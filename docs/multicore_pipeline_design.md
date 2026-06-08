# Multicore Pipeline Execution Design

## Goal
Execute independent nodes within a graph in parallel across multiple CPU cores, while preserving the correctness guarantees of the single-threaded executor.

## Current Limitation

The existing `execute_pipelined_graph` runs nodes sequentially:

```rust
for node_id in nodes.iter() {
    execute_node(*node_id);
}
```

For a graph like `MaxDataRef` with 8+ independent input nodes, only one core is used.

## Wave-Based Parallelism Model

### Concept

After `vxVerifyGraph`, nodes are in topological order. We can compute **waves** (also called levels or layers):

- **Wave 0:** Nodes with no dependencies (in-degree 0)
- **Wave 1:** Nodes whose dependencies are all in wave 0
- **Wave 2:** Nodes whose dependencies are all in waves ≤ 1
- etc.

Within a wave, all nodes are independent and can execute in parallel. Between waves, we need a barrier (all nodes in wave N must finish before wave N+1 starts).

### Example

```
Graph: image -> blur -> output
       scalar -> add_scalar
       
Nodes:
  n0: read image (wave 0)
  n1: read scalar (wave 0)
  n2: blur image (wave 1, depends on n0)
  n3: add scalar (wave 1, depends on n1)
  n4: write output (wave 2, depends on n2+n3)

Execution:
  Wave 0: n0 || n1   (parallel)
  Wave 1: n2 || n3   (parallel)
  Wave 2: n4         (sequential)
```

### Data Safety

Within a wave:
- Each node only **reads** refs produced by earlier waves
- Each node only **writes** its own output refs
- No shared mutable state between nodes in the same wave

Between waves:
- Wave N+1 nodes read outputs from wave N nodes
- The barrier ensures all writes from wave N are visible

## Design Decisions

### 1. Thread Pool Model: Global, Shared Across Graphs

**Decision:** Use a single global thread pool, not per-graph pools.

**Rationale:**
- Multiple graphs in QUEUE_AUTO mode each have their own executor thread
- The thread pool is only used for intra-graph parallel node dispatch
- A global pool avoids creating/destroying threads per graph
- Total worker threads = `min(num_cpus, OPENVX_PIPELINING_THREADS)`

```rust
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;
```

### 2. Wave Computation: Pre-compute at Verify Time

**Decision:** Compute waves during `vxVerifyGraph` and store in `GraphData`.

**Rationale:**
- Graph topology is immutable after verify
- No runtime overhead for computing waves during execution
- Waves only change if graph is re-verified (which re-creates nodes)

```rust
pub struct GraphData {
    // ... existing fields ...
    pub topo_waves: Vec<Vec<u64>>,  // wave[i] = list of node IDs in wave i
}
```

**Algorithm:**
```rust
fn compute_waves(topo_order: &[u64], dependencies: &HashMap<u64, Vec<u64>>) -> Vec<Vec<u64>> {
    let mut wave_map: HashMap<u64, usize> = HashMap::new();
    
    for node_id in topo_order {
        let deps = dependencies.get(node_id).unwrap_or(&vec![]);
        let max_dep_wave = deps.iter()
            .map(|d| wave_map.get(d).unwrap_or(&0))
            .max()
            .unwrap_or(0);
        let wave = max_dep_wave + 1;
        wave_map.insert(*node_id, wave);
    }
    
    // Group by wave
    let num_waves = wave_map.values().max().unwrap_or(&0) + 1;
    let mut waves: Vec<Vec<u64>> = vec![vec![]; num_waves];
    for (node_id, wave) in wave_map {
        waves[wave].push(node_id);
    }
    waves
}
```

### 3. Execution Strategy: Barrier Between Waves

**Decision:** Use a barrier between waves (all nodes in wave N finish before N+1 starts).

**Rationale:**
- Simpler than per-node notification
- Matches the current sequential model's semantics exactly
- Easier to reason about correctness
- Slightly higher latency for small graphs, but acceptable

**Alternative considered:** Per-node notification (each node signals dependents when done). Rejected because it adds complexity and the barrier model is already correct.

### 4. Error Handling: Fail Fast, Propagate First Error

**Decision:** If any node in a wave fails, abort remaining waves and propagate the error.

**Implementation:**
```rust
struct WaveResult {
    statuses: Vec<vx_status>,
    node_ids: Vec<u64>,
}

// After executing a wave, check for failures
if result.statuses.iter().any(|s| *s != VX_SUCCESS) {
    // Find first error
    let first_error = result.statuses.iter()
        .zip(&result.node_ids)
        .find(|(s, _)| **s != VX_SUCCESS);
    return first_error.map(|(s, _)| *s).unwrap_or(VX_FAILURE);
}
```

### 5. Thread Safety of `execute_node`

**Audit of shared state accessed by `execute_node`:**

| State | Type | Thread-Safe? | Notes |
|-------|------|--------------|-------|
| `node_data.parameters` | `Mutex<Vec<Option<u64>>>` | ✅ Yes | Lock per access |
| `node_data.status` | `AtomicI32` | ✅ Yes | Atomic ops |
| `node_data.run_count` | `AtomicU64` | ✅ Yes | Atomic ops |
| `kernel_output_indices` | `Lazy<HashMap<...>>` | ✅ Yes | Read-only after init |
| `REF_SUBSTITUTIONS` | `thread_local!` | ✅ Yes | Per-thread, no sharing |
| `USER_KERNELS` | `Lazy<Mutex<...>>` | ⚠️ Lock needed | Read-heavy, rare writes |
| `USER_KERNEL_PARAMS` | `Lazy<Mutex<...>>` | ⚠️ Lock needed | Read-heavy, rare writes |
| `GRAPHS_DATA` | `Lazy<Mutex<...>>` | ⚠️ Lock needed | Read during execution |
| `NODES` | `Lazy<Mutex<...>>` | ⚠️ Lock needed | Read during execution |
| `REFERENCE_TYPES` | `Lazy<Mutex<...>>` | ⚠️ Lock needed | Read during execution |

**Conclusion:** `execute_node` is thread-safe for concurrent execution on different nodes. The only concern is lock contention on `USER_KERNELS` / `USER_KERNEL_PARAMS` / `NODES` — but these are read-heavy and locks are held briefly.

### 6. Configuration

**Environment variable:** `OPENVX_PIPELINING_THREADS`

| Value | Behavior |
|-------|----------|
| `0` or unset | Auto-detect `num_cpus::get()` |
| `1` | Single-threaded (current behavior, no thread pool) |
| `N` | Exactly N worker threads |

**API (optional future work):**
```c
vx_context_attribute_e VX_CONTEXT_PIPELINING_THREADS = /* new enum */;
vxSetContextAttribute(context, VX_CONTEXT_PIPELINING_THREADS, &N, sizeof(N));
```

### 7. Performance Thresholds

**Decision:** Always use wave-based execution when `OPENVX_PIPELINING_THREADS > 1`, but only spawn threads for waves with >1 nodes.

**Rationale:**
- Single-node waves run on the caller thread (no spawn overhead)
- Multi-node waves spawn to thread pool
- The overhead of computing waves is negligible compared to node execution time

## Modified Execution Flow

### `execute_pipelined_graph` (multicore version)

```rust
fn execute_pipelined_graph(graph_id: u64) -> i32 {
    // ... setup (same as before) ...

    let waves = &g.topo_waves;
    
    for wave in waves.iter() {
        if wave.len() == 1 {
            // Fast path: single node, execute on caller thread
            let node_id = wave[0];
            match execute_node(node_id) {
                Some(0) => notify_node_completed(graph_id, node_id, context_id),
                Some(status) => return status,
                None => return -1,
            }
        } else {
            // Parallel path: spawn all nodes to thread pool
            let mut handles = Vec::with_capacity(wave.len());
            for node_id in wave {
                let nid = *node_id;
                let gid = graph_id;
                let cid = context_id;
                let handle = thread_pool.spawn(move || {
                    let status = execute_node(nid);
                    (nid, status)
                });
                handles.push(handle);
            }
            
            // Wait for all nodes in wave
            for handle in handles {
                let (node_id, status) = handle.join();
                match status {
                    Some(0) => notify_node_completed(graph_id, node_id, context_id),
                    Some(status) => return status,
                    None => return -1,
                }
            }
        }
    }
    
    finish_pipelined_execution(graph_id, context_id);
    0
}
```

### `vxVerifyGraph` (add wave computation)

```rust
// After computing topo_order, also compute waves:
let waves = compute_waves(&topo_order, &param_to_producer);
{
    let mut g = graphs.get_mut(&graph_id).unwrap();
    g.topo_waves = waves;
}
```

## Edge Cases

1. **Cyclic graph:** Already rejected by `vxVerifyGraph` — waves are only computed for valid DAGs
2. **Graph with single node:** Waves = `[[n0]]`, executes on caller thread (no spawn overhead)
3. **Graph release during execution:** `execution_mutex` prevents concurrent execution; `vxReleaseGraph` stops executor first
4. **Node failure in wave:** First error is propagated, remaining nodes in wave are still waited for (to avoid dangling threads)
5. **Thread pool exhaustion:** Use work-stealing or bounded queue; fallback to caller thread if pool full
6. **Cross-graph parallelism:** Multiple graphs each have their own executor thread; the global thread pool is shared

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hidden shared mutable state in `execute_node` | Medium | High | Audit all globals (Step 5 of implementation plan) |
| Lock contention on `USER_KERNELS`/`NODES` | Medium | Medium | Profile; consider RwLock if reads dominate |
| Barrier overhead for small graphs | High | Low | Fast path for single-node waves |
| Thread pool creation failure | Low | High | Fallback to single-threaded mode |
| `REF_SUBSTITUTIONS` not initialized in worker | Low | High | Initialize thread-local on first use |

## Testing Strategy

1. **Correctness:** Run full `GraphPipeline.*` suite with `OPENVX_PIPELINING_THREADS=1,2,4,auto`
2. **Stress:** Run 1000 graph executions concurrently
3. **Regression:** Full baseline suite (non-pipelining) must pass
4. **Performance:** Benchmark `MaxDataRef` (many parallel nodes) and `LoopCarriedDependency` (sequential chain)

## Implementation Order

1. Compute waves in `vxVerifyGraph`
2. Add `topo_waves` to `GraphData`
3. Create global thread pool
4. Modify `execute_pipelined_graph` for wave-based dispatch
5. Add `OPENVX_PIPELINING_THREADS` configuration
6. Test and benchmark

## References

- `docs/pipelining_architecture.md` — Current single-threaded architecture
- rustVX PR #45 — Pipelining extension implementation
- OpenVX 1.3.1 Pipelining KHR extension spec
