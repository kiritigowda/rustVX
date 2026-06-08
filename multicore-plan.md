# Multicore Pipeline Execution Plan

## Task
Design and implement multicore execution for the OpenVX pipelining extension in rustVX. The current pipelining implementation is single-threaded per graph — we need wave-based intra-graph parallelism that executes independent nodes on multiple CPU cores.

## Background
Current pipelining is **single-threaded per graph**:
- One background executor thread per graph in `QUEUE_AUTO` mode
- `execution_mutex` serializes all executions of the same graph
- `execute_pipelined_graph` runs nodes sequentially in topological order
- `finish_pipelined_execution` runs after all nodes

The topological sort now correctly orders nodes including scalar dependencies (fixed earlier today). This means nodes at the same topological "wave" can execute in parallel.

## Plan

### Step 1: [DOCUMENT] Document current single-threaded architecture ✅ DONE
**Status:** `docs/pipelining_architecture.md` written

### Step 2: [DESIGN] Design wave-based parallel execution model ✅ DONE
**Status:** `docs/multicore_pipeline_design.md` written

### Step 3: [IMPLEMENT] Create global thread pool for node dispatch ✅ DONE
**Status:** `openvx-core/src/thread_pool.rs` created and tested

### Step 4: [IMPLEMENT] Wave-based node execution in `execute_pipelined_graph` ✅ DONE
**Status:** `pipelining_executor.rs` modified with wave-based dispatch
**Tests:** All 9 UserKernel tests pass, 12 representative tests pass

### Step 5: [IMPLEMENT] Per-node execution wrapper with error propagation
**Dependencies:** Step 4
**Approach:**
- Create `execute_node_parallel` wrapper:
  - Calls existing `execute_node` (from `unified_c_api.rs`)
  - Catches return status
  - Stores result in a shared `Vec<vx_status>` (indexed by node position)
  - Uses `Arc<Mutex<Vec<vx_status>>>` for error collection
- Verify `execute_node` is thread-safe:
  - `node_data.parameters` is a `Mutex<Vec<Option<u64>>>` — lock per access, safe
  - `node_data.status` is `AtomicI32` — safe
  - `kernel_output_indices` is a `Lazy<HashMap<...>>` — read-only after init, safe
  - `REF_SUBSTITUTIONS` is thread-local — safe
  - `USER_KERNELS` / `USER_KERNEL_PARAMS` are `Lazy<Mutex<...>>` — read-heavy, lock per access
**Verification:** Unit test: spawn 100 threads, each calls `execute_node` on different nodes, verify no data races
**Files:** `openvx-core/src/pipelining_executor.rs`

### Step 6: [IMPLEMENT] Add `OPENVX_PIPELINING_THREADS` configuration
**Dependencies:** Step 3
**Approach:**
- Read env var at thread pool creation time
- Add to `pipelining.rs` config struct:
  ```rust
  pub struct PipeliningConfig {
      pub thread_count: usize,
  }
  ```
- Expose via C API (optional): `vxSetContextAttribute` with a new attribute enum
- Default behavior:
  - `OPENVX_PIPELINING_THREADS=1` → single-threaded (current behavior, useful for debugging)
  - `OPENVX_PIPELINING_THREADS=0` or unset → auto-detect `num_cpus::get()`
  - `OPENVX_PIPELINING_THREADS=N` → exactly N threads
**Verification:** Test with `OPENVX_PIPELINING_THREADS=1` produces same results as before
**Files:** `openvx-core/src/pipelining.rs`, `openvx-core/src/thread_pool.rs`

### Step 7: [TEST] Write correctness tests for parallel execution
**Dependencies:** Steps 4, 5, 6
**Approach:**
- Test 1: `GraphPipeline.TwoNodes` with different thread counts (1, 2, 4, auto)
  - Verify same results regardless of thread count
- Test 2: `GraphPipeline.UserKernel` with parallel execution
  - The scalar dependency test is a good stress test for wave ordering
- Test 3: `GraphPipeline.MaxDataRef` — many parallel inputs
  - This graph has many independent inputs that can execute in parallel
- Test 4: Error propagation test
  - Create a graph where one node fails, verify remaining nodes don't execute
- Test 5: Thread safety test
  - Run 1000 graph executions concurrently with different graphs
  - Verify no crashes, no data races
**Verification:** All tests pass with `OPENVX_PIPELINING_THREADS=1`, `=2`, `=4`, and `=auto`
**Files:** `OpenVX-cts/test_conformance/test_graph_pipeline.c` (add new test cases), or separate Rust test module

### Step 8: [TEST] Run full conformance suite with multicore enabled
**Dependencies:** Step 7
**Approach:**
- Run full `GraphPipeline.*` suite with `OPENVX_PIPELINING_THREADS=auto`
- Run full baseline suite (non-pipelining) to verify no regressions
- Run with `OPENVX_PIPELINING_THREADS=1` to verify single-threaded mode still works
- Run stress tests with high loop counts and multicore
**Verification:** All 109 pipelining tests pass, all baseline tests pass, no new failures
**Files:** CI workflow `conformance.yml` (add matrix with different thread counts)

### Step 9: [BENCHMARK] Measure performance improvement from multicore
**Dependencies:** Step 8
**Approach:**
- Use `openvx-mark` benchmark suite
- Compare single-threaded vs multicore for:
  - Graphs with many independent nodes (e.g., `MaxDataRef`)
  - Graphs with long chains (e.g., `LoopCarriedDependency`)
  - Real-world vision graphs (e.g., feature extraction pipelines)
- Metrics:
  - Throughput (MP/s)
  - Latency (ms per graph execution)
  - CPU utilization (% of cores used)
**Verification:** Multicore shows improvement for parallelizable graphs, no regression for sequential graphs
**Files:** `docs/multicore_performance_report.md`

### Step 10: [DOCUMENT] Write user-facing documentation
**Dependencies:** Steps 1–9
**Approach:**
- Update README.md:
  - Add section on pipelining configuration
  - Document `OPENVX_PIPELINING_THREADS` env var
  - Explain when multicore helps (graphs with parallel branches)
- Update API docs:
  - Document new context attribute (if added)
  - Clarify thread-safety guarantees of pipelining API
- Write `docs/multicore_pipeline.md`:
  - Architecture overview
  - Configuration guide
  - Performance tuning tips
  - Troubleshooting (e.g., "my graph is slower with multicore — why?")
**Verification:** Docs are clear enough for a user to enable and tune multicore pipelining
**Files:** `README.md`, `docs/multicore_pipeline.md`

## Risk Analysis
**Potential blockers:**
1. `execute_node` may have hidden shared mutable state that's not thread-safe
   - Mitigation: Audit all globals accessed by `execute_node` (see Step 5)
2. Wave-based execution may increase latency for small graphs (barrier overhead)
   - Mitigation: Add threshold — only use multicore for graphs with >N nodes or >M waves
3. Thread pool contention with multiple graphs in QUEUE_AUTO mode
   - Mitigation: Use work-stealing queue, limit total threads
4. `REF_SUBSTITUTIONS` thread-local may not be initialized in worker threads
   - Mitigation: Ensure thread-local init in worker thread startup
5. Performance may not improve on CI runners (which may have 2 cores)
   - Mitigation: Benchmark locally, set expectations accordingly

## Rollback Plan
**If fails at any step:**
- Keep single-threaded path intact (never remove it)
- `OPENVX_PIPELINING_THREADS=1` always works as fallback
- Can disable multicore at runtime if errors detected

## Decision: Start with Step 1?
This is a significant engineering effort (10 steps, ~2-3 hours each). Recommend starting with Step 1 (architecture docs) to ensure we understand the current system before modifying it.

Ready to proceed?
