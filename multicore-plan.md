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

### Step 1: [DOCUMENT] Document current single-threaded architecture
**Dependencies:** None
**Approach:**
- Write `docs/pipelining_architecture.md` describing:
  - How QUEUE_AUTO executor thread works
  - How `execute_pipelined_graph` currently works (sequential node dispatch)
  - How `finish_pipelined_execution` works (moving refs to done, emitting events)
  - Where synchronization points are (`execution_mutex`, `active_executions`, `active_cv`)
  - Why current approach is safe but not parallel
**Verification:** Document is clear enough for another engineer to understand
**Files:** `docs/pipelining_architecture.md`

### Step 2: [DESIGN] Design wave-based parallel execution model
**Dependencies:** Step 1
**Approach:**
- Design a wave-based execution model:
  - **Wave 0:** All nodes with in-degree 0 execute in parallel
  - **Wave 1:** After all wave-0 nodes finish, nodes whose dependencies are satisfied execute in parallel
  - Continue until all waves complete
- Determine synchronization strategy:
  - Per-wave barrier (all nodes in wave N must finish before wave N+1 starts)
  - Or per-node notification (each node notifies its dependents when done)
- Decide thread pool model:
  - Global thread pool (shared across all graphs) vs per-graph thread pool
  - Configuration via environment variable or build flag
- Define memory model for shared state:
  - `REF_SUBSTITUTIONS` is thread-local — safe for parallel reads, needs coordination for writes
  - Node output refs are written by producer, read by consumers in later waves
  - `NODE_PARAMETER_BINDINGS` / `GRAPH_PARAMETER_BINDINGS` are read-only during execution
**Verification:** Design doc covers edge cases (cyclic graphs, error handling, graph release during execution)
**Files:** `docs/multicore_pipeline_design.md`

### Step 3: [IMPLEMENT] Create global thread pool for node dispatch
**Dependencies:** Step 2
**Approach:**
- Create `openvx-core/src/thread_pool.rs`:
  - `ThreadPool` struct with configurable worker count
  - `spawn` method that accepts closures
  - `join` or `wait_all` for barrier synchronization
  - Use `std::thread` + channels (or `crossbeam` if available)
- Configuration:
  - Default thread count = `num_cpus::get()` or env var `OPENVX_PIPELINING_THREADS`
  - Minimum 1, maximum 64 (sanity cap)
- Lazy initialization: thread pool created on first use, shared across all graphs
**Verification:** Thread pool compiles, basic spawn/join test passes
**Files:** `openvx-core/src/thread_pool.rs`, update `openvx-core/src/lib.rs`

### Step 4: [IMPLEMENT] Wave-based node execution in `execute_pipelined_graph`
**Dependencies:** Step 3
**Approach:**
- Modify `execute_pipelined_graph` in `pipelining_executor.rs`:
  1. Build wave map: group nodes by topological depth (already computed by `vxVerifyGraph`)
  2. For each wave:
     a. Spawn all nodes in wave to thread pool
     b. Wait for all wave nodes to complete
     c. Check for errors — if any node failed, abort remaining waves
  3. After all waves: run `finish_pipelined_execution`
- Key changes:
  - `execute_node` must be safe to call from multiple threads concurrently (verify no shared mutable state)
  - Error propagation: collect status from all nodes in wave, fail fast
  - Keep thread-local `REF_SUBSTITUTIONS` per worker thread (already thread-local)
**Verification:** Single-graph pipelining tests pass, no regressions in non-pipelining tests
**Files:** `openvx-core/src/pipelining_executor.rs`

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
