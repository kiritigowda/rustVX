# rustVX Pipelining Architecture

## Overview

rustVX implements the OpenVX Pipelining, Streaming & Batch Processing KHR extension. This document describes the **current single-threaded** architecture as of the merge of PR #45.

## Components

### 1. Pipelining State (`pipelining.rs`)

```rust
pub struct VxGraphPipeliningState {
    pub schedule_mode: Mutex<VxGraphScheduleMode>,
    pub parameter_queues: Mutex<HashMap<u32, Arc<VxGraphParameterQueue>>>,
    pub streaming_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    pub active_executions: AtomicUsize,        // Count of in-flight executions
    pub active_mutex: Mutex<()>,              // Guards active_executions + active_cv
    pub active_cv: Condvar,                    // Signals when active_executions drops to 0
    pub executor_running: AtomicBool,          // QUEUE_AUTO executor loop control
    pub executor_notify: Mutex<Arc<Condvar>>, // Wake executor when work arrives
}
```

### 2. Global Registries

| Registry | Type | Purpose |
|----------|------|---------|
| `GRAPH_PIPELINING` | `Lazy<Mutex<HashMap<u64, Arc<VxGraphPipeliningState>>>>` | Per-graph pipelining config |
| `EVENT_SYSTEMS` | `Lazy<Mutex<HashMap<u64, Arc<VxEventSystem>>>>` | Per-context event queues |
| `ACTIVE_PIPELINING_GRAPHS` | `AtomicUsize` | Fast-path counter (non-Normal graphs) |

### 3. Execution Entry Points

#### `vxScheduleGraph` (`unified_c_api.rs:5050`)
- Called by application to schedule a graph for execution
- For pipelining graphs: pre-increments `active_executions`, spawns background thread
- For non-pipelining graphs: spawns background thread (no pipelining state)

#### `execute_graph_nodes` (`unified_c_api.rs:2296`)
- Called by the background thread spawned by `vxScheduleGraph`
- **Single-threaded**: executes nodes sequentially in topological order
- Clears `REF_SUBSTITUTIONS` (thread-local) at start
- Uses `ActiveExecGuard` to decrement `active_executions` on completion

#### `execute_pipelined_graph` (`pipelining_executor.rs`)
- Called by QUEUE_AUTO executor thread (or by `execute_graph_nodes` for QUEUE_MANUAL)
- **Single-threaded**: loops over nodes in `topo_order`, calling `execute_node` for each
- After all nodes: calls `finish_pipelined_execution`

### 4. QUEUE_AUTO Executor Thread

```
Graph released / program exit
      │
      ▼
+-------------+     +------------------+
│  App Thread │────▶│ vxSetGraphSchedule │
+-------------+     │   Config(...,      │
                    │   QueueAuto)         │
                    +------------------+
                              │
                              ▼
                    +------------------+
                    │  Start executor  │
                    │  thread (spawn)  │
                    +------------------+
                              │
                              ▼
                    +------------------+
                    │  executor_loop() │◄──────┐
                    │  Wait on executor│       │
                    │  _notify condvar │       │
                    +------------------+       │
                              │                │
                              ▼                │
                    +------------------+       │
                    │  Check for work    │       │
                    │  (ref enqueued?)   │       │
                    +------------------+       │
                              │                │
                              ▼                │
                    +------------------+       │
                    │ execute_pipelined│       │
                    │    _graph()      │       │
                    +------------------+       │
                              │                │
                              ▼                │
                    +------------------+       │
                    │  Signal completion │───────┘
                    │  (active_executions--)
                    +------------------+
```

### 5. Synchronization Points

| Synchronization | Location | Purpose |
|----------------|----------|---------|
| `execution_mutex` | `unified_c_api.rs` (around `execute_graph_nodes`) | Serializes all executions of the same graph |
| `active_executions` + `active_cv` | `VxGraphPipeliningState` | `vxWaitGraph` waits for background thread completion |
| `parameter_queues` mutex | `VxGraphPipeliningState` | Enqueue/dequeue refs |
| `executor_notify` condvar | `VxGraphPipeliningState` | Wakes executor when refs are enqueued |
| `GRAPH_PIPELINING` mutex | Global | Access pipelining state (fast-path via `ACTIVE_PIPELINING_GRAPHS`) |

### 6. Node Execution Flow (QUEUE_MANUAL)

```
vxScheduleGraph(graph)
    │
    ▼
Lock execution_mutex (per-graph)
    │
    ▼
execute_graph_nodes(graph)
    │
    ├── Clear REF_SUBSTITUTIONS
    ├── Check pipelining mode (fast-path via ACTIVE_PIPELINING_GRAPHS)
    ├── Loop over queued ref sets:
    │     For each set:
    │       Build REF_SUBSTITUTIONS map (graph param -> node param)
    │       For each node in topo_order:
    │         resolve_graph_parameter() (apply substitutions)
    │         execute_node(node_id)
    │       Move consumed refs -> done
    │       Emit parameter_consumed events
    ├── If all iterations succeeded:
    │     Move all refs -> done
    │     Emit graph_completed events
    │     auto_age_delays()
    ├── ActiveExecGuard drops: active_executions--, signals active_cv
    │
    ▼
Unlock execution_mutex
```

### 7. Node Execution Flow (QUEUE_AUTO)

```
vxSetGraphScheduleConfig(..., QueueAuto)
    │
    ▼
Start executor_loop thread
    │
    ├── Waits on executor_notify condvar
    │
    ├── Wakes when refs enqueued
    │     │
    │     ▼
    │   execute_pipelined_graph(graph_id)
    │     ├── Read topo_order (immutable after verify)
    │     ├── For each node in topo_order:
    │     │   resolve_graph_parameter()
    │     │   execute_node(node_id)
    │     ├── finish_pipelined_execution()
    │     │   ├── Move refs -> done
    │     │   ├── Emit events
    │     │   └── auto_age_delays()
    │     └── Loop back (or sleep if no refs)
```

## Key Invariants

1. **Topological order is immutable** after `vxVerifyGraph`
2. **REF_SUBSTITUTIONS is thread-local** — safe per-thread, no cross-thread sharing
3. **execution_mutex serializes per-graph** — only one execution of a given graph at a time
4. **Node outputs are only read by later waves** — enforced by topological sort
5. **Graph parameter queues are FIFO** — refs dequeued in enqueue order

## Why Current Approach is Safe but Not Parallel

### Safety Guarantees
- Single thread per graph means no race conditions on node execution
- `REF_SUBSTITUTIONS` thread-local means no cross-thread ref confusion
- `execution_mutex` means no overlapping executions of the same graph
- Topological sort means producers always execute before consumers

### Performance Limitations
- Nodes in the same topological "wave" execute sequentially even though they have no dependencies
- For graphs with many parallel branches (e.g., `MaxDataRef`), this leaves cores idle
- The `AbsDiff` benchmark regression was from the `GRAPH_PIPELINING` lock check on every execution (fixed by `ACTIVE_PIPELINING_GRAPHS` fast-path)

## Files

| File | Purpose |
|------|---------|
| `openvx-core/src/pipelining.rs` | Core types: `VxGraphPipeliningState`, `VxGraphParameterQueue`, `VxEventSystem` |
| `openvx-core/src/pipelining_api.rs` | C API: `vxSetGraphScheduleConfig`, `vxGraphParameterEnqueueReadyRef`, events |
| `openvx-core/src/pipelining_executor.rs` | Executor thread: `executor_loop`, `execute_pipelined_graph`, `finish_pipelined_execution` |
| `openvx-core/src/unified_c_api.rs` | Graph execution: `execute_graph_nodes`, `execute_node`, `resolve_graph_parameter`, topological sort |
| `openvx-core/src/c_api.rs` | Graph/node lifecycle: `vxReleaseGraph`, `vxReleaseNode`, `vxCreateGenericNode` |

## References

- OpenVX 1.3.1 Pipelining, Streaming & Batch Processing KHR extension
- rustVX PR #45: "OpenVX Pipelining, Streaming & Batch Processing KHR extension"
