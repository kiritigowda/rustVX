# OpenVX Pipelining Multicore Sample

Demonstrates wave-based parallel execution using the OpenVX Pipelining extension on multicore CPUs.

## What It Shows

- **4 parallel image-processing branches** (Gaussian blur, Box filter, Dilate, Erode)
- **2 execution waves** computed automatically by `vxVerifyGraph`:
  - Wave 0: All 4 filter nodes (no dependencies → run in parallel)
  - Wave 1: All 4 fill nodes (depend on Wave 0 → run in parallel after barrier)
- **QUEUE_AUTO mode** for overlapping graph executions
- **Environment variable** `OPENVX_PIPELINING_THREADS` to tune parallelism

## Requirements

- rustVX built with `-DOPENVX_USE_PIPELINING=ON`
- GCC or Clang
- OpenVX headers (from rustVX `include/` directory)

## Build

```bash
cd samples/pipelining_multicore
make OPENVX_INCLUDE=/path/to/rustVX/include OPENVX_LIB=/path/to/rustVX/target/release
```

Or manually:
```bash
gcc -O3 -o pipelining_multicore pipelining_multicore.c \
    -I/path/to/rustVX/include \
    -L/path/to/rustVX/target/release \
    -lopenvx -Wl,-rpath,/path/to/rustVX/target/release
```

## Run

```bash
# Auto-detect thread pool size (hardware cores, capped at 64)
./pipelining_multicore

# Force single-threaded (sequential fallback)
OPENVX_PIPELINING_THREADS=1 ./pipelining_multicore

# Use exactly 4 threads
OPENVX_PIPELINING_THREADS=4 ./pipelining_multicore

# Debug: show thread pool size being used (rustVX logs at init)
RUST_LOG=info OPENVX_PIPELINING_THREADS=4 ./pipelining_multicore
```

## Expected Output

```
✓ OpenVX Pipelining Extension available
✓ Graph verified (topological waves computed)
✓ Pipelining mode set to QUEUE_AUTO
✓ Graph scheduled (executor thread started)
Warming up...
Running benchmark (100 iterations)...

=== Results ===
Total time:  245.32 ms
Iterations:   100
Throughput:   407.63 FPS

Notes:
- Nodes in Wave 0 (Gaussian, Box, Dilate, Erode) execute in parallel
- Nodes in Wave 1 (4× Fill) execute in parallel after Wave 0
- Set OPENVX_PIPELINING_THREADS=N to control thread pool size
```

## Architecture

```
Input Image (640×480)
    │
    ├──→ [Gaussian3x3] ──→ tmp_a ──→ [Fill] ──→ out_a
    ├──→ [Box3x3]      ──→ tmp_b ──→ [Fill] ──→ out_b
    ├──→ [Dilate3x3]   ──→ tmp_c ──→ [Fill] ──→ out_c
    └──→ [Erode3x3]    ──→ tmp_d ──→ [Fill] ──→ out_d
```

### Wave Schedule

| Wave | Nodes | Why Parallel? |
|------|-------|---------------|
| 0 | Gaussian, Box, Dilate, Erode | All read same input, no inter-dependencies |
| 1 | Fill A, Fill B, Fill C, Fill D | All depend only on Wave 0 outputs |

Between waves: barrier ensures Wave 0 completes before Wave 1 starts.

## Performance Tips

1. **More parallel branches = more speedup** — add more independent nodes
2. **`OPENVX_PIPELINING_THREADS`** — match your CPU core count (or slightly less to leave cores for OS)
3. **Queue depth** — enqueue multiple frames ahead to hide latency
4. **Avoid false dependencies** — use virtual images for intermediates

## Files

- `pipelining_multicore.c` — main sample
- `Makefile` — build automation
- `README.md` — this file

## See Also

- `docs/pipelining_architecture.md` — rustVX pipelining internals
- `docs/multicore_pipeline_design.md` — wave-based execution design
- OpenVX 1.3 Specification — Pipelining Extension (khronos.org)
