# OpenVX Pipelining Multicore Samples

Demonstrates wave-based parallel execution using the OpenVX Pipelining extension on multicore CPUs.

## Samples

### 1. `pipelining_multicore` — Basic Wave-Parallel Demo

A minimal 4-branch parallel filter graph showing how the wave executor works.

```
Input
  ├──→ Gaussian3x3 → Fill → out_a
  ├──→ Box3x3      → Fill → out_b
  ├──→ Dilate3x3   → Fill → out_c
  └──→ Erode3x3    → Fill → out_d
```

- **Wave 0**: All 4 filters execute in parallel
- **Wave 1**: All 4 fills execute in parallel after barrier

### 2. `pipelining_vs_nonpipelining` — Throughput Comparison

Runs the *same* graph twice to measure the performance benefit:

```
Input
  ├──→ Gaussian3x3 ──→ tmp_a ─┐
  ├──→ Erode3x3    ──→ tmp_b ─┤→ AND → AND → Output
  └──→ Dilate3x3  ──→ tmp_c ─┘
```

- **Non-pipelining mode**: `vxProcessGraph`, one frame at a time
- **Pipelining mode**: `QUEUE_AUTO` with 3-deep buffer queue, overlapping execution

Prints throughput (FPS) and speedup ratio.

### 3. `multiscale_feature_extraction` — Real-World CV Pipeline

A realistic preprocessing stage inspired by object-detection networks (YOLO, SSD):

```
RGB Input (1920×1080)
    └──→ ColorConvert → Y (grayscale)
            ├──→ Gaussian3x3 ──→ Canny(full-res)   ──┐
            ├──→ HalfScale → Gaussian3x3 ──→ Canny(half) ──→ ScaleUp ──┤→ OR → OR → Confidence Map
            └──→ HalfScale → HalfScale → Gaussian3x3 ──→ Canny(quarter) ──→ ScaleUp ──┘
```

- **Wave 0**: ColorConvert + 3 Gaussian blurs (parallel)
- **Wave 1**: 2 HalfScale (sequential dependency)
- **Wave 2**: 3 Canny edge detectors (parallel, independent scales)
- **Wave 3**: 2 ScaleUp + 2 OR (fusion)

This is where multicore pipelining shines — the 3 Canny detectors are expensive and embarrassingly parallel.

## Requirements

- rustVX built with `-DOPENVX_USE_PIPELINING=ON`
- GCC or Clang with POSIX support (`-D_POSIX_C_SOURCE=200809L`)
- OpenVX headers (from rustVX `include/`)

## Build

```bash
cd samples/pipelining_multicore
make OPENVX_INCLUDE=../../include OPENVX_LIB=../../target/release
```

Builds all three samples.

## Run

### Basic demo
```bash
./pipelining_multicore                    # auto-detect threads
OPENVX_PIPELINING_THREADS=1 ./pipelining_multicore   # single-threaded
OPENVX_PIPELINING_THREADS=4 ./pipelining_multicore   # 4 threads
```

### Throughput comparison
```bash
./pipelining_vs_nonpipelining
```

Expected output:
```
[1/2] NON-PIPELINING (vxProcessGraph)...
      12.345 ms/frame = 81.01 FPS

[2/2] PIPELINING (QUEUE_AUTO + enqueue/dequeue)...
      7.234 ms/frame = 138.24 FPS

=== Summary ===
Non-pipelining throughput:  81.01 FPS
Pipelining throughput:      138.24 FPS
Throughput speedup:         1.71x
```

### Real-world pipeline
```bash
./multiscale_feature_extraction
OPENVX_PIPELINING_THREADS=4 ./multiscale_feature_extraction
```

## Performance Tips

1. **More parallel branches = more speedup** — add independent nodes
2. **`OPENVX_PIPELINING_THREADS`** — match CPU core count (or slightly less)
3. **Queue depth** — `NUM_BUF=3` is a good default; increase for bursty inputs
4. **Avoid false dependencies** — use virtual images for intermediates
5. **Large nodes benefit most** — Canny, GaussianPyramid, OpticalFlow

## Architecture

```
┌─────────────────────────────────────┐
│         Wave Scheduler              │
│  (computed by vxVerifyGraph)        │
├─────────────────────────────────────┤
│  Wave 0: Node A │ Node B │ Node C  │  ← parallel on thread pool
│  ───────────────────────────────────│  ← barrier
│  Wave 1: Node D │ Node E            │  ← parallel on thread pool
│  ───────────────────────────────────│  ← barrier
│  Wave 2: Node F                     │  ← single node, fast path
└─────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `pipelining_multicore.c` | Basic parallel-branch demo |
| `pipelining_vs_nonpipelining.c` | Head-to-head throughput benchmark |
| `multiscale_feature_extraction.c` | Real-world multi-scale CV pipeline |
| `Makefile` | Build all three samples |
| `README.md` | This file |

## See Also

- `docs/pipelining_architecture.md` — rustVX pipelining internals
- `docs/multicore_pipeline_design.md` — wave-based execution design
- OpenVX 1.3 Pipelining Extension Spec — khronos.org
