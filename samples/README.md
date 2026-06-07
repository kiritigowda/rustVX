# rustVX Samples

This directory contains example applications demonstrating OpenVX features implemented in rustVX.

## Available Samples

### `pipelining_multicore/` — OpenVX Pipelining Extension Demos

Demonstrates wave-based parallel execution using the OpenVX Pipelining extension on multicore CPUs.

| Sample | Description |
|--------|-------------|
| **`pipelining_multicore.c`** | Basic 4-branch parallel filter (Gaussian, Box, Dilate, Erode). Shows wave scheduling. |
| **`benchmark_pipelining.c`** | **Head-to-head throughput benchmark** — runs the same graph with `vxProcessGraph` (sequential) vs `QUEUE_AUTO` (pipelined). Prints FPS and speedup ratio. |
| **`multiscale_feature_extraction.c`** | **Real-world CV pipeline** — multi-scale edge detection (Sobel + Magnitude at 3 scales), fusion via OR. Inspired by YOLO/SSD preprocessing. |

### Performance (measured on 4-core Ubuntu 22.04)

| Sample | Non-Pipelining | Pipelining (auto) | Pipelining (4 threads) | Speedup |
|--------|---------------|-------------------|------------------------|---------|
| Basic 3-filter | 195 FPS | 357 FPS | 408 FPS | **~2.1x** |
| Multi-scale feature extraction | 19.7 FPS | 41.8 FPS | **46.7 FPS** | **~2.2x** |

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `OPENVX_PIPELINING_THREADS` | `1` | Single-threaded sequential fallback |
| | `0` or unset | Auto-detect hardware cores (capped at 64) |
| | `N` | Exactly N threads in pool |

### Build

```bash
cd samples/pipelining_multicore
make OPENVX_INCLUDE=../../include OPENVX_LIB=../../target/release
```

### Run

```bash
# Basic demo
./pipelining_multicore

# Throughput comparison (non-pipelining vs pipelining)
./benchmark_pipelining
OPENVX_PIPELINING_THREADS=4 ./benchmark_pipelining

# Real-world multi-scale pipeline
./multiscale_feature_extraction
OPENVX_PIPELINING_THREADS=1 ./multiscale_feature_extraction   # single-threaded
OPENVX_PIPELINING_THREADS=4 ./multiscale_feature_extraction   # 4 cores
```

## Adding New Samples

To add a sample:
1. Create a new subdirectory under `samples/`
2. Include a `README.md` explaining what the sample does
3. Provide a `Makefile` (or `Cargo.toml` for Rust samples)
4. Update this top-level README

## See Also

- `docs/pipelining_architecture.md` — rustVX pipelining internals
- `docs/multicore_pipeline_design.md` — wave-based execution design
- [OpenVX 1.3 Specification](https://www.khronos.org/registry/OpenVX/) — Pipelining Extension
