# rustVX

OpenVX 1.3.1 Vision Conformant Implementation in Rust

## Overview

rustVX is a complete, conformant implementation of the Khronos OpenVX 1.3.1 specification written in Rust. It provides a safe, fast, and portable vision processing framework with full Vision Feature Set support.

## Features

- **Complete OpenVX 1.3.1 API** - All base and vision functions
- **Vision Conformance** - Passes Khronos CTS for Vision Feature Set
- **Memory Safe** - Rust's ownership model prevents use-after-free, data races
- **SIMD Optimized** - SSE2/AVX2/NEON vectorized kernels
- **C API Compatible** - Drop-in replacement for existing OpenVX applications
- **Thread Safe** - Lock-free graph execution where possible

## Vision Feature Set

### Color Conversions
- vxColorConvert (RGB↔YUV, RGB↔NV12, RGB↔IYUV)
- vxChannelExtract, vxChannelCombine

### Filters
- vxConvolve (generic NxM)
- vxGaussian3x3, vxGaussian5x5
- vxMedian3x3
- vxBox3x3

### Morphology
- vxDilate3x3, vxErode3x3

### Gradients
- vxSobel3x3, vxSobel5x5
- vxMagnitude, vxPhase

### Arithmetic
- vxAdd, vxSubtract, vxMultiply
- vxWeightedAverage

### Geometric
- vxScaleImage (bilinear, nearest, area)
- vxWarpAffine, vxWarpPerspective
- vxRemap

### Optical Flow
- vxOpticalFlowPyrLK (Lucas-Kanade)

### Feature Detection
- vxHarrisCorners
- vxFASTCorners
- vxMinMaxLoc

## Building

```bash
# Clone repository
git clone https://github.com/yourusername/rustVX.git
cd rustVX

# Build with cargo
cargo build --release --all

# Run tests
cargo test --all
```

## Usage

### C API
```c
#include <VX/vx.h>

vx_context context = vxCreateContext();
vx_graph graph = vxCreateGraph(context);

// Create nodes...
vxVerifyGraph(graph);
vxProcessGraph(graph);

vxReleaseGraph(&graph);
vxReleaseContext(&context);
```

### Rust API
```rust
use rustVX::prelude::*;

let context = Context::new()?;
let mut graph = context.create_graph()?;

// Create nodes...
graph.verify()?;
graph.execute()?;
```

## Conformance

This implementation passes the Khronos OpenVX Conformance Test Suite for:
- Base Feature Set
- Vision Conformance Feature Set

See CONFORMANCE.md for detailed results.

## License

MIT License - See LICENSE file

## Acknowledgements

Based on the Khronos OpenVX specification and reference implementations from:
- Khronos Group OpenVX Sample Implementation
- AMD MIVisionX
- Texas Instruments TIOVX
