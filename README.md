<p align="center">
  <a href="https://www.khronos.org/openvx/">
    <img src="docs/openvx-logo.svg" alt="OpenVX" width="320">
  </a>
</p>

# rustVX

[![OpenVX Conformance](../../actions/workflows/conformance.yml/badge.svg?branch=develop)](../../actions/workflows/conformance.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

An [OpenVX 1.3.1](https://www.khronos.org/openvx/) implementation written in Rust. rustVX provides the complete OpenVX Vision Feature Set through a standard C API (`libopenvx_ffi`), enabling existing OpenVX applications to use a memory-safe, portable backend with no source changes.

## Conformance Status

rustVX passes the full [Khronos OpenVX 1.3 Conformance Test Suite](https://github.com/KhronosGroup/OpenVX-cts) for both required profiles:

| Profile | Required tests | Passing |
|---------|----------------|---------|
| OpenVX baseline | 863 | **863 / 863** |
| Vision conformance profile | 4957 | **4957 / 4957** |
| **Total enabled** | **5820** | **5820 / 5820** |

Latest CTS run results are published on each push and pull request via the [Actions tab](../../actions).

## Project Structure

```
rustVX/
├── openvx-core/       # Core framework: context, graph, node, types, C API
├── openvx-image/      # Image data object and channel operations
├── openvx-buffer/     # Generic buffer for intermediate data
├── openvx-vision/     # All vision kernels (filters, geometric, features, etc.)
├── openvx-ffi/        # C shared library (cdylib) exporting the full OpenVX API
├── include/VX/        # Standard OpenVX C headers
└── OpenVX-cts/        # Khronos Conformance Test Suite (git submodule)
```

The workspace compiles into a single shared library (`libopenvx_ffi.so` / `.dylib` / `.dll`) that any OpenVX application can link against.

## Prerequisites

| Tool | Version |
|------|---------|
| [Rust](https://rustup.rs/) | stable |
| C compiler | gcc, clang, or MSVC |
| [CMake](https://cmake.org/) | 3.10+ |
| Make or Ninja | for building the CTS |

## Build

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/kiritigowda/rustVX.git
cd rustVX

# Build the library
cargo build --release
```

The shared library is produced at:

| Platform | Path |
|----------|------|
| Linux | `target/release/libopenvx_ffi.so` |
| macOS | `target/release/libopenvx_ffi.dylib` |
| Windows | `target/release/openvx_ffi.dll` |

## Running Conformance Tests

The [Khronos OpenVX Conformance Test Suite](https://github.com/KhronosGroup/OpenVX-cts) is included as a git submodule. Build and run it against the rustVX library:

> [!NOTE]
> The `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag below is needed when configuring with CMake 4.0+, which dropped compatibility with the older `cmake_minimum_required` versions used by the upstream CTS. It is harmless on older CMake.

### Linux

```bash
# Build the CTS
cd OpenVX-cts
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_STANDARD_LIBRARIES="-lm" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-lm" \
  -DOPENVX_INCLUDES="$(pwd)/../../include;$(pwd)/../include" \
  -DOPENVX_LIBRARIES="$(pwd)/../../target/release/libopenvx_ffi.so;m" \
  -DOPENVX_CONFORMANCE_VISION=ON
make -j$(nproc)

# Run all tests
export LD_LIBRARY_PATH=$(pwd)/../../target/release
export VX_TEST_DATA_PATH=$(pwd)/../test_data/
./bin/vx_test_conformance
```

### macOS

```bash
# Build the CTS
cd OpenVX-cts
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOPENVX_INCLUDES="$(pwd)/../../include;$(pwd)/../include" \
  -DOPENVX_LIBRARIES="$(pwd)/../../target/release/libopenvx_ffi.dylib" \
  -DOPENVX_CONFORMANCE_VISION=ON
make -j$(sysctl -n hw.ncpu)

# Run all tests
export DYLD_LIBRARY_PATH=$(pwd)/../../target/release
export VX_TEST_DATA_PATH=$(pwd)/../test_data/
./bin/vx_test_conformance
```

### Windows (MSVC)

```powershell
# Build the CTS
cd OpenVX-cts
mkdir build; cd build
cmake .. `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 `
  -DOPENVX_INCLUDES="$PWD\..\..\include;$PWD\..\include" `
  -DOPENVX_LIBRARIES="$PWD\..\..\target\release\openvx_ffi.dll.lib" `
  -DOPENVX_CONFORMANCE_VISION=ON
cmake --build . --config Release

# Run all tests
$env:PATH = "$PWD\..\..\target\release;$env:PATH"
$env:VX_TEST_DATA_PATH = "$PWD\..\test_data\"
.\bin\Release\vx_test_conformance.exe
```

### Running Specific Test Categories

Use the `--filter` flag to run a subset of tests:

```bash
# Run only filter tests
./bin/vx_test_conformance --filter="Gaussian3x3.*:Median3x3.*:Box3x3.*"

# Run only feature detection tests
./bin/vx_test_conformance --filter="HarrisCorners.*:FastCorners.*:vxCanny.*"
```

## Continuous Integration

GitHub Actions builds and runs the full CTS on every push and pull request. The workflow splits the suite into parallel jobs for faster feedback:

| Job | Test categories |
|-----|-----------------|
| **baseline** | GraphBase, Logging, SmokeTest, Target |
| **graph** | Graph framework (cycles, virtual data, multi-run, replicate node), GraphCallback, GraphDelay, GraphROI, UserNode |
| **data-objects** | Scalar, Array, ObjectArray, Matrix, Convolution, Distribution, LUT, Histogram |
| **image-ops** | Image, CopyImagePatch, MapImagePatch, CreateImageFromChannel, Remap |
| **vision-color** | ColorConvert, ChannelExtract, ChannelCombine, ConvertDepth |
| **vision-filters** | Box, Gaussian, Median, Dilate, Erode, Sobel, Magnitude, Phase, NonLinearFilter, Convolve, EqualizeHistogram |
| **vision-arithmetic** | Add, Subtract, Multiply, Bitwise (And/Or/Xor/Not), WeightedAverage, Threshold |
| **vision-geometric** | Scale, WarpAffine, WarpPerspective, Remap, HalfScaleGaussian |
| **vision-features** | HarrisCorners, FastCorners, Canny |
| **vision-statistics** | MeanStdDev, MinMaxLoc, Integral |
| **vision-pyramid** | GaussianPyramid, LaplacianPyramid, LaplacianReconstruct, OptFlowPyrLK |

Per-job status for the latest run is visible from the [Actions tab](../../actions/workflows/conformance.yml).

## License

This project is licensed under the [MIT License](LICENSE).

The OpenVX logo is a trademark of [The Khronos Group Inc.](https://www.khronos.org/legal/trademarks) The vector logo file in `docs/openvx-logo.svg` is sourced from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:OpenVX_logo.svg) and is included for identification purposes only.
