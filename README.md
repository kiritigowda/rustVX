# rustVX

[![OpenVX Conformance](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml/badge.svg?branch=develop)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml)

An [OpenVX 1.3.1](https://www.khronos.org/openvx/) implementation written in Rust. rustVX provides the complete OpenVX Vision Feature Set through a standard C API (`libopenvx_ffi`), enabling existing OpenVX applications to use a memory-safe, portable backend with no source changes.

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

The [OpenVX Conformance Test Suite](https://github.com/simonCatBot/OpenVX-cts) is included as a git submodule. Build and run it against the rustVX library:

### Linux

```bash
# Build the CTS
cd OpenVX-cts
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
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

GitHub Actions builds and runs the full CTS on every push and pull request. The workflow splits tests into parallel jobs for faster feedback:

- **baseline** -- Graph, Logging, SmokeTest, Target
- **graph** -- Graph callbacks, delays, ROI, UserNode
- **data-objects** -- Scalar, Array, Matrix, Convolution, Distribution, LUT, Histogram
- **image-ops** -- Image, CopyImagePatch, MapImagePatch, Remap
- **vision-color** -- ColorConvert, ChannelExtract, ChannelCombine, ConvertDepth
- **vision-filters** -- Box, Gaussian, Median, Dilate, Erode, Sobel, Convolve, EqualizeHistogram, NonLinearFilter
- **vision-arithmetic** -- Add, Subtract, Multiply, Bitwise, Not, WeightedAverage, Threshold
- **vision-geometric** -- Scale, WarpAffine, WarpPerspective, Remap, HalfScaleGaussian
- **vision-features** -- HarrisCorners, FastCorners, Canny
- **vision-statistics** -- MeanStdDev, MinMaxLoc, Integral
- **vision-pyramid** -- GaussianPyramid, LaplacianPyramid, LaplacianReconstruct, OptFlowPyrLK

See the [Actions tab](https://github.com/kiritigowda/rustVX/actions) for latest results.

## License

MIT
