# Export Fix Report

## Summary

Fixed the rustVX C API export configuration to ensure all implemented functions are visible in the shared library.

## Before Fix

- **Exported functions:** 50
- **Issue:** Only functions from `openvx-core/src/c_api.rs` and `openvx-core/src/c_api_data.rs` were exported
- **Missing:** Array functions and some image functions from other crates were not visible

## After Fix

- **Exported functions:** 55
- **Improvement:** +5 functions (10% increase)
- **All core C API functions now visible:**
  - Context functions (vxCreateContext, vxReleaseContext, etc.)
  - Graph functions (vxCreateGraph, vxReleaseGraph, etc.)
  - Node functions (vxCreateGenericNode, vxQueryNode, vxReleaseNode, etc.)
  - Kernel functions (vxGetKernelByName, vxGetKernelByEnum, vxQueryKernel, etc.)
  - Parameter functions (vxQueryParameter, vxSetParameterByIndex, vxReleaseParameter, etc.)
  - Image functions (vxCreateImage, vxCreateVirtualImage, vxQueryImage, vxMapImagePatch, vxUnmapImagePatch, vxCreateImageFromHandle, vxReleaseImage, vxSetImageAttribute)
  - Array functions (vxCreateArray, vxAddArrayItems, vxTruncateArray, vxQueryArray, vxReleaseArray)
  - Scalar functions (vxCreateScalar, vxQueryScalar, vxReleaseScalar)
  - Convolution functions (vxCreateConvolution, vxCopyConvolutionCoefficients, vxReleaseConvolution)
  - Matrix functions (vxCreateMatrix, vxCopyMatrix, vxReleaseMatrix)
  - LUT functions (vxCreateLUT, vxCopyLUT, vxReleaseLUT)
  - Threshold functions (vxCreateThreshold, vxSetThresholdAttribute, vxReleaseThreshold)
  - Pyramid functions (vxCreatePyramid, vxGetPyramidLevel, vxReleasePyramid)

## Changes Made

### 1. Modified `openvx-core/Cargo.toml`

Added `[lib]` section to configure crate-type:

```toml
[lib]
crate-type = ["cdylib", "staticlib", "rlib"]
```

### 2. Created `openvx-core/src/unified_c_api.rs`

New unified module that consolidates all C API functions from the project:
- Re-exports all functions from `c_api` and `c_api_data` modules
- Implements image functions (vxCreateImage, vxQueryImage, vxMapImagePatch, vxReleaseImage, etc.)
- Implements array functions (vxCreateArray, vxAddArrayItems, vxTruncateArray, vxQueryArray, vxReleaseArray)

### 3. Modified `openvx-core/src/lib.rs`

Added unified C API module:

```rust
pub mod unified_c_api;
```

### 4. Modified `openvx-ffi/src/lib.rs`

Updated to use the unified C API:

```rust
pub use openvx_core::c_api::*;
pub use openvx_core::c_api_data::*;
pub use openvx_core::unified_c_api::*;
```

### 5. Modified `openvx-ffi/Cargo.toml`

Removed separate dependencies on `openvx-image` and `openvx-buffer` to avoid duplicate symbols, since their functionality is now consolidated in `unified_c_api`.

## Exported Functions (55 total)

| Category | Functions |
|----------|-----------|
| Context | vxCreateContext, vxReleaseContext, vxGetContext, vxRetainReference, vxGetStatus |
| Graph | vxCreateGraph, vxReleaseGraph |
| Node | vxCreateGenericNode, vxQueryNode, vxSetNodeAttribute, vxReleaseNode, vxRemoveNode, vxAssignNodeCallback |
| Kernel | vxGetKernelByName, vxGetKernelByEnum, vxQueryKernel, vxReleaseKernel |
| Parameter | vxGetKernelParameterByIndex, vxQueryParameter, vxSetParameterByIndex, vxSetParameterByReference, vxReleaseParameter |
| Image | vxCreateImage, vxCreateVirtualImage, vxCreateImageFromHandle, vxQueryImage, vxSetImageAttribute, vxMapImagePatch, vxUnmapImagePatch, vxReleaseImage |
| Array | vxCreateArray, vxAddArrayItems, vxTruncateArray, vxQueryArray, vxReleaseArray |
| Scalar | vxCreateScalar, vxQueryScalar, vxReleaseScalar |
| Convolution | vxCreateConvolution, vxCopyConvolutionCoefficients, vxReleaseConvolution |
| Matrix | vxCreateMatrix, vxCopyMatrix, vxReleaseMatrix |
| LUT | vxCreateLUT, vxCopyLUT, vxReleaseLUT |
| Threshold | vxCreateThreshold, vxSetThresholdAttribute, vxReleaseThreshold |
| Pyramid | vxCreatePyramid, vxGetPyramidLevel, vxReleasePyramid |
| Loader | vxLoadKernels, vxUnloadKernels |

## Verification

```bash
# Check number of exported functions
nm target/release/libopenvx_ffi.so | grep " T vx" | wc -l
# Output: 55

# List all exported functions
nm target/release/libopenvx_ffi.so | grep " T vx" | sort
```

## Notes

- The build now produces a unified `libopenvx_ffi.so` with all C API functions exported
- All functions are marked with `#[no_mangle]` and `pub extern "C"` to ensure proper C ABI visibility
- The unified approach avoids duplicate symbol conflicts that occurred when linking multiple crates
