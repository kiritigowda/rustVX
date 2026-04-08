# RustVX OpenVX Conformance Fix Plan

## Problem Statement

The rustVX implementation fails to link because critical OpenVX API functions are missing or stubbed. While 27 baseline tests pass (context/graph), the vision API functions required for full conformance are not properly exported or implemented.

## Root Cause

1. **Exported Functions**: Only ~33 functions exported vs ~300 required for full OpenVX 1.3 conformance
2. **Missing Vision Kernels**: All vision kernels are stubs or missing
3. **Missing Data Objects**: Scalars, Thresholds, Arrays, Convolution, Matrix, LUT, Pyramid, Remap not implemented

## Functions to Add/Fix

### Reference Management (Critical)
- `vxRetainReference` - Increment reference count
- `vxReleaseReference` - Decrement reference count  
- `vxQueryReference` - Query reference attributes
- `vxSetReferenceName` - Set reference name for debugging
- `vxGetContext` - Get context from reference
- `vxGetStatus` - Get error status

### Image Objects (Critical)
- `vxCreateImage` - Create image with format
- `vxCreateVirtualImage` - Create virtual image
- `vxCreateImageFromHandle` - Create image from handle
- `vxQueryImage` - Query image attributes
- `vxMapImagePatch` - Map image for CPU access
- `vxUnmapImagePatch` - Unmap image
- `vxSetImageAttribute` - Set image attributes
- `vxFormatImagePatchAddress2d` - Calculate patch address

### Vision Kernels (Critical - Stubs need implementation)
- `vxColorConvert` - Color space conversion
- `vxGaussian3x3` - 3x3 Gaussian filter
- `vxSobel3x3` - Sobel edge detection
- `vxAdd` - Element-wise addition
- `vxThreshold` - Threshold operation
- `vxErode3x3` - Morphological erosion
- `vxDilate3x3` - Morphological dilation
- `vxAnd` - Bitwise AND
- `vxOr` - Bitwise OR
- `vxBox3x3` - Box filter
- `vxMedian3x3` - Median filter

### Data Objects (Critical)
- `vxCreateScalar` - Scalar data
- `vxCreateThreshold` - Threshold object
- `vxCreateArray` - Array object
- `vxCreateConvolution` - Convolution object
- `vxCreateMatrix` - Matrix object
- `vxCreateLUT` - Lookup table
- `vxCreatePyramid` - Image pyramid
- `vxCreateRemap` - Remap table

### Graph/Node Management
- `vxQueryNode` - Query node attributes
- `vxReleaseNode` - Release node
- `vxSetNodeAttribute` - Set node attributes
- `vxReplicateNode` - Replicate node for batch processing

### Kernel Management
- `vxLoadKernels` - Load kernel libraries
- `vxGetKernelByName` - Find kernel by name
- `vxUnloadKernels` - Unload kernel libraries
- `vxGetKernelParameterByIndex` - Access kernel parameters

## Implementation Steps

1. Add all vx* functions to FFI exports in `lib.rs`
2. Implement actual vision kernel algorithms
3. Add proper image object constructors and accessors
4. Ensure all vx* functions link correctly

Next: Update the FFI exports to include all required functions for OpenVX 1.3 conformance.