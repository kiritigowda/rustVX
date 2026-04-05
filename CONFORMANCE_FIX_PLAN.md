# RustVX OpenVX 1.3 Conformance Fix Plan

## Executive Summary

**Current Status**: 27/27 Baseline Tests Pass (Context, Graph, Basic Nodes)  
**Target**: Full OpenVX 1.3 Conformance (~300 API functions)  
**Gap**: ~270 functions missing or stub-only

## Phase 1: Reference Management (Week 1)

### 1.1 Core Reference Functions
These are required for proper memory management and debugging:

| Function | Status | Action |
|----------|--------|--------|
| `vxRetainReference` | ❌ Missing | Add reference count increment |
| `vxReleaseReference` | ⚠️ Partial | Fix unified registry cleanup |
| `vxQueryReference` | ❌ Missing | Add reference attribute queries |
| `vxSetReferenceName` | ❌ Missing | Add debug naming support |
| `vxGetContext` | ❌ Missing | Add context retrieval from reference |
| `vxGetStatus` | ❌ Missing | Add error status retrieval |

**Implementation**: Extend `openvx-core/src/unified_c_api.rs`
- Add `REFERENCE_COUNTS` registry integration
- Ensure all reference types (Image, Scalar, Threshold, etc.) register properly
- Add name storage in `REFERENCE_NAMES` HashMap

### 1.2 Reference Type Registration
Update all create/release functions to:
1. Register in `REFERENCE_COUNTS` on creation
2. Clean up `REFERENCE_NAMES` on release
3. Support `vxQueryReference` for type introspection

## Phase 2: Image API (Week 1-2)

### 2.1 Image Creation Functions
| Function | Priority | Notes |
|----------|----------|-------|
| `vxCreateImage` | 🔴 Critical | U8, RGB, RGBX formats |
| `vxCreateVirtualImage` | 🔴 Critical | For graph intermediate results |
| `vxCreateImageFromHandle` | 🔴 Critical | For external memory |
| `vxCreateImageFromChannel` | 🟡 Medium | Channel extraction |
| `vxCreateImageFromROI` | 🟡 Medium | Region of interest |

### 2.2 Image Access Functions
| Function | Priority | Notes |
|----------|----------|-------|
| `vxQueryImage` | 🔴 Critical | Width, height, format queries |
| `vxMapImagePatch` | 🔴 Critical | CPU read/write access |
| `vxUnmapImagePatch` | 🔴 Critical | Release mapped memory |
| `vxSetImageAttribute` | 🟡 Medium | Valid rectangle, etc. |
| `vxFormatImagePatchAddress2d` | 🔴 Critical | Patch addressing math |
| `vxGetValidRegionImage` | 🟡 Medium | Get valid pixel region |
| `vxSetImageValidRectangle` | 🟡 Medium | Set valid region |
| `vxSwapImageHandle` | 🟢 Low | Handle swapping |

**Implementation**: Extend `openvx-image/src/c_api.rs`
- Use existing `VxCImage` struct from unified_c_api
- Implement proper memory layout for planar formats
- Support NV12, RGB, U8 formats at minimum

## Phase 3: Vision Kernels (Week 2-3)

### 3.1 Filter Kernels
All currently stubs - need actual implementations:

| Kernel | Algorithm | Test Status |
|--------|-----------|-------------|
| `vxGaussian3x3` | 3x3 Gaussian blur | ❌ Fails |
| `vxBox3x3` | 3x3 box filter | ❌ Fails |
| `vxMedian3x3` | 3x3 median filter | ❌ Fails |
| `vxErode3x3` | Morphological erosion | ❌ Fails |
| `vxDilate3x3` | Morphological dilation | ❌ Fails |
| `vxConvolve` | Generic convolution | ❌ Fails |
| `vxGaussianPyramid` | Multi-scale pyramid | ❌ Fails |
| `vxScaleImage` | Resize (nearest/bilinear) | ❌ Fails |

**Implementation**: `openvx-vision/src/filter_simd.rs` or new file
- Start with simple scalar implementations
- Add SIMD optimizations (AVX2/NEON) as Phase 3b
- Ensure boundary handling matches OpenVX spec

### 3.2 Edge Detection Kernels
| Kernel | Algorithm | Priority |
|--------|-----------|----------|
| `vxSobel3x3` | Sobel edge detection | 🔴 Critical |
| `vxMagnitude` | Gradient magnitude | 🔴 Critical |
| `vxPhase` | Gradient phase | 🟡 Medium |
| `vxCannyEdgeDetector` | Canny edges | 🟡 Medium |
| `vxHarrisCorners` | Harris corner detection | 🟡 Medium |
| `vxFastCorners` | FAST corners | 🟢 Low |

### 3.3 Arithmetic/Logic Kernels
| Kernel | Operation | Priority |
|--------|-----------|----------|
| `vxAdd` | Element-wise add | 🔴 Critical |
| `vxSubtract` | Element-wise subtract | 🔴 Critical |
| `vxMultiply` | Element-wise multiply | 🔴 Critical |
| `vxAnd` | Bitwise AND | 🔴 Critical |
| `vxOr` | Bitwise OR | 🔴 Critical |
| `vxXor` | Bitwise XOR | 🟡 Medium |
| `vxNot` | Bitwise NOT | 🟡 Medium |
| `vxAbsDiff` | Absolute difference | 🟡 Medium |

### 3.4 Color/Conversion Kernels
| Kernel | Operation | Priority |
|--------|-----------|----------|
| `vxColorConvert` | Color space conversion | 🔴 Critical |
| `vxChannelExtract` | Extract single channel | 🟡 Medium |
| `vxChannelCombine` | Combine channels | 🟡 Medium |
| `vxConvertDepth` | Bit depth conversion | 🟡 Medium |
| `vxThreshold` | Binary thresholding | 🔴 Critical |

## Phase 4: Data Objects (Week 3)

### 4.1 Scalar Objects
| Function | Status |
|----------|--------|
| `vxCreateScalar` | ❌ Missing |
| `vxReleaseScalar` | ❌ Missing |
| `vxQueryScalar` | ❌ Missing |
| `vxCopyScalar` | ❌ Missing |
| `vxCreateVirtualScalar` | ❌ Missing |

**Implementation**: New file `openvx-core/src/scalar.rs`

### 4.2 Threshold Objects
| Function | Status |
|----------|--------|
| `vxCreateThreshold` | ❌ Missing |
| `vxCreateThresholdForImage` | ❌ Missing |
| `vxReleaseThreshold` | ❌ Missing |
| `vxQueryThreshold` | ❌ Missing |
| `vxSetThresholdAttribute` | ❌ Missing |
| `vxCopyThresholdValue` | ❌ Missing |
| `vxCopyThresholdRange` | ❌ Missing |
| `vxCopyThresholdOutput` | ❌ Missing |

**Implementation**: New file `openvx-core/src/threshold.rs`

### 4.3 Array Objects
| Function | Status |
|----------|--------|
| `vxCreateArray` | ❌ Missing |
| `vxCreateVirtualArray` | ❌ Missing |
| `vxReleaseArray` | ❌ Missing |
| `vxQueryArray` | ❌ Missing |
| `vxAddArrayItems` | ❌ Missing |
| `vxTruncateArray` | ❌ Missing |
| `vxCopyArrayRange` | ❌ Missing |
| `vxMapArrayRange` | ❌ Missing |
| `vxUnmapArrayRange` | ❌ Missing |

**Implementation**: New file `openvx-core/src/array.rs`

### 4.4 Convolution Objects
| Function | Status |
|----------|--------|
| `vxCreateConvolution` | ❌ Missing |
| `vxCreateVirtualConvolution` | ❌ Missing |
| `vxReleaseConvolution` | ❌ Missing |
| `vxQueryConvolution` | ❌ Missing |
| `vxSetConvolutionAttribute` | ❌ Missing |
| `vxCopyConvolutionCoefficients` | ❌ Missing |

**Implementation**: New file `openvx-core/src/convolution.rs`

### 4.5 Matrix Objects
| Function | Status |
|----------|--------|
| `vxCreateMatrix` | ❌ Missing |
| `vxCreateMatrixFromPattern` | ❌ Missing |
| `vxCreateMatrixFromPatternAndOrigin` | ❌ Missing |
| `vxCreateVirtualMatrix` | ❌ Missing |
| `vxReleaseMatrix` | ❌ Missing |
| `vxQueryMatrix` | ❌ Missing |
| `vxCopyMatrix` | ❌ Missing |

**Implementation**: New file `openvx-core/src/matrix.rs`

### 4.6 LUT Objects
| Function | Status |
|----------|--------|
| `vxCreateLUT` | ❌ Missing |
| `vxCreateVirtualLUT` | ❌ Missing |
| `vxReleaseLUT` | ❌ Missing |
| `vxQueryLUT` | ❌ Missing |
| `vxCopyLUT` | ❌ Missing |
| `vxMapLUT` | ❌ Missing |
| `vxUnmapLUT` | ❌ Missing |

**Implementation**: New file `openvx-core/src/lut.rs`

### 4.7 Pyramid Objects
| Function | Status |
|----------|--------|
| `vxCreatePyramid` | ❌ Missing |
| `vxCreateVirtualPyramid` | ❌ Missing |
| `vxReleasePyramid` | ❌ Missing |
| `vxQueryPyramid` | ❌ Missing |
| `vxGetPyramidLevel` | ❌ Missing |

**Implementation**: New file `openvx-core/src/pyramid.rs`

### 4.8 Distribution Objects
| Function | Status |
|----------|--------|
| `vxCreateDistribution` | ❌ Missing |
| `vxCreateVirtualDistribution` | ❌ Missing |
| `vxReleaseDistribution` | ❌ Missing |
| `vxQueryDistribution` | ❌ Missing |
| `vxCopyDistribution` | ❌ Missing |
| `vxMapDistribution` | ❌ Missing |
| `vxUnmapDistribution` | ❌ Missing |

**Implementation**: New file `openvx-core/src/distribution.rs`

### 4.9 Remap Objects
| Function | Status |
|----------|--------|
| `vxCreateRemap` | ❌ Missing |
| `vxCreateVirtualRemap` | ❌ Missing |
| `vxReleaseRemap` | ❌ Missing |
| `vxQueryRemap` | ❌ Missing |
| `vxCopyRemapPatch` | ❌ Missing |
| `vxMapRemapPatch` | ❌ Missing |
| `vxUnmapRemapPatch` | ❌ Missing |

**Implementation**: New file `openvx-core/src/remap.rs`

## Phase 5: Graph/Node Management (Week 4)

### 5.1 Node Functions
| Function | Status | Notes |
|----------|--------|-------|
| `vxQueryNode` | ❌ Missing | Query node attributes |
| `vxReleaseNode` | ❌ Missing | Proper cleanup |
| `vxAssignNodeCallback` | 🟡 Medium | User callbacks |
| `vxRetrieveNodeCallback` | 🟡 Medium | Get callback |
| `vxSetNodeAttribute` | ⚠️ Partial | Complete implementation |
| `vxReplicateNode` | ❌ Missing | Batch processing |
| `vxSetNodeTarget` | ⚠️ Partial | Target selection |

### 5.2 Graph Execution
| Function | Status | Notes |
|----------|--------|-------|
| `vxScheduleGraph` | ⚠️ Partial | Async execution |
| `vxWaitGraph` | ⚠️ Partial | Wait for completion |
| `vxIsGraphVerified` | ⚠️ Partial | Verification check |

## Phase 6: Kernel Management (Week 4)

### 6.1 Kernel Loading
| Function | Status |
|----------|--------|
| `vxLoadKernels` | ❌ Missing |
| `vxUnloadKernels` | ❌ Missing |
| `vxGetKernelByName` | ❌ Missing |
| `vxGetKernelByEnum` | ❌ Missing |
| `vxQueryKernel` | ❌ Missing |
| `vxGetKernelParameterByIndex` | ❌ Missing |

### 6.2 User Kernels
| Function | Status |
|----------|--------|
| `vxAddUserKernel` | ⚠️ Partial |
| `vxFinalizeKernel` | ❌ Missing |
| `vxRemoveKernel` | ❌ Missing |
| `vxSetKernelAttribute` | ⚠️ Partial |
| `vxAddParameterToKernel` | ❌ Missing |

## Phase 7: Testing & Validation (Week 5)

### 7.1 CTS Test Groups
Run and fix failures in each group:

1. **GraphBase** - 14 tests (currently passing)
2. **SmokeTestBase** - 7 tests (currently passing)
3. **TargetBase** - 3 tests (currently passing)
4. **Image** - Image operations
5. **Filter** - Gaussian, Box, Median, Erode, Dilate
6. **Edge** - Sobel, Magnitude, Canny
7. **Arithmetic** - Add, Subtract, Multiply
8. **Color** - ColorConvert, ChannelExtract
9. **Histogram** - Histogram, EqualizeHist
10. **Integral** - IntegralImage
11. **Warp** - WarpAffine, WarpPerspective, Remap
12. **OpticalFlow** - OpticalFlowPyrLK

### 7.2 Performance Optimization
- Add SIMD implementations (AVX2 for x86_64, NEON for ARM)
- Optimize memory access patterns
- Add multi-threading for large images

## Implementation Priority

### Critical (Must Have for Conformance)
1. Reference management (vxRetainReference, vxReleaseReference, etc.)
2. Image creation and access (vxCreateImage, vxMapImagePatch, etc.)
3. Vision kernels: Gaussian3x3, Sobel3x3, Add, Threshold
4. Data objects: Scalar, Threshold, Array

### Medium (Important)
1. More vision kernels: ColorConvert, Box3x3, Median3x3
2. Data objects: Convolution, Matrix, LUT
3. Node management improvements

### Low (Nice to Have)
1. Advanced kernels: OpticalFlow, WarpPerspective
2. Pyramid, Remap, Distribution
3. User kernel support

## Files to Modify/Create

### Modify Existing
1. `openvx-ffi/src/lib.rs` - Ensure all symbols exported
2. `openvx-core/src/c_api.rs` - Add missing functions
3. `openvx-core/src/unified_c_api.rs` - Fix reference management
4. `openvx-image/src/c_api.rs` - Complete image API
5. `openvx-vision/src/filter_simd.rs` - Add kernel implementations

### Create New
1. `openvx-core/src/scalar.rs`
2. `openvx-core/src/threshold.rs`
3. `openvx-core/src/array.rs`
4. `openvx-core/src/convolution.rs`
5. `openvx-core/src/matrix.rs`
6. `openvx-core/src/lut.rs`
7. `openvx-core/src/pyramid.rs`
8. `openvx-core/src/distribution.rs`
9. `openvx-core/src/remap.rs`

## Success Criteria

1. All 25 baseline tests pass (✅ Already achieved)
2. All Image tests pass
3. All Filter tests pass
4. All Edge Detection tests pass
5. All Arithmetic tests pass
6. All Color tests pass
7. All Data Object tests pass
8. Linker errors resolved (all vx* symbols exported)

## Next Steps

1. Start with **Phase 1: Reference Management** (core infrastructure)
2. Move to **Phase 2: Image API** (required for all vision tests)
3. Implement **Phase 3: Vision Kernels** (actual algorithms)
4. Add **Phase 4: Data Objects** (supporting types)
5. Fix **Phase 5: Graph/Node** management
6. Complete **Phase 6: Kernel Management**
7. **Phase 7: Testing** until all CTS tests pass

---

**Estimated Timeline**: 5 weeks  
**Risk**: Vision kernel algorithms require careful implementation to match Khronos reference results