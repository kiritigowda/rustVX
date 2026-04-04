# OpenVX CTS Test Groups - Implementation Plan

## Analysis: 15,277 tests from 69 test cases

## Logical Grouping Strategy

### Group 1: Core Framework (Foundation) ✅ MOSTLY DONE
**Test Count:** ~50 tests
**Priority:** CRITICAL (blocks everything else)

**Includes:**
- GraphBase.* (14 tests) - ✅ PASSING
- SmokeTestBase.* (7 tests) - ✅ PASSING  
- TargetBase.* (3 tests)
- SmokeTest.* (14 tests) - ⚠️ 7 failing

**APIs Required:**
- vxCreateContext, vxReleaseContext
- vxCreateGraph, vxReleaseGraph, vxQueryGraph
- vxCreateReference, vxRetainReference, vxReleaseReference
- vxQueryReference, vxSetReferenceName
- vxGetStatus, vxGetContext
- vxRegisterUserStruct, vxRegisterLogCallback
- vxHint

**Status:** 
- ✅ 14 GraphBase tests passing
- ✅ 7 SmokeTestBase tests passing
- ⚠️ 7 SmokeTest tests failing (needs fixes)

---

### Group 2: Graph Management & Execution
**Test Count:** ~150 tests
**Priority:** HIGH (foundation for kernels)

**Includes:**
- Graph.TwoNodes, Graph.GraphFactory
- Graph.VirtualImage, Graph.VirtualArray
- Graph.NodeRemove, Graph.NodeFromEnum
- Graph.MultipleRun, Graph.MultipleRunAsync
- Graph.NodePerformance, Graph.GraphPerformance
- Graph.ReplicateNode/* (16 tests)
- Graph.KernelName/* (42 tests)

**APIs Required:**
- vxCreateNode, vxQueryNode, vxSetNodeAttribute
- vxAddParameterToKernel, vxSetParameterByIndex
- vxVerifyGraph, vxProcessGraph
- vxReplicateNode
- Graph performance queries

**Dependencies:** Group 1

---

### Group 3: Image Management
**Test Count:** ~400 tests
**Priority:** HIGH (most vision data)

**Includes:**
- Image.* (230 tests)
- vxCreateImageFromChannel.* (54 tests)
- vxMapImagePatch.* (156 tests)
- vxCopyImagePatch.* (117 tests)
- vxuCopyImagePatch

**APIs Required:**
- vxCreateImage, vxCreateVirtualImage, vxCreateImageFromROI
- vxQueryImage (all attributes)
- vxMapImagePatch, vxUnmapImagePatch
- vxCopyImagePatch
- vxCreateImageFromChannel
- Image format conversions

**Dependencies:** Group 1

---

### Group 4: Vision Kernels - Color & Channel
**Test Count:** ~100 tests
**Priority:** MEDIUM

**Includes:**
- ColorConvert.* (56 tests)
- ChannelExtract.* (51 tests)
- ChannelCombine.* (17 tests)

**APIs Required:**
- ColorConvert (RGB↔YUV, RGB↔Gray, etc.)
- ChannelExtract, ChannelCombine

**Dependencies:** Group 3 (images)

---

### Group 5: Vision Kernels - Filters
**Test Count:** ~1,100 tests
**Priority:** HIGH (core vision operations)

**Includes:**
- Convolve.* (1009 tests) - Custom convolution
- NonLinearFilter.* (172 tests)
- Box3x3.* (23 tests)
- Gaussian3x3.* (from Graph.KernelName)
- Median3x3.* (12 tests)
- Dilate3x3.* (12 tests)
- Erode3x3.* (12 tests)
- HalfScaleGaussian.* (25 tests)

**APIs Required:**
- vxConvolve
- vxBox3x3, vxGaussian3x3, vxMedian3x3
- vxDilate3x3, vxErode3x3
- vxNonLinearFilter
- vxHalfScaleGaussian
- vxCreateConvolution, vxCopyConvolutionCoefficients

**Dependencies:** Group 3 (images), Group 4 (channels)

---

### Group 6: Vision Kernels - Geometric
**Test Count:** ~1,700 tests
**Priority:** HIGH

**Includes:**
- Scale.* (982 tests)
- WarpAffine.* (305 tests)
- WarpPerspective.* (361 tests)
- Remap.* (380 tests)

**APIs Required:**
- vxScaleImage
- vxWarpAffine, vxWarpPerspective
- vxRemap
- vxCreateMatrix, vxCopyMatrix (for affine/perspective)
- vxCreateRemap

**Dependencies:** Group 3 (images), Group 7 (matrices for affine)

---

### Group 7: Data Objects & Buffers
**Test Count:** ~300 tests
**Priority:** MEDIUM

**Includes:**
- Array.* (23 tests)
- Scalar.* (102 tests)
- Matrix.* (13 tests)
- LUT.* (38 tests)
- Threshold.* (20 tests)
- vxCreateDistribution
- ObjectArray.* (12 tests)

**APIs Required:**
- vxCreateArray, vxQueryArray, vxCopyArray
- vxCreateScalar, vxQueryScalar, vxCopyScalar
- vxCreateMatrix, vxQueryMatrix, vxCopyMatrix
- vxCreateLUT, vxQueryLUT, vxCopyLUT
- vxCreateThreshold, vxQueryThreshold
- vxCreateDistribution
- vxCreateObjectArray

**Dependencies:** Group 1

---

### Group 8: Vision Kernels - Arithmetic
**Test Count:** ~500 tests
**Priority:** MEDIUM

**Includes:**
- vxAddSub.* (76 tests)
- vxuAddSub.* (60 tests)
- vxMultiply.* (306 tests)
- vxuMultiply.* (170 tests)
- WeightedAverage.* (102 tests)
- vxConvertDepth.* (20 tests)
- vxuConvertDepth.* (20 tests)

**APIs Required:**
- vxAdd, vxSubtract
- vxMultiply (with overflow policy)
- vxWeightedAverage
- vxConvertDepth

**Dependencies:** Group 3 (images), Group 7 (scalars for policies)

---

### Group 9: Vision Kernels - Feature Detection
**Test Count:** ~800 tests
**Priority:** MEDIUM

**Includes:**
- HarrisCorners.* (433 tests)
- FastCorners.* (24 tests)
- vxCanny.* (28 tests)
- vxuCanny.* (28 tests)

**APIs Required:**
- vxHarrisCorners
- vxFastCorners
- vxCannyEdgeDetector
- vxCreateDistribution (for histogram in Canny)

**Dependencies:** Group 3 (images), Group 7 (distributions)

---

### Group 10: Pyramid Operations
**Test Count:** ~100 tests
**Priority:** MEDIUM

**Includes:**
- GaussianPyramid.* (25 tests)
- LaplacianPyramid.*
- LaplacianReconstruct.*
- Graph.ReplicateNode/* (uses pyramids)

**APIs Required:**
- vxCreatePyramid, vxQueryPyramid
- vxGetPyramidLevel
- vxGaussianPyramid
- vxLaplacianPyramid, vxLaplacianReconstruct

**Dependencies:** Group 3 (images), Group 5 (gaussian)

---

### Group 11: Advanced Features
**Test Count:** ~200 tests
**Priority:** LOW (optional)

**Includes:**
- UserNode.* (74 tests)
- GraphDelay.* (12 tests)
- IntegralImage.*
- Histogram.*
- EqualizeHistogram.*
- TableLookup.*
- MinMaxLoc.*
- MeanStdDev.*
- AbsDiff.*
- And, Or, Xor, Not

**APIs Required:**
- User kernels
- vxDelay, vxAgeDelay
- vxIntegralImage
- vxHistogram, vxEqualizeHistogram
- vxTableLookup
- vxMinMaxLoc
- vxMeanStdDev
- vxAbsDiff
- Bitwise operations

**Dependencies:** Multiple groups

---

## Implementation Strategy

### Phase 1: Complete Core (Groups 1-2) ✅ IN PROGRESS
- Finish remaining SmokeTest fixes
- Complete parameter/node functions
- Goal: 14/14 SmokeTest passing

### Phase 2: Image Foundation (Group 3)
- Complete image API
- Fix remaining image patch operations
- Goal: Image.* tests passing

### Phase 3: Vision Core (Groups 4-6)
- Implement color conversions
- Implement filters (box, gaussian, etc.)
- Implement geometric transforms
- Goal: Filter and geometric tests passing

### Phase 4: Data & Arithmetic (Groups 7-8)
- Complete all data objects
- Implement arithmetic kernels
- Goal: Array, Scalar, arithmetic tests passing

### Phase 5: Advanced Vision (Groups 9-10)
- Feature detection (Harris, FAST, Canny)
- Pyramid operations
- Goal: Feature detection tests passing

### Phase 6: Polish (Group 11)
- User kernels
- Remaining advanced features
- Goal: Full conformance

---

## Current Status Summary

| Group | Tests | Status | Priority |
|-------|-------|--------|----------|
| 1. Core Framework | ~50 | ⚠️ 7 failing | CRITICAL |
| 2. Graph Management | ~150 | Not started | HIGH |
| 3. Image Management | ~400 | Partial | HIGH |
| 4. Color/Channel | ~100 | Partial | MEDIUM |
| 5. Filters | ~1,100 | Partial | HIGH |
| 6. Geometric | ~1,700 | Partial | HIGH |
| 7. Data Objects | ~300 | Partial | MEDIUM |
| 8. Arithmetic | ~500 | Partial | MEDIUM |
| 9. Feature Detection | ~800 | Not started | MEDIUM |
| 10. Pyramid | ~100 | Partial | MEDIUM |
| 11. Advanced | ~200 | Not started | LOW |

**Total:** ~4,700 tests in scope for Vision Conformance
