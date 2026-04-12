# rustVX OpenVX 1.3.1 Conformance Plan

**Branch:** fix/openvx-1.3.1-conformance  
**Created:** April 12, 2026  
**Goal:** Achieve OpenVX 1.3.1 Full Vision Conformance

---

## Current Status

### Baseline Tests (Required for any conformance)
- **22/24 tests passing** (91.7%)
- **2 failing:** TargetBase.vxCreateContext, TargetBase.vxReleaseContext
- **Root cause:** Reference counting overflow (dangling_refs_count: 4294967254)

### Vision Feature Set
- ~4,700 tests across 11 groups
- Some groups partially working, need fixes for full conformance

---

## Plan

### Step 1: [FOUNDATION] Fix Baseline Reference Counting Issues
**Dependencies:** None  
**Priority:** CRITICAL - Blocks all conformance

**Problem:** TargetBase tests failing due to reference counting overflow
- `Expected: 0 == dangling_refs_count`
- `Actual: 0 != 4294967254`

**Approach:**
1. Investigate unified_c_api.rs reference counting
2. Check vxCreateContext for proper initialization
3. Check vxReleaseContext for proper cleanup
4. Ensure REFERENCE_COUNTS registry is properly initialized/cleared

**Verification:** All 24 Baseline tests pass
**Files:** openvx-core/src/unified_c_api.rs

---

### Step 2: [GROUP 2] Graph Management & Kernel Functions
**Dependencies:** Step 1
**Priority:** HIGH

**Scope:** 
- Graph.TwoNodes, Graph.GraphFactory
- Graph.VirtualImage, Graph.VirtualArray
- Graph.NodeRemove, Graph.NodeFromEnum
- Graph.MultipleRun, Graph.MultipleRunAsync
- Graph.ReplicateNode
- Graph.KernelName/* (42 kernel name tests)

**APIs to verify/fix:**
- vxCreateNode, vxQueryNode, vxSetNodeAttribute
- vxAddParameterToKernel, vxSetParameterByIndex
- vxVerifyGraph, vxProcessGraph
- vxReplicateNode

**Verification:** All Group 2 tests pass
**Files:** openvx-core/src/unified_c_api.rs (graph functions)

---

### Step 3: [GROUP 3] Image Management API
**Dependencies:** Step 2
**Priority:** HIGH

**Scope:**
- Image.* (~230 tests)
- vxCreateImageFromChannel.* (54 tests)
- vxMapImagePatch.* (156 tests)
- vxCopyImagePatch.* (117 tests)

**APIs to verify/fix:**
- vxCreateImage, vxCreateVirtualImage, vxCreateImageFromROI
- vxQueryImage (all attributes)
- vxMapImagePatch, vxUnmapImagePatch
- vxCopyImagePatch
- vxCreateImageFromChannel

**Verification:** All Image tests pass
**Files:** 
- openvx-core/src/unified_c_api.rs (image functions)
- openvx-image/src/*.rs

---

### Step 4: [GROUP 4] Color & Channel Operations
**Dependencies:** Step 3
**Priority:** MEDIUM

**Scope:**
- ColorConvert.* (56 tests) - Currently 14/56 passing
- ChannelExtract.* (51 tests) - Currently 13/51 passing
- ChannelCombine.* (17 tests)

**APIs to verify/fix:**
- ColorConvert (RGB↔YUV, RGB↔Gray, etc.)
- ChannelExtract, ChannelCombine

**Verification:** ColorConvert and Channel tests pass
**Files:** openvx-vision/src/color.rs

---

### Step 5: [GROUP 5] Vision Filters - 3x3 Operations
**Dependencies:** Step 3
**Priority:** HIGH

**Scope:**
- Box3x3.* (23 tests) - Should be working
- Gaussian3x3.* - Should be working
- Median3x3.* (12 tests) - Should be working
- Dilate3x3.* (12 tests)
- Erode3x3.* (12 tests)
- Convolve.* (1009 tests) - Custom convolution

**APIs to verify/fix:**
- vxBox3x3, vxGaussian3x3, vxMedian3x3
- vxDilate3x3, vxErode3x3
- vxConvolve (with custom kernels)
- vxCreateConvolution, vxCopyConvolutionCoefficients

**Verification:** All filter tests pass
**Files:** openvx-vision/src/filters.rs, openvx-vision/src/convolution.rs

---

### Step 6: [GROUP 6] Geometric Operations
**Dependencies:** Step 3, Step 7 (matrices)
**Priority:** HIGH

**Scope:**
- Scale.* (982 tests) - Currently working well
- WarpAffine.* (305 tests)
- WarpPerspective.* (361 tests)
- Remap.* (380 tests)

**APIs to verify/fix:**
- vxScaleImage
- vxWarpAffine, vxWarpPerspective
- vxRemap
- vxCreateMatrix, vxCopyMatrix (for affine/perspective)
- vxCreateRemap

**Verification:** All geometric tests pass
**Files:** openvx-vision/src/geometric.rs

---

### Step 7: [GROUP 7] Data Objects & Buffers
**Dependencies:** Step 1
**Priority:** MEDIUM (blocks some vision kernels)

**Scope:**
- Array.* (23 tests)
- Scalar.* (102 tests)
- Matrix.* (13 tests)
- LUT.* (38 tests)
- Threshold.* (20 tests)
- Distribution.* tests

**APIs to verify/fix:**
- vxCreateArray, vxQueryArray, vxCopyArray
- vxCreateScalar, vxQueryScalar, vxCopyScalar
- vxCreateMatrix, vxQueryMatrix, vxCopyMatrix
- vxCreateLUT, vxQueryLUT, vxCopyLUT
- vxCreateThreshold, vxQueryThreshold
- vxCreateDistribution, vxQueryDistribution

**Verification:** All data object tests pass
**Files:** 
- openvx-core/src/unified_c_api.rs (data object functions)
- openvx-buffer/src/*.rs

---

### Step 8: [GROUP 8] Arithmetic Operations
**Dependencies:** Step 3 (images), Step 7 (scalars)
**Priority:** MEDIUM

**Scope:**
- vxAddSub.* (76 tests)
- vxuAddSub.* (60 tests) - VXU immediate functions
- vxMultiply.* (306 tests)
- vxuMultiply.* (170 tests)
- WeightedAverage.* (102 tests)
- vxConvertDepth.* (20 tests)

**APIs to verify/fix:**
- vxAdd, vxSubtract
- vxMultiply (with overflow policy)
- vxWeightedAverage
- vxConvertDepth
- VXU immediate versions (vxuAdd, vxuSubtract, etc.)

**Verification:** All arithmetic tests pass
**Files:** 
- openvx-vision/src/arithmetic.rs
- openvx-core/src/vxu_impl.rs (VXU functions)

---

### Step 9: [GROUP 9] Feature Detection
**Dependencies:** Step 3 (images), Step 7 (distributions)
**Priority:** MEDIUM

**Scope:**
- HarrisCorners.* (433 tests)
- FastCorners.* (24 tests)
- vxCanny.* (28 tests)

**APIs to verify/fix:**
- vxHarrisCorners (needs accuracy tuning)
- vxFastCorners (needs accuracy tuning)
- vxCannyEdgeDetector (needs accuracy tuning)
- vxCreateDistribution (for histogram in Canny)

**Note:** These tests work but need algorithm accuracy tuning to match reference within 2% tolerance

**Verification:** Feature detection tests pass with acceptable tolerance
**Files:** openvx-vision/src/feature_detection.rs

---

### Step 10: [GROUP 10] Pyramid Operations
**Dependencies:** Step 3 (images), Step 5 (gaussian)
**Priority:** MEDIUM

**Scope:**
- GaussianPyramid.* (25 tests)
- LaplacianPyramid.*
- LaplacianReconstruct.*
- HalfScaleGaussian.* (25 tests)

**APIs to verify/fix:**
- vxCreatePyramid, vxQueryPyramid
- vxGetPyramidLevel
- vxGaussianPyramid
- vxLaplacianPyramid, vxLaplacianReconstruct
- vxHalfScaleGaussian

**Verification:** All pyramid tests pass
**Files:** openvx-vision/src/pyramid.rs

---

### Step 11: [GROUP 11] Advanced Features & Non-Linear Filters
**Dependencies:** Step 3-10
**Priority:** LOW (optional but good for completeness)

**Scope:**
- NonLinearFilter.* (172 tests)
- IntegralImage.*
- Histogram.*
- EqualizeHistogram.*
- TableLookup.*
- MinMaxLoc.*
- MeanStdDev.*
- AbsDiff.*
- Bitwise operations (And, Or, Xor, Not)

**APIs to verify/fix:**
- vxNonLinearFilter
- vxIntegralImage
- vxHistogram, vxEqualizeHistogram
- vxTableLookup
- vxMinMaxLoc, vxMeanStdDev
- vxAbsDiff
- Bitwise operations

**Verification:** Advanced tests pass
**Files:** openvx-vision/src/

---

### Step 12: [INTEGRATION] Full CTS Run & Final Verification
**Dependencies:** Steps 1-11
**Priority:** CRITICAL

**Approach:**
1. Run full CTS suite
2. Document any remaining failures
3. Fix critical issues
4. Verify no regressions

**Verification:** 
- All Baseline tests pass
- All Vision Feature Set tests pass
- No memory leaks or crashes

---

## Risk Analysis

### Blockers
1. **Reference counting overflow** - Critical baseline issue
2. **Algorithm accuracy** - Feature detection needs tuning
3. **Color conversion coverage** - Limited format support
4. **Memory issues** - Potential leaks in image operations

### Mitigation
1. Fix reference counting first (Step 1)
2. Use reference implementation for accuracy comparison
3. Prioritize common color formats (RGB, NV12, YUV)
4. Run Valgrind/ASan for memory validation

---

## Rollback Plan

**If tests fail:**
1. Each step is isolated to specific files
2. Git branches allow reverting individual steps
3. Run filtered tests to isolate issues
4. Compare with reference implementation

---

## Success Criteria

### Phase 1 (Baseline):
- [ ] All 24 Baseline tests pass (100%)

### Phase 2 (Vision Core):
- [ ] Groups 2-6: 80%+ tests passing
- [ ] Image, Filter, Geometric operations working

### Phase 3 (Data & Arithmetic):
- [ ] Groups 7-8: 80%+ tests passing
- [ ] All data objects working

### Phase 4 (Advanced):
- [ ] Groups 9-11: 70%+ tests passing
- [ ] Feature detection with acceptable accuracy

### Final:
- [ ] Full Vision Conformance achieved
- [ ] No regressions in passing tests
- [ ] Memory safe (no leaks)

---

## Test Commands

```bash
# Set environment
export LD_LIBRARY_PATH=/home/simon/.openclaw/workspace/rustVX/target/release
export VX_TEST_DATA_PATH=/home/simon/.openclaw/workspace/rustVX/OpenVX-cts/test_data

# Baseline tests
./bin/vx_test_conformance --filter="*Base*"

# Individual groups
./bin/vx_test_conformance --filter="*Graph.TwoNodes*,*Graph.GraphFactory*"
./bin/vx_test_conformance --filter="*Image.*"
./bin/vx_test_conformance --filter="*ColorConvert*,*ChannelExtract*"
./bin/vx_test_conformance --filter="*Box3x3*,*Gaussian3x3*,*Median3x3*"
./bin/vx_test_conformance --filter="*Scale*,*Warp*,*Remap*"
./bin/vx_test_conformance --filter="*Harris*,*Fast*,*Canny*"

# Full run
./bin/vx_test_conformance
```

---

## Team Assignment Strategy

**Using Team Code with 4-6 agents in parallel:**

1. **Agent 1 (Foundation):** Step 1 - Fix reference counting
2. **Agent 2 (Graph & Vision Core):** Steps 2, 5, 6 - Graph, Filters, Geometric
3. **Agent 3 (Images & Color):** Steps 3, 4 - Image API, Color operations
4. **Agent 4 (Data Objects):** Step 7 - Arrays, Scalars, Matrices, etc.
5. **Agent 5 (Arithmetic & VXU):** Step 8 - Arithmetic operations, VXU fixes
6. **Agent 6 (Advanced):** Steps 9-11 - Feature detection, Pyramids, Advanced

**Manager (me):** Coordination, integration, final verification

---

*Plan created: April 12, 2026*  
*Target: OpenVX 1.3.1 Full Vision Conformance*
