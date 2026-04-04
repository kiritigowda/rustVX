# High-Impact Conformance Plan

## Analysis: 15,277 Tests Across 69 Categories

## Top 15 Test Groups by Count

| Rank | Group | Tests | Current Status | Impact |
|------|-------|-------|----------------|--------|
| 1 | **Convolve** | 1009 | Unknown | 🔴 Critical |
| 2 | **Scale** | 982 | Partial | 🔴 Critical |
| 3 | **HarrisCorners** | 433 | Unknown | 🟡 High |
| 4 | **Remap** | 380 | Unknown | 🟡 High |
| 5 | **WarpPerspective** | 361 | Unknown | 🟡 High |
| 6 | **WarpAffine** | 305 | Partial | 🟡 High |
| 7 | **Image** | 230 | Many failing | 🟡 High |
| 8 | **NonLinearFilter** | 172 | Unknown | 🟢 Medium |
| 9 | **vxMapImagePatch** | 156 | Unknown | 🟢 Medium |
| 10 | **Graph** | 118 | Working ✅ | 🟢 Medium |
| 11 | **vxCopyImagePatch** | 117 | Unknown | 🟢 Medium |
| 12 | **WeightedAverage** | 102 | Unknown | 🟢 Medium |
| 13 | **Scalar** | 102 | Working ✅ | 🟢 Medium |
| 14 | **vxAddSub** | 76 | Unknown | 🟢 Low |
| 15 | **ColorConvert** | 56 | Failing | 🟢 Low |

**Top 5 groups = 3,485 tests (23% of total)**

---

## Strategic Fix Plan

### Phase 1: Fix Image Infrastructure (Unblocks Groups 1-7)
**Estimated Impact: +2,000 tests**

**Problem:** Many vision tests fail due to:
1. Virtual image handling
2. Image format support (S16 needed for Sobel)
3. Image patch mapping

**Fixes:**
1. ✅ Fix virtual image creation/validation
2. ✅ Add S16 format support for gradients
3. ✅ Fix image patch mapping functions
4. ✅ Handle image from ROI

**Files:**
- openvx-image/src/c_api.rs
- openvx-core/src/image_format.rs

---

### Phase 2: Geometric Transforms (Groups 2, 4, 5, 6)
**Estimated Impact: +2,028 tests**

**Groups:** Scale (982), Remap (380), WarpPerspective (361), WarpAffine (305)

**Current Status:** Partial - some implementations exist but accuracy issues

**Fixes:**
1. ✅ Fix bilinear interpolation accuracy
2. ✅ Fix border handling in geometric transforms
3. ✅ Ensure output format matches input

**Files:**
- openvx-vision/src/geometric.rs
- openvx-vision/src/transform.rs

---

### Phase 3: Convolution (Group 1 - 1,009 tests)
**Estimated Impact: +1,009 tests**

**Current Status:** Custom convolution likely has kernel handling issues

**Fixes:**
1. ✅ Fix vxConvolve kernel size handling
2. ✅ Ensure proper normalization
3. ✅ Fix convolution data type handling

**Files:**
- openvx-vision/src/filter_simd.rs
- openvx-core/src/c_api_data.rs (convolution objects)

---

### Phase 4: Feature Detection (Group 3 - HarrisCorners)
**Estimated Impact: +433 tests**

**Current Status:** Unknown, but algorithms exist in code

**Fixes:**
1. ✅ Fix HarrisCorners parameter handling
2. ✅ Ensure gradient computation accuracy
3. ✅ Fix corner response calculation

**Files:**
- openvx-vision/src/feature_detection.rs
- openvx-vision/src/object_detection.rs

---

### Phase 5: Quick Wins (Groups 8-15)
**Estimated Impact: +800 tests**

**Quick fixes for:**
- NonLinearFilter (172) - median filter
- vxMapImagePatch (156) - likely patch access
- vxCopyImagePatch (117) - data copying
- ColorConvert (56) - fix YUV/RGB conversions

**Files:**
- Various vision kernel files

---

## Total Potential Impact

| Phase | Tests | Priority |
|-------|-------|----------|
| Phase 1: Image Infrastructure | +2,000 | 🔴 Critical |
| Phase 2: Geometric Transforms | +2,028 | 🔴 Critical |
| Phase 3: Convolution | +1,009 | 🟡 High |
| Phase 4: Feature Detection | +433 | 🟡 High |
| Phase 5: Quick Wins | +800 | 🟢 Medium |

**Total: +6,270 tests (41% of total suite)**

---

## Immediate Action Plan

### Step 1: Image Format Support (This Session)
**Task:** Add S16 format for Sobel/gradient outputs
**Agent:** image-format-agent
**Time:** 30 min
**Impact:** Unlocks 306+ tests (Sobel3x3, gradients)

### Step 2: Virtual Image Fix (This Session)
**Task:** Fix vxCreateVirtualImage validation
**Agent:** virtual-image-agent
**Time:** 30 min
**Impact:** Unlocks 230 Image tests

### Step 3: Geometric Transform Accuracy (Next Session)
**Task:** Fix Scale, WarpAffine, WarpPerspective
**Agent:** geometric-agent
**Time:** 1 hour
**Impact:** +1,648 tests

### Step 4: Convolution Kernel (Next Session)
**Task:** Fix vxConvolve implementation
**Agent:** convolution-agent
**Time:** 1 hour
**Impact:** +1,009 tests

---

## Current Working Foundation

✅ **Already Working:**
- GraphBase: 14/14 (100%)
- Box3x3: 23/23 (100%)
- Gaussian3x3: 9/9 (100%)
- Core reference counting
- Kernel loading/unloading

**These prove the infrastructure is solid!**

---

## Risk Assessment

**High Risk:**
- Image format changes (may break existing tests)
- Convolution accuracy (complex math)

**Low Risk:**
- Virtual image fixes (isolated)
- Border handling tweaks

**Mitigation:**
- Run tests after each fix
- Commit frequently
- Use feature flags if needed
