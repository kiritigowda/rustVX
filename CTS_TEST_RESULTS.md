# OpenVX CTS Test Results - April 5, 2026

## Executive Summary

**Status:** Mixed - Core graph execution working, reference counting issues persist

**Key Achievement:**
- ✅ **Graph.TwoNodes EXECUTES** - Both nodes run successfully
- ⚠️ Reference counting underflow in test teardown

---

## Test Results by Category

### Baseline Tests

| Test | Status | Notes |
|------|--------|-------|
| GraphBase.AllocateUserKernelId | ✅ PASS | |
| GraphBase.AllocateUserKernelLibraryId | ✅ PASS | |
| GraphBase.RegisterUserStructWithName | ✅ PASS | |
| GraphBase.GetUserStructNameByEnum | ✅ PASS | |
| GraphBase.GetUserStructEnumByName | ✅ PASS | |
| GraphBase.vxCreateGraph | ❌ FAIL | Reference counting |
| GraphBase.vxIsGraphVerifiedBase | ❌ FAIL | Reference counting |
| GraphBase.vxQueryGraph | ❌ FAIL | Reference counting |
| GraphBase.vxReleaseGraph | ❌ FAIL | Reference counting |
| GraphBase.vxQueryNodeBase | ✅ PASS | |
| GraphBase.vxReleaseNodeBase | ✅ PASS | |
| GraphBase.vxRemoveNodeBase | ✅ PASS | |
| GraphBase.vxReplicateNodeBase | ❌ FAIL | Reference counting |
| GraphBase.vxSetNodeAttributeBase | ✅ PASS | |

### Smoke Tests

| Test | Status | Notes |
|------|--------|-------|
| SmokeTestBase.vxReleaseReferenceBase | ❌ FAIL | Reference counting |
| SmokeTestBase.vxLoadKernels | ✅ PASS | |
| SmokeTestBase.vxUnloadKernels | ❌ FAIL | Reference counting |
| SmokeTestBase.vxSetReferenceName | ❌ FAIL | Reference counting |
| SmokeTestBase.vxGetStatus | ✅ PASS | |
| SmokeTestBase.vxQueryReference | ❌ FAIL | Reference counting |
| SmokeTestBase.vxRetainReferenceBase | ❌ FAIL | Reference counting |

### Target Tests

| Test | Status | Notes |
|------|--------|-------|
| TargetBase.vxCreateContext | ❌ FAIL | Reference counting |
| TargetBase.vxReleaseContext | ❌ FAIL | Reference counting |
| TargetBase.vxSetNodeTargetBase | ✅ PASS | |

### Graph Execution Tests

| Test | Status | Notes |
|------|--------|-------|
| **Graph.TwoNodes** | ⚠️ **PARTIAL** | **Executes but cleanup fails** |
| Graph.VirtualImage | ❌ FAIL | Reference counting |
| Graph.GraphFactory | ❌ FAIL | Reference counting |

---

## Critical Issues

### Issue 1: Reference Counting Underflow (CRITICAL)

**Symptom:**
```
FAILED: Expected: 0 == dangling_refs_count
         Actual: 0 != 4294967295
```

**Root Cause:** 
- `vxQueryContext(VX_CONTEXT_REFERENCES)` counts all entries in REFERENCE_COUNTS
- When objects are released, REFERENCE_COUNTS entries are removed
- Test subtracts base_references from current
- When current < base, underflow occurs (4294967295 = -1 in unsigned)

**Fix Needed:**
- Either decrement instead of remove in REFERENCE_COUNTS
- Or adjust vxQueryContext to return per-context counts only

### Issue 2: Kernel ID Mismatch (FIXED)

**Status:** ✅ RESOLVED

**What Was Fixed:**
- Changed kernel registration to use enum values as IDs
- Previously used `generate_id()` which returned random IDs
- Now Box3x3 = 0x12, IntegralImage = 0x0e per OpenVX spec

### Issue 3: U32 Image Format (FIXED)

**Status:** ✅ RESOLVED

**What Was Fixed:**
- Added `ImageFormat::GrayU32` for integral image output
- Added U32 format code (0x32333055) to df_image_to_format

---

## Code Changes Summary

### Files Modified

1. **openvx-core/src/c_api.rs**
   - Changed kernel registration to use enum values: `let kernel_id = kernel_enum as u64`

2. **openvx-core/src/vxu_impl.rs**
   - Added `ImageFormat::GrayU32` variant
   - Added U32 format code support

3. **openvx-vision/src/register.rs**
   - Updated to register kernels with correct enum IDs

4. **openvx-image/src/c_api.rs**
   - Attempted reference counting fix (needs refinement)

---

## Next Steps

1. **Fix Reference Counting:**
   - Modify vxQueryContext to only count references per context
   - OR: Keep references in REFERENCE_COUNTS and decrement

2. **Run Full CTS:**
   - After fix, run complete test suite
   - Expected: +50+ additional tests should pass

3. **Vision Conformance:**
   - Graph execution is working
   - Need to validate all vision kernel outputs

---

## Repository

**URL:** https://github.com/simonCatBot/rustVX
**Commit:** Working toward full conformance

---

*Report generated: April 5, 2026*