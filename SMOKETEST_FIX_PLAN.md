# Plan: Fix Remaining 6 SmokeTest Failures for OpenVX CTS

## Task
Fix all 6 failing SmokeTest tests to achieve 100% baseline conformance.

## Current Status
- GraphBase: 14/14 ✅ PASSING
- SmokeTestBase: 7/7 ✅ PASSING  
- SmokeTest: 8/14 ⚠️ (6 FAILURES)

## Failing Tests Analysis

### 1. SmokeTest.vxRetainReference
**Error:** Expected num_refs2 == num_refs1+1, Actual: 0 != 1
**Root Cause:** vxRetainReference returns 0, vxQueryReference returns 0 for VX_REFERENCE_COUNT
**Files:** openvx-core/src/c_api.rs (vxRetainReference, vxQueryReference)
**Registry Issue:** REFERENCE_COUNTS not properly initialized when objects are created

### 2. SmokeTest.vxReleaseReference
**Error:** Error type of "array = vxCreateArray(context, VX_TYPE_KEYPOINT, 32)"
**Root Cause:** Array created but reference counting not working
**Files:** openvx-buffer/src/c_api.rs (vxCreateArray, vxReleaseArray)
**Note:** VX_TYPE_KEYPOINT is now defined, but release counting fails

### 3. SmokeTest.vxRegisterUserStruct
**Error:** Assertion mytype >= VX_TYPE_USER_STRUCT_START failed
**Root Cause:** vxRegisterUserStructWithName returns value < 0x100
**Files:** openvx-core/src/c_api.rs
**Status:** Should be fixed by userstruct-agent, verify it's working

### 4. SmokeTest.vxSetParameterByIndex
**Error:** Parameter setting not implemented or failing
**Files:** openvx-core/src/c_api.rs (vxSetParameterByIndex)
**Needs:** Implementation to set node parameters by index

### 5. SmokeTest.vxSetParameterByReference  
**Error:** Parameter setting by reference not working
**Files:** openvx-core/src/c_api.rs (vxSetParameterByReference)
**Needs:** Implementation

### 6. SmokeTest.vxGetParameterByIndex
**Error:** Parameter retrieval failing
**Files:** openvx-core/src/c_api.rs (vxGetParameterByIndex)
**Needs:** Implementation to get node parameter by index

## Dependency Graph

```
vxCreate* functions
    ↓
REFERENCE_COUNTS registry (shared across all types)
    ↓
vxRetainReference → vxQueryReference(VX_REFERENCE_COUNT)
    ↓
vxReleaseReference
    ↓
vxSetParameterByIndex → requires working nodes/parameters
    ↓
vxSetParameterByReference
    ↓
vxGetParameterByIndex
```

## Execution Plan

### Round 1: Fix Reference Counting Foundation
**Agent:** refcount-fix-agent
**Dependencies:** None
**Task:** 
- Verify REFERENCE_COUNTS registry is initialized with count=1 for ALL object types
- Fix vxRetainReference to increment and return new count
- Fix vxQueryReference to return count from registry
- Fix vxReleaseReference to decrement and return new count
- Ensure vxCreateArray, vxCreateImage, vxCreateScalar, vxCreateGraph, vxCreateContext all register in REFERENCE_COUNTS

**Verification:** 
- Build: cargo build --release -p openvx-core
- Test: SmokeTest.vxRetainReference passes
- Test: SmokeTest.vxReleaseReference passes

**Files:** 
- openvx-core/src/c_api.rs (vxRetainReference, vxQueryReference, vxReleaseReference, vxCreateContext, vxCreateGraph)
- openvx-buffer/src/c_api.rs (vxCreateArray, vxReleaseArray)

---

### Round 2: Fix Parameter Functions
**Agent:** parameter-agent  
**Dependencies:** Round 1 complete
**Task:**
- Implement vxSetParameterByIndex
- Implement vxSetParameterByReference
- Implement vxGetParameterByIndex
- Ensure parameters are properly linked to nodes

**Verification:**
- Build: cargo build --release
- Test: SmokeTest.vxSetParameterByIndex passes
- Test: SmokeTest.vxSetParameterByReference passes
- Test: SmokeTest.vxGetParameterByIndex passes

**Files:**
- openvx-core/src/c_api.rs (vxSetParameterByIndex, vxSetParameterByReference, vxGetParameterByIndex)
- openvx-core/src/unified_c_api.rs (if parameter structures exist there)

---

### Round 3: Verify User Struct Fix
**Agent:** verify-userstruct-agent
**Dependencies:** None (independent)
**Task:**
- Verify vxRegisterUserStructWithName returns enum >= 0x100
- Check USER_STRUCTS registry is properly populated
- Verify vxGetUserStructNameByEnum and vxGetUserStructEnumByName work

**Verification:**
- Build: cargo build --release
- Test: SmokeTest.vxRegisterUserStruct passes

**Files:**
- openvx-core/src/c_api.rs (vxRegisterUserStructWithName, vxGetUserStructNameByEnum, vxGetUserStructEnumByName)

---

### Round 4: Integration & Full Smoke Test
**Agent:** integration-agent
**Dependencies:** Rounds 1, 2, 3 complete
**Task:**
- Run full SmokeTest suite
- Verify all 14 tests pass
- Fix any remaining issues
- Rebuild CTS and verify

**Verification:**
- All SmokeTest* tests pass (14/14)
- CTS builds successfully

**Files:** All affected files

## Risk Analysis

**Potential Blockers:**
1. Reference counting may require changes across multiple crates (core, buffer, image)
2. Parameter functions may need node/parameter structures that don't exist
3. Circular dependencies between objects

**Mitigation:**
1. Use a centralized REFERENCE_COUNTS registry in c_api.rs
2. Create minimal parameter structures if needed
3. Initialize all ref_counts to 1 at creation

## Rollback Plan

**If fails at any round:**
- Each round is isolated to specific functions
- Can revert individual commits
- Integration tests verify each step

## Success Criteria

- ✅ SmokeTest: 14/14 tests passing
- ✅ SmokeTestBase: 7/7 tests passing (already done)
- ✅ GraphBase: 14/14 tests passing (already done)
- ✅ CTS builds without errors
- ✅ Total: 35/35 baseline tests passing

## Team Assignment

| Agent | Task | Files | Dependencies |
|-------|------|-------|--------------|
| refcount-fix-agent | Fix reference counting | c_api.rs, buffer/c_api.rs | None |
| parameter-agent | Implement parameter functions | c_api.rs | Round 1 |
| verify-userstruct-agent | Verify user struct fix | c_api.rs | None |
| integration-agent | Integration test | All | Rounds 1-3 |
