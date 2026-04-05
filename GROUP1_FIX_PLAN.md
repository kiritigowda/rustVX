# Group 1 Fix Plan: Core Framework (7 Remaining SmokeTest Failures)

## Task
Fix all 7 remaining SmokeTest failures to achieve 100% Group 1 (Core Framework) conformance.

## Current Status
- ✅ GraphBase: 14/14 PASSING
- ✅ SmokeTestBase: 7/7 PASSING  
- ⚠️ SmokeTest: 7/14 (7 FAILURES remaining)

## The 7 Failing Tests

1. **SmokeTest.vxRegisterUserStruct** - User struct enum allocation
2. **SmokeTest.vxHint** - vxHint implementation
3. **SmokeTest.vxReleaseReference** - Reference counting for objects
4. **SmokeTest.vxSetParameterByIndex** - Parameter setting on nodes
5. **SmokeTest.vxSetParameterByReference** - Parameter setting by reference
6. **SmokeTest.vxGetParameterByIndex** - Parameter retrieval
7. **SmokeTestBase.vxReleaseReferenceBase** - Base reference release

## Analysis of Failures

### Failure 1: vxRegisterUserStruct
**Error:** `Assertion mytype >= VX_TYPE_USER_STRUCT_START failed`
**Root Cause:** vxRegisterUserStructWithName returns enum value < 0x100
**Fix:** Ensure atomic counter starts at 0x100

### Failure 2: vxHint  
**Error:** vxHint not implemented or returns error
**Root Cause:** Function missing or stub
**Fix:** Implement vxHint to accept and store hints

### Failure 3: vxReleaseReference
**Error:** `Expected ref_count1 - ref_count0 == 1, Actual: 0 != 1`
**Root Cause:** Objects (pyramid, remap, scalar, etc.) not counted in vxQueryContext VX_CONTEXT_REFERENCES
**Fix:** Ensure ALL object creation functions register in REFERENCE_COUNTS and are counted

### Failure 4-6: Parameter Functions
**Error:** vxQueryNode VX_NODE_PARAMETERS not implemented, vxSet/GetParameterByIndex failing
**Root Cause:** Parameter management incomplete
**Fix:** Implement full parameter lifecycle

### Failure 7: vxReleaseReferenceBase
**Error:** Base test failing
**Root Cause:** Related to reference counting
**Fix:** Same as Failure 3

## Execution Plan

### Step 1: [ANALYSIS] Identify all object creation functions
**Dependencies:** None
**Approach:**
- Search for all vxCreate* functions in codebase
- Verify each registers in REFERENCE_COUNTS
- Verify each registers in REFERENCE_TYPES for type detection
**Verification:** List of 20+ create functions with registration status
**Files:** All c_api.rs and unified_c_api.rs

---

### Step 2: [FIX] Implement vxRegisterUserStruct
**Dependencies:** None
**Approach:**
- Find vxRegisterUserStructWithName in c_api.rs or unified_c_api.rs
- Add atomic counter: `static NEXT_USER_STRUCT: AtomicI32 = AtomicI32::new(0x100)`
- Change return to use `fetch_add(1, Ordering::SeqCst)`
- Ensure VX_TYPE_USER_STRUCT_START = 0x100 is defined
**Verification:** SmokeTest.vxRegisterUserStruct passes
**Files:** openvx-core/src/c_api.rs

---

### Step 3: [IMPLEMENTATION] Add vxHint
**Dependencies:** None
**Approach:**
- Add vxHint function definition
- Accept reference, hint enum, and data pointer
- Validate inputs (non-null reference)
- Store hint in registry or just return SUCCESS for now
**Verification:** SmokeTest.vxHint passes
**Files:** openvx-core/src/c_api.rs

---

### Step 4: [CRITICAL] Fix Reference Counting for ALL Objects
**Dependencies:** Step 1 (analysis complete)
**Approach:**
- For each vxCreate* function:
  1. Add REFERENCE_COUNTS registration: `counts.insert(ptr as usize, AtomicUsize::new(1))`
  2. Add REFERENCE_TYPES registration with correct type
- Update vxQueryContext VX_CONTEXT_REFERENCES to count all types
**Verification:** SmokeTest.vxReleaseReference passes
**Files:** 
- openvx-core/src/c_api.rs (context, graph, node, kernel)
- openvx-core/src/c_api_data.rs (scalar, matrix, lut, threshold, pyramid, convolution)
- openvx-core/src/unified_c_api.rs (distribution, remap)
- openvx-image/src/c_api.rs (image)
- openvx-buffer/src/c_api.rs (array)

---

### Step 5: [IMPLEMENTATION] Complete Parameter Functions
**Dependencies:** None (parallel to Step 4)
**Approach:**
- vxQueryNode: Add VX_NODE_PARAMETERS case (already done? verify)
- vxSetParameterByIndex: Set parameter in node's parameter vector
- vxSetParameterByReference: Find parameter by reference, set value
- vxGetParameterByIndex: Return parameter object (create if needed)
- vxQueryParameter: Add VX_PARAMETER_REF case
**Verification:** 
- SmokeTest.vxSetParameterByIndex passes
- SmokeTest.vxSetParameterByReference passes
- SmokeTest.vxGetParameterByIndex passes
**Files:** openvx-core/src/c_api.rs, openvx-core/src/unified_c_api.rs

---

### Step 6: [INTEGRATION] Build and Test
**Dependencies:** Steps 2-5 complete
**Approach:**
- Build: `cargo build --release`
- Rebuild CTS: `make -j$(nproc)`
- Run all SmokeTest: `--filter="Smoke*"`
**Verification:** All 14 SmokeTest tests pass
**Files:** N/A (build only)

---

### Step 7: [VALIDATION] Full Group 1 Verification
**Dependencies:** Step 6
**Approach:**
- Run GraphBase.*: All 14 should pass
- Run SmokeTestBase.*: All 7 should pass
- Run SmokeTest.*: All 14 should pass (no failures)
**Verification:** 35/35 Group 1 tests passing
**Files:** N/A (test only)

---

## Risk Analysis

**Blockers:**
1. REFERENCE_COUNTS type mismatch (HashMap value type issues)
2. Circular dependencies between crates for type registration
3. Parameter structure differences between c_api and unified_c_api

**Mitigation:**
1. Use AtomicUsize consistently, fix operations with load/store
2. Use centralized REFERENCE_TYPES registry in unified_c_api
3. Focus on c_api implementation, unified can delegate

**Rollback:**
- Each step isolated to specific functions
- Can revert individual commits
- Tests verify each step

## Parallel Execution Strategy

Steps 2, 3, and 5 can run in parallel (independent fixes).
Step 4 is large and can be parallelized by object type:
- Agent A: Context, Graph, Node, Kernel
- Agent B: Image (in openvx-image crate)
- Agent C: Array (in openvx-buffer crate)  
- Agent D: Scalar, Matrix, LUT, Threshold, Pyramid, Convolution, Distribution, Remap

Step 6 (build/test) depends on all previous steps.
Step 7 (validation) final check.
