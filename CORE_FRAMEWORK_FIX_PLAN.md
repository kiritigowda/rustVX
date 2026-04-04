# Core Framework Fix Plan - Iteration 1

## Target: 35/35 Core Tests (100% PASSING)

## Current Status: 32/35 (91%)

### ❌ Remaining Failures (3 tests):

1. **SmokeTest.vxGetParameterByIndex** - Double-free crash
2. **SmokeTest.vxSetParameterByReference** - Parameter lifecycle issue  
3. **SmokeTestBase.vxReleaseReferenceBase** - Reference counting

## Root Cause Analysis

### Double-Free Problem:
Parameter storage has **dual ownership**:
1. Node stores: `parameters: Mutex<Vec<Option<u64>>>` in c_api.rs
2. Global registry: `PARAMETERS: Mutex<HashMap<u64, Arc<VxCParameter>>>` in unified_c_api.rs

When vxReleaseParameter is called:
- Removes from PARAMETERS HashMap → Arc drop
- Also tries to free from other location
- **DOUBLE FREE**

## Fix Strategy

### Approach: Simplify to Single Ownership

**Current (BROKEN):**
```rust
// Two places store the same data
node.parameters: Vec<Option<u64>>  // c_api.rs
PARAMETERS: HashMap<u64, Arc<VxCParameter>>  // unified_c_api.rs
```

**Fixed (SIMPLE):**
```rust
// Only one place stores data
node.parameters: Vec<Option<u64>>  // Just store reference IDs, no Arc
// Remove PARAMETERS HashMap entirely
```

## Implementation Steps

### Step 1: Remove PARAMETERS Registry
- Comment out `static PARAMETERS` in unified_c_api.rs
- Remove from vxGetParameterByIndex
- Remove from vxReleaseParameter
- Keep only node.parameters vector

### Step 2: Fix vxGetParameterByIndex
- Just return (node_id << 32) | index as handle
- No Arc storage needed
- No global registry lookup

### Step 3: Fix vxReleaseParameter
- Only decrement REFERENCE_COUNTS
- Don't try to remove from PARAMETERS
- Simple reference counting only

### Step 4: Test
- Run SmokeTest* tests
- Should pass all 3 remaining failures

## Expected Result
35/35 Core tests passing (100%)

## Files to Modify
- openvx-core/src/unified_c_api.rs
- openvx-core/src/c_api.rs (minor changes)

## Timeline
- Iteration 1: 30 minutes
- If fails, debug and Iteration 2
