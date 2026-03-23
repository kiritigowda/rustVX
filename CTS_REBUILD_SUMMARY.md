# rustVX CTS Vision Rebuild - Summary

## Date: 2026-03-22

## Overview
Rebuilt the rustVX vision implementation to fix CTS (Conformance Test Suite) failures. Focused on the key issues identified in the CTS test reports.

## Key Issues Fixed

### 1. Reference Counting Initialization (CRITICAL)
**Problem**: References were starting with count 0 instead of 1
**Impact**: `vxRetainReferenceBase` test failing - expected count 1, got 0

**Fix Applied**:
- Modified `vxCreateContext()` in `c_api.rs` - initializes reference count to 1 at creation
- Modified `vxCreateGraph()` in `c_api.rs` - initializes reference count to 1, also registers graph in unified registry
- Modified `vxRetainReference()` in `c_api.rs` - now returns VX_ERROR_INVALID_REFERENCE if reference doesn't exist in registry

### 2. Graph Registration in Unified Registry
**Problem**: Graphs created via c_api weren't visible to vxQueryReference in unified_c_api
**Impact**: Graph query operations failing

**Fix Applied**:
- Modified `vxCreateGraph()` to register graphs in both `GRAPHS` (c_api) and `GRAPHS_DATA` (unified_c_api) registries
- Added proper unified graph data structure creation with all required fields

### 3. Error Code Corrections
**Problem**: Several functions returning incorrect error codes per CTS expectations
**Impact**: Tests expecting specific error codes failing

**Fix Applied**:
- `vxGetStatus()` now returns `VX_ERROR_NO_RESOURCES` (-12) for null reference (per CTS expectation)
- Error constants aligned in `c_api.rs`

### 4. vxIsGraphVerified Behavior
**Problem**: Already fixed - returns vx_bool (0 or 1) not vx_status
**Location**: `unified_c_api.rs` line 390-411

**Status**: Already correct - returns 0 (vx_false_e) for null graph, not error code

### 5. vxQueryGraph Cross-Registry Lookup
**Problem**: vxQueryGraph only checking unified registry, not c_api registry
**Impact**: Graphs created via c_api not queryable

**Fix Applied**:
- Added check for graph in c_api registry within VX_GRAPH_ATTRIBUTE_STATE case

## Files Modified

1. `/home/simon/.openclaw/workspace/rustVX/openvx-core/src/c_api.rs`
   - `vxCreateContext()` - reference count initialization
   - `vxCreateGraph()` - complete rewrite with unified registry support
   - `vxRetainReference()` - proper validation
   - `vxGetStatus()` - correct error code

2. `/home/simon/.openclaw/workspace/rustVX/openvx-core/src/unified_c_api.rs`
   - `vxQueryGraph()` - added c_api registry check

## Testing Recommendation

After building with `cargo build`, run CTS tests to verify:

```bash
cd openvx-rust/OpenVX-cts/build
./bin/test_vx -f base -e "TestCaseName" --verbose
```

Specific tests to verify:
- `SmokeTestBase.vxRetainReferenceBase` - should now pass
- `SmokeTestBase.vxReleaseReferenceBase` - should now pass
- `GraphBase.vxIsGraphVerifiedBase` - should return 0 (false) for null
- `GraphBase.vxQueryGraph` - should handle c_api graphs
- `SmokeTestBase.vxGetStatus` - should return -12 for null

## Remaining Work

The following may still need attention:

1. **vxLoadKernels** - May need additional implementation for test module loading
2. **vxUnloadKernels** - Cleanup logic may need refinement
3. **Reference name handling** - vxQueryReference for VX_REFERENCE_NAME already fixed
4. **Node creation** - May need similar cross-registry registration

## Build Instructions

```bash
cd /home/simon/.openclaw/workspace/rustVX
cargo build --release
```

The shared library will be at `target/release/libopenvx_core.so`
