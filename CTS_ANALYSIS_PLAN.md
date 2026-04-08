# OpenVX 1.3.1 CTS Conformance Analysis Plan

**Date:** April 5, 2026
**Goal:** Achieve Full Vision Conformance
**Strategy:** Iterative team-based fixing

---

## Current Status Summary

| Category | Tests | Passing | Status |
|----------|-------|---------|--------|
| **Baseline** | 25 | 25 | ✅ **100%** |
| **KernelName** | 42 | 42 | ✅ **100%** |
| **Vision Tests** | ~15,000+ | ~65 | 🔄 **~0.4%** |
| **Total Before Crash** | ~120 | ~67 | **~56%** |

**Crash Point:** Graph.TwoNodes (during graph execution with vision kernels)

---

## Failure Categories (Grouped by Root Cause)

### Group 1: Graph Execution Failures 🔴 CRITICAL
**Symptom:** Segfault during graph execution
**Tests Affected:** Graph.TwoNodes, Graph.GraphFactory, Graph.VirtualImage, etc.

**Root Causes:**
1. Kernel function pointers NULL
2. Node parameter validation failing
3. Image data access during graph execution
4. Memory corruption in graph processing

**Fix Strategy:**
- Ensure all vision kernel functions are properly linked
- Fix node parameter handling
- Add null checks before kernel execution
- Validate image data before processing

---

### Group 2: Image Patch Address Functions 🔴 CRITICAL
**Symptom:** vxFormatImagePatchAddress1d/2d fails for planar formats
**Tests Affected:** Image.FormatImagePatchAddress1d, Image.FormatImagePatchAddress2d

**Root Causes:**
1. Incorrect plane offset calculations for YUV/NV12
2. Not handling planar formats in address calculation
3. Missing stride calculations for UV planes

**Fix Strategy:**
- Implement proper planar format address calculation
- Add plane-aware offset computation
- Handle YUV4, IYUV, NV12, NV21 formats correctly

---

### Group 3: Image Swap Handle Issues 🔴 HIGH
**Symptom:** vxSwapImageHandle crashes or fails
**Tests Affected:** Image.SwapImageHandle

**Root Causes:**
1. Planar image plane swapping not implemented
2. Memory ownership confusion
3. External pointer handling incorrect

**Fix Strategy:**
- Implement proper multi-plane swap
- Handle external vs owned memory correctly
- Add validation for planar format swaps

---

### Group 4: Virtual Image Clone Issues 🔴 HIGH
**Symptom:** vxCloneImage fails for virtual images
**Tests Affected:** Image.CloneImage, Image.VirtualImage

**Root Causes:**
1. Virtual image backing memory not allocated
2. Clone doesn't handle virtual image state
3. Memory layout mismatch

**Fix Strategy:**
- Allocate backing memory during vxVerifyGraph
- Handle virtual image flag in clone
- Proper memory copying for all formats

---

### Group 5: Remap/Map Function Failures 🟡 MEDIUM
**Symptom:** vxRemapPoint fails or returns wrong values
**Tests Affected:** Remap.Point, Remap.Region

**Root Causes:**
1. Remap coordinate transformation incorrect
2. Interpolation not implemented
3. Boundary handling wrong

**Fix Strategy:**
- Implement bilinear interpolation for remap
- Fix coordinate transformation math
- Handle edge cases properly

---

### Group 6: Distribution/Array Issues 🟡 MEDIUM
**Symptom:** Array/Distribution operations fail
**Tests Affected:** Array.*, Distribution.*

**Root Causes:**
1. Range validation incorrect
2. Memory mapping issues
3. Element access out of bounds

**Fix Strategy:**
- Fix range validation
- Ensure proper memory alignment
- Add bounds checking

---

### Group 7: Pyramid/Scale Issues 🟡 MEDIUM
**Symptom:** Pyramid construction fails
**Tests Affected:** Pyramid.*, ScaleImage.*

**Root Causes:**
1. Pyramid level computation wrong
2. Scale factor calculation incorrect
3. Memory allocation for levels

**Fix Strategy:**
- Fix pyramid level calculation
- Implement correct scale factors
- Proper memory allocation per level

---

## Iterative Fix Plan

### Round 1: Graph Execution Stability
**Goal:** Get Graph.TwoNodes and similar tests passing
**Agents:** 3 parallel agents
- Agent 1: Fix kernel function pointer linking
- Agent 2: Fix node parameter validation
- Agent 3: Fix image data access in graph execution

### Round 2: Image Patch Functions
**Goal:** Fix Image.FormatImagePatchAddress* tests
**Agents:** 2 parallel agents
- Agent 1: Implement planar format address calculation
- Agent 2: Fix stride calculations for all formats

### Round 3: Image Handle Operations
**Goal:** Fix Swap/Clone operations
**Agents:** 2 parallel agents
- Agent 1: Fix vxSwapImageHandle for planar formats
- Agent 2: Fix vxCloneImage for virtual images

### Round 4: Remap and Geometric
**Goal:** Fix Remap and pyramid tests
**Agents:** 2 parallel agents
- Agent 1: Implement Remap bilinear interpolation
- Agent 2: Fix pyramid construction

### Round 5: Polish and Edge Cases
**Goal:** Fix remaining Array/Distribution issues
**Agents:** 2 parallel agents
- Agent 1: Fix Array operations
- Agent 2: Fix Distribution operations

---

## Success Metrics

| Round | Target Pass Rate | Expected Tests Passing |
|-------|-----------------|----------------------|
| Start | ~0.4% | ~67 |
| Round 1 | 5% | ~750 |
| Round 2 | 15% | ~2,250 |
| Round 3 | 30% | ~4,500 |
| Round 4 | 50% | ~7,500 |
| Round 5 | 75% | ~11,250 |
| Final | 90%+ | ~13,500+ |

---

## Execution Command

```bash
cd /home/simon/.openclaw/workspace/rustvx/OpenVX-cts/build
LD_LIBRARY_PATH=/home/simon/.openclaw/workspace/rustvx/target/release ./bin/vx_test_conformance
```

---

## Notes

- **Baseline is SOLID:** 100% passing (25/25)
- **KernelName is SOLID:** 100% passing (42/42)
- **Vision kernels IMPLEMENTED:** 40+ with real algorithms
- **Main issue:** Graph execution and image handling

**We're closer than it looks!** The crashes are fixable structural issues, not missing algorithms.
