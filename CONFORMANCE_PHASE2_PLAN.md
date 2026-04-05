# rustVX OpenVX Conformance Phase 2 Plan

## Goal: Achieve Vision Conformance Feature Set

### Current Status
- ✅ Core framework (context, graph)
- ✅ Basic vision kernels (49 integration tests passing)
- ✅ Scalars, Arrays (newly added)
- ❌ CTS link still fails - missing many symbols

### CTS Link Errors (From RUSTVX_CONFORMANCE.md)

#### Critical Missing Functions for Link:
1. **vxAllocateUserKernelId**
2. **vxAllocateUserKernelLibraryId**
3. **vxRegisterUserStructWithName**
4. **vxGetUserStructNameByEnum**
5. **vxGetUserStructEnumByName**
6. **vxIsGraphVerified** ✅ (should be done)
7. **vxQueryNode** - Critical
8. **vxReleaseNode** - Critical
9. **vxRemoveNode**
10. **vxSetNodeAttribute**
11. **vxReplicateNode** ✅ (should be done)
12. **vxLoadKernels**
13. **vxGetKernelByName**
14. **vxUnloadKernels**
15. **vxGetKernelParameterByIndex**
16. **vxRetainReference**
17. **vxGetContext**
18. **vxGetStatus**
19. **vxCreateImage** - Critical
20. **vxQueryImage** - Critical
21. **vxMapImagePatch** - Critical
22. **vxUnmapImagePatch** - Critical
23. **vxReleaseImage** - Critical
24. **vxCreateThreshold** - Critical
25. **vxQueryThreshold** - Critical
26. **vxReleaseThreshold** - Critical

### Phase 2 Implementation Plan

#### Round 1: Core Reference & Context (Link Critical)
Dependencies: None
- vxRetainReference
- vxGetContext
- vxGetStatus
- vxGetStatus

#### Round 2: Node Management (Link Critical)
Dependencies: Round 1
- vxQueryNode (full implementation)
- vxReleaseNode
- vxRemoveNode
- vxSetNodeAttribute

#### Round 3: Kernel Management (Link Critical)
Dependencies: Round 1
- vxLoadKernels
- vxGetKernelByName
- vxUnloadKernels
- vxGetKernelParameterByIndex
- vxGetKernelByEnum

#### Round 4: User Kernel Support (Link Critical)
Dependencies: Round 3
- vxAllocateUserKernelId
- vxAllocateUserKernelLibraryId
- vxRegisterUserStructWithName
- vxGetUserStructNameByEnum
- vxGetUserStructEnumByName

#### Round 5: Complete Image API (Runtime Critical)
Dependencies: Round 1
- vxCreateImage (all formats)
- vxQueryImage (all attributes)
- vxMapImagePatch / vxUnmapImagePatch
- vxCreateVirtualImage
- vxCreateImageFromHandle
- vxSetImageAttribute
- vxReleaseImage

#### Round 6: Complete Threshold API (Runtime Critical)
Dependencies: Round 1
- vxCreateThreshold
- vxQueryThreshold
- vxSetThresholdAttribute
- vxReleaseThreshold

#### Round 7: Convolution, Matrix, LUT, Distribution, Pyramid
Dependencies: Round 5
- vxCreateConvolution / vxQueryConvolution / vxReleaseConvolution
- vxCreateMatrix / vxQueryMatrix / vxReleaseMatrix
- vxCreateLUT / vxQueryLUT / vxReleaseLUT
- vxCreateDistribution / vxQueryDistribution / vxReleaseDistribution
- vxCreatePyramid / vxQueryPyramid / vxReleasePyramid

#### Round 8: CTS Build & Run
Dependencies: Round 1-7
- Fix any remaining link errors
- Build CTS
- Run CTS with filters for implemented functions
- Debug and fix failing tests

### Team Assignment

Using 4 agents in parallel:
- **Agent 1**: Round 1 + Round 2 (Reference/Context + Node Management)
- **Agent 2**: Round 3 + Round 4 (Kernel Management + User Kernel)
- **Agent 3**: Round 5 (Complete Image API) - largest task
- **Agent 4**: Round 6 + Round 7 (Threshold + Other Data Objects)

### Success Criteria

- ✅ CTS builds successfully without link errors
- ✅ Can run CTS with `--filter` for vision tests
- ✅ At least 50% of Vision Feature Set tests pass
- ✅ No regressions in existing 49 integration tests

### Risk Analysis

**Blockers:**
1. Image patch mapping (vxMapImagePatch) is complex - needs careful memory management
2. User kernel callbacks require C-to-Rust FFI bridge
3. CTS test data may reveal algorithm accuracy issues

**Mitigation:**
- Follow existing patterns from c_api.rs
- Test each component incrementally
- Use Khronos reference for algorithm correctness
