# Full OpenVX Conformance Plan for rustVX

## Current Status Assessment

**Exported Functions**: ~50 functions (from nm analysis)
**Target Functions**: ~300 functions needed for conformance
**CTS Build**: Fails with link errors (undefined references)

### Critical Missing Functions for CTS Link

#### Priority 1: Graph Operations (Blocks Base Tests)
- `vxVerifyGraph` - Compiles graph for execution
- `vxProcessGraph` - Executes graph synchronously
- `vxQueryGraph` - Queries graph attributes
- `vxWaitGraph` - Waits for async graph completion
- `vxScheduleGraph` - Schedules graph for async execution
- `vxIsGraphVerified` - Checks if graph is verified
- `vxReplicateNode` - Replicates nodes for batch processing

#### Priority 2: Reference Management (Blocks Smoke Tests)
- `vxQueryReference` - Queries reference attributes including count
- `vxReleaseReference` - Generic reference release
- `vxRetainReference` - Already exported but may need fixes

#### Priority 3: Context Operations
- `vxQueryContext` - Queries context attributes
- `vxSetContextAttribute` - Sets context attributes

#### Priority 4: User Kernel Support
- `vxAllocateUserKernelId` - Allocates user kernel ID
- `vxAllocateUserKernelLibraryId` - Allocates library ID
- `vxRegisterUserStructWithName` - Registers user struct
- `vxGetUserStructNameByEnum` - Gets struct name from enum
- `vxGetUserStructEnumByName` - Gets struct enum from name

#### Priority 5: Array Operations
- `vxQueryArray` - Queries array attributes
- `vxMapArrayRange` - Maps array for access
- `vxUnmapArrayRange` - Unmaps array

#### Priority 6: Logging/Debugging
- `vxRegisterLogCallback` - Registers log callback
- `vxAddLogEntry` - Adds log entry
- `vxDirective` - Sets directives

#### Priority 7: Image Utilities
- `vxFormatImagePatchAddress2d` - Formats image patch address

#### Priority 8: Scalar Operations
- `vxCopyScalar` - Copies scalar data

## Dependency Graph

```
Round 1: Reference Management (Foundation)
  ├─ vxQueryReference
  ├─ vxReleaseReference
  └─ vxRetainReference (verify/fix)

Round 2: Context Operations (Foundation)
  ├─ vxQueryContext
  └─ vxSetContextAttribute

Round 3: Graph Operations (Critical for CTS)
  ├─ vxVerifyGraph
  ├─ vxProcessGraph
  ├─ vxQueryGraph
  ├─ vxWaitGraph
  ├─ vxScheduleGraph
  ├─ vxIsGraphVerified
  └─ vxReplicateNode

Round 4: User Kernel Support
  ├─ vxAllocateUserKernelId
  ├─ vxAllocateUserKernelLibraryId
  ├─ vxRegisterUserStructWithName
  ├─ vxGetUserStructNameByEnum
  └─ vxGetUserStructEnumByName

Round 5: Array Operations
  ├─ vxQueryArray
  ├─ vxMapArrayRange
  └─ vxUnmapArrayRange

Round 6: Logging & Debugging
  ├─ vxRegisterLogCallback
  ├─ vxAddLogEntry
  └─ vxDirective

Round 7: Image & Scalar Utilities
  ├─ vxFormatImagePatchAddress2d
  └─ vxCopyScalar
```

## Team Code Execution Plan

### Phase 1: Foundation (Round 1-2)
**Parallel Agents:**
- Agent 1: Reference Management (Round 1)
- Agent 2: Context Operations (Round 2)

### Phase 2: Graph Core (Round 3)
**Sequential Dependency:** Requires Phase 1
- Agent 3: Graph Operations (Round 3)

### Phase 3: User Kernel & Array (Round 4-5)
**Parallel Agents:**
- Agent 4: User Kernel Support (Round 4)
- Agent 5: Array Operations (Round 5)

### Phase 4: Logging & Utilities (Round 6-7)
**Parallel Agents:**
- Agent 6: Logging & Debugging (Round 6)
- Agent 7: Image & Scalar Utilities (Round 7)

### Phase 5: Integration & Testing
- Agent 8: Build CTS, fix any remaining issues, run tests

## Success Criteria

1. ✅ CTS builds without link errors
2. ✅ All SmokeTest tests pass (14/14)
3. ✅ All SmokeTestBase tests pass (7/7)
4. ✅ All GraphBase tests pass (14/14)
5. ✅ At least 50% of Vision Feature Set tests pass
6. ✅ No regressions in existing integration tests

## Risk Analysis

**Blockers:**
1. Graph execution model needs proper topological sort
2. Reference counting must be consistent across all types
3. Async graph operations may need threading support

**Mitigation:**
1. Study existing topological sort implementation
2. Use centralized reference counting registry
3. Use std::sync primitives for thread safety

## Rollback Plan

Each agent works in isolation with git worktrees:
- If a round fails, revert that worktree
- Other rounds remain unaffected
- Integration happens only after all rounds pass local tests
