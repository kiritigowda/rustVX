# rustVX OpenVX Conformance Master Plan

## Current Status (April 5, 2026)

### Baseline Tests: ✅ 100% (25/25 passing)
- GraphBase: 14/14
- SmokeTestBase: 7/7
- Logging: 1/1
- TargetBase: 3/3

### Vision Tests: ⚠️ ~26% passing
- Tests crash during Image.CreateImageFromHandle/9 (NV12 format)
- Graph tests fail because vision kernels are stubs
- Memory corruption in image cloning

---

## Phase 1: Fix Memory & Stability Issues (CRITICAL)

### Step 1.1: Fix vxCloneImage
**Problem:** Image.VirtualImageCreation tests fail with "Unable to make a clone of vx_image"
**Files:** openvx-image/src/c_api.rs
**Agent:** image-clone-agent

### Step 1.2: Fix NV12 Memory Corruption
**Problem:** free(): invalid pointer during NV12 image creation
**Files:** openvx-image/src/c_api.rs
**Agent:** image-nv12-agent

---

## Phase 2: Implement Vision Kernel Algorithms

### Priority 1: Core Filters (Enable basic graph execution)
1. vxColorConvert (RGB↔YUV, RGB↔NV12)
2. vxGaussian3x3 / vxBox3x3
3. vxSobel3x3
4. vxAdd / vxSubtract

### Priority 2: Morphology & Threshold
1. vxDilate3x3 / vxErode3x3
2. vxThreshold
3. vxAnd / vxOr / vxXor / vxNot

### Priority 3: Advanced Features
1. vxMagnitude / vxPhase
2. vxScaleImage
3. vxWarpAffine / vxWarpPerspective
4. vxHarrisCorners / vxFASTCorners
5. vxOpticalFlowPyrLK

---

## Phase 3: Data Objects & Supporting Functions

### Step 3.1: Scalar Operations
- vxCopyScalar properly implemented

### Step 3.2: Pyramid Support
- vxCreatePyramid / vxReleasePyramid
- vxGetPyramidLevel

### Step 3.3: Remap Support
- vxCreateRemap / vxRemapImageNode

---

## Dependency Graph

```
Phase 1: Memory Fixes
    ├─ Fix vxCloneImage
    └─ Fix NV12 corruption
           ↓
Phase 2: Vision Kernels (Priority order)
    ├─ ColorConvert (enables RGB↔YUV pipelines)
    ├─ Gaussian/Box Filter (basic smoothing)
    ├─ Sobel (edge detection)
    ├─ Arithmetic (Add/Sub)
    ├─ Morphology (Dilate/Erode)
    ├─ Threshold (binary operations)
    └─ Advanced kernels (Magnitude, Warp, Corners, OpticalFlow)
           ↓
Phase 3: Data Objects
    ├─ Pyramid
    ├─ Remap
    └─ Advanced scalar ops
```

---

## Team Code Execution Plan

### Round 1: Foundation (2 agents, parallel)
- **Agent A**: Fix vxCloneImage implementation
- **Agent B**: Fix NV12 memory corruption

### Round 2: Core Kernels (3 agents, staged)
- **Agent C**: ColorConvert + Gaussian/Box (after Round 1)
- **Agent D**: Sobel + Arithmetic kernels (after Round 1)
- **Agent E**: Morphology + Threshold (after Round 1)

### Round 3: Advanced Kernels (2 agents, parallel after Round 2)
- **Agent F**: Magnitude + Phase + ScaleImage
- **Agent G**: WarpAffine + WarpPerspective + Remap

### Round 4: Feature Detection (1 agent, after Round 3)
- **Agent H**: HarrisCorners + FASTCorners + OpticalFlow

### Round 5: Data Objects & Integration (1 agent, after Round 2)
- **Agent I**: Pyramid + advanced fixes

### Round 6: Validation (1 agent, after all rounds)
- **Agent J**: Full CTS run, fix remaining issues

---

## Success Criteria

1. Baseline tests: 25/25 (maintain current)
2. Vision tests: >80% passing
3. No crashes during CTS execution
4. All smoke tests pass
5. No regressions in existing functionality

---

## Risk Analysis

**Blockers:**
1. Image format handling is complex (multiple planar formats)
2. Vision algorithms need numerical accuracy validation
3. Graph execution requires proper node scheduling

**Mitigation:**
1. Use reference implementations for format conversion
2. Compare outputs against OpenCV or Khronos reference
3. Implement topological sort for graph execution

---

## Test Commands

```bash
# Build
cd /home/simon/.openclaw/workspace/rustvx
source ~/.cargo/env
cargo build --release

# Run CTS
cd OpenVX-cts/build
LD_LIBRARY_PATH=/home/simon/.openclaw/workspace/rustvx/target/release ./bin/vx_test_conformance

# Check specific test
cd OpenVX-cts/build
LD_LIBRARY_PATH=/home/simon/.openclaw/workspace/rustvx/target/release ./bin/vx_test_conformance 2>&1 | grep -E "Image.CreateImageFromHandle"
```
