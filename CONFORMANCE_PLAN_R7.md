# rustVX Conformance Plan — Baseline + Vision

## Current Status (April 19, 2026)

### Baseline Profile
| Test | Required | Pass | Fail | Status |
|---|---|---|---|---|
| GraphBase | 14 | 14 | 0 | ✅ |
| Graph.TwoNodes | 1 | 1 | 0 | ✅ |
| Graph.VirtualImage | 1 | 1 | 0 | ✅ |
| Graph.NodeRemove | 1 | 1 | 0 | ✅ |
| Graph.TwoNodesWithSameDst | 1 | 1 | 0 | ✅ |
| Graph.Cycle | 1 | 1 | 0 | ✅ |
| Graph.Cycle2 | 1 | 1 | 0 | ✅ |
| Graph.MultipleRun | 1 | 1 | 0 | ✅ |
| Graph.MultipleRunAsync | 1 | 1 | 0 | ✅ |
| Graph.NodePerformance | 1 | 1 | 0 | ✅ |
| Graph.GraphPerformance | 1 | 1 | 0 | ✅ |
| GraphCallback | 4 | 3 | 1 | ❌ (1/Reverse) |
| GraphDelay | 12 | 0 | 12 | ❌ |
| GraphROI | 3 | 1 | 2 | ❌ |
| SmokeTest | 7 | 3 | 4 | ❌ |
| SmokeTestBase | 7 | 7 | 0 | ✅ |
| TargetBase | 3 | 3 | 0 | ✅ |
| Logging | 1 | 1 | 0 | ✅ |
| Scalar | 102 | 49 | 53 | ❌ |
| Array | 23 | 0 | 23 | ❌ |
| Matrix | 13 | 12 | 1 | ❌ |
| Distribution | 1 | 0 | 1 | ❌ |
| Convolution | 4 | 3 | 1 | ❌ |
| ObjectArray | 12 | 0 | 12 | ❌ |
| Image | ? | HANG | — | ❌ |

**Baseline: ~102/214+ failing**

### Vision Profile
| Test | Required | Pass | Fail | Status |
|---|---|---|---|---|
| Scale | 982 | 982 | 0 | ✅ |
| Remap | 380 | 380 | 0 | ✅ |
| WarpPerspective | 361 | 361 | 0 | ✅ |
| HalfScaleGaussian | 25 | 25 | 0 | ✅ |
| WeightedAverage | 102 | 102 | 0 | ✅ |
| vxuMultiply | 170 | 170 | 0 | ✅ |
| vxuAddSub | 60 | 60 | 0 | ✅ |
| LUT | 38 | 38 | 0 | ✅ |
| WarpAffine | 305 | 293 | 12 | ❌ |
| Convolve | 1009 | 1008 | 1 | ❌ |
| vxMultiply | 306 | 170 | 136 | ❌ |
| Sobel3x3 | 9 | 9 | 0 | ✅ |
| Magnitude | 4 | 4 | 0 | ✅ |
| Phase | 4 | 4 | 0 | ✅ |
| Threshold | 20 | 20 | 0 | ✅ |
| vxuConvertDepth | 20 | 20 | 0 | ✅ |
| Box3x3 | 23 | 23 | 0 | ✅ |
| Gaussian3x3 | 9 | 9 | 0 | ✅ |
| Median3x3 | 12 | 12 | 0 | ✅ |
| Dilate3x3 | 12 | 12 | 0 | ✅ |
| Erode3x3 | 12 | 12 | 0 | ✅ |
| vxBinOp8u | 8 | 4 | 4 | ❌ |
| vxNot | 2 | 2 | 0 | ✅ |
| vxuBinOp8u | 4 | 4 | 0 | ✅ |
| vxuNot | 1 | 1 | 0 | ✅ |
| vxAddSub | 76 | 60 | 16 | ❌ |
| NonLinearFilter | 172 | 43 | 129 | ❌ |
| ColorConvert | 56 | 18 | 38 | ❌ |
| ChannelExtract | 51 | 16 | 35 | ❌ |
| ChannelCombine | 17 | 7 | 10 | ❌ |
| FastCorners | 24 | 0 | 24 | ❌ |
| vxCanny | 28 | 0 | 28 | ❌ |
| Histogram | 2 | 0 | 2 | ❌ |
| MeanStdDev | 4 | 0 | 4 | ❌ |
| Integral | 9 | 1 | 8 | ❌ |
| EqualizeHistogram | 2 | 0 | 2 | ❌ |
| OptFlowPyrLK | 5 | 1 | 4 | ❌ |
| HarrisCorners | 433 | HANG | — | ❌ |
| vxConvertDepth | 20 | HANG | — | ❌ |
| MinMaxLoc | ? | HANG | — | ❌ |
| GaussianPyramid | ? | HANG | — | ❌ |
| LaplacianPyramid | ? | HANG | — | ❌ |
| Image | ? | HANG | — | ❌ |

**Vision: ~3,825/4,753 pass (~80.5%), ~928 failing + HANGs**

---

## Plan: Round 7 — Multi-Agent Attack

### Agent 1: Baseline Core (GraphDelay, GraphCallback, GraphROI, SmokeTest)
**Priority: HIGH** — Baseline is required for Vision conformance

**Tasks:**
1. **GraphDelay (0/12)**: Implement `vxCreateDelay`, `vxGetDelayValue`, `vxDelayAge`, `vxAssociateDelayWithNode`. The delay mechanism allows temporal buffering of data between graph executions.
2. **GraphCallback (3/4)**: Fix CallbackOrder/Reverse — needs topological sort of nodes in graph execution (currently executes in insertion order, but the test creates nodes in reverse order and expects data-dependency execution).
3. **GraphROI (1/3)**: Implement graph-level ROI support.
4. **SmokeTest (3/7)**: Fix remaining smoke tests (likely missing API functions).

**Key files:** `openvx-core/src/unified_c_api.rs`, `openvx-core/src/c_api.rs`

### Agent 2: Baseline Data (Array, ObjectArray, Scalar, Distribution, Convolution, Matrix)
**Priority: HIGH** — Baseline is required

**Tasks:**
1. **Array (0/23)**: Implement `vxAddArrayItems`, `vxCopyArrayRange`, `vxMapArrayRange`, `vxUnmapArrayRange`, `vxQueryArray` attributes.
2. **ObjectArray (0/12)**: Implement `vxCreateObjectArray`, `vxGetStatus`, `vxQueryObjectArray`, item access.
3. **Scalar (49/102)**: Fix remaining scalar queries and operations.
4. **Distribution (0/1)**: Implement `vxQueryDistribution`.
5. **Convolution (3/4)**: Fix 1 failing test.
6. **Matrix (12/13)**: Fix 1 failing test.

**Key files:** `openvx-core/src/c_api_data.rs`, `openvx-core/src/unified_c_api.rs`, `openvx-buffer/src/c_api.rs`

### Agent 3: Vision Algorithms (HarrisCorners, NonLinearFilter, ColorConvert, Canny)
**Priority: HIGH** — biggest Vision gains

**Tasks:**
1. **HarrisCorners (HANG → 433)**: Fix the hang (likely infinite loop in corner detection), then fix accuracy. This is the single biggest gain.
2. **NonLinearFilter (43/172)**: Fix border handling for VX_BORDER_UNDEFINED (output image wrong size), implement remaining filter modes.
3. **ColorConvert (18/56)**: Implement missing color space conversions (IYUV↔NV12, YUV4↔RGB, etc.)
4. **vxCanny (0/28)**: Implement Canny edge detection properly.
5. **FastCorners (0/24)**: Implement fast corner detection.

**Key files:** `openvx-vision/src/features.rs`, `openvx-vision/src/color.rs`, `openvx-vision/src/filter.rs`, `openvx-core/src/unified_c_api.rs`

### Agent 4: Vision Misc + Fix Hangs
**Priority: MEDIUM**

**Tasks:**
1. **Fix HANGs**: HarrisCorners, vxConvertDepth, MinMaxLoc, GaussianPyramid, LaplacianPyramid, Image tests. Most likely caused by infinite loops or missing kernel execution. Add timeouts/guards.
2. **vxMultiply (170/306)**: Fix graph mode — 136 failing. Likely missing S16*S16=S16 and S16+S16=S16 combinations in graph mode (immediate mode works).
3. **vxAddSub (60/76)**: Similar to vxMultiply — graph mode combinations missing.
4. **WarpAffine (293/305)**: Fix remaining 12 tests (specific border/mode combos).
5. **ChannelExtract/Combine**: Fix planar format issues.
6. **Histogram (0/2), MeanStdDev (0/4), Integral (1/9), EqualizeHistogram (0/2)**: Implement missing graph kernel versions.
7. **vxBinOp8u (4/8)**: Fix remaining 4 tests.

**Key files:** `openvx-core/src/unified_c_api.rs`, `openvx-core/src/vxu_impl.rs`, `openvx-vision/src/`

### Cross-cutting concerns:
- **Graph kernel execution**: Many graph-mode tests fail because the kernel dispatch in `execute_node()` is incomplete. Need to implement proper kernel dispatch for ALL vision kernels, not just the ones that work in immediate mode.
- **HANG prevention**: Add iteration limits or timeouts in graph processing to prevent infinite loops.
- **Enum values**: OpenVX uses `VX_ENUM_BASE(VENDOR, TYPE)` for enum values, not simple 0/1/2. Our implementation has many hardcoded small values that should be the proper Khronos enum values (e.g., VX_ACTION_CONTINUE = 0x1000, not 0).

---

## Success Criteria
- **Baseline**: All baseline tests pass
- **Vision**: All vision profile tests pass
- **Both required** for OpenVX 1.3.1 Vision Conformance Profile