# rustVX Vision Conformance Execution Plan
**Date:** April 19, 2026
**Goal:** Pass OpenVX 1.3.1 Vision Conformance Profile
**Current Branch:** fix/openvx-1.3.1-conformance-phase2

---

## Current CTS Test Results (April 19, 2026)

### Baseline (24 tests) — ✅ 24/24 PASS
- GraphBase: 14/14 ✅
- Logging: 1/1 ✅
- SmokeTestBase: 7/7 ✅
- TargetBase: 3/3 ✅

### Graph Tests (54+ tests)
| Test | Status | Notes |
|------|--------|-------|
| Graph.TwoNodes | ✅ PASS | |
| Graph.GraphFactory | ✅ PASS | |
| Graph.VirtualImage | ✅ PASS | |
| Graph.VirtualArray | ❌ FAIL | vxProcessGraph node fails with VX_ERROR_INVALID_PARAMETERS |
| Graph.NodeRemove | ✅ PASS | |
| Graph.NodeFromEnum | ⏰ HANG | Infinite loop |
| Graph.TwoNodesWithSameDst | ✅ PASS | |
| Graph.Cycle | ✅ PASS | |
| Graph.Cycle2 | ✅ PASS | |
| Graph.MultipleRun | ✅ PASS | |
| Graph.MultipleRunAsync | ✅ PASS | |
| Graph.NodePerformance | ❌ FAIL | VX_ERROR_NOT_IMPLEMENTED |
| Graph.GraphPerformance | ❌ FAIL | perf.num assertion |
| Graph.ReplicateNode | ❌ FAIL (0/48) | NULL pointer issues |
| Graph.KernelName | ✅ 42/42 | All kernel names registered |

### Image Tests
| Test | Pass/Total | Notes |
|------|-----------|-------|
| Image.RngImageCreation | 14/14 ✅ | |
| Image.ImageCreation_U1 | 1/1 ✅ | |
| Image.VirtualImageCreation | 19/19 ✅ | |
| Image.VirtualImageCreationDims | 4/4 ✅ | |
| Image.Convert_CT_Image | 13/13 ✅ | |
| Image.QueryImage | 0/1 ❌ | |
| Image.UniformImage | 7/13 | |
| Image.CreateImageFromHandle | 0/39 ❌ | |
| Image.SwapImageHandle | 0/78 ❌ | |
| Image.FormatImagePatchAddress1d | ⏰ HANG | |
| Image.vxSetImagePixelValues | 0/13 ❌ | |

### Image Patch Operations
| Test | Pass/Total |
|------|-----------|
| vxMapImagePatch | 87/156 |
| vxCopyImagePatch | 60/117 |
| vxCreateImageFromChannel | 18/54 |

### Data Objects
| Test | Pass/Total | Notes |
|------|-----------|-------|
| Array | 5/23 | |
| Scalar.Create* | 15/34 | Type issues |
| Scalar.CopyScalar | ⏰ HANG | |
| Scalar.CopyScalarWithSize | 0/19 | |
| Scalar.CreateVirtualScalar | 0/19 | |
| Scalar.QueryScalar | 0/15 | |
| Matrix | 6/13 | |
| LUT | 0/38 | |
| Threshold | 20/20 ✅ | |
| Distribution | 0/1 | |
| ObjectArray | 0/12 | |

### Vision Kernels - Filters (WORKING)
| Test | Pass/Total |
|------|-----------|
| Box3x3 | 23/23 ✅ |
| Gaussian3x3 | 9/9 ✅ |
| Median3x3 | 12/12 ✅ |
| Dilate3x3 | 12/12 ✅ |
| Erode3x3 | 12/12 ✅ |
| Convolve | 1008/1009 |
| Convolution | 3/4 |

### Vision Kernels - Color & Channels
| Test | Pass/Total |
|------|-----------|
| ColorConvert | 14/56 |
| ChannelExtract | 16/51 |
| ChannelCombine | 7/17 |

### Vision Kernels - Gradient
| Test | Pass/Total |
|------|-----------|
| Sobel3x3 | 1/9 |
| Magnitude | 0/4 |
| Phase | 0/4 |

### Vision Kernels - Geometric
| Test | Pass/Total |
|------|-----------|
| Scale | 488/982 |
| WarpAffine | 71/305 |
| WarpPerspective | 174/361 |
| Remap | 52/380 |

### Vision Kernels - Arithmetic
| Test | Pass/Total |
|------|-----------|
| vxAddSub | 42/76 |
| vxMultiply | 5/306 |
| vxConvertDepth | 0/20 |
| WeightedAverage | 0/102 |
| vxNot | 0/2 |
| vxBinOp8u | 4/8 |
| vxBinOp16s | 0/2 |

### Vision Kernels - Feature Detection
| Test | Pass/Total |
|------|-----------|
| HarrisCorners | 0/433 |
| FastCorners | 0/24 |
| vxCanny | 0/28 |

### Vision Kernels - Pyramids
| Test | Pass/Total |
|------|-----------|
| HalfScaleGaussian | 0/25 |
| GaussianPyramid | ⏰ HANG |
| LaplacianPyramid | ⏰ HANG |
| LaplacianReconstruct | 0/9 |

### Vision Kernels - Statistics & Advanced
| Test | Pass/Total |
|------|-----------|
| NonLinearFilter | 0/172 |
| Histogram | 0/2 |
| EqualizeHistogram | 0/2 |
| MinMaxLoc | ⏰ HANG |
| MeanStdDev | 0/4 |
| Integral | 1/9 |

### VXU Immediate Functions
| Test | Pass/Total |
|------|-----------|
| vxuAddSub | 42/60 |
| vxuMultiply | 0/170 |
| vxuConvertDepth | 2/20 |
| vxuBinOp8u | 4/4 ✅ |
| vxuBinOp16s | 0/1 |
| vxuCanny | 0/28 |
| vxuNot | 1/1 ✅ |

### Other Groups
| Test | Pass/Total | Notes |
|------|-----------|-------|
| SmokeTest | 5/7 | |
| Target | 4/8 | |
| GraphCallback | 0/4 | |
| GraphDelay | 0/12 | |
| GraphROI | 1/3 | |
| UserNode | 0/74 | |
| OptFlowPyrLK | 1/5 | |
| vxCopyRemapPatch | 1/1 ✅ | |
| vxMapRemapPatch | ⏰ HANG | |

---

## Agent Assignment Plan

### Round 1: Foundation Fixes (4 agents, parallel)

**Agent 1: Graph & Core Fixes**
- Fix Graph.VirtualArray (user kernel execution)
- Fix Graph.NodeFromEnum hang (infinite loop)
- Fix Graph.ReplicateNode (0/48)
- Fix Graph.NodePerformance / Graph.GraphPerformance
- Files: openvx-core/src/unified_c_api.rs
- Verify: CTS Graph.* tests pass

**Agent 2: Image API Fixes**
- Fix Image.CreateImageFromHandle (0/39)
- Fix Image.SwapImageHandle (0/78)
- Fix Image.vxSetImagePixelValues (0/13)
- Fix Image.QueryImage (0/1)
- Fix Image.UniformImage (7/13)
- Fix Image.FormatImagePatchAddress1d hang
- Fix vxMapImagePatch (87/156 → target 156/156)
- Fix vxCopyImagePatch (60/117 → target 117/117)
- Fix vxCreateImageFromChannel (18/54 → target 54/54)
- Files: openvx-image/src/c_api.rs, openvx-core/src/unified_c_api.rs
- Verify: CTS Image.* tests pass

**Agent 3: Data Objects + Color/Channel**
- Fix Scalar (hang + query + copy + virtual)
- Fix LUT (0/38)
- Fix Matrix (6/13 → 13/13)
- Fix Array (5/23 → 23/23)
- Fix ObjectArray (0/12)
- Fix Distribution (0/1)
- Fix ColorConvert (14/56 → target 56/56)
- Fix ChannelExtract (16/51 → 51/51)
- Fix ChannelCombine (7/17 → 17/17)
- Files: openvx-buffer/src/c_api.rs, openvx-core/src/unified_c_api.rs, openvx-vision/src/color.rs
- Verify: CTS Scalar.*, LUT.*, Matrix.*, Array.*, Color*.* tests pass

**Agent 4: Arithmetic + Gradient + ConvertDepth**
- Fix vxMultiply (5/306 → target 306/306)
- Fix vxAddSub (42/76 → 76/76)
- Fix vxConvertDepth (0/20 → 20/20)
- Fix WeightedAverage (0/102 → 102/102)
- Fix vxNot (0/2)
- Fix vxBinOp8u (4/8 → 8/8)
- Fix vxBinOp16s (0/2)
- Fix Sobel3x3 (1/9 → 9/9)
- Fix Magnitude (0/4 → 4/4)
- Fix Phase (0/4 → 4/4)
- Fix VXU counterparts
- Files: openvx-vision/src/arithmetic.rs, openvx-vision/src/gradient.rs, openvx-core/src/vxu_impl.rs, openvx-core/src/unified_c_api.rs
- Verify: CTS vxAddSub.*, vxMultiply.*, vxConvertDepth.*, Sobel3x3.*, etc.

### Round 2: Advanced Features (3 agents, after Round 1 merges)

**Agent 5: Geometric Operations**
- Fix Scale (488/982 → 982/982)
- Fix WarpAffine (71/305 → 305/305)
- Fix WarpPerspective (174/361 → 361/361)
- Fix Remap (52/380 → 380/380) + fix vxMapRemapPatch hang
- Files: openvx-vision/src/geometric.rs, openvx-core/src/unified_c_api.rs
- Verify: CTS Scale.*, WarpAffine.*, WarpPerspective.*, Remap.*

**Agent 6: Feature Detection + Pyramids + Statistics**
- Fix HarrisCorners (0/433 → 433/433)
- Fix FastCorners (0/24 → 24/24)
- Fix vxCanny/vxuCanny (0/28 → 28/28)
- Fix HalfScaleGaussian (0/25 → 25/25)
- Fix GaussianPyramid hang
- Fix LaplacianPyramid hang
- Fix LaplacianReconstruct (0/9 → 9/9)
- Fix NonLinearFilter (0/172 → 172/172)
- Fix Histogram (0/2), EqualizeHistogram (0/2)
- Fix MinMaxLoc hang
- Fix MeanStdDev (0/4), Integral (1/9 → 9/9)
- Files: openvx-vision/src/features.rs, openvx-vision/src/statistics.rs, openvx-vision/src/filter.rs, openvx-core/src/unified_c_api.rs
- Verify: CTS HarrisCorners.*, FastCorners.*, vxCanny.*, etc.

**Agent 7: Graph Advanced + UserNode + SmokeTest + Target**
- Fix UserNode (0/74)
- Fix GraphCallback (0/4)
- Fix GraphDelay (0/12)
- Fix GraphROI (1/3 → 3/3)
- Fix SmokeTest (5/7 → 7/7)
- Fix Target (4/8 → 8/8)
- Fix OptFlowPyrLK (1/5 → 5/5)
- Files: openvx-core/src/unified_c_api.rs, openvx-vision/src/optical_flow.rs
- Verify: CTS UserNode.*, GraphCallback.*, etc.

---

## Integration Strategy

1. After each round, merge agent branches back to fix/openvx-1.3.1-conformance-phase2
2. Run full CTS to check for regressions
3. Fix merge conflicts (agent who created resolves)
4. Final integration run after Round 2