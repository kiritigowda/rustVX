# Vision Conformance Implementation Plan

**Goal:** Achieve OpenVX Vision Conformance
**Date:** April 4, 2026
**Strategy:** Implement vision kernel algorithms in parallel groups

---

## Vision Kernel Groups (Priority Order)

### Group 1: Color Conversion ⭐ HIGHEST PRIORITY
**Functions:**
- `vxColorConvertNode` / `vxuColorConvert`
  - RGB ↔ RGBX
  - RGB ↔ NV12/NV21 (YUV420)
  - RGB ↔ YUV4
  - RGB ↔ IYUV (I420)

**Why First:** Most basic operation, used by many other tests
**Algorithm:** Pixel-wise color space conversion using standard matrices

---

### Group 2: Filtering/Convolution ⭐ HIGH PRIORITY
**Functions:**
- `vxGaussian3x3Node` / `vxuGaussian3x3`
- `vxGaussian5x5Node` / `vxuGaussian5x5`
- `vxGaussianPyramidNode` / `vxuGaussianPyramid`
- `vxHalfScaleGaussianNode` / `vxuHalfScaleGaussian`
- `vxBox3x3Node` / `vxuBox3x3`
- `vxMedian3x3Node` / `vxuMedian3x3`
- `vxConvolveNode` / `vxuConvolve` (custom convolution)

**Algorithm:** Separable 1D convolution, Gaussian kernel generation

---

### Group 3: Edge Detection ⭐ HIGH PRIORITY
**Functions:**
- `vxSobel3x3Node` / `vxuSobel3x3`
- `vxMagnitudeNode` / `vxuMagnitude`
- `vxPhaseNode` / `vxuPhase`
- `vxCannyEdgeDetectorNode` / `vxuCannyEdgeDetector`

**Algorithm:** Sobel operators, gradient calculation, non-max suppression, hysteresis

---

### Group 4: Morphology ⭐ MEDIUM PRIORITY
**Functions:**
- `vxErode3x3Node` / `vxuErode3x3`
- `vxDilate3x3Node` / `vxuDilate3x3`
- `vxErode5x5Node` / `vxuErode5x5`
- `vxDilate5x5Node` / `vxuDilate5x5`

**Algorithm:** Min/Max filtering over neighborhood

---

### Group 5: Arithmetic Operations ⭐ MEDIUM PRIORITY
**Functions:**
- `vxAddNode` / `vxuAdd`
- `vxSubtractNode` / `vxuSubtract`
- `vxMultiplyNode` / `vxuMultiply`
- `vxNotNode` / `vxuNot`
- `vxAndNode` / `vxuAnd`
- `vxOrNode` / `vxuOr`
- `vxXorNode` / `vxuXor`
- `vxAbsDiffNode` / `vxuAbsDiff`
- `vxWeightedAverageNode` / `vxuWeightedAverage`

**Algorithm:** Element-wise pixel operations

---

### Group 6: Geometric Transforms ⭐ MEDIUM PRIORITY
**Functions:**
- `vxScaleImageNode` / `vxuScaleImage` (bilinear interpolation)
- `vxWarpAffineNode` / `vxuWarpAffine`
- `vxWarpPerspectiveNode` / `vxuWarpPerspective`
- `vxRemapNode` / `vxuRemap`

**Algorithm:** Bilinear/bicubic interpolation, matrix transforms

---

### Group 7: Feature Detection ⭐ MEDIUM PRIORITY
**Functions:**
- `vxHarrisCornersNode` / `vxuHarrisCorners`
- `vxFastCornersNode` / `vxuFastCorners`
- `vxHoughLinesPNode` / `vxuHoughLinesP`

**Algorithm:** Corner response calculation, non-max suppression, Hough transform

---

### Group 8: Optical Flow ⭐ LOWER PRIORITY
**Functions:**
- `vxOpticalFlowPyrLKNode` / `vxuOpticalFlowPyrLK`

**Algorithm:** Lucas-Kanade optical flow on image pyramid

---

### Group 9: Histogram & Statistics ⭐ LOWER PRIORITY
**Functions:**
- `vxHistogramNode` / `vxuHistogram`
- `vxEqualizeHistNode` / `vxuEqualizeHist`
- `vxMeanStdDevNode` / `vxuMeanStdDev`
- `vxMinMaxLocNode` / `vxuMinMaxLoc`
- `vxIntegralImageNode` / `vxuIntegralImage`
- `vxTableLookupNode` / `vxuTableLookup`

**Algorithm:** Histogram accumulation, equalization LUT, integral image

---

### Group 10: Pyramid Operations ⭐ LOWER PRIORITY
**Functions:**
- `vxLaplacianPyramidNode` / `vxuLaplacianPyramid`
- `vxLaplacianReconstructNode` / `vxuLaplacianReconstruct`

**Algorithm:** Gaussian pyramid subtraction/addition

---

### Group 11: Channel Operations ⭐ LOWER PRIORITY
**Functions:**
- `vxChannelExtractNode` / `vxuChannelExtract`
- `vxChannelCombineNode` / `vxuChannelCombine`

**Algorithm:** Channel splitting/combining

---

## Implementation Strategy

### Phase 1: Core Image Processing (Groups 1-3)
**Parallel Implementation:**
- Agent 1: Color Conversion (Group 1)
- Agent 2: Filtering (Group 2)
- Agent 3: Edge Detection (Group 3)

### Phase 2: Basic Operations (Groups 4-6)
**Parallel Implementation:**
- Agent 4: Morphology (Group 4)
- Agent 5: Arithmetic (Group 5)
- Agent 6: Geometric (Group 6)

### Phase 3: Advanced Features (Groups 7-11)
**Parallel Implementation:**
- Agent 7: Feature Detection (Group 7)
- Agent 8: Optical Flow (Group 8)
- Agent 9: Histogram & Statistics (Group 9)
- Agent 10: Pyramid & Channel (Groups 10-11)

---

## Implementation Notes

### Color Conversion Matrices
```
RGB to YUV (BT.601):
Y = 0.299R + 0.587G + 0.114B
U = -0.169R - 0.331G + 0.5B + 128
V = 0.5R - 0.419G - 0.081B + 128
```

### Gaussian Kernel (3x3, sigma=1.0)
```
[1 2 1]
[2 4 2] / 16
[1 2 1]
```

### Sobel Operators
```
Gx = [-1 0 1]    Gy = [-1 -2 -1]
     [-2 0 2]         [ 0  0  0]
     [-1 0 1]         [ 1  2  1]
```

---

## Success Criteria

- [ ] All vision kernel functions have actual algorithm implementations (not stubs)
- [ ] CTS vision tests pass without crashes
- [ ] Numerical accuracy within tolerance of reference implementation
- [ ] At least 80% of vision feature set tests pass

---

## File Locations

**Vision Kernels:** `openvx-vision/src/kernels/`

**Pattern:** Each group gets its own module file:
- `color.rs` - Color conversion
- `filter.rs` - Gaussian, box, median filtering
- `edge.rs` - Sobel, Canny
- `morph.rs` - Erode, dilate
- `arithmetic.rs` - Add, sub, mul, etc.
- `geometric.rs` - Scale, warp, remap
- `feature.rs` - Harris, FAST, Hough
- `optical_flow.rs` - Lucas-Kanade
- `histogram.rs` - Histogram, equalization
- `pyramid.rs` - Laplacian pyramid
- `channel.rs` - Channel operations

**VXU Functions:** `openvx-vision/src/vxu/` - Immediate mode wrappers

---

*Plan generated: April 4, 2026*
