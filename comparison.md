# OpenVX Benchmark Comparison

**khronos.sample** vs **RustVX OpenVX Implementation**

## System Info

| Property | Value |
|:---|:---|
| CPU | AMD EPYC 7763 64-Core Processor |
| Cores | 4 |
| RAM | 15.6 GB |
| OS | Linux 6.8.0-1052-azure |

> Same hardware — both benchmarks ran on identical hardware.

## Conformance & Scores

| Metric | khronos.sample | RustVX OpenVX Implementation |
|:---|---:|---:|
| Vision Score (MP/s) | 44.15 | 197.25 |
| Conformance | PASS (41/41) | PASS (41/41) |

## Category Sub-Scores

| Category | khronos.sample (MP/s) | RustVX OpenVX Implementation (MP/s) | Change % |
|:---|---:|---:|---:|
| color | 125.10 | 329.69 | +163.5 |
| feature | 4.14 | 25.19 | +508.5 |
| filters | 18.08 | 293.81 | +1525.1 |
| geometric | 29.32 | 68.66 | +134.2 |
| misc | 91.29 | 269.13 | +194.8 |
| multiscale | 715.63 | 76.26 | -89.3 |
| pipeline_feature | 13.88 | 60.52 | +336.0 |
| pipeline_vision | 7.79 | 242.60 | +3014.2 |
| pixelwise | 111.18 | 364.40 | +227.8 |
| statistical | 132.15 | 700.81 | +430.3 |

## Summary

| Metric | Count |
|:---|---:|
| Total benchmarks compared | 52 |
| Both verified | 52 |

## Detailed Comparison

> Speedup = RustVX OpenVX Implementation throughput / khronos.sample throughput. Values >1.00 mean RustVX OpenVX Implementation is faster.

| Benchmark | Mode | Resolution | khronos.sample (ms) | khronos.sample (MP/s) | khronos.sample Verified | RustVX OpenVX Implementation (ms) | RustVX OpenVX Implementation (MP/s) | RustVX OpenVX Implementation Verified | Speedup |
|:---|:---|:---|---:|---:|:---:|---:|---:|:---:|---:|
| LaplacianPyramid | graph | FHD | 0.002 | 1254446.5 | PASS | 69.637 | 29.8 | PASS | 0.00x |
| ScaleImage_Half | graph | FHD | 13.686 | 151.5 | PASS | 8.431 | 245.9 | PASS | 1.62x |
| NonLinearFilter | graph | FHD | 224.776 | 9.2 | PASS | 135.354 | 15.3 | PASS | 1.66x |
| ScaleImage_Double | graph | FHD | 217.881 | 9.5 | PASS | 130.694 | 15.9 | PASS | 1.67x |
| IntegralImage | graph | FHD | 2.470 | 839.4 | PASS | 1.468 | 1412.0 | PASS | 1.68x |
| Multiply | graph | FHD | 18.796 | 110.3 | PASS | 11.023 | 188.1 | PASS | 1.71x |
| Phase | graph | FHD | 101.832 | 20.4 | PASS | 58.320 | 35.6 | PASS | 1.75x |
| Magnitude | graph | FHD | 17.660 | 117.4 | PASS | 9.450 | 219.4 | PASS | 1.87x |
| ChannelExtract | graph | FHD | 11.635 | 178.2 | PASS | 6.206 | 334.1 | PASS | 1.87x |
| Threshold_Range | graph | FHD | 21.247 | 97.6 | PASS | 11.327 | 183.1 | PASS | 1.88x |
| SobelMagnitudePhase | graph | FHD | 232.042 | 8.9 | PASS | 118.655 | 17.5 | PASS | 1.96x |
| Not | graph | FHD | 11.671 | 177.7 | PASS | 5.703 | 363.6 | PASS | 2.05x |
| WarpPerspective | graph | FHD | 107.007 | 19.4 | PASS | 49.946 | 41.5 | PASS | 2.14x |
| Remap | graph | FHD | 97.754 | 21.2 | PASS | 42.550 | 48.7 | PASS | 2.30x |
| And | graph | FHD | 18.150 | 114.2 | PASS | 7.853 | 264.0 | PASS | 2.31x |
| Xor | graph | FHD | 18.129 | 114.4 | PASS | 7.833 | 264.7 | PASS | 2.31x |
| Or | graph | FHD | 18.135 | 114.3 | PASS | 7.774 | 266.7 | PASS | 2.33x |
| ChannelCombine | graph | FHD | 20.778 | 99.8 | PASS | 8.547 | 242.6 | PASS | 2.43x |
| Sobel3x3 | graph | FHD | 125.626 | 16.5 | PASS | 51.131 | 40.5 | PASS | 2.46x |
| ThresholdedEdge | graph | FHD | 169.631 | 12.2 | PASS | 66.605 | 31.1 | PASS | 2.55x |
| ColorConvert_RGB2NV12 | graph | FHD | 19.008 | 109.1 | PASS | 6.977 | 297.2 | PASS | 2.72x |
| ColorConvert_RGB2IYUV | graph | FHD | 20.003 | 103.7 | PASS | 7.196 | 288.2 | PASS | 2.78x |
| WeightedAverage | graph | FHD | 16.103 | 128.8 | PASS | 5.448 | 380.6 | PASS | 2.96x |
| MeanStdDev | graph | FHD | 15.704 | 132.0 | PASS | 5.152 | 402.5 | PASS | 3.05x |
| CannyEdgeDetector | graph | FHD | 313.817 | 6.6 | PASS | 96.394 | 21.5 | PASS | 3.25x |
| ConvertDepth | graph | FHD | 13.613 | 152.3 | PASS | 3.697 | 560.9 | PASS | 3.68x |
| EdgeDetection | graph | FHD | 406.238 | 5.1 | PASS | 98.425 | 21.1 | PASS | 4.13x |
| AbsDiff | graph | FHD | 24.533 | 84.5 | PASS | 5.697 | 364.0 | PASS | 4.31x |
| HistogramEqualize | graph | FHD | 51.971 | 39.9 | PASS | 11.660 | 177.8 | PASS | 4.46x |
| TableLookup | graph | FHD | 11.020 | 188.2 | PASS | 2.410 | 860.4 | PASS | 4.57x |
| EqualizeHist | graph | FHD | 20.348 | 101.9 | PASS | 3.991 | 519.5 | PASS | 5.10x |
| WarpAffine | graph | FHD | 56.694 | 36.6 | PASS | 10.727 | 193.3 | PASS | 5.28x |
| FastCorners | graph | FHD | 1021.028 | 2.0 | PASS | 190.607 | 10.9 | PASS | 5.36x |
| HalfScaleGaussian | graph | FHD | 71.544 | 29.0 | PASS | 10.715 | 193.5 | PASS | 6.68x |
| HarrisTracker | graph | FHD | 378.023 | 5.5 | PASS | 51.798 | 40.0 | PASS | 7.29x |
| GaussianPyramid | graph | FHD | 205.711 | 10.1 | PASS | 26.948 | 77.0 | PASS | 7.63x |
| Threshold_Binary | graph | FHD | 20.251 | 102.4 | PASS | 2.552 | 812.4 | PASS | 7.93x |
| Subtract | graph | FHD | 21.351 | 97.1 | PASS | 2.547 | 814.0 | PASS | 8.38x |
| Add | graph | FHD | 21.353 | 97.1 | PASS | 2.520 | 822.8 | PASS | 8.47x |
| HarrisCorners | graph | FHD | 376.042 | 5.5 | PASS | 44.238 | 46.9 | PASS | 8.51x |
| Histogram | graph | FHD | 15.972 | 129.8 | PASS | 1.794 | 1155.7 | PASS | 8.90x |
| OpticalFlowPyrLK | graph | FHD | 520.339 | 4.0 | PASS | 56.483 | 36.7 | PASS | 9.20x |
| MinMaxLoc | graph | FHD | 75.451 | 27.5 | PASS | 4.186 | 495.4 | PASS | 18.03x |
| CustomConvolution | graph | FHD | 88.983 | 23.3 | PASS | 4.284 | 484.0 | PASS | 20.77x |
| Gaussian3x3 | graph | FHD | 65.153 | 31.8 | PASS | 2.917 | 711.0 | PASS | 22.34x |
| Box3x3 | graph | FHD | 65.195 | 31.8 | PASS | 2.855 | 726.3 | PASS | 22.83x |
| Dilate3x3 | graph | FHD | 72.847 | 28.5 | PASS | 2.905 | 713.9 | PASS | 25.08x |
| Erode3x3 | graph | FHD | 72.900 | 28.4 | PASS | 2.867 | 723.2 | PASS | 25.43x |
| MorphologyOpen | graph | FHD | 158.702 | 13.1 | PASS | 1.552 | 1336.4 | PASS | 102.25x |
| MorphologyClose | graph | FHD | 158.837 | 13.1 | PASS | 1.543 | 1343.9 | PASS | 102.98x |
| Median3x3 | graph | FHD | 527.396 | 3.9 | PASS | 2.994 | 692.6 | PASS | 176.23x |
| DualFilter | graph | FHD | 563.033 | 3.7 | PASS | 1.632 | 1270.8 | PASS | 345.34x |

