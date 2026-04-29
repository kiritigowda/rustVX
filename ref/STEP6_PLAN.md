# Step 6: Feature Detection Fixes

## Current Status
- FastCorners: 0/24 — Algorithm correctness issue
- HarrisCorners: 0/~433 — NodeCreation fails (dangling refs)
- vxCanny: 0/28 — Implementation issues
- Histogram: 0/2 — Distribution data type mismatch (u32 vs i32)

## Key Issues and Fixes

### 1. Histogram (easiest fix)
- **Bug**: VxCDistribution.data is `Vec<u32>` but OpenVX spec says bins are `vx_int32`
- **Fix**: Change `Vec<u32>` to `Vec<i32>` throughout, update vxCopyDistribution and vxMapDistribution
- **Files**: `openvx-core/src/unified_c_api.rs` (VxCDistribution struct + copy/map functions), `openvx-core/src/vxu_impl.rs` (histogram function)

### 2. HarrisCorners NodeCreation
- **Bug**: Dangling image references — graph cleanup doesn't properly handle intermediate images created by HarrisCorners
- **Fix**: Check vxReleaseGraph cleanup for VX_TYPE_ARRAY (keypoints array), and ensure intermediate images are released properly
- **Files**: `openvx-core/src/unified_c_api.rs` (graph cleanup), `openvx-core/src/c_api.rs`

### 3. FastCorners (algorithmic fix)
- **Bug**: Corner detection results don't match reference — likely issues with NMS and/or strength computation
- **Reference**: MIVisionX `ago_haf_cpu_fast_corners.cpp` in this directory
- **Key differences from reference**:
  - Reference uses contiguous arc check (isCorner with 16-bit mask)
  - Reference NMS uses strict `>` for horizontal neighbors (not `>=`)
  - Reference writes strength to scratch buffer, then does NMS in separate pass
- **Fix**: Rewrite to match MIVisionX algorithm (without SSE)
- **Files**: `openvx-core/src/vxu_impl.rs` (vxu_fast_corners_impl)

### 4. vxCanny (algorithmic fix)
- **Bug**: Hysteresis edge tracking only does single pass, not flood-fill
- **Reference**: MIVisionX `ago_haf_cpu_canny.cpp` in this directory
- **Key differences from reference**:
  - Reference uses proper flood-fill: strong edges are pushed to a stack, then the stack processes connected weak edges
  - Reference uses L1 norm (|Gx| + |Gy|) for magnitude, not L2 norm (sqrt(Gx²+Gy²))
  - Reference stores magnitude and direction together in a U16 (mag<<2 | dir)
  - Reference NMS uses proper direction quantization (0=0°, 1=45°, 2=90°, 3=135°)
  - Final pass: converts all 127 (weak/not-connected) to 0, all strong to 255
- **Fix**: Rewrite Canny to match MIVisionX (3x3 Sobel only for CTS, proper flood-fill hysteresis)
- **Files**: `openvx-core/src/vxu_impl.rs` (canny_edge_detector)

## Build & Test Commands
```bash
cd /home/simon/.openclaw/workspace/rustvx
cargo build --release
cd OpenVX-cts/build && make -j$(nproc)
LD_LIBRARY_PATH=../target/release VX_TEST_DATA_PATH=../OpenVX-cts/test_data/ ../OpenVX-cts/build/bin/vx_test_conformance --filter="Histogram.*"
LD_LIBRARY_PATH=../target/release VX_TEST_DATA_PATH=../OpenVX-cts/test_data/ ../OpenVX-cts/build/bin/vx_test_conformance --filter="FastCorners.*"
LD_LIBRARY_PATH=../target/release VX_TEST_DATA_PATH=../OpenVX-cts/test_data/ ../OpenVX-cts/build/bin/vx_test_conformance --filter="vxCanny.*"
LD_LIBRARY_PATH=../target/release VX_TEST_DATA_PATH=../OpenVX-cts/test_data/ ../OpenVX-cts/build/bin/vx_test_conformance --filter="HarrisCorners.*"
```