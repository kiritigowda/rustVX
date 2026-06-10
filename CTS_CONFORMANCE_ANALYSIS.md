# rustVX Implementation Analysis vs OpenVX CTS Conformance Issues

## Repository: kiritigowda/rustVX
## Analysis Date: 2025-06-05

---

## 1. HOG (Histogram of Oriented Gradients) Implementation

### 1.1 vxHOGCellsNode Implementation

**Location**: `openvx-core/src/vxu_impl.rs` - `vxu_hog_cells_impl`

**Key Implementation Details**:
- **Dimensions**: Uses 2D tensors for magnitudes `[num_cells_x, num_cells_y]` and 3D tensors for bins `[num_cells_x, num_cells_y, num_bins]`
- **Data Types**: Hardcoded to `VX_TYPE_INT16` for both magnitudes and bins
- **Cell calculation**: `num_cells_x = width / cell_w`, `num_cells_y = height / cell_h`
- **Border handling**: Uses replicate border for Sobel gradient calculation
- **Accumulation**: Accumulates directly into INT16 arrays with truncation at each pixel

**Potential CTS Issues**:
1. **Data Type Validation**: The code validates `mag_dtype != VX_TYPE_INT16` and `bins_dtype != VX_TYPE_INT16` - CTS may expect different type handling
2. **Dimension Ordering**: Uses `[num_cells_x, num_cells_y]` layout - CTS may expect different dimension ordering
3. **Magnitude Calculation**: Uses `sqrt(gx*gx + gy*gy)` - CTS may expect L1 norm or different magnitude calculation
4. **Bin Indexing**: `(orientation * num_div_360).floor()` - CTS may expect rounding differences

### 1.2 vxHOGFeaturesNode Implementation

**Location**: `openvx-core/src/vxu_impl.rs` - `vxu_hog_features_impl`

**Key Implementation Details**:
- **Input**: Takes magnitudes (2D INT16), bins (3D INT16), and HOG params struct
- **Output**: 3D INT16 tensor `[num_windows_w, num_windows_h, feature_dim]`
- **Q7.8 Format**: Stores features as Q7.8 fixed-point (`value * 256.0`)
- **L2-Hys Normalization**: Implements renormalization using truncated Q7.8 values
- **Window calculation**: 
  - `num_windows_w = (width - window_w) / window_stride + 1`
  - `blocks_per_window_w = (window_w - block_w) / block_stride + 1`

**Potential CTS Issues**:
1. **Feature Dimension Calculation**: May differ from CTS reference if cell/block arithmetic varies
2. **Normalization**: L2-Hys uses truncation at Q7.8 conversion - CTS may expect different rounding
3. **Feature Tensor Layout**: 3D tensor `[windows_w, windows_h, feature_dim]` - dimension ordering critical
4. **Threshold Application**: `min(threshold)` applied before Q7.8 conversion

---

## 2. BilateralFilter Implementation

### 2.1 3D Tensor Handling

**Location**: `openvx-core/src/vxu_impl.rs` - `vxu_bilateral_filter_impl_with_border`

**Key Implementation Details**:
- **Dimension Support**: Supports 2D and 3D tensors
- **Data Types**: U8 (`VX_TYPE_UINT8`) and S16 (`VX_TYPE_INT16`)
- **Dimension Layout**: 
  - 2D: `[width, height]`
  - 3D: `[channels, width, height]` or `[width, height, channels]`
- **Strides calculation**: Built from dimensions for multi-dimensional access

**3D Tensor Logic**:
```rust
let (width, height) = if num_dims >= 2 {
    (dims[num_dims - 2] as i32, dims[num_dims - 1] as i32)
} else {
    (dims[0] as i32, 1)
};
let channels = if width * height > 0 {
    total_elements / (width * height)
} else { 1 };
```

**Potential CTS Issues**:
1. **Dimension Interpretation**: Last 2 dimensions treated as spatial (width, height) - CTS may expect different layout
2. **Channel Handling**: Multi-channel processing may differ from CTS reference
3. **Stride Calculation**: Row-major strides may not match CTS expectations for certain layouts

### 2.2 Border Mode Handling

**Key Implementation Details**:
- **Supported modes**: Undefined, Constant, Replicate
- **Constant border**: Uses `border.constant_value.U8` for U8, may not handle S16 constant values properly
- **Undefined mode**: Skips border pixels (`low_x = radius`, `high_x = width - radius`)
- **Replicate mode**: Clamps coordinates to image bounds

**Border Implementation**:
```rust
let border_mode = match border {
    Some(b) => match b.mode {
        0x0000C000 => BorderMode::Undefined,
        0x0000C001 => BorderMode::Constant(val),
        0x0000C002 => BorderMode::Replicate,
        _ => BorderMode::Undefined,
    },
    None => BorderMode::Undefined,
};
```

**Potential CTS Issues**:
1. **S16 Constant Border**: Constant value reading only handles U8 via `b.constant_value.U8` - S16 constant borders may be broken
2. **Border Pixel Skipping**: Undefined mode skips pixels but may leave them uninitialized
3. **Border Mode Default**: Defaults to Undefined when not specified - CTS may expect different default

### 2.3 Color Weight Tables

**S16 Specific Logic**:
- Calculates min/max values in tensor to determine range
- Creates lookup table with `k_exp_num_bins = 4096 * channels`
- Uses `scale_index_s16 = k_exp_num_bins / (max_val - min_val)`

**Potential CTS Issues**:
1. **Range Calculation**: Min/max scan over entire tensor may differ from CTS
2. **Table Size**: 4096 bins per channel may not match CTS implementation
3. **Edge Case**: When range is near zero, copies input to output - CTS may expect different behavior

---

## 3. TensorConvertDepth Implementation

### 3.1 Data Type Handling

**Location**: `openvx-core/src/vxu_impl.rs` - `vxu_tensor_convert_depth_impl`

**Key Implementation Details**:
- **Input types supported**: INT16, UINT8, INT8
- **Output types supported**: INT16, UINT8, INT8
- **Formula**: `converted = (val - offset) * scale` where `scale = 1.0 / norm`
- **INT16 handling**: Assumes Q7.8 format (`val / 256.0` for input, `converted * 256.0` for output)

**Type Conversion Logic**:
```rust
let val = match in_dtype {
    VX_TYPE_INT16 => {
        let bytes = [in_data[i*2], in_data[i*2+1]];
        i16::from_ne_bytes(bytes) as f64 / 256.0  // Q7.8 decode
    }
    VX_TYPE_UINT8 => in_data[i] as f64,
    VX_TYPE_INT8 => in_data[i] as i8 as f64,
    _ => return VX_ERROR_INVALID_PARAMETERS,
};
```

**Potential CTS Issues**:
1. **Q7.8 Assumption**: Hardcoded division/multiplication by 256 for INT16 - CTS may expect raw values
2. **Policy Handling**: Wrap policy uses bitwise AND (`tmp as i64 & 0xFFFF`) - may differ from CTS overflow behavior
3. **Saturation**: Saturate policy uses `clamp()` - CTS may expect different saturation bounds
4. **Endianness**: Uses `from_ne_bytes()` - may differ on big-endian systems

### 3.2 Output Type Handling

**Output Type Logic**:
```rust
match out_dtype {
    VX_TYPE_INT16 => {
        let tmp = converted * 256.0;
        let clamped = if _policy == VX_CONVERT_POLICY_WRAP {
            (tmp as i64 & 0xFFFF) as i16
        } else {
            tmp.clamp(i16::MIN as f64, i16::MAX as f64) as i16
        };
    }
    VX_TYPE_UINT8 => { /* similar */ }
    VX_TYPE_INT8 => { /* similar */ }
}
```

**Potential CTS Issues**:
1. **INT16 Output**: Multiplies by 256 (Q7.8 encoding) - CTS may expect raw INT16 output
2. **Sign Extension**: Wrap policy for INT16 uses `0xFFFF` mask - may not handle sign correctly
3. **Float Precision**: Uses f64 for intermediate calculations - may differ from CTS fixed-point math

---

## 4. WeightedAverage Kernel Registration

### 4.1 Kernel Registration

**Location**: 
- `openvx-vision/src/arithmetic.rs` - `WeightedAverageKernel`
- `openvx-core/src/unified_c_api.rs` - C API wrapper

**Key Implementation Details**:
- **Kernel name**: `"org.khronos.openvx.weighted_average"`
- **C API Registration**: Uses `create_node_with_params` with kernel name string
- **Graph node creation**: `vxWeightedAverageNode` creates node via `get_kernel_by_name`

**Kernel Registration in openvx-vision**:
```rust
impl KernelTrait for WeightedAverageKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.weighted_average"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::WeightedAverage
    }
    // ...
}
```

**C API Registration**:
```rust
pub extern "C" fn vxWeightedAverageNode(
    graph: vx_graph,
    img1: vx_image,
    alpha: vx_scalar,
    img2: vx_image,
    output: vx_image,
) -> vx_node {
    create_node_with_params(
        graph,
        "org.khronos.openvx.weighted_average",
        &[img1 as vx_reference, alpha as vx_reference, 
          img2 as vx_reference, output as vx_reference],
    )
}
```

**Potential CTS Issues**:
1. **Kernel Name Mismatch**: CTS may expect different kernel name format
2. **Parameter Count**: 4 parameters (src1, alpha, src2, dst) - CTS may validate parameter count strictly
3. **Alpha Type**: Uses `vx_scalar` for alpha - CTS may expect specific scalar type validation
4. **Kernel Enum**: `VX_KERNEL_WEIGHTED_AVERAGE = 0x40` - CTS may expect different enum value

### 4.2 Implementation Formula

**Formula**: `(src1 * alpha + src2 * (256 - alpha)) / 256`

**Potential CTS Issues**:
1. **Rounding**: Integer division truncates - CTS may expect rounding
2. **Overflow**: Intermediate calculation may overflow - CTS may expect saturation
3. **Alpha Range**: May not validate alpha is in [0, 256] range

---

## 5. MinMaxLoc Implementation

### 5.1 minCount/maxCount Types

**Location**: `openvx-core/src/vxu_impl.rs` - `vxu_min_max_loc_impl`

**Key Implementation Details**:
- **minCount/maxCount**: Written as `u32` (32-bit unsigned)
- **Type detection**: Uses `is_s16` to determine image type
- **Value writing**: min_val/max_val written as native image type (u8 or i16)

**Count Writing Logic**:
```rust
if !min_count_scalar.is_null() {
    let count = result.min_locs.len() as u32;
    crate::c_api_data::vxCopyScalarData(
        min_count_scalar,
        &count as *const u32 as *mut c_void,
        0x11002, // VX_TYPE_UINT32
        0x0,
    );
}
```

**Potential CTS Issues**:
1. **Count Type**: Uses `u32` - CTS OpenVX spec may expect `vx_size` (usize/size_t)
2. **Scalar Type**: Hardcoded `0x11002` (VX_TYPE_UINT32) - CTS may expect type checking against scalar's actual type
3. **Null Handling**: Optional parameters checked with `is_null()` - CTS may pass invalid but non-null scalars

### 5.2 Value Type Handling

**Value Writing Logic**:
```rust
if !min_val_scalar.is_null() {
    if is_s16 {
        let v = result.min_val as i16;
        crate::c_api_data::vxCopyScalarData(
            min_val_scalar, &v as *const i16 as *mut c_void, 0x11002, 0x0);
    } else {
        let v = result.min_val as u8;
        crate::c_api_data::vxCopyScalarData(
            min_val_scalar, &v as *const u8 as *mut c_void, 0x11002, 0x0);
    }
}
```

**Potential CTS Issues**:
1. **Type Code**: Uses `0x11002` for all types - CTS may expect type-specific codes
2. **S16 Values**: Written as i16 - CTS may expect raw values or scaled values
3. **U8 Values**: Written as u8 - CTS may expect different handling

### 5.3 Location Array Handling

**Location Writing**:
- Uses `vxTruncateArray` followed by `vxAddArrayItems`
- Converts to `vx_coordinates2d_t` {x, y} format
- Respects array capacity

**Potential CTS Issues**:
1. **Array Type**: CTS may expect VX_TYPE_COORDINATES2D specifically
2. **Order**: Locations written in discovery order - CTS may expect sorted order
3. **Duplicates**: All locations with min/max value added - CTS may limit count

---

## 6. CopyImagePatch Implementation

### 6.1 Uniform Image Handling

**Location**: `openvx-image/src/c_api.rs` - `vxCopyImagePatch`

**Key Implementation Details**:
- **No special uniform handling**: The code does NOT check for uniform images
- **Direct memory access**: Copies from `img.data` regardless of uniform status
- **Uniform image creation**: `vxCreateUniformImage` fills data with uniform value but doesn't set special flags

**Relevant Code**:
```rust
pub extern "C" fn vxCopyImagePatch(
    image: vx_image,
    rect: *const vx_rectangle_t,
    plane_index: vx_uint32,
    user_addr: *const vx_imagepatch_addressing_t,
    user_ptr: *mut c_void,
    usage: vx_enum,
    mem_type: vx_enum,
    _flags: vx_uint32,
) -> vx_status {
    // ... validation ...
    let img = unsafe { &mut *(image as *mut VxCImage) };
    
    // Direct data access - no uniform check
    if let Ok(data) = img.data.read() {
        for y in 0..height {
            let src_start = offset + y * stride_y;
            // ... copy ...
        }
    }
}
```

**Potential CTS Issues**:
1. **Uniform Image Optimization**: CTS may expect uniform images to return uniform value without actual memory read
2. **Virtual Image Handling**: May not properly handle virtual images without backing
3. **External Memory**: Has special handling for `is_external_memory` but may miss edge cases

### 6.2 Image Format Handling

**Format Support**:
- Planar formats (IYUV, NV12, NV21): Plane index validation and offset calculation
- Packed formats (YUYV, UYVY): Direct pixel access
- RGB/RGBA: Standard interleaved access

**Potential CTS Issues**:
1. **Plane Validation**: `plane_index` validation may reject valid indices for certain formats
2. **Region Clamping**: Width/height clamped to plane dimensions - CTS may expect error on oversized rects
3. **Stride Handling**: Uses calculated strides - CTS may expect specific stride patterns

---

## Summary of Critical CTS Breakage Risks

| Kernel | Risk Level | Key Issue |
|--------|------------|-----------|
| HOGCells | **HIGH** | INT16 data type requirement may conflict with CTS U8 expectations |
| HOGFeatures | **HIGH** | Q7.8 encoding and dimension ordering critical for CTS match |
| BilateralFilter | **MEDIUM** | S16 constant border handling broken (only reads U8 constant_value) |
| TensorConvertDepth | **HIGH** | Q7.8 hardcoded for INT16 - CTS may expect raw values |
| WeightedAverage | **LOW** | Kernel name and enum registration appears correct |
| MinMaxLoc | **MEDIUM** | minCount/maxCount as u32 may not match CTS vx_size expectation |
| CopyImagePatch | **MEDIUM** | No uniform image optimization - CTS may expect special handling |

---

## Recommendations for CTS Compliance

1. **HOG**: Verify CTS expects INT16 vs U8 for magnitudes/bins
2. **BilateralFilter**: Fix S16 constant border value reading
3. **TensorConvertDepth**: Add option for raw INT16 (non-Q7.8) handling
4. **MinMaxLoc**: Change count type to vx_size/usize instead of u32
5. **CopyImagePatch**: Add uniform image fast-path for CTS performance tests
6. **General**: Add comprehensive type validation for all scalar outputs