//! C API for OpenVX Data Objects (Scalar, Convolution, Matrix, LUT, Threshold, Pyramid)

use std::ffi::c_void;
use std::sync::RwLock;
use crate::c_api::*;

// ============================================================================
// Scalar Implementation
// ============================================================================

/// Scalar structure for C API
pub struct VxCScalarData {
    data_type: vx_enum,
    data: Vec<u8>,
    context: vx_context,
}

impl VxCScalarData {
    fn type_size(data_type: vx_enum) -> usize {
        match data_type {
            0x003 | 0x002 => 1, // VX_TYPE_UINT8 | VX_TYPE_INT8
            0x005 | 0x004 => 2, // VX_TYPE_UINT16 | VX_TYPE_INT16
            0x007 | 0x006 | 0x00A => 4, // VX_TYPE_UINT32 | VX_TYPE_INT32 | VX_TYPE_FLOAT32
            0x009 | 0x008 | 0x00B => 8, // VX_TYPE_UINT64 | VX_TYPE_INT64 | VX_TYPE_FLOAT64
            _ => 4,
        }
    }
}

/// Create a scalar
#[no_mangle]
pub extern "C" fn vxCreateScalar(
    context: vx_context,
    data_type: vx_enum,
    ptr: *const c_void,
) -> vx_scalar {
    if context.is_null() {
        return std::ptr::null_mut();
    }

    let type_size = VxCScalarData::type_size(data_type);
    let mut data = vec![0u8; type_size];

    if !ptr.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, data.as_mut_ptr(), type_size);
        }
    }

    let scalar = Box::new(VxCScalarData {
        data_type,
        data,
        context,
    });

    Box::into_raw(scalar) as vx_scalar
}

/// Query scalar attributes
#[no_mangle]
pub extern "C" fn vxQueryScalar(
    scalar: vx_scalar,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if scalar.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() || size == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let s = unsafe { &*(scalar as *const VxCScalarData) };

    unsafe {
        match attribute {
            VX_SCALAR_TYPE => {
                if size != std::mem::size_of::<vx_enum>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_enum) = s.data_type;
            }
            _ => return VX_ERROR_NOT_IMPLEMENTED,
        }
    }

    VX_SUCCESS
}

/// Release scalar
#[no_mangle]
pub extern "C" fn vxReleaseScalar(scalar: *mut vx_scalar) -> vx_status {
    if scalar.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*scalar).is_null() {
            let _ = Box::from_raw(*scalar as *mut VxCScalarData);
            *scalar = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

// ============================================================================
// Convolution Implementation
// ============================================================================

/// Convolution structure for C API
pub struct VxCConvolutionData {
    columns: vx_size,
    rows: vx_size,
    scale: vx_uint32,
    data: RwLock<Vec<i16>>,
    context: vx_context,
}

/// Create a convolution
#[no_mangle]
pub extern "C" fn vxCreateConvolution(
    context: vx_context,
    columns: vx_size,
    rows: vx_size,
) -> vx_convolution {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if columns == 0 || rows == 0 || columns > 9 || rows > 9 {
        return std::ptr::null_mut();
    }

    let size = columns * rows;
    let conv = Box::new(VxCConvolutionData {
        columns,
        rows,
        scale: 1,
        data: RwLock::new(vec![0i16; size]),
        context,
    });

    Box::into_raw(conv) as vx_convolution
}

/// Copy convolution coefficients
#[no_mangle]
pub extern "C" fn vxCopyConvolutionCoefficients(
    conv: vx_convolution,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if conv.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let c = unsafe { &*(conv as *const VxCConvolutionData) };
    let size = c.columns * c.rows;
    let data_size = size * std::mem::size_of::<i16>();

    unsafe {
        match usage {
            VX_READ_ONLY => {
                let data = c.data.read().unwrap();
                let src = data.as_ptr() as *const c_void;
                std::ptr::copy_nonoverlapping(src, user_ptr, data_size);
            }
            VX_WRITE_ONLY => {
                let mut data = c.data.write().unwrap();
                let dst = data.as_mut_ptr() as *mut c_void;
                std::ptr::copy_nonoverlapping(user_ptr, dst, data_size);
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Release convolution
#[no_mangle]
pub extern "C" fn vxReleaseConvolution(conv: *mut vx_convolution) -> vx_status {
    if conv.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*conv).is_null() {
            let _ = Box::from_raw(*conv as *mut VxCConvolutionData);
            *conv = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

// ============================================================================
// Matrix Implementation
// ============================================================================

/// Matrix structure for C API
pub struct VxCMatrixData {
    data_type: vx_enum,
    columns: vx_size,
    rows: vx_size,
    data: RwLock<Vec<u8>>,
    context: vx_context,
}

impl VxCMatrixData {
    fn element_size(data_type: vx_enum) -> usize {
        match data_type {
            0x003 | 0x002 => 1, // VX_TYPE_UINT8 | VX_TYPE_INT8
            0x005 | 0x004 => 2, // VX_TYPE_UINT16 | VX_TYPE_INT16
            0x007 | 0x006 | 0x00A => 4, // VX_TYPE_UINT32 | VX_TYPE_INT32 | VX_TYPE_FLOAT32
            0x009 | 0x008 | 0x00B => 8, // VX_TYPE_UINT64 | VX_TYPE_INT64 | VX_TYPE_FLOAT64
            _ => 4,
        }
    }
}

/// Create a matrix
#[no_mangle]
pub extern "C" fn vxCreateMatrix(
    context: vx_context,
    data_type: vx_enum,
    columns: vx_size,
    rows: vx_size,
) -> vx_matrix {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if columns == 0 || rows == 0 {
        return std::ptr::null_mut();
    }

    let elem_size = VxCMatrixData::element_size(data_type);
    let total_size = columns * rows * elem_size;

    let matrix = Box::new(VxCMatrixData {
        data_type,
        columns,
        rows,
        data: RwLock::new(vec![0u8; total_size]),
        context,
    });

    Box::into_raw(matrix) as vx_matrix
}

/// Copy matrix data
#[no_mangle]
pub extern "C" fn vxCopyMatrix(
    matrix: vx_matrix,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if matrix.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let m = unsafe { &*(matrix as *const VxCMatrixData) };
    let elem_size = VxCMatrixData::element_size(m.data_type);
    let data_size = m.columns * m.rows * elem_size;

    unsafe {
        match usage {
            VX_READ_ONLY => {
                let data = m.data.read().unwrap();
                let src = data.as_ptr() as *const c_void;
                std::ptr::copy_nonoverlapping(src, user_ptr, data_size);
            }
            VX_WRITE_ONLY => {
                let mut data = m.data.write().unwrap();
                let dst = data.as_mut_ptr() as *mut c_void;
                std::ptr::copy_nonoverlapping(user_ptr, dst, data_size);
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Release matrix
#[no_mangle]
pub extern "C" fn vxReleaseMatrix(matrix: *mut vx_matrix) -> vx_status {
    if matrix.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*matrix).is_null() {
            let _ = Box::from_raw(*matrix as *mut VxCMatrixData);
            *matrix = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

// ============================================================================
// LUT Implementation
// ============================================================================

/// LUT structure for C API
pub struct VxCLUTData {
    data_type: vx_enum,
    count: vx_size,
    offset: vx_int32,
    data: RwLock<Vec<u8>>,
    context: vx_context,
}

impl VxCLUTData {
    fn element_size(data_type: vx_enum) -> usize {
        match data_type {
            0x003 | 0x002 => 1, // VX_TYPE_UINT8 | VX_TYPE_INT8
            0x005 | 0x004 => 2, // VX_TYPE_UINT16 | VX_TYPE_INT16
            _ => 1,
        }
    }
}

/// Create a LUT
#[no_mangle]
pub extern "C" fn vxCreateLUT(
    context: vx_context,
    data_type: vx_enum,
    count: vx_size,
) -> vx_lut {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if count == 0 {
        return std::ptr::null_mut();
    }

    let elem_size = VxCLUTData::element_size(data_type);
    let total_size = count * elem_size;

    let lut = Box::new(VxCLUTData {
        data_type,
        count,
        offset: 0,
        data: RwLock::new(vec![0u8; total_size]),
        context,
    });

    Box::into_raw(lut) as vx_lut
}

/// Copy LUT data
#[no_mangle]
pub extern "C" fn vxCopyLUT(
    lut: vx_lut,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if lut.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let l = unsafe { &*(lut as *const VxCLUTData) };
    let elem_size = VxCLUTData::element_size(l.data_type);
    let data_size = l.count * elem_size;

    unsafe {
        match usage {
            VX_READ_ONLY => {
                let data = l.data.read().unwrap();
                let src = data.as_ptr() as *const c_void;
                std::ptr::copy_nonoverlapping(src, user_ptr, data_size);
            }
            VX_WRITE_ONLY => {
                let mut data = l.data.write().unwrap();
                let dst = data.as_mut_ptr() as *mut c_void;
                std::ptr::copy_nonoverlapping(user_ptr, dst, data_size);
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Release LUT
#[no_mangle]
pub extern "C" fn vxReleaseLUT(lut: *mut vx_lut) -> vx_status {
    if lut.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*lut).is_null() {
            let _ = Box::from_raw(*lut as *mut VxCLUTData);
            *lut = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

// ============================================================================
// Threshold Implementation
// ============================================================================

/// Threshold structure for C API
pub struct VxCThresholdData {
    thresh_type: vx_enum,
    data_type: vx_enum,
    value: vx_int32,
    lower: vx_int32,
    upper: vx_int32,
    true_value: vx_int32,
    false_value: vx_int32,
    context: vx_context,
}

/// Create a threshold
#[no_mangle]
pub extern "C" fn vxCreateThreshold(
    context: vx_context,
    thresh_type: vx_enum,
    data_type: vx_enum,
) -> vx_threshold {
    if context.is_null() {
        return std::ptr::null_mut();
    }

    let threshold = Box::new(VxCThresholdData {
        thresh_type,
        data_type,
        value: 127,
        lower: 0,
        upper: 255,
        true_value: 255,
        false_value: 0,
        context,
    });

    Box::into_raw(threshold) as vx_threshold
}

/// Set threshold attribute
#[no_mangle]
pub extern "C" fn vxSetThresholdAttribute(
    thresh: vx_threshold,
    attribute: vx_enum,
    ptr: *const c_void,
    size: vx_size,
) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() || size == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let t = unsafe { &mut *(thresh as *mut VxCThresholdData) };

    unsafe {
        match attribute {
            VX_THRESHOLD_VALUE => {
                if size != std::mem::size_of::<vx_int32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                t.value = *(ptr as *const vx_int32);
            }
            VX_THRESHOLD_LOWER => {
                if size != std::mem::size_of::<vx_int32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                t.lower = *(ptr as *const vx_int32);
            }
            VX_THRESHOLD_UPPER => {
                if size != std::mem::size_of::<vx_int32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                t.upper = *(ptr as *const vx_int32);
            }
            VX_THRESHOLD_TRUE_VALUE => {
                if size != std::mem::size_of::<vx_int32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                t.true_value = *(ptr as *const vx_int32);
            }
            VX_THRESHOLD_FALSE_VALUE => {
                if size != std::mem::size_of::<vx_int32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                t.false_value = *(ptr as *const vx_int32);
            }
            _ => return VX_ERROR_NOT_IMPLEMENTED,
        }
    }

    VX_SUCCESS
}

/// Release threshold
#[no_mangle]
pub extern "C" fn vxReleaseThreshold(thresh: *mut vx_threshold) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*thresh).is_null() {
            let _ = Box::from_raw(*thresh as *mut VxCThresholdData);
            *thresh = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

// ============================================================================
// Pyramid Implementation
// ============================================================================

/// Pyramid level structure
pub struct VxCPyramidLevel {
    width: vx_uint32,
    height: vx_uint32,
    data: Vec<u8>,
}

/// Pyramid structure for C API
pub struct VxCPyramidData {
    levels: vx_size,
    scale: vx_float32,
    format: vx_df_image,
    images: Vec<VxCPyramidLevel>,
    context: vx_context,
}

/// Create a pyramid
#[no_mangle]
pub extern "C" fn vxCreatePyramid(
    context: vx_context,
    levels: vx_size,
    scale: vx_float32,
    width: vx_uint32,
    height: vx_uint32,
    format: vx_df_image,
) -> vx_pyramid {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if levels == 0 || width == 0 || height == 0 {
        return std::ptr::null_mut();
    }

    let mut pyramid_images = Vec::with_capacity(levels);
    let mut current_width = width;
    let mut current_height = height;

    // Calculate bytes per pixel based on format
    let bpp = match format {
        VX_DF_IMAGE_U8 => 1,
        VX_DF_IMAGE_U16 | VX_DF_IMAGE_S16 => 2,
        VX_DF_IMAGE_RGB => 3,
        VX_DF_IMAGE_RGBA | VX_DF_IMAGE_RGBX => 4,
        _ => 1,
    };

    for _ in 0..levels {
        let size = (current_width as usize) * (current_height as usize) * bpp;
        pyramid_images.push(VxCPyramidLevel {
            width: current_width,
            height: current_height,
            data: vec![0u8; size],
        });

        // Calculate next level dimensions
        current_width = (current_width as f32 * scale) as vx_uint32;
        current_height = (current_height as f32 * scale) as vx_uint32;
        if current_width == 0 || current_height == 0 {
            break;
        }
    }

    let pyramid = Box::new(VxCPyramidData {
        levels: pyramid_images.len(),
        scale,
        format,
        images: pyramid_images,
        context,
    });

    Box::into_raw(pyramid) as vx_pyramid
}

/// Get pyramid level as image
#[no_mangle]
pub extern "C" fn vxGetPyramidLevel(
    pyr: vx_pyramid,
    index: vx_uint32,
) -> vx_image {
    if pyr.is_null() {
        return std::ptr::null_mut();
    }

    let p = unsafe { &*(pyr as *const VxCPyramidData) };
    if (index as usize) >= p.levels {
        return std::ptr::null_mut();
    }

    // Create a simple image wrapper for the pyramid level
    // In a full implementation, this would return a proper vx_image
    // For now, return a pointer that represents the level
    let level = &p.images[index as usize];
    let img_wrapper = Box::new(level.data.as_ptr() as usize);
    Box::into_raw(img_wrapper) as vx_image
}

/// Release pyramid
#[no_mangle]
pub extern "C" fn vxReleasePyramid(pyr: *mut vx_pyramid) -> vx_status {
    if pyr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*pyr).is_null() {
            let _ = Box::from_raw(*pyr as *mut VxCPyramidData);
            *pyr = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}
