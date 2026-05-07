//! C API for OpenVX Data Objects (Scalar, Convolution, Matrix, LUT, Threshold, Pyramid)

#![allow(non_camel_case_types)]
#![allow(dead_code, hidden_glob_reexports, non_snake_case, unused_unsafe)]

pub use crate::c_api::*;
use crate::unified_c_api::{REFERENCE_COUNTS, REFERENCE_TYPES};
use crate::unified_c_api::{VX_TYPE_CONVOLUTION, VX_TYPE_LUT, VX_TYPE_MATRIX, VX_TYPE_THRESHOLD};
use std::ffi::c_void;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::sync::RwLock;

// Pixel value union (needed for image operations)
// Match the C OpenVX definition with proper reserved padding
// The reserved array must overlap with the other fields for test compatibility
// In C, union members share the same starting address, so U8 overlaps with reserved[0]
#[repr(C)]
#[derive(Copy, Clone)]
pub union vx_pixel_value_t {
    /// Reserved array that overlaps with other fields (matches C definition)
    /// When reserved[0]=0x11, reserved[1]=0x22, etc., then:
    /// - U8 = reserved[0] = 0x11
    /// - U16 = reserved[0..1] = 0x2211 (little-endian)
    pub reserved: [u8; 16],
    /// VX_DF_IMAGE_RGB format in the R,G,B order
    pub RGB: [u8; 3],
    /// VX_DF_IMAGE_RGBX format in the R,G,B,X order  
    pub RGBX: [u8; 4],
    /// VX_DF_IMAGE_RGBA format in the R,G,B,A order
    pub RGBA: [u8; 4],
    /// All YUV formats in the Y,U,V order
    pub YUV: [u8; 3],
    /// VX_DF_IMAGE_U1
    pub U1: u8,
    /// VX_DF_IMAGE_U8 (overlaps with reserved[0])
    pub U8: u8,
    /// VX_DF_IMAGE_U16 (overlaps with reserved[0..1])
    pub U16: u16,
    /// VX_DF_IMAGE_S16 (overlaps with reserved[0..1])
    pub S16: i16,
    /// VX_DF_IMAGE_U32 (overlaps with reserved[0..3])
    pub U32: u32,
    /// VX_DF_IMAGE_S32 (overlaps with reserved[0..3])
    pub S32: i32,
}

// Implement Debug manually since unions don't derive Debug
impl std::fmt::Debug for vx_pixel_value_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            f.debug_struct("vx_pixel_value_t")
                .field("U32", &self.U32)
                .finish()
        }
    }
}

// ============================================================================
// Scalar Implementation
// ============================================================================

/// Scalar structure for C API
pub struct VxCScalarData {
    pub data_type: vx_enum,
    pub data: Vec<u8>,
    pub context: vx_context,
}

impl VxCScalarData {
    pub fn type_size(data_type: vx_enum) -> usize {
        match data_type {
            0x001 | 0x003 | 0x002 => 1, // VX_TYPE_CHAR | VX_TYPE_UINT8 | VX_TYPE_INT8
            0x005 | 0x004 => 2,         // VX_TYPE_UINT16 | VX_TYPE_INT16
            0x007 | 0x006 | 0x00A | 0x00C => 4, // VX_TYPE_UINT32 | VX_TYPE_INT32 | VX_TYPE_FLOAT32 | VX_TYPE_ENUM
            0x009 | 0x008 | 0x00B | 0x00D => 8, // VX_TYPE_UINT64 | VX_TYPE_INT64 | VX_TYPE_FLOAT64 | VX_TYPE_SIZE
            0x00E => 4,                         // VX_TYPE_DF_IMAGE
            0x010 => 4,                         // VX_TYPE_BOOL
            0x020 => 16,                        // VX_TYPE_RECTANGLE
            0x021 => 28,                        // VX_TYPE_KEYPOINT
            0x022 => 8,                         // VX_TYPE_COORDINATES2D
            0x023 => 12,                        // VX_TYPE_COORDINATES3D
            0x024 => 8,                         // VX_TYPE_COORDINATES2DF (2 floats)
            _ => {
                // Check user-registered struct sizes
                if data_type >= 0x100 {
                    if let Ok(structs) = crate::unified_c_api::USER_STRUCTS.lock() {
                        if let Some((_, size)) = structs.get(&data_type) {
                            return *size;
                        }
                    }
                }
                4
            }
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

    let scalar_ptr = Box::into_raw(scalar) as vx_scalar;

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(scalar_ptr as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(scalar_ptr as usize, VX_TYPE_SCALAR);
        }
    }

    scalar_ptr
}

/// Copy scalar data (works with VxCScalarData layout)
#[no_mangle]
pub extern "C" fn vxCopyScalarData(
    scalar: vx_scalar,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if scalar.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST && user_mem_type != 0x0 {
        // Also accept 0x0 for legacy compatibility
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    unsafe {
        let s = &*(scalar as *const VxCScalarData);
        match usage {
            0x11001 | 0x1 => {
                // VX_READ_ONLY
                std::ptr::copy_nonoverlapping(s.data.as_ptr(), user_ptr as *mut u8, s.data.len());
            }
            0x11002 | 0x2 => {
                // VX_WRITE_ONLY
                let s_mut = &mut *(scalar as *mut VxCScalarData);
                std::ptr::copy_nonoverlapping(
                    user_ptr as *const u8,
                    s_mut.data.as_mut_ptr(),
                    s_mut.data.len(),
                );
            }
            0x11003 | 0x3 => {
                // VX_READ_AND_WRITE
                std::ptr::copy_nonoverlapping(s.data.as_ptr(), user_ptr as *mut u8, s.data.len());
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
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

/// Create a virtual scalar (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualScalar(graph: vx_graph, data_type: vx_enum) -> vx_scalar {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    // Virtual scalars are created like regular scalars but associated with graph
    // In a full implementation, memory would be allocated during graph execution
    vxCreateScalar(graph as vx_context, data_type, std::ptr::null())
}

/// Release scalar
#[no_mangle]
pub extern "C" fn vxReleaseScalar(scalar: *mut vx_scalar) -> vx_status {
    if scalar.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*scalar).is_null() {
            let addr = *scalar as usize;

            // Check reference count before freeing
            let should_free = if let Ok(counts) = REFERENCE_COUNTS.lock() {
                if let Some(cnt) = counts.get(&addr) {
                    let current = cnt.load(std::sync::atomic::Ordering::SeqCst);
                    if current > 1 {
                        // Decrement and don't free
                        cnt.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        false
                    } else {
                        // Last reference - free it
                        true
                    }
                } else {
                    // Not in registry - just free
                    true
                }
            } else {
                false
            };

            if should_free {
                // Remove from reference counts and types
                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                    counts.remove(&addr);
                }
                if let Ok(mut types) = REFERENCE_TYPES.lock() {
                    types.remove(&addr);
                }

                let _ = Box::from_raw(*scalar as *mut VxCScalarData);
            }

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
    pub columns: vx_size,
    pub rows: vx_size,
    pub scale: vx_uint32,
    pub data: RwLock<Vec<i16>>,
    pub context: vx_context,
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

    let size = columns
        .checked_mul(rows)
        .and_then(|s| s.try_into().ok())
        .unwrap_or(0);
    if size == 0 {
        return std::ptr::null_mut();
    }
    let conv = Box::new(VxCConvolutionData {
        columns,
        rows,
        scale: 1,
        data: RwLock::new(vec![0i16; size]),
        context,
    });

    let conv_ptr = Box::into_raw(conv) as vx_convolution;

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(conv_ptr as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(conv_ptr as usize, VX_TYPE_CONVOLUTION);
        }
    }

    conv_ptr
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
    let size = c
        .columns
        .checked_mul(c.rows)
        .and_then(|s| s.checked_mul(std::mem::size_of::<i16>()))
        .unwrap_or(0);
    if size == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    let data_size = size;

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

/// Create a virtual convolution (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualConvolution(
    graph: vx_graph,
    columns: vx_size,
    rows: vx_size,
) -> vx_convolution {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    // Virtual convolutions are created like regular ones but associated with graph
    vxCreateConvolution(graph as vx_context, columns, rows)
}

/// Release convolution
#[no_mangle]
pub extern "C" fn vxReleaseConvolution(conv: *mut vx_convolution) -> vx_status {
    if conv.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*conv).is_null() {
            let addr = *conv as usize;

            // Remove from reference counts and types
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.remove(&addr);
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.remove(&addr);
            }

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
    pub data_type: vx_enum,
    pub columns: vx_size,
    pub rows: vx_size,
    pub data: RwLock<Vec<u8>>,
    pub context: vx_context,
    /// Matrix pattern type (VX_PATTERN_BOX, VX_PATTERN_CROSS, etc.)
    /// Stored as 0 for non-pattern matrices (VX_PATTERN_OTHER)
    pub pattern: vx_enum,
    /// Origin x coordinate (column offset of the center)
    pub origin_x: vx_size,
    /// Origin y coordinate (row offset of the center)
    pub origin_y: vx_size,
}

impl VxCMatrixData {
    pub fn element_size(data_type: vx_enum) -> usize {
        match data_type {
            0x003 | 0x002 => 1,         // VX_TYPE_UINT8 | VX_TYPE_INT8
            0x005 | 0x004 => 2,         // VX_TYPE_UINT16 | VX_TYPE_INT16
            0x007 | 0x006 | 0x00A => 4, // VX_TYPE_UINT32 | VX_TYPE_INT32 | VX_TYPE_FLOAT32
            0x009 | 0x008 | 0x00B => 8, // VX_TYPE_UINT64 | VX_TYPE_INT64 | VX_TYPE_FLOAT64
            _ => 4,
        }
    }

    /// Get matrix data as a slice of f32 values
    pub fn as_f32_slice(&self) -> Option<Vec<f32>> {
        if self.data_type != VX_TYPE_FLOAT32 {
            return None;
        }
        let data = self.data.read().ok()?;
        let len = data.len() / 4;
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let bytes = [
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ];
            result.push(f32::from_le_bytes(bytes));
        }
        Some(result)
    }

    /// Create a reference from a raw pointer
    pub fn from_ptr(ptr: vx_matrix) -> Option<&'static VxCMatrixData> {
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { &*(ptr as *const VxCMatrixData) })
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
    let total_size = columns
        .checked_mul(rows)
        .and_then(|s| s.checked_mul(elem_size))
        .and_then(|s| s.try_into().ok())
        .unwrap_or(0);
    if total_size == 0 && (columns > 0 && rows > 0) {
        return std::ptr::null_mut();
    }

    let matrix = Box::new(VxCMatrixData {
        data_type,
        columns,
        rows,
        data: RwLock::new(vec![0u8; total_size]),
        context,
        pattern: 0, // VX_PATTERN_OTHER for manually created matrices
        origin_x: columns / 2,
        origin_y: rows / 2,
    });

    let matrix_ptr = Box::into_raw(matrix) as vx_matrix;

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(matrix_ptr as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(matrix_ptr as usize, VX_TYPE_MATRIX);
        }
    }

    matrix_ptr
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
    let data_size = m
        .columns
        .checked_mul(m.rows)
        .and_then(|s| s.checked_mul(elem_size))
        .unwrap_or(0);
    if data_size == 0 && (m.columns > 0 && m.rows > 0) {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    unsafe {
        match usage {
            VX_READ_ONLY => {
                let data = match m.data.read() {
                    Ok(d) => d,
                    Err(_) => return VX_ERROR_INVALID_REFERENCE,
                };
                let src = data.as_ptr() as *const c_void;
                std::ptr::copy_nonoverlapping(src, user_ptr, data_size);
            }
            VX_WRITE_ONLY => {
                let mut data = match m.data.write() {
                    Ok(d) => d,
                    Err(_) => return VX_ERROR_INVALID_REFERENCE,
                };
                let dst = data.as_mut_ptr() as *mut c_void;
                std::ptr::copy_nonoverlapping(user_ptr, dst, data_size);
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Create a virtual matrix (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualMatrix(
    graph: vx_graph,
    data_type: vx_enum,
    columns: vx_size,
    rows: vx_size,
) -> vx_matrix {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    // Virtual matrices are created like regular ones but associated with graph
    vxCreateMatrix(graph as vx_context, data_type, columns, rows)
}

/// Release matrix
#[no_mangle]
pub extern "C" fn vxReleaseMatrix(matrix: *mut vx_matrix) -> vx_status {
    if matrix.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*matrix).is_null() {
            let addr = *matrix as usize;

            // Remove from reference counts and types
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.remove(&addr);
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.remove(&addr);
            }

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
    pub data_type: vx_enum,
    pub count: vx_size,
    pub offset: vx_int32,
    pub data: RwLock<Vec<u8>>,
    pub context: vx_context,
    /// Structure for tracking mapped LUTs: (map_id, mapped_data, usage)
    pub mapped_luts: Arc<RwLock<Vec<(usize, Vec<u8>, vx_enum)>>>,
}

impl VxCLUTData {
    pub fn element_size(data_type: vx_enum) -> usize {
        match data_type {
            0x003 | 0x002 => 1, // VX_TYPE_UINT8 | VX_TYPE_INT8
            0x005 | 0x004 => 2, // VX_TYPE_UINT16 | VX_TYPE_INT16
            _ => 1,
        }
    }
}

/// Create a LUT
#[no_mangle]
pub extern "C" fn vxCreateLUT(context: vx_context, data_type: vx_enum, count: vx_size) -> vx_lut {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if count == 0 {
        return std::ptr::null_mut();
    }

    let elem_size = VxCLUTData::element_size(data_type);
    let total_size = count
        .checked_mul(elem_size)
        .and_then(|s| s.try_into().ok())
        .unwrap_or(0);
    if total_size == 0 && count > 0 {
        return std::ptr::null_mut();
    }

    let lut = Box::new(VxCLUTData {
        data_type,
        count,
        offset: if data_type == 0x004 {
            (count / 2) as vx_int32
        } else {
            0
        },
        data: RwLock::new(vec![0u8; total_size]),
        context,
        mapped_luts: Arc::new(RwLock::new(Vec::new())),
    });

    let lut_ptr = Box::into_raw(lut) as vx_lut;

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(lut_ptr as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(lut_ptr as usize, VX_TYPE_LUT);
        }
    }

    lut_ptr
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
    let data_size = l.count.checked_mul(elem_size).unwrap_or(0);
    if data_size == 0 && l.count > 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

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

/// Create a virtual LUT (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualLUT(
    graph: vx_graph,
    data_type: vx_enum,
    count: vx_size,
) -> vx_lut {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    // Virtual LUTs are created like regular ones but associated with graph
    vxCreateLUT(graph as vx_context, data_type, count)
}

/// Release LUT
#[no_mangle]
pub extern "C" fn vxReleaseLUT(lut: *mut vx_lut) -> vx_status {
    if lut.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*lut).is_null() {
            let addr = *lut as usize;

            // Honor reference counting: only the *last* release frees the
            // underlying VxCLUTData. Object arrays, retained handles from
            // `vxGetObjectArrayItem`, etc. all bump REFERENCE_COUNTS, and
            // releasing the handle prematurely (as the previous version did)
            // turned every other holder's pointer into use-after-free memory.
            let should_free = if let Ok(counts) = REFERENCE_COUNTS.lock() {
                if let Some(cnt) = counts.get(&addr) {
                    let current = cnt.load(std::sync::atomic::Ordering::SeqCst);
                    if current > 1 {
                        cnt.store(
                            current - 1,
                            std::sync::atomic::Ordering::SeqCst,
                        );
                        false
                    } else {
                        true
                    }
                } else {
                    // No tracking entry — treat as last reference.
                    true
                }
            } else {
                false
            };

            if should_free {
                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                    counts.remove(&addr);
                }
                if let Ok(mut types) = REFERENCE_TYPES.lock() {
                    types.remove(&addr);
                }
                let _ = Box::from_raw(*lut as *mut VxCLUTData);
            }
            *lut = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

/// Query LUT attributes
#[no_mangle]
pub extern "C" fn vxQueryLUT(lut: vx_lut, attribute: i32, ptr: *mut c_void, size: usize) -> i32 {
    if lut.is_null() || ptr.is_null() {
        return -2;
    }

    unsafe {
        let l = &*(lut as *const VxCLUTData);
        match attribute {
            0x80700 => {
                // VX_LUT_TYPE
                if size >= std::mem::size_of::<vx_enum>() {
                    *(ptr as *mut vx_enum) = l.data_type;
                    return 0;
                }
            }
            0x80701 => {
                // VX_LUT_COUNT
                if size >= std::mem::size_of::<vx_size>() {
                    *(ptr as *mut vx_size) = l.count;
                    return 0;
                }
            }
            0x80702 => {
                // VX_LUT_SIZE
                if size >= std::mem::size_of::<vx_size>() {
                    let elem_size = VxCLUTData::element_size(l.data_type);
                    *(ptr as *mut vx_size) = l.count * elem_size;
                    return 0;
                }
            }
            0x80703 => {
                // VX_LUT_OFFSET
                if size >= std::mem::size_of::<vx_int32>() {
                    *(ptr as *mut vx_int32) = l.offset;
                    return 0;
                }
            }
            _ => {}
        }
    }

    -30
}

/// Map LUT for CPU access
#[no_mangle]
pub extern "C" fn vxMapLUT(
    lut: vx_lut,
    map_id: *mut vx_map_id,
    ptr: *mut *mut c_void,
    usage: vx_enum,
    mem_type: vx_enum,
    _flags: vx_uint32,
) -> vx_status {
    if lut.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if map_id.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let l = unsafe { &mut *(lut as *mut VxCLUTData) };

    unsafe {
        // Get LUT data
        let data_guard = match l.data.read() {
            Ok(guard) => guard,
            Err(_) => return VX_ERROR_INVALID_REFERENCE,
        };

        // Create a copy of the data for the mapped LUT
        let mapped_data = data_guard.clone();

        // Store the mapped data
        let map_id_val = if let Ok(mut mappings) = l.mapped_luts.write() {
            let id = mappings.len() + 1;
            mappings.push((id, mapped_data, usage));
            id
        } else {
            return VX_ERROR_INVALID_REFERENCE;
        };

        // Set output parameters
        *map_id = map_id_val;

        // Return pointer to the STORED mapped data
        if let Ok(mappings) = l.mapped_luts.read() {
            if let Some(mapping) = mappings.iter().find(|(id, _, _)| *id == map_id_val) {
                *ptr = mapping.1.as_ptr() as *mut c_void;
            }
        }

        drop(data_guard);
    }

    VX_SUCCESS
}

/// Unmap LUT
#[no_mangle]
pub extern "C" fn vxUnmapLUT(lut: vx_lut, map_id: vx_map_id) -> vx_status {
    if lut.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let l = unsafe { &mut *(lut as *mut VxCLUTData) };

    if let Ok(mut mappings) = l.mapped_luts.write() {
        if let Some(pos) = mappings.iter().position(|(id, _, _)| *id == map_id) {
            let (_, mapped_data, usage) = mappings.remove(pos);

            // If write access, copy data back
            if usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE {
                if let Ok(mut data) = l.data.write() {
                    data.copy_from_slice(&mapped_data);
                }
            }

            return VX_SUCCESS;
        }
    }

    VX_ERROR_INVALID_REFERENCE
}

// ============================================================================
// Threshold Implementation
// ============================================================================

/// Threshold structure for C API
pub struct VxCThresholdData {
    pub thresh_type: vx_enum,
    pub data_type: vx_enum,
    pub value: vx_int32,
    pub lower: vx_int32,
    pub upper: vx_int32,
    pub true_value: vx_int32,
    pub false_value: vx_int32,
    pub input_format: vx_df_image,
    pub output_format: vx_df_image,
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
        input_format: VX_DF_IMAGE_U8,
        output_format: VX_DF_IMAGE_U8,
        context,
    });

    let thresh_ptr = Box::into_raw(threshold) as vx_threshold;

    // Register in unified THRESHOLDS registry for type tracking
    crate::unified_c_api::register_threshold(thresh_ptr as usize);

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(thresh_ptr as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(thresh_ptr as usize, VX_TYPE_THRESHOLD);
        }
    }

    thresh_ptr
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

/// Create a virtual threshold for image (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualThresholdForImage(
    graph: vx_graph,
    thresh_type: vx_enum,
    _input_format: vx_df_image,
    output_format: vx_df_image,
) -> vx_threshold {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    // Virtual thresholds are created like regular ones
    // The input/output format parameters are stored for validation
    vxCreateThreshold(graph as vx_context, thresh_type, output_format as vx_enum)
}

/// Release threshold
#[no_mangle]
pub extern "C" fn vxReleaseThreshold(thresh: *mut vx_threshold) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*thresh).is_null() {
            let addr = *thresh as usize;

            // Unregister from unified THRESHOLDS registry
            crate::unified_c_api::unregister_threshold(addr);

            // Remove from reference counts and types
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.remove(&addr);
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.remove(&addr);
            }

            let _ = Box::from_raw(*thresh as *mut VxCThresholdData);
            *thresh = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

// ============================================================================
// Extended Threshold API
// ============================================================================

// Additional threshold attributes
// VX_ATTRIBUTE_BASE(VX_ID_KHRONOS=0, VX_TYPE_THRESHOLD=0x80A) = 0x80A00
// VX_THRESHOLD_INPUT_FORMAT = 0x80A00 + 7 = 0x80A07
// VX_THRESHOLD_OUTPUT_FORMAT = 0x80A00 + 8 = 0x80A08
pub const VX_THRESHOLD_TYPE: vx_enum = 0x80A00;
pub const VX_THRESHOLD_DATA_TYPE: vx_enum = 0x80A06;
pub const VX_THRESHOLD_INPUT_FORMAT: vx_enum = 0x80A07;
pub const VX_THRESHOLD_OUTPUT_FORMAT: vx_enum = 0x80A08;

// Threshold types
// VX_ENUM_BASE(VX_ID_KHRONOS=0, VX_ENUM_THRESHOLD_TYPE=0x0B) = 0x0B000
// VX_THRESHOLD_TYPE_BINARY = 0x0B000 + 0 = 0x0B000 (45056)
// VX_THRESHOLD_TYPE_RANGE = 0x0B000 + 1 = 0x0B001 (45057)
pub const VX_THRESHOLD_TYPE_BINARY: vx_enum = 0x0B000;
pub const VX_THRESHOLD_TYPE_RANGE: vx_enum = 0x0B001;

/// Create a threshold for image
#[no_mangle]
pub extern "C" fn vxCreateThresholdForImage(
    context: vx_context,
    thresh_type: vx_enum,
    input_format: vx_df_image,
    output_format: vx_df_image,
) -> vx_threshold {
    if context.is_null() {
        return std::ptr::null_mut();
    }

    let thresh = vxCreateThreshold(context, thresh_type, 0);
    if !thresh.is_null() {
        let t = unsafe { &mut *(thresh as *mut VxCThresholdData) };
        t.input_format = input_format;
        t.output_format = output_format;
    }
    thresh
}

/// Query threshold attributes
#[no_mangle]
pub extern "C" fn vxQueryThresholdData(
    thresh: vx_threshold,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let t = unsafe { &*(thresh as *const VxCThresholdData) };

    unsafe {
        match attribute {
            VX_THRESHOLD_TYPE => {
                if size != std::mem::size_of::<vx_enum>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_enum) = t.thresh_type;
            }
            VX_THRESHOLD_DATA_TYPE => {
                if size != std::mem::size_of::<vx_enum>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_enum) = t.data_type;
            }
            VX_THRESHOLD_INPUT_FORMAT => {
                if size != std::mem::size_of::<vx_df_image>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_df_image) = t.input_format;
            }
            VX_THRESHOLD_OUTPUT_FORMAT => {
                if size != std::mem::size_of::<vx_df_image>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_df_image) = t.output_format;
            }
            _ => return VX_ERROR_NOT_IMPLEMENTED,
        }
    }

    VX_SUCCESS
}

/// Copy threshold value (for binary threshold)
#[no_mangle]
pub extern "C" fn vxCopyThresholdValue(
    thresh: vx_threshold,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let t = unsafe { &mut *(thresh as *mut VxCThresholdData) };

    unsafe {
        match usage {
            VX_READ_ONLY => {
                // Copy value to user
                let val = t.value;
                std::ptr::write(user_ptr as *mut vx_int32, val);
            }
            VX_WRITE_ONLY => {
                // Copy from user
                t.value = *(user_ptr as *const vx_int32);
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Copy threshold range (for range threshold)
#[no_mangle]
pub extern "C" fn vxCopyThresholdRange(
    thresh: vx_threshold,
    lower: *mut c_void,
    upper: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if lower.is_null() || upper.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let t = unsafe { &mut *(thresh as *mut VxCThresholdData) };

    unsafe {
        match usage {
            VX_READ_ONLY => {
                // Copy range to user
                std::ptr::write(lower as *mut vx_int32, t.lower);
                std::ptr::write(upper as *mut vx_int32, t.upper);
            }
            VX_WRITE_ONLY => {
                // Copy from user
                t.lower = *(lower as *const vx_int32);
                t.upper = *(upper as *const vx_int32);
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Copy threshold output values
#[no_mangle]
pub extern "C" fn vxCopyThresholdOutput(
    thresh: vx_threshold,
    true_value: *mut c_void,
    false_value: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if true_value.is_null() || false_value.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let t = unsafe { &mut *(thresh as *mut VxCThresholdData) };

    unsafe {
        match usage {
            VX_READ_ONLY => {
                // Copy output values to user
                std::ptr::write(true_value as *mut vx_int32, t.true_value);
                std::ptr::write(false_value as *mut vx_int32, t.false_value);
            }
            VX_WRITE_ONLY => {
                // Copy from user
                t.true_value = *(true_value as *const vx_int32);
                t.false_value = *(false_value as *const vx_int32);
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Query threshold (wrapper to match unified_c_api)
#[no_mangle]
pub extern "C" fn vxQueryThreshold(
    thresh: vx_threshold,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    vxQueryThresholdData(thresh, attribute, ptr, size)
}

/// Copy threshold (wrapper for unified API compatibility)
#[no_mangle]
pub extern "C" fn vxCopyThreshold(
    thresh: vx_threshold,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if thresh.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let t = unsafe { &*(thresh as *const VxCThresholdData) };

    unsafe {
        match usage {
            VX_READ_ONLY => {
                // Copy all threshold values
                std::ptr::write(user_ptr as *mut vx_int32, t.value);
            }
            VX_WRITE_ONLY => {
                // Write value
                let val = *(user_ptr as *const vx_int32);
                let t = unsafe { &mut *(thresh as *mut VxCThresholdData) };
                t.value = val;
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

// Pyramid functions (vxCreatePyramid, vxGetPyramidLevel, vxReleasePyramid, vxCreateVirtualPyramid)
// are implemented in the openvx-image crate
