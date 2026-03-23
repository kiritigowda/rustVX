//! C API for OpenVX Image

use std::ffi::c_void;
use std::sync::{RwLock, Arc};
use std::sync::atomic::Ordering;
use openvx_core::unified_c_api::{register_image, unregister_image};
use openvx_core::c_api::{
    vx_context, vx_graph, vx_image, vx_status, vx_enum, vx_size, vx_uint32,
    vx_rectangle_t, vx_imagepatch_addressing_t, vx_map_id, vx_df_image,
    VX_SUCCESS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_INVALID_PARAMETERS,
    VX_ERROR_NOT_IMPLEMENTED,
    VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGBA, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_NV12,
    VX_DF_IMAGE_NV21, VX_DF_IMAGE_UYVY, VX_DF_IMAGE_YUYV, VX_DF_IMAGE_IYUV,
    VX_DF_IMAGE_YUV4, VX_DF_IMAGE_U8, VX_DF_IMAGE_U16, VX_DF_IMAGE_S16,
    VX_DF_IMAGE_U32, VX_DF_IMAGE_S32, VX_DF_IMAGE_VIRT,
    VX_IMAGE_FORMAT, VX_IMAGE_WIDTH, VX_IMAGE_HEIGHT, VX_IMAGE_PLANES,
    VX_IMAGE_IS_UNIFORM, VX_IMAGE_UNIFORM_VALUE,
    VX_READ_ONLY, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST,
};

// Global image registry
static IMAGE_ID_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// Image struct for C API
#[derive(Debug, Clone)]
pub struct VxCImage {
    width: vx_uint32,
    height: vx_uint32,
    format: vx_df_image,
    is_virtual: bool,
    context: vx_context,
    data: Arc<RwLock<Vec<u8>>>,
    mapped_patches: Arc<RwLock<Vec<(vx_map_id, Vec<u8>)>>>,
}

impl VxCImage {
    fn bytes_per_pixel(format: vx_df_image) -> usize {
        match format {
            VX_DF_IMAGE_U8 => 1,
            VX_DF_IMAGE_U16 | VX_DF_IMAGE_S16 => 2,
            VX_DF_IMAGE_U32 | VX_DF_IMAGE_S32 | VX_DF_IMAGE_RGB => 4,
            VX_DF_IMAGE_RGBA | VX_DF_IMAGE_RGBX => 4,
            VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => 1, // Luma only
            VX_DF_IMAGE_IYUV => 1,
            VX_DF_IMAGE_UYVY | VX_DF_IMAGE_YUYV => 2,
            VX_DF_IMAGE_YUV4 => 3,
            _ => 1,
        }
    }

    fn channels(format: vx_df_image) -> usize {
        match format {
            VX_DF_IMAGE_U8 => 1,
            VX_DF_IMAGE_U16 | VX_DF_IMAGE_S16 => 1,
            VX_DF_IMAGE_U32 | VX_DF_IMAGE_S32 => 1,
            VX_DF_IMAGE_RGB => 3,
            VX_DF_IMAGE_RGBA | VX_DF_IMAGE_RGBX => 4,
            VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => 3,
            VX_DF_IMAGE_IYUV => 3,
            VX_DF_IMAGE_UYVY | VX_DF_IMAGE_YUYV => 2,
            VX_DF_IMAGE_YUV4 => 3,
            _ => 1,
        }
    }

    fn calculate_size(width: vx_uint32, height: vx_uint32, format: vx_df_image) -> usize {
        // Validate dimensions to prevent overflow
        if width == 0 || height == 0 {
            return 0;
        }
        
        // Check for potential overflow
        let w = width as usize;
        let h = height as usize;
        let channels = Self::channels(format);
        
        // Limit maximum allocation to ~1GB (sanity check)
        let max_size = 1024 * 1024 * 1024; // 1GB
        let size = w.saturating_mul(h).saturating_mul(channels);
        
        if size > max_size {
            eprintln!("Image size {}x{}x{} = {} exceeds maximum allocation limit", w, h, channels, size);
            return 0; // Return 0 to trigger null image creation
        }
        
        size
    }
}

/// Create an image
#[no_mangle]
pub extern "C" fn vxCreateImage(
    context: vx_context,
    width: vx_uint32,
    height: vx_uint32,
    color: vx_df_image,
) -> vx_image {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if width == 0 || height == 0 {
        return std::ptr::null_mut();
    }

    let size = VxCImage::calculate_size(width, height, color);
    if size == 0 {
        return std::ptr::null_mut(); // Invalid dimensions or overflow
    }
    let data = vec![0u8; size];

    let image = Box::new(VxCImage {
        width,
        height,
        format: color,
        is_virtual: false,
        context,
        data: Arc::new(RwLock::new(data)),
        mapped_patches: Arc::new(RwLock::new(Vec::new())),
    });

    let image_ptr = Box::into_raw(image) as vx_image;
    
    // Register image address in unified registry for type queries (vxQueryReference)
    register_image(image_ptr as usize);
    
    image_ptr
}

/// Create a virtual image (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualImage(
    graph: vx_graph,
    width: vx_uint32,
    height: vx_uint32,
    color: vx_df_image,
) -> vx_image {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    if width == 0 || height == 0 {
        return std::ptr::null_mut();
    }

    // Virtual images don't allocate memory immediately
    let image = Box::new(VxCImage {
        width,
        height,
        format: color,
        is_virtual: true,
        context: std::ptr::null_mut(), // Virtual images use graph context
        data: Arc::new(RwLock::new(Vec::new())),
        mapped_patches: Arc::new(RwLock::new(Vec::new())),
    });

    let image_ptr = Box::into_raw(image) as vx_image;
    
    // Register image address in unified registry for type queries (vxQueryReference)
    register_image(image_ptr as usize);
    
    image_ptr
}

/// Create an image from existing handles
#[no_mangle]
pub extern "C" fn vxCreateImageFromHandle(
    context: vx_context,
    color: vx_df_image,
    addrs: *const vx_imagepatch_addressing_t,
    ptrs: *mut *mut c_void,
    num_planes: vx_uint32,
) -> vx_image {
    if context.is_null() || addrs.is_null() || ptrs.is_null() {
        return std::ptr::null_mut();
    }

    if num_planes == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let addr = &*addrs;
        let width = addr.dim_x;
        let height = addr.dim_y;

        // Calculate size from addressing with overflow protection
        let total_size = if addr.stride_y > 0 {
            (height as usize).checked_mul(addr.stride_y as usize)
                .and_then(|s| if s > 0 { Some(s) } else { None })
        } else {
            (width as usize)
                .checked_mul(height as usize)
                .and_then(|s| s.checked_mul(VxCImage::bytes_per_pixel(color)))
                .and_then(|s| if s > 0 { Some(s) } else { None })
        };
        let Some(total_size) = total_size else {
            return std::ptr::null_mut();
        };

        // For now, we copy the data. A full implementation might keep references
        let data = vec![0u8; total_size];

        let image = Box::new(VxCImage {
            width,
            height,
            format: color,
            is_virtual: false,
            context,
            data: Arc::new(RwLock::new(data)),
            mapped_patches: Arc::new(RwLock::new(Vec::new())),
        });

        let image_ptr = Box::into_raw(image) as vx_image;
        
        // Register image address in unified registry for type queries (vxQueryReference)
        register_image(image_ptr as usize);
        
        image_ptr
    }
}

/// Release an image
#[no_mangle]
pub extern "C" fn vxReleaseImage(image: *mut vx_image) -> vx_status {
    if image.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*image).is_null() {
            // Unregister from unified registry
            unregister_image(*image as usize);
            let _ = Box::from_raw(*image as *mut VxCImage);
            *image = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

/// Query image attributes
#[no_mangle]
pub extern "C" fn vxQueryImage(
    image: vx_image,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if image.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if size == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let img = unsafe { &*(image as *const VxCImage) };

    unsafe {
        match attribute {
            VX_IMAGE_FORMAT => {
                if size != std::mem::size_of::<vx_df_image>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_df_image) = img.format;
            }
            VX_IMAGE_WIDTH => {
                if size != std::mem::size_of::<vx_uint32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_uint32) = img.width;
            }
            VX_IMAGE_HEIGHT => {
                if size != std::mem::size_of::<vx_uint32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_uint32) = img.height;
            }
            VX_IMAGE_PLANES => {
                if size != std::mem::size_of::<vx_size>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                let planes = match img.format {
                    VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 | VX_DF_IMAGE_IYUV => 2,
                    VX_DF_IMAGE_YUV4 => 3,
                    _ => 1,
                };
                *(ptr as *mut vx_size) = planes;
            }
            VX_IMAGE_IS_UNIFORM => {
                if size != std::mem::size_of::<vx_bool>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                // Currently not supporting uniform images - always return false (0)
                *(ptr as *mut vx_bool) = 0;
            }
            VX_IMAGE_UNIFORM_VALUE => {
                return VX_ERROR_NOT_IMPLEMENTED;
            }
            _ => return VX_ERROR_NOT_IMPLEMENTED,
        }
    }

    VX_SUCCESS
}

/// vx_bool type alias
pub type vx_bool = i32;

/// Set image attributes
#[no_mangle]
pub extern "C" fn vxSetImageAttribute(
    _image: vx_image,
    _attribute: vx_enum,
    _ptr: *const c_void,
    _size: vx_size,
) -> vx_status {
    // Most image attributes are read-only
    VX_ERROR_NOT_IMPLEMENTED
}

/// Map an image patch for CPU access
#[no_mangle]
pub extern "C" fn vxMapImagePatch(
    image: vx_image,
    rect: *const vx_rectangle_t,
    plane_index: vx_uint32,
    map_id: *mut vx_map_id,
    addr: *mut vx_imagepatch_addressing_t,
    ptr: *mut *mut c_void,
    usage: vx_enum,
    mem_type: vx_enum,
    _flags: vx_uint32,
) -> vx_status {
    if image.is_null() || rect.is_null() || map_id.is_null() || addr.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    if mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    if usage != VX_READ_ONLY && usage != VX_WRITE_ONLY {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let img = unsafe { &*(image as *const VxCImage) };
    let rect = unsafe { &*rect };

    // Validate rectangle
    if rect.start_x >= img.width || rect.start_y >= img.height {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if rect.end_x > img.width || rect.end_y > img.height {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if rect.end_x <= rect.start_x || rect.end_y <= rect.start_y {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let bpp = VxCImage::bytes_per_pixel(img.format);
    let stride_x = bpp as i32;
    let stride_y = (img.width as i32).checked_mul(bpp as i32)
        .and_then(|s| if s > 0 { Some(s) } else { None })
        .unwrap_or(0);
    
    // If stride_y is 0 due to overflow or invalid dimensions, return error
    if stride_y == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    // Set addressing structure
    unsafe {
        (*addr).dim_x = rect.end_x - rect.start_x;
        (*addr).dim_y = rect.end_y - rect.start_y;
        (*addr).stride_x = stride_x;
        (*addr).stride_y = stride_y;
        (*addr).scale_x = 0;
        (*addr).scale_y = 0;
        (*addr).step_x = 1;
        (*addr).step_y = 1;
    }

    // Get data pointer with overflow-safe calculation
    let data = img.data.read().unwrap();
    
    // Calculate offset with proper overflow checking
    let start_x_offset = (rect.start_x as usize).checked_mul(bpp);
    let row_offset = (rect.start_y as usize)
        .checked_mul(img.width as usize)
        .and_then(|s| s.checked_mul(bpp));
    
    // If any multiplication overflowed, return error
    let (Some(row_off), Some(col_off)) = (row_offset, start_x_offset) else {
        return VX_ERROR_INVALID_PARAMETERS;
    };
    
    let Some(offset) = row_off.checked_add(col_off) else {
        return VX_ERROR_INVALID_PARAMETERS;
    };

    if offset >= data.len() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let patch_ptr = unsafe {
        data.as_ptr().add(offset) as *mut c_void
    };

    unsafe {
        *ptr = patch_ptr;
    }

    // Generate map ID
    let id = IMAGE_ID_COUNTER.fetch_add(1, Ordering::SeqCst) as vx_map_id;
    unsafe {
        *map_id = id;
    }

    VX_SUCCESS
}

/// Unmap an image patch
#[no_mangle]
pub extern "C" fn vxUnmapImagePatch(
    image: vx_image,
    _map_id: vx_map_id,
) -> vx_status {
    if image.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // In this simple implementation, we just validate the image exists
    // Full implementation would use the map_id to validate and cleanup
    VX_SUCCESS
}
