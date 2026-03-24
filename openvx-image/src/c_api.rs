//! C API for OpenVX Image

use std::ffi::c_void;
use std::sync::{RwLock, Arc};
use openvx_core::unified_c_api::{register_image, unregister_image};
use openvx_core::unified_c_api::VxCImage;
use openvx_core::c_api::{
    vx_context, vx_graph, vx_image, vx_status, vx_enum, vx_size, vx_uint32,
    vx_rectangle_t, vx_imagepatch_addressing_t, vx_map_id, vx_df_image, vx_int32,
    VX_SUCCESS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_INVALID_PARAMETERS,
    VX_ERROR_NOT_IMPLEMENTED,
    VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGBA, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_NV12,
    VX_DF_IMAGE_NV21, VX_DF_IMAGE_IYUV, VX_DF_IMAGE_UYVY, VX_DF_IMAGE_YUYV,
    VX_DF_IMAGE_YUV4, VX_DF_IMAGE_U8, VX_DF_IMAGE_U16, VX_DF_IMAGE_S16,
    VX_DF_IMAGE_U32, VX_DF_IMAGE_S32,
    VX_IMAGE_FORMAT, VX_IMAGE_WIDTH, VX_IMAGE_HEIGHT, VX_IMAGE_PLANES,
    VX_IMAGE_IS_UNIFORM, VX_IMAGE_UNIFORM_VALUE,
    VX_READ_ONLY, VX_WRITE_ONLY, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST,
};

// Re-export pixel value type and keypoint type
pub use openvx_core::c_api_data::vx_pixel_value_t;
pub use openvx_core::unified_c_api::vx_keypoint_t;

// Global image registry
static IMAGE_ID_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

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
        parent: None,
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
    // Note: Virtual images CAN have width/height of 0 - they get dimensions
    // from connected nodes during graph verification

    // Virtual images don't allocate memory immediately
    let image = Box::new(VxCImage {
        width,
        height,
        format: color,
        is_virtual: true,
        context: std::ptr::null_mut(), // Virtual images use graph context
        data: Arc::new(RwLock::new(Vec::new())),
        mapped_patches: Arc::new(RwLock::new(Vec::new())),
        parent: None,
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

        // Validate dimensions
        if width == 0 || height == 0 {
            return std::ptr::null_mut();
        }

        // Calculate size from addressing with overflow protection
        // and sanity limits to prevent massive allocations
        const MAX_ALLOCATION_SIZE: usize = 1024 * 1024 * 1024; // 1GB limit
        
        let total_size = if addr.stride_y > 0 {
            // Validate stride_y is reasonable (not larger than a single row * 100)
            let expected_stride = width as usize * VxCImage::bytes_per_pixel(color);
            let stride = addr.stride_y as usize;

            if stride > expected_stride * 100 {
                return std::ptr::null_mut(); // Unreasonable stride
            }

            // Calculate total size: stride_y * height
            stride
                .checked_mul(height as usize)
                .unwrap_or(0)
                .min(MAX_ALLOCATION_SIZE)
        } else {
            return std::ptr::null_mut();
        };

        if total_size == 0 {
            return std::ptr::null_mut();
        }

        // Allocate data buffer and copy from user handles
        let mut data = vec![0u8; total_size];

        // Copy from first plane pointer (user handles)
        let user_ptr = *ptrs;
        if !user_ptr.is_null() {
            std::ptr::copy_nonoverlapping(
                user_ptr as *const u8,
                data.as_mut_ptr(),
                total_size
            );
        }

        let image = Box::new(VxCImage {
            width,
            height,
            format: color,
            is_virtual: false,
            context,
            data: Arc::new(RwLock::new(data)),
            mapped_patches: Arc::new(RwLock::new(Vec::new())),
            parent: None,
        });

        let image_ptr = Box::into_raw(image) as vx_image;

        // Register image address in unified registry for type queries (vxQueryReference)
        register_image(image_ptr as usize);

        image_ptr
    }
}

/// \brief Creates an image object with the specified attributes and sets all pixels to the uniform value specified by value pointer.
/// 
/// Creates an image with the specified width, height, and color format, where all pixels
/// are initialized to the same uniform value. This is useful for creating test images or
/// constant images for graph operations.
/// 
/// \param [in] context The reference to the overall context
/// \param [in] width The image width in pixels
/// \param [in] height The image height in pixels
/// \param [in] color The VX_DF_IMAGE format code
/// \param [in] value A pointer to the \ref vx_pixel_value_t union to use for the uniform image
/// \return An image reference VX_SUCCESS
#[no_mangle]
pub extern "C" fn vxCreateUniformImage(
    context: vx_context,
    width: vx_uint32,
    height: vx_uint32,
    color: vx_df_image,
    value: *const vx_pixel_value_t,
) -> vx_image {
    // Validate context parameter
    if context.is_null() {
        return std::ptr::null_mut();
    }
    
    // Validate value parameter
    if value.is_null() {
        return std::ptr::null_mut();
    }
    
    // Validate dimensions (width and height must be > 0)
    if width == 0 || height == 0 {
        return std::ptr::null_mut();
    }

    // Calculate the required buffer size
    let size = VxCImage::calculate_size(width, height, color);
    if size == 0 {
        return std::ptr::null_mut();
    }

    // Create data buffer and fill with uniform value
    let mut data = vec![0u8; size];
    
    unsafe {
        let val = std::ptr::read(value);
        
        // Fill data based on format using uppercase field names to match C OpenVX spec
        match color {
            VX_DF_IMAGE_U8 => {
                data.fill(val.U8);
            }
            VX_DF_IMAGE_U16 => {
                let v = val.U16.to_le_bytes();
                for chunk in data.chunks_exact_mut(2) {
                    chunk[0] = v[0];
                    chunk[1] = v[1];
                }
            }
            VX_DF_IMAGE_S16 => {
                let v = val.S16.to_le_bytes();
                for chunk in data.chunks_exact_mut(2) {
                    chunk[0] = v[0];
                    chunk[1] = v[1];
                }
            }
            VX_DF_IMAGE_U32 => {
                let v = val.U32.to_le_bytes();
                for chunk in data.chunks_exact_mut(4) {
                    chunk[0] = v[0];
                    chunk[1] = v[1];
                    chunk[2] = v[2];
                    chunk[3] = v[3];
                }
            }
            VX_DF_IMAGE_S32 => {
                let v = val.S32.to_le_bytes();
                for chunk in data.chunks_exact_mut(4) {
                    chunk[0] = v[0];
                    chunk[1] = v[1];
                    chunk[2] = v[2];
                    chunk[3] = v[3];
                }
            }
            VX_DF_IMAGE_RGB => {
                for chunk in data.chunks_exact_mut(3) {
                    chunk[0] = val.RGB[0];
                    chunk[1] = val.RGB[1];
                    chunk[2] = val.RGB[2];
                }
            }
            VX_DF_IMAGE_RGBA => {
                for chunk in data.chunks_exact_mut(4) {
                    chunk[0] = val.RGBA[0];
                    chunk[1] = val.RGBA[1];
                    chunk[2] = val.RGBA[2];
                    chunk[3] = val.RGBA[3];
                }
            }
            VX_DF_IMAGE_RGBX => {
                for chunk in data.chunks_exact_mut(4) {
                    chunk[0] = val.RGBX[0];
                    chunk[1] = val.RGBX[1];
                    chunk[2] = val.RGBX[2];
                    chunk[3] = val.RGBX[3];
                }
            }
            VX_DF_IMAGE_YUV4 | VX_DF_IMAGE_IYUV | VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => {
                // YUV formats use YUV field - fill with Y value as default
                data.fill(val.YUV[0]);
            }
            VX_DF_IMAGE_UYVY | VX_DF_IMAGE_YUYV => {
                // Packed YUV formats - fill with Y value
                data.fill(val.YUV[0]);
            }
            _ => {
                // Default: fill with U8 value for other formats
                data.fill(val.U8);
            }
        }
    }

    // Create the image structure
    let image = Box::new(VxCImage {
        width,
        height,
        format: color,
        is_virtual: false,
        context,
        data: Arc::new(RwLock::new(data)),
        mapped_patches: Arc::new(RwLock::new(Vec::new())),
        parent: None,
    });

    // Convert to raw pointer
    let image_ptr = Box::into_raw(image) as vx_image;
    
    // Register image address in unified registry for type queries (vxQueryReference)
    register_image(image_ptr as usize);
    
    image_ptr
}

/// Create an image from a channel of another image
/// 
/// According to OpenVX spec, this creates a sub-image from a specific channel
/// of a YUV formatted parent image. Valid channels are:
/// - VX_CHANNEL_Y (0): Y plane (valid for IYUV, NV12, NV21, YUV4)
/// - VX_CHANNEL_U (1): U plane (valid for IYUV, YUV4)
/// - VX_CHANNEL_V (2): V plane (valid for IYUV, YUV4)
/// 
/// The function extracts the context from the parent image.
#[no_mangle]
pub extern "C" fn vxCreateImageFromChannel(
    img: vx_image,
    channel: vx_enum,
) -> vx_image {
    if img.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let source_img = &*(img as *const VxCImage);
        
        // Per OpenVX spec, only YUV formats support channel extraction
        // Validate channel based on source format
        match channel {
            // Y channel is valid for IYUV, NV12, NV21, YUV4
            4 /* VX_CHANNEL_Y */ => {
                match source_img.format {
                    VX_DF_IMAGE_YUV4 | VX_DF_IMAGE_IYUV | VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => {
                        // Valid - continue
                    }
                    _ => return std::ptr::null_mut(),
                }
            }
            // U and V channels are only valid for YUV4 and IYUV
            5 /* VX_CHANNEL_U */ | 6 /* VX_CHANNEL_V */ => {
                match source_img.format {
                    VX_DF_IMAGE_YUV4 | VX_DF_IMAGE_IYUV => {
                        // Valid - continue
                    }
                    _ => return std::ptr::null_mut(),
                }
            }
            _ => return std::ptr::null_mut(),
        }

        // Get context from parent image
        let context = source_img.context;
        if context.is_null() {
            return std::ptr::null_mut();
        }

        // Calculate dimensions based on plane
        // For YUV formats:
        // - Y plane: full width x height
        // - U/V planes for IYUV: width/2 x height/2 (4:2:0 subsampling)
        // - U/V planes for YUV4: full width x height (4:4:4, no subsampling)
        let (output_width, output_height) = match channel {
            4 /* VX_CHANNEL_Y */ => {
                // Y plane is full resolution
                (source_img.width, source_img.height)
            }
            5 /* VX_CHANNEL_U */ | 6 /* VX_CHANNEL_V */ => {
                // U/V planes depend on format
                match source_img.format {
                    VX_DF_IMAGE_IYUV | VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => {
                        // 4:2:0 subsampling - half resolution
                        ((source_img.width + 1) / 2, (source_img.height + 1) / 2)
                    }
                    VX_DF_IMAGE_YUV4 => {
                        // 4:4:4 - full resolution
                        (source_img.width, source_img.height)
                    }
                    _ => return std::ptr::null_mut(),
                }
            }
            _ => return std::ptr::null_mut(),
        };

        // Create channel image that shares data with parent
        // Note: This is a simplified implementation. A full implementation
        // would need to handle plane offsets for YUV formats properly.
        // Store the parent image pointer to keep parent alive while sub-image exists
        let parent_ptr = img as usize;
        let channel_image = Box::new(VxCImage {
            width: output_width,
            height: output_height,
            format: VX_DF_IMAGE_U8, // Channel images are always U8
            is_virtual: false,
            context: context,
            data: Arc::clone(&source_img.data),
            mapped_patches: Arc::new(RwLock::new(Vec::new())),
            parent: Some(parent_ptr),
        });

        let image_ptr = Box::into_raw(channel_image) as vx_image;

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
        let img = *image;
        if !img.is_null() {
            // Unregister from unified registry
            unregister_image(img as usize);
            
            // Free the image
            let _ = Box::from_raw(img as *mut VxCImage);
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

    let img = unsafe { &*(image as *const VxCImage) };

    unsafe {
        match attribute {
            VX_IMAGE_FORMAT => {
                if size != std::mem::size_of::<vx_df_image>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_df_image) = img.format;
                VX_SUCCESS
            }
            VX_IMAGE_WIDTH => {
                if size != std::mem::size_of::<vx_uint32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_uint32) = img.width;
                VX_SUCCESS
            }
            VX_IMAGE_HEIGHT => {
                if size != std::mem::size_of::<vx_uint32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_uint32) = img.height;
                VX_SUCCESS
            }
            VX_IMAGE_PLANES => {
                if size != std::mem::size_of::<vx_uint32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                // Number of planes based on format
                let planes = VxCImage::num_planes(img.format);
                *(ptr as *mut vx_uint32) = planes as vx_uint32;
                VX_SUCCESS
            }
            VX_IMAGE_IS_UNIFORM => {
                if size != std::mem::size_of::<vx_enum>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                // Currently not tracking uniform status
                *(ptr as *mut vx_enum) = 0; // vx_false_e
                VX_SUCCESS
            }
            VX_IMAGE_UNIFORM_VALUE => {
                // Not implemented
                VX_ERROR_NOT_IMPLEMENTED
            }
            _ => VX_ERROR_NOT_IMPLEMENTED,
        }
    }
}

/// Set image attributes
#[no_mangle]
pub extern "C" fn vxSetImageAttribute(
    _image: vx_image,
    _attribute: vx_enum,
    _ptr: *const c_void,
    _size: vx_size,
) -> vx_status {
    VX_ERROR_NOT_IMPLEMENTED
}

/// Map image patch for CPU access
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

    let img = unsafe { &mut *(image as *mut VxCImage) };

    unsafe {
        let rect_ref = &*rect;

        // Calculate the mapped region
        let start_x = rect_ref.start_x as usize;
        let start_y = rect_ref.start_y as usize;
        let end_x = rect_ref.end_x as usize;
        let end_y = rect_ref.end_y as usize;
        
        let width = end_x.saturating_sub(start_x);
        let height = end_y.saturating_sub(start_y);
        
        if width == 0 || height == 0 {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        // Get image data
        let data_guard = match img.data.read() {
            Ok(guard) => guard,
            Err(_) => return VX_ERROR_INVALID_REFERENCE,
        };

        // Determine plane-specific parameters for planar YUV formats
        let is_planar = VxCImage::is_planar_format(img.format);
        let (plane_width, plane_height) = if is_planar {
            let (pw, ph) = VxCImage::plane_dimensions(img.width, img.height, img.format, plane_index as usize);
            (pw as usize, ph as usize)
        } else {
            (img.width as usize, img.height as usize)
        };

        // Validate the plane_index
        if is_planar && plane_index as usize >= VxCImage::num_planes(img.format) {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        // Calculate stride and offset based on format and plane
        let (bpp, stride_y, plane_offset) = if is_planar {
            // For planar formats, each plane is 1 byte per pixel (luma/chroma)
            let bpp = 1usize;
            let stride_y = plane_width * bpp;
            let plane_offset = VxCImage::plane_offset(img.width, img.height, img.format, plane_index as usize);
            (bpp, stride_y, plane_offset)
        } else {
            // For packed formats, use standard bytes per pixel
            let bpp = VxCImage::bytes_per_pixel(img.format);
            let stride_y = plane_width * bpp;
            (bpp, stride_y, 0)
        };

        let offset = plane_offset + start_y * stride_y + start_x * bpp;

        // Create a copy of the data for the mapped patch
        let patch_size = height * width * bpp;
        let mut patch_data = vec![0u8; patch_size];
        
        // Copy data row by row
        for y in 0..height {
            let src_start = offset + y * stride_y;
            let dst_start = y * width * bpp;
            if src_start + width * bpp <= data_guard.len() {
                patch_data[dst_start..dst_start + width * bpp]
                    .copy_from_slice(&data_guard[src_start..src_start + width * bpp]);
            }
        }
        
        // Store the patch
        let map_id_val = if let Ok(mut patches) = img.mapped_patches.write() {
            let id = patches.len() + 1;
            // Store plane_index in the offset field (using upper bits) to track which plane was mapped
            // This is a hack; we encode plane_index * stride_y in the stored stride_y to track it
            let encoded_stride_y = stride_y | (plane_index as usize) << 48;
            patches.push((id, patch_data.clone(), usage, offset, encoded_stride_y));
            id
        } else {
            return VX_ERROR_INVALID_REFERENCE;
        };

        // Fill addressing structure
        (*addr).dim_x = width as vx_uint32;
        (*addr).dim_y = height as vx_uint32;
        (*addr).stride_x = bpp as vx_int32;
        (*addr).stride_y = stride_y as vx_int32;
        (*addr).scale_x = 0;
        (*addr).scale_y = 0;
        
        // Set output parameters
        *map_id = map_id_val;
        
        // Return pointer to the patch data
        if let Ok(patches) = img.mapped_patches.read() {
            if let Some(patch) = patches.iter().find(|(id, _, _, _, _)| *id == map_id_val) {
                *ptr = patch.1.as_ptr() as *mut c_void;
            }
        }

        // Keep the data_guard alive until after we've set the ptr
        drop(data_guard);
    }

    VX_SUCCESS
}

/// Unmap image patch
#[no_mangle]
pub extern "C" fn vxUnmapImagePatch(
    image: vx_image,
    map_id: vx_map_id,
) -> vx_status {
    if image.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let img = unsafe { &mut *(image as *mut VxCImage) };

    if let Ok(mut patches) = img.mapped_patches.write() {
        if let Some(pos) = patches.iter().position(|(id, _, _, _, _)| *id == map_id) {
            let (_, patch_data, usage, offset, encoded_stride_y) = patches.remove(pos);
            
            // Decode the plane index and actual stride_y
            let is_planar = VxCImage::is_planar_format(img.format);
            let (stride_y, plane_index) = if is_planar {
                // Extract plane index from upper bits
                let plane_idx = (encoded_stride_y >> 48) as u32;
                let actual_stride_y = encoded_stride_y & 0xFFFFFFFFFFFF;
                (actual_stride_y, plane_idx)
            } else {
                (encoded_stride_y, 0)
            };
            
            // If write access, copy data back
            if usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE {
                if let Ok(mut data) = img.data.write() {
                    let bpp = if is_planar {
                        1 // Planar formats are 1 byte per pixel
                    } else {
                        VxCImage::bytes_per_pixel(img.format)
                    };
                    
                    // Calculate width based on stride_y and bpp
                    let width = if stride_y > 0 {
                        stride_y / bpp
                    } else {
                        0
                    };
                    let height = if stride_y > 0 && width > 0 {
                        patch_data.len() / stride_y
                    } else {
                        0
                    };
                    
                    // Copy data back row by row
                    for y in 0..height {
                        let src_start = y * width * bpp;
                        let dst_start = offset + y * stride_y;
                        if dst_start + width * bpp <= data.len() && src_start + width * bpp <= patch_data.len() {
                            data[dst_start..dst_start + width * bpp]
                                .copy_from_slice(&patch_data[src_start..src_start + width * bpp]);
                        }
                    }
                }
            }
            
            VX_SUCCESS
        } else {
            VX_ERROR_INVALID_PARAMETERS
        }
    } else {
        VX_ERROR_INVALID_REFERENCE
    }
}

/// Copy image patch
#[no_mangle]
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
    if image.is_null() || rect.is_null() || user_addr.is_null() || user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let img = unsafe { &mut *(image as *mut VxCImage) };

    unsafe {
        let rect_ref = &*rect;
        let addr = &*user_addr;
        
        // Calculate the region
        let start_x = rect_ref.start_x as usize;
        let start_y = rect_ref.start_y as usize;
        let end_x = rect_ref.end_x as usize;
        let end_y = rect_ref.end_y as usize;
        
        let width = end_x.saturating_sub(start_x);
        let height = end_y.saturating_sub(start_y);
        
        if width == 0 || height == 0 {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        // Determine plane-specific parameters for planar YUV formats
        let is_planar = VxCImage::is_planar_format(img.format);
        
        // Validate the plane_index
        if is_planar && plane_index as usize >= VxCImage::num_planes(img.format) {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        let (plane_width, plane_height) = if is_planar {
            let (pw, ph) = VxCImage::plane_dimensions(img.width, img.height, img.format, plane_index as usize);
            (pw as usize, ph as usize)
        } else {
            (img.width as usize, img.height as usize)
        };

        // Calculate stride and offset based on format and plane
        let (bpp, stride_y, plane_offset) = if is_planar {
            // For planar formats, each plane is 1 byte per pixel (luma/chroma)
            let bpp = 1usize;
            let stride_y = plane_width * bpp;
            let plane_offset = VxCImage::plane_offset(img.width, img.height, img.format, plane_index as usize);
            (bpp, stride_y, plane_offset)
        } else {
            // For packed formats, use standard bytes per pixel
            let bpp = VxCImage::bytes_per_pixel(img.format);
            let stride_y = plane_width * bpp;
            (bpp, stride_y, 0)
        };

        let offset = plane_offset + start_y * stride_y + start_x * bpp;

        match usage {
            VX_READ_ONLY => {
                // Copy from image to user buffer
                if let Ok(data) = img.data.read() {
                    for y in 0..height {
                        let src_start = offset + y * stride_y;
                        let dst_start = y * addr.stride_y as usize;
                        if src_start + width * bpp <= data.len() {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr().add(src_start),
                                (user_ptr as *mut u8).add(dst_start),
                                width * bpp
                            );
                        }
                    }
                }
            }
            VX_WRITE_ONLY => {
                // Copy from user buffer to image
                if let Ok(mut data) = img.data.write() {
                    for y in 0..height {
                        let dst_start = offset + y * stride_y;
                        let src_start = y * addr.stride_y as usize;
                        if dst_start + width * bpp <= data.len() {
                            std::ptr::copy_nonoverlapping(
                                (user_ptr as *const u8).add(src_start),
                                data.as_mut_ptr().add(dst_start),
                                width * bpp
                            );
                        }
                    }
                }
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    VX_SUCCESS
}

/// Get valid region image
#[no_mangle]
pub extern "C" fn vxGetValidRegionImage(
    image: vx_image,
    rect: *mut vx_rectangle_t,
) -> vx_status {
    if image.is_null() || rect.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let img = unsafe { &*(image as *const VxCImage) };

    unsafe {
        // Return full image as valid region
        (*rect).start_x = 0;
        (*rect).start_y = 0;
        (*rect).end_x = img.width;
        (*rect).end_y = img.height;
    }

    VX_SUCCESS
}

/// Set image valid rectangle
#[no_mangle]
pub extern "C" fn vxSetImageValidRectangle(
    _image: vx_image,
    _rect: *const vx_rectangle_t,
) -> vx_status {
    // Stub - valid rectangle tracking not implemented
    VX_SUCCESS
}

/// Swap image handle
#[no_mangle]
pub extern "C" fn vxSwapImageHandle(
    _image: vx_image,
    _new_ptrs: *const *mut c_void,
    _prev_ptrs: *mut *mut c_void,
    _num_planes: vx_uint32,
) -> vx_status {
    VX_ERROR_NOT_IMPLEMENTED
}

/// Compute image pattern
#[no_mangle]
pub extern "C" fn vxComputeImagePattern(
    _image: vx_image,
    _rect: *const vx_rectangle_t,
    _num_points: vx_uint32,
    _points: *const vx_keypoint_t,
    _pattern: *mut vx_enum,
) -> vx_status {
    VX_ERROR_NOT_IMPLEMENTED
}

/// Copy image
#[no_mangle]
pub extern "C" fn vxCopyImage(
    image: vx_image,
    ptr: *mut c_void,
    usage: vx_enum,
    mem_type: vx_enum,
) -> vx_status {
    if image.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let img = unsafe { &*(image as *const VxCImage) };

    unsafe {
        if let Ok(data) = img.data.read() {
            match usage {
                VX_READ_ONLY => {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ptr as *mut u8,
                        data.len()
                    );
                }
                VX_WRITE_ONLY => {
                    if let Ok(mut data) = img.data.write() {
                        std::ptr::copy_nonoverlapping(
                            ptr as *const u8,
                            data.as_mut_ptr(),
                            data.len()
                        );
                    }
                }
                _ => return VX_ERROR_INVALID_PARAMETERS,
            }
        }
    }

    VX_SUCCESS
}

/// Copy image plane
#[no_mangle]
pub extern "C" fn vxCopyImagePlane(
    image: vx_image,
    plane_index: vx_uint32,
    ptr: *mut c_void,
    usage: vx_enum,
    mem_type: vx_enum,
) -> vx_status {
    if image.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if mem_type != VX_MEMORY_TYPE_HOST {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    let img = unsafe { &*(image as *const VxCImage) };

    unsafe {
        if let Ok(data) = img.data.read() {
            // For planar formats, calculate plane offsets using the unified helper
            let is_planar = VxCImage::is_planar_format(img.format);
            if plane_index as usize >= VxCImage::num_planes(img.format) {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let plane_offset = VxCImage::plane_offset(img.width, img.height, img.format, plane_index as usize);
            let plane_size = VxCImage::plane_size(img.width, img.height, img.format, plane_index as usize);

            if plane_size == 0 || plane_offset + plane_size > data.len() {
                return VX_ERROR_INVALID_PARAMETERS;
            }

            match usage {
                VX_READ_ONLY => {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(plane_offset),
                        ptr as *mut u8,
                        plane_size
                    );
                }
                VX_WRITE_ONLY => {
                    if let Ok(mut data) = img.data.write() {
                        std::ptr::copy_nonoverlapping(
                            ptr as *const u8,
                            data.as_mut_ptr().add(plane_offset),
                            plane_size
                        );
                    }
                }
                _ => return VX_ERROR_INVALID_PARAMETERS,
            }
        }
    }

    VX_SUCCESS
}

/// Allocate image memory
#[no_mangle]
pub extern "C" fn vxAllocateImageMemory(
    _image: vx_image,
    _type: vx_enum,
) -> vx_status {
    VX_SUCCESS
}

/// Release image memory
#[no_mangle]
pub extern "C" fn vxReleaseImageMemory(
    _image: vx_image,
    _type: vx_enum,
) -> vx_status {
    VX_SUCCESS
}

/// Create image from ROI
#[no_mangle]
pub extern "C" fn vxCreateImageFromROI(
    img: vx_image,
    rect: *const vx_rectangle_t,
) -> vx_image {
    if img.is_null() || rect.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let source_img = &*(img as *const VxCImage);
        let rect_ref = &*rect;

        // Calculate ROI dimensions
        let roi_width = rect_ref.end_x.saturating_sub(rect_ref.start_x);
        let roi_height = rect_ref.end_y.saturating_sub(rect_ref.start_y);

        if roi_width == 0 || roi_height == 0 {
            return std::ptr::null_mut();
        }

        // Create ROI image that references parent data
        // Note: This is a simplified implementation that copies data
        // A full implementation would handle ROI as a view into the parent
        let bpp = VxCImage::bytes_per_pixel(source_img.format);
        let parent_stride = source_img.width as usize * bpp;
        let roi_stride = roi_width as usize * bpp;
        let roi_size = roi_height as usize * roi_stride;

        let mut roi_data = vec![0u8; roi_size];

        if let Ok(parent_data) = source_img.data.read() {
            for y in 0..roi_height as usize {
                let src_y = rect_ref.start_y as usize + y;
                let src_offset = src_y * parent_stride + rect_ref.start_x as usize * bpp;
                let dst_offset = y * roi_stride;
                
                if src_offset + roi_stride <= parent_data.len() {
                    roi_data[dst_offset..dst_offset + roi_stride]
                        .copy_from_slice(&parent_data[src_offset..src_offset + roi_stride]);
                }
            }
        }

        let roi_image = Box::new(VxCImage {
            width: roi_width,
            height: roi_height,
            format: source_img.format,
            is_virtual: false,
            context: source_img.context,
            data: Arc::new(RwLock::new(roi_data)),
            mapped_patches: Arc::new(RwLock::new(Vec::new())),
            parent: Some(img as usize), // Store parent reference
        });

        let image_ptr = Box::into_raw(roi_image) as vx_image;

        // Register image address in unified registry for type queries (vxQueryReference)
        register_image(image_ptr as usize);

        image_ptr
    }
}

/// Map image
#[no_mangle]
pub extern "C" fn vxMapImage(
    _image: vx_image,
    _map_id: *mut vx_map_id,
    _usage: vx_enum,
) -> vx_status {
    VX_ERROR_NOT_IMPLEMENTED
}

/// Unmap image
#[no_mangle]
pub extern "C" fn vxUnmapImage(
    _image: vx_image,
    _map_id: vx_map_id,
) -> vx_status {
    VX_ERROR_NOT_IMPLEMENTED
}

/// Lock image
#[no_mangle]
pub extern "C" fn vxLockImage(
    _image: vx_image,
    _usage: vx_enum,
) -> *mut c_void {
    std::ptr::null_mut()
}

/// Unlock image
#[no_mangle]
pub extern "C" fn vxUnlockImage(
    _image: vx_image,
) -> vx_status {
    VX_ERROR_NOT_IMPLEMENTED
}

/// Get image plane count
#[no_mangle]
pub extern "C" fn vxGetImagePlaneCount(
    image: vx_image,
) -> vx_uint32 {
    if image.is_null() {
        return 0;
    }

    let img = unsafe { &*(image as *const VxCImage) };
    
    VxCImage::num_planes(img.format) as vx_uint32
}

/// Create image from ROI handle
#[no_mangle]
pub extern "C" fn vxCreateImageFromROIH(
    _context: vx_context,
    _parent: vx_image,
    _rect: *const vx_rectangle_t,
) -> vx_image {
    std::ptr::null_mut()
}

/// Create uniform image from handle
#[no_mangle]
pub extern "C" fn vxCreateUniformImageFromHandle(
    _context: vx_context,
    _width: vx_uint32,
    _height: vx_uint32,
    _color: vx_df_image,
    _value: *const vx_pixel_value_t,
) -> vx_image {
    std::ptr::null_mut()
}

/// Clone an image - creates a deep copy of the source image
///
/// This function creates a new image with the same dimensions and format as the source,
/// then copies all pixel data from the source to the new image.
///
/// # Arguments
/// * `context` - The OpenVX context
/// * `source` - The source image to clone
///
/// # Returns
/// A new vx_image handle that is a deep copy of the source, or null on error
#[no_mangle]
pub extern "C" fn vxCloneImage(
    context: vx_context,
    source: vx_image,
) -> vx_image {
    // Validate parameters
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if source.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let source_img = &*(source as *const VxCImage);

        // Get source image properties
        let width = source_img.width;
        let height = source_img.height;
        let format = source_img.format;

        // Handle virtual images - they don't have allocated data
        // Per OpenVX spec, cloning a virtual image creates a regular image
        if source_img.is_virtual && source_img.data.read().unwrap().is_empty() {
            // Virtual image without data - just create regular image with same dimensions
            return vxCreateImage(context, width, height, format);
        }

        // Create the destination image
        let mut dest_image = vxCreateImage(context, width, height, format);
        if dest_image.is_null() {
            return std::ptr::null_mut();
        }

        // Get the destination image data structure
        let dest_img = &mut *(dest_image as *mut VxCImage);

        // Copy all data from source to destination
        if let Ok(source_data) = source_img.data.read() {
            if let Ok(mut dest_data) = dest_img.data.write() {
                // Ensure destination has enough space
                if dest_data.len() >= source_data.len() {
                    // Perform deep copy of pixel data
                    dest_data.copy_from_slice(&source_data[..source_data.len()]);
                } else {
                    // Shouldn't happen if vxCreateImage worked correctly, but handle it
                    vxReleaseImage(&mut dest_image);
                    return std::ptr::null_mut();
                }
            } else {
                vxReleaseImage(&mut dest_image);
                return std::ptr::null_mut();
            }
        } else {
            vxReleaseImage(&mut dest_image);
            return std::ptr::null_mut();
        }

        // Also copy mapped patches metadata if any exist
        if let Ok(source_patches) = source_img.mapped_patches.read() {
            if let Ok(mut dest_patches) = dest_img.mapped_patches.write() {
                // Copy the patch metadata (not the actual data which is owned by the patches)
                *dest_patches = source_patches.clone();
            }
        }

        dest_image
    }
}

/// Clone an image for use with a specific graph
///
/// This variant creates a virtual image if the source is virtual,
/// otherwise creates a regular image. This is used by CTS for cloning
/// images within graph contexts.
///
/// # Arguments
/// * `context` - The OpenVX context (used if source is not virtual)
/// * `graph` - The OpenVX graph (used if source is virtual)
/// * `source` - The source image to clone
///
/// # Returns
/// A new vx_image handle that is a clone of the source, or null on error
#[no_mangle]
pub extern "C" fn vxCloneImageWithGraph(
    context: vx_context,
    graph: vx_graph,
    source: vx_image,
) -> vx_image {
    // Validate source
    if source.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let source_img = &*(source as *const VxCImage);
        let width = source_img.width;
        let height = source_img.height;
        let format = source_img.format;
        let is_virtual = source_img.is_virtual;

        // Determine if we need a virtual image
        let needs_virtual = is_virtual || source_img.data.read().unwrap().is_empty();

        if needs_virtual {
            // Validate graph for virtual image creation
            if graph.is_null() {
                return std::ptr::null_mut();
            }
            // Create a virtual image with same dimensions
            vxCreateVirtualImage(graph, width, height, format)
        } else {
            // Create a regular image with data copy
            vxCloneImage(context, source)
        }
    }
}
