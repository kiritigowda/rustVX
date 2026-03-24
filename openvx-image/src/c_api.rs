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
    VX_READ_ONLY, VX_WRITE_ONLY, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST,
};

// Global image registry
static IMAGE_ID_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// Image struct for C API
#[derive(Debug)]
pub struct VxCImage {
    width: vx_uint32,
    height: vx_uint32,
    format: vx_df_image,
    is_virtual: bool,
    context: vx_context,
    data: Arc<RwLock<Vec<u8>>>,
    // map_id, buffer, usage, offset, stride
    mapped_patches: Arc<RwLock<Vec<(vx_map_id, Vec<u8>, vx_enum, usize, usize)>>>,
}

impl VxCImage {
    fn bytes_per_pixel(format: vx_df_image) -> usize {
        match format {
            VX_DF_IMAGE_U8 => 1,
            VX_DF_IMAGE_U16 | VX_DF_IMAGE_S16 => 2,
            VX_DF_IMAGE_U32 | VX_DF_IMAGE_S32 => 4,
            VX_DF_IMAGE_RGB => 3,
            VX_DF_IMAGE_RGBA | VX_DF_IMAGE_RGBX => 4,
            VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => 1, // Luma plane only per-pixel
            VX_DF_IMAGE_IYUV => 1,                    // Y plane only per-pixel
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
        // Use bytes_per_pixel for allocation, NOT channels
        // channels() returns logical color channels, bytes_per_pixel() returns actual memory needed
        let bpp = Self::bytes_per_pixel(format);

        // Limit maximum allocation to ~1GB (sanity check)
        let max_size = 1024 * 1024 * 1024; // 1GB
        let size = w.saturating_mul(h).saturating_mul(bpp);

        if size > max_size {
            eprintln!("Image size {}x{}x{} = {} exceeds maximum allocation limit", w, h, bpp, size);
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
                // stride_y seems unreasonably large, use calculated stride
                expected_stride.checked_mul(height as usize)
            } else {
                (height as usize).checked_mul(stride)
            }
        } else {
            (width as usize)
                .checked_mul(height as usize)
                .and_then(|s| s.checked_mul(VxCImage::bytes_per_pixel(color)))
        };
        
        let total_size = match total_size {
            Some(size) if size > 0 && size <= MAX_ALLOCATION_SIZE => size,
            _ => return std::ptr::null_mut(),
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

    // Accept READ_ONLY, WRITE_ONLY, or READ_AND_WRITE
    if usage != VX_READ_ONLY && usage != VX_WRITE_ONLY && usage != VX_READ_AND_WRITE {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let img = unsafe { &mut *(image as *mut VxCImage) };
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

    // Calculate patch dimensions
    let patch_width = (rect.end_x - rect.start_x) as usize;
    let patch_height = (rect.end_y - rect.start_y) as usize;
    let patch_stride = stride_y as usize;
    
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

    // For WRITE_ONLY: allocate a temporary buffer to write into
    // For READ_ONLY: allocate a buffer and copy the data
    let patch_size = patch_height.saturating_mul(patch_stride);
    if patch_size == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    
    // Get mutable access to data for potential copying
    {
        let data = img.data.read().unwrap();
        
        if offset >= data.len() {
            return VX_ERROR_INVALID_PARAMETERS;
        }
        
        // Check if we have enough data
        let required_size = offset + patch_size;
        if required_size > data.len() && usage == VX_READ_ONLY {
            // Not enough data to read
            return VX_ERROR_INVALID_PARAMETERS;
        }
    }
    
    // Allocate a persistent buffer for the mapped patch
    // This ensures the pointer remains valid until unmap
    let mut patch_buffer: Vec<u8> = vec![0; patch_size];
    
    if usage == VX_READ_ONLY || usage == VX_READ_AND_WRITE {
        // Copy data from image to buffer
        let data = img.data.read().unwrap();
        for row in 0..patch_height {
            let src_offset = offset + (row * patch_stride);
            let row_size = patch_width.saturating_mul(bpp);
            if src_offset + row_size <= data.len() {
                let dst_offset = row * patch_stride;
                patch_buffer[dst_offset..dst_offset + row_size]
                    .copy_from_slice(&data[src_offset..src_offset + row_size]);
            }
        }
    }
    
    // Generate map ID
    let id = IMAGE_ID_COUNTER.fetch_add(1, Ordering::SeqCst) as vx_map_id;
    
    // Store the patch buffer and metadata, then get pointer from stored buffer
    let patch_ptr = {
        let mut mapped = img.mapped_patches.write().unwrap();
        mapped.push((id, patch_buffer, usage, offset, patch_stride));
        // Get pointer from the stored buffer
        mapped.last_mut().unwrap().1.as_mut_ptr() as *mut c_void
    };
    
    unsafe {
        *ptr = patch_ptr;
        *map_id = id;
    }

    VX_SUCCESS
}

/// Unmap an image patch
#[no_mangle]
pub extern "C" fn vxUnmapImagePatch(
    image: vx_image,
    map_id: vx_map_id,
) -> vx_status {
    if image.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let img = unsafe { &mut *(image as *mut VxCImage) };
    
    let mut mapped = img.mapped_patches.write().unwrap();
    
    // Find the patch by map_id
    if let Some(pos) = mapped.iter().position(|(id, _, _, _, _)| *id == map_id) {
        let (_, buffer, usage, offset, stride) = mapped.remove(pos);
        
        // If this was a write or read-write mapping, copy data back to the image
        if usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE {
            let mut data = img.data.write().unwrap();
            let patch_height = buffer.len() / stride;
            
            for row in 0..patch_height {
                let src_offset = row * stride;
                let dst_offset = offset + (row * stride);
                let row_size = stride.min(buffer.len() - src_offset);
                
                if dst_offset + row_size <= data.len() && src_offset + row_size <= buffer.len() {
                    data[dst_offset..dst_offset + row_size]
                        .copy_from_slice(&buffer[src_offset..src_offset + row_size]);
                }
            }
        }
        
        VX_SUCCESS
    } else {
        // Map ID not found
        VX_ERROR_INVALID_PARAMETERS
    }
}

/// Channel constants (re-exported from unified_c_api for local use)
pub const VX_CHANNEL_0: vx_enum = 0;
pub const VX_CHANNEL_1: vx_enum = 1;
pub const VX_CHANNEL_2: vx_enum = 2;
pub const VX_CHANNEL_3: vx_enum = 3;
pub const VX_CHANNEL_R: vx_enum = 0;
pub const VX_CHANNEL_G: vx_enum = 1;
pub const VX_CHANNEL_B: vx_enum = 2;
pub const VX_CHANNEL_A: vx_enum = 3;
pub const VX_CHANNEL_Y: vx_enum = 4;
pub const VX_CHANNEL_U: vx_enum = 5;
pub const VX_CHANNEL_V: vx_enum = 6;

/// Image struct for channel-extracted image views
/// This is a specialized variant that references a parent image's channel data
#[derive(Debug, Clone)]
pub struct VxCChannelImage {
    width: vx_uint32,
    height: vx_uint32,
    format: vx_df_image,
    channel: vx_enum,
    parent_context: vx_context,
    parent_image: vx_image,
    channel_offset: usize,
    channel_stride: usize,
    parent_data: Arc<RwLock<Vec<u8>>>,
}

/// Create an image from a specific channel of a multi-channel source image
/// 
/// This function creates a new image that references data from a single channel
/// of a multi-channel parent image. The channel image shares the underlying data
/// buffer with the parent.
/// 
/// # Arguments
/// * `img` - The source multi-channel image
/// * `channel` - The channel to extract (VX_CHANNEL_0, VX_CHANNEL_R, VX_CHANNEL_Y, etc.)
/// 
/// # Returns
/// * A new vx_image on success, NULL on failure
#[no_mangle]
pub extern "C" fn vxCreateImageFromChannel(
    img: vx_image,
    channel: vx_enum,
) -> vx_image {
    if img.is_null() {
        return std::ptr::null_mut();
    }

    let source_img = unsafe { &*(img as *const VxCImage) };
    let source_channels = VxCImage::channels(source_img.format);
    let source_bpp = VxCImage::bytes_per_pixel(source_img.format);

    // Validate channel index
    if channel < 0 {
        return std::ptr::null_mut();
    }
    let channel_idx = channel as usize;
    if channel_idx >= source_channels {
        return std::ptr::null_mut();
    }

    // Validate that the source image is actually multi-channel
    if source_channels == 1 {
        return std::ptr::null_mut();
    }

    // Determine the output format and channel offset
    let (output_format, channel_offset, channel_stride) = match source_img.format {
        VX_DF_IMAGE_RGB => {
            // RGB: 3 channels interleaved
            (VX_DF_IMAGE_U8, channel_idx, 3)
        }
        VX_DF_IMAGE_RGBA | VX_DF_IMAGE_RGBX => {
            // RGBA/RGBX: 4 channels interleaved
            (VX_DF_IMAGE_U8, channel_idx, 4)
        }
        VX_DF_IMAGE_YUV4 => {
            // YUV4: 3 planes, separate Y, U, V
            let width = source_img.width as usize;
            let height = source_img.height as usize;
            let plane_size = match width.checked_mul(height) {
                Some(size) if size <= MAX_ALLOCATION_SIZE => size,
                _ => return std::ptr::null_mut(),
            };
            let offset = match channel_idx.checked_mul(plane_size) {
                Some(off) => off,
                None => return std::ptr::null_mut(),
            };
            let stride = 1; // Single channel, no interleaving
            (VX_DF_IMAGE_U8, offset, stride)
        }
        VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => {
            // NV12/NV21: Y plane + interleaved UV
            let width = source_img.width as usize;
            let height = source_img.height as usize;
            let y_plane_size = match width.checked_mul(height) {
                Some(size) if size <= MAX_ALLOCATION_SIZE => size,
                _ => return std::ptr::null_mut(),
            };
            if channel_idx == 0 {
                // Y channel - full Y plane
                (VX_DF_IMAGE_U8, 0, 1)
            } else if channel_idx == 1 || channel_idx == 2 {
                // U or V channel - interleaved in UV plane
                // For NV12: UV, for NV21: VU
                let channel_offset = if channel_idx == 1 { 0 } else { 1 };
                let uv_offset = match y_plane_size.checked_add(channel_offset) {
                    Some(off) => off,
                    None => return std::ptr::null_mut(),
                };
                (VX_DF_IMAGE_U8, uv_offset, 2)
            } else {
                return std::ptr::null_mut();
            }
        }
        VX_DF_IMAGE_IYUV => {
            // IYUV (I420): Y plane + U plane + V plane
            let width = source_img.width as usize;
            let height = source_img.height as usize;
            let y_plane_size = match width.checked_mul(height) {
                Some(size) if size <= MAX_ALLOCATION_SIZE => size,
                _ => return std::ptr::null_mut(),
            };
            let uv_plane_size = match (width / 2).checked_mul(height / 2) {
                Some(size) if size <= MAX_ALLOCATION_SIZE => size,
                _ => return std::ptr::null_mut(),
            };
            let offset = match channel_idx {
                0 => 0,                                    // Y
                1 => y_plane_size,                         // U
                2 => match y_plane_size.checked_add(uv_plane_size) {
                    Some(off) => off,                      // V
                    None => return std::ptr::null_mut(),
                },
                _ => return std::ptr::null_mut(),
            };
            (VX_DF_IMAGE_U8, offset, 1)
        }
        _ => {
            // Unsupported format for channel extraction
            return std::ptr::null_mut();
        }
    };

    // Create the channel image with reference to parent's data
    let channel_image = Box::new(VxCChannelImage {
        width: source_img.width,
        height: source_img.height,
        format: output_format,
        channel,
        parent_context: source_img.context,
        parent_image: img,
        channel_offset,
        channel_stride,
        parent_data: Arc::clone(&source_img.data),
    });

    // Store the channel image as a new type of image reference
    // We use a different representation - Box<VxCChannelImage> cast to vx_image
    let image_ptr = Box::into_raw(channel_image) as vx_image;

    // Register image address in unified registry for type queries (vxQueryReference)
    register_image(image_ptr as usize);

    image_ptr
}

/// Query channel image attributes
/// This is a helper function for querying VxCChannelImage objects
pub fn query_channel_image(
    image: *const VxCChannelImage,
    attribute: vx_enum,
    ptr: *mut c_void,
) -> vx_status {
    let img = unsafe { &*image };

    unsafe {
        match attribute {
            VX_IMAGE_FORMAT => {
                *(ptr as *mut vx_df_image) = img.format;
                VX_SUCCESS
            }
            VX_IMAGE_WIDTH => {
                *(ptr as *mut vx_uint32) = img.width;
                VX_SUCCESS
            }
            VX_IMAGE_HEIGHT => {
                *(ptr as *mut vx_uint32) = img.height;
                VX_SUCCESS
            }
            VX_IMAGE_PLANES => {
                *(ptr as *mut vx_size) = 1; // Single channel image has 1 plane
                VX_SUCCESS
            }
            _ => VX_ERROR_NOT_IMPLEMENTED,
        }
    }
}
