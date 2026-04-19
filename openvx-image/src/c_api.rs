//! C API for OpenVX Image

#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::sync::{RwLock, Arc, atomic::AtomicUsize};
// FFI declarations for register/unregister image - ensure we use the same symbol
// as defined in openvx-core's unified_c_api
extern "C" {
    fn register_image(addr: usize);
    fn unregister_image(addr: usize);
    fn vxGetContext(ref_: vx_reference) -> vx_context;
}
use openvx_core::unified_c_api::VxCImage;
use openvx_core::c_api::{
    vx_context, vx_graph, vx_image, vx_status, vx_enum, vx_size, vx_uint32,
    vx_rectangle_t, vx_imagepatch_addressing_t, vx_map_id, vx_df_image, vx_int32,
    vx_float32, vx_reference, VxImage,
    VX_SUCCESS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_INVALID_PARAMETERS,
    VX_ERROR_NOT_IMPLEMENTED,
    VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGBA, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_NV12,
    VX_DF_IMAGE_NV21, VX_DF_IMAGE_IYUV, VX_DF_IMAGE_UYVY, VX_DF_IMAGE_YUYV,
    VX_DF_IMAGE_YUV4, VX_DF_IMAGE_U8, VX_DF_IMAGE_U16, VX_DF_IMAGE_S16,
    VX_DF_IMAGE_U32, VX_DF_IMAGE_S32, VX_DF_IMAGE_VIRT,
    VX_IMAGE_FORMAT, VX_IMAGE_WIDTH, VX_IMAGE_HEIGHT, VX_IMAGE_PLANES,
    VX_IMAGE_IS_UNIFORM, VX_IMAGE_UNIFORM_VALUE, VX_IMAGE_SPACE, VX_IMAGE_RANGE,
    VX_IMAGE_IS_VIRTUAL,
    VX_READ_ONLY, VX_WRITE_ONLY, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST,
};
use openvx_core::unified_c_api::{REFERENCE_COUNTS, REFERENCE_TYPES, VX_TYPE_IMAGE};

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
    // VX_DF_IMAGE_VIRT is only valid for virtual images, not regular images
    if color == VX_DF_IMAGE_VIRT {
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
        is_external_memory: false,
        external_ptrs: Vec::new(),
        external_strides: Vec::new(),
        external_stride_x: Vec::new(),
        external_dim_x: Vec::new(),
        external_dim_y: Vec::new(),
        roi_offsets: Vec::new(),
        is_from_handle: false,
    });

    let image_ptr = Box::into_raw(image) as vx_image;

    // Register image address in unified registry for type queries (vxQueryReference)
    unsafe {
        register_image(image_ptr as usize);
    }
    
    // Register as valid image for double-free protection
    register_valid_image(image_ptr as usize);

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(image_ptr as usize, std::sync::atomic::AtomicUsize::new(1));
        }
    }

    // Register in REFERENCE_TYPES for type detection
    unsafe {
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(image_ptr as usize, VX_TYPE_IMAGE);
        }
    }

    image_ptr
}

/// Use unified VIRTUAL_IMAGES registry from openvx_core
use openvx_core::unified_c_api::{VIRTUAL_IMAGES, VirtualImageInfo as VxCVirtualImageInfo};
use openvx_core::unified_c_api as core_unified;

/// Re-export virtual image functions using unified registry
pub fn get_virtual_image_info(addr: usize) -> Option<VxCVirtualImageInfo> {
    if let Ok(registry) = VIRTUAL_IMAGES.lock() {
        registry.get(&addr).cloned()
    } else {
        None
    }
}

/// Update virtual image with backing image
pub fn allocate_virtual_image_backing(addr: usize, backing_image: vx_image) -> bool {
    if let Ok(mut registry) = VIRTUAL_IMAGES.lock() {
        if let Some(info) = registry.get_mut(&addr) {
            info.backing_image = Some(backing_image as usize);
            return true;
        }
    }
    false
}

/// Check if an image is virtual
pub fn is_virtual_image(addr: usize) -> bool {
    if let Ok(registry) = VIRTUAL_IMAGES.lock() {
        registry.get(&addr).map(|info| info.is_virtual).unwrap_or(false)
    } else {
        false
    }
}

/// Register a virtual image in the unified registry
fn register_virtual_image(addr: usize, info: VxCVirtualImageInfo) {
    if let Ok(mut registry) = VIRTUAL_IMAGES.lock() {
        registry.insert(addr, info);
    }
}

/// Unregister a virtual image from the unified registry
fn unregister_virtual_image(addr: usize) -> Option<VxCVirtualImageInfo> {
    if let Ok(mut registry) = VIRTUAL_IMAGES.lock() {
        registry.remove(&addr)
    } else {
        None
    }
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

    // Get the context from the graph
    let context = unsafe { vxGetContext(graph as vx_reference) };
    if context.is_null() {
        return std::ptr::null_mut();
    }

    // Note: Virtual images CAN have width/height of 0 - they get dimensions
    // from connected nodes during graph verification
    // VX_DF_IMAGE_VIRT is valid for virtual images

    // Virtual images don't allocate memory immediately
    // For VX_DF_IMAGE_VIRT format, we store 0 dimensions initially
    let (store_width, store_height, store_format) = if color == VX_DF_IMAGE_VIRT {
        (0, 0, VX_DF_IMAGE_VIRT)
    } else {
        (width, height, color)
    };

    let image = Box::new(VxCImage {
        width: store_width,
        height: store_height,
        format: store_format,
        is_virtual: true,
        context, // Store the context from the graph
        data: Arc::new(RwLock::new(Vec::new())),
        mapped_patches: Arc::new(RwLock::new(Vec::new())),
        parent: None,
        is_external_memory: false,
        external_ptrs: Vec::new(),
        external_strides: Vec::new(),
        external_stride_x: Vec::new(),
        external_dim_x: Vec::new(),
        external_dim_y: Vec::new(),
        roi_offsets: Vec::new(),
        is_from_handle: false,
    });

    let image_ptr = Box::into_raw(image) as vx_image;

    // Register virtual image info
    use openvx_core::unified_c_api::VirtualImageInfo;
    register_virtual_image(
        image_ptr as usize,
        VirtualImageInfo {
            width: store_width,
            height: store_height,
            format: store_format as u32,
            is_virtual: true,
            backing_image: None,
        }
    );

    // Register image address in unified registry for type queries (vxQueryReference)
    unsafe {
        register_image(image_ptr as usize);
    }
    
    // Register as valid image for double-free protection
    register_valid_image(image_ptr as usize);

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(image_ptr as usize, std::sync::atomic::AtomicUsize::new(1));
        }
    }

    // Register in REFERENCE_TYPES for type detection
    unsafe {
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(image_ptr as usize, VX_TYPE_IMAGE);
        }
    }

    image_ptr
}

/// Create an image from existing handles
/// 
/// Per OpenVX spec:
///   vx_image vxCreateImageFromHandle(
///       vx_context context,
///       vx_df_image color,
///       vx_imagepatch_addressing_t addrs[],
///       void *ptrs[],
///       vx_enum memory_type)
/// 
/// IMPORTANT: The image created does NOT own the memory - it references external memory
/// provided by the caller. vxReleaseImage will NOT free this memory.
#[no_mangle]
pub extern "C" fn vxCreateImageFromHandle(
    context: vx_context,
    color: vx_df_image,
    addrs: *const vx_imagepatch_addressing_t,
    ptrs: *mut *mut c_void,
    memory_type: vx_enum,
) -> vx_image {
    if context.is_null() || addrs.is_null() || ptrs.is_null() {
        return std::ptr::null_mut();
    }

    // Validate memory type
    if memory_type != VX_MEMORY_TYPE_HOST {
        return std::ptr::null_mut();
    }

    // Determine number of planes from the format
    let num_planes = VxCImage::num_planes(color);
    if num_planes == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        // For planar formats, addrs is an array with one entry per plane
        // The first plane (usually Y) has the full image dimensions
        let addr0 = &*addrs;
        let width = addr0.dim_x;
        let height = addr0.dim_y;

        // Validate dimensions
        if width == 0 || height == 0 {
            return std::ptr::null_mut();
        }

        const MAX_DIMENSION: u32 = 65536;
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return std::ptr::null_mut();
        }

        // Validate all plane pointers and collect them along with stride info
        let mut external_ptrs: Vec<*mut u8> = Vec::with_capacity(num_planes);
        let mut external_strides: Vec<i32> = Vec::with_capacity(num_planes);
        let mut external_stride_x: Vec<i32> = Vec::with_capacity(num_planes);
        let mut external_dim_x: Vec<u32> = Vec::with_capacity(num_planes);
        let mut external_dim_y: Vec<u32> = Vec::with_capacity(num_planes);
        for plane_idx in 0..num_planes {
            let plane_ptr = *ptrs.add(plane_idx);
            if plane_ptr.is_null() {
                return std::ptr::null_mut();
            }
            external_ptrs.push(plane_ptr as *mut u8);
            let plane_addr = &*addrs.add(plane_idx);
            external_strides.push(plane_addr.stride_y);
            external_stride_x.push(plane_addr.stride_x);
            external_dim_x.push(plane_addr.dim_x);
            external_dim_y.push(plane_addr.dim_y);
        }

        // Create an empty Vec - we won't use it for external memory
        // The Vec will have capacity 0 and won't allocate
        let data = Vec::with_capacity(0);

        let image = Box::new(VxCImage {
            width,
            height,
            format: color,
            is_virtual: false,
            context,
            data: Arc::new(RwLock::new(data)),
            mapped_patches: Arc::new(RwLock::new(Vec::new())),
            parent: None,
            is_external_memory: true,
            external_ptrs,
            external_strides,
            external_stride_x,
            external_dim_x,
            external_dim_y,
            roi_offsets: Vec::new(),
            is_from_handle: true,
        });

        let image_ptr = Box::into_raw(image) as vx_image;

        // Register image address in unified registry for type queries (vxQueryReference)
        register_image(image_ptr as usize);
        
        // Register as valid image for double-free protection
        register_valid_image(image_ptr as usize);

        // Register in reference counting
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(image_ptr as usize, std::sync::atomic::AtomicUsize::new(1));
        }

        // Register in REFERENCE_TYPES for type detection
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(image_ptr as usize, VX_TYPE_IMAGE);
        }

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
        is_external_memory: false,
        external_ptrs: Vec::new(),
        external_strides: Vec::new(),
        external_stride_x: Vec::new(),
        external_dim_x: Vec::new(),
        external_dim_y: Vec::new(),
        roi_offsets: Vec::new(),
        is_from_handle: false,
    });

    // Convert to raw pointer
    let image_ptr = Box::into_raw(image) as vx_image;
    
    // Register image address in unified registry for type queries (vxQueryReference)
    unsafe {
        register_image(image_ptr as usize);
    }
    
    // Register as valid image for double-free protection
    register_valid_image(image_ptr as usize);
    
    image_ptr
}

/// Create an image from a channel of another image
/// 
/// According to OpenVX spec, this creates a sub-image from a specific channel
/// of a YUV formatted parent image. Valid channels are:
/// - VX_CHANNEL_Y: Y plane (valid for IYUV, NV12, NV21, YUV4)
/// - VX_CHANNEL_U: U plane (valid for IYUV, YUV4)
/// - VX_CHANNEL_V: V plane (valid for IYUV, YUV4)
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
        match channel {
            0x00009014 /* VX_CHANNEL_Y */ => {
                match source_img.format {
                    VX_DF_IMAGE_YUV4 | VX_DF_IMAGE_IYUV | VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => {}
                    _ => return std::ptr::null_mut(),
                }
            }
            0x00009015 /* VX_CHANNEL_U */ | 0x00009016 /* VX_CHANNEL_V */ => {
                match source_img.format {
                    VX_DF_IMAGE_YUV4 | VX_DF_IMAGE_IYUV => {}
                    _ => return std::ptr::null_mut(),
                }
            }
            _ => return std::ptr::null_mut(),
        }

        let context = source_img.context;
        if context.is_null() {
            return std::ptr::null_mut();
        }

        // Calculate dimensions based on plane
        let (output_width, output_height) = match channel {
            0x00009014 /* VX_CHANNEL_Y */ => (source_img.width, source_img.height),
            0x00009015 | 0x00009016 => {
                match source_img.format {
                    VX_DF_IMAGE_IYUV | VX_DF_IMAGE_NV12 | VX_DF_IMAGE_NV21 => ((source_img.width + 1) / 2, (source_img.height + 1) / 2),
                    VX_DF_IMAGE_YUV4 => (source_img.width, source_img.height),
                    _ => return std::ptr::null_mut(),
                }
            }
            _ => return std::ptr::null_mut(),
        };

        // Determine the plane index for the channel
        let plane_index = match channel {
            0x00009014 => 0, // Y is plane 0
            0x00009015 => 1, // U is plane 1
            0x00009016 => 2, // V is plane 2
            _ => 0,
        };

        let parent_ptr = img as usize;
        
        // For channel images, we create a new image with its own data buffer
        // that contains a copy of just the channel plane data.
        // This avoids complex offset calculations during map/unmap.
        let plane_size = VxCImage::plane_size(output_width, output_height, source_img.format, plane_index);
        let plane_offset = VxCImage::plane_offset(source_img.width, source_img.height, source_img.format, plane_index);
        
        // Create data buffer for the channel image
        let mut channel_data = vec![0u8; plane_size];
        
        // Copy the channel data from parent
        if source_img.is_external_memory {
            // Read from external memory
            let ext_ptr = if plane_index < source_img.external_ptrs.len() {
                source_img.external_ptrs[plane_index]
            } else {
                std::ptr::null_mut()
            };
            if !ext_ptr.is_null() {
                std::ptr::copy_nonoverlapping(ext_ptr, channel_data.as_mut_ptr(), plane_size);
            }
        } else {
            // Read from internal data
            if let Ok(data) = source_img.data.read() {
                if plane_offset + plane_size <= data.len() {
                    channel_data.copy_from_slice(&data[plane_offset..plane_offset + plane_size]);
                }
            }
        }
        
        let channel_image = Box::new(VxCImage {
            width: output_width,
            height: output_height,
            format: VX_DF_IMAGE_U8, // Channel images are always U8
            is_virtual: false,
            context: context,
            data: Arc::new(RwLock::new(channel_data)),
            mapped_patches: Arc::new(RwLock::new(Vec::new())),
            parent: Some(parent_ptr),
            is_external_memory: false,
            external_ptrs: Vec::new(),
        external_strides: Vec::new(),
        external_stride_x: Vec::new(),
        external_dim_x: Vec::new(),
        external_dim_y: Vec::new(),
        roi_offsets: Vec::new(),
        is_from_handle: false,
        });

        let image_ptr = Box::into_raw(channel_image) as vx_image;

        register_image(image_ptr as usize);
        register_valid_image(image_ptr as usize);
        
        image_ptr
    }
}

/// Find the root parent image (the one created from handle or the top-level image)
/// by following the parent chain
unsafe fn find_root_parent(img: &VxCImage) -> (usize, bool) {
    let mut current = img as *const VxCImage;
    let mut addr = current as usize;
    let mut has_from_handle = false;
    
    loop {
        let cur = &*current;
        if cur.is_from_handle {
            has_from_handle = true;
            return (addr, has_from_handle);
        }
        if let Some(parent_addr) = cur.parent {
            current = parent_addr as *const VxCImage;
            addr = parent_addr;
        } else {
            return (addr, has_from_handle);
        }
    }
}

use std::collections::HashSet;

/// Registry to track valid (non-freed) images

static VALID_IMAGES: std::sync::LazyLock<std::sync::Mutex<HashSet<usize>>> = std::sync::LazyLock::new(|| {
    std::sync::Mutex::new(HashSet::new())
});

/// Register an image as valid
fn register_valid_image(addr: usize) {
    if let Ok(mut images) = VALID_IMAGES.lock() {
        images.insert(addr);
    }
}

/// Unregister an image (mark as freed)
fn unregister_valid_image(addr: usize) -> bool {
    if let Ok(mut images) = VALID_IMAGES.lock() {
        images.remove(&addr);
        true
    } else {
        false
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
            let addr = img as usize;
            
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
                    // Not in registry - free it
                    true
                }
            } else {
                false
            };
            
            if should_free {
                // Check if this image was already freed
                if !unregister_valid_image(addr) {
                    return VX_ERROR_INVALID_REFERENCE;
                }
                
                // Unregister from unified registry
                unregister_image(addr);

                // Remove from counts
                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                    counts.remove(&addr);
                }
                
                // Don't remove from types - keep for reference queries

                // Clean up virtual image info if this was a virtual image
                unregister_virtual_image(addr);

                // IMPORTANT: Access image data BEFORE freeing the Box
                // The external_ptrs Vec will be dropped when the Box is freed
                // For external memory images, we don't free the external data
                // but the Vec container itself is properly cleaned up
                let img_data = &mut *(img as *mut VxCImage);
                
                // Clear external_ptrs to drop the Vec properly
                img_data.external_ptrs.clear();
                
                // Free the image - this drops the Box and all its fields
                let _ = Box::from_raw(img as *mut VxCImage);
            }
            
            *image = std::ptr::null_mut();
        } else {
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
                // Accept both vx_uint32 and vx_size sizes for compatibility
                let planes = VxCImage::num_planes(img.format);
                if size == std::mem::size_of::<vx_size>() {
                    *(ptr as *mut vx_size) = planes as vx_size;
                    VX_SUCCESS
                } else if size == std::mem::size_of::<vx_uint32>() {
                    *(ptr as *mut vx_uint32) = planes as vx_uint32;
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
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
            VX_IMAGE_SPACE => {
                // Image space/color space - return VX_COLOR_SPACE_UNDEFINED (0)
                if size == std::mem::size_of::<vx_enum>() {
                    *(ptr as *mut vx_enum) = 0; // VX_COLOR_SPACE_UNDEFINED
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            VX_IMAGE_RANGE => {
                // Image range - return VX_CHANNEL_RANGE_FULL (0)
                if size == std::mem::size_of::<vx_enum>() {
                    *(ptr as *mut vx_enum) = 0; // VX_CHANNEL_RANGE_FULL
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            VX_IMAGE_IS_VIRTUAL => {
                if size != std::mem::size_of::<vx_enum>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                // Return vx_true_e (1) if virtual, vx_false_e (0) if not
                let is_virt = if img.is_virtual { 1 } else { 0 };
                *(ptr as *mut vx_enum) = is_virt;
                VX_SUCCESS
            }
            _ => VX_ERROR_NOT_IMPLEMENTED,
        }
    }
}

/// Set image attributes
#[no_mangle]
pub extern "C" fn vxSetImageAttribute(
    _image: vx_image,
    attribute: vx_enum,
    _ptr: *const c_void,
    _size: vx_size,
) -> vx_status {
    if _image.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    
    // Handle attributes that should succeed
    match attribute {
        VX_IMAGE_SPACE => {
            // Accept setting image space - currently no-op
            VX_SUCCESS
        }
        VX_IMAGE_RANGE => {
            // Accept setting image range - currently no-op
            VX_SUCCESS
        }
        _ => VX_ERROR_NOT_IMPLEMENTED,
    }
}

/// Map image patch for CPU access
///
/// Per OpenVX spec, if the image was created via vxCreateImageFromHandle,
/// the returned address (*ptr) will be the address of the patch in the
/// original pixel buffer provided when the image was created.
/// The returned memory layout will be identical to that of the addressing
/// structure provided when vxCreateImageFromHandle was called.
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

        // Determine plane-specific parameters for planar YUV formats
        let is_planar = VxCImage::is_planar_format(img.format);
        let (plane_width, plane_height) = if is_planar {
            let (pw, ph) = VxCImage::plane_dimensions(img.width, img.height, img.format, plane_index as usize);
            (pw as usize, ph as usize)
        } else {
            (img.width as usize, img.height as usize)
        };

        // Validate the plane_index
        if plane_index as usize >= VxCImage::num_planes(img.format) {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        // Calculate the mapped region, clamped to plane bounds
        let start_x = (rect_ref.start_x as usize).min(plane_width);
        let start_y = (rect_ref.start_y as usize).min(plane_height);
        let end_x = (rect_ref.end_x as usize).min(plane_width);
        let end_y = (rect_ref.end_y as usize).min(plane_height);
        
        let width = end_x.saturating_sub(start_x);
        let height = end_y.saturating_sub(start_y);
        
        if width == 0 || height == 0 {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        // For external memory images (created from handle), return pointer directly into user memory
        if img.is_external_memory {
            // For ROI images, we need to get the root parent's current external pointers
            // because the parent may have had its handles swapped.
            // We follow the parent chain to find the image created from handle.
            let (effective_ptrs, effective_strides, effective_stride_x, roi_start_x, roi_start_y) = {
                if img.parent.is_some() {
                    // This is a sub-image (ROI or channel) - find the root parent
                    let (root_addr, root_is_from_handle) = unsafe { find_root_parent(img) };
                    if root_is_from_handle {
                        let root_img = unsafe { &*(root_addr as *const VxCImage) };
                        let p_ptrs = root_img.external_ptrs.clone();
                        let p_strides = root_img.external_strides.clone();
                        let p_stride_x = root_img.external_stride_x.clone();
                        // Accumulate ROI offsets through the parent chain
                        // We need to sum up all the offsets from the current image up to the root
                        let (mut accum_rx, mut accum_ry) = (0usize, 0usize);
                        let mut current = img as *const VxCImage;
                        loop {
                            let cur = unsafe { &*current };
                            if (plane_index as usize) < cur.roi_offsets.len() {
                                let (rx, ry) = cur.roi_offsets[plane_index as usize];
                                accum_rx += rx;
                                accum_ry += ry;
                            }
                            if let Some(parent_addr) = cur.parent {
                                current = parent_addr as *const VxCImage;
                                // Stop at the root (from-handle image)
                                if unsafe { &*current }.is_from_handle {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        (p_ptrs, p_strides, p_stride_x, accum_rx, accum_ry)
                    } else {
                        // Root parent is not from handle - use our own pointers
                        (img.external_ptrs.clone(), img.external_strides.clone(), img.external_stride_x.clone(), 0, 0)
                    }
                } else {
                    // Not a sub-image - use our own pointers
                    (img.external_ptrs.clone(), img.external_strides.clone(), img.external_stride_x.clone(), 0, 0)
                }
            };

            let ext_ptr = if (plane_index as usize) < effective_ptrs.len() {
                effective_ptrs[plane_index as usize]
            } else {
                return VX_ERROR_INVALID_PARAMETERS;
            };
            if ext_ptr.is_null() {
                // After vxSwapImageHandle with NULL new_ptrs, the image has no
                // backing memory. Per OpenVX spec, vxMapImagePatch should return
                // VX_ERROR_NO_MEMORY in this case.
                use openvx_core::c_api::VX_ERROR_NO_MEMORY;
                return VX_ERROR_NO_MEMORY;
            }

            // Use the original strides from the image handle
            let ext_stride_y = if (plane_index as usize) < effective_strides.len() {
                effective_strides[plane_index as usize]
            } else {
                // Fallback: compute stride from plane dimensions
                plane_width as vx_int32
            };
            let ext_stride_x = if (plane_index as usize) < effective_stride_x.len() {
                effective_stride_x[plane_index as usize]
            } else {
                // For planar formats, stride_x is typically 1 byte per pixel
                // For packed formats, it's bytes_per_pixel
                if is_planar { 1 } else { VxCImage::bytes_per_pixel(img.format) as vx_int32 }
            };

            // Calculate the byte offset to the start of the patch in the plane
            // Include the ROI offset for ROI images
            let offset_bytes = (roi_start_y + start_y) as isize * ext_stride_y as isize 
                + (roi_start_x + start_x) as isize * ext_stride_x as isize;
            let patch_ptr = ext_ptr.offset(offset_bytes) as *mut c_void;

            // Fill addressing structure with ORIGINAL strides from handle
            (*addr).dim_x = width as vx_uint32;
            (*addr).dim_y = height as vx_uint32;
            (*addr).stride_x = ext_stride_x;
            (*addr).stride_y = ext_stride_y;
            (*addr).step_x = 1;
            (*addr).step_y = 1u16;
            (*addr).scale_x = 1024; // VX_SCALE_UNITY
            (*addr).scale_y = 1024; // VX_SCALE_UNITY
            *ptr = patch_ptr;

            // For external memory, we don't need to copy data.
            // The user directly accesses the memory.
            // We still need a map_id for unmap tracking.
            // Store a minimal entry so vxUnmapImagePatch knows this was mapped.
            let map_id_val = if let Ok(mut patches) = img.mapped_patches.write() {
                let id = patches.len() + 1;
                // For external memory, we store empty data since no copy was made
                // The (offset, stride_y, plane_index, mapped_width) are still stored for potential unmap use
                patches.push((id, Vec::new(), usage, 0, ext_stride_y as usize, plane_index, width as u32));
                id
            } else {
                return VX_ERROR_INVALID_REFERENCE;
            };
            *map_id = map_id_val;

            return VX_SUCCESS;
        }

        // For regular (internal memory) images, copy the data to a separate buffer
        // Calculate stride and offset based on format and plane
        let bpp = if is_planar {
            1usize // For planar formats, each plane is 1 byte per pixel (luma/chroma)
        } else {
            VxCImage::bytes_per_pixel(img.format)
        };
        
        let stride_y = plane_width * bpp;
        let plane_offset = if is_planar {
            VxCImage::plane_offset(img.width, img.height, img.format, plane_index as usize)
        } else {
            0
        };

        let offset = plane_offset + start_y * stride_y + start_x * bpp;

        // Create a copy of the data for the mapped patch
        let mapped_stride_y = width * bpp;
        let patch_size = height * mapped_stride_y;
        let mut patch_data = vec![0u8; patch_size];
        
        // Copy data row by row from internal data buffer
        let data_guard = match img.data.read() {
            Ok(guard) => guard,
            Err(_) => return VX_ERROR_INVALID_REFERENCE,
        };
        for y in 0..height {
            let src_start = offset + y * stride_y;
            let dst_start = y * mapped_stride_y;
            if src_start + width * bpp <= data_guard.len() {
                patch_data[dst_start..dst_start + width * bpp]
                    .copy_from_slice(&data_guard[src_start..src_start + width * bpp]);
            }
        }
        drop(data_guard);
        
        let store_stride_y = stride_y;
        
        let map_id_val = if let Ok(mut patches) = img.mapped_patches.write() {
            let id = patches.len() + 1;
            patches.push((id, patch_data, usage, offset, store_stride_y, plane_index, width as u32));
            id
        } else {
            return VX_ERROR_INVALID_REFERENCE;
        };

        // Fill addressing structure
        (*addr).dim_x = width as vx_uint32;
        (*addr).dim_y = height as vx_uint32;
        (*addr).stride_x = bpp as vx_int32;
        (*addr).stride_y = mapped_stride_y as vx_int32;
        (*addr).step_x = 1;
        (*addr).step_y = 1u16;
        (*addr).scale_x = 1024; // VX_SCALE_UNITY
        (*addr).scale_y = 1024; // VX_SCALE_UNITY
        *map_id = map_id_val;
        
        // Return pointer to the STORED patch data (not the local variable)
        if let Ok(patches) = img.mapped_patches.read() {
            if let Some(patch) = patches.iter().find(|(id, _, _, _, _, _, _)| *id == map_id_val) {
                *ptr = patch.1.as_ptr() as *mut c_void;
            }
        }
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
        if let Some(pos) = patches.iter().position(|p| {
            let (id, _, _, _, _, _, _) = p;
            *id == map_id
        }) {
            let patch_tuple = patches.remove(pos);
            let (_, patch_data, usage, offset, stride_y, plane_index, mapped_width) = patch_tuple;
            
            // For external memory images, no copy-back needed since the user
            // directly modified the external memory buffer.
            // The patch_data will be empty for external memory maps.
            if img.is_external_memory && patch_data.is_empty() {
                // External memory: direct pointer was returned, nothing to copy back
                return VX_SUCCESS;
            }
            
            // If write access, copy data back
            if usage == VX_WRITE_ONLY || usage == VX_READ_AND_WRITE {
                let is_planar = VxCImage::is_planar_format(img.format);
                let bpp = if is_planar { 1 } else { VxCImage::bytes_per_pixel(img.format) };
                let width = mapped_width as usize;
                let mapped_stride = width * bpp;
                let height = if mapped_stride > 0 { patch_data.len() / mapped_stride } else { 0 };
                
                if img.is_external_memory {
                    // Write back to external memory
                    let ext_ptr = if (plane_index as usize) < img.external_ptrs.len() {
                        img.external_ptrs[plane_index as usize]
                    } else {
                        return VX_ERROR_INVALID_PARAMETERS;
                    };
                    if !ext_ptr.is_null() {
                        for y in 0..height {
                            let src_start = y * mapped_stride;
                            let dst_start = offset + y * stride_y;
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    patch_data.as_ptr().add(src_start),
                                    ext_ptr.add(dst_start),
                                    width * bpp,
                                );
                            }
                        }
                    }
                } else {
                    // Write back to internal data buffer
                    if let Ok(mut data) = img.data.write() {
                        for y in 0..height {
                            let src_start = y * mapped_stride;
                            let dst_start = offset + y * stride_y;
                            if dst_start + width * bpp <= data.len() && src_start + width * bpp <= patch_data.len() {
                                data[dst_start..dst_start + width * bpp]
                                    .copy_from_slice(&patch_data[src_start..src_start + width * bpp]);
                            }
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
        let (bpp, stride_y, plane_offset) = if img.is_external_memory {
            // For external memory images, use the strides from vxCreateImageFromHandle
            let ext_bpp = if (plane_index as usize) < img.external_stride_x.len() && img.external_stride_x[plane_index as usize] > 0 {
                img.external_stride_x[plane_index as usize] as usize
            } else if is_planar {
                1usize
            } else {
                VxCImage::bytes_per_pixel(img.format)
            };
            let ext_stride_y = if (plane_index as usize) < img.external_strides.len() {
                img.external_strides[plane_index as usize] as usize
            } else {
                plane_width * ext_bpp
            };
            // No plane_offset for external memory - each plane has its own pointer
            (ext_bpp, ext_stride_y, 0usize)
        } else if is_planar {
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
                if img.is_external_memory {
                    let ext_ptr = if (plane_index as usize) < img.external_ptrs.len() {
                        img.external_ptrs[plane_index as usize]
                    } else {
                        return VX_ERROR_INVALID_PARAMETERS;
                    };
                    if !ext_ptr.is_null() {
                        for y in 0..height {
                            let src_start = offset + y * stride_y;
                            let dst_start = y * addr.stride_y as usize;
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    ext_ptr.add(src_start),
                                    (user_ptr as *mut u8).add(dst_start),
                                    width * bpp,
                                );
                            }
                        }
                    }
                } else {
                    if let Ok(data) = img.data.read() {
                        for y in 0..height {
                            let src_start = offset + y * stride_y;
                            let dst_start = y * addr.stride_y as usize;
                            if src_start + width * bpp <= data.len() {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        data.as_ptr().add(src_start),
                                        (user_ptr as *mut u8).add(dst_start),
                                        width * bpp
                                    );
                                }
                            }
                        }
                    }
                }
            }
            VX_WRITE_ONLY => {
                // Copy from user buffer to image
                if img.is_external_memory {
                    let ext_ptr = if (plane_index as usize) < img.external_ptrs.len() {
                        img.external_ptrs[plane_index as usize]
                    } else {
                        return VX_ERROR_INVALID_PARAMETERS;
                    };
                    if !ext_ptr.is_null() {
                        for y in 0..height {
                            let dst_start = offset + y * stride_y;
                            let src_start = y * addr.stride_y as usize;
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    (user_ptr as *const u8).add(src_start),
                                    ext_ptr.add(dst_start),
                                    width * bpp,
                                );
                            }
                        }
                    }
                } else {
                    if let Ok(mut data) = img.data.write() {
                        for y in 0..height {
                            let dst_start = offset + y * stride_y;
                            let src_start = y * addr.stride_y as usize;
                            if dst_start + width * bpp <= data.len() {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        (user_ptr as *const u8).add(src_start),
                                        data.as_mut_ptr().add(dst_start),
                                        width * bpp
                                    );
                                }
                            }
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
///
/// Per OpenVX spec:
///   vx_status vxSwapImageHandle(vx_image image, void* const new_ptrs[], void* prev_ptrs[], vx_size num_planes)
///
/// - If new_ptrs is non-NULL, the image's plane pointers are replaced with new_ptrs[].
/// - If new_ptrs is NULL, the image reclaims its pointers (sets them to NULL).
/// - If prev_ptrs is non-NULL, the previous pointers are written to prev_ptrs[].
/// - If prev_ptrs is NULL, previous pointers are not returned (but still swapped).
#[no_mangle]
pub extern "C" fn vxSwapImageHandle(
    image: vx_image,
    new_ptrs: *const *mut c_void,
    prev_ptrs: *mut *mut c_void,
    num_planes: vx_size,
) -> vx_status {
    if image.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let img = unsafe { &mut *(image as *mut VxCImage) };

    // Per OpenVX spec, this only works for images created directly from vxCreateImageFromHandle
    // ROI images and channel images should fail even if they reference external memory
    if !img.is_from_handle {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let num_expected_planes = VxCImage::num_planes(img.format);
    if num_planes as usize != num_expected_planes {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    unsafe {
        for plane_idx in 0..num_planes {
            // Return old pointers if prev_ptrs is provided
            if !prev_ptrs.is_null() {
                if plane_idx < img.external_ptrs.len() {
                    *prev_ptrs.add(plane_idx) = img.external_ptrs[plane_idx] as *mut c_void;
                } else {
                    *prev_ptrs.add(plane_idx) = std::ptr::null_mut();
                }
            }
            // Set new pointers if new_ptrs is provided
            if !new_ptrs.is_null() {
                let new_ptr = *new_ptrs.add(plane_idx);
                if plane_idx < img.external_ptrs.len() {
                    img.external_ptrs[plane_idx] = new_ptr as *mut u8;
                }
            } else {
                // new_ptrs is NULL: reclaim the image's pointers (set to NULL)
                // This means the image no longer has valid external memory
                if plane_idx < img.external_ptrs.len() {
                    img.external_ptrs[plane_idx] = std::ptr::null_mut();
                }
            }
        }
    }

    VX_SUCCESS
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

        let context = source_img.context;
        if context.is_null() {
            return std::ptr::null_mut();
        }

        // Per OpenVX spec, ROI images share memory with the parent.
        // For external memory images, the ROI also shares external memory.
        // When the parent's handles are swapped, the ROI sees the new data.

        let is_planar = VxCImage::is_planar_format(source_img.format);
        let num_planes = VxCImage::num_planes(source_img.format);

        // Calculate per-plane ROI offsets and dimensions
        let mut roi_external_ptrs = Vec::new();
        let mut roi_external_strides = Vec::new();
        let mut roi_external_stride_x = Vec::new();
        let mut roi_external_dim_x = Vec::new();
        let mut roi_external_dim_y = Vec::new();

        let mut roi_offsets: Vec<(usize, usize)> = Vec::new();

        for plane_idx in 0..num_planes {
            let (plane_width, plane_height) = if is_planar {
                let (pw, ph) = VxCImage::plane_dimensions(source_img.width, source_img.height, source_img.format, plane_idx);
                (pw, ph)
            } else {
                (source_img.width, source_img.height)
            };

            let subsamp_x = if is_planar && plane_idx > 0 {
                (source_img.width as usize) / (plane_width as usize)
            } else {
                1
            };
            let subsamp_y = if is_planar && plane_idx > 0 {
                (source_img.height as usize) / (plane_height as usize)
            } else {
                1
            };

            let roi_plane_start_x = (rect_ref.start_x as usize) / subsamp_x;
            let roi_plane_start_y = (rect_ref.start_y as usize) / subsamp_y;
            let roi_plane_end_x = (rect_ref.end_x as usize + subsamp_x - 1) / subsamp_x;
            let roi_plane_end_y = (rect_ref.end_y as usize + subsamp_y - 1) / subsamp_y;
            let roi_plane_width = roi_plane_end_x.saturating_sub(roi_plane_start_x);
            let roi_plane_height = roi_plane_end_y.saturating_sub(roi_plane_start_y);

            roi_external_dim_x.push(roi_plane_width as u32);
            roi_external_dim_y.push(roi_plane_height as u32);
            roi_offsets.push((roi_plane_start_x, roi_plane_start_y));

            if source_img.is_external_memory {
                // For external memory, compute the offset to the ROI start within the plane
                let ext_stride_y = if plane_idx < source_img.external_strides.len() {
                    source_img.external_strides[plane_idx]
                } else {
                    plane_width as i32
                };
                let ext_stride_x = if plane_idx < source_img.external_stride_x.len() {
                    source_img.external_stride_x[plane_idx]
                } else {
                    if is_planar { 1 } else { VxCImage::bytes_per_pixel(source_img.format) as i32 }
                };

                roi_external_strides.push(ext_stride_y);
                roi_external_stride_x.push(ext_stride_x);

                // Calculate the offset from the plane's base pointer to the ROI start
                let offset_bytes = roi_plane_start_y as isize * ext_stride_y as isize
                    + roi_plane_start_x as isize * ext_stride_x as isize;

                // Get the parent's plane pointer and offset it
                let parent_plane_ptr = if plane_idx < source_img.external_ptrs.len() {
                    source_img.external_ptrs[plane_idx]
                } else {
                    std::ptr::null_mut()
                };
                let roi_ptr = if parent_plane_ptr.is_null() {
                    std::ptr::null_mut()
                } else {
                    parent_plane_ptr.offset(offset_bytes)
                };
                roi_external_ptrs.push(roi_ptr);
            } else {
                // For internal memory, we still share the parent's Arc<RwLock<Vec<u8>>
                // The offset is stored separately and used during map operations
                roi_external_ptrs.push(std::ptr::null_mut());
                let bpp = if is_planar { 1i32 } else { VxCImage::bytes_per_pixel(source_img.format) as i32 };
                roi_external_stride_x.push(bpp);
                roi_external_strides.push(plane_width as i32 * bpp);
            }
        }

        let roi_image = Box::new(VxCImage {
            width: roi_width,
            height: roi_height,
            format: source_img.format,
            is_virtual: false,
            context,
            // Share the parent's data buffer - changes to parent data are visible in ROI
            data: Arc::clone(&source_img.data),
            mapped_patches: Arc::new(RwLock::new(Vec::new())),
            parent: Some(img as usize),
            // If parent has external memory, ROI also uses external memory
            // (with offset pointers to the ROI region)
            is_external_memory: source_img.is_external_memory,
            external_ptrs: roi_external_ptrs,
            external_strides: roi_external_strides,
            external_stride_x: roi_external_stride_x,
            external_dim_x: roi_external_dim_x,
            external_dim_y: roi_external_dim_y,
            roi_offsets,
            is_from_handle: false, // ROI images should NOT support vxSwapImageHandle
        });

        let image_ptr = Box::into_raw(roi_image) as vx_image;

        // Register image address in unified registry for type queries (vxQueryReference)
        register_image(image_ptr as usize);

        // Register as valid image for double-free protection
        register_valid_image(image_ptr as usize);

        // Register in reference counting
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(image_ptr as usize, std::sync::atomic::AtomicUsize::new(1));
        }

        // Register in REFERENCE_TYPES for type detection
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(image_ptr as usize, VX_TYPE_IMAGE);
        }

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
/// If a rectangle is specified, only that region is cloned. If rect is NULL,
/// the entire image is cloned.
///
/// # Arguments
/// * `image` - The source image to clone
/// * `rect` - Optional rectangle defining the region to clone (NULL for entire image)
///
/// # Returns
/// A new vx_image handle that is a deep copy of the source (or region), or null on error
#[no_mangle]
pub extern "C" fn vxCloneImage(
    image: vx_image,
    rect: *const vx_rectangle_t,
) -> vx_image {
    // Validate image parameter
    if image.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let source_img = &*(image as *const VxCImage);

        // Get context from the source image
        let context = vxGetContext(image as vx_reference);
        if context.is_null() {
            return std::ptr::null_mut();
        }

        // Get source image properties
        let src_width = source_img.width;
        let src_height = source_img.height;
        let format = source_img.format;

        // Determine clone dimensions and offset
        let (clone_width, clone_height, offset_x, offset_y) = if rect.is_null() {
            // Clone entire image
            (src_width, src_height, 0, 0)
        } else {
            // Clone specified region
            let rect_ref = &*rect;
            
            // Validate rectangle bounds
            if rect_ref.start_x >= rect_ref.end_x || rect_ref.start_y >= rect_ref.end_y {
                return std::ptr::null_mut();
            }
            if rect_ref.end_x > src_width || rect_ref.end_y > src_height {
                return std::ptr::null_mut();
            }
            
            let width = rect_ref.end_x - rect_ref.start_x;
            let height = rect_ref.end_y - rect_ref.start_y;
            (width, height, rect_ref.start_x, rect_ref.start_y)
        };

        // Handle zero dimensions
        if clone_width == 0 || clone_height == 0 {
            return std::ptr::null_mut();
        }

        // Handle virtual images - they don't have allocated data
        // Per OpenVX spec, cloning a virtual image creates a regular image
        if source_img.is_virtual && source_img.data.read().unwrap().is_empty() {
            // Virtual image without data - just create regular image with clone dimensions
            return vxCreateImage(context, clone_width, clone_height, format);
        }

        // Create the destination image
        let mut dest_image = vxCreateImage(context, clone_width, clone_height, format);
        if dest_image.is_null() {
            return std::ptr::null_mut();
        }

        // Get the destination image data structure
        let dest_img = &mut *(dest_image as *mut VxCImage);

        // Calculate bytes per pixel for the format
        let bpp = VxCImage::bytes_per_pixel(format);
        let src_stride = src_width as usize * bpp;
        let dest_stride = clone_width as usize * bpp;

        // Copy data from source to destination
        if let Ok(source_data) = source_img.data.read() {
            if let Ok(mut dest_data) = dest_img.data.write() {
                // Ensure destination has enough space
                if dest_data.len() >= clone_height as usize * dest_stride {
                    // Copy pixel data row by row
                    for y in 0..clone_height as usize {
                        let src_y = (offset_y as usize) + y;
                        let src_offset = src_y * src_stride + (offset_x as usize) * bpp;
                        let dest_offset = y * dest_stride;
                        let row_bytes = dest_stride;
                        
                        if src_offset + row_bytes <= source_data.len() {
                            dest_data[dest_offset..dest_offset + row_bytes]
                                .copy_from_slice(&source_data[src_offset..src_offset + row_bytes]);
                        }
                    }
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
            // Create a regular image with data copy (clone entire image)
            vxCloneImage(source, std::ptr::null())
        }
    }
}

// ============================================================================
// Pyramid Operations
// ============================================================================

use openvx_core::unified_c_api::{VxCPyramid, VX_TYPE_PYRAMID, vx_pyramid};

/// Create a pyramid object
/// 
/// Creates a pyramid with the specified number of levels, scale factor,
/// and base image dimensions.
/// 
/// # Arguments
/// * `context` - The OpenVX context
/// * `num_levels` - The number of levels in the pyramid
/// * `scale` - The scale factor between levels (typically VX_SCALE_PYRAMID_HALF = 0.5)
/// * `width` - The width of the base level (level 0)
/// * `height` - The height of the base level (level 0)
/// * `format` - The image format for all levels
/// 
/// # Returns
/// A pyramid handle on success, or NULL on failure
#[no_mangle]
pub extern "C" fn vxCreatePyramid(
    context: vx_context,
    num_levels: vx_size,
    scale: vx_float32,
    width: vx_uint32,
    height: vx_uint32,
    format: vx_df_image,
) -> vx_pyramid {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if num_levels == 0 || width == 0 || height == 0 {
        return std::ptr::null_mut();
    }
    
    // Validate scale factor (must be positive and less than 1.0)
    if scale <= 0.0 || scale >= 1.0 {
        return std::ptr::null_mut();
    }

    let levels_usize = num_levels as usize;
    let mut level_images: Vec<usize> = Vec::with_capacity(levels_usize);

    // Create images for each level
    for level in 0..levels_usize {
        // Calculate dimensions for this level using scale^level
        let level_scale = scale.powi(level as i32);
        let level_width = (width as f32 * level_scale) as vx_uint32;
        let level_height = (height as f32 * level_scale) as vx_uint32;
        
        // Ensure minimum dimensions of 1x1
        let level_width = level_width.max(1);
        let level_height = level_height.max(1);
        
        let img = vxCreateImage(context, level_width, level_height, format);
        if img.is_null() {
            // Failed to create image - clean up already created images
            for existing_img in level_images.iter_mut() {
                let mut img_ptr = *existing_img as *mut VxImage;
                vxReleaseImage(&mut img_ptr);
            }
            return std::ptr::null_mut();
        }
        
        level_images.push(img as usize);
    }

    // Create the pyramid structure
    let pyramid = Box::new(VxCPyramid {
        context: context as usize,
        num_levels: levels_usize,
        scale,
        width,
        height,
        format,
        levels: level_images,
    });

    let pyramid_ptr = Box::into_raw(pyramid) as vx_pyramid;

    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(pyramid_ptr as usize, std::sync::atomic::AtomicUsize::new(1));
        }
    }

    // Register in REFERENCE_TYPES for type detection
    unsafe {
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(pyramid_ptr as usize, VX_TYPE_PYRAMID);
        }
    }

    pyramid_ptr
}

/// Release a pyramid object
/// 
/// Releases the pyramid and all its level images.
/// 
/// # Arguments
/// * `pyramid` - Pointer to the pyramid handle
/// 
/// # Returns
/// VX_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn vxReleasePyramid(pyramid: *mut vx_pyramid) -> vx_status {
    if pyramid.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let pyr = *pyramid;
        if !pyr.is_null() {
            let addr = pyr as usize;
            
            // Get the pyramid struct
            let pyramid_data = &mut *(pyr as *mut VxCPyramid);
            
            // Release all level images
            for level_img in pyramid_data.levels.iter_mut() {
                let mut img = *level_img as vx_image;
                vxReleaseImage(&mut img);
            }

            // Remove from reference counts and types
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.remove(&addr);
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.remove(&addr);
            }

            // Free the pyramid
            let _ = Box::from_raw(pyr as *mut VxCPyramid);
            *pyramid = std::ptr::null_mut();
        }
    }

    VX_SUCCESS
}

/// Get a level image from a pyramid
/// 
/// Returns a reference to the image at the specified level.
/// The returned image reference must not be released - it is owned by the pyramid.
/// 
/// # Arguments
/// * `pyramid` - The pyramid handle
/// * `index` - The level index (0 to num_levels-1)
/// 
/// # Returns
/// An image handle on success, or NULL on failure
#[no_mangle]
pub extern "C" fn vxGetPyramidLevel(pyramid: vx_pyramid, index: vx_uint32) -> vx_image {
    if pyramid.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let pyramid_data = &*(pyramid as *const VxCPyramid);
        
        // Validate index
        let idx = index as usize;
        if idx >= pyramid_data.num_levels {
            return std::ptr::null_mut();
        }
        
        // Return the level image (not a copy, just the reference)
        pyramid_data.levels.get(idx).map(|&img| img as vx_image).unwrap_or(std::ptr::null_mut())
    }
}

/// Query pyramid attributes
/// 
/// Queries various attributes of a pyramid object.
/// 
/// # Arguments
/// * `pyramid` - The pyramid handle
/// * `attribute` - The attribute to query
/// * `ptr` - Pointer to the memory to store the result
/// * `size` - Size of the memory pointed to by ptr
/// 
/// # Returns
/// VX_SUCCESS on success, error code on failure
#[no_mangle]
pub extern "C" fn vxQueryPyramid(
    pyramid: vx_pyramid,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if pyramid.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let pyr = unsafe { &*(pyramid as *const VxCPyramid) };

    unsafe {
        // VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_PYRAMID) = 0x80900
        match attribute {
            // VX_PYRAMID_LEVELS = 0x80900
            0x80900 | 0x00 => {
                if size != std::mem::size_of::<vx_size>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_size) = pyr.num_levels as vx_size;
                VX_SUCCESS
            }
            // VX_PYRAMID_SCALE = 0x80901
            0x80901 | 0x01 => {
                if size != std::mem::size_of::<vx_float32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_float32) = pyr.scale;
                VX_SUCCESS
            }
            // VX_PYRAMID_FORMAT = 0x80902
            0x80902 | 0x02 => {
                if size != std::mem::size_of::<vx_df_image>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_df_image) = pyr.format;
                VX_SUCCESS
            }
            // VX_PYRAMID_WIDTH = 0x80903
            0x80903 | 0x03 => {
                if size != std::mem::size_of::<vx_uint32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_uint32) = pyr.width;
                VX_SUCCESS
            }
            // VX_PYRAMID_HEIGHT = 0x80904
            0x80904 | 0x04 => {
                if size != std::mem::size_of::<vx_uint32>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_uint32) = pyr.height;
                VX_SUCCESS
            }
            _ => VX_ERROR_NOT_IMPLEMENTED,
        }
    }
}

/// Create a virtual pyramid
#[no_mangle]
pub extern "C" fn vxCreateVirtualPyramid(
    graph: vx_graph,
    num_levels: vx_size,
    scale: vx_float32,
    width: vx_uint32,
    height: vx_uint32,
    format: vx_df_image,
) -> vx_pyramid {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    if num_levels == 0 {
        return std::ptr::null_mut();
    }
    
    // Get context from graph
    let context = unsafe { vxGetContext(graph as vx_reference) };
    if context.is_null() {
        return std::ptr::null_mut();
    }
    
    // Virtual pyramids can have 0 width/height/format - they're resolved during graph verification
    // when connected to output-producing nodes
    let actual_width = if width == 0 { 1 } else { width };
    let actual_height = if height == 0 { 1 } else { height };
    let actual_format = if format == VX_DF_IMAGE_VIRT { VX_DF_IMAGE_U8 } else { format };
    
    // Virtual pyramids are created like regular ones with placeholder dimensions
    vxCreatePyramid(context, num_levels, scale, actual_width, actual_height, actual_format)
}
