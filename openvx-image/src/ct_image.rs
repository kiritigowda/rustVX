//! CT Image Framework - Conformance Test Image utilities for OpenVX
//!
//! This module implements the CT_Image framework used by the OpenVX Conformance Test Suite.
//! CT_Image provides a convenient way to allocate, manipulate, and convert images for testing.

#![allow(non_camel_case_types)]

use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openvx_core::c_api::{
    vx_context, vx_image, vx_df_image, vx_enum, vx_status,
    vx_rectangle_t, vx_imagepatch_addressing_t, vx_map_id,
    vx_uint32, vx_uint64, vx_size,
    VX_SUCCESS, VX_ERROR_INVALID_PARAMETERS, VX_ERROR_INVALID_REFERENCE,
    VX_READ_ONLY, VX_WRITE_ONLY, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST,
    VX_IMAGE_WIDTH, VX_IMAGE_HEIGHT, VX_IMAGE_FORMAT,
};

// Import VxCImage from core
use openvx_core::unified_c_api::VxCImage;

// VX_DF_IMAGE format constants
const VX_DF_IMAGE_U8: u32 = 0x20080100;
const VX_DF_IMAGE_U16: u32 = 0x20100100;
const VX_DF_IMAGE_S16: u32 = 0x20100200;
const VX_DF_IMAGE_U32: u32 = 0x20200100;
const VX_DF_IMAGE_S32: u32 = 0x20200200;
const VX_DF_IMAGE_RGB: u32 = 0x21000300;
const VX_DF_IMAGE_RGBA: u32 = 0x21000400;
const VX_DF_IMAGE_RGBX: u32 = 0x21010400;
const VX_DF_IMAGE_NV12: u32 = 0x3231564E;
const VX_DF_IMAGE_NV21: u32 = 0x3132564E;
const VX_DF_IMAGE_IYUV: u32 = 0x56555949;
const VX_DF_IMAGE_YUV4: u32 = 0x34555659;
const VX_DF_IMAGE_UYVY: u32 = 0x59565955;
const VX_DF_IMAGE_YUYV: u32 = 0x56595559;

/// CT Rectangle structure for ROI operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CT_Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// CT Image copy direction
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CT_ImageCopyDirection {
    CopyCTImageToVXImage = 0,
    CopyVXImageToCTImage,
}

/// CT Image data union - supports various pixel formats
#[repr(C)]
pub union CT_ImageData {
    pub y: *mut u8,
    pub u16: *mut u16,
    pub s16: *mut i16,
    pub u32: *mut u32,
    pub s32: *mut i32,
    pub rgb: *mut CT_RGBPixel,
    pub yuv: *mut CT_YUVPixel,
    pub rgbx: *mut CT_RGBXPixel,
    pub yuyv: *mut CT_YUYVPixel,
    pub uyvy: *mut CT_UYVYPixel,
}

/// RGB pixel structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CT_RGBPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// YUV pixel structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CT_YUVPixel {
    pub y: u8,
    pub u: u8,
    pub v: u8,
}

/// RGBX pixel structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CT_RGBXPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub x: u8,
}

/// YUYV pixel structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CT_YUYVPixel {
    pub y0: u8,
    pub u: u8,
    pub y1: u8,
    pub v: u8,
}

/// UYVY pixel structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CT_UYVYPixel {
    pub u: u8,
    pub y0: u8,
    pub v: u8,
    pub y1: u8,
}

/// CT Image header structure
#[repr(C)]
pub struct CT_ImageHdr {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: vx_df_image,
    pub roi: CT_Rect,
    pub data: CT_ImageData,
    // Private area
    data_begin_: *mut u8,
    data_size_: usize,  // Size of data buffer for proper deallocation
    refcount_: Arc<AtomicUsize>,
}

/// CT Image handle
pub type CT_Image = *mut CT_ImageHdr;

/// Get the number of channels for a format
#[no_mangle]
pub extern "C" fn ct_channels(format: vx_df_image) -> i32 {
    match format as u32 {
        0x20080100 => 1, // VX_DF_IMAGE_U8
        0x20100100 | 0x20100200 => 1, // VX_DF_IMAGE_U16, VX_DF_IMAGE_S16
        0x20200100 | 0x20200200 => 1, // VX_DF_IMAGE_U32, VX_DF_IMAGE_S32
        0x21000300 => 3, // VX_DF_IMAGE_RGB
        0x21000400 | 0x21010400 => 4, // VX_DF_IMAGE_RGBA, VX_DF_IMAGE_RGBX
        0x3231564E | 0x3132564E => 3, // VX_DF_IMAGE_NV12, VX_DF_IMAGE_NV21
        0x56555949 => 3, // VX_DF_IMAGE_IYUV
        0x59565955 | 0x56595559 => 2, // VX_DF_IMAGE_UYVY, VX_DF_IMAGE_YUYV
        0x34555659 => 3, // VX_DF_IMAGE_YUV4
        _ => 1,
    }
}

/// Get stride in bytes
#[no_mangle]
pub extern "C" fn ct_stride_bytes(image: CT_Image) -> u32 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        img.stride * ct_image_bits_per_pixel(img.format) / 8
    }
}

/// Get bits per pixel for a format
#[no_mangle]
pub extern "C" fn ct_image_bits_per_pixel(format: vx_df_image) -> u32 {
    match format as u32 {
        0x20080100 => 8, // VX_DF_IMAGE_U8
        0x20100100 | 0x20100200 => 16, // VX_DF_IMAGE_U16, VX_DF_IMAGE_S16
        0x20200100 | 0x20200200 => 32, // VX_DF_IMAGE_U32, VX_DF_IMAGE_S32
        0x21000300 => 24, // VX_DF_IMAGE_RGB
        0x21000400 | 0x21010400 => 32, // VX_DF_IMAGE_RGBA, VX_DF_IMAGE_RGBX
        0x3231564E | 0x3132564E => 12, // VX_DF_IMAGE_NV12, VX_DF_IMAGE_NV21 (average)
        0x56555949 => 12, // VX_DF_IMAGE_IYUV (average)
        0x59565955 | 0x56595559 => 16, // VX_DF_IMAGE_UYVY, VX_DF_IMAGE_YUYV
        0x34555659 => 24, // VX_DF_IMAGE_YUV4
        _ => 8,
    }
}

/// Get number of planes for a format
#[no_mangle]
pub extern "C" fn ct_get_num_planes(format: vx_df_image) -> u32 {
    match format as u32 {
        0x3231564E | 0x3132564E => 2, // NV12, NV21
        0x56555949 => 3, // IYUV
        0x34555659 => 3, // YUV4
        _ => 1,
    }
}

/// Get plane base address
#[no_mangle]
pub extern "C" fn ct_image_get_plane_base(img: CT_Image, plane: i32) -> *mut u8 {
    if img.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let image = &*img;
        let base = image.data_begin_;
        if base.is_null() {
            return std::ptr::null_mut();
        }
        
        // Calculate offset based on format and plane
        let offset = match (image.format as u32, plane) {
            (0x3231564E | 0x3132564E, 0) => 0, // NV12/NV21 Y plane
            (0x3231564E | 0x3132564E, 1) => {
                // NV12/NV21 UV plane - after Y
                let y_size = image.stride * image.height;
                y_size as usize
            }
            (0x56555949, 0) => 0, // IYUV Y plane
            (0x56555949, 1) => {
                // IYUV U plane
                let y_size = image.stride * image.height;
                let uv_size = (image.stride / 2) * (image.height / 2);
                (y_size + uv_size) as usize
            }
            (0x56555949, 2) => {
                // IYUV V plane
                let y_size = image.stride * image.height;
                let uv_size = (image.stride / 2) * (image.height / 2);
                (y_size + 2 * uv_size) as usize
            }
            (0x34555659, 0) => 0, // YUV4 Y plane
            (0x34555659, 1) => {
                // YUV4 U plane
                let plane_size = image.stride * image.height;
                plane_size as usize
            }
            (0x34555659, 2) => {
                // YUV4 V plane
                let plane_size = image.stride * image.height;
                (2 * plane_size) as usize
            }
            _ => 0, // Single plane
        };
        
        base.add(offset)
    }
}

/// Get channel step X
#[no_mangle]
pub extern "C" fn ct_image_get_channel_step_x(image: CT_Image, channel: vx_enum) -> i32 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        match (img.format as u32, channel) {
            // For planar formats, Y/U/V channels have different step patterns
            (0x56555949, 4) => 1, // Y channel in IYUV
            (0x56555949, 5 | 6) => 1, // U/V channels in IYUV
            (0x3231564E | 0x3132564E, 4) => 1, // Y channel in NV12/NV21
            (0x3231564E | 0x3132564E, 5 | 6) => 2, // U/V in NV (interleaved)
            _ => 1,
        }
    }
}

/// Get channel step Y
#[no_mangle]
pub extern "C" fn ct_image_get_channel_step_y(image: CT_Image, channel: vx_enum) -> i32 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        let stride = img.stride as i32;
        match (img.format as u32, channel) {
            // For planar formats
            (0x56555949, 4) => stride, // Y channel
            (0x56555949, 5 | 6) => stride / 2, // U/V channels (subsampled)
            (0x3231564E | 0x3132564E, 4) => stride, // Y channel
            (0x3231564E | 0x3132564E, 5 | 6) => stride, // UV plane stride
            _ => stride,
        }
    }
}

/// Get channel subsampling X
#[no_mangle]
pub extern "C" fn ct_image_get_channel_subsampling_x(image: CT_Image, channel: vx_enum) -> i32 {
    if image.is_null() {
        return 1;
    }
    unsafe {
        let img = &*image;
        match (img.format as u32, channel) {
            (0x56555949, 5 | 6) => 2, // U/V in IYUV (4:2:0)
            (0x3231564E | 0x3132564E, 5 | 6) => 2, // U/V in NV
            _ => 1,
        }
    }
}

/// Get channel subsampling Y
#[no_mangle]
pub extern "C" fn ct_image_get_channel_subsampling_y(image: CT_Image, channel: vx_enum) -> i32 {
    if image.is_null() {
        return 1;
    }
    unsafe {
        let img = &*image;
        match (img.format as u32, channel) {
            (0x56555949, 5 | 6) => 2, // U/V in IYUV (4:2:0)
            (0x3231564E | 0x3132564E, 5 | 6) => 2, // U/V in NV
            _ => 1,
        }
    }
}

/// Get channel plane index
#[no_mangle]
pub extern "C" fn ct_image_get_channel_plane(image: CT_Image, channel: vx_enum) -> i32 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        match (img.format as u32, channel) {
            (0x56555949, 4) => 0, // Y
            (0x56555949, 5) => 1, // U
            (0x56555949, 6) => 2, // V
            (0x3231564E | 0x3132564E, 4) => 0, // Y
            (0x3231564E | 0x3132564E, 5 | 6) => 1, // UV plane
            _ => 0,
        }
    }
}

/// Get channel component index within a pixel
#[no_mangle]
pub extern "C" fn ct_image_get_channel_component(image: CT_Image, channel: vx_enum) -> i32 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        match (img.format as u32, channel) {
            (0x3231564E, 5) => 0, // NV12 U
            (0x3231564E, 6) => 1, // NV12 V
            (0x3132564E, 5) => 1, // NV21 U
            (0x3132564E, 6) => 0, // NV21 V
            _ => 0,
        }
    }
}

/// Get number of channels
#[no_mangle]
pub extern "C" fn ct_get_num_channels(format: vx_df_image) -> i32 {
    ct_channels(format)
}

/// Allocate a CT Image
#[no_mangle]
pub extern "C" fn ct_allocate_image(width: u32, height: u32, format: vx_df_image) -> CT_Image {
    if width == 0 || height == 0 {
        return std::ptr::null_mut();
    }
    
    // Use VxCImage::calculate_size for accurate planar format sizing
    let size = VxCImage::calculate_size(width, height, format);
    if size == 0 {
        return std::ptr::null_mut();
    }
    
    let stride = width; // In pixels (for CT image compatibility)
    
    // Allocate data buffer using std::alloc
    let layout = Layout::from_size_align(size, 1).unwrap();
    let data_ptr = unsafe { alloc(layout) };
    if data_ptr.is_null() {
        return std::ptr::null_mut();
    }
    // Initialize to zero
    unsafe {
        std::ptr::write_bytes(data_ptr, 0, size);
    }
    
    // Create image header
    let image = Box::new(CT_ImageHdr {
        width,
        height,
        stride,
        format,
        roi: CT_Rect { x: 0, y: 0, width, height },
        data: CT_ImageData { y: data_ptr },
        data_begin_: data_ptr,
        data_size_: size,
        refcount_: Arc::new(AtomicUsize::new(1)),
    });
    
    Box::into_raw(image)
}

/// Allocate CT Image with custom stride and data
#[no_mangle]
pub extern "C" fn ct_allocate_image_hdr(
    width: u32,
    height: u32,
    stride: u32,
    format: vx_df_image,
    data: *mut c_void,
) -> CT_Image {
    if width == 0 || height == 0 || data.is_null() {
        return std::ptr::null_mut();
    }
    
    // Calculate size based on format using VxCImage
    let size = VxCImage::calculate_size(width, height, format);
    
    let image = Box::new(CT_ImageHdr {
        width,
        height,
        stride,
        format,
        roi: CT_Rect { x: 0, y: 0, width, height },
        data: CT_ImageData { y: data as *mut u8 },
        data_begin_: data as *mut u8,
        data_size_: size,
        refcount_: Arc::new(AtomicUsize::new(1)),
    });
    
    Box::into_raw(image)
}

/// Get ROI from CT Image
#[no_mangle]
pub extern "C" fn ct_get_image_roi(img: CT_Image, roi: CT_Rect) -> CT_Image {
    if img.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let source = &*img;
        
        // Calculate data offset for ROI
        let bpp = ct_image_bits_per_pixel(source.format) / 8;
        let offset = (roi.y * source.stride + roi.x) * bpp as u32;
        let roi_data = if source.data_begin_.is_null() {
            std::ptr::null_mut()
        } else {
            source.data_begin_.add(offset as usize)
        };
        
        // ROI images share the parent data, use parent's size
        let roi_image = Box::new(CT_ImageHdr {
            width: roi.width,
            height: roi.height,
            stride: source.stride,
            format: source.format,
            roi: CT_Rect { x: roi.x, y: roi.y, width: roi.width, height: roi.height },
            data: CT_ImageData { y: roi_data },
            data_begin_: source.data_begin_,
            // For ROI images, the data size is shared with parent, but we track it separately
            // The actual allocated size is still source.data_size_
            data_size_: 0, // Mark as 0 to indicate this is a sub-image that shouldn't free the data
            refcount_: Arc::clone(&source.refcount_),
        });
        
        // Increment reference count
        source.refcount_.fetch_add(1, Ordering::SeqCst);
        
        Box::into_raw(roi_image)
    }
}

/// Get ROI with individual parameters
#[no_mangle]
pub extern "C" fn ct_get_image_roi_(
    img: CT_Image,
    x_start: u32,
    y_start: u32,
    width: u32,
    height: u32,
) -> CT_Image {
    ct_get_image_roi(img, CT_Rect { x: x_start, y: y_start, width, height })
}

/// Adjust ROI
#[no_mangle]
pub extern "C" fn ct_adjust_roi(img: CT_Image, left: i32, top: i32, right: i32, bottom: i32) {
    if img.is_null() {
        return;
    }
    unsafe {
        let image = &mut *img;
        let new_x = (image.roi.x as i32 + left).max(0) as u32;
        let new_y = (image.roi.y as i32 + top).max(0) as u32;
        let new_width = (image.roi.width as i32 - left - right).max(0) as u32;
        let new_height = (image.roi.height as i32 - top - bottom).max(0) as u32;
        
        image.roi.x = new_x;
        image.roi.y = new_y;
        image.roi.width = new_width;
        image.roi.height = new_height;
        image.width = new_width;
        image.height = new_height;
        
        // Recalculate data pointer
        let bpp = ct_image_bits_per_pixel(image.format) / 8;
        let offset = (new_y * image.stride + new_x) * bpp as u32;
        image.data.y = if image.data_begin_.is_null() {
            std::ptr::null_mut()
        } else {
            image.data_begin_.add(offset as usize)
        };
    }
}

/// Get data pointer for 1-bit images
#[no_mangle]
pub extern "C" fn ct_image_data_ptr_1u(image: CT_Image, x: u32, y: u32) -> *mut u8 {
    if image.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let img = &*image;
        let byte_offset = (y * img.stride + x) / 8;
        img.data_begin_.add(byte_offset as usize)
    }
}

/// Get data with replicate border for 1-bit
#[no_mangle]
pub extern "C" fn ct_image_data_replicate_1u(image: CT_Image, x: i32, y: i32) -> u8 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        let clamped_x = x.max(0).min(img.width as i32 - 1) as u32;
        let clamped_y = y.max(0).min(img.height as i32 - 1) as u32;
        let byte_offset = (clamped_y * img.stride + clamped_x) / 8;
        let bit_offset = (clamped_y * img.stride + clamped_x) % 8;
        let byte = *img.data_begin_.add(byte_offset as usize);
        (byte >> bit_offset) & 1
    }
}

/// Get data with constant border for 1-bit
#[no_mangle]
pub extern "C" fn ct_image_data_constant_1u(
    image: CT_Image,
    x: i32,
    y: i32,
    constant_value: vx_enum,
) -> u8 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        if x < 0 || x >= img.width as i32 || y < 0 || y >= img.height as i32 {
            return if constant_value != 0 { 1 } else { 0 };
        }
        ct_image_data_replicate_1u(image, x, y)
    }
}

/// Get data pointer for 8-bit images
#[no_mangle]
pub extern "C" fn ct_image_data_ptr_8u(image: CT_Image, x: u32, y: u32) -> *mut u8 {
    if image.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let img = &*image;
        img.data.y.add((y * img.stride + x) as usize)
    }
}

/// Get data with replicate border for 8-bit
#[no_mangle]
pub extern "C" fn ct_image_data_replicate_8u(image: CT_Image, x: i32, y: i32) -> u8 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        let clamped_x = x.max(0).min(img.width as i32 - 1) as u32;
        let clamped_y = y.max(0).min(img.height as i32 - 1) as u32;
        *img.data.y.add((clamped_y * img.stride + clamped_x) as usize)
    }
}

/// Get data with constant border for 8-bit
#[no_mangle]
pub extern "C" fn ct_image_data_constant_8u(
    image: CT_Image,
    x: i32,
    y: i32,
    constant_value: u32,
) -> u8 {
    if image.is_null() {
        return 0;
    }
    unsafe {
        let img = &*image;
        if x < 0 || x >= img.width as i32 || y < 0 || y >= img.height as i32 {
            return (constant_value & 0xFF) as u8;
        }
        ct_image_data_replicate_8u(image, x, y)
    }
}

/// Free CT Image
#[no_mangle]
pub extern "C" fn ct_free_image(image: CT_Image) {
    if image.is_null() {
        return;
    }
    unsafe {
        let img = &*image;
        let new_count = img.refcount_.fetch_sub(1, Ordering::SeqCst) - 1;
        
        if new_count == 0 {
            // Last reference, free the data
            if !img.data_begin_.is_null() && img.data_size_ > 0 {
                // Deallocate using std::alloc
                let layout = Layout::from_size_align(img.data_size_, 1).unwrap();
                dealloc(img.data_begin_, layout);
            }
        }
        
        // Free the header
        let _ = Box::from_raw(image);
    }
}

/// Convert CT Image to VX Image
/// Note: This creates a copy of the data
#[no_mangle]
pub extern "C" fn ct_image_to_vx_image(
    ctimg: CT_Image,
    context: vx_context,
) -> vx_image {
    use crate::c_api::*;
    
    if ctimg.is_null() || context.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let img = &*ctimg;
        let width = img.width;
        let height = img.height;
        let format = img.format;
        
        // Map CT format to VX format
        let vx_format = match format {
            0x20080100 => VX_DF_IMAGE_U8,
            0x21000300 => VX_DF_IMAGE_RGB,
            0x21000400 => VX_DF_IMAGE_RGBA,
            0x21010400 => VX_DF_IMAGE_RGBX,
            0x3231564E => VX_DF_IMAGE_NV12,
            0x3132564E => VX_DF_IMAGE_NV21,
            0x56555949 => VX_DF_IMAGE_IYUV,
            0x59565955 => VX_DF_IMAGE_UYVY,
            0x56595559 => VX_DF_IMAGE_YUYV,
            0x34555659 => VX_DF_IMAGE_YUV4,
            _ => return std::ptr::null_mut(),
        };
        
        // Create vx_image
        let vximg = vxCreateImage(context, width, height, vx_format);
        if vximg.is_null() {
            return std::ptr::null_mut();
        }
        
        // For planar formats, copy all planes
        let is_planar = VxCImage::is_planar_format(vx_format);
        let num_planes = VxCImage::num_planes(vx_format);
        
        if is_planar {
            // Copy each plane separately
            for plane_idx in 0..num_planes {
                let mut addr = std::mem::zeroed::<vx_imagepatch_addressing_t>();
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                
                // Get plane dimensions
                let (plane_width, plane_height) = VxCImage::plane_dimensions(width, height, vx_format, plane_idx);
                let rect = vx_rectangle_t { 
                    start_x: 0, start_y: 0, 
                    end_x: plane_width, end_y: plane_height 
                };
                let mut map_id: vx_map_id = 0;
                
                let status = vxMapImagePatch(vximg, &rect, plane_idx as vx_uint32, 
                                             &mut map_id, &mut addr, &mut ptr, 
                                             VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
                if status != VX_SUCCESS {
                    let mut img_ptr = vximg;
                    vxReleaseImage(&mut img_ptr);
                    return std::ptr::null_mut();
                }
                
                // Copy data row by row
                if !ptr.is_null() && !img.data_begin_.is_null() {
                    let plane_offset = VxCImage::plane_offset(width, height, vx_format, plane_idx);
                    
                    // For planar formats, each pixel is 1 byte
                    let row_size = plane_width as usize;  // Bytes to copy per row
                    let dst_stride = addr.stride_y as usize;
                    
                    for y in 0..plane_height as usize {
                        let src_row = img.data_begin_.add(plane_offset + y * plane_width as usize);
                        let dst_row = (ptr as *mut u8).add(y * dst_stride);
                        if row_size > 0 && plane_offset + y * plane_width as usize + row_size <= img.data_size_ {
                            std::ptr::copy_nonoverlapping(src_row, dst_row, row_size);
                        }
                    }
                }
                
                vxUnmapImagePatch(vximg, map_id);
            }
        } else {
            // Single plane format - use vxMapImagePatch
            let mut addr = std::mem::zeroed::<vx_imagepatch_addressing_t>();
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let rect = vx_rectangle_t { start_x: 0, start_y: 0, end_x: width, end_y: height };
            let mut map_id: vx_map_id = 0;
            
            let status = vxMapImagePatch(vximg, &rect, 0, &mut map_id, &mut addr, &mut ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
            if status != VX_SUCCESS {
                let mut img_ptr = vximg;
                vxReleaseImage(&mut img_ptr);
                return std::ptr::null_mut();
            }
            
            // Copy data row by row (accounting for stride)
            if !ptr.is_null() && !img.data_begin_.is_null() {
                let bpp = VxCImage::bytes_per_pixel(vx_format);
                let row_size = width as usize * bpp;  // Bytes to copy per row
                let src_stride = img.stride as usize * bpp;  // CT stride in bytes
                let dst_stride = addr.stride_y as usize;
                
                for y in 0..height as usize {
                    let src_row = img.data_begin_.add(y * src_stride);
                    let dst_row = (ptr as *mut u8).add(y * dst_stride);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, row_size);
                }
            }
            
            vxUnmapImagePatch(vximg, map_id);
        }
        
        vximg
    }
}

/// Convert VX Image to CT Image
/// Note: This creates a copy of the data
#[no_mangle]
pub extern "C" fn ct_image_from_vx_image(vximg: vx_image) -> CT_Image {
    use crate::c_api::*;
    
    if vximg.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        // Query image attributes
        let mut width: vx_uint32 = 0;
        let mut height: vx_uint32 = 0;
        let mut format: vx_df_image = 0;
        
        if vxQueryImage(vximg, VX_IMAGE_WIDTH, &mut width as *mut _ as *mut std::ffi::c_void, 
                       std::mem::size_of::<vx_uint32>()) != VX_SUCCESS {
            return std::ptr::null_mut();
        }
        
        if vxQueryImage(vximg, VX_IMAGE_HEIGHT, &mut height as *mut _ as *mut std::ffi::c_void,
                       std::mem::size_of::<vx_uint32>()) != VX_SUCCESS {
            return std::ptr::null_mut();
        }
        
        if vxQueryImage(vximg, VX_IMAGE_FORMAT, &mut format as *mut _ as *mut std::ffi::c_void,
                       std::mem::size_of::<vx_df_image>()) != VX_SUCCESS {
            return std::ptr::null_mut();
        }
        
        // Create CT_Image
        let ctimg = ct_allocate_image(width, height, format);
        if ctimg.is_null() {
            return std::ptr::null_mut();
        }
        
        let img = &mut *ctimg;
        
        // For planar formats, copy all planes
        let is_planar = VxCImage::is_planar_format(format);
        let num_planes = VxCImage::num_planes(format);
        
        if is_planar {
            // Copy each plane separately
            let mut total_copied = 0usize;
            for plane_idx in 0..num_planes {
                let mut addr = std::mem::zeroed::<vx_imagepatch_addressing_t>();
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                
                // Get plane dimensions
                let (plane_width, plane_height) = VxCImage::plane_dimensions(width, height, format, plane_idx);
                let rect = vx_rectangle_t { 
                    start_x: 0, start_y: 0, 
                    end_x: plane_width, end_y: plane_height 
                };
                let mut map_id: vx_map_id = 0;
                
                let status = vxMapImagePatch(vximg, &rect, plane_idx as vx_uint32, &mut map_id, &mut addr, &mut ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
                if status != VX_SUCCESS {
                    ct_free_image(ctimg);
                    return std::ptr::null_mut();
                }
                
                // Copy data row by row
                if !ptr.is_null() && !img.data_begin_.is_null() {
                    let plane_offset = VxCImage::plane_offset(width, height, format, plane_idx);
                    let plane_size = VxCImage::plane_size(width, height, format, plane_idx);
                    
                    // For planar formats, each pixel is 1 byte
                    let row_size = plane_width as usize;  // Bytes to copy per row
                    let src_stride = addr.stride_y as usize;
                    
                    for y in 0..plane_height as usize {
                        let dst_row = img.data_begin_.add(plane_offset + y * plane_width as usize);
                        let src_row = (ptr as *const u8).add(y * src_stride);
                        if row_size > 0 && plane_offset + y * plane_width as usize + row_size <= img.data_size_ {
                            std::ptr::copy_nonoverlapping(src_row, dst_row, row_size);
                        }
                    }
                    total_copied += plane_size;
                }
                
                vxUnmapImagePatch(vximg, map_id);
            }
        } else {
            // Single plane format - use vxMapImagePatch
            let mut addr = std::mem::zeroed::<vx_imagepatch_addressing_t>();
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let rect = vx_rectangle_t { start_x: 0, start_y: 0, end_x: width, end_y: height };
            let mut map_id: vx_map_id = 0;
            
            let status = vxMapImagePatch(vximg, &rect, 0, &mut map_id, &mut addr, &mut ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
            if status != VX_SUCCESS {
                ct_free_image(ctimg);
                return std::ptr::null_mut();
            }
            
            // Copy data row by row (accounting for stride)
            if !ptr.is_null() && !img.data_begin_.is_null() {
                let bpp = VxCImage::bytes_per_pixel(format);
                let row_size = width as usize * bpp;  // Bytes to copy per row
                let dst_stride = img.stride as usize * bpp;  // CT stride in bytes
                let src_stride = addr.stride_y as usize;
                
                for y in 0..height as usize {
                    let dst_row = img.data_begin_.add(y * dst_stride);
                    let src_row = (ptr as *const u8).add(y * src_stride);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, row_size);
                }
            }
            
            vxUnmapImagePatch(vximg, map_id);
        }
        
        ctimg
    }
}

/// Copy between CT Image and VX Image
#[no_mangle]
pub extern "C" fn ct_image_copy(
    ctimg: CT_Image,
    vximg: vx_image,
    dir: CT_ImageCopyDirection,
) {
    if ctimg.is_null() || vximg.is_null() {
        return;
    }
    
    // For now, stub implementation - would need proper VxCImage access
    // This would copy data between CT_Image and vx_image structures
    let _ = dir; // silence warning
}

/// Clone a CT Image
#[no_mangle]
pub extern "C" fn ct_image_create_clone(image: CT_Image) -> CT_Image {
    if image.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let img = &*image;
        let clone = ct_allocate_image(img.width, img.height, img.format);
        if clone.is_null() {
            return std::ptr::null_mut();
        }
        
        let cl = &mut *clone;
        cl.roi = img.roi;
        
        // Copy data - use the stored data_size_ for accurate copying
        if !img.data_begin_.is_null() && !cl.data_begin_.is_null() && img.data_size_ > 0 {
            std::ptr::copy_nonoverlapping(img.data_begin_, cl.data_begin_, img.data_size_.min(cl.data_size_));
        }
        
        clone
    }
}

/// Fill CT Image with random values
#[no_mangle]
pub extern "C" fn ct_fill_ct_image_random(
    image: CT_Image,
    seed: *mut u64,
    a: i32,
    b: i32,
) {
    if image.is_null() {
        return;
    }
    
    unsafe {
        let img = &mut *image;
        let mut rng = if seed.is_null() {
            123456789u64
        } else {
            *seed
        };
        
        let min_val = a.min(b) as u8;
        let max_val = a.max(b) as u8;
        let range = max_val.saturating_sub(min_val);
        
        for y in 0..img.height {
            for x in 0..img.width {
                // Simple LCG random number generator
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let val = if range > 0 {
                    min_val + ((rng % range as u64) as u8)
                } else {
                    min_val
                };
                
                if !img.data.y.is_null() {
                    *img.data.y.add((y * img.stride + x) as usize) = val;
                }
            }
        }
        
        if !seed.is_null() {
            *seed = rng;
        }
    }
}

/// Allocate random CT Image
#[no_mangle]
pub extern "C" fn ct_allocate_ct_image_random(
    width: u32,
    height: u32,
    format: vx_df_image,
    rng: *mut u64,
    a: i32,
    b: i32,
) -> CT_Image {
    let image = ct_allocate_image(width, height, format);
    if !image.is_null() {
        ct_fill_ct_image_random(image, rng, a, b);
    }
    image
}

/// Convert U8 CT Image to U1 CT Image
#[no_mangle]
pub extern "C" fn U8_ct_image_to_U1_ct_image(img_in: CT_Image, img_out: CT_Image) {
    if img_in.is_null() || img_out.is_null() {
        return;
    }
    unsafe {
        let input = &*img_in;
        let output = &mut *img_out;
        
        for y in 0..input.height.min(output.height) {
            for x in 0..input.width.min(output.width) {
                let val = *input.data.y.add((y * input.stride + x) as usize);
                let bit = if val > 127 { 1 } else { 0 };
                
                // Pack into output
                let byte_idx = (y * output.stride + x) / 8;
                let bit_idx = (y * output.stride + x) % 8;
                if byte_idx < output.stride * output.height / 8 {
                    let byte = output.data.y.add(byte_idx as usize);
                    if bit == 1 {
                        *byte |= 1 << bit_idx;
                    } else {
                        *byte &= !(1 << bit_idx);
                    }
                }
            }
        }
    }
}

/// Convert U1 CT Image to U8 CT Image
#[no_mangle]
pub extern "C" fn U1_ct_image_to_U8_ct_image(img_in: CT_Image, img_out: CT_Image) {
    if img_in.is_null() || img_out.is_null() {
        return;
    }
    unsafe {
        let input = &*img_in;
        let output = &mut *img_out;
        
        for y in 0..input.height.min(output.height) {
            for x in 0..input.width.min(output.width) {
                let byte_idx = (y * input.stride + x) / 8;
                let bit_idx = (y * input.stride + x) % 8;
                let bit = if byte_idx < input.stride * input.height / 8 {
                    (*input.data.y.add(byte_idx as usize) >> bit_idx) & 1
                } else {
                    0
                };
                
                let val = if bit == 1 { 255 } else { 0 };
                *output.data.y.add((y * output.stride + x) as usize) = val;
            }
        }
    }
}

/// Apply threshold to U8 CT Image
#[no_mangle]
pub extern "C" fn threshold_U8_ct_image(img: CT_Image, thresh: u8) {
    if img.is_null() {
        return;
    }
    unsafe {
        let image = &mut *img;
        for y in 0..image.height {
            for x in 0..image.width {
                let idx = (y * image.stride + x) as usize;
                let val = *image.data.y.add(idx);
                *image.data.y.add(idx) = if val > thresh { 255 } else { 0 };
            }
        }
    }
}

/// Assert two CT images are equal
#[no_mangle]
pub extern "C" fn ct_assert_eq_ctimage(
    expected: CT_Image,
    actual: CT_Image,
    threshold: u32,
) -> i32 {
    if expected.is_null() || actual.is_null() {
        return 0;
    }
    
    unsafe {
        let exp = &*expected;
        let act = &*actual;
        
        if exp.width != act.width || exp.height != act.height {
            return 0;
        }
        
        for y in 0..exp.height {
            for x in 0..exp.width {
                let idx = (y * exp.stride + x) as usize;
                let exp_val = *exp.data.y.add(idx) as i32;
                let act_val = *act.data.y.add(idx) as i32;
                let diff = (exp_val - act_val).abs() as u32;
                
                if diff > threshold {
                    return 0;
                }
            }
        }
        
        1
    }
}

/// Dump image info
#[no_mangle]
pub extern "C" fn ct_dump_image_info(image: CT_Image) {
    if image.is_null() {
        println!("CT_Image: null");
        return;
    }
    unsafe {
        let img = &*image;
        println!("CT_Image: {}x{} stride={} format=0x{:08X}",
            img.width, img.height, img.stride, img.format as u32);
        println!("  ROI: ({}, {}) {}x{}",
            img.roi.x, img.roi.y, img.roi.width, img.roi.height);
    }
}

/// Read image from file (stub)
#[no_mangle]
pub extern "C" fn ct_read_image(file_name: *const i8, dcn: i32) -> CT_Image {
    // Stub implementation - would need image format parsing
    std::ptr::null_mut()
}

/// Write image to file (stub)
#[no_mangle]
pub extern "C" fn ct_write_image(file_name: *const i8, image: CT_Image) {
    // Stub implementation
}

/// Read rect from S32 image
#[no_mangle]
pub extern "C" fn ct_image_read_rect_S32(
    img: CT_Image,
    dst: *mut i32,
    sx: i32,
    sy: i32,
    ex: i32,
    ey: i32,
    _border: i32, // vx_border_t
) -> i32 {
    if img.is_null() || dst.is_null() {
        return -1;
    }
    
    unsafe {
        let image = &*img;
        let width = (ex - sx) as u32;
        let height = (ey - sy) as u32;
        
        for y in 0..height {
            for x in 0..width {
                let src_x = (sx + x as i32).max(0).min(image.width as i32 - 1) as u32;
                let src_y = (sy + y as i32).max(0).min(image.height as i32 - 1) as u32;
                let src_idx = (src_y * image.stride + src_x) as usize;
                let dst_idx = (y * width + x) as usize;
                
                if image.format as u32 == 0x20200100 { // VX_DF_IMAGE_S32
                    *dst.add(dst_idx) = *image.data.s32.add(src_idx);
                } else {
                    *dst.add(dst_idx) = *image.data.y.add(src_idx) as i32;
                }
            }
        }
        
        0
    }
}
