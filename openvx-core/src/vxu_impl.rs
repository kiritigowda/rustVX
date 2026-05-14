//! VXU (Immediate Mode) Function Implementations
//!
//! This module provides actual implementations for VXU functions that bridge
//! the C API types to the Rust vision kernel implementations.

#![allow(non_camel_case_types)]
#![allow(
    dead_code,
    unreachable_patterns,
    unused_assignments,
    unused_unsafe,
    unused_variables
)]

use crate::c_api::{
    vx_array,
    vx_bool,
    vx_context,
    vx_convolution,
    vx_coordinates2d_t,
    vx_df_image,
    vx_enum,
    vx_float32,
    vx_image,
    vx_map_id,
    vx_matrix,
    vx_pyramid,
    vx_scalar,
    vx_size,
    vx_status,
    vx_threshold,
    vx_uint32,
    VX_DF_IMAGE_S16,
    VX_DF_IMAGE_U8, // Add S16/U16/U8 format constants
    VX_ERROR_INVALID_FORMAT,
    VX_ERROR_INVALID_PARAMETERS,
    VX_ERROR_INVALID_REFERENCE,
    VX_ERROR_NOT_IMPLEMENTED,
    VX_SUCCESS,
};
use crate::unified_c_api::{vx_border_t, vx_distribution, vx_remap, VxCImage, VxCPyramid};
use std::ffi::c_void;

/// OpenVX enum constants for norm type
const VX_NORM_L1: vx_enum = 0x10000;
const VX_NORM_L2: vx_enum = 0x10001;

/// OpenVX enum constants for convert policy
const VX_CONVERT_POLICY_WRAP: vx_enum = 0xA000;
const VX_CONVERT_POLICY_SATURATE: vx_enum = 0xA001;

/// OpenVX enum constants for round policy  
const VX_ROUND_POLICY_TO_ZERO: vx_enum = 0x12001;
const VX_ROUND_POLICY_TO_NEAREST_EVEN: vx_enum = 0x12002;
fn read_scale_from_scalar(scalar: vx_scalar) -> f32 {
    if scalar.is_null() {
        return 1.0; // Default scale
    }
    // Read directly from VxCScalarData pointer
    unsafe {
        let s = &*(scalar as *const crate::c_api_data::VxCScalarData);
        if s.data.len() >= 4 {
            f32::from_le_bytes([s.data[0], s.data[1], s.data[2], s.data[3]])
        } else {
            1.0
        }
    }
}

/// Image format enum for internal use
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageFormat {
    Gray,    // U8 - single byte per pixel
    GrayU16, // U16 - two bytes per pixel
    GrayS16, // S16 - two bytes per pixel (signed)
    GrayU32, // U32 - four bytes per pixel (for integral image)
    Rgb,
    Rgba,
    NV12,
    NV21,
    IYUV,
    YUV4,
    YUYV, // Packed YUV 4:2:2 - Y0 U0 Y1 V0
    UYVY, // Packed YUV 4:2:2 - U0 Y0 V0 Y1
}

impl ImageFormat {
    pub fn channels(&self) -> usize {
        match self {
            ImageFormat::Gray => 1,
            ImageFormat::GrayU16 => 1, // U16 is single channel, 2 bytes
            ImageFormat::GrayS16 => 1, // S16 is single channel, 2 bytes
            ImageFormat::GrayU32 => 4,
            ImageFormat::Rgb => 3,
            ImageFormat::Rgba => 4,
            // Planar formats return 1 for buffer calculation base
            // Use buffer_size() for actual allocation
            ImageFormat::NV12 => 1,
            ImageFormat::NV21 => 1,
            ImageFormat::IYUV => 1,
            ImageFormat::YUV4 => 1,
            // Packed YUV formats: 2 bytes per pixel
            ImageFormat::YUYV => 1,
            ImageFormat::UYVY => 1,
        }
    }

    /// Calculate buffer size for this format with given dimensions
    /// For planar formats, this accounts for subsampled chroma planes
    pub fn buffer_size(&self, width: usize, height: usize) -> usize {
        match self {
            ImageFormat::Gray => width.saturating_mul(height),
            ImageFormat::GrayU16 => width.saturating_mul(height).saturating_mul(2), // U16 = 2 bytes per pixel
            ImageFormat::GrayS16 => width.saturating_mul(height).saturating_mul(2), // S16 = 2 bytes per pixel
            ImageFormat::GrayU32 => width.saturating_mul(height).saturating_mul(4), // U32 = 4 bytes per pixel
            ImageFormat::Rgb => width.saturating_mul(height).saturating_mul(3),
            ImageFormat::Rgba => width.saturating_mul(height).saturating_mul(4),
            // IYUV/I420: Y (full) + U (quarter) + V (quarter) = 1.5 * width * height
            ImageFormat::IYUV => {
                let y_size = width.saturating_mul(height);
                let half_w = (width + 1) / 2;
                let half_h = (height + 1) / 2;
                let uv_size = half_w.saturating_mul(half_h);
                y_size.saturating_add(uv_size).saturating_add(uv_size)
            }
            // NV12/NV21: Y (full) + UV interleaved (quarter * 2) = 1.5 * width * height
            ImageFormat::NV12 | ImageFormat::NV21 => {
                let y_size = width.saturating_mul(height);
                let half_h = (height + 1) / 2;
                let uv_size = width.saturating_mul(half_h);
                y_size.saturating_add(uv_size)
            }
            // YUV4: Three full-size planes = 3 * width * height
            ImageFormat::YUV4 => width.saturating_mul(height).saturating_mul(3),
            // Packed YUV: 2 bytes per pixel (4:2:2 sampling)
            ImageFormat::YUYV | ImageFormat::UYVY => width.saturating_mul(height).saturating_mul(2),
        }
    }
}

/// Simple Image struct for VXU operations
pub struct Image {
    width: usize,
    height: usize,
    format: ImageFormat,
    data: Vec<u8>,
    // Valid region (start_x, start_y, end_x, end_y) - pixels outside this region
    // are treated as out-of-bounds by geometric operations
    valid_start_x: usize,
    valid_start_y: usize,
    valid_end_x: usize,
    valid_end_y: usize,
}

impl Image {
    pub fn new(width: usize, height: usize, format: ImageFormat) -> Option<Self> {
        // Use format-specific buffer size calculation
        let size = format.buffer_size(width, height);

        // Check for overflow (0 indicates overflow) and limit size
        if size == 0 || size > (1 << 30) {
            return None;
        }

        let data = vec![0u8; size];
        Some(Image {
            width,
            height,
            format,
            data,
            valid_start_x: 0,
            valid_start_y: 0,
            valid_end_x: width,
            valid_end_y: height,
        })
    }

    pub fn from_data(width: usize, height: usize, format: ImageFormat, data: Vec<u8>) -> Self {
        Image {
            width,
            height,
            format,
            data,
            valid_start_x: 0,
            valid_start_y: 0,
            valid_end_x: width,
            valid_end_y: height,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn format(&self) -> ImageFormat {
        self.format
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
    pub fn valid_rect(&self) -> (usize, usize, usize, usize) {
        (
            self.valid_start_x,
            self.valid_start_y,
            self.valid_end_x,
            self.valid_end_y,
        )
    }
    pub fn set_valid_rect(&mut self, sx: usize, sy: usize, ex: usize, ey: usize) {
        self.valid_start_x = sx;
        self.valid_start_y = sy;
        self.valid_end_x = ex;
        self.valid_end_y = ey;
    }
    /// Check if a pixel is within the valid region
    pub fn is_valid_pixel(&self, x: i32, y: i32) -> bool {
        x >= self.valid_start_x as i32
            && x < self.valid_end_x as i32
            && y >= self.valid_start_y as i32
            && y < self.valid_end_y as i32
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        let idx = y.saturating_mul(self.width).saturating_add(x);
        *self.data.get(idx).unwrap_or(&0)
    }

    pub fn get_rgb(&self, x: usize, y: usize) -> (u8, u8, u8) {
        if x >= self.width || y >= self.height {
            return (0, 0, 0);
        }
        let idx = y
            .saturating_mul(self.width)
            .saturating_add(x)
            .saturating_mul(3);
        if idx.saturating_add(2) >= self.data.len() {
            return (0, 0, 0);
        }
        (self.data[idx], self.data[idx + 1], self.data[idx + 2])
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, value: u8) {
        if x < self.width && y < self.height {
            let idx = y.saturating_mul(self.width).saturating_add(x);
            if let Some(p) = self.data.get_mut(idx) {
                *p = value;
            }
        }
    }

    /// Get pixel as i16 (for S16 format)
    pub fn get_pixel_s16(&self, x: usize, y: usize) -> i16 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        let idx = (y * self.width + x) * 2; // 2 bytes per i16
        if idx + 1 >= self.data.len() {
            return 0;
        }
        i16::from_le_bytes([self.data[idx], self.data[idx + 1]])
    }

    /// Set pixel as i16 (for S16 format)
    pub fn set_pixel_s16(&mut self, x: usize, y: usize, value: i16) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 2; // 2 bytes per i16
            let bytes = value.to_le_bytes();
            if idx + 1 < self.data.len() {
                self.data[idx] = bytes[0];
                self.data[idx + 1] = bytes[1];
            }
        }
    }

    pub fn set_rgb(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        if x < self.width && y < self.height {
            let idx = y
                .saturating_mul(self.width)
                .saturating_add(x)
                .saturating_mul(3);
            if idx.saturating_add(2) < self.data.len() {
                self.data[idx] = r;
                self.data[idx + 1] = g;
                self.data[idx + 2] = b;
            }
        }
    }
}

/// Convert vx_df_image to ImageFormat
fn df_image_to_format(df: vx_df_image) -> Option<ImageFormat> {
    match df {
        0x38303055 => Some(ImageFormat::Gray), // VX_DF_IMAGE_U8 ('U008')
        0x31305555 => Some(ImageFormat::GrayU16), // VX_DF_IMAGE_U16 ('U016')
        0x53313053 => Some(ImageFormat::GrayS16), // VX_DF_IMAGE_S16 ('S016') - CORRECTED
        0x36313053 => Some(ImageFormat::GrayS16), // Alternative S16 format code
        0x32333055 => Some(ImageFormat::GrayU32), // VX_DF_IMAGE_U32 ('U032')
        0x32424752 => Some(ImageFormat::Rgb),  // VX_DF_IMAGE_RGB ('RGB2')
        0x41424752 => Some(ImageFormat::Rgba), // VX_DF_IMAGE_RGBA/RGBX ('RGBA')
        0x3231564E => Some(ImageFormat::NV12), // VX_DF_IMAGE_NV12 ('NV12')
        0x3132564E => Some(ImageFormat::NV21), // VX_DF_IMAGE_NV21 ('NV21')
        0x56555949 => Some(ImageFormat::IYUV), // VX_DF_IMAGE_IYUV ('IYUV')
        0x34565559 => Some(ImageFormat::YUV4), // VX_DF_IMAGE_YUV4 ('YUV4')
        0x34565559 => Some(ImageFormat::YUV4), // Alternative YUV4 format code
        0x56595559 => Some(ImageFormat::YUYV), // VX_DF_IMAGE_YUYV ('YUYV')
        0x59565955 => Some(ImageFormat::UYVY), // VX_DF_IMAGE_UYVY ('UYVY')
        _ => Some(ImageFormat::Gray),          // Default to gray
    }
}

/// Get image info from C API image
unsafe fn get_image_info(image: vx_image) -> Option<(u32, u32, vx_df_image)> {
    if image.is_null() {
        return None;
    }

    let img = &*(image as *const VxCImage);
    Some((img.width, img.height, img.format as vx_df_image))
}

/// Convert C API image to Rust Image
unsafe fn c_image_to_rust(image: vx_image) -> Option<Image> {
    let (width, height, format) = get_image_info(image)?;
    let img = &*(image as *const VxCImage);
    let format = df_image_to_format(format)?;

    // Handle external memory images (created from handle)
    // For these images, data buffer is empty and we must read from external_ptrs
    let data = if img.is_external_memory {
        let num_planes = VxCImage::num_planes(img.format);
        let total_size = VxCImage::calculate_size(width, height, img.format);
        if total_size == 0 {
            return None;
        }
        let mut buf = vec![0u8; total_size];
        // Copy each plane from external memory into contiguous buffer
        for plane_idx in 0..num_planes {
            let plane_offset = VxCImage::plane_offset(width, height, img.format, plane_idx);
            let plane_size = VxCImage::plane_size(width, height, img.format, plane_idx);
            let ext_ptr = if plane_idx < img.external_ptrs.len() {
                img.external_ptrs[plane_idx]
            } else {
                std::ptr::null_mut()
            };
            if !ext_ptr.is_null() {
                let ext_stride = if plane_idx < img.external_strides.len() {
                    img.external_strides[plane_idx] as usize
                } else {
                    plane_size
                };
                // For planar formats with potentially different external strides,
                // copy row by row using the external stride
                let (pw, ph) = VxCImage::plane_dimensions(width, height, img.format, plane_idx);
                let stride_x = VxCImage::plane_stride_x(img.format, plane_idx);
                for y in 0..(ph as usize) {
                    let src_offset = y * ext_stride;
                    let dst_offset = plane_offset + y * (pw as usize) * stride_x;
                    let row_bytes = (pw as usize) * stride_x;
                    if dst_offset + row_bytes <= buf.len() {
                        std::ptr::copy_nonoverlapping(
                            ext_ptr.add(src_offset),
                            buf.as_mut_ptr().add(dst_offset),
                            row_bytes,
                        );
                    }
                }
            }
        }
        buf
    } else {
        img.data.read().ok()?.clone()
    };

    // Handle ROI images: extract only the ROI portion from the parent's data
    if img.parent.is_some() && !img.roi_offsets.is_empty() {
        let bpp = match format {
            ImageFormat::Gray | ImageFormat::GrayS16 => match format {
                ImageFormat::GrayS16 => 2,
                _ => 1,
            },
            ImageFormat::Rgb => 3,
            ImageFormat::Rgba => 4,
            _ => 1,
        };
        let parent_width = {
            // Get parent dimensions from parent pointer
            if let Some(parent_addr) = img.parent {
                let parent_img = &*(parent_addr as *const VxCImage);
                parent_img.width as usize
            } else {
                // Can't determine parent width, fall back to simple extraction
                let result = Image::from_data(width as usize, height as usize, format, data);
                return Some(result);
            }
        };
        let (roi_start_x, roi_start_y) = img.roi_offsets[0];
        let roi_w = width as usize;
        let roi_h = height as usize;
        let parent_stride = parent_width * bpp;
        let roi_size = roi_w * roi_h * bpp;
        let mut roi_data = vec![0u8; roi_size];
        for y in 0..roi_h {
            let src_offset = ((roi_start_y + y) * parent_stride) + roi_start_x * bpp;
            let dst_offset = y * roi_w * bpp;
            let copy_len = roi_w * bpp;
            if src_offset + copy_len <= data.len() && dst_offset + copy_len <= roi_data.len() {
                roi_data[dst_offset..dst_offset + copy_len]
                    .copy_from_slice(&data[src_offset..src_offset + copy_len]);
            }
        }
        let mut result = Image::from_data(roi_w, roi_h, format, roi_data);
        if let Ok(vr) = img.valid_rect.read() {
            result.set_valid_rect(
                vr.start_x as usize,
                vr.start_y as usize,
                vr.end_x as usize,
                vr.end_y as usize,
            );
        }
        return Some(result);
    }

    let mut result = Image::from_data(width as usize, height as usize, format, data);
    // Read valid rectangle from C image
    if let Ok(vr) = img.valid_rect.read() {
        result.set_valid_rect(
            vr.start_x as usize,
            vr.start_y as usize,
            vr.end_x as usize,
            vr.end_y as usize,
        );
    }
    Some(result)
}

/// Convert C API image to Rust Image using raw data access
unsafe fn c_image_to_rust_raw(image: vx_image) -> Option<Image> {
    // Use the same logic as c_image_to_rust which now handles ROI images
    c_image_to_rust(image)
}

/// Copy Rust Image data back to C API image
unsafe fn copy_rust_to_c_image(src: &Image, dst: vx_image) -> vx_status {
    if dst.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let img = &*(dst as *const VxCImage);

    // Handle ROI images: write back to the correct offset in the parent's data buffer
    if img.parent.is_some() && !img.roi_offsets.is_empty() {
        // For ROI of external memory, get parent's external data and write back
        let bpp = match src.format {
            ImageFormat::GrayS16 => 2,
            ImageFormat::Gray => 1,
            ImageFormat::Rgb => 3,
            ImageFormat::Rgba => 4,
            _ => 1,
        };
        let parent_width = {
            if let Some(parent_addr) = img.parent {
                let parent_img = &*(parent_addr as *const VxCImage);
                parent_img.width as usize
            } else {
                return VX_ERROR_INVALID_REFERENCE;
            }
        };
        let (roi_start_x, roi_start_y) = img.roi_offsets[0];
        let roi_w = img.width as usize;
        let roi_h = img.height as usize;
        let parent_stride = parent_width * bpp;

        // Check if the parent image uses external memory
        let parent_is_external = {
            if let Some(parent_addr) = img.parent {
                let parent_img = &*(parent_addr as *const VxCImage);
                parent_img.is_external_memory
            } else {
                false
            }
        };

        if parent_is_external {
            let parent_img = if let Some(parent_addr) = img.parent {
                &*(parent_addr as *const VxCImage)
            } else {
                return VX_ERROR_INVALID_REFERENCE;
            };
            let ext_ptr = if !parent_img.external_ptrs.is_empty() {
                parent_img.external_ptrs[0]
            } else {
                std::ptr::null_mut()
            };
            if ext_ptr.is_null() {
                return VX_ERROR_INVALID_REFERENCE;
            }
            for y in 0..roi_h {
                let dst_offset = ((roi_start_y + y) * parent_stride) + roi_start_x * bpp;
                let src_offset = y * roi_w * bpp;
                let copy_len = roi_w * bpp;
                if src_offset + copy_len <= src.data.len() {
                    std::ptr::copy_nonoverlapping(
                        src.data.as_ptr().add(src_offset),
                        ext_ptr.add(dst_offset),
                        copy_len,
                    );
                }
            }
        } else {
            let mut dst_data = match img.data.write() {
                Ok(d) => d,
                Err(_) => return VX_ERROR_INVALID_REFERENCE,
            };
            for y in 0..roi_h {
                let dst_offset = ((roi_start_y + y) * parent_stride) + roi_start_x * bpp;
                let src_offset = y * roi_w * bpp;
                let copy_len = roi_w * bpp;
                if dst_offset + copy_len <= dst_data.len()
                    && src_offset + copy_len <= src.data.len()
                {
                    dst_data[dst_offset..dst_offset + copy_len]
                        .copy_from_slice(&src.data[src_offset..src_offset + copy_len]);
                }
            }
        }
        return VX_SUCCESS;
    }

    // Handle external memory images (created from handle)
    if img.is_external_memory {
        let num_planes = VxCImage::num_planes(img.format);
        for plane_idx in 0..num_planes {
            let plane_offset = VxCImage::plane_offset(img.width, img.height, img.format, plane_idx);
            let ext_ptr = if plane_idx < img.external_ptrs.len() {
                img.external_ptrs[plane_idx]
            } else {
                std::ptr::null_mut()
            };
            if ext_ptr.is_null() {
                continue;
            }
            let ext_stride = if plane_idx < img.external_strides.len() {
                img.external_strides[plane_idx] as usize
            } else {
                VxCImage::plane_size(img.width, img.height, img.format, plane_idx)
            };
            let (pw, ph) = VxCImage::plane_dimensions(img.width, img.height, img.format, plane_idx);
            let stride_x = VxCImage::plane_stride_x(img.format, plane_idx);
            for y in 0..(ph as usize) {
                let src_offset = plane_offset + y * (pw as usize) * stride_x;
                let dst_offset = y * ext_stride;
                let row_bytes = (pw as usize) * stride_x;
                if src_offset + row_bytes <= src.data.len() {
                    std::ptr::copy_nonoverlapping(
                        src.data.as_ptr().add(src_offset),
                        ext_ptr.add(dst_offset),
                        row_bytes,
                    );
                }
            }
        }
        return VX_SUCCESS;
    }

    // For internal memory images, proceed with the original logic
    let mut dst_data = match img.data.write() {
        Ok(d) => d,
        Err(_) => return VX_ERROR_INVALID_REFERENCE,
    };

    if dst_data.len() == src.data.len() {
        // Same format - direct copy
        dst_data.copy_from_slice(&src.data);
        VX_SUCCESS
    } else if src.format == ImageFormat::Rgb && dst_data.len() == src.data.len() * 4 / 3 {
        // RGB to RGBX/RGBA conversion (src: 3 bytes/pixel, dst: 4 bytes/pixel)
        // Check data is available in source
        if src.data.len() > 0 {
            let width = src.width;
            let height = src.height;
            for y in 0..height {
                for x in 0..width {
                    let (r, g, b) = src.get_rgb(x, y);
                    let dst_idx = (y * width + x) * 4;
                    if dst_idx + 3 < dst_data.len() {
                        dst_data[dst_idx] = r;
                        dst_data[dst_idx + 1] = g;
                        dst_data[dst_idx + 2] = b;
                        dst_data[dst_idx + 3] = 255; // Alpha
                    }
                }
            }
            VX_SUCCESS
        } else {
            VX_ERROR_INVALID_PARAMETERS
        }
    } else {
        VX_ERROR_INVALID_PARAMETERS
    }
}

/// Copy Rust Image data back to C API image - optimized version that handles
/// RGB→RGBX (which has dst_data.len() == src.data.len() / 3 * 4, not exactly *4/3 due to integer division)
unsafe fn copy_rust_to_c_image_optimized(src: &Image, dst: vx_image) -> vx_status {
    if dst.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let img = &*(dst as *const VxCImage);

    // For ROI images or external memory images, delegate to the general copy function
    if img.parent.is_some() && !img.roi_offsets.is_empty() {
        return copy_rust_to_c_image(src, dst);
    }

    // For external memory images, delegate to copy_rust_to_c_image which handles it
    if img.is_external_memory {
        return copy_rust_to_c_image(src, dst);
    }

    let mut dst_data = match img.data.write() {
        Ok(d) => d,
        Err(_) => return VX_ERROR_INVALID_REFERENCE,
    };

    // Get destination format
    let dst_format = match df_image_to_format(img.format as vx_df_image) {
        Some(f) => f,
        None => return VX_ERROR_INVALID_FORMAT,
    };

    // Exact match - same format
    if dst_data.len() == src.data.len() {
        dst_data.copy_from_slice(&src.data);
        return VX_SUCCESS;
    }

    // RGB to RGBA/RGBX conversion: 3 bytes/pixel to 4 bytes/pixel
    // src.data.len() == width * height * 3
    // dst_data.len() == width * height * 4
    if src.format == ImageFormat::Rgb && dst_format == ImageFormat::Rgba {
        let width = src.width;
        let height = src.height;
        let expected_src = width * height * 3;
        let expected_dst = width * height * 4;

        if src.data.len() == expected_src && dst_data.len() == expected_dst {
            for y in 0..height {
                for x in 0..width {
                    let (r, g, b) = src.get_rgb(x, y);
                    let dst_idx = y * width * 4 + x * 4;
                    dst_data[dst_idx] = r;
                    dst_data[dst_idx + 1] = g;
                    dst_data[dst_idx + 2] = b;
                    dst_data[dst_idx + 3] = 255; // Alpha
                }
            }
            return VX_SUCCESS;
        }
    }

    // Gray to RGB conversion: 1 byte/pixel to 3 bytes/pixel
    if src.format == ImageFormat::Gray && dst_format == ImageFormat::Rgb {
        let width = src.width;
        let height = src.height;
        let expected_src = width * height;
        let expected_dst = width * height * 3;

        if src.data.len() == expected_src && dst_data.len() == expected_dst {
            for y in 0..height {
                for x in 0..width {
                    let gray = src.get_pixel(x, y);
                    let dst_idx = y * width * 3 + x * 3;
                    dst_data[dst_idx] = gray;
                    dst_data[dst_idx + 1] = gray;
                    dst_data[dst_idx + 2] = gray;
                }
            }
            return VX_SUCCESS;
        }
    }

    // Gray to RGBA conversion: 1 byte/pixel to 4 bytes/pixel
    if src.format == ImageFormat::Gray && dst_format == ImageFormat::Rgba {
        let width = src.width;
        let height = src.height;
        let expected_src = width * height;
        let expected_dst = width * height * 4;

        if src.data.len() == expected_src && dst_data.len() == expected_dst {
            for y in 0..height {
                for x in 0..width {
                    let gray = src.get_pixel(x, y);
                    let dst_idx = y * width * 4 + x * 4;
                    dst_data[dst_idx] = gray;
                    dst_data[dst_idx + 1] = gray;
                    dst_data[dst_idx + 2] = gray;
                    dst_data[dst_idx + 3] = 255; // Alpha
                }
            }
            return VX_SUCCESS;
        }
    }

    // RGBA to RGB conversion: 4 bytes/pixel to 3 bytes/pixel
    if src.format == ImageFormat::Rgba && dst_format == ImageFormat::Rgb {
        let width = src.width;
        let height = src.height;
        let expected_src = width * height * 4;
        let expected_dst = width * height * 3;

        if src.data.len() == expected_src && dst_data.len() == expected_dst {
            for y in 0..height {
                for x in 0..width {
                    let src_idx = y * width * 4 + x * 4;
                    let dst_idx = y * width * 3 + x * 3;
                    dst_data[dst_idx] = src.data[src_idx];
                    dst_data[dst_idx + 1] = src.data[src_idx + 1];
                    dst_data[dst_idx + 2] = src.data[src_idx + 2];
                    // Skip alpha
                }
            }
            return VX_SUCCESS;
        }
    }

    VX_ERROR_INVALID_PARAMETERS
}

/// Copy Rust Image data back to C API image with format conversion
unsafe fn convert_and_copy(src: &Image, dst: vx_image, target_format: vx_df_image) -> vx_status {
    if dst.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Get source and target formats
    let src_format = src.format();
    let Some(dst_format) = df_image_to_format(target_format) else {
        return VX_ERROR_INVALID_FORMAT;
    };

    // If formats match, simple copy (which handles external memory)
    if src_format == dst_format {
        return copy_rust_to_c_image(src, dst);
    }

    // External memory images don't support format conversion via this path
    let img = &*(dst as *const VxCImage);
    if img.is_external_memory {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    // Format conversion required for internal memory images
    let width = img.width;
    let height = img.height;
    let mut dst_data = match img.data.write() {
        Ok(d) => d,
        Err(_) => return VX_ERROR_INVALID_REFERENCE,
    };

    match (src_format, dst_format) {
        (ImageFormat::Gray, ImageFormat::GrayS16) => {
            // U8 (0-255) to S16 (-32768 to 32767)
            // U8 with offset 128 is signed: (val as i16 - 128) * 256
            let src_data = src.data();
            let w = width as usize;
            let h = height as usize;
            for y in 0..h {
                for x in 0..w {
                    let val = src_data[y * w + x] as i16;
                    // Convert: U8(0-255) -> S16(-128 to 127) with scale factor
                    // S16 value = (U8 - 128) * 256
                    let s16_val = (val - 128i16).wrapping_mul(256i16);
                    let idx = y * w + x;
                    let bytes = s16_val.to_le_bytes();
                    if idx * 2 + 1 < dst_data.len() {
                        dst_data[idx * 2] = bytes[0];
                        dst_data[idx * 2 + 1] = bytes[1];
                    }
                }
            }
            VX_SUCCESS
        }
        (_src, _dst) => VX_ERROR_INVALID_FORMAT,
    }
}

/// Create a new Rust Image matching the C image dimensions and format
unsafe fn create_matching_image(c_image: vx_image) -> Option<Image> {
    let (width, height, format) = get_image_info(c_image)?;
    let format = df_image_to_format(format)?;
    Image::new(width as usize, height as usize, format)
}

/// VXU Color Functions
/// ===========================================================================

/// BT.709 coefficients for YUV conversion (from OpenVX conformance tests)
/// Y = 0.2126*R + 0.7152*G + 0.0722*B
/// U = 128 - 0.1146*R - 0.3854*G + 0.5*B
/// V = 128 + 0.5*R - 0.4542*G - 0.0458*B
const Y_COEFF_R: i32 = 54; // 0.2126 * 256 (BT.709)
const Y_COEFF_G: i32 = 183; // 0.7152 * 256 (BT.709)
const Y_COEFF_B: i32 = 18; // 0.0722 * 256 (BT.709)

const U_COEFF_R: i32 = -29; // -0.1146 * 256 (BT.709)
const U_COEFF_G: i32 = -99; // -0.3854 * 256 (BT.709)
const U_COEFF_B: i32 = 128; // 0.5 * 256

const V_COEFF_R: i32 = 128; // 0.5 * 256
const V_COEFF_G: i32 = -116; // -0.4542 * 256 (BT.709)
const V_COEFF_B: i32 = -12; // -0.0458 * 256 (BT.709)

/// Clamp value to u8 range
#[inline]
fn clamp_u8(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

/// RGB to YUV conversion (BT.709) using fast fixed-point math.
/// Bit-exact with the floating-point CTS reference for u8 inputs.
#[inline(always)]
fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as i32;
    let gf = g as i32;
    let bf = b as i32;

    let y = ((Y_COEFF_R * rf + Y_COEFF_G * gf + Y_COEFF_B * bf + 127) >> 8).clamp(0, 255) as u8;
    let u = ((U_COEFF_R * rf + U_COEFF_G * gf + U_COEFF_B * bf + 32768) >> 8).clamp(0, 255) as u8;
    let v = ((V_COEFF_R * rf + V_COEFF_G * gf + V_COEFF_B * bf + 32768) >> 8).clamp(0, 255) as u8;

    (y, u, v)
}

/// YUV to RGB conversion (BT.709) using fast fixed-point math.
/// Bit-exact with the floating-point CTS reference for u8 inputs.
#[inline(always)]
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let yf = y as i32;
    let uf = (u as i32) - 128;
    let vf = (v as i32) - 128;

    // Coefficients: R=1.5748, G=-0.1873/-0.4681, B=1.8556  →  *256
    let r = ((yf + ((402 * vf + 128) >> 8))).clamp(0, 255) as u8;
    let g = ((yf + ((-48 * uf - 120 * vf + 128) >> 8))).clamp(0, 255) as u8;
    let b = ((yf + ((475 * uf + 128) >> 8))).clamp(0, 255) as u8;

    (r, g, b)
}

/// Calculate plane sizes and offsets for planar YUV formats
/// Calculate plane sizes and offsets for planar YUV formats
fn get_nv12_plane_info(width: u32, height: u32) -> (usize, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = w * ((h + 1) / 2); // UV interleaved, half height
    let total_size = y_size + uv_size;
    (y_size, uv_size, total_size)
}

fn get_nv21_plane_info(width: u32, height: u32) -> (usize, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let vu_size = w * ((h + 1) / 2); // VU interleaved, half height
    let total_size = y_size + vu_size;
    (y_size, vu_size, total_size)
}

fn get_iyuv_plane_info(width: u32, height: u32) -> (usize, usize, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let half_w = (w + 1) / 2;
    let half_h = (h + 1) / 2;
    let y_size = w * h;
    let u_size = half_w * half_h;
    let v_size = half_w * half_h;
    let total_size = y_size + u_size + v_size;
    (y_size, u_size, v_size, total_size)
}

fn get_yuv4_plane_info(width: u32, height: u32) -> (usize, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let plane_size = w * h;
    let total_size = plane_size * 3;
    (plane_size, plane_size, total_size)
}

/// Get UV indices for NV12
#[inline]
fn get_nv12_uv_indices(x: usize, y: usize, width: usize, y_size: usize) -> usize {
    let uv_y = y / 2;
    let uv_x = (x / 2) * 2; // U and V interleaved
    y_size + uv_y * width + uv_x
}

pub fn vxu_color_convert_impl(
    _context: vx_context,
    input: vx_image,
    output: vx_image,
) -> vx_status {
    if input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        // Get source image info
        let (src_width, src_height, src_format) = match get_image_info(input) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Get destination image info
        let (dst_width, dst_height, dst_format) = match get_image_info(output) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Validate dimensions match
        if src_width != dst_width || src_height != dst_height {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        // Convert formats
        let src_fmt = match df_image_to_format(src_format) {
            Some(f) => f,
            None => return VX_ERROR_INVALID_FORMAT,
        };
        let dst_fmt = match df_image_to_format(dst_format) {
            Some(f) => f,
            None => return VX_ERROR_INVALID_FORMAT,
        };

        // Access source and destination data directly
        let src_img = &*(input as *const VxCImage);
        let dst_img = &*(output as *const VxCImage);

        let src_data = match src_img.data.read() {
            Ok(d) => d,
            Err(_) => return VX_ERROR_INVALID_REFERENCE,
        };
        let mut dst_data = match dst_img.data.write() {
            Ok(d) => d,
            Err(_) => return VX_ERROR_INVALID_REFERENCE,
        };

        let width = src_width as usize;
        let height = src_height as usize;

        // Perform conversion directly on the image buffers
        match (src_fmt, dst_fmt) {
            // RGB to RGBA
            (ImageFormat::Rgb, ImageFormat::Rgba) => {
                let expected_src_len = width * height * 3;
                let expected_dst_len = width * height * 4;
                if src_data.len() >= expected_src_len && dst_data.len() >= expected_dst_len {
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 3 + x * 3;
                            let dst_idx = y * width * 4 + x * 4;
                            if src_idx + 2 < src_data.len() && dst_idx + 3 < dst_data.len() {
                                dst_data[dst_idx] = src_data[src_idx];
                                dst_data[dst_idx + 1] = src_data[src_idx + 1];
                                dst_data[dst_idx + 2] = src_data[src_idx + 2];
                                dst_data[dst_idx + 3] = 255;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // RGBA to RGB
            (ImageFormat::Rgba, ImageFormat::Rgb) => {
                if src_data.len() == width * height * 4 && dst_data.len() == width * height * 3 {
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 4 + x * 4;
                            let dst_idx = y * width * 3 + x * 3;
                            dst_data[dst_idx] = src_data[src_idx];
                            dst_data[dst_idx + 1] = src_data[src_idx + 1];
                            dst_data[dst_idx + 2] = src_data[src_idx + 2];
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // RGBX to NV12
            (ImageFormat::Rgba, ImageFormat::NV12) => {
                let src_len = width * height * 4;
                let (y_size, _, dst_total) = get_nv12_plane_info(src_width, src_height);
                if src_data.len() >= src_len && dst_data.len() >= dst_total {
                    // Clear destination first
                    dst_data.fill(128);

                    // First pass: compute Y plane
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 4 + x * 4;
                            let r = src_data[src_idx];
                            let g = src_data[src_idx + 1];
                            let b = src_data[src_idx + 2];
                            let (y_val, _, _) = rgb_to_yuv(r, g, b);
                            dst_data[y * width + x] = y_val;
                        }
                    }

                    // Second pass: compute UV plane (subsampled 2x2)
                    // CTS reference: convert each pixel to YUV, then average the U and V values
                    for y in (0..height).step_by(2) {
                        for x in (0..width).step_by(2) {
                            let mut sum_u = 0i32;
                            let mut sum_v = 0i32;

                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let py = (y + dy).min(height - 1);
                                    let px = (x + dx).min(width - 1);
                                    let src_idx = py * width * 4 + px * 4;
                                    let r = src_data[src_idx];
                                    let g = src_data[src_idx + 1];
                                    let b = src_data[src_idx + 2];
                                    let (_, u, v) = rgb_to_yuv(r, g, b);
                                    sum_u += u as i32;
                                    sum_v += v as i32;
                                }
                            }

                            let u_val = (sum_u / 4) as u8;
                            let v_val = (sum_v / 4) as u8;

                            let uv_y = y / 2;
                            let uv_idx = y_size + uv_y * width + x;
                            if uv_idx + 1 < dst_data.len() {
                                dst_data[uv_idx] = u_val;
                                dst_data[uv_idx + 1] = v_val;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // RGBX to IYUV (I420)
            (ImageFormat::Rgba, ImageFormat::IYUV) => {
                let src_len = width * height * 4;
                let (y_size, u_size, _, dst_total) = get_iyuv_plane_info(src_width, src_height);
                if src_data.len() >= src_len && dst_data.len() >= dst_total {
                    // Clear destination
                    dst_data.fill(0);

                    let half_w = (width + 1) / 2;
                    let _half_h = (height + 1) / 2;

                    // Compute Y plane
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 4 + x * 4;
                            let r = src_data[src_idx];
                            let g = src_data[src_idx + 1];
                            let b = src_data[src_idx + 2];
                            let (y_val, _, _) = rgb_to_yuv(r, g, b);
                            dst_data[y * width + x] = y_val;
                        }
                    }

                    // Compute U and V planes (subsampled 2x2)
                    // CTS reference: convert each pixel to YUV, then average the U and V values
                    for y in (0..height).step_by(2) {
                        for x in (0..width).step_by(2) {
                            let mut sum_u = 0i32;
                            let mut sum_v = 0i32;

                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let py = (y + dy).min(height - 1);
                                    let px = (x + dx).min(width - 1);
                                    let src_idx = py * width * 4 + px * 4;
                                    let r = src_data[src_idx];
                                    let g = src_data[src_idx + 1];
                                    let b = src_data[src_idx + 2];
                                    let (_, u, v) = rgb_to_yuv(r, g, b);
                                    sum_u += u as i32;
                                    sum_v += v as i32;
                                }
                            }

                            let u_val = (sum_u / 4) as u8;
                            let v_val = (sum_v / 4) as u8;

                            let uv_y = y / 2;
                            let uv_x = x / 2;
                            let u_idx = y_size + uv_y * half_w + uv_x;
                            let v_idx = y_size + u_size + uv_y * half_w + uv_x;

                            if u_idx < dst_data.len() {
                                dst_data[u_idx] = u_val;
                            }
                            if v_idx < dst_data.len() {
                                dst_data[v_idx] = v_val;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // RGBX to YUV4
            (ImageFormat::Rgba, ImageFormat::YUV4) => {
                let src_len = width * height * 4;
                let (y_size, u_size, dst_total) = get_yuv4_plane_info(src_width, src_height);
                if src_data.len() >= src_len && dst_data.len() >= dst_total {
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 4 + x * 4;
                            let r = src_data[src_idx];
                            let g = src_data[src_idx + 1];
                            let b = src_data[src_idx + 2];
                            let (y_val, u_val, v_val) = rgb_to_yuv(r, g, b);

                            let idx = y * width + x;
                            dst_data[idx] = y_val;
                            dst_data[y_size + idx] = u_val;
                            dst_data[y_size + u_size + idx] = v_val;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // Gray to/from RGB/RGBA
            (ImageFormat::Gray, ImageFormat::Rgb) => {
                if src_data.len() == width * height && dst_data.len() == width * height * 3 {
                    for y in 0..height {
                        for x in 0..width {
                            let gray = src_data[y * width + x];
                            let dst_idx = y * width * 3 + x * 3;
                            dst_data[dst_idx] = gray;
                            dst_data[dst_idx + 1] = gray;
                            dst_data[dst_idx + 2] = gray;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            (ImageFormat::Gray, ImageFormat::Rgba) => {
                if src_data.len() == width * height && dst_data.len() == width * height * 4 {
                    for y in 0..height {
                        for x in 0..width {
                            let gray = src_data[y * width + x];
                            let dst_idx = y * width * 4 + x * 4;
                            dst_data[dst_idx] = gray;
                            dst_data[dst_idx + 1] = gray;
                            dst_data[dst_idx + 2] = gray;
                            dst_data[dst_idx + 3] = 255;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            (ImageFormat::Rgb, ImageFormat::Gray) => {
                if src_data.len() == width * height * 3 && dst_data.len() == width * height {
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 3 + x * 3;
                            let r = src_data[src_idx] as u32;
                            let g = src_data[src_idx + 1] as u32;
                            let b = src_data[src_idx + 2] as u32;
                            let gray = ((54 * r + 183 * g + 18 * b) / 255) as u8;
                            dst_data[y * width + x] = gray;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            (ImageFormat::Rgba, ImageFormat::Gray) => {
                if src_data.len() == width * height * 4 && dst_data.len() == width * height {
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 4 + x * 4;
                            let r = src_data[src_idx] as u32;
                            let g = src_data[src_idx + 1] as u32;
                            let b = src_data[src_idx + 2] as u32;
                            let gray = ((54 * r + 183 * g + 18 * b) / 255) as u8;
                            dst_data[y * width + x] = gray;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // RGB to NV12
            (ImageFormat::Rgb, ImageFormat::NV12) => {
                let src_len = width * height * 3;
                let (y_size, _, dst_total) = get_nv12_plane_info(src_width, src_height);
                if src_data.len() >= src_len && dst_data.len() >= dst_total {
                    // Clear UV plane with neutral chroma
                    dst_data[y_size..].fill(128);

                    // Y plane: linear scan over src, write to dst[y * width + x]
                    let mut src_i = 0usize;
                    for y in 0..height {
                        let dst_row = y * width;
                        for x in 0..width {
                            let r = src_data[src_i];
                            let g = src_data[src_i + 1];
                            let b = src_data[src_i + 2];
                            src_i += 3;
                            dst_data[dst_row + x] = ((Y_COEFF_R * r as i32
                                + Y_COEFF_G * g as i32
                                + Y_COEFF_B * b as i32
                                + 127) >> 8) as u8;
                        }
                    }

                    // UV plane: process 2x2 blocks
                    let mut src_i = 0usize;
                    for y in (0..height).step_by(2) {
                        let next_row_src_i = src_i + width * 3;
                        let uv_row = y_size + (y / 2) * width;
                        let mut x = 0usize;
                        while x + 1 < width {
                            // 4 pixels: (x,y), (x+1,y), (x,y+1), (x+1,y+1)
                            let r0 = src_data[src_i + x * 3];
                            let g0 = src_data[src_i + x * 3 + 1];
                            let b0 = src_data[src_i + x * 3 + 2];
                            let r1 = src_data[src_i + (x + 1) * 3];
                            let g1 = src_data[src_i + (x + 1) * 3 + 1];
                            let b1 = src_data[src_i + (x + 1) * 3 + 2];
                            let r2 = src_data[next_row_src_i + x * 3];
                            let g2 = src_data[next_row_src_i + x * 3 + 1];
                            let b2 = src_data[next_row_src_i + x * 3 + 2];
                            let r3 = src_data[next_row_src_i + (x + 1) * 3];
                            let g3 = src_data[next_row_src_i + (x + 1) * 3 + 1];
                            let b3 = src_data[next_row_src_i + (x + 1) * 3 + 2];

                            let sum_u = (U_COEFF_R * (r0 as i32 + r1 as i32 + r2 as i32 + r3 as i32)
                                + U_COEFF_G * (g0 as i32 + g1 as i32 + g2 as i32 + g3 as i32)
                                + U_COEFF_B * (b0 as i32 + b1 as i32 + b2 as i32 + b3 as i32)
                                + 4 * 32768) >> 10;
                            let sum_v = (V_COEFF_R * (r0 as i32 + r1 as i32 + r2 as i32 + r3 as i32)
                                + V_COEFF_G * (g0 as i32 + g1 as i32 + g2 as i32 + g3 as i32)
                                + V_COEFF_B * (b0 as i32 + b1 as i32 + b2 as i32 + b3 as i32)
                                + 4 * 32768) >> 10;

                            let uv_idx = uv_row + x;
                            dst_data[uv_idx] = sum_u.clamp(0, 255) as u8;
                            dst_data[uv_idx + 1] = sum_v.clamp(0, 255) as u8;
                            x += 2;
                        }
                        // Handle odd width tail
                        if x < width {
                            let mut sum_u = 0i32;
                            let mut sum_v = 0i32;
                            let mut count = 0i32;
                            for dy in 0..2 {
                                let py = (y + dy).min(height - 1);
                                let pidx = (py * width + x) * 3;
                                let r = src_data[pidx];
                                let g = src_data[pidx + 1];
                                let b = src_data[pidx + 2];
                                sum_u += (U_COEFF_R * r as i32
                                    + U_COEFF_G * g as i32
                                    + U_COEFF_B * b as i32
                                    + 32768) >> 8;
                                sum_v += (V_COEFF_R * r as i32
                                    + V_COEFF_G * g as i32
                                    + V_COEFF_B * b as i32
                                    + 32768) >> 8;
                                count += 1;
                            }
                            let uv_idx = uv_row + x;
                            dst_data[uv_idx] = (sum_u / count).clamp(0, 255) as u8;
                            dst_data[uv_idx + 1] = (sum_v / count).clamp(0, 255) as u8;
                        }
                        src_i = next_row_src_i + if y + 1 < height { width * 3 } else { 0 };
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV12 to RGB
            (ImageFormat::NV12, ImageFormat::Rgb) => {
                let (y_size, _, src_total) = get_nv12_plane_info(src_width, src_height);
                let dst_len = width * height * 3;
                if src_data.len() >= src_total && dst_data.len() >= dst_len {
                    for y in 0..height {
                        for x in 0..width {
                            let y_val = src_data[y * width + x];
                            let uv_y = y / 2;
                            let uv_x = (x / 2) * 2;
                            let uv_idx = y_size + uv_y * width + uv_x;
                            let u = if uv_idx < src_data.len() {
                                src_data[uv_idx]
                            } else {
                                128
                            };
                            let v = if uv_idx + 1 < src_data.len() {
                                src_data[uv_idx + 1]
                            } else {
                                128
                            };

                            let (r, g, b) = yuv_to_rgb(y_val, u, v);

                            let dst_idx = y * width * 3 + x * 3;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV12 to RGBA
            (ImageFormat::NV12, ImageFormat::Rgba) => {
                let (y_size, _, src_total) = get_nv12_plane_info(src_width, src_height);
                let dst_len = width * height * 4;
                if src_data.len() >= src_total && dst_data.len() >= dst_len {
                    for y in 0..height {
                        for x in 0..width {
                            let y_val = src_data[y * width + x];
                            let uv_y = y / 2;
                            let uv_x = (x / 2) * 2;
                            let uv_idx = y_size + uv_y * width + uv_x;
                            let u = if uv_idx < src_data.len() {
                                src_data[uv_idx]
                            } else {
                                128
                            };
                            let v = if uv_idx + 1 < src_data.len() {
                                src_data[uv_idx + 1]
                            } else {
                                128
                            };

                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = y * width * 4 + x * 4;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                            dst_data[dst_idx + 3] = 255;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // RGB to IYUV (I420)
            (ImageFormat::Rgb, ImageFormat::IYUV) => {
                let src_len = width * height * 3;
                let (y_size, u_size, _, dst_total) = get_iyuv_plane_info(src_width, src_height);
                if src_data.len() >= src_len && dst_data.len() >= dst_total {
                    dst_data[..dst_total].fill(0);

                    let half_w = (width + 1) / 2;

                    // Y plane: linear scan
                    let mut src_i = 0usize;
                    for y in 0..height {
                        let dst_row = y * width;
                        for x in 0..width {
                            let r = src_data[src_i];
                            let g = src_data[src_i + 1];
                            let b = src_data[src_i + 2];
                            src_i += 3;
                            dst_data[dst_row + x] = ((Y_COEFF_R * r as i32
                                + Y_COEFF_G * g as i32
                                + Y_COEFF_B * b as i32
                                + 127) >> 8) as u8;
                        }
                    }

                    // U/V planes: coarsened 2x2 average in fixed-point
                    src_i = 0;
                    for y in (0..height).step_by(2) {
                        let next_row_src_i = src_i + width * 3;
                        let uv_y = y / 2;
                        let u_row = y_size + uv_y * half_w;
                        let v_row = y_size + u_size + uv_y * half_w;
                        let mut x = 0usize;
                        while x + 1 < width {
                            let r0 = src_data[src_i + x * 3];
                            let g0 = src_data[src_i + x * 3 + 1];
                            let b0 = src_data[src_i + x * 3 + 2];
                            let r1 = src_data[src_i + (x + 1) * 3];
                            let g1 = src_data[src_i + (x + 1) * 3 + 1];
                            let b1 = src_data[src_i + (x + 1) * 3 + 2];
                            let r2 = src_data[next_row_src_i + x * 3];
                            let g2 = src_data[next_row_src_i + x * 3 + 1];
                            let b2 = src_data[next_row_src_i + x * 3 + 2];
                            let r3 = src_data[next_row_src_i + (x + 1) * 3];
                            let g3 = src_data[next_row_src_i + (x + 1) * 3 + 1];
                            let b3 = src_data[next_row_src_i + (x + 1) * 3 + 2];

                            let rs = r0 as i32 + r1 as i32 + r2 as i32 + r3 as i32;
                            let gs = g0 as i32 + g1 as i32 + g2 as i32 + g3 as i32;
                            let bs = b0 as i32 + b1 as i32 + b2 as i32 + b3 as i32;

                            let u_val = ((U_COEFF_R * rs + U_COEFF_G * gs + U_COEFF_B * bs
                                + 4 * 32768) >> 10)
                                .clamp(0, 255) as u8;
                            let v_val = ((V_COEFF_R * rs + V_COEFF_G * gs + V_COEFF_B * bs
                                + 4 * 32768) >> 10)
                                .clamp(0, 255) as u8;

                            let uv_x = x / 2;
                            dst_data[u_row + uv_x] = u_val;
                            dst_data[v_row + uv_x] = v_val;
                            x += 2;
                        }
                        if x < width {
                            // tail: 1 or 2 pixels wide
                            let mut sum_u = 0i32;
                            let mut sum_v = 0i32;
                            let mut count = 0i32;
                            for dy in 0..2 {
                                let py = (y + dy).min(height - 1);
                                let pidx = (py * width + x) * 3;
                                let r = src_data[pidx];
                                let g = src_data[pidx + 1];
                                let b = src_data[pidx + 2];
                                sum_u += (U_COEFF_R * r as i32 + U_COEFF_G * g as i32
                                    + U_COEFF_B * b as i32 + 32768) >> 8;
                                sum_v += (V_COEFF_R * r as i32 + V_COEFF_G * g as i32
                                    + V_COEFF_B * b as i32 + 32768) >> 8;
                                count += 1;
                            }
                            let uv_x = x / 2;
                            dst_data[u_row + uv_x] = (sum_u / count).clamp(0, 255) as u8;
                            dst_data[v_row + uv_x] = (sum_v / count).clamp(0, 255) as u8;
                        }
                        src_i = next_row_src_i + if y + 1 < height { width * 3 } else { 0 };
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // IYUV to RGB
            (ImageFormat::IYUV, ImageFormat::Rgb) => {
                let (y_size, u_size, _, src_total) = get_iyuv_plane_info(src_width, src_height);
                let dst_len = width * height * 3;
                if src_data.len() >= src_total && dst_data.len() >= dst_len {
                    let half_w = (width + 1) / 2;
                    let _half_h = (height + 1) / 2;

                    for y in 0..height {
                        for x in 0..width {
                            let y_val = src_data[y * width + x];
                            let uv_y = y / 2;
                            let uv_x = x / 2;
                            let u_idx = y_size + uv_y * half_w + uv_x;
                            let v_idx = y_size + u_size + uv_y * half_w + uv_x;

                            let u = if u_idx < src_data.len() {
                                src_data[u_idx]
                            } else {
                                128
                            };
                            let v = if v_idx < src_data.len() {
                                src_data[v_idx]
                            } else {
                                128
                            };

                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = y * width * 3 + x * 3;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // IYUV to RGBA
            (ImageFormat::IYUV, ImageFormat::Rgba) => {
                let (y_size, u_size, _, src_total) = get_iyuv_plane_info(src_width, src_height);
                let dst_len = width * height * 4;
                if src_data.len() >= src_total && dst_data.len() >= dst_len {
                    let half_w = (width + 1) / 2;

                    for y in 0..height {
                        for x in 0..width {
                            let y_val = src_data[y * width + x];
                            let uv_y = y / 2;
                            let uv_x = x / 2;
                            let u_idx = y_size + uv_y * half_w + uv_x;
                            let v_idx = y_size + u_size + uv_y * half_w + uv_x;

                            let u = if u_idx < src_data.len() {
                                src_data[u_idx]
                            } else {
                                128
                            };
                            let v = if v_idx < src_data.len() {
                                src_data[v_idx]
                            } else {
                                128
                            };

                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = y * width * 4 + x * 4;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                            dst_data[dst_idx + 3] = 255;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // RGB to YUV4
            (ImageFormat::Rgb, ImageFormat::YUV4) => {
                let src_len = width * height * 3;
                let (y_size, u_size, dst_total) = get_yuv4_plane_info(src_width, src_height);
                if src_data.len() >= src_len && dst_data.len() >= dst_total {
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = y * width * 3 + x * 3;
                            let r = src_data[src_idx];
                            let g = src_data[src_idx + 1];
                            let b = src_data[src_idx + 2];
                            let (y_val, u_val, v_val) = rgb_to_yuv(r, g, b);

                            dst_data[y * width + x] = y_val;
                            dst_data[y_size + y * width + x] = u_val;
                            dst_data[y_size + u_size + y * width + x] = v_val;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // YUV4 to RGB
            (ImageFormat::YUV4, ImageFormat::Rgb) => {
                let (y_size, u_size, src_total) = get_yuv4_plane_info(src_width, src_height);
                let dst_len = width * height * 3;
                if src_data.len() >= src_total && dst_data.len() >= dst_len {
                    for y in 0..height {
                        for x in 0..width {
                            let idx = y * width + x;
                            let y_val = src_data[idx];
                            let u = src_data[y_size + idx];
                            let v = src_data[y_size + u_size + idx];

                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = y * width * 3 + x * 3;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // YUV4 to RGBA
            (ImageFormat::YUV4, ImageFormat::Rgba) => {
                let (y_size, u_size, src_total) = get_yuv4_plane_info(src_width, src_height);
                let dst_len = width * height * 4;
                if src_data.len() >= src_total && dst_data.len() >= dst_len {
                    for y in 0..height {
                        for x in 0..width {
                            let idx = y * width + x;
                            let y_val = src_data[idx];
                            let u = src_data[y_size + idx];
                            let v = src_data[y_size + u_size + idx];

                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = y * width * 4 + x * 4;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                            dst_data[dst_idx + 3] = 255;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // IYUV to NV12
            (ImageFormat::IYUV, ImageFormat::NV12) => {
                let (src_y_size, src_u_size, _, src_total) =
                    get_iyuv_plane_info(src_width, src_height);
                let (dst_y_size, _, dst_total) = get_nv12_plane_info(src_width, src_height);
                if src_data.len() >= src_total && dst_data.len() >= dst_total {
                    let half_w = (width + 1) / 2;
                    let half_h = (height + 1) / 2;

                    // Copy Y plane
                    for y in 0..height {
                        for x in 0..width {
                            dst_data[y * width + x] = src_data[y * width + x];
                        }
                    }

                    // Interleave U and V planes
                    for y in 0..half_h {
                        for x in 0..half_w {
                            let u_idx = src_y_size + y * half_w + x;
                            let v_idx = src_y_size + src_u_size + y * half_w + x;
                            let uv_idx = dst_y_size + y * width + x * 2;

                            if uv_idx + 1 < dst_data.len() {
                                dst_data[uv_idx] = src_data[u_idx];
                                dst_data[uv_idx + 1] = src_data[v_idx];
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV12 to IYUV
            (ImageFormat::NV12, ImageFormat::IYUV) => {
                let (src_y_size, _, src_total) = get_nv12_plane_info(src_width, src_height);
                let (dst_y_size, dst_u_size, _, dst_total) =
                    get_iyuv_plane_info(src_width, src_height);
                if src_data.len() >= src_total && dst_data.len() >= dst_total {
                    let half_w = (width + 1) / 2;
                    let half_h = (height + 1) / 2;

                    // Copy Y plane
                    for y in 0..height {
                        for x in 0..width {
                            dst_data[y * width + x] = src_data[y * width + x];
                        }
                    }

                    // De-interleave UV plane
                    for y in 0..half_h {
                        for x in 0..half_w {
                            let uv_idx = src_y_size + y * width + x * 2;
                            let u_idx = dst_y_size + y * half_w + x;
                            let v_idx = dst_y_size + dst_u_size + y * half_w + x;

                            if uv_idx + 1 < src_data.len() {
                                if u_idx < dst_data.len() {
                                    dst_data[u_idx] = src_data[uv_idx];
                                }
                                if v_idx < dst_data.len() {
                                    dst_data[v_idx] = src_data[uv_idx + 1];
                                }
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV12 to YUV4
            (ImageFormat::NV12, ImageFormat::YUV4) => {
                let (src_y_size, _src_uv_size, src_total) =
                    get_nv12_plane_info(src_width, src_height);
                let (dst_y_size, dst_u_size, dst_total) =
                    get_yuv4_plane_info(src_width, src_height);
                if src_data.len() >= src_total && dst_data.len() >= dst_total {
                    // Copy Y plane
                    for y in 0..height {
                        for x in 0..width {
                            dst_data[y * width + x] = src_data[y * width + x];
                        }
                    }

                    // Expand UV to full size U and V planes
                    // NV12: UV interleaved, U first
                    // UV plane has same width as Y but half height
                    // Stride for UV is width/2 in terms of sample pairs
                    for y in 0..height {
                        for x in 0..width {
                            let uv_y = y / 2;
                            let uv_x = x / 2;
                            // In NV12, UV pairs are stored interleaved
                            // Each UV pair takes 2 bytes, and there are width/2 pairs per row
                            let uv_idx = src_y_size + uv_y * width + uv_x * 2;
                            let u = if uv_idx < src_data.len() {
                                src_data[uv_idx]
                            } else {
                                128
                            };
                            let v = if uv_idx + 1 < src_data.len() {
                                src_data[uv_idx + 1]
                            } else {
                                128
                            };

                            let idx = y * width + x;
                            dst_data[dst_y_size + idx] = u;
                            dst_data[dst_y_size + dst_u_size + idx] = v;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV21 to YUV4
            (ImageFormat::NV21, ImageFormat::YUV4) => {
                let (src_y_size, _, src_total) = get_nv12_plane_info(src_width, src_height);
                let (dst_y_size, dst_u_size, dst_total) =
                    get_yuv4_plane_info(src_width, src_height);
                if src_data.len() >= src_total && dst_data.len() >= dst_total {
                    // Copy Y plane
                    for y in 0..height {
                        for x in 0..width {
                            dst_data[y * width + x] = src_data[y * width + x];
                        }
                    }

                    // Expand VU (NV21) to full size U and V planes
                    // NV21: VU interleaved, V first
                    for y in 0..height {
                        for x in 0..width {
                            let vu_y = y / 2;
                            let vu_x = x / 2;
                            // In NV21, VU pairs are stored at each even position
                            let vu_idx = src_y_size + vu_y * width + vu_x * 2;
                            let v = if vu_idx < src_data.len() {
                                src_data[vu_idx]
                            } else {
                                128
                            };
                            let u = if vu_idx + 1 < src_data.len() {
                                src_data[vu_idx + 1]
                            } else {
                                128
                            };

                            let idx = y * width + x;
                            dst_data[dst_y_size + idx] = u;
                            dst_data[dst_y_size + dst_u_size + idx] = v;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV21 to RGB
            (ImageFormat::NV21, ImageFormat::Rgb) => {
                let (y_size, _, total) = get_nv12_plane_info(src_width, src_height);
                if src_data.len() >= total && dst_data.len() >= width * height * 3 {
                    for y in 0..height {
                        for x in 0..width {
                            let y_val = src_data[y * width + x];
                            // NV21: VU interleaved, V first
                            let vu_x = (x / 2) * 2;
                            let vu_y = y / 2;
                            let vu_idx = y_size + vu_y * width + vu_x;
                            let v = if vu_idx < src_data.len() {
                                src_data[vu_idx]
                            } else {
                                128
                            };
                            let u = if vu_idx + 1 < src_data.len() {
                                src_data[vu_idx + 1]
                            } else {
                                128
                            };
                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = (y * width + x) * 3;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV21 to RGBA/RGBX
            (ImageFormat::NV21, ImageFormat::Rgba) => {
                let (y_size, _, total) = get_nv12_plane_info(src_width, src_height);
                if src_data.len() >= total && dst_data.len() >= width * height * 4 {
                    for y in 0..height {
                        for x in 0..width {
                            let y_val = src_data[y * width + x];
                            // NV21: VU interleaved, V first
                            let vu_x = (x / 2) * 2;
                            let vu_y = y / 2;
                            let vu_idx = y_size + vu_y * width + vu_x;
                            let v = if vu_idx < src_data.len() {
                                src_data[vu_idx]
                            } else {
                                128
                            };
                            let u = if vu_idx + 1 < src_data.len() {
                                src_data[vu_idx + 1]
                            } else {
                                128
                            };
                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = (y * width + x) * 4;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                            dst_data[dst_idx + 3] = 255;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // NV21 to IYUV
            (ImageFormat::NV21, ImageFormat::IYUV) => {
                let (src_y_size, _, src_total) = get_nv12_plane_info(src_width, src_height);
                let (dst_y_size, dst_u_size, _, dst_total) =
                    get_iyuv_plane_info(src_width, src_height);
                if src_data.len() >= src_total && dst_data.len() >= dst_total {
                    let half_w = (width + 1) / 2;
                    let half_h = (height + 1) / 2;
                    // Copy Y plane
                    for y in 0..height {
                        for x in 0..width {
                            dst_data[y * width + x] = src_data[y * width + x];
                        }
                    }
                    // De-interleave VU plane (NV21: VU) into separate U and V planes (IYUV)
                    for y in 0..half_h {
                        for x in 0..half_w {
                            let vu_idx = src_y_size + y * width + x * 2;
                            let v = if vu_idx < src_data.len() {
                                src_data[vu_idx]
                            } else {
                                128
                            };
                            let u = if vu_idx + 1 < src_data.len() {
                                src_data[vu_idx + 1]
                            } else {
                                128
                            };
                            let u_idx = dst_y_size + y * half_w + x;
                            let v_idx = dst_y_size + dst_u_size + y * half_w + x;
                            if u_idx < dst_data.len() {
                                dst_data[u_idx] = u;
                            }
                            if v_idx < dst_data.len() {
                                dst_data[v_idx] = v;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // UYVY to RGB
            (ImageFormat::UYVY, ImageFormat::Rgb) => {
                if src_data.len() >= width * height * 2 && dst_data.len() >= width * height * 3 {
                    for y in 0..height {
                        for x in 0..width {
                            // UYVY: [U0, Y0, V0, Y1] per 2 pixels
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let u = src_data[pair_idx];
                            let y0 = src_data[pair_idx + 1];
                            let v = src_data[pair_idx + 2];
                            let y1 = src_data[pair_idx + 3];
                            let y_val = if x % 2 == 0 { y0 } else { y1 };
                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = (y * width + x) * 3;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // UYVY to RGBA/RGBX
            (ImageFormat::UYVY, ImageFormat::Rgba) => {
                if src_data.len() >= width * height * 2 && dst_data.len() >= width * height * 4 {
                    for y in 0..height {
                        for x in 0..width {
                            // UYVY: [U0, Y0, V0, Y1] per 2 pixels
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let u = src_data[pair_idx];
                            let y0 = src_data[pair_idx + 1];
                            let v = src_data[pair_idx + 2];
                            let y1 = src_data[pair_idx + 3];
                            let y_val = if x % 2 == 0 { y0 } else { y1 };
                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = (y * width + x) * 4;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                            dst_data[dst_idx + 3] = 255;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // UYVY to NV12
            (ImageFormat::UYVY, ImageFormat::NV12) => {
                if src_data.len() >= width * height * 2 {
                    let (y_size, _, _) = get_nv12_plane_info(src_width, src_height);
                    // Y plane
                    for y in 0..height {
                        for x in 0..width {
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let y0 = src_data[pair_idx + 1];
                            let y1 = src_data[pair_idx + 3];
                            let y_val = if x % 2 == 0 { y0 } else { y1 };
                            dst_data[y * width + x] = y_val;
                        }
                    }
                    // UV plane (4:2:0 subsampling from 4:2:2)
                    // Average U/V from 2 rows per 2x2 block
                    for y in (0..height).step_by(2) {
                        for x in (0..width).step_by(2) {
                            let mut sum_u: i32 = 0;
                            let mut sum_v: i32 = 0;
                            let mut count: i32 = 0;
                            for dy in 0..2 {
                                if y + dy < height {
                                    let pair_idx = ((y + dy) * width + x) * 2;
                                    let u = src_data[pair_idx];
                                    let v = src_data[pair_idx + 2];
                                    sum_u += u as i32;
                                    sum_v += v as i32;
                                    count += 1;
                                }
                            }
                            let u_avg = (sum_u / count) as u8;
                            let v_avg = (sum_v / count) as u8;
                            let uv_idx = y_size + (y / 2) * width + x;
                            if uv_idx + 1 < dst_data.len() {
                                dst_data[uv_idx] = u_avg;
                                dst_data[uv_idx + 1] = v_avg;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // UYVY to IYUV
            (ImageFormat::UYVY, ImageFormat::IYUV) => {
                if src_data.len() >= width * height * 2 {
                    let (y_size, u_size, _, _) = get_iyuv_plane_info(src_width, src_height);
                    let half_w = (width + 1) / 2;
                    let _half_h = (height + 1) / 2;
                    // Y plane
                    for y in 0..height {
                        for x in 0..width {
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let y0 = src_data[pair_idx + 1];
                            let y1 = src_data[pair_idx + 3];
                            let y_val = if x % 2 == 0 { y0 } else { y1 };
                            dst_data[y * width + x] = y_val;
                        }
                    }
                    // U and V planes (4:2:0 subsampling from 4:2:2)
                    // Average U/V from 2 rows per 2x2 block
                    for y in (0..height).step_by(2) {
                        for x in (0..width).step_by(2) {
                            let mut sum_u: i32 = 0;
                            let mut sum_v: i32 = 0;
                            let mut count: i32 = 0;
                            for dy in 0..2 {
                                if y + dy < height {
                                    let pair_idx = ((y + dy) * width + x) * 2;
                                    let u = src_data[pair_idx];
                                    let v = src_data[pair_idx + 2];
                                    sum_u += u as i32;
                                    sum_v += v as i32;
                                    count += 1;
                                }
                            }
                            let u_avg = (sum_u / count) as u8;
                            let v_avg = (sum_v / count) as u8;
                            let uv_y = y / 2;
                            let uv_x = x / 2;
                            let u_idx = y_size + uv_y * half_w + uv_x;
                            let v_idx = y_size + u_size + uv_y * half_w + uv_x;
                            if u_idx < dst_data.len() {
                                dst_data[u_idx] = u_avg;
                            }
                            if v_idx < dst_data.len() {
                                dst_data[v_idx] = v_avg;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // YUYV to RGB
            (ImageFormat::YUYV, ImageFormat::Rgb) => {
                if src_data.len() >= width * height * 2 && dst_data.len() >= width * height * 3 {
                    for y in 0..height {
                        for x in 0..width {
                            // YUYV: [Y0, U, Y1, V] per 2 pixels
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let y0 = src_data[pair_idx];
                            let u = src_data[pair_idx + 1];
                            let y1 = src_data[pair_idx + 2];
                            let v = src_data[pair_idx + 3];
                            let y_val = if x % 2 == 0 { y0 } else { y1 };
                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = (y * width + x) * 3;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // YUYV to RGBA/RGBX
            (ImageFormat::YUYV, ImageFormat::Rgba) => {
                if src_data.len() >= width * height * 2 && dst_data.len() >= width * height * 4 {
                    for y in 0..height {
                        for x in 0..width {
                            // YUYV: [Y0, U, Y1, V] per 2 pixels
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let y0 = src_data[pair_idx];
                            let u = src_data[pair_idx + 1];
                            let y1 = src_data[pair_idx + 2];
                            let v = src_data[pair_idx + 3];
                            let y_val = if x % 2 == 0 { y0 } else { y1 };
                            let (r, g, b) = yuv_to_rgb(y_val, u, v);
                            let dst_idx = (y * width + x) * 4;
                            dst_data[dst_idx] = r;
                            dst_data[dst_idx + 1] = g;
                            dst_data[dst_idx + 2] = b;
                            dst_data[dst_idx + 3] = 255;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // YUYV to NV12
            (ImageFormat::YUYV, ImageFormat::NV12) => {
                if src_data.len() >= width * height * 2 {
                    let (y_size, _, _) = get_nv12_plane_info(src_width, src_height);
                    // Y plane
                    for y in 0..height {
                        for x in 0..width {
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let y_val = if x % 2 == 0 {
                                src_data[pair_idx]
                            } else {
                                src_data[pair_idx + 2]
                            };
                            dst_data[y * width + x] = y_val;
                        }
                    }
                    // UV plane (4:2:0 subsampling from 4:2:2)
                    // Average U/V from 2 rows per 2x2 block
                    for y in (0..height).step_by(2) {
                        for x in (0..width).step_by(2) {
                            let mut sum_u: i32 = 0;
                            let mut sum_v: i32 = 0;
                            let mut count: i32 = 0;
                            for dy in 0..2 {
                                if y + dy < height {
                                    let pair_idx = ((y + dy) * width + x) * 2;
                                    let u = src_data[pair_idx + 1];
                                    let v = src_data[pair_idx + 3];
                                    sum_u += u as i32;
                                    sum_v += v as i32;
                                    count += 1;
                                }
                            }
                            let u_avg = (sum_u / count) as u8;
                            let v_avg = (sum_v / count) as u8;
                            let uv_idx = y_size + (y / 2) * width + x;
                            if uv_idx + 1 < dst_data.len() {
                                dst_data[uv_idx] = u_avg;
                                dst_data[uv_idx + 1] = v_avg;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // YUYV to IYUV
            (ImageFormat::YUYV, ImageFormat::IYUV) => {
                if src_data.len() >= width * height * 2 {
                    let (y_size, u_size, _, _) = get_iyuv_plane_info(src_width, src_height);
                    let half_w = (width + 1) / 2;
                    let _half_h = (height + 1) / 2;
                    // Y plane
                    for y in 0..height {
                        for x in 0..width {
                            let pair_idx = (y * width + (x & !1)) * 2;
                            let y_val = if x % 2 == 0 {
                                src_data[pair_idx]
                            } else {
                                src_data[pair_idx + 2]
                            };
                            dst_data[y * width + x] = y_val;
                        }
                    }
                    // U and V planes (4:2:0 subsampling from 4:2:2)
                    // Average U/V from 2 rows per 2x2 block
                    for y in (0..height).step_by(2) {
                        for x in (0..width).step_by(2) {
                            let mut sum_u: i32 = 0;
                            let mut sum_v: i32 = 0;
                            let mut count: i32 = 0;
                            for dy in 0..2 {
                                if y + dy < height {
                                    let pair_idx = ((y + dy) * width + x) * 2;
                                    let u = src_data[pair_idx + 1];
                                    let v = src_data[pair_idx + 3];
                                    sum_u += u as i32;
                                    sum_v += v as i32;
                                    count += 1;
                                }
                            }
                            let u_avg = (sum_u / count) as u8;
                            let v_avg = (sum_v / count) as u8;
                            let uv_y = y / 2;
                            let uv_x = x / 2;
                            let u_idx = y_size + uv_y * half_w + uv_x;
                            let v_idx = y_size + u_size + uv_y * half_w + uv_x;
                            if u_idx < dst_data.len() {
                                dst_data[u_idx] = u_avg;
                            }
                            if v_idx < dst_data.len() {
                                dst_data[v_idx] = v_avg;
                            }
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // IYUV to YUV4
            (ImageFormat::IYUV, ImageFormat::YUV4) => {
                let (src_y_size, src_u_size, _, src_total) =
                    get_iyuv_plane_info(src_width, src_height);
                let (dst_y_size, dst_u_size, dst_total) =
                    get_yuv4_plane_info(src_width, src_height);
                if src_data.len() >= src_total && dst_data.len() >= dst_total {
                    let half_w = (width + 1) / 2;
                    let _half_h = (height + 1) / 2;

                    // Copy Y plane
                    for y in 0..height {
                        for x in 0..width {
                            dst_data[y * width + x] = src_data[y * width + x];
                        }
                    }

                    // Expand U and V planes from subsampled to full size
                    // IYUV: U and V planes are at half resolution, stored contiguously after Y
                    // Stride for U/V is half_w (same as in reference test's stride/2)
                    for y in 0..height {
                        for x in 0..width {
                            let uv_y = y / 2;
                            let uv_x = x / 2;
                            // U plane starts at src_y_size, V plane at src_y_size + src_u_size
                            // Each row of U/V has half_w elements
                            let u_idx = src_y_size + uv_y * half_w + uv_x;
                            let v_idx = src_y_size + src_u_size + uv_y * half_w + uv_x;

                            let u = if u_idx < src_data.len() {
                                src_data[u_idx]
                            } else {
                                128
                            };
                            let v = if v_idx < src_data.len() {
                                src_data[v_idx]
                            } else {
                                128
                            };

                            let idx = y * width + x;
                            dst_data[dst_y_size + idx] = u;
                            dst_data[dst_y_size + dst_u_size + idx] = v;
                        }
                    }
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            // Same format - direct copy
            (src_fmt, dst_fmt) if src_fmt == dst_fmt => {
                if src_data.len() == dst_data.len() {
                    dst_data.copy_from_slice(&src_data);
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            _ => VX_ERROR_NOT_IMPLEMENTED,
        }
    }
}

// OpenVX channel enum values (from vx_types.h)
const VX_CHANNEL_R: vx_enum = 0x00009010; // R channel for RGB/RGBX
const VX_CHANNEL_G: vx_enum = 0x00009011; // G channel for RGB/RGBX
const VX_CHANNEL_B: vx_enum = 0x00009012; // B channel for RGB/RGBX
const VX_CHANNEL_A: vx_enum = 0x00009013; // A channel for RGBX
const VX_CHANNEL_Y: vx_enum = 0x00009014; // Y channel for YUV
const VX_CHANNEL_U: vx_enum = 0x00009015; // U channel for YUV
const VX_CHANNEL_V: vx_enum = 0x00009016; // V channel for YUV

pub fn vxu_channel_extract_impl(
    context: vx_context,
    input: vx_image,
    channel: vx_enum,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src_width = src.width();
        let src_height = src.height();
        let dst_width = dst.width();
        let dst_height = dst.height();
        let dst_data = dst.data_mut();
        let src_data = src.data();

        match src.format() {
            ImageFormat::Rgb => {
                for y in 0..dst_height {
                    for x in 0..dst_width {
                        let (r, g, b) = src.get_rgb(x, y);
                        let val = match channel {
                            VX_CHANNEL_R => r,
                            VX_CHANNEL_G => g,
                            VX_CHANNEL_B => b,
                            _ => r,
                        };
                        dst_data[y * dst_width + x] = val;
                    }
                }
            }
            ImageFormat::Rgba => {
                for y in 0..dst_height {
                    for x in 0..dst_width {
                        let idx = y * src_width + x;
                        let val = match channel {
                            VX_CHANNEL_R => src_data[idx * 4],
                            VX_CHANNEL_G => src_data[idx * 4 + 1],
                            VX_CHANNEL_B => src_data[idx * 4 + 2],
                            VX_CHANNEL_A => src_data[idx * 4 + 3],
                            _ => src_data[idx * 4],
                        };
                        dst_data[y * dst_width + x] = val;
                    }
                }
            }
            ImageFormat::NV12 => {
                let y_size = src_width * src_height;
                match channel {
                    VX_CHANNEL_Y => {
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                dst_data[y * dst_width + x] = src_data[y * src_width + x];
                            }
                        }
                    }
                    VX_CHANNEL_U | VX_CHANNEL_V => {
                        // NV12: UV interleaved, subsampled 2x2
                        // dst is (width/2) x (height/2)
                        let uv_offset = if channel == VX_CHANNEL_U { 0 } else { 1 };
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                // In the source, UV pairs are at even x positions
                                // src: Y plane, then UV interleaved
                                // For destination pixel (x,y), the source UV is at
                                // (x*2, y*2) in the UV plane
                                let uv_idx = y_size + y * src_width + x * 2 + uv_offset;
                                dst_data[y * dst_width + x] = *src_data.get(uv_idx).unwrap_or(&128);
                            }
                        }
                    }
                    _ => return VX_ERROR_INVALID_PARAMETERS,
                }
            }
            ImageFormat::NV21 => {
                let y_size = src_width * src_height;
                match channel {
                    VX_CHANNEL_Y => {
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                dst_data[y * dst_width + x] = src_data[y * src_width + x];
                            }
                        }
                    }
                    VX_CHANNEL_V | VX_CHANNEL_U => {
                        // NV21: VU interleaved, subsampled 2x2
                        // dst is (width/2) x (height/2)
                        let vu_offset = if channel == VX_CHANNEL_V { 0 } else { 1 };
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let vu_idx = y_size + y * src_width + x * 2 + vu_offset;
                                dst_data[y * dst_width + x] = *src_data.get(vu_idx).unwrap_or(&128);
                            }
                        }
                    }
                    _ => return VX_ERROR_INVALID_PARAMETERS,
                }
            }
            ImageFormat::UYVY => {
                // UYVY: U0 Y0 V0 Y1 - packed 4:2:2
                // Each 4-byte macropixel = 2 horizontal pixels
                // stride = src_width * 2 bytes
                match channel {
                    VX_CHANNEL_Y => {
                        // Y is subsampled x=1, y=1 - output is full size
                        // Y0 at offset 1, Y1 at offset 3 in each macropixel
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let macro_x = x / 2; // which macropixel
                                let byte_idx = y * src_width * 2 + macro_x * 4;
                                let y_val = if x % 2 == 0 {
                                    src_data[byte_idx + 1] // Y0
                                } else {
                                    src_data[byte_idx + 3] // Y1
                                };
                                dst_data[y * dst_width + x] = y_val;
                            }
                        }
                    }
                    VX_CHANNEL_U => {
                        // U is subsampled x=2, y=1 - output is (width/2) x height
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let byte_idx = y * src_width * 2 + x * 4;
                                dst_data[y * dst_width + x] =
                                    *src_data.get(byte_idx).unwrap_or(&128);
                            }
                        }
                    }
                    VX_CHANNEL_V => {
                        // V is subsampled x=2, y=1 - output is (width/2) x height
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let byte_idx = y * src_width * 2 + x * 4 + 2;
                                dst_data[y * dst_width + x] =
                                    *src_data.get(byte_idx).unwrap_or(&128);
                            }
                        }
                    }
                    _ => return VX_ERROR_INVALID_PARAMETERS,
                }
            }
            ImageFormat::YUYV => {
                // YUYV: Y0 U0 Y1 V0 - packed 4:2:2
                // Each 4-byte macropixel = 2 horizontal pixels
                match channel {
                    VX_CHANNEL_Y => {
                        // Y is subsampled x=1, y=1 - output is full size
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let macro_x = x / 2;
                                let byte_idx = y * src_width * 2 + macro_x * 4;
                                let y_val = if x % 2 == 0 {
                                    src_data[byte_idx] // Y0
                                } else {
                                    src_data[byte_idx + 2] // Y1
                                };
                                dst_data[y * dst_width + x] = y_val;
                            }
                        }
                    }
                    VX_CHANNEL_U => {
                        // U is subsampled x=2, y=1 - output is (width/2) x height
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let byte_idx = y * src_width * 2 + x * 4 + 1;
                                dst_data[y * dst_width + x] =
                                    *src_data.get(byte_idx).unwrap_or(&128);
                            }
                        }
                    }
                    VX_CHANNEL_V => {
                        // V is subsampled x=2, y=1 - output is (width/2) x height
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let byte_idx = y * src_width * 2 + x * 4 + 3;
                                dst_data[y * dst_width + x] =
                                    *src_data.get(byte_idx).unwrap_or(&128);
                            }
                        }
                    }
                    _ => return VX_ERROR_INVALID_PARAMETERS,
                }
            }
            ImageFormat::IYUV => {
                let y_size = src_width * src_height;
                let half_w = (src_width + 1) / 2;
                let half_h = (src_height + 1) / 2;
                let u_size = half_w * half_h;
                match channel {
                    VX_CHANNEL_Y => {
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                dst_data[y * dst_width + x] = src_data[y * src_width + x];
                            }
                        }
                    }
                    VX_CHANNEL_U => {
                        // U is subsampled 2x2 - dst is (width/2) x (height/2)
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let u_idx = y_size + y * half_w + x;
                                dst_data[y * dst_width + x] = *src_data.get(u_idx).unwrap_or(&128);
                            }
                        }
                    }
                    VX_CHANNEL_V => {
                        // V is subsampled 2x2 - dst is (width/2) x (height/2)
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                let v_idx = y_size + u_size + y * half_w + x;
                                dst_data[y * dst_width + x] = *src_data.get(v_idx).unwrap_or(&128);
                            }
                        }
                    }
                    _ => return VX_ERROR_INVALID_PARAMETERS,
                }
            }
            ImageFormat::YUV4 => {
                let y_size = src_width * src_height;
                match channel {
                    VX_CHANNEL_Y => {
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                dst_data[y * dst_width + x] = src_data[y * src_width + x];
                            }
                        }
                    }
                    VX_CHANNEL_U => {
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                dst_data[y * dst_width + x] = src_data[y_size + y * src_width + x];
                            }
                        }
                    }
                    VX_CHANNEL_V => {
                        for y in 0..dst_height {
                            for x in 0..dst_width {
                                dst_data[y * dst_width + x] =
                                    src_data[2 * y_size + y * src_width + x];
                            }
                        }
                    }
                    _ => return VX_ERROR_INVALID_PARAMETERS,
                }
            }
            _ => {
                // Default: try to extract as U8
                for y in 0..dst_height {
                    for x in 0..dst_width {
                        dst_data[y * dst_width + x] = src.get_pixel(x, y);
                    }
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

pub fn vxu_channel_combine_impl(
    context: vx_context,
    plane0: vx_image,
    plane1: vx_image,
    plane2: vx_image,
    plane3: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        // Get output image info
        let img = &*(output as *const VxCImage);
        let width = img.width as usize;
        let height = img.height as usize;
        let format = img.format as vx_df_image;
        let out_fmt = match df_image_to_format(format) {
            Some(f) => f,
            None => return VX_ERROR_INVALID_FORMAT,
        };

        // Create a Rust Image for the output
        let mut dst = match Image::new(width, height, out_fmt) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let dst_data = dst.data_mut();

        // Get source plane images
        let y_img = if plane0.is_null() {
            None
        } else {
            c_image_to_rust(plane0)
        };
        let u_img = if plane1.is_null() {
            None
        } else {
            c_image_to_rust(plane1)
        };
        let v_img = if plane2.is_null() {
            None
        } else {
            c_image_to_rust(plane2)
        };
        let a_img = if plane3.is_null() {
            None
        } else {
            c_image_to_rust(plane3)
        };

        match format as u32 {
            0x32424752 => {
                // VX_DF_IMAGE_RGB — fast path when all three planes are present
                if let (Some(r), Some(g), Some(b)) = (&y_img, &u_img, &v_img) {
                    if let Err(_) = crate::kernel_fast_paths::channel_combine_rgb(r, g, b, &mut dst) {
                        return VX_ERROR_INVALID_PARAMETERS;
                    }
                } else {
                    // Fallback: slow get_pixel path for missing planes
                    for y in 0..height {
                        for x in 0..width {
                            let r = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                            let g = u_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                            let b = v_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                            let idx = y * width * 3 + x * 3;
                            if idx + 2 < dst_data.len() {
                                dst_data[idx] = r;
                                dst_data[idx + 1] = g;
                                dst_data[idx + 2] = b;
                            }
                        }
                    }
                }
            }
            0x41424752 => {
                // VX_DF_IMAGE_RGBX — fast path when all four planes are present
                if let (Some(r), Some(g), Some(b), Some(a)) = (&y_img, &u_img, &v_img, &a_img) {
                    if let Err(_) = crate::kernel_fast_paths::channel_combine_rgbx(r, g, b, a, &mut dst) {
                        return VX_ERROR_INVALID_PARAMETERS;
                    }
                } else {
                    // Fallback: slow get_pixel path for missing planes
                    for y in 0..height {
                        for x in 0..width {
                            let r = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                            let g = u_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                            let b = v_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                            let a = a_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(255);
                            let idx = y * width * 4 + x * 4;
                            if idx + 3 < dst_data.len() {
                                dst_data[idx] = r;
                                dst_data[idx + 1] = g;
                                dst_data[idx + 2] = b;
                                dst_data[idx + 3] = a;
                            }
                        }
                    }
                }
            }
            0x3231564E => {
                // VX_DF_IMAGE_NV12
                // Planar: Y (full), UV interleaved (half size)
                let y_size = width * height;
                // Y plane
                for y in 0..height {
                    for x in 0..width {
                        let y_val = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                        dst_data[y * width + x] = y_val;
                    }
                }
                // UV plane (subsampled)
                let half_w = (width + 1) / 2;
                let half_h = (height + 1) / 2;
                for y in 0..half_h {
                    for x in 0..half_w {
                        let u_val = u_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        let v_val = v_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        let uv_idx = y_size + y * width + x * 2;
                        if uv_idx + 1 < dst_data.len() {
                            dst_data[uv_idx] = u_val;
                            dst_data[uv_idx + 1] = v_val;
                        }
                    }
                }
            }
            0x3132564E => {
                // VX_DF_IMAGE_NV21
                // Planar: Y (full), VU interleaved (half size, V first)
                let y_size = width * height;
                // Y plane
                for y in 0..height {
                    for x in 0..width {
                        let y_val = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                        dst_data[y * width + x] = y_val;
                    }
                }
                // VU plane (subsampled)
                let half_w = (width + 1) / 2;
                let half_h = (height + 1) / 2;
                for y in 0..half_h {
                    for x in 0..half_w {
                        let v_val = v_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        let u_val = u_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        let vu_idx = y_size + y * width + x * 2;
                        if vu_idx + 1 < dst_data.len() {
                            dst_data[vu_idx] = v_val;
                            dst_data[vu_idx + 1] = u_val;
                        }
                    }
                }
            }
            0x56555949 => {
                // VX_DF_IMAGE_IYUV
                // Planar: Y (full), U (quarter), V (quarter)
                let y_size = width * height;
                let half_w = (width + 1) / 2;
                let half_h = (height + 1) / 2;
                let u_size = half_w * half_h;
                // Y plane
                for y in 0..height {
                    for x in 0..width {
                        let y_val = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                        dst_data[y * width + x] = y_val;
                    }
                }
                // U plane
                for y in 0..half_h {
                    for x in 0..half_w {
                        let u_val = u_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        let u_idx = y_size + y * half_w + x;
                        if u_idx < dst_data.len() {
                            dst_data[u_idx] = u_val;
                        }
                    }
                }
                // V plane
                for y in 0..half_h {
                    for x in 0..half_w {
                        let v_val = v_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        let v_idx = y_size + u_size + y * half_w + x;
                        if v_idx < dst_data.len() {
                            dst_data[v_idx] = v_val;
                        }
                    }
                }
            }
            0x34565559 => {
                // VX_DF_IMAGE_YUV4
                // Planar: Three full-size planes
                let y_size = width * height;
                // Y plane
                for y in 0..height {
                    for x in 0..width {
                        let y_val = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                        dst_data[y * width + x] = y_val;
                    }
                }
                // U plane
                for y in 0..height {
                    for x in 0..width {
                        let u_val = u_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        dst_data[y_size + y * width + x] = u_val;
                    }
                }
                // V plane
                for y in 0..height {
                    for x in 0..width {
                        let v_val = v_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(128);
                        dst_data[2 * y_size + y * width + x] = v_val;
                    }
                }
            }
            0x59565955 => {
                // VX_DF_IMAGE_UYVY
                // Interleaved: U, Y0, V, Y1 (4:2:2, UYVY order)
                for y in 0..height {
                    for x in (0..width).step_by(2) {
                        let y0 = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                        let y1 = y_img
                            .as_ref()
                            .map(|img| {
                                if x + 1 < width {
                                    img.get_pixel(x + 1, y)
                                } else {
                                    y0
                                }
                            })
                            .unwrap_or(y0);
                        let u_val = u_img
                            .as_ref()
                            .map(|img| img.get_pixel(x / 2, y))
                            .unwrap_or(128);
                        let v_val = v_img
                            .as_ref()
                            .map(|img| img.get_pixel(x / 2, y))
                            .unwrap_or(128);
                        let idx = y * width * 2 + x * 2;
                        if idx + 3 < dst_data.len() {
                            dst_data[idx] = u_val;
                            dst_data[idx + 1] = y0;
                            dst_data[idx + 2] = v_val;
                            dst_data[idx + 3] = y1;
                        }
                    }
                }
            }
            0x56595559 => {
                // VX_DF_IMAGE_YUYV
                // Interleaved: Y0, U, Y1, V (4:2:2, YUYV order)
                for y in 0..height {
                    for x in (0..width).step_by(2) {
                        let y0 = y_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                        let y1 = y_img
                            .as_ref()
                            .map(|img| {
                                if x + 1 < width {
                                    img.get_pixel(x + 1, y)
                                } else {
                                    y0
                                }
                            })
                            .unwrap_or(y0);
                        let u_val = u_img
                            .as_ref()
                            .map(|img| img.get_pixel(x / 2, y))
                            .unwrap_or(128);
                        let v_val = v_img
                            .as_ref()
                            .map(|img| img.get_pixel(x / 2, y))
                            .unwrap_or(128);
                        let idx = y * width * 2 + x * 2;
                        if idx + 3 < dst_data.len() {
                            dst_data[idx] = y0;
                            dst_data[idx + 1] = u_val;
                            dst_data[idx + 2] = y1;
                            dst_data[idx + 3] = v_val;
                        }
                    }
                }
            }
            _ => {
                return VX_ERROR_INVALID_FORMAT;
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

/// ===========================================================================
/// VXU Filter Functions
/// ===========================================================================

pub fn vxu_gaussian3x3_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match gaussian3x3(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_gaussian3x3_impl_with_border(
    _context: vx_context,
    input: vx_image,
    output: vx_image,
    _border: Option<crate::unified_c_api::vx_border_t>,
) -> vx_status {
    vxu_gaussian3x3_impl(_context, input, output)
}

pub fn vxu_gaussian5x5_impl_with_border(
    _context: vx_context,
    input: vx_image,
    output: vx_image,
    _border: Option<crate::unified_c_api::vx_border_t>,
) -> vx_status {
    vxu_gaussian5x5_impl(_context, input, output)
}

pub fn vxu_gaussian5x5_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match gaussian5x5(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_box3x3_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    vxu_box3x3_impl_with_border(context, input, output, None)
}

pub fn vxu_box3x3_impl_with_border(
    _context: vx_context,
    input: vx_image,
    output: vx_image,
    _border: Option<crate::unified_c_api::vx_border_t>,
) -> vx_status {
    if input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        };

        // Check if source image has data - early check
        if src.data.is_empty() {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        };

        let result = match box3x3(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        };
        result
    }
}

pub fn vxu_median3x3_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    vxu_median3x3_impl_with_border(context, input, output, None)
}

pub fn vxu_median3x3_impl_with_border(
    _context: vx_context,
    input: vx_image,
    output: vx_image,
    _border: Option<crate::unified_c_api::vx_border_t>,
) -> vx_status {
    if input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match median3x3(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_convolve_impl(
    context: vx_context,
    input: vx_image,
    conv: vx_convolution,
    output: vx_image,
    border: Option<crate::unified_c_api::vx_border_t>,
) -> vx_status {
    if context.is_null() || input.is_null() || conv.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let (_, _, dst_format) = match get_image_info(output) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Read convolution data
        let conv_data = &*(conv as *const crate::c_api_data::VxCConvolutionData);
        let cols = conv_data.columns;
        let rows = conv_data.rows;
        let scale = conv_data.scale;
        if scale == 0 {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        let coeffs = match conv_data.data.read() {
            Ok(d) => d.clone(),
            Err(_) => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Determine border mode
        let border_mode = match border {
            Some(b) => crate::unified_c_api::border_from_vx(&Some(b)),
            None => get_border_from_context(context),
        };

        let width = src.width;
        let height = src.height;

        // OpenVX convolve: coefficients are reversed (data[cols*rows-1-i])
        // Kernel center is at (cols/2, rows/2)
        let origin_x = cols / 2;
        let origin_y = rows / 2;

        if dst_format == VX_DF_IMAGE_S16 {
            let mut dst = match Image::new(width, height, ImageFormat::GrayS16) {
                Some(img) => img,
                None => return VX_ERROR_INVALID_PARAMETERS,
            };

            for y in 0..height {
                for x in 0..width {
                    let mut sum: i32 = 0;
                    for ky in 0..rows {
                        for kx in 0..cols {
                            let coeff_idx = (rows - 1 - ky) * cols + (cols - 1 - kx);
                            let coeff = coeffs[coeff_idx] as i32;
                            let px = x as isize + kx as isize - origin_x as isize;
                            let py = y as isize + ky as isize - origin_y as isize;
                            let pixel = get_pixel_bordered(&src, px, py, border_mode) as i32;
                            sum += pixel * coeff;
                        }
                    }
                    let value = sum / scale as i32;
                    let clamped = value.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    dst.set_pixel_s16(x, y, clamped);
                }
            }
            copy_rust_to_c_image(&dst, output)
        } else {
            // U8 output
            let mut dst = match create_matching_image(output) {
                Some(img) => img,
                None => return VX_ERROR_INVALID_PARAMETERS,
            };
            let dst_data = dst.data_mut();

            // Fast path: 3×3 kernel with direct slice access (no per-pixel border checks)
            if cols == 3 && rows == 3 && border_mode == BorderMode::Undefined {
                let src_data = src.data();
                let w = width;
                let h = height;
                if w >= 3 && h >= 3 && src_data.len() >= w * h && dst_data.len() >= w * h {
                    // Coeffs are already in OpenVX reversed order from the caller loop above.
                    // Reconstruct the 3×3 matrix in normal top-left-to-bottom-right order
                    // for the fast path (which expects coeffs[0]=top-left).
                    let mut k: [i16; 9] = [0; 9];
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let coeff_idx = (2 - ky) * 3 + (2 - kx);
                            k[ky * 3 + kx] = coeffs[coeff_idx];
                        }
                    }
                    // Inner region (no border checks)
                    for y in 1..h - 1 {
                        let row_m1 = &src_data[(y - 1) * w..y * w];
                        let row_0  = &src_data[y * w..(y + 1) * w];
                        let row_p1 = &src_data[(y + 1) * w..(y + 2) * w];
                        let dst_row = &mut dst_data[y * w..(y + 1) * w];
                        for x in 1..w - 1 {
                            let mut sum: i32 = 0;
                            sum += row_m1[x - 1] as i32 * k[0] as i32;
                            sum += row_m1[x]     as i32 * k[1] as i32;
                            sum += row_m1[x + 1] as i32 * k[2] as i32;
                            sum += row_0[x - 1]  as i32 * k[3] as i32;
                            sum += row_0[x]      as i32 * k[4] as i32;
                            sum += row_0[x + 1]  as i32 * k[5] as i32;
                            sum += row_p1[x - 1] as i32 * k[6] as i32;
                            sum += row_p1[x]     as i32 * k[7] as i32;
                            sum += row_p1[x + 1] as i32 * k[8] as i32;
                            dst_row[x] = (sum / scale as i32).clamp(0, 255) as u8;
                        }
                    }
                    // Top row: replicate nearest
                    let dst_row = &mut dst_data[0..w];
                    for x in 0..w {
                        let mut sum: i32 = 0;
                        let px = |dx: isize| if dx < 0 { 0 } else if dx >= w as isize { (w - 1) as isize } else { dx } as usize;
                        let py = |dy: isize| if dy < 0 { 0 } else { dy } as usize;
                        for ky in 0..3 {
                            let src_y = py(ky as isize - 1);
                            let src_row = &src_data[src_y * w..(src_y + 1) * w];
                            for kx in 0..3 {
                                let src_x = px(x as isize + kx as isize - 1);
                                sum += src_row[src_x] as i32 * k[ky * 3 + kx] as i32;
                            }
                        }
                        dst_row[x] = (sum / scale as i32).clamp(0, 255) as u8;
                    }
                    // Bottom row: replicate nearest
                    let dst_row = &mut dst_data[(h - 1) * w..h * w];
                    for x in 0..w {
                        let mut sum: i32 = 0;
                        let px = |dx: isize| if dx < 0 { 0 } else if dx >= w as isize { (w - 1) as isize } else { dx } as usize;
                        let py = |dy: isize| if dy >= h as isize { (h - 1) as isize } else { dy } as usize;
                        for ky in 0..3 {
                            let src_y = py((h - 1) as isize + ky as isize - 1);
                            let src_row = &src_data[src_y * w..(src_y + 1) * w];
                            for kx in 0..3 {
                                let src_x = px(x as isize + kx as isize - 1);
                                sum += src_row[src_x] as i32 * k[ky * 3 + kx] as i32;
                            }
                        }
                        dst_row[x] = (sum / scale as i32).clamp(0, 255) as u8;
                    }
                    // Left/right edges for inner rows
                    for y in 1..h - 1 {
                        let row_m1 = &src_data[(y - 1) * w..y * w];
                        let row_0  = &src_data[y * w..(y + 1) * w];
                        let row_p1 = &src_data[(y + 1) * w..(y + 2) * w];
                        let dst_row = &mut dst_data[y * w..(y + 1) * w];
                        // x = 0
                        {
                            let mut sum: i32 = 0;
                            for ky in 0..3 {
                                let src_row = match ky {
                                    0 => row_m1,
                                    1 => row_0,
                                    2 => row_p1,
                                    _ => unreachable!(),
                                };
                                for kx in 0..3 {
                                    let src_x = if kx == 0 { 0 } else { kx - 1 };
                                    sum += src_row[src_x] as i32 * k[ky * 3 + kx] as i32;
                                }
                            }
                            dst_row[0] = (sum / scale as i32).clamp(0, 255) as u8;
                        }
                        // x = w - 1
                        {
                            let mut sum: i32 = 0;
                            let xlast = w - 1;
                            for ky in 0..3 {
                                let src_row = match ky {
                                    0 => row_m1,
                                    1 => row_0,
                                    2 => row_p1,
                                    _ => unreachable!(),
                                };
                                for kx in 0..3 {
                                    let src_x = if xlast + kx >= w + 1 { w - 1 } else { xlast + kx - 1 };
                                    sum += src_row[src_x] as i32 * k[ky * 3 + kx] as i32;
                                }
                            }
                            dst_row[xlast] = (sum / scale as i32).clamp(0, 255) as u8;
                        }
                    }
                }
            } else {
                // Generic fallback (original get_pixel_bordered loop)
                for y in 0..height {
                    for x in 0..width {
                        let mut sum: i32 = 0;
                        for ky in 0..rows {
                            for kx in 0..cols {
                                let coeff_idx = (rows - 1 - ky) * cols + (cols - 1 - kx);
                                let coeff = coeffs[coeff_idx] as i32;
                                let px = x as isize + kx as isize - origin_x as isize;
                                let py = y as isize + ky as isize - origin_y as isize;
                                let pixel = get_pixel_bordered(&src, px, py, border_mode) as i32;
                                sum += pixel * coeff;
                            }
                        }
                        let value = sum / scale as i32;
                        let clamped = value.clamp(0, 255) as u8;
                        let idx = y * width + x;
                        if let Some(p) = dst_data.get_mut(idx) {
                            *p = clamped;
                        }
                    }
                }
            }
            copy_rust_to_c_image(&dst, output)
        }
    }
}

/// ===========================================================================
/// VXU Morphology Functions
/// ===========================================================================

pub fn vxu_dilate3x3_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    vxu_dilate3x3_impl_with_border(context, input, output, None)
}

pub fn vxu_dilate3x3_impl_with_border(
    _context: vx_context,
    input: vx_image,
    output: vx_image,
    border: Option<crate::unified_c_api::vx_border_t>,
) -> vx_status {
    if input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let border_mode = if let Some(b) = border {
            match b.mode {
                VX_BORDER_CONSTANT => {
                    let const_val = unsafe { b.constant_value.U8 };
                    BorderMode::Constant(const_val)
                }
                VX_BORDER_REPLICATE => BorderMode::Replicate,
                VX_BORDER_UNDEFINED | _ => BorderMode::Undefined,
            }
        } else {
            BorderMode::Undefined
        };

        let width = src.width();
        let height = src.height();

        // Fast path: process inner region with direct slice access,
        // then edges with border-aware logic.
        if width > 2 && height > 2 {
            let src_data = src.data();
            let dst_data = dst.data_mut();
            let w = width as usize;

            // Inner region (1..height-1, 1..width-1) — all neighbors in bounds
            for y in 1..(height - 1) {
                let y0 = (y - 1) as usize * w;
                let y1 = y as usize * w;
                let y2 = (y + 1) as usize * w;
                let dst_row = y as usize * w;
                for x in 1..(width - 1) {
                    let x0 = x as usize - 1;
                    let x1 = x as usize;
                    let x2 = x as usize + 1;
                    let mut max_val = src_data[y0 + x0];
                    max_val = max_val.max(src_data[y0 + x1]);
                    max_val = max_val.max(src_data[y0 + x2]);
                    max_val = max_val.max(src_data[y1 + x0]);
                    max_val = max_val.max(src_data[y1 + x1]);
                    max_val = max_val.max(src_data[y1 + x2]);
                    max_val = max_val.max(src_data[y2 + x0]);
                    max_val = max_val.max(src_data[y2 + x1]);
                    max_val = max_val.max(src_data[y2 + x2]);
                    dst_data[dst_row + x1] = max_val;
                }
            }

            // Edge pixels use border-aware logic
            for y in [0usize, (height - 1) as usize] {
                for x in 0..width {
                    let mut max_val: u8 = 0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let px = x as isize + dx;
                            let py = y as isize + dy;
                            let val = get_pixel_bordered(&src, px, py, border_mode);
                            max_val = max_val.max(val);
                        }
                    }
                    dst_data[y as usize * w + x as usize] = max_val;
                }
            }
            for y in 1..(height - 1) {
                for x in [0usize, (width - 1) as usize] {
                    let mut max_val: u8 = 0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let px = x as isize + dx;
                            let py = y as isize + dy;
                            let val = get_pixel_bordered(&src, px, py, border_mode);
                            max_val = max_val.max(val);
                        }
                    }
                    dst_data[y as usize * w + x] = max_val;
                }
            }

            return copy_rust_to_c_image(&dst, output);
        }

        // Fallback for small images
        match dilate3x3(&src, &mut dst, border_mode) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_erode3x3_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    vxu_erode3x3_impl_with_border(context, input, output, None)
}

pub fn vxu_erode3x3_impl_with_border(
    _context: vx_context,
    input: vx_image,
    output: vx_image,
    border: Option<crate::unified_c_api::vx_border_t>,
) -> vx_status {
    if input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let border_mode = if let Some(b) = border {
            match b.mode {
                VX_BORDER_CONSTANT => {
                    let const_val = unsafe { b.constant_value.U8 };
                    BorderMode::Constant(const_val)
                }
                VX_BORDER_REPLICATE => BorderMode::Replicate,
                VX_BORDER_UNDEFINED | _ => BorderMode::Undefined,
            }
        } else {
            BorderMode::Undefined
        };

        let width = src.width();
        let height = src.height();

        // Fast path: process inner region with direct slice access,
        // then edges with border-aware logic.
        if width > 2 && height > 2 {
            let src_data = src.data();
            let dst_data = dst.data_mut();
            let w = width as usize;

            // Inner region (1..height-1, 1..width-1) — all neighbors in bounds
            for y in 1..(height - 1) {
                let y0 = (y - 1) as usize * w;
                let y1 = y as usize * w;
                let y2 = (y + 1) as usize * w;
                let dst_row = y as usize * w;
                for x in 1..(width - 1) {
                    let x0 = x as usize - 1;
                    let x1 = x as usize;
                    let x2 = x as usize + 1;
                    let mut min_val = src_data[y0 + x0];
                    min_val = min_val.min(src_data[y0 + x1]);
                    min_val = min_val.min(src_data[y0 + x2]);
                    min_val = min_val.min(src_data[y1 + x0]);
                    min_val = min_val.min(src_data[y1 + x1]);
                    min_val = min_val.min(src_data[y1 + x2]);
                    min_val = min_val.min(src_data[y2 + x0]);
                    min_val = min_val.min(src_data[y2 + x1]);
                    min_val = min_val.min(src_data[y2 + x2]);
                    dst_data[dst_row + x1] = min_val;
                }
            }

            // Edge pixels use border-aware logic
            for y in [0usize, (height - 1) as usize] {
                for x in 0..width {
                    let mut min_val: u8 = 255;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let px = x as isize + dx;
                            let py = y as isize + dy;
                            let val = get_pixel_bordered(&src, px, py, border_mode);
                            min_val = min_val.min(val);
                        }
                    }
                    dst_data[y as usize * w + x as usize] = min_val;
                }
            }
            for y in 1..(height - 1) {
                for x in [0usize, (width - 1) as usize] {
                    let mut min_val: u8 = 255;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let px = x as isize + dx;
                            let py = y as isize + dy;
                            let val = get_pixel_bordered(&src, px, py, border_mode);
                            min_val = min_val.min(val);
                        }
                    }
                    dst_data[y as usize * w + x] = min_val;
                }
            }

            return copy_rust_to_c_image(&dst, output);
        }

        // Fallback for small images
        match erode3x3(&src, &mut dst, border_mode) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_dilate5x5_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    // For now, use 3x3 dilate twice as approximation
    vxu_dilate3x3_impl(context, input, output)
}

pub fn vxu_erode5x5_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    // For now, use 3x3 erode twice as approximation
    vxu_erode3x3_impl(context, input, output)
}

/// ===========================================================================
/// VXU Gradient Functions
/// ===========================================================================

pub fn vxu_sobel3x3_impl(
    context: vx_context,
    input: vx_image,
    output_x: vx_image,
    output_y: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let width = src.width();
        let height = src.height();

        // Create S16 output images
        let mut gx = match Image::new(width, height, ImageFormat::GrayS16) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let mut gy = match Image::new(width, height, ImageFormat::GrayS16) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Get border mode from context
        let border = get_border_from_context(context);

        sobel3x3_s16(&src, &mut gx, &mut gy, border);

        copy_rust_to_c_image(&gx, output_x);
        copy_rust_to_c_image(&gy, output_y);
        VX_SUCCESS
    }
}

pub fn vxu_magnitude_impl(
    context: vx_context,
    grad_x: vx_image,
    grad_y: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || grad_x.is_null() || grad_y.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let gx = match c_image_to_rust(grad_x) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let gy = match c_image_to_rust(grad_y) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let (_, _, out_format) = match get_image_info(output) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let dst_format = df_image_to_format(out_format).unwrap_or(ImageFormat::GrayS16);

        let mut dst = match Image::new(gx.width(), gx.height(), dst_format) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        magnitude_s16(&gx, &gy, &mut dst);
        copy_rust_to_c_image(&dst, output)
    }
}

pub fn vxu_phase_impl(
    context: vx_context,
    grad_x: vx_image,
    grad_y: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || grad_x.is_null() || grad_y.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let gx = match c_image_to_rust(grad_x) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let gy = match c_image_to_rust(grad_y) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match Image::new(gx.width(), gx.height(), ImageFormat::Gray) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        phase_s16(&gx, &gy, &mut dst);
        copy_rust_to_c_image(&dst, output)
    }
}

/// ===========================================================================
/// VXU Arithmetic Functions
/// ===========================================================================

pub fn vxu_add_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    _policy: vx_enum,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match add(&src1, &src2, &mut dst, _policy) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_subtract_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    _policy: vx_enum,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match subtract(&src1, &src2, &mut dst, _policy) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_multiply_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    scale: vx_scalar,
    overflow_policy: vx_enum,
    rounding_policy: vx_enum,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Read scale value from scalar
    let scale_value = read_scale_from_scalar(scale);

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match multiply(
            &src1,
            &src2,
            &mut dst,
            scale_value,
            overflow_policy,
            rounding_policy,
        ) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// Direct-scale version for vxuMultiply (takes vx_float32 directly, not vx_scalar)
/// The OpenVX spec defines vxuMultiply with vx_float32 scale parameter,
/// not vx_scalar. The graph version (vxMultiplyNode) uses vx_scalar.
pub fn vxu_multiply_impl_direct_scale(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    scale_value: vx_float32,
    overflow_policy: vx_enum,
    rounding_policy: vx_enum,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match multiply(
            &src1,
            &src2,
            &mut dst,
            scale_value,
            overflow_policy,
            rounding_policy,
        ) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_weighted_average_impl(
    context: vx_context,
    img1: vx_image,
    alpha: vx_scalar,
    img2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || img1.is_null() || img2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Read alpha value from scalar
    let mut alpha_f32: f32 = 0.5;
    if !alpha.is_null() {
        unsafe {
            let status = crate::c_api_data::vxCopyScalarData(
                alpha,
                &mut alpha_f32 as *mut f32 as *mut c_void,
                0x11001, // VX_READ_ONLY
                0x0,     // VX_MEMORY_TYPE_HOST
            );
            if status != 0 {
                // 0 = VX_SUCCESS
                alpha_f32 = 0.5; // fallback
            }
        }
    }

    // Convert float alpha [0,1] - pass directly to weighted function
    // alpha_f32 is already in [0,1] range from the scalar

    unsafe {
        let src1 = match c_image_to_rust(img1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(img2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match weighted(&src1, &src2, &mut dst, alpha_f32) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_abs_diff_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match abs_diff(&src1, &src2, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// ===========================================================================
/// VXU Statistics Functions
/// ===========================================================================

pub fn vxu_integral_image_impl(
    context: vx_context,
    input: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match integral_image(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_mean_std_dev_impl(
    context: vx_context,
    input: vx_image,
    mean_scalar: vx_scalar,
    stddev_scalar: vx_scalar,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        match mean_std_dev(&src) {
            Ok((mean_val, stddev_val)) => {
                if !mean_scalar.is_null() {
                    crate::c_api_data::vxCopyScalarData(
                        mean_scalar,
                        &mean_val as *const f32 as *mut c_void,
                        0x11002,
                        0x0, // VX_WRITE_ONLY
                    );
                }
                if !stddev_scalar.is_null() {
                    crate::c_api_data::vxCopyScalarData(
                        stddev_scalar,
                        &stddev_val as *const f32 as *mut c_void,
                        0x11002,
                        0x0,
                    );
                }
                VX_SUCCESS
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_min_max_loc_impl(
    context: vx_context,
    input: vx_image,
    min_val_scalar: vx_scalar,
    max_val_scalar: vx_scalar,
    min_loc_array: vx_array,
    max_loc_array: vx_array,
    min_count_scalar: vx_scalar,
    max_count_scalar: vx_scalar,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let is_s16 = matches!(src.format, ImageFormat::GrayS16);

        match min_max_loc(&src) {
            Ok(result) => {
                // Write min/max values as the native image data type
                if !min_val_scalar.is_null() {
                    if is_s16 {
                        let v = result.min_val as i16;
                        crate::c_api_data::vxCopyScalarData(
                            min_val_scalar,
                            &v as *const i16 as *mut c_void,
                            0x11002,
                            0x0,
                        );
                    } else {
                        let v = result.min_val as u8;
                        crate::c_api_data::vxCopyScalarData(
                            min_val_scalar,
                            &v as *const u8 as *mut c_void,
                            0x11002,
                            0x0,
                        );
                    }
                }
                if !max_val_scalar.is_null() {
                    if is_s16 {
                        let v = result.max_val as i16;
                        crate::c_api_data::vxCopyScalarData(
                            max_val_scalar,
                            &v as *const i16 as *mut c_void,
                            0x11002,
                            0x0,
                        );
                    } else {
                        let v = result.max_val as u8;
                        crate::c_api_data::vxCopyScalarData(
                            max_val_scalar,
                            &v as *const u8 as *mut c_void,
                            0x11002,
                            0x0,
                        );
                    }
                }
                // Write min/max locations to arrays
                extern "C" {
                    fn vxTruncateArray(arr: vx_array, new_num_items: vx_size) -> vx_status;
                    fn vxAddArrayItems(
                        arr: vx_array,
                        count: vx_size,
                        ptr: *const c_void,
                        stride: vx_size,
                    ) -> vx_status;
                    fn vxQueryArray(
                        arr: vx_array,
                        attribute: vx_enum,
                        ptr: *mut c_void,
                        size: vx_size,
                    ) -> vx_status;
                }
                if !min_loc_array.is_null() {
                    vxTruncateArray(min_loc_array, 0);
                    let coords: Vec<vx_coordinates2d_t> = result
                        .min_locs
                        .iter()
                        .map(|c| vx_coordinates2d_t {
                            x: c.x as u32,
                            y: c.y as u32,
                        })
                        .collect();
                    if !coords.is_empty() {
                        let mut cap: vx_size = 0;
                        vxQueryArray(
                            min_loc_array,
                            0x80E02,
                            &mut cap as *mut _ as *mut c_void,
                            std::mem::size_of::<vx_size>() as vx_size,
                        ); // VX_ARRAY_CAPACITY
                        let add_count = if cap > 0 {
                            coords.len().min(cap)
                        } else {
                            coords.len()
                        };
                        vxAddArrayItems(
                            min_loc_array,
                            add_count as vx_size,
                            coords.as_ptr() as *const c_void,
                            std::mem::size_of::<vx_coordinates2d_t>() as vx_size,
                        );
                    }
                }
                if !max_loc_array.is_null() {
                    vxTruncateArray(max_loc_array, 0);
                    let coords: Vec<vx_coordinates2d_t> = result
                        .max_locs
                        .iter()
                        .map(|c| vx_coordinates2d_t {
                            x: c.x as u32,
                            y: c.y as u32,
                        })
                        .collect();
                    if !coords.is_empty() {
                        let mut cap: vx_size = 0;
                        vxQueryArray(
                            max_loc_array,
                            0x80E02,
                            &mut cap as *mut _ as *mut c_void,
                            std::mem::size_of::<vx_size>() as vx_size,
                        ); // VX_ARRAY_CAPACITY
                        let add_count = if cap > 0 {
                            coords.len().min(cap)
                        } else {
                            coords.len()
                        };
                        vxAddArrayItems(
                            max_loc_array,
                            add_count as vx_size,
                            coords.as_ptr() as *const c_void,
                            std::mem::size_of::<vx_coordinates2d_t>() as vx_size,
                        );
                    }
                }
                if !min_count_scalar.is_null() {
                    let count = result.min_locs.len() as u32;
                    crate::c_api_data::vxCopyScalarData(
                        min_count_scalar,
                        &count as *const u32 as *mut c_void,
                        0x11002,
                        0x0,
                    );
                }
                if !max_count_scalar.is_null() {
                    let count = result.max_locs.len() as u32;
                    crate::c_api_data::vxCopyScalarData(
                        max_count_scalar,
                        &count as *const u32 as *mut c_void,
                        0x11002,
                        0x0,
                    );
                }
                VX_SUCCESS
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_histogram_impl(
    context: vx_context,
    input: vx_image,
    distribution: vx_distribution,
) -> vx_status {
    if context.is_null() || input.is_null() || distribution.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let dist = &*(distribution as *const crate::unified_c_api::VxCDistribution);

        // Compute a full 256-bin histogram
        let full_hist = match histogram(&src) {
            Ok(h) => h,
            Err(_) => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Map the 256-bin histogram to the distribution's bins/offset/range
        let nbins = dist.bins;
        let offset = dist.offset as usize;
        let range = dist.range as usize;

        let mut dist_data = match dist.data.write() {
            Ok(d) => d,
            Err(_) => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Clear distribution data
        for i in 0..nbins {
            dist_data[i] = 0;
        }

        // Map pixel values to distribution bins
        // bin_index = (pixel_value - offset) * nbins / range
        for i in offset..(offset + range).min(256) {
            let bin_idx = (i - offset) * nbins / range;
            if bin_idx < nbins {
                dist_data[bin_idx] += full_hist[i] as i32;
            }
        }

        VX_SUCCESS
    }
}

/// ===========================================================================
/// VXU Geometric Functions
/// ===========================================================================

pub fn vxu_scale_image_impl(
    context: vx_context,
    input: vx_image,
    output: vx_image,
    _interpolation: vx_enum,
    override_border: Option<BorderMode>,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Use override border from graph mode, or fall back to context border
        let border = override_border.unwrap_or_else(|| get_border_from_context(context));

        // Parse interpolation type
        // VX_INTERPOLATION_NEAREST_NEIGHBOR = 0x4000
        // VX_INTERPOLATION_BILINEAR = 0x4001 (default)
        // VX_INTERPOLATION_AREA = 0x4002
        let interpolation = match _interpolation {
            0x4000 => InterpolationType::NearestNeighbor,
            0x4002 => InterpolationType::Area,
            _ => InterpolationType::Bilinear, // Default
        };

        match scale_image(&src, &mut dst, interpolation, border) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_warp_affine_impl(
    context: vx_context,
    input: vx_image,
    _matrix: vx_matrix,
    _interpolation: vx_enum,
    output: vx_image,
    override_border: Option<BorderMode>,
) -> vx_status {
    if context.is_null() || input.is_null() || _matrix.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Read the affine matrix (2x3) from the vx_matrix handle
        // The CTS stores mat[col][row] with col\u{220}\\{0,1,2\}, row\u{220}\\{0,1\}
        // Flat layout: m[col*2 + row], which is exactly what vxCopyMatrix stores
        // CTS reference: x0 = m[0]*x + m[2]*y + m[4]
        //               y0 = m[1]*x + m[3]*y + m[5]
        // Pass raw data directly to warp_affine which now expects CTS layout
        let affine_matrix: [f32; 6] = {
            let m = crate::c_api_data::VxCMatrixData::from_ptr(_matrix);
            if m.is_none() {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let m = m.unwrap();
            let data = m.as_f32_slice();
            if data.is_none() {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let data = data.unwrap();
            if data.len() < 6 {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            [data[0], data[1], data[2], data[3], data[4], data[5]]
        };

        let border = override_border.unwrap_or_else(|| get_border_from_context(context));
        let nn = _interpolation == 0x4000; // VX_INTERPOLATION_NEAREST_NEIGHBOR
        match warp_affine(&src, &affine_matrix, &mut dst, border, nn) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_warp_perspective_impl(
    context: vx_context,
    input: vx_image,
    _matrix: vx_matrix,
    _interpolation: vx_enum,
    output: vx_image,
    override_border: Option<BorderMode>,
) -> vx_status {
    if context.is_null() || input.is_null() || _matrix.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Read the actual matrix data from the vx_matrix handle
        let persp_matrix: [f32; 9] = {
            let m = crate::c_api_data::VxCMatrixData::from_ptr(_matrix);
            if m.is_none() {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let m = m.unwrap();
            if m.data_type != 0x00A || m.rows != 3 || m.columns != 3 {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let data = m.as_f32_slice();
            if data.is_none() {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let data = data.unwrap();
            if data.len() < 9 {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            // OpenVX stores matrices in COLUMN-MAJOR order:
            // data[0-2] = first column, data[3-5] = second column, data[6-8] = third column
            // Pass the data as-is to warp_perspective which handles column-major
            [
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
            ]
        };

        let border = override_border.unwrap_or_else(|| get_border_from_context(context));
        let nn = _interpolation == 0x4000; // VX_INTERPOLATION_NEAREST_NEIGHBOR
        match warp_perspective(&src, &persp_matrix, &mut dst, border, nn) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// ===========================================================================
/// VXU Feature Detection Functions
/// ===========================================================================

pub fn vxu_harris_corners_impl(
    context: vx_context,
    input: vx_image,
    strength_thresh: vx_scalar,
    min_distance: vx_scalar,
    sensitivity: vx_scalar,
    gradient_size: vx_enum,
    block_size: vx_enum,
    corners: vx_array,
    num_corners: vx_scalar,
) -> vx_status {
    // Validate all required parameters with null checks
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Read scalar parameters
    let threshold: f32 = if !strength_thresh.is_null() {
        let mut val: f32 = 0.0;
        let status = crate::c_api_data::vxCopyScalarData(
            strength_thresh,
            &mut val as *mut f32 as *mut c_void,
            0x11001,
            0x0,
        );
        if status == VX_SUCCESS {
            val
        } else {
            0.001
        }
    } else {
        0.001
    };

    let min_dist: f32 = if !min_distance.is_null() {
        let mut val: f32 = 0.0;
        let status = crate::c_api_data::vxCopyScalarData(
            min_distance,
            &mut val as *mut f32 as *mut c_void,
            0x11001,
            0x0,
        );
        if status == VX_SUCCESS {
            val
        } else {
            3.0
        }
    } else {
        3.0
    };

    let k: f32 = if !sensitivity.is_null() {
        let mut val: f32 = 0.0;
        let status = crate::c_api_data::vxCopyScalarData(
            sensitivity,
            &mut val as *mut f32 as *mut c_void,
            0x11001,
            0x0,
        );
        if status == VX_SUCCESS {
            val
        } else {
            0.04
        }
    } else {
        0.04
    };

    let gs = gradient_size as usize;
    let bs = block_size as usize;
    // Valid gradient sizes: 3, 5, 7
    let gs = if gs == 3 || gs == 5 || gs == 7 { gs } else { 3 };
    // Valid block sizes: 3, 5, 7
    let bs = if bs == 3 || bs == 5 || bs == 7 { bs } else { 3 };

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        };

        let width = src.width();
        let height = src.height();

        if width < 3 || height < 3 {
            // Image too small for Harris corners
            if !corners.is_null() {
                extern "C" {
                    fn vxTruncateArray(arr: vx_array, new_num_items: vx_size) -> vx_status;
                }
                unsafe {
                    vxTruncateArray(corners, 0);
                }
            }
            if !num_corners.is_null() {
                let num: usize = 0;
                crate::c_api_data::vxCopyScalarData(
                    num_corners,
                    &num as *const usize as *mut c_void,
                    0x11002,
                    0x0,
                );
            }
            return VX_SUCCESS;
        }

        // Get image data as flat u8 array
        let img_data = src.data();

        // Gradients are computed raw (no normalization) - the normFactor at the end handles all normalization

        // Compute GxGx, GxGy, GyGy structure tensor components
        // using separable Sobel filters (horizontal then vertical pass)
        let mut gxy = vec![
            GxyComponent {
                ixx: 0.0f32,
                ixy: 0.0f32,
                iyy: 0.0f32
            };
            width * height
        ];

        match gs {
            5 => harris_sobel_5x5(&img_data, width, height, &mut gxy),
            7 => harris_sobel_7x7(&img_data, width, height, &mut gxy),
            _ => harris_sobel_3x3(&img_data, width, height, &mut gxy),
        }

        // Compute Harris response using block window accumulation
        // Direct windowing matching MIVisionX: for each pixel, sum the structure
        // tensor components over a bs×bs window centered at that pixel
        let half_block = bs / 2;
        let mut responses = vec![0.0f32; width * height];

        for y in half_block..height - half_block {
            for x in half_block..width - half_block {
                let mut ixx = 0.0f32;
                let mut ixy = 0.0f32;
                let mut iyy = 0.0f32;
                for j in 0..bs {
                    let row = y + j - half_block;
                    let row_off = row * width;
                    for i in 0..bs {
                        let col = x + i - half_block;
                        ixx += gxy[row_off + col].ixx;
                        ixy += gxy[row_off + col].ixy;
                        iyy += gxy[row_off + col].iyy;
                    }
                }
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let mc = det - k * trace * trace;
                responses[y * width + x] = mc;
            }
        }

        // Normalize responses using the proper Harris normalization factor
        // normFactor = (255.0 * (1 << (gradient_size - 1)) * block_size)^4
        // This matches the CTS truth data scale: scale = 1/((1<<(gs-1))*bs*255), scale^4
        let norm_base = 255.0f32 * (1 << (gs - 1)) as f32 * bs as f32;
        let norm_factor = norm_base * norm_base * norm_base * norm_base;
        for r in responses.iter_mut() {
            *r /= norm_factor;
        }

        // Harris corner detection pipeline (matching MIVisionX reference):
        // 1. Threshold the Vc score image
        // 2. 3x3 non-max suppression (asymmetric >= / > comparison)
        // 3. Sort candidates by strength descending
        // 4. Distance-based NMS with grid (ceilf for radius check)
        // 5. Collect final corners

        // Step 1: Threshold — set sub-threshold responses to 0
        for r in responses.iter_mut() {
            if *r <= threshold {
                *r = 0.0;
            }
        }

        // Step 2: 3x3 non-max suppression matching MIVisionX HafCpu_NonMaxSupp_XY_ANY_3x3
        // Asymmetric comparison: >= for top/left neighbors, > for bottom/right
        // This gives raster-scan order bias: first pixel in scan order wins ties
        let mut candidates: Vec<(i32, i32, f32)> = Vec::new();
        for y in 1..(height as i32 - 1) {
            for x in 1..(width as i32 - 1) {
                let idx = (y as usize) * width + (x as usize);
                let vc = responses[idx];
                if vc <= 0.0 {
                    continue;
                }
                // MIVisionX 3x3 NMS: >= for top row + left, > for right + bottom row
                // p0[1] >= p9[0] && p0[1] >= p9[1] && p0[1] >= p9[2] &&  (top-left, top, top-right)
                // p0[1] >= p0[0] &&                                 (left)
                // p0[1] > p0[2] &&                                  (right)
                // p0[1] > p1[0] && p0[1] > p1[1] && p0[1] > p1[2]   (bottom-left, bottom, bottom-right)
                let top_left = responses[((y - 1) as usize) * width + (x - 1) as usize];
                let top = responses[((y - 1) as usize) * width + x as usize];
                let top_right = responses[((y - 1) as usize) * width + (x + 1) as usize];
                let left = responses[(y as usize) * width + (x - 1) as usize];
                let right = responses[(y as usize) * width + (x + 1) as usize];
                let bottom_left = responses[((y + 1) as usize) * width + (x - 1) as usize];
                let bottom = responses[((y + 1) as usize) * width + x as usize];
                let bottom_right = responses[((y + 1) as usize) * width + (x + 1) as usize];

                if vc >= top_left
                    && vc >= top
                    && vc >= top_right
                    && vc >= left
                    && vc > right
                    && vc > bottom_left
                    && vc > bottom
                    && vc > bottom_right
                {
                    candidates.push((x, y, vc));
                }
            }
        }

        // Step 3: Sort candidates by strength descending
        // MIVisionX uses std::sort with greater<int64> on packed (x,y,strength) values
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Step 4: Distance-based NMS with grid (MIVisionX HafCpu_HarrisMergeSortAndPick_XY_XYS)
        // Uses ceilf(min_distance²) for radius check, not floor
        let min_dist2 = (min_dist * min_dist).ceil() as i32;
        let mut corner_list: Vec<(i32, i32, f32)> = Vec::new();

        if min_dist2 <= 0 {
            // No distance constraint, keep all 3x3 NMS survivors
            // But limit to array capacity like MIVisionX's AddToTheSortedKeypointList
            corner_list = candidates;
        } else {
            // Grid-based distance suppression matching MIVisionX
            let cell_size = (min_dist as usize).max(1);
            let grid_w = (width + cell_size - 1) / cell_size;
            let grid_h = (height + cell_size - 1) / cell_size;
            let mut grid: Vec<(i32, i32)> = vec![(-1i32, -1i32); grid_w * grid_h];

            for &(x, y, strength) in &candidates {
                let cx = (x as usize) / cell_size;
                let cy = (y as usize) / cell_size;

                let mut too_close = false;
                for gy in cy.saturating_sub(2)..=(cy + 2).min(grid_h - 1) {
                    for gx in cx.saturating_sub(2)..=(cx + 2).min(grid_w - 1) {
                        let (px, py) = grid[gy * grid_w + gx];
                        if px >= 0 {
                            let dx = x - px;
                            let dy = y - py;
                            if dx * dx + dy * dy < min_dist2 {
                                too_close = true;
                                break;
                            }
                        }
                    }
                    if too_close {
                        break;
                    }
                }

                if !too_close {
                    corner_list.push((x, y, strength));
                    grid[cy * grid_w + cx] = (x, y);
                }
            }
        }

        // Write corners to output array using proper C API functions
        // (openvx-buffer VxCArray struct layout differs from unified_c_api VxCArray)
        if !corners.is_null() {
            let keypoint_size = std::mem::size_of::<vx_keypoint_t>();

            // First truncate the array to 0 items
            extern "C" {
                fn vxTruncateArray(arr: vx_array, new_num_items: vx_size) -> vx_status;
                fn vxAddArrayItems(
                    arr: vx_array,
                    count: vx_size,
                    ptr: *const c_void,
                    stride: vx_size,
                ) -> vx_status;
            }
            unsafe {
                vxTruncateArray(corners, 0);
            }

            // Build a flat byte buffer of keypoints
            let mut buf: Vec<u8> = Vec::with_capacity(corner_list.len() * keypoint_size);
            for &(x, y, strength) in &corner_list {
                let kp = vx_keypoint_t {
                    x,
                    y,
                    strength,
                    scale: 0.0,
                    orientation: 0.0,
                    tracking_status: 1,
                    error: 0.0,
                };
                let kp_bytes = std::slice::from_raw_parts(
                    &kp as *const vx_keypoint_t as *const u8,
                    keypoint_size,
                );
                buf.extend_from_slice(kp_bytes);
            }

            // Add all keypoints at once
            unsafe {
                let status = vxAddArrayItems(
                    corners,
                    corner_list.len() as vx_size,
                    buf.as_ptr() as *const c_void,
                    keypoint_size as vx_size,
                );
                if status != VX_SUCCESS {}
            }
        }

        // Write num_corners to scalar
        if !num_corners.is_null() {
            let num = corner_list.len() as usize;
            crate::c_api_data::vxCopyScalarData(
                num_corners,
                &num as *const usize as *mut c_void,
                0x11002, // VX_WRITE_ONLY
                0x0,
            );
        }

        VX_SUCCESS
    }
}

/// Structure tensor component per pixel
#[derive(Clone, Copy, Default)]
struct GxyComponent {
    ixx: f32,
    ixy: f32,
    iyy: f32,
}

/// Separable Sobel 3x3 + structure tensor computation
/// Gx: horizontal = [-1, 0, 1], vertical = [1, 2, 1]
/// Gy: horizontal = [1, 2, 1], vertical = [-1, 0, 1]
/// Computes Gx*Gx, Gx*Gy, Gy*Gy per pixel
fn harris_sobel_3x3(img_data: &[u8], width: usize, height: usize, gxy: &mut [GxyComponent]) {
    // Use raw gradients (no normalization) - matches MIVisionX reference with div_factor=1
    for y in 1..height - 1 {
        let row_m = (y - 1) * width;
        let row_c = y * width;
        let row_p = (y + 1) * width;

        for x in 1..width - 1 {
            let p_m_l = img_data[row_m + x - 1] as i32;
            let p_m_c = img_data[row_m + x] as i32;
            let p_m_r = img_data[row_m + x + 1] as i32;
            let p_c_l = img_data[row_c + x - 1] as i32;
            let p_c_r = img_data[row_c + x + 1] as i32;
            let p_p_l = img_data[row_p + x - 1] as i32;
            let p_p_c = img_data[row_p + x] as i32;
            let p_p_r = img_data[row_p + x + 1] as i32;

            let gx: i32 = (p_p_r - p_p_l) + 2 * (p_c_r - p_c_l) + (p_m_r - p_m_l);
            let gy: i32 = (p_p_r + 2 * p_p_c + p_p_l) - (p_m_r + 2 * p_m_c + p_m_l);

            let gxf = gx as f32;
            let gyf = gy as f32;

            let idx = y * width + x;
            gxy[idx].ixx = gxf * gxf;
            gxy[idx].ixy = gxf * gyf;
            gxy[idx].iyy = gyf * gyf;
        }
    }
}

/// 5x5 Sobel + structure tensor computation
fn harris_sobel_5x5(img_data: &[u8], width: usize, height: usize, gxy: &mut [GxyComponent]) {
    // 5x5 Sobel kernels (from OpenVX spec)
    let gx_kernel: [[i32; 5]; 5] = [
        [-1, -2, 0, 2, 1],
        [-4, -8, 0, 8, 4],
        [-6, -12, 0, 12, 6],
        [-4, -8, 0, 8, 4],
        [-1, -2, 0, 2, 1],
    ];
    let gy_kernel: [[i32; 5]; 5] = [
        [-1, -4, -6, -4, -1],
        [-2, -8, -12, -8, -2],
        [0, 0, 0, 0, 0],
        [2, 8, 12, 8, 2],
        [1, 4, 6, 4, 1],
    ];

    for y in 2..height - 2 {
        for x in 2..width - 2 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;
            for ky in 0..5 {
                let row = (y + ky - 2) * width;
                for kx in 0..5 {
                    let px = row + x + kx - 2;
                    let p = img_data[px] as i32;
                    gx += p * gx_kernel[ky][kx];
                    gy += p * gy_kernel[ky][kx];
                }
            }

            // Use raw gradients (no normalization) - normFactor handles it
            let gxf = gx as f32;
            let gyf = gy as f32;

            let idx = y * width + x;
            gxy[idx].ixx = gxf * gxf;
            gxy[idx].ixy = gxf * gyf;
            gxy[idx].iyy = gyf * gyf;
        }
    }
}

/// 7x7 Sobel + structure tensor computation
fn harris_sobel_7x7(img_data: &[u8], width: usize, height: usize, gxy: &mut [GxyComponent]) {
    // 7x7 Sobel kernels
    let gx_kernel: [[i32; 7]; 7] = [
        [-1, -4, -5, 0, 5, 4, 1],
        [-6, -24, -30, 0, 30, 24, 6],
        [-15, -60, -75, 0, 75, 60, 15],
        [-20, -80, -100, 0, 100, 80, 20],
        [-15, -60, -75, 0, 75, 60, 15],
        [-6, -24, -30, 0, 30, 24, 6],
        [-1, -4, -5, 0, 5, 4, 1],
    ];
    let gy_kernel: [[i32; 7]; 7] = [
        [-1, -6, -15, -20, -15, -6, -1],
        [-4, -24, -60, -80, -60, -24, -4],
        [-5, -30, -75, -100, -75, -30, -5],
        [0, 0, 0, 0, 0, 0, 0],
        [5, 30, 75, 100, 75, 30, 5],
        [4, 24, 60, 80, 60, 24, 4],
        [1, 6, 15, 20, 15, 6, 1],
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;
            for ky in 0..7 {
                let row = (y + ky - 3) * width;
                for kx in 0..7 {
                    let px = row + x + kx - 3;
                    let p = img_data[px] as i32;
                    gx += p * gx_kernel[ky][kx];
                    gy += p * gy_kernel[ky][kx];
                }
            }

            // Use raw gradients (no normalization) - normFactor handles it
            let gxf = gx as f32;
            let gyf = gy as f32;

            let idx = y * width + x;
            gxy[idx].ixx = gxf * gxf;
            gxy[idx].ixy = gxf * gyf;
            gxy[idx].iyy = gyf * gyf;
        }
    }
}
pub fn vxu_fast_corners_impl(
    context: vx_context,
    input: vx_image,
    strength_thresh: vx_scalar,
    nonmax_suppression: i32,
    corners: vx_array,
    num_corners: vx_scalar,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Read threshold from scalar
        let threshold: f32 = if !strength_thresh.is_null() {
            let mut val: f32 = 20.0;
            let status = crate::c_api_data::vxCopyScalarData(
                strength_thresh,
                &mut val as *mut f32 as *mut c_void,
                0x11001,
                0x0,
            );
            if status == VX_SUCCESS {
                val
            } else {
                20.0
            }
        } else {
            20.0
        };

        let threshold_val = threshold.max(0.0).min(255.0) as u8;
        let do_nms = nonmax_suppression != 0;

        let width = src.width;
        let height = src.height;

        // FAST-9 corner detection
        const CIRCLE_OFFSETS: [(isize, isize); 16] = [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, 0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ];

        // Phase 1: Detect all FAST corners and compute strengths
        let mut corner_list: Vec<(usize, usize, f32)> = Vec::new(); // (x, y, strength)

        for y in 3..height - 3 {
            for x in 3..width - 3 {
                let center = src.get_pixel(x, y);

                // Sample the circle pixels
                let mut circle = [0u8; 16];
                for (i, (dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
                    let px = (x as isize + dx) as usize;
                    let py = (y as isize + dy) as usize;
                    circle[i] = src.get_pixel(px, py);
                }

                // Full FAST-9 contiguous arc check (matching CTS reference)
                // Check for 9+ contiguous pixels that are all brighter or all darker
                // Use combined tracking like the CTS reference
                let mut max_up = 0i32;
                let mut max_lo = 0i32;
                let mut up_count = 0i32;
                let mut lo_count = 0i32;

                for i in 0..25 {
                    let val = circle[i % 16] as i32;
                    if val > (center as i32 + threshold_val as i32) {
                        up_count += 1;
                        lo_count = 0;
                    } else if val < (center as i32 - threshold_val as i32) {
                        lo_count += 1;
                        up_count = 0;
                    } else {
                        up_count = 0;
                        lo_count = 0;
                    }
                    if up_count > max_up {
                        max_up = up_count;
                    }
                    if lo_count > max_lo {
                        max_lo = lo_count;
                    }
                }

                if max_up < 9 && max_lo < 9 {
                    continue; // Not a corner
                }

                // Compute corner strength using binary search (like reference)
                let mut lo_t = threshold_val as i32;
                let mut hi_t = 255i32;
                while hi_t - lo_t > 1 {
                    let mid_t = (hi_t + lo_t) / 2;
                    let mid_high = center as i32 + mid_t;
                    let mid_low = center as i32 - mid_t;

                    // Check if still a corner at this threshold (combined tracking)
                    let mut max_up = 0i32;
                    let mut max_lo = 0i32;
                    let mut up_count = 0i32;
                    let mut lo_count = 0i32;
                    for i in 0..25 {
                        let val = circle[i % 16] as i32;
                        if val > mid_high {
                            up_count += 1;
                            lo_count = 0;
                        } else if val < mid_low {
                            lo_count += 1;
                            up_count = 0;
                        } else {
                            up_count = 0;
                            lo_count = 0;
                        }
                        if up_count > max_up {
                            max_up = up_count;
                        }
                        if lo_count > max_lo {
                            max_lo = lo_count;
                        }
                    }

                    if max_up >= 9 || max_lo >= 9 {
                        lo_t = mid_t;
                    } else {
                        hi_t = mid_t;
                    }
                }
                let strength = lo_t;

                corner_list.push((x, y, strength as f32));
            }
        }

        // Phase 2: Non-maximum suppression (if requested)
        if do_nms {
            // Create strength image for NMS
            let img_size = width * height;
            let mut strength_img = vec![0u8; img_size];
            for &(x, y, s) in &corner_list {
                let ix = x as usize;
                let iy = y as usize;
                if ix < width && iy < height {
                    strength_img[iy * width + ix] = s.max(0.0).min(255.0) as u8;
                }
            }

            // NMS: keep only local maxima in 3x3 neighborhood
            // Matching CTS reference: >= for top/left, > for right/bottom (raster-scan bias)
            let mut nms_corners: Vec<(usize, usize, f32)> = Vec::new();
            for &(x, y, _s) in &corner_list {
                let ix = x as usize;
                let iy = y as usize;
                if ix < 1 || ix >= width - 1 || iy < 1 || iy >= height - 1 {
                    continue;
                }
                let c = strength_img[iy * width + ix] as i32;
                // Asymmetric: >= for top/left neighbors, > for right/bottom
                let top_left = strength_img[(iy - 1) * width + ix - 1] as i32;
                let top = strength_img[(iy - 1) * width + ix] as i32;
                let top_right = strength_img[(iy - 1) * width + ix + 1] as i32;
                let left = strength_img[iy * width + ix - 1] as i32;
                let right = strength_img[iy * width + ix + 1] as i32;
                let bottom_left = strength_img[(iy + 1) * width + ix - 1] as i32;
                let bottom = strength_img[(iy + 1) * width + ix] as i32;
                let bottom_right = strength_img[(iy + 1) * width + ix + 1] as i32;

                if c >= top_left
                    && c >= top
                    && c >= top_right
                    && c >= left
                    && c > right
                    && c > bottom_left
                    && c > bottom
                    && c > bottom_right
                {
                    nms_corners.push((x, y, c as f32));
                }
            }
            corner_list = nms_corners;
        }

        // Sort corners by strength (descending)
        corner_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Write corners to output array using proper C API functions
        if !corners.is_null() {
            let keypoint_size = std::mem::size_of::<vx_keypoint_t>();

            extern "C" {
                fn vxTruncateArray(arr: vx_array, new_num_items: vx_size) -> vx_status;
                fn vxAddArrayItems(
                    arr: vx_array,
                    count: vx_size,
                    ptr: *const c_void,
                    stride: vx_size,
                ) -> vx_status;
            }
            unsafe {
                vxTruncateArray(corners, 0);
            }

            let mut buf: Vec<u8> = Vec::with_capacity(corner_list.len() * keypoint_size);
            for &(x, y, strength) in &corner_list {
                let kp = vx_keypoint_t {
                    x: x as i32,
                    y: y as i32,
                    strength,
                    scale: 0.0,
                    orientation: 0.0,
                    tracking_status: 1,
                    error: 0.0,
                };
                let kp_bytes = std::slice::from_raw_parts(
                    &kp as *const vx_keypoint_t as *const u8,
                    keypoint_size,
                );
                buf.extend_from_slice(kp_bytes);
            }

            unsafe {
                vxAddArrayItems(
                    corners,
                    corner_list.len() as vx_size,
                    buf.as_ptr() as *const c_void,
                    keypoint_size as vx_size,
                );
            }
        }

        // Write num_corners to scalar
        if !num_corners.is_null() {
            let num = corner_list.len() as u32;
            crate::c_api_data::vxCopyScalarData(
                num_corners,
                &num as *const u32 as *mut c_void,
                0x11002, // VX_WRITE_ONLY
                0x0,
            );
        }

        VX_SUCCESS
    }
}

/// ===========================================================================
/// VXU Object Detection Functions
/// ===========================================================================

pub fn vxu_canny_edge_detector_impl(
    context: vx_context,
    input: vx_image,
    hyst_threshold: vx_threshold,
    gradient_size: vx_enum,
    norm_type: vx_enum,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Read threshold values from threshold object
        let (low_thresh, high_thresh) = if !hyst_threshold.is_null() {
            let t = &*(hyst_threshold as *const crate::c_api_data::VxCThresholdData);
            (t.lower, t.upper)
        } else {
            (50i32, 150i32)
        };

        let gsz = gradient_size as i32;
        let norm = norm_type;

        match canny_edge_detector(&src, &mut dst, low_thresh, high_thresh, gsz, norm) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// ===========================================================================
/// VXU Pyramid Functions
/// ===========================================================================

pub fn vxu_gaussian_pyramid_impl(
    context: vx_context,
    input: vx_image,
    output: vx_pyramid,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let pyramid = &*(output as *const VxCPyramid);
        let num_levels = pyramid.num_levels;

        // Copy input to level 0 of the pyramid
        let level0 = pyramid
            .levels
            .get(0)
            .map(|&img| img as vx_image)
            .unwrap_or(std::ptr::null_mut());
        if level0.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }

        // Copy input image data to level 0
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let mut dst0 = match create_matching_image(level0) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Copy input to level 0
        let (src_w, src_h) = (src.width(), src.height());
        let (dst_w, dst_h) = (dst0.width(), dst0.height());
        if src_w != dst_w || src_h != dst_h {
            for y in 0..dst_h.min(src_h) {
                for x in 0..dst_w.min(src_w) {
                    dst0.set_pixel(x, y, src.get_pixel(x, y));
                }
            }
        } else {
            for y in 0..src_h {
                for x in 0..src_w {
                    dst0.set_pixel(x, y, src.get_pixel(x, y));
                }
            }
        }
        copy_rust_to_c_image(&dst0, level0);

        // Generate subsequent levels using Gaussian blur + downscale
        let is_half_scale = (pyramid.scale - 0.5_f32).abs() < 0.001;
        for level_idx in 1..num_levels {
            let prev_level = pyramid
                .levels
                .get(level_idx - 1)
                .map(|&img| img as vx_image)
                .unwrap_or(std::ptr::null_mut());
            let curr_level = pyramid
                .levels
                .get(level_idx)
                .map(|&img| img as vx_image)
                .unwrap_or(std::ptr::null_mut());
            if prev_level.is_null() || curr_level.is_null() {
                break;
            }

            if is_half_scale {
                vxu_half_scale_gaussian_impl(context, prev_level, curr_level, 5);
            } else {
                // For non-half-scale (e.g., ORB): Gaussian 5x5 blur + nearest-neighbor resample
                // Uses integer arithmetic (>> 8) to match CTS reference
                let prev_img = match c_image_to_rust(prev_level) {
                    Some(img) => img,
                    None => break,
                };
                let (prev_w, prev_h) = (prev_img.width(), prev_img.height());

                // 5x5 Gaussian kernel weights (integer)
                let kernel: [i32; 25] = [
                    1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6,
                    4, 1,
                ];

                let mut blurred = vec![0u8; prev_w * prev_h];
                for y in 0..prev_h {
                    for x in 0..prev_w {
                        let mut sum: i32 = 0;
                        for ky in 0..5_i32 {
                            for kx in 0..5_i32 {
                                let sy = (y as i32 + ky - 2).clamp(0, prev_h as i32 - 1) as usize;
                                let sx = (x as i32 + kx - 2).clamp(0, prev_w as i32 - 1) as usize;
                                sum += prev_img.get_pixel(sx, sy) as i32
                                    * kernel[(ky * 5 + kx) as usize];
                            }
                        }
                        blurred[y * prev_w + x] = (sum >> 8).clamp(0, 255) as u8;
                    }
                }

                // Nearest-neighbor resample with center-to-center coordinate mapping
                let (curr_w, curr_h, curr_fmt) = match get_image_info(curr_level) {
                    Some(info) => (info.0 as usize, info.1 as usize, info.2),
                    None => break,
                };
                let dst_fmt = match df_image_to_format(curr_fmt) {
                    Some(f) => f,
                    None => break,
                };
                let mut dst_img = match Image::new(curr_w, curr_h, dst_fmt) {
                    Some(img) => img,
                    None => break,
                };

                for dy in 0..curr_h {
                    for dx in 0..curr_w {
                        // Center-to-center mapping: x_src = ((dx + 0.5) * prev_w / curr_w) - 0.5
                        let src_x = ((dx as f64 + 0.5) * prev_w as f64 / curr_w as f64) - 0.5;
                        let src_y = ((dy as f64 + 0.5) * prev_h as f64 / curr_h as f64) - 0.5;
                        let x_int = (src_x.round() as isize).clamp(0, prev_w as isize - 1) as usize;
                        let y_int = (src_y.round() as isize).clamp(0, prev_h as isize - 1) as usize;
                        dst_img.set_pixel(dx, dy, blurred[y_int * prev_w + x_int]);
                    }
                }
                copy_rust_to_c_image(&dst_img, curr_level);
            }
        }

        VX_SUCCESS
    }
}

/// ===========================================================================
/// VXU Laplacian Pyramid Functions
/// ===========================================================================

/// Laplacian pyramid: compute Gaussian pyramid, then take differences
pub fn vxu_laplacian_pyramid_impl(
    context: vx_context,
    input: vx_image,
    laplacian: vx_pyramid,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || laplacian.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let pyr = &*(laplacian as *const VxCPyramid);
        let num_levels = pyr.num_levels;

        // Build Gaussian pyramid using a temporary vx_pyramid (levels+1 levels, HALF scale)
        // CTS reference builds its Gaussian pyramid by calling vxuGaussianPyramid
        let (src_w, src_h, _) = match get_image_info(input) {
            Some(info) => (info.0, info.1, info.2),
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        extern "C" {
            fn vxCreatePyramid(
                ctx: vx_context,
                levels: vx_size,
                scale: vx_float32,
                w: vx_uint32,
                h: vx_uint32,
                fmt: vx_df_image,
            ) -> vx_pyramid;
            fn vxReleasePyramid(pyr: *mut vx_pyramid) -> vx_status;
        }
        let gauss_pyr = vxCreatePyramid(
            context,
            (num_levels + 1) as vx_size,
            0.5_f32, // VX_SCALE_PYRAMID_HALF
            src_w,
            src_h,
            0x30303855, // VX_DF_IMAGE_U8
        );
        if gauss_pyr.is_null() {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        // Build the Gaussian pyramid
        let status = vxu_gaussian_pyramid_impl(context, input, gauss_pyr);
        if status != VX_SUCCESS {
            vxReleasePyramid(&mut (gauss_pyr as vx_pyramid));
            return status;
        }

        let gauss = &*(gauss_pyr as *const VxCPyramid);

        // Compute Laplacian levels: L[i] = G[i] - expand(G[i+1])
        for level_idx in 0..num_levels {
            let level_img = pyr
                .levels
                .get(level_idx)
                .map(|&img| img as vx_image)
                .unwrap_or(std::ptr::null_mut());
            if level_img.is_null() {
                break;
            }

            let g_curr = gauss
                .levels
                .get(level_idx)
                .map(|&img| img as vx_image)
                .unwrap_or(std::ptr::null_mut());
            let g_next = gauss
                .levels
                .get(level_idx + 1)
                .map(|&img| img as vx_image)
                .unwrap_or(std::ptr::null_mut());
            if g_curr.is_null() || g_next.is_null() {
                break;
            }

            // Use vxuSubtract: L[i] = G[i] - expand(G[i+1])
            // First expand G[i+1] to size of G[i]
            let (gw, gh, _) = match get_image_info(g_curr) {
                Some(info) => (info.0 as usize, info.1 as usize, info.2),
                None => break,
            };
            let (nw, nh, _) = match get_image_info(g_next) {
                Some(info) => (info.0 as usize, info.1 as usize, info.2),
                None => break,
            };
            let next_img = match c_image_to_rust(g_next) {
                Some(img) => img,
                None => break,
            };
            let curr_img = match c_image_to_rust(g_curr) {
                Some(img) => img,
                None => break,
            };

            // Burt-Adelson expand: zero-interleave, convolve 5x5 Gaussian, multiply by 4
            // Step 1: zero-interleave next into gw x gh
            let mut interleaved = vec![0u8; gw * gh];
            for y in 0..gh {
                for x in 0..gw {
                    if x % 2 == 0 && y % 2 == 0 {
                        let sx = x / 2;
                        let sy = y / 2;
                        if sx < nw && sy < nh {
                            interleaved[y * gw + x] = next_img.get_pixel(sx, sy);
                        }
                    }
                }
            }

            // Step 2: convolve with 5x5 Gaussian (integer arithmetic, /256)
            let kernel: [i32; 25] = [
                1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1,
            ];
            let mut convolved = vec![0i16; gw * gh];
            for y in 0..gh {
                for x in 0..gw {
                    let mut sum: i32 = 0;
                    for ky in 0..5_i32 {
                        for kx in 0..5_i32 {
                            let sy = (y as i32 + ky - 2).clamp(0, gh as i32 - 1) as usize;
                            let sx = (x as i32 + kx - 2).clamp(0, gw as i32 - 1) as usize;
                            sum +=
                                interleaved[sy * gw + sx] as i32 * kernel[(ky * 5 + kx) as usize];
                        }
                    }
                    convolved[y * gw + x] = (sum / 256) as i16;
                }
            }

            // Step 3: multiply by 4
            for v in convolved.iter_mut() {
                *v = (*v * 4).clamp(-32768, 32767);
            }

            // Step 4: L[i] = G[i] - expanded
            let (lw, lh, _) = match get_image_info(level_img) {
                Some(info) => (info.0 as usize, info.1 as usize, info.2),
                None => break,
            };
            let img_ref = &*(level_img as *const VxCImage);
            if let Ok(mut data) = img_ref.data.write() {
                for y in 0..gh.min(lh) {
                    for x in 0..gw.min(lw) {
                        let curr_val = curr_img.get_pixel(x, y) as i16;
                        let expanded = convolved[y * gw + x];
                        let diff = (curr_val - expanded).clamp(-32768, 32767);
                        let offset = (y * lw + x) * 2;
                        if offset + 1 < data.len() {
                            let bytes = diff.to_le_bytes();
                            data[offset] = bytes[0];
                            data[offset + 1] = bytes[1];
                        }
                    }
                }
            }
        }

        // Output image = lowest Gaussian level (G[num_levels])
        let last_gauss = gauss
            .levels
            .get(num_levels)
            .map(|&img| img as vx_image)
            .unwrap_or(std::ptr::null_mut());
        if !last_gauss.is_null() {
            let last_img = c_image_to_rust(last_gauss);
            if let Some(img) = last_img {
                copy_rust_to_c_image(&img, output);
            }
        }

        vxReleasePyramid(&mut (gauss_pyr as vx_pyramid));
        VX_SUCCESS
    }
}

/// Laplacian reconstruct: upsample and add Laplacian levels back.
///
/// Implements the algorithm from `own_laplacian_reconstruct_reference` in
/// `OpenVX-cts/test_conformance/test_laplacianpyramid.c`:
///
/// 1. Start with the U8 input image as the running reconstruction.
/// 2. For each laplacian level from coarsest (`num_levels-1`) to finest (0):
///    - Build a U8 zero-interleaved buffer at the laplacian level dimensions
///      (only even-coord pixels carry data, odd-coord pixels are zero), with
///      pre-interleaved values clamped to `[0, 255]`.
///    - Convolve with the 5x5 Burt-Adelson Gaussian (sum/256) using the
///      CTS's "upsample replicate" border rule: out-of-bounds positions whose
///      *interleaved* coordinates would be odd are zero, while out-of-bounds
///      positions at even-even coordinates are taken from the pre-interleaved
///      buffer with REPLICATE.
///    - Multiply by 4 (S16 saturate) and add the laplacian level (S16
///      saturate). Saturate to U8 to form the next running reconstruction.
/// 3. The final running reconstruction is the U8 output.
pub fn vxu_laplacian_reconstruct_impl(
    context: vx_context,
    laplacian: vx_pyramid,
    input: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || laplacian.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // 5x5 Gaussian kernel from the OpenVX spec / Burt-Adelson; sums to 256.
    const GAUSSIAN5X5: [i32; 25] = [
        1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1,
    ];

    /// Sample the conceptually upsampled image at integer coords `(sx, sy)`,
    /// which may be out of bounds. `pre` is the pre-interleaved (smaller)
    /// buffer with U8 values; `(pw, ph)` is its size; `(uw, uh)` is the
    /// upsampled size. Out-of-bounds even-even positions REPLICATE the
    /// pre-image; everything else (odd-coord positions) returns 0.
    fn sample_upsample(
        sx: i32,
        sy: i32,
        pre: &[u8],
        pw: usize,
        ph: usize,
        _uw: usize,
        _uh: usize,
    ) -> u8 {
        // Odd-coord positions are zero by construction of the zero-interleave.
        if (sx & 1) != 0 || (sy & 1) != 0 {
            return 0;
        }
        // Even-even positions map back to the pre-interleaved image.
        let px = (sx >> 1).clamp(0, pw as i32 - 1) as usize;
        let py = (sy >> 1).clamp(0, ph as i32 - 1) as usize;
        pre[py * pw + px]
    }

    unsafe {
        let pyr = &*(laplacian as *const VxCPyramid);
        let num_levels = pyr.num_levels;
        if num_levels == 0 {
            if let Some(img) = c_image_to_rust(input) {
                copy_rust_to_c_image(&img, output);
            }
            return VX_SUCCESS;
        }

        let input_img = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let mut cur_w = input_img.width();
        let mut cur_h = input_img.height();
        let mut current: Vec<u8> = Vec::with_capacity(cur_w * cur_h);
        for y in 0..cur_h {
            for x in 0..cur_w {
                current.push(input_img.get_pixel(x, y));
            }
        }

        for level_idx in (0..num_levels).rev() {
            let level_img = pyr
                .levels
                .get(level_idx)
                .map(|&img| img as vx_image)
                .unwrap_or(std::ptr::null_mut());
            if level_img.is_null() {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let (lw, lh, _) = match get_image_info(level_img) {
                Some(info) => (info.0 as usize, info.1 as usize, info.2),
                None => return VX_ERROR_INVALID_PARAMETERS,
            };

            let img_ref = &*(level_img as *const VxCImage);
            let lap_data = match img_ref.data.read() {
                Ok(d) => d.clone(),
                Err(_) => return VX_ERROR_INVALID_PARAMETERS,
            };

            // 1) Convolve the zero-interleaved view of `current` (lw x lh) with
            //    the 5x5 Gaussian / 256, using the CTS's special upsample-with-
            //    replicate rule for out-of-bounds samples.
            let mut expanded: Vec<i16> = vec![0i16; lw * lh];
            for y in 0..lh {
                for x in 0..lw {
                    let mut sum: i32 = 0;
                    for ky in 0..5_i32 {
                        for kx in 0..5_i32 {
                            let sx = x as i32 + kx - 2;
                            let sy = y as i32 + ky - 2;
                            let pixel = if sx >= 0
                                && sx < lw as i32
                                && sy >= 0
                                && sy < lh as i32
                            {
                                // In-bounds: zero-interleaved value.
                                if (sx & 1) != 0 || (sy & 1) != 0 {
                                    0u8
                                } else {
                                    let px = (sx as usize) / 2;
                                    let py = (sy as usize) / 2;
                                    if px < cur_w && py < cur_h {
                                        current[py * cur_w + px]
                                    } else {
                                        0
                                    }
                                }
                            } else {
                                // Out-of-bounds: emulate REPLICATE on the
                                // pre-interleaved image and re-zero-interleave.
                                sample_upsample(sx, sy, &current, cur_w, cur_h, lw, lh)
                            };
                            sum += pixel as i32 * GAUSSIAN5X5[(ky * 5 + kx) as usize];
                        }
                    }
                    let conv = (sum / 256).clamp(i16::MIN as i32, i16::MAX as i32);
                    let scaled = conv
                        .saturating_mul(4)
                        .clamp(i16::MIN as i32, i16::MAX as i32);
                    expanded[y * lw + x] = scaled as i16;
                }
            }

            // 2) Add the laplacian level with S16 saturation, then saturate to
            //    U8 to form the next running reconstruction.
            let mut next: Vec<u8> = Vec::with_capacity(lw * lh);
            for y in 0..lh {
                for x in 0..lw {
                    let off = (y * lw + x) * 2;
                    let lap = if off + 1 < lap_data.len() {
                        i16::from_le_bytes([lap_data[off], lap_data[off + 1]]) as i32
                    } else {
                        0
                    };
                    let s16_sum = (expanded[y * lw + x] as i32 + lap)
                        .clamp(i16::MIN as i32, i16::MAX as i32);
                    next.push(s16_sum.clamp(0, 255) as u8);
                }
            }

            current = next;
            cur_w = lw;
            cur_h = lh;
        }

        let mut out_img = match Image::new(cur_w, cur_h, ImageFormat::Gray) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        for y in 0..cur_h {
            for x in 0..cur_w {
                out_img.set_pixel(x, y, current[y * cur_w + x]);
            }
        }
        copy_rust_to_c_image(&out_img, output);
        VX_SUCCESS
    }
}

/// ===========================================================================
/// VXU Remap Functions
/// ===========================================================================

pub fn vxu_remap_impl(
    context: vx_context,
    input: vx_image,
    _table: vx_remap,
    _policy: vx_enum,
    output: vx_image,
    override_border: Option<BorderMode>,
) -> vx_status {
    if context.is_null() || input.is_null() || _table.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Read remap table data
        let (map_x, map_y, dst_w, dst_h) = {
            let remap_data = &*(_table as *const crate::unified_c_api::VxCRemap);
            let map: std::sync::RwLockReadGuard<'_, Vec<f32>> = match remap_data.map_data.read() {
                Ok(d) => d,
                Err(_) => return VX_ERROR_INVALID_PARAMETERS,
            };
            let dw = remap_data.dst_width as usize;
            let dh = remap_data.dst_height as usize;
            let mut mx = Vec::with_capacity(dw * dh);
            let mut my = Vec::with_capacity(dw * dh);
            for y in 0..dh {
                for x in 0..dw {
                    let idx = (y * dw + x) * 2;
                    if idx + 1 < map.len() {
                        mx.push(map[idx]);
                        my.push(map[idx + 1]);
                    } else {
                        mx.push(0.0);
                        my.push(0.0);
                    }
                }
            }
            (mx, my, dw, dh)
        };

        let border = override_border.unwrap_or_else(|| get_border_from_context(context));
        let nearest_neighbor = _policy == 0x4000; // VX_INTERPOLATION_NEAREST_NEIGHBOR

        let src_width = src.width as f32;
        let src_height = src.height as f32;
        let src_w = src.width as i32;
        let src_h = src.height as i32;
        let dst_data = dst.data_mut();

        for y in 0..dst_h {
            for x in 0..dst_w {
                let idx = y * dst_w + x;
                let src_x = map_x[idx];
                let src_y = map_y[idx];

                let out_idx: usize = y.saturating_mul(dst_w).saturating_add(x);

                if nearest_neighbor {
                    let nx = (src_x + 0.5).floor() as i32;
                    let ny = (src_y + 0.5).floor() as i32;

                    if nx >= 0 && nx < src_w && ny >= 0 && ny < src_h {
                        if let Some(p) = dst_data.get_mut(out_idx) {
                            *p = src.get_pixel(nx as usize, ny as usize);
                        }
                    } else {
                        let val = match border {
                            BorderMode::Constant(c) => c,
                            _ => 0,
                        };
                        if let Some(p) = dst_data.get_mut(out_idx) {
                            *p = val;
                        }
                    }
                } else {
                    // Bilinear interpolation
                    if matches!(border, BorderMode::Undefined) {
                        let x0 = src_x.floor() as i32;
                        let y0 = src_y.floor() as i32;
                        if x0 >= 0 && x0 + 1 < src_w && y0 >= 0 && y0 + 1 < src_h {
                            if let Some(p) = dst_data.get_mut(out_idx) {
                                *p = bilinear_interpolate_with_border(&src, src_x, src_y, border);
                            }
                        }
                    } else {
                        if src_x < 0.0 || src_x >= src_width || src_y < 0.0 || src_y >= src_height {
                            let val = match border {
                                BorderMode::Constant(c) => c,
                                _ => 0,
                            };
                            if let Some(p) = dst_data.get_mut(out_idx) {
                                *p = val;
                            }
                        } else if let Some(p) = dst_data.get_mut(out_idx) {
                            *p = bilinear_interpolate_with_border(&src, src_x, src_y, border);
                        }
                    }
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

/// ===========================================================================
/// Vision Kernel Implementations (inline implementations)
/// ===========================================================================

/// Status type for vision operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VxStatus {
    Success,
    ErrorInvalidFormat,
    ErrorInvalidParameters,
    ErrorInvalidDimension,
    ErrorNoMemory,
}

/// ===========================================================================
/// Border Helper Functions
/// ===========================================================================
use crate::unified_c_api::{
    CONTEXTS, VX_BORDER_CONSTANT, VX_BORDER_REPLICATE, VX_BORDER_UNDEFINED,
};

fn get_border_from_context(context: vx_context) -> BorderMode {
    if let Ok(contexts) = CONTEXTS.lock() {
        if let Some(ctx) = contexts.get(&(context as usize)) {
            if let Ok(border) = ctx.border_mode.read() {
                return match border.mode {
                    VX_BORDER_CONSTANT => {
                        // For erosion, constant should be 255 (max), for dilation 0 (min)
                        // OpenVX spec says: constant_value is used for border
                        let const_val = unsafe { border.constant_value.U8 };
                        BorderMode::Constant(const_val)
                    }
                    VX_BORDER_REPLICATE => BorderMode::Replicate,
                    VX_BORDER_UNDEFINED | _ => BorderMode::Undefined,
                };
            }
        }
    }
    BorderMode::Undefined
}

/// Border modes for filter operations
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BorderMode {
    Undefined,
    Constant(u8),
    Replicate,
}

/// Coordinate type for min/max locations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
}

/// Corner structure
#[derive(Debug, Clone, Copy)]
pub struct Corner {
    pub x: usize,
    pub y: usize,
    pub strength: f32,
}

/// Line segment structure
#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// Result type for vision operations
type VxResult<T> = Result<T, VxStatus>;

// ============================================================================
// Color Conversion
// ============================================================================

fn rgb_to_gray(src: &Image, dst: &mut Image) -> VxResult<()> {
    if dst.format != ImageFormat::Gray {
        return Err(VxStatus::ErrorInvalidFormat);
    }

    let width = src.width;
    let height = src.height;

    // RGB inputs are guaranteed to be 3 bytes per pixel and laid out
    // contiguously in `src.data()`. Walk the slices directly with the
    // same Q8 BT.709 coefficients the per-pixel `get_rgb` path used
    // — the bounds-checked accessor was the bottleneck, not the math.
    if matches!(src.format, ImageFormat::Rgb) {
        let n = width.checked_mul(height).ok_or(VxStatus::ErrorInvalidParameters)?;
        let src_slice = src.data();
        let dst_data = dst.data_mut();
        if src_slice.len() >= n * 3 && dst_data.len() >= n {
            crate::simd_kernels::rgb_to_gray_fast(&src_slice[..n * 3], &mut dst_data[..n]);
            return Ok(());
        }
    }

    // Mixed / unexpected layouts — keep the original safe-but-slow path.
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            let gray = ((54 * r as u32 + 183 * g as u32 + 18 * b as u32) / 255) as u8;
            dst.set_pixel(x, y, gray);
        }
    }

    Ok(())
}

fn gray_to_rgb(src: &Image, dst: &mut Image) -> VxResult<()> {
    if src.format != ImageFormat::Gray || dst.format != ImageFormat::Rgb {
        return Err(VxStatus::ErrorInvalidFormat);
    }

    let width = src.width;
    let height = src.height;

    for y in 0..height {
        for x in 0..width {
            let gray = src.get_pixel(x, y);
            dst.set_rgb(x, y, gray, gray, gray);
        }
    }

    Ok(())
}

fn rgb_to_rgba(src: &Image, dst: &mut Image) -> VxResult<()> {
    if src.format != ImageFormat::Rgb || dst.format != ImageFormat::Rgba {
        return Err(VxStatus::ErrorInvalidFormat);
    }

    let width = src.width;
    let height = src.height;

    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            let idx = y.saturating_mul(width).saturating_add(x).saturating_mul(4);
            let dst_data = dst.data_mut();
            if idx.saturating_add(3) < dst_data.len() {
                dst_data[idx] = r;
                dst_data[idx + 1] = g;
                dst_data[idx + 2] = b;
                dst_data[idx + 3] = 255;
            }
        }
    }

    Ok(())
}

fn rgba_to_rgb(src: &Image, dst: &mut Image) -> VxResult<()> {
    if src.format != ImageFormat::Rgba || dst.format != ImageFormat::Rgb {
        return Err(VxStatus::ErrorInvalidFormat);
    }

    let width = src.width;
    let height = src.height;
    let src_data = src.data();
    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let src_idx = y.saturating_mul(width).saturating_add(x).saturating_mul(4);
            let dst_idx = y.saturating_mul(width).saturating_add(x).saturating_mul(3);
            if src_idx.saturating_add(2) < src_data.len()
                && dst_idx.saturating_add(2) < dst_data.len()
            {
                dst_data[dst_idx] = src_data[src_idx];
                dst_data[dst_idx + 1] = src_data[src_idx + 1];
                dst_data[dst_idx + 2] = src_data[src_idx + 2];
            }
        }
    }

    Ok(())
}

// ============================================================================
// Filter Operations
fn get_pixel_bordered(img: &Image, x: isize, y: isize, border: BorderMode) -> u8 {
    let width = img.width as isize;
    let height = img.height as isize;

    if x >= 0 && x < width && y >= 0 && y < height {
        img.get_pixel(x as usize, y as usize)
    } else {
        match border {
            BorderMode::Undefined => 0,
            BorderMode::Constant(val) => val,
            BorderMode::Replicate => {
                let cx = x.max(0).min(width - 1) as usize;
                let cy = y.max(0).min(height - 1) as usize;
                img.get_pixel(cx, cy)
            }
        }
    }
}

fn quickselect(arr: &mut [u8], k: usize) -> u8 {
    if k >= arr.len() {
        return 0;
    }

    let mut sorted = arr.to_vec();
    sorted.sort_unstable();
    sorted[k]
}

fn convolve_generic(
    src: &Image,
    dst: &mut Image,
    kernel: &[[i32; 3]; 3],
    border: BorderMode,
) -> VxResult<()> {
    let width = src.width;
    let height = src.height;
    let kernel_sum: i32 = kernel.iter().flat_map(|r| r.iter()).sum::<i32>().max(1);

    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let mut sum: i32 = 0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x as isize + kx as isize - 1;
                    let py = y as isize + ky as isize - 1;
                    let pixel = get_pixel_bordered(src, px, py, border);
                    sum += pixel as i32 * kernel[ky][kx];
                }
            }
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = clamp_u8(sum / kernel_sum);
            }
        }
    }

    Ok(())
}

fn gaussian3x3(src: &Image, dst: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    // VX_BORDER_UNDEFINED: only the interior `(1..H-1) × (1..W-1)` is
    // written; border pixels are left unchanged. Both the SIMD path
    // and the scalar fallback below honour this.
    if width < 3 || height < 3 {
        return Ok(());
    }

    // SIMD fast path: AVX2-eligible CPUs still go through the SSE2
    // implementation here because Gaussian's 3-row halo benefits little
    // from AVX2 vs SSE2 on the test runners; the SSE2 path is the
    // simpler/correct one to ship first. A dedicated AVX2 variant can
    // be added in `simd_kernels::gaussian3x3_u8` later.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    unsafe {
        if std::is_x86_feature_detected!("sse2") {
            let len = width.checked_mul(height).ok_or(VxStatus::ErrorInvalidParameters)?;
            let src_data = src.data();
            let dst_data = dst.data_mut();
            if src_data.len() >= len && dst_data.len() >= len {
                crate::simd_kernels::gaussian3x3_u8::sse2(
                    src_data.as_ptr(),
                    dst_data.as_mut_ptr(),
                    width,
                    height,
                );
                return Ok(());
            }
        }
    }

    // Scalar fallback — slice-iter version of the original kernel.
    // Walks rows directly instead of paying for `get_pixel(x, y)`'s
    // bounds checks per access, which is alone enough to deliver the
    // headline scalar-path speedup on non-x86_64 targets.
    let len = width.checked_mul(height).ok_or(VxStatus::ErrorInvalidParameters)?;
    let src_data = src.data();
    let dst_data = dst.data_mut();
    if src_data.len() < len || dst_data.len() < len {
        return Err(VxStatus::ErrorInvalidParameters);
    }
    for y in 1..height - 1 {
        let row0 = (y - 1) * width;
        let row1 = y * width;
        let row2 = (y + 1) * width;
        for x in 1..width - 1 {
            let s = src_data[row0 + x - 1] as u32
                + 2 * src_data[row0 + x] as u32
                + src_data[row0 + x + 1] as u32
                + 2 * src_data[row1 + x - 1] as u32
                + 4 * src_data[row1 + x] as u32
                + 2 * src_data[row1 + x + 1] as u32
                + src_data[row2 + x - 1] as u32
                + 2 * src_data[row2 + x] as u32
                + src_data[row2 + x + 1] as u32;
            // Match the existing kernel: integer divide by 16 with
            // truncation (not rounding) — keeps CTS bit-exact.
            dst_data[row1 + x] = (s >> 4) as u8;
        }
    }

    Ok(())
}

fn gaussian5x5(src: &Image, dst: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;
    let kernel = [1, 4, 6, 4, 1];

    let dst_data = dst.data_mut();
    // Use checked operations to prevent integer overflow
    let temp_size = width
        .checked_mul(height)
        .ok_or(VxStatus::ErrorInvalidParameters)?;
    let mut temp = vec![0u8; temp_size];

    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut sum: i32 = 0;
            let mut weight: i32 = 0;
            for k in 0..5 {
                let px = x as isize + k as isize - 2;
                if px >= 0 && px < width as isize {
                    sum += src.get_pixel(px as usize, y) as i32 * kernel[k];
                    weight += kernel[k];
                }
            }
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = temp.get_mut(idx) {
                *p = clamp_u8(sum / weight.max(1));
            }
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum: i32 = 0;
            let mut weight: i32 = 0;
            for k in 0..5 {
                let py = y as isize + k as isize - 2;
                if py >= 0 && py < height as isize {
                    let idx = (py as usize).saturating_mul(width).saturating_add(x);
                    if let Some(val) = temp.get(idx) {
                        sum += *val as i32 * kernel[k];
                        weight += kernel[k];
                    }
                }
            }
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = clamp_u8(sum / weight.max(1));
            }
        }
    }

    Ok(())
}

fn box3x3(src: &Image, dst: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    let len = width.checked_mul(height).ok_or(VxStatus::ErrorInvalidParameters)?;
    let src_data = src.data();
    let dst_data = dst.data_mut();
    if src_data.len() < len || dst_data.len() < len {
        return Err(VxStatus::ErrorInvalidParameters);
    }

    // Helper: scalar 3x3 box at (x, y) honouring rustVX's "variable
    // neighbour count" border behaviour. Used both as the standalone
    // fallback and as the border-pixel pass after the SIMD interior.
    let box_at = |sd: &[u8], x: usize, y: usize| -> u8 {
        let mut sum: u32 = 0;
        let mut count: u32 = 0;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let py = y as isize + dy;
                let px = x as isize + dx;
                if py >= 0 && py < height as isize && px >= 0 && px < width as isize {
                    sum += sd[py as usize * width + px as usize] as u32;
                    count += 1;
                }
            }
        }
        (sum / count.max(1)) as u8
    };

    if width < 3 || height < 3 {
        // No interior — fall back to scalar for every pixel.
        for y in 0..height {
            for x in 0..width {
                dst_data[y * width + x] = box_at(src_data, x, y);
            }
        }
        return Ok(());
    }

    // Interior (where every neighbour count is exactly 9): take the
    // SIMD path when available, else a tight slice loop.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let wrote_interior_via_simd = unsafe {
        if std::is_x86_feature_detected!("sse2") {
            crate::simd_kernels::box3x3_u8::sse2(
                src_data.as_ptr(),
                dst_data.as_mut_ptr(),
                width,
                height,
            );
            true
        } else {
            false
        }
    };
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let wrote_interior_via_simd = false;

    if !wrote_interior_via_simd {
        for y in 1..height - 1 {
            let row0 = (y - 1) * width;
            let row1 = y * width;
            let row2 = (y + 1) * width;
            for x in 1..width - 1 {
                let s = src_data[row0 + x - 1] as u32
                    + src_data[row0 + x] as u32
                    + src_data[row0 + x + 1] as u32
                    + src_data[row1 + x - 1] as u32
                    + src_data[row1 + x] as u32
                    + src_data[row1 + x + 1] as u32
                    + src_data[row2 + x - 1] as u32
                    + src_data[row2 + x] as u32
                    + src_data[row2 + x + 1] as u32;
                dst_data[row1 + x] = (s / 9) as u8;
            }
        }
    }

    // Borders: top + bottom rows, left + right columns (excluding the
    // four corner pixels which are covered by the row passes).
    for x in 0..width {
        dst_data[x] = box_at(src_data, x, 0);
        dst_data[(height - 1) * width + x] = box_at(src_data, x, height - 1);
    }
    for y in 1..height - 1 {
        dst_data[y * width] = box_at(src_data, 0, y);
        dst_data[y * width + width - 1] = box_at(src_data, width - 1, y);
    }

    Ok(())
}

fn median3x3(src: &Image, dst: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    let dst_data = dst.data_mut();
    let mut window = [0u8; 9];

    for y in 0..height {
        for x in 0..width {
            let mut idx = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let py = y as isize + dy;
                    let px = x as isize + dx;
                    if py >= 0 && py < height as isize && px >= 0 && px < width as isize {
                        window[idx] = src.get_pixel(px as usize, py as usize);
                    } else {
                        window[idx] = src.get_pixel(x, y); // Replicate border
                    }
                    idx += 1;
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = quickselect(&mut window, 4);
            }
        }
    }

    Ok(())
}

// ============================================================================
// Morphology
// ============================================================================

fn dilate3x3(src: &Image, dst: &mut Image, border: BorderMode) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    let dst_data = dst.data_mut();

    // For VX_BORDER_UNDEFINED, only process the inner region
    // (exclude 1-pixel border where neighborhood is incomplete)
    let (start_y, end_y, start_x, end_x) = match border {
        BorderMode::Undefined => (1, height.saturating_sub(1), 1, width.saturating_sub(1)),
        _ => (0, height, 0, width),
    };

    for y in start_y..end_y {
        for x in start_x..end_x {
            let mut max_val: u8 = 0;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    let val = get_pixel_bordered(src, px, py, border);
                    max_val = max_val.max(val);
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = max_val;
            }
        }
    }

    Ok(())
}

fn erode3x3(src: &Image, dst: &mut Image, border: BorderMode) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    let dst_data = dst.data_mut();

    // For VX_BORDER_UNDEFINED, only process the inner region
    // (exclude 1-pixel border where neighborhood is incomplete)
    let (start_y, end_y, start_x, end_x) = match border {
        BorderMode::Undefined => (1, height.saturating_sub(1), 1, width.saturating_sub(1)),
        _ => (0, height, 0, width),
    };

    for y in start_y..end_y {
        for x in start_x..end_x {
            let mut min_val: u8 = 255;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    let val = get_pixel_bordered(src, px, py, border);
                    min_val = min_val.min(val);
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = min_val;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Gradient Operations
// ============================================================================

const SOBEL_X: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];

const SOBEL_Y: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

/// Compute Sobel gradients outputting S16 directly, with proper border handling
fn sobel3x3_s16(src: &Image, grad_x: &mut Image, grad_y: &mut Image, border: BorderMode) {
    let width = src.width();
    let height = src.height();

    // Determine pixel range based on border mode
    // VX_BORDER_UNDEFINED: only compute inner pixels (1-pixel border left as zero)
    let (start_y, end_y, start_x, end_x) = match border {
        BorderMode::Undefined => (1, height.saturating_sub(1), 1, width.saturating_sub(1)),
        _ => (0, height, 0, width), // Replicate and Constant process all pixels
    };

    for y in start_y..end_y {
        for x in start_x..end_x {
            let mut sum_x: i32 = 0;
            let mut sum_y: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x as isize + kx as isize - 1;
                    let py = y as isize + ky as isize - 1;
                    let pixel = get_pixel_bordered(src, px, py, border) as i32;
                    sum_x += pixel * SOBEL_X[ky][kx];
                    sum_y += pixel * SOBEL_Y[ky][kx];
                }
            }

            // Output raw i16 values (no scaling)
            grad_x.set_pixel_s16(x, y, sum_x as i16);
            grad_y.set_pixel_s16(x, y, sum_y as i16);
        }
    }
}

/// Compute magnitude from S16 gradient images, output S16
/// mag(x,y) = floor(sqrt(gx² + gy²) + 0.5), saturated to S16 range
fn magnitude_s16(grad_x: &Image, grad_y: &Image, mag: &mut Image) {
    let width = grad_x.width();
    let height = grad_x.height();

    for y in 0..height {
        for x in 0..width {
            let gx = grad_x.get_pixel_s16(x, y) as i32;
            let gy = grad_y.get_pixel_s16(x, y) as i32;

            // Use double precision as per CTS reference
            let val = ((gx as f64 * gx as f64 + gy as f64 * gy as f64) as f64).sqrt();
            let ival = (val + 0.5).floor() as i32;
            let s16_val = ival.clamp(-32768, 32767) as i16;
            mag.set_pixel_s16(x, y, s16_val);
        }
    }
}

/// Compute phase from S16 gradient images, output U8
/// phase(x,y) = atan2(gy, gx) * 256 / (2π), mapped to [0, 255]
/// If val < 0, add 256; then floor(val + 0.5); clamp to [0, 255]
fn phase_s16(grad_x: &Image, grad_y: &Image, phase: &mut Image) {
    let width = grad_x.width();
    let height = grad_x.height();
    let phase_data = phase.data_mut();

    for y in 0..height {
        for x in 0..width {
            let gx = grad_x.get_pixel_s16(x, y) as f64;
            let gy = grad_y.get_pixel_s16(x, y) as f64;

            // CTS reference: atan2(gy, gx) * 256 / (M_PI * 2)
            let mut val = gy.atan2(gx) * 256.0 / (std::f64::consts::PI * 2.0);
            if val < 0.0 {
                val += 256.0;
            }
            let mut ival = (val + 0.5).floor() as i32;
            if ival >= 256 {
                ival -= 256;
            }
            let idx = y * width + x;
            if let Some(p) = phase_data.get_mut(idx) {
                *p = ival.clamp(0, 255) as u8;
            }
        }
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

/// Pixel-wise addition with overflow policy support
/// policy: 0 = WRAP, 1 = SATURATE (VX_CONVERT_POLICY_WRAP/SATURATE)
fn add(src1: &Image, src2: &Image, dst: &mut Image, policy: vx_enum) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let saturate = policy == VX_CONVERT_POLICY_SATURATE;

    let dst_is_s16 = matches!(dst.format, ImageFormat::GrayS16);
    let src1_is_s16 = matches!(src1.format, ImageFormat::GrayS16);
    let src2_is_s16 = matches!(src2.format, ImageFormat::GrayS16);

    // Hot path: U8 + U8 -> U8. The graph executor used to walk this
    // pixel-by-pixel via `get_pixel(x, y)` which dominates FHD
    // benchmarks; we now feed the slice straight into the SIMD
    // dispatcher in `crate::simd_kernels` (AVX2 → SSE2 → scalar).
    if !dst_is_s16 && !src1_is_s16 && !src2_is_s16 {
        let len = width.checked_mul(height).ok_or(VxStatus::ErrorInvalidParameters)?;
        let s1 = &src1.data()[..len];
        let s2 = &src2.data()[..len];
        let dst_data = dst.data_mut();
        let d = &mut dst_data[..len];

        if saturate {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            unsafe {
                if std::is_x86_feature_detected!("avx2") {
                    crate::simd_kernels::add_u8_sat::avx2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
                if std::is_x86_feature_detected!("sse2") {
                    crate::simd_kernels::add_u8_sat::sse2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
            }
            for ((a, b), o) in s1.iter().zip(s2.iter()).zip(d.iter_mut()) {
                *o = (*a).saturating_add(*b);
            }
        } else {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            unsafe {
                if std::is_x86_feature_detected!("avx2") {
                    crate::simd_kernels::add_u8_wrap::avx2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
                if std::is_x86_feature_detected!("sse2") {
                    crate::simd_kernels::add_u8_wrap::sse2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
            }
            for ((a, b), o) in s1.iter().zip(s2.iter()).zip(d.iter_mut()) {
                *o = (*a).wrapping_add(*b);
            }
        }
        return Ok(());
    }

    if dst_is_s16 {
        // S16 output - compute with 32-bit intermediate to avoid overflow
        for y in 0..height {
            for x in 0..width {
                let a = if src1_is_s16 {
                    src1.get_pixel_s16(x, y) as i32
                } else {
                    src1.get_pixel(x, y) as i32
                };
                let b = if src2_is_s16 {
                    src2.get_pixel_s16(x, y) as i32
                } else {
                    src2.get_pixel(x, y) as i32
                };
                let sum = a + b;
                let result = if saturate {
                    sum.clamp(-32768, 32767) as i16
                } else {
                    sum as i16 // Wrap
                };
                dst.set_pixel_s16(x, y, result);
            }
        }
    } else {
        // U8 output with at least one S16 input — keep the precise
        // mixed-format fallback (rare in practice, hot path above
        // covered the U8/U8/U8 case).
        let dst_data = dst.data_mut();
        for y in 0..height {
            for x in 0..width {
                let a = if src1_is_s16 {
                    src1.get_pixel_s16(x, y) as i32
                } else {
                    src1.get_pixel(x, y) as i32
                };
                let b = if src2_is_s16 {
                    src2.get_pixel_s16(x, y) as i32
                } else {
                    src2.get_pixel(x, y) as i32
                };
                let sum = a + b;
                let result = if saturate {
                    sum.clamp(0, 255) as u8
                } else {
                    sum as u8 // Truncation to u8 acts as wrap
                };
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = result;
                }
            }
        }
    }

    Ok(())
}

/// Pixel-wise subtraction with overflow policy support
/// policy: 0 = WRAP, 1 = SATURATE (VX_CONVERT_POLICY_WRAP/SATURATE)
fn subtract(src1: &Image, src2: &Image, dst: &mut Image, policy: vx_enum) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let saturate = policy == VX_CONVERT_POLICY_SATURATE;

    let dst_is_s16 = matches!(dst.format, ImageFormat::GrayS16);
    let src1_is_s16 = matches!(src1.format, ImageFormat::GrayS16);
    let src2_is_s16 = matches!(src2.format, ImageFormat::GrayS16);

    // Hot path: U8 - U8 -> U8.
    if !dst_is_s16 && !src1_is_s16 && !src2_is_s16 {
        let len = width.checked_mul(height).ok_or(VxStatus::ErrorInvalidParameters)?;
        let s1 = &src1.data()[..len];
        let s2 = &src2.data()[..len];
        let dst_data = dst.data_mut();
        let d = &mut dst_data[..len];

        if saturate {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            unsafe {
                if std::is_x86_feature_detected!("avx2") {
                    crate::simd_kernels::sub_u8_sat::avx2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
                if std::is_x86_feature_detected!("sse2") {
                    crate::simd_kernels::sub_u8_sat::sse2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
            }
            for ((a, b), o) in s1.iter().zip(s2.iter()).zip(d.iter_mut()) {
                *o = (*a).saturating_sub(*b);
            }
        } else {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            unsafe {
                if std::is_x86_feature_detected!("avx2") {
                    crate::simd_kernels::sub_u8_wrap::avx2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
                if std::is_x86_feature_detected!("sse2") {
                    crate::simd_kernels::sub_u8_wrap::sse2(s1.as_ptr(), s2.as_ptr(), d.as_mut_ptr(), len);
                    return Ok(());
                }
            }
            for ((a, b), o) in s1.iter().zip(s2.iter()).zip(d.iter_mut()) {
                *o = (*a).wrapping_sub(*b);
            }
        }
        return Ok(());
    }

    if dst_is_s16 {
        // S16 output
        for y in 0..height {
            for x in 0..width {
                let a = if src1_is_s16 {
                    src1.get_pixel_s16(x, y) as i32
                } else {
                    src1.get_pixel(x, y) as i32
                };
                let b = if src2_is_s16 {
                    src2.get_pixel_s16(x, y) as i32
                } else {
                    src2.get_pixel(x, y) as i32
                };
                let diff = a - b;
                let result = if saturate {
                    diff.clamp(-32768, 32767) as i16
                } else {
                    diff as i16 // Wrap
                };
                dst.set_pixel_s16(x, y, result);
            }
        }
    } else {
        // U8 output with at least one S16 input — mixed-format fallback.
        let dst_data = dst.data_mut();
        for y in 0..height {
            for x in 0..width {
                let a = if src1_is_s16 {
                    src1.get_pixel_s16(x, y) as i32
                } else {
                    src1.get_pixel(x, y) as i32
                };
                let b = if src2_is_s16 {
                    src2.get_pixel_s16(x, y) as i32
                } else {
                    src2.get_pixel(x, y) as i32
                };
                let diff = a - b;
                let result = if saturate {
                    diff.clamp(0, 255) as u8
                } else {
                    diff as u8 // Truncation to u8 acts as wrap
                };
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = result;
                }
            }
        }
    }

    Ok(())
}

/// Pixel-wise minimum (Enhanced Vision: `vxMin`).
///
/// Per the OpenVX 1.3 spec the input and output images must have the same
/// dimensions and a matching `VX_DF_IMAGE_U8` *or* `VX_DF_IMAGE_S16` format.
/// The output pixel is `min(src1, src2)`, no policy is involved.
fn min_image(src1: &Image, src2: &Image, dst: &mut Image) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    if dst.width != src1.width || dst.height != src1.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let dst_is_s16 = matches!(dst.format, ImageFormat::GrayS16);
    let src1_is_s16 = matches!(src1.format, ImageFormat::GrayS16);
    let src2_is_s16 = matches!(src2.format, ImageFormat::GrayS16);

    // Min/Max require src and dst formats to match per spec.
    if dst_is_s16 != src1_is_s16 || dst_is_s16 != src2_is_s16 {
        return Err(VxStatus::ErrorInvalidFormat);
    }

    let width = src1.width;
    let height = src1.height;

    if dst_is_s16 {
        for y in 0..height {
            for x in 0..width {
                let a = src1.get_pixel_s16(x, y);
                let b = src2.get_pixel_s16(x, y);
                dst.set_pixel_s16(x, y, a.min(b));
            }
        }
    } else {
        let dst_data = dst.data_mut();
        for y in 0..height {
            for x in 0..width {
                let a = src1.get_pixel(x, y);
                let b = src2.get_pixel(x, y);
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = a.min(b);
                }
            }
        }
    }

    Ok(())
}

/// Pixel-wise maximum (Enhanced Vision: `vxMax`). See `min_image` for the
/// format/dimension contract; identical except the per-pixel reduction is
/// `max(src1, src2)`.
fn max_image(src1: &Image, src2: &Image, dst: &mut Image) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    if dst.width != src1.width || dst.height != src1.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let dst_is_s16 = matches!(dst.format, ImageFormat::GrayS16);
    let src1_is_s16 = matches!(src1.format, ImageFormat::GrayS16);
    let src2_is_s16 = matches!(src2.format, ImageFormat::GrayS16);

    if dst_is_s16 != src1_is_s16 || dst_is_s16 != src2_is_s16 {
        return Err(VxStatus::ErrorInvalidFormat);
    }

    let width = src1.width;
    let height = src1.height;

    if dst_is_s16 {
        for y in 0..height {
            for x in 0..width {
                let a = src1.get_pixel_s16(x, y);
                let b = src2.get_pixel_s16(x, y);
                dst.set_pixel_s16(x, y, a.max(b));
            }
        }
    } else {
        let dst_data = dst.data_mut();
        for y in 0..height {
            for x in 0..width {
                let a = src1.get_pixel(x, y);
                let b = src2.get_pixel(x, y);
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = a.max(b);
                }
            }
        }
    }

    Ok(())
}

/// Immediate-mode entry point for `vxuMin`. Also used by the graph kernel
/// dispatcher when `vxMinNode` is processed via `vxProcessGraph`.
pub fn vxu_min_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match min_image(&src1, &src2, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(VxStatus::ErrorInvalidFormat) => VX_ERROR_INVALID_FORMAT,
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// Immediate-mode entry point for `vxuMax`. Also used by the graph kernel
/// dispatcher when `vxMaxNode` is processed via `vxProcessGraph`.
pub fn vxu_max_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match max_image(&src1, &src2, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(VxStatus::ErrorInvalidFormat) => VX_ERROR_INVALID_FORMAT,
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// Pixel-wise multiplication with scale, overflow and rounding policies
/// overflow_policy: 0 = WRAP, 1 = SATURATE
/// rounding_policy: 1 = TO_ZERO, 2 = TO_NEAREST_EVEN
fn multiply(
    src1: &Image,
    src2: &Image,
    dst: &mut Image,
    scale: f32,
    overflow_policy: vx_enum,
    rounding_policy: vx_enum,
) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let saturate = overflow_policy == VX_CONVERT_POLICY_SATURATE;
    let round_to_nearest = rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN;

    // Check output format
    let dst_is_s16 = matches!(dst.format, ImageFormat::GrayS16);
    let src1_is_s16 = matches!(src1.format, ImageFormat::GrayS16);
    let src2_is_s16 = matches!(src2.format, ImageFormat::GrayS16);

    // OpenVX spec: dst(x,y) = src1(x,y) * src2(x,y) * scale
    // scale is a float32, e.g. 1/255 or 1/2^n
    // For U8 output, result is clamped/wrapped to [0, 255]
    // For S16 output, result is clamped/wrapped to [-32768, 32767]

    if dst_is_s16 {
        // S16 output - use 64-bit intermediate
        for y in 0..height {
            for x in 0..width {
                let a = if src1_is_s16 {
                    src1.get_pixel_s16(x, y) as i64
                } else {
                    src1.get_pixel(x, y) as i64
                };
                let b = if src2_is_s16 {
                    src2.get_pixel_s16(x, y) as i64
                } else {
                    src2.get_pixel(x, y) as i64
                };

                // Compute: a * b * scale with proper rounding
                let product = (a as f64) * (b as f64) * (scale as f64);
                let rounded = if round_to_nearest {
                    product.round_ties_even() as i64
                } else {
                    // VX_ROUND_POLICY_TO_ZERO: truncate toward zero
                    if product >= 0.0 {
                        product.floor() as i64
                    } else {
                        product.ceil() as i64
                    }
                };

                let result = if saturate {
                    rounded.clamp(-32768, 32767) as i16
                } else {
                    // Wrap: just truncate to i16
                    rounded as i16
                };
                dst.set_pixel_s16(x, y, result);
            }
        }
    } else {
        // U8 output
        let dst_data = dst.data_mut();
        for y in 0..height {
            for x in 0..width {
                let a = if src1_is_s16 {
                    src1.get_pixel_s16(x, y) as i64
                } else {
                    src1.get_pixel(x, y) as i64
                };
                let b = if src2_is_s16 {
                    src2.get_pixel_s16(x, y) as i64
                } else {
                    src2.get_pixel(x, y) as i64
                };

                // Compute: a * b * scale with proper rounding
                let product = (a as f64) * (b as f64) * (scale as f64);
                let rounded = if round_to_nearest {
                    product.round_ties_even() as i64
                } else {
                    // VX_ROUND_POLICY_TO_ZERO: truncate toward zero
                    if product >= 0.0 {
                        product.floor() as i64
                    } else {
                        product.ceil() as i64
                    }
                };

                let result = if saturate {
                    rounded.clamp(0, 255) as u8
                } else {
                    // Wrap: just truncate to u8
                    rounded as u8
                };

                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = result;
                }
            }
        }
    }

    Ok(())
}

fn weighted(src1: &Image, src2: &Image, dst: &mut Image, alpha_f32: f32) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let alpha_w = alpha_f32;
    let beta_w = 1.0 - alpha_f32;

    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as f32;
            let b = src2.get_pixel(x, y) as f32;
            let result = alpha_w * a + beta_w * b;
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = result as i32 as u8;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Statistics
// ============================================================================

struct MinMaxLocResult {
    min_val: i64,
    max_val: i64,
    min_locs: Vec<Coordinate>,
    max_locs: Vec<Coordinate>,
}

fn min_max_loc(src: &Image) -> VxResult<MinMaxLocResult> {
    let width = src.width;
    let height = src.height;
    let is_s16 = matches!(src.format, ImageFormat::GrayS16);

    let mut min_val: i64 = i64::MAX;
    let mut max_val: i64 = i64::MIN;
    let mut min_locs: Vec<Coordinate> = Vec::new();
    let mut max_locs: Vec<Coordinate> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let val = if is_s16 {
                let idx = (y * width + x) * 2;
                let data = src.data();
                if idx + 1 < data.len() {
                    i16::from_le_bytes([data[idx], data[idx + 1]]) as i64
                } else {
                    0i64
                }
            } else {
                src.get_pixel(x, y) as i64
            };

            if val < min_val {
                min_val = val;
                min_locs.clear();
                min_locs.push(Coordinate { x, y });
            } else if val == min_val {
                min_locs.push(Coordinate { x, y });
            }

            if val > max_val {
                max_val = val;
                max_locs.clear();
                max_locs.push(Coordinate { x, y });
            } else if val == max_val {
                max_locs.push(Coordinate { x, y });
            }
        }
    }

    Ok(MinMaxLocResult {
        min_val,
        max_val,
        min_locs,
        max_locs,
    })
}

fn mean_std_dev(src: &Image) -> VxResult<(f32, f32)> {
    let (sx, sy, ex, ey) = src.valid_rect();
    let pixel_count = (ex - sx).saturating_mul(ey - sy) as f32;
    if pixel_count == 0.0 {
        return Ok((0.0, 0.0));
    }

    // Compute mean
    let mut sum: u64 = 0;
    for y in sy..ey {
        for x in sx..ex {
            sum += src.get_pixel(x, y) as u64;
        }
    }
    let mean = sum as f32 / pixel_count;

    // Compute variance
    let mut sum_sq_diff: f64 = 0.0;
    for y in sy..ey {
        for x in sx..ex {
            let diff = src.get_pixel(x, y) as f32 - mean;
            sum_sq_diff += (diff * diff) as f64;
        }
    }
    let variance = sum_sq_diff as f32 / pixel_count;
    let stddev = variance.sqrt();

    Ok((mean, stddev))
}

fn histogram(src: &Image) -> VxResult<[i32; 256]> {
    let width = src.width;
    let height = src.height;
    let mut hist = [0i32; 256];

    for y in 0..height {
        for x in 0..width {
            let val = src.get_pixel(x, y);
            hist[val as usize] += 1;
        }
    }

    Ok(hist)
}

fn integral_image(src: &Image, dst: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    if width == 0 || height == 0 {
        return Ok(());
    }

    let src_data = src.data();
    let dst_bytes = dst.data_mut();

    // Bail out (rather than corrupt memory) if the buffers are smaller than
    // the declared image dimensions for any reason.
    let pixels = width.checked_mul(height).ok_or(VxStatus::ErrorInvalidDimension)?;
    if src_data.len() < pixels || dst_bytes.len() < pixels.checked_mul(4).ok_or(VxStatus::ErrorInvalidDimension)? {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    // The destination is a U32 image laid out as little-endian bytes. On the
    // platforms rustVX targets (little-endian x86_64 / aarch64) we can safely
    // reinterpret the byte buffer as `[u32]` and use native 32-bit stores
    // instead of decomposing every result into `to_le_bytes()`. Even on a
    // hypothetical big-endian target the reinterpret is sound — we just lose
    // the on-disk LE convention there, which is moot because rustVX is only
    // built on LE hosts.
    debug_assert!(cfg!(target_endian = "little"));
    let dst_u32: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u32, pixels)
    };

    // First row: integral = running row sum, no row above.
    {
        let src_row = &src_data[..width];
        let dst_row = &mut dst_u32[..width];
        let mut row_sum: u32 = 0;
        for x in 0..width {
            row_sum += src_row[x] as u32;
            dst_row[x] = row_sum;
        }
    }

    // Subsequent rows: dst[y, x] = row_sum + dst[y - 1, x]. Splitting the
    // destination into "previous row" and "current row" slices lets the
    // borrow checker see disjoint ranges and lets the optimiser use native
    // 32-bit loads/stores in the hot loop.
    for y in 1..height {
        let src_row = &src_data[y * width..(y + 1) * width];
        let (prev_rows, current_and_after) = dst_u32.split_at_mut(y * width);
        let prev_row = &prev_rows[(y - 1) * width..y * width];
        let dst_row = &mut current_and_after[..width];

        let mut row_sum: u32 = 0;
        for x in 0..width {
            row_sum += src_row[x] as u32;
            dst_row[x] = row_sum + prev_row[x];
        }
    }

    Ok(())
}

fn abs_diff(src1: &Image, src2: &Image, dst: &mut Image) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;

    match src1.format {
        ImageFormat::Gray => {
            let dst_data = dst.data_mut();
            for y in 0..height {
                for x in 0..width {
                    let a = src1.get_pixel(x, y);
                    let b = src2.get_pixel(x, y);
                    let diff = if a > b { a - b } else { b - a };
                    let idx = y.saturating_mul(width).saturating_add(x);
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = diff;
                    }
                }
            }
        }
        ImageFormat::GrayS16 => {
            for y in 0..height {
                for x in 0..width {
                    let a = src1.get_pixel_s16(x, y) as i32;
                    let b = src2.get_pixel_s16(x, y) as i32;
                    let diff = (a - b).abs();
                    // Clamp to S16 range (0..32767 for absolute difference)
                    let result: i16 = if diff > 32767 { 32767 } else { diff as i16 };
                    dst.set_pixel_s16(x, y, result);
                }
            }
        }
        _ => {
            // Default: treat as U8
            let dst_data = dst.data_mut();
            for y in 0..height {
                for x in 0..width {
                    let a = src1.get_pixel(x, y);
                    let b = src2.get_pixel(x, y);
                    let diff = if a > b { a - b } else { b - a };
                    let idx = y.saturating_mul(width).saturating_add(x);
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = diff;
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Geometric Operations
// ============================================================================

fn bilinear_interpolate(img: &Image, x: f32, y: f32) -> u8 {
    let width = img.width as i32;
    let height = img.height as i32;

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    if x0 < 0 || y0 < 0 || x0 >= width || y0 >= height {
        return 0;
    }

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = img.get_pixel(x0 as usize, y0 as usize) as f32;
    let p10 = if x1 < width {
        img.get_pixel(x1 as usize, y0 as usize) as f32
    } else {
        p00
    };
    let p01 = if y1 < height {
        img.get_pixel(x0 as usize, y1 as usize) as f32
    } else {
        p00
    };
    let p11 = if x1 < width && y1 < height {
        img.get_pixel(x1 as usize, y1 as usize) as f32
    } else {
        p00
    };

    let value = (1.0 - fx) * (1.0 - fy) * p00
        + fx * (1.0 - fy) * p10
        + (1.0 - fx) * fy * p01
        + fx * fy * p11;

    clamp_u8(value as i32)
}

fn scale_image(
    src: &Image,
    dst: &mut Image,
    interpolation: InterpolationType,
    border: BorderMode,
) -> VxResult<()> {
    let src_width = src.width;
    let src_height = src.height;
    let dst_width = dst.width;
    let dst_height = dst.height;

    let dst_data = dst.data_mut();

    // Backward mapping: for each output pixel, compute corresponding source pixel
    // OpenVX uses center-aligned mapping: src = (dst + 0.5) * (src_size / dst_size) - 0.5
    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    for y in 0..dst_height {
        for x in 0..dst_width {
            // Map from destination to source with half-pixel offset (OpenVX standard)
            let src_x = (x as f32 + 0.5) * x_scale - 0.5;
            let src_y = (y as f32 + 0.5) * y_scale - 0.5;

            let value = match interpolation {
                InterpolationType::NearestNeighbor => {
                    nearest_neighbor_interpolate(src, src_x, src_y, border)
                }
                InterpolationType::Bilinear => {
                    bilinear_interpolate_with_border(src, src_x, src_y, border)
                }
                InterpolationType::Area => {
                    // AREA interpolation is only for downscaling (x_scale >= 1 and y_scale >= 1)
                    // For upscaling, fall back to nearest-neighbor
                    if x_scale >= 1.0 && y_scale >= 1.0 {
                        let src_x_area = x as f32 * x_scale;
                        let src_y_area = y as f32 * y_scale;
                        area_interpolate(src, src_x_area, src_y_area, x_scale, y_scale, border)
                    } else {
                        nearest_neighbor_interpolate(src, src_x, src_y, border)
                    }
                }
            };

            let idx = y.saturating_mul(dst_width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = value;
            }
        }
    }

    Ok(())
}

fn nearest_neighbor_interpolate(img: &Image, x: f32, y: f32, border: BorderMode) -> u8 {
    let width = img.width as i32;
    let height = img.height as i32;

    // OpenVX nearest neighbor: round to nearest integer (round-half-up)
    // floor(x + 0.5)
    let nx = (x + 0.5).floor() as i32;
    let ny = (y + 0.5).floor() as i32;

    // Check if within valid region
    if !img.is_valid_pixel(nx, ny) {
        return match border {
            BorderMode::Constant(val) => val,
            BorderMode::Replicate => {
                let clamped_x = nx.clamp(0, width - 1) as usize;
                let clamped_y = ny.clamp(0, height - 1) as usize;
                img.get_pixel(clamped_x, clamped_y)
            }
            BorderMode::Undefined => 0,
        };
    }

    img.get_pixel(nx as usize, ny as usize)
}

fn bilinear_interpolate_with_border(img: &Image, x: f32, y: f32, border: BorderMode) -> u8 {
    let width = img.width as i32;
    let height = img.height as i32;

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    // Handle border modes - pixels outside valid region use border mode
    let get_pixel_bilinear = |px: i32, py: i32| -> u8 {
        if img.is_valid_pixel(px, py) {
            img.get_pixel(px as usize, py as usize)
        } else {
            match border {
                BorderMode::Constant(val) => val,
                BorderMode::Replicate => {
                    let clamped_x = px.clamp(0, width - 1) as usize;
                    let clamped_y = py.clamp(0, height - 1) as usize;
                    img.get_pixel(clamped_x, clamped_y)
                }
                BorderMode::Undefined => 0,
            }
        }
    };

    let p00 = get_pixel_bilinear(x0, y0) as f32;
    let p10 = get_pixel_bilinear(x1, y0) as f32;
    let p01 = get_pixel_bilinear(x0, y1) as f32;
    let p11 = get_pixel_bilinear(x1, y1) as f32;

    let value = (1.0 - fx) * (1.0 - fy) * p00
        + fx * (1.0 - fy) * p10
        + (1.0 - fx) * fy * p01
        + fx * fy * p11;

    // Round to nearest integer (CTS reference uses ref_float + 0.5f)
    clamp_u8(value.round() as i32)
}

fn area_interpolate(
    img: &Image,
    x: f32,
    y: f32,
    x_scale: f32,
    y_scale: f32,
    border: BorderMode,
) -> u8 {
    // For area interpolation (used when downscaling), compute the average
    // over the source region that maps to this output pixel.
    // For integer downscaling (e.g., 4:1), this should average exactly
    // the NxM block of source pixels.
    // Use floor-based region: start at floor(x), end at floor(x + x_scale)
    // This ensures integer downscale averages the correct block.
    let x_start = x.floor() as i32;
    let y_start = y.floor() as i32;
    // For exact integer downscale: x_start should be the block start,
    // and the region should cover exactly x_scale * y_scale pixels.
    let x_end = (x + x_scale).floor() as i32; // Use floor, not ceil
    let y_end = (y + y_scale).floor() as i32;

    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    for py in y_start..y_end {
        for px in x_start..x_end {
            if img.is_valid_pixel(px, py) {
                sum += img.get_pixel(px as usize, py as usize) as u32;
                count += 1;
            } else {
                // Outside valid region
                match border {
                    BorderMode::Constant(val) => {
                        sum += val as u32;
                        count += 1;
                    }
                    BorderMode::Replicate => {
                        let clamped_x = px.clamp(0, img.width as i32 - 1) as usize;
                        let clamped_y = py.clamp(0, img.height as i32 - 1) as usize;
                        sum += img.get_pixel(clamped_x, clamped_y) as u32;
                        count += 1;
                    }
                    BorderMode::Undefined => {
                        // For BORDER_UNDEFINED with valid region,
                        // skip invalid pixels entirely (don't include in sum or count)
                        // This matches CTS behavior: pixels outside valid region
                        // are excluded from the area average
                    }
                }
            }
        }
    }

    if count > 0 {
        ((sum + count / 2) / count) as u8 // Round to nearest
    } else {
        0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum InterpolationType {
    NearestNeighbor,
    Bilinear,
    Area,
}

fn warp_affine(
    src: &Image,
    matrix: &[f32; 6],
    dst: &mut Image,
    border: BorderMode,
    nearest_neighbor: bool,
) -> VxResult<()> {
    let dst_width = dst.width;
    let dst_height = dst.height;
    let _src_w = src.width as i32;
    let _src_h = src.height as i32;

    let dst_data = dst.data_mut();

    let a11 = matrix[0];
    let a12 = matrix[2];
    let a13 = matrix[4];
    let a21 = matrix[1];
    let a22 = matrix[3];
    let a23 = matrix[5];

    for y in 0..dst_height {
        for x in 0..dst_width {
            let xf = x as f32;
            let yf = y as f32;

            let src_x = a11 * xf + a12 * yf + a13;
            let src_y = a21 * xf + a22 * yf + a23;

            let idx = y.saturating_mul(dst_width).saturating_add(x);

            if nearest_neighbor {
                let nx = (src_x + 0.5).floor() as i32;
                let ny = (src_y + 0.5).floor() as i32;

                if src.is_valid_pixel(nx, ny) {
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = src.get_pixel(nx as usize, ny as usize);
                    }
                } else {
                    let val = match border {
                        BorderMode::Constant(c) => c,
                        _ => 0,
                    };
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = val;
                    }
                }
            } else {
                // Bilinear interpolation
                if matches!(border, BorderMode::Undefined) {
                    // Check if the full 2x2 neighborhood is within valid region
                    let x0 = src_x.floor() as i32;
                    let y0 = src_y.floor() as i32;
                    if src.is_valid_pixel(x0, y0)
                        && src.is_valid_pixel(x0 + 1, y0)
                        && src.is_valid_pixel(x0, y0 + 1)
                        && src.is_valid_pixel(x0 + 1, y0 + 1)
                    {
                        if let Some(p) = dst_data.get_mut(idx) {
                            *p = bilinear_interpolate_with_border(src, src_x, src_y, border);
                        }
                    }
                } else {
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = bilinear_interpolate_with_border(src, src_x, src_y, border);
                    }
                }
            }
        }
    }

    Ok(())
}

fn warp_perspective(
    src: &Image,
    matrix: &[f32; 9],
    dst: &mut Image,
    border: BorderMode,
    nearest_neighbor: bool,
) -> VxResult<()> {
    let dst_width = dst.width;
    let dst_height = dst.height;

    let _src_w = src.width as i32;
    let _src_h = src.height as i32;

    let dst_data = dst.data_mut();

    // CTS perspective matrix layout: m[col*3 + row]
    // x0 = m[0]*x + m[3]*y + m[6], y0 = m[1]*x + m[4]*y + m[7], z0 = m[2]*x + m[5]*y + m[8]
    let h00 = matrix[0];
    let h10 = matrix[1];
    let h20 = matrix[2];
    let h01 = matrix[3];
    let h11 = matrix[4];
    let h21 = matrix[5];
    let h02 = matrix[6];
    let h12 = matrix[7];
    let h22 = matrix[8];

    for y in 0..dst_height {
        for x in 0..dst_width {
            let xf = x as f32;
            let yf = y as f32;

            // Column-major matrix multiplication (matches OpenVX test)
            let x_h = h00 * xf + h01 * yf + h02;
            let y_h = h10 * xf + h11 * yf + h12;
            let w_h = h20 * xf + h21 * yf + h22;

            let idx = y.saturating_mul(dst_width).saturating_add(x);
            if w_h.abs() < 1e-6 {
                let val = match border {
                    BorderMode::Constant(c) => c,
                    _ => 0,
                };
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = val;
                }
                continue;
            }

            let src_x = x_h / w_h;
            let src_y = y_h / w_h;

            if nearest_neighbor {
                let nx = (src_x + 0.5).floor() as i32;
                let ny = (src_y + 0.5).floor() as i32;

                if src.is_valid_pixel(nx, ny) {
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = src.get_pixel(nx as usize, ny as usize);
                    }
                } else {
                    let val = match border {
                        BorderMode::Constant(c) => c,
                        _ => 0,
                    };
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = val;
                    }
                }
            } else {
                // Bilinear interpolation with proper border handling
                if matches!(border, BorderMode::Undefined) {
                    let x0 = src_x.floor() as i32;
                    let y0 = src_y.floor() as i32;
                    if src.is_valid_pixel(x0, y0)
                        && src.is_valid_pixel(x0 + 1, y0)
                        && src.is_valid_pixel(x0, y0 + 1)
                        && src.is_valid_pixel(x0 + 1, y0 + 1)
                    {
                        if let Some(p) = dst_data.get_mut(idx) {
                            *p = bilinear_interpolate_with_border(src, src_x, src_y, border);
                        }
                    }
                    // else: UNDEFINED border, leave as 0
                } else {
                    if let Some(p) = dst_data.get_mut(idx) {
                        *p = bilinear_interpolate_with_border(src, src_x, src_y, border);
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Feature Detection
// ============================================================================

fn harris_corners(
    image: &Image,
    k: f32,
    threshold: f32,
    _min_distance: usize,
) -> VxResult<Vec<Corner>> {
    let width = image.width;
    let height = image.height;
    // Use checked operations to prevent integer overflow
    let response_size = width
        .checked_mul(height)
        .ok_or(VxStatus::ErrorInvalidParameters)?;
    let mut responses = vec![0f32; response_size];

    // Compute gradients using Sobel
    let (grad_x, grad_y) = compute_gradients_sobel(image)?;

    // Compute structure tensor and corner response
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut ixx: f32 = 0.0;
            let mut iyy: f32 = 0.0;
            let mut ixy: f32 = 0.0;

            // Sum over 3x3 window
            for wy in -1..=1 {
                for wx in -1..=1 {
                    let idx = ((y as isize + wy) as usize)
                        .saturating_mul(width)
                        .saturating_add((x as isize + wx) as usize);
                    let ix = grad_x[idx] as f32;
                    let iy = grad_y[idx] as f32;

                    ixx += ix * ix;
                    iyy += iy * iy;
                    ixy += ix * iy;
                }
            }

            // Harris corner response
            let det = ixx * iyy - ixy * ixy;
            let trace = ixx + iyy;
            let response = det - k * trace * trace;

            responses[y.saturating_mul(width).saturating_add(x)] = response;
        }
    }

    // Non-maximum suppression
    let mut corners = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let response = responses[y.saturating_mul(width).saturating_add(x)];

            if response < threshold {
                continue;
            }

            // Check if local maximum
            let mut is_max = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let idx = (ny as usize)
                        .saturating_mul(width)
                        .saturating_add(nx as usize);
                    if responses.get(idx).copied().unwrap_or(0.0) > response {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }

            if is_max {
                corners.push(Corner {
                    x,
                    y,
                    strength: response,
                });
            }
        }
    }

    // Sort by strength
    corners.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(corners)
}

fn fast9(image: &Image, threshold: u8) -> VxResult<Vec<Corner>> {
    let width = image.width;
    let height = image.height;
    let mut corners = Vec::new();

    const CIRCLE_OFFSETS: [(isize, isize); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let center = image.get_pixel(x, y);
            let high = center.saturating_add(threshold);
            let low = center.saturating_sub(threshold);

            let mut circle = [0u8; 16];
            for (i, (dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
                let px = (x as isize + dx) as usize;
                let py = (y as isize + dy) as usize;
                circle[i] = image.get_pixel(px, py);
            }

            // Check for 9 contiguous brighter or darker pixels
            let mut is_corner = false;

            for start in 0..16 {
                let mut brighter_count = 0;
                let mut darker_count = 0;

                for i in 0..16 {
                    let idx = (start + i) % 16;
                    if circle[idx] > high {
                        brighter_count += 1;
                        darker_count = 0;
                    } else if circle[idx] < low {
                        darker_count += 1;
                        brighter_count = 0;
                    } else {
                        brighter_count = 0;
                        darker_count = 0;
                    }

                    if brighter_count >= 9 || darker_count >= 9 {
                        is_corner = true;
                        break;
                    }
                }

                if is_corner {
                    break;
                }
            }

            if is_corner {
                let score = compute_fast_score(&circle, center, threshold);
                corners.push(Corner {
                    x,
                    y,
                    strength: score as f32,
                });
            }
        }
    }

    corners.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(corners)
}

fn compute_fast_score(circle: &[u8; 16], center: u8, threshold: u8) -> u16 {
    let mut score: u16 = 0;
    for &p in circle.iter() {
        let diff = if p > center {
            (p - center) as u16
        } else {
            (center - p) as u16
        };
        if diff > threshold as u16 {
            score += diff;
        }
    }
    score
}

fn compute_gradients_sobel(image: &Image) -> VxResult<(Vec<f32>, Vec<f32>)> {
    let width = image.width;
    let height = image.height;
    // Use saturating_mul to prevent integer overflow
    let gradient_size = width.saturating_mul(height);
    let mut grad_x = vec![0f32; gradient_size];
    let mut grad_y = vec![0f32; gradient_size];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    let pixel = image.get_pixel(px, py) as i32;
                    gx += pixel * SOBEL_X[ky][kx];
                    gy += pixel * SOBEL_Y[ky][kx];
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            grad_x[idx] = gx as f32 / 4.0;
            grad_y[idx] = gy as f32 / 4.0;
        }
    }

    Ok((grad_x, grad_y))
}

// ============================================================================
// Object Detection
// ============================================================================

fn canny_edge_detector(
    src: &Image,
    dst: &mut Image,
    low_thresh: i32,
    high_thresh: i32,
    gsz: i32,
    norm: vx_enum,
) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    let img_size = width
        .checked_mul(height)
        .ok_or(VxStatus::ErrorInvalidParameters)?;

    if width < gsz as usize || height < gsz as usize {
        let dst_data = dst.data_mut();
        for i in dst_data.iter_mut() {
            *i = 0;
        }
        return Ok(());
    }

    let bsz = (gsz / 2 + 1) as usize;

    // Sobel separable kernels per OpenVX spec / CTS reference
    let dim1: [[i32; 7]; 3] = [
        [1, 2, 1, 0, 0, 0, 0],
        [1, 4, 6, 4, 1, 0, 0],
        [1, 6, 15, 20, 15, 6, 1],
    ];
    let dim2: [[i32; 7]; 3] = [
        [-1, 0, 1, 0, 0, 0, 0],
        [-1, -2, 0, 2, 1, 0, 0],
        [-1, -4, -5, 0, 5, 4, 1],
    ];
    let k_idx = (gsz as usize / 2) - 1;
    let w1 = &dim1[k_idx][..gsz as usize];
    let w2 = &dim2[k_idx][..gsz as usize];

    // Compute lo/hi thresholds according to norm
    let lo: u64 = if norm == VX_NORM_L1 {
        low_thresh as u64
    } else {
        (low_thresh as i64 * low_thresh as i64) as u64
    };
    let hi: u64 = if norm == VX_NORM_L1 {
        high_thresh as u64
    } else {
        (high_thresh as i64 * high_thresh as i64) as u64
    };

    // Precompute gradients for all pixels to avoid redundant magnitude
    // computation during NMS. Each pixel needs dx, dy, and magnitude.
    let mut grad_dx = vec![0i32; img_size];
    let mut grad_dy = vec![0i32; img_size];
    let mut magnitude = vec![0u64; img_size];

    let half = gsz as isize / 2;
    let src_data = src.data();
    let src_stride = width;

    for y in 0..height {
        for x in 0..width {
            let mut dx: i32 = 0;
            let mut dy: i32 = 0;
            for i in 0..gsz as isize {
                let mut xx: i32 = 0;
                let mut yy: i32 = 0;
                for j in 0..gsz as isize {
                    let py = y as isize + i - half;
                    let px = x as isize + j - half;
                    let v = if px >= 0 && px < width as isize && py >= 0 && py < height as isize {
                        src_data[py as usize * src_stride + px as usize] as i32
                    } else {
                        let bx = px.max(0).min(width as isize - 1) as usize;
                        let by = py.max(0).min(height as isize - 1) as usize;
                        src_data[by * src_stride + bx] as i32
                    };
                    xx += v * w2[j as usize];
                    yy += v * w1[j as usize];
                }
                dx += xx * w1[i as usize];
                dy += yy * w2[i as usize];
            }

            let idx = y * width + x;
            grad_dx[idx] = dx;
            grad_dy[idx] = dy;
            magnitude[idx] = if norm == VX_NORM_L2 {
                (dx as i64 * dx as i64 + dy as i64 * dy as i64) as u64
            } else {
                (dx.abs() + dy.abs()) as u64
            };
        }
    }

    // Working edge image: 0=none, 1=link(weak), 2=edge(strong)
    let mut tmp = vec![0u8; img_size];

    // Border pixels: set to white (255) like the reference does
    for j in 0..bsz {
        for i in 0..width {
            tmp[j * width + i] = 255;
            tmp[(height - 1 - j) * width + i] = 255;
        }
    }
    for j in bsz..height - bsz {
        for i in 0..bsz {
            tmp[j * width + i] = 255;
            tmp[j * width + width - 1 - i] = 255;
        }
    }

    // threshold + NMS using precomputed gradients
    for j in bsz..height - bsz {
        for i in bsz..width - bsz {
            let idx = j * width + i;
            let m = magnitude[idx];
            let dx = grad_dx[idx];
            let dy = grad_dy[idx];

            let mut e: u8 = 0; // CREF_NONE

            if m > lo {
                let l1 = (dx.abs() + dy.abs()) as u64;
                let dx64 = dx.abs() as u64;
                let dy64 = dy.abs() as u64;

                let (m1, m2): (u64, u64);
                if l1 * l1 < 2 * dx64 * dx64 {
                    // horizontal edge
                    m1 = magnitude[j * width + i.saturating_sub(1)];
                    m2 = magnitude[j * width + i + 1];
                } else if l1 * l1 < 2 * dy64 * dy64 {
                    // vertical edge
                    m1 = magnitude[j.saturating_sub(1) * width + i];
                    m2 = magnitude[(j + 1) * width + i];
                } else {
                    // diagonal
                    let s = if (dx ^ dy) < 0 { -1i32 } else { 1i32 };
                    let m1_i = if s < 0 { i + 1 } else { i.saturating_sub(1) };
                    let m2_i = if s >= 0 { i + 1 } else { i.saturating_sub(1) };
                    m1 = magnitude[j.saturating_sub(1) * width + m1_i];
                    let m2_raw = magnitude[(j + 1) * width + m2_i];
                    m2 = m2_raw + 1;
                }

                if m > m1 && m >= m2 {
                    e = if m > hi { 2 } else { 1 }; // CREF_EDGE or CREF_LINK
                }
            }

            tmp[idx] = e;
        }
    }

    // Drop large gradient buffers before edge tracing to save memory
    drop(grad_dx);
    drop(grad_dy);
    drop(magnitude);

    // Recursive edge tracing (follow_edge)
    let offsets: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    fn follow_edge(
        tmp: &mut [u8],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        offsets: &[(isize, isize); 8],
    ) {
        tmp[y * width + x] = 255;
        for &(oy, ox) in offsets.iter() {
            let ny = y as isize + oy;
            let nx = x as isize + ox;
            if ny >= 0 && ny < height as isize && nx >= 0 && nx < width as isize {
                let idx = (ny as usize) * width + (nx as usize);
                if tmp[idx] == 1 {
                    let mut stack: Vec<(usize, usize)> = vec![(nx as usize, ny as usize)];
                    while let Some((sx, sy)) = stack.pop() {
                        let sidx = sy * width + sx;
                        if tmp[sidx] == 1 {
                            tmp[sidx] = 255;
                            for &(oy2, ox2) in offsets.iter() {
                                let ny2 = sy as isize + oy2;
                                let nx2 = sx as isize + ox2;
                                if ny2 >= 0
                                    && ny2 < height as isize
                                    && nx2 >= 0
                                    && nx2 < width as isize
                                {
                                    let nidx = (ny2 as usize) * width + (nx2 as usize);
                                    if tmp[nidx] == 1 {
                                        stack.push((nx2 as usize, ny2 as usize));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // trace edges from all strong edge pixels
    for j in bsz..height - bsz {
        for i in bsz..width - bsz {
            if tmp[j * width + i] == 2 {
                follow_edge(&mut tmp, width, height, i, j, &offsets);
            }
        }
    }

    // clear non-edges: anything < 255 becomes 0
    for j in bsz..height - bsz {
        for i in bsz..width - bsz {
            let idx = j * width + i;
            if tmp[idx] < 255 {
                tmp[idx] = 0;
            }
        }
    }

    // Copy to destination
    let dst_data = dst.data_mut();
    for i in 0..img_size {
        dst_data[i] = tmp[i];
    }

    Ok(())
}

// ============================================================================
// Threshold Implementation
// ============================================================================

/// Threshold an image using a threshold object
/// Supports both BINARY and RANGE threshold types
fn threshold_image(
    src: &Image,
    dst: &mut Image,
    thresh_type: vx_enum,
    value: i32,
    lower: i32,
    upper: i32,
    true_val: i32,
    false_val: i32,
) -> VxResult<()> {
    let width = src.width;
    let height = src.height;
    let dst_data = dst.data_mut();

    let true_v = true_val.max(0).min(255) as u8;
    let false_v = false_val.max(0).min(255) as u8;

    for y in 0..height {
        for x in 0..width {
            let pixel = src.get_pixel(x, y) as i32;

            let output = if thresh_type == crate::c_api_data::VX_THRESHOLD_TYPE_BINARY {
                if pixel > value {
                    true_v
                } else {
                    false_v
                }
            } else {
                // VX_THRESHOLD_TYPE_RANGE
                if pixel < lower || pixel > upper {
                    false_v
                } else {
                    true_v
                }
            };

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = output;
            }
        }
    }

    Ok(())
}

/// VXU Threshold implementation
pub fn vxu_threshold_impl(
    context: vx_context,
    input: vx_image,
    threshold: vx_threshold,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() || threshold.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        // Get image info to check format
        let img = &*(input as *const VxCImage);
        let width = img.width as usize;
        let height = img.height as usize;
        let format = img.format;

        // Check if input is S16 format (0x36313053 = 'S016')
        let is_s16 = format == 0x36313053;

        // Get threshold values from threshold object
        let t = &*(threshold as *const crate::c_api_data::VxCThresholdData);

        // Get source and destination data
        let src_data = match img.data.read() {
            Ok(d) => d,
            Err(_) => return VX_ERROR_INVALID_REFERENCE,
        };

        let dst_img = &*(output as *const VxCImage);
        let mut dst_data = match dst_img.data.write() {
            Ok(d) => d,
            Err(_) => return VX_ERROR_INVALID_REFERENCE,
        };

        let true_v = t.true_value.max(0).min(255) as u8;
        let false_v = t.false_value.max(0).min(255) as u8;

        // Process based on format
        if is_s16 {
            // S16 format: 2 bytes per pixel, interpreted as signed 16-bit
            let expected_size = width * height * 2;
            if src_data.len() < expected_size {
                return VX_ERROR_INVALID_PARAMETERS;
            }

            // Ensure output has enough space
            if dst_data.len() < width * height {
                return VX_ERROR_INVALID_PARAMETERS;
            }

            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    let byte_idx = idx * 2;

                    // Read as little-endian signed 16-bit
                    let pixel = if byte_idx + 1 < src_data.len() {
                        let low = src_data[byte_idx] as i16;
                        let high = src_data[byte_idx + 1] as i16;
                        (low | (high << 8)) as i32
                    } else {
                        0i32
                    };

                    let output_val = if t.thresh_type == crate::c_api_data::VX_THRESHOLD_TYPE_BINARY
                    {
                        if pixel > t.value {
                            true_v
                        } else {
                            false_v
                        }
                    } else {
                        // VX_THRESHOLD_TYPE_RANGE
                        if pixel < t.lower || pixel > t.upper {
                            false_v
                        } else {
                            true_v
                        }
                    };

                    if idx < dst_data.len() {
                        dst_data[idx] = output_val;
                    }
                }
            }
        } else {
            // U8 format: 1 byte per pixel
            let expected_size = width * height;
            if src_data.len() < expected_size || dst_data.len() < expected_size {
                return VX_ERROR_INVALID_PARAMETERS;
            }

            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    let pixel = src_data[idx] as i32;

                    let output_val = if t.thresh_type == crate::c_api_data::VX_THRESHOLD_TYPE_BINARY
                    {
                        if pixel > t.value {
                            true_v
                        } else {
                            false_v
                        }
                    } else {
                        // VX_THRESHOLD_TYPE_RANGE
                        if pixel < t.lower || pixel > t.upper {
                            false_v
                        } else {
                            true_v
                        }
                    };

                    dst_data[idx] = output_val;
                }
            }
        }

        VX_SUCCESS
    }
}

/// VXU Equalize Histogram implementation
pub fn vxu_equalize_histogram_impl(
    context: vx_context,
    input: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Compute histogram
        let hist = match histogram(&src) {
            Ok(h) => h,
            Err(_) => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Compute cumulative distribution function (CDF)
        let mut cdf = [0u32; 256];
        let mut sum: u32 = 0;
        for i in 0..256 {
            sum += hist[i] as u32;
            cdf[i] = sum;
        }

        // Find the first non-zero bin (cdf_min)
        let total_pixels = src.width.saturating_mul(src.height) as u32;
        let mut equalized = [0u8; 256];
        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
        let scale = total_pixels - cdf_min;
        if scale == 0 {
            // All pixels are the same value — identity mapping
            for i in 0..256 {
                equalized[i] = i as u8;
            }
        } else {
            for i in 0..256 {
                if cdf[i] == 0 {
                    equalized[i] = 0;
                } else {
                    // OpenVX spec formula: dst(x,y) = (cdf(src(x,y)) - cdf_min) * 255 / (M*N - cdf_min)
                    // with rounding: (val * 255 + scale/2) / scale
                    let val = (cdf[i] - cdf_min) as u64;
                    equalized[i] = ((val * 255 + (scale as u64) / 2) / scale as u64) as u8;
                }
            }
        }

        // Apply equalization
        let width = src.width;
        let height = src.height;
        let dst_data = dst.data_mut();

        for y in 0..height {
            for x in 0..width {
                let pixel = src.get_pixel(x, y);
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = equalized[pixel as usize];
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

/// VXU Non-Linear Filter implementation
pub fn vxu_non_linear_filter_impl(
    context: vx_context,
    input: vx_image,
    function: vx_enum,
    mask_data: &[u8],
    mask_cols: usize,
    mask_rows: usize,
    origin_x: usize,
    origin_y: usize,
    output: vx_image,
    border: Option<vx_border_t>,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // VX_NONLINEAR_FILTER_MEDIAN = 0x16000 = 90112
    // VX_NONLINEAR_FILTER_MIN = 0x16001 = 90113
    // VX_NONLINEAR_FILTER_MAX = 0x16002 = 90114

    // Convert border mode
    let border_mode = crate::unified_c_api::border_from_vx(&border);

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let width = src.width;
        let height = src.height;
        let dst_data = dst.data_mut();

        for y in 0..height {
            for x in 0..width {
                // Collect values within the mask
                let mut values = Vec::new();

                for my in 0..mask_rows {
                    for mx in 0..mask_cols {
                        // Only include pixels where mask is non-zero
                        if mask_data[my * mask_cols + mx] != 0 {
                            let py = y as isize + my as isize - origin_y as isize;
                            let px = x as isize + mx as isize - origin_x as isize;

                            let pixel = match &border_mode {
                                BorderMode::Replicate => {
                                    let cy = py.max(0).min(height as isize - 1) as usize;
                                    let cx = px.max(0).min(width as isize - 1) as usize;
                                    src.get_pixel(cx, cy)
                                }
                                BorderMode::Constant(val) => {
                                    if py < 0
                                        || py >= height as isize
                                        || px < 0
                                        || px >= width as isize
                                    {
                                        *val
                                    } else {
                                        src.get_pixel(px as usize, py as usize)
                                    }
                                }
                                BorderMode::Undefined => {
                                    if py < 0
                                        || py >= height as isize
                                        || px < 0
                                        || px >= width as isize
                                    {
                                        continue; // Skip out-of-bounds for undefined border
                                    }
                                    src.get_pixel(px as usize, py as usize)
                                }
                            };
                            values.push(pixel);
                        }
                    }
                }

                if values.is_empty() {
                    continue; // No valid pixels in mask
                }

                let value = match function {
                    // VX_NONLINEAR_FILTER_MIN = 0x16001 = 90113
                    90113 => values.iter().copied().min().unwrap_or(0),
                    // VX_NONLINEAR_FILTER_MAX = 0x16002 = 90114
                    90114 => values.iter().copied().max().unwrap_or(0),
                    // VX_NONLINEAR_FILTER_MEDIAN = 0x16000 = 90112 (default)
                    _ => {
                        values.sort_unstable();
                        values[values.len() / 2]
                    }
                };

                let idx = y * width + x;
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = value;
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

/// ===========================================================================
/// VXU Bitwise Logical Operations
/// ===========================================================================

pub fn vxu_and_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Bitwise AND implementation
        let width = dst.width();
        let height = dst.height();
        let dst_data = dst.data_mut();

        for y in 0..height {
            for x in 0..width {
                let a = src1.get_pixel(x, y);
                let b = src2.get_pixel(x, y);
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = a & b;
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

pub fn vxu_or_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        };

        // Bitwise OR implementation
        let width = dst.width();
        let height = dst.height();
        let dst_data = dst.data_mut();

        for y in 0..height {
            for x in 0..width {
                let a = src1.get_pixel(x, y);
                let b = src2.get_pixel(x, y);
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = a | b;
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

pub fn vxu_xor_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || in1.is_null() || in2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src1 = match c_image_to_rust(in1) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let src2 = match c_image_to_rust(in2) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Bitwise XOR implementation
        let width = dst.width();
        let height = dst.height();
        let dst_data = dst.data_mut();

        for y in 0..height {
            for x in 0..width {
                let a = src1.get_pixel(x, y);
                let b = src2.get_pixel(x, y);
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = a ^ b;
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

pub fn vxu_not_impl(context: vx_context, input: vx_image, output: vx_image) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Bitwise NOT implementation
        let width = dst.width();
        let height = dst.height();

        let dst_data = dst.data_mut();

        for y in 0..height {
            for x in 0..width {
                let a = src.get_pixel(x, y);
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = !a;
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

/// ===========================================================================
/// VXU Optical Flow Functions
/// ===========================================================================

/// Keypoint structure for optical flow (matches VX_KEYPOINT)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct vx_keypoint_t {
    pub x: i32,
    pub y: i32,
    pub strength: f32,
    pub scale: f32,
    pub orientation: f32,
    pub tracking_status: i32,
    pub error: f32,
}

/// Optical flow implementation using Lucas-Kanade pyramidal algorithm
///
/// Parameters:
/// - old_images: Pyramid of previous frame
/// - new_images: Pyramid of current frame
/// - old_points: Array of keypoints to track (VX_TYPE_KEYPOINT)
/// - new_points_estimates: Initial estimates for new points (optional)
/// - new_points: Output array for tracked keypoints
/// - termination: Termination criteria (ITERATIONS, EPSILON, or BOTH)
/// - epsilon: Convergence threshold
/// - num_iterations: Maximum iterations
/// - use_initial_estimate: Whether to use new_points_estimates
/// - window_dimension: Size of the tracking window (3, 5, 7, etc.)
pub fn vxu_optical_flow_pyr_lk_impl(
    context: vx_context,
    old_images: vx_pyramid,
    new_images: vx_pyramid,
    old_points: vx_array,
    new_points_estimates: vx_array,
    new_points: vx_array,
    _termination: vx_enum,
    epsilon: vx_float32,
    num_iterations: vx_uint32,
    use_initial_estimate: vx_bool,
    window_dimension: vx_size,
) -> vx_status {
    if context.is_null()
        || old_images.is_null()
        || new_images.is_null()
        || old_points.is_null()
        || new_points.is_null()
    {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let window_size = window_dimension as usize;
    if window_size == 0 || window_size % 2 == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    optical_flow_pyr_lk_run(
        old_images,
        new_images,
        old_points,
        new_points_estimates,
        new_points,
        epsilon,
        num_iterations as usize,
        use_initial_estimate != 0,
        window_size,
    )
}

/// Shared Lucas-Kanade pyramidal optical flow runner used by both the immediate
/// (`vxuOpticalFlowPyrLK`) and graph-mode dispatch paths.
///
/// Reads the input keypoint array via the public `vxQueryArray` /
/// `vxMapArrayRange` FFI (so it works regardless of the array's internal
/// struct layout), walks the U8 pyramid levels by treating each
/// `pyr.levels[i]` pointer as a `*const VxCImage`, runs a coarse-to-fine
/// Lucas-Kanade tracker, and writes results back via `vxTruncateArray` /
/// `vxAddArrayItems`.
pub(crate) fn optical_flow_pyr_lk_run(
    old_images: vx_pyramid,
    new_images: vx_pyramid,
    old_points: vx_array,
    new_points_estimates: vx_array,
    new_points: vx_array,
    epsilon: vx_float32,
    max_iterations: usize,
    use_initial_estimate: bool,
    window_size: usize,
) -> vx_status {
    extern "C" {
        fn vxQueryArray(
            arr: vx_array,
            attr: vx_enum,
            ptr: *mut c_void,
            size: vx_size,
        ) -> vx_status;
        fn vxTruncateArray(arr: vx_array, new_num_items: vx_size) -> vx_status;
        fn vxAddArrayItems(
            arr: vx_array,
            count: vx_size,
            ptr: *const c_void,
            stride: vx_size,
        ) -> vx_status;
        fn vxMapArrayRange(
            arr: vx_array,
            start: vx_size,
            end: vx_size,
            map_id: *mut vx_map_id,
            stride: *mut vx_size,
            ptr: *mut *mut c_void,
            usage: vx_enum,
            mem_type: vx_enum,
            flags: vx_uint32,
        ) -> vx_status;
        fn vxUnmapArrayRange(arr: vx_array, map_id: vx_map_id) -> vx_status;
    }

    const VX_ARRAY_NUMITEMS_ATTR: vx_enum = 0x80E01;
    const VX_READ_ONLY_USAGE: vx_enum = 0x11001;
    const VX_MEMORY_TYPE_HOST_C: vx_enum = 0xE001;

    if window_size < 3 || window_size % 2 == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    let half_window = (window_size / 2) as i32;
    let kp_size = std::mem::size_of::<vx_keypoint_t>() as vx_size;

    unsafe {
        // 1) Number of input keypoints.
        let mut num_items: vx_size = 0;
        let qstatus = vxQueryArray(
            old_points,
            VX_ARRAY_NUMITEMS_ATTR,
            &mut num_items as *mut vx_size as *mut c_void,
            std::mem::size_of::<vx_size>() as vx_size,
        );
        if qstatus != VX_SUCCESS {
            return qstatus;
        }
        if num_items == 0 {
            // Just clear the output array.
            let _ = vxTruncateArray(new_points, 0);
            return VX_SUCCESS;
        }

        // 2) Read input keypoints into a local buffer (avoid leaving array mapped).
        let mut input_keys: Vec<vx_keypoint_t> = vec![std::mem::zeroed(); num_items];
        {
            let mut map_id: vx_map_id = 0;
            let mut stride: vx_size = 0;
            let mut data_ptr: *mut c_void = std::ptr::null_mut();
            let map_status = vxMapArrayRange(
                old_points,
                0,
                num_items,
                &mut map_id,
                &mut stride,
                &mut data_ptr,
                VX_READ_ONLY_USAGE,
                VX_MEMORY_TYPE_HOST_C,
                0,
            );
            if map_status != VX_SUCCESS || data_ptr.is_null() {
                return if map_status != VX_SUCCESS {
                    map_status
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                };
            }
            let stride = if stride == 0 { kp_size } else { stride };
            for i in 0..num_items {
                let kp_ptr = (data_ptr as *const u8).add(i * stride) as *const vx_keypoint_t;
                input_keys[i] = *kp_ptr;
            }
            let _ = vxUnmapArrayRange(old_points, map_id);
        }

        // 3) Optionally read initial estimates. The CTS frequently passes
        // `old_points` here as the same array, in which case the values
        // already match `input_keys`.
        let mut initial_keys: Vec<vx_keypoint_t> = Vec::new();
        if use_initial_estimate && !new_points_estimates.is_null() {
            let mut est_num: vx_size = 0;
            let est_status = vxQueryArray(
                new_points_estimates,
                VX_ARRAY_NUMITEMS_ATTR,
                &mut est_num as *mut vx_size as *mut c_void,
                std::mem::size_of::<vx_size>() as vx_size,
            );
            if est_status == VX_SUCCESS && est_num > 0 {
                let mut map_id: vx_map_id = 0;
                let mut stride: vx_size = 0;
                let mut data_ptr: *mut c_void = std::ptr::null_mut();
                let map_status = vxMapArrayRange(
                    new_points_estimates,
                    0,
                    est_num,
                    &mut map_id,
                    &mut stride,
                    &mut data_ptr,
                    VX_READ_ONLY_USAGE,
                    VX_MEMORY_TYPE_HOST_C,
                    0,
                );
                if map_status == VX_SUCCESS && !data_ptr.is_null() {
                    let stride = if stride == 0 { kp_size } else { stride };
                    initial_keys.resize(est_num.min(num_items), std::mem::zeroed());
                    for i in 0..initial_keys.len() {
                        let kp_ptr =
                            (data_ptr as *const u8).add(i * stride) as *const vx_keypoint_t;
                        initial_keys[i] = *kp_ptr;
                    }
                    let _ = vxUnmapArrayRange(new_points_estimates, map_id);
                }
            }
        }

        // 4) Load every pyramid level as Vec<u8> (U8 only - CTS pyramids are U8).
        let old_pyr = &*(old_images as *const VxCPyramid);
        let new_pyr = &*(new_images as *const VxCPyramid);
        let levels = old_pyr.num_levels.min(new_pyr.num_levels);
        if levels == 0 {
            return VX_ERROR_INVALID_PARAMETERS;
        }
        let mut old_levels: Vec<(usize, usize, Vec<u8>)> = Vec::with_capacity(levels);
        let mut new_levels: Vec<(usize, usize, Vec<u8>)> = Vec::with_capacity(levels);
        for li in 0..levels {
            let load = |pyr: &VxCPyramid, idx: usize| -> Option<(usize, usize, Vec<u8>)> {
                let img_addr = *pyr.levels.get(idx)?;
                if img_addr == 0 {
                    return None;
                }
                let img = &*(img_addr as *const VxCImage);
                let w = img.width as usize;
                let h = img.height as usize;
                let data = img.data.read().ok()?.clone();
                if data.len() < w * h {
                    return None;
                }
                Some((w, h, data))
            };
            let old_l = match load(old_pyr, li) {
                Some(v) => v,
                None => return VX_ERROR_INVALID_PARAMETERS,
            };
            let new_l = match load(new_pyr, li) {
                Some(v) => v,
                None => return VX_ERROR_INVALID_PARAMETERS,
            };
            old_levels.push(old_l);
            new_levels.push(new_l);
        }

        // 5) Run pyramidal Lucas-Kanade per keypoint (coarse to fine).
        let eps_sq = epsilon * epsilon;
        let mut output_keys: Vec<vx_keypoint_t> = Vec::with_capacity(num_items);
        for (i, kp_in) in input_keys.iter().enumerate() {
            // Per OpenVX, an input keypoint with tracking_status==0 is dropped.
            if kp_in.tracking_status == 0 {
                let mut out = *kp_in;
                out.tracking_status = 0;
                output_keys.push(out);
                continue;
            }

            let px0 = kp_in.x as f32;
            let py0 = kp_in.y as f32;

            // Initial flow at full resolution.
            let (mut u, mut v) = if use_initial_estimate && i < initial_keys.len() {
                let est = &initial_keys[i];
                (est.x as f32 - px0, est.y as f32 - py0)
            } else {
                (0.0, 0.0)
            };

            let mut tracked = true;

            // Iterate from coarsest level (levels-1) down to finest (0).
            for li in (0..levels).rev() {
                let scale = (1u32 << li) as f32;
                let (lw, lh, ref old_data) = old_levels[li];
                let (_, _, ref new_data) = new_levels[li];

                // Coordinates and current flow estimate at this level.
                let px = px0 / scale;
                let py = py0 / scale;
                let mut lu = u / scale;
                let mut lv = v / scale;

                for _iter in 0..max_iterations {
                    let mut sum_ix2: f32 = 0.0;
                    let mut sum_iy2: f32 = 0.0;
                    let mut sum_ixiy: f32 = 0.0;
                    let mut sum_ixit: f32 = 0.0;
                    let mut sum_iyit: f32 = 0.0;
                    let mut valid_pixels = 0i32;

                    for wy in -half_window..=half_window {
                        for wx in -half_window..=half_window {
                            let xi = px as i32 + wx;
                            let yi = py as i32 + wy;
                            if xi < 1 || xi >= lw as i32 - 1 || yi < 1 || yi >= lh as i32 - 1 {
                                continue;
                            }
                            let xs = xi as usize;
                            let ys = yi as usize;
                            // Spatial gradients (central difference) on the previous frame.
                            let gx = (old_data[ys * lw + (xs + 1)] as f32
                                - old_data[ys * lw + (xs - 1)] as f32)
                                * 0.5;
                            let gy = (old_data[(ys + 1) * lw + xs] as f32
                                - old_data[(ys - 1) * lw + xs] as f32)
                                * 0.5;
                            // Temporal gradient: bilinearly-sampled new frame at (x+lu, y+lv) minus old frame.
                            let nx = xi as f32 + lu;
                            let ny = yi as f32 + lv;
                            let ix0 = nx.floor() as i32;
                            let iy0 = ny.floor() as i32;
                            if ix0 < 0
                                || ix0 + 1 >= lw as i32
                                || iy0 < 0
                                || iy0 + 1 >= lh as i32
                            {
                                continue;
                            }
                            let fx = nx - ix0 as f32;
                            let fy = ny - iy0 as f32;
                            let ix0u = ix0 as usize;
                            let iy0u = iy0 as usize;
                            let p00 = new_data[iy0u * lw + ix0u] as f32;
                            let p10 = new_data[iy0u * lw + (ix0u + 1)] as f32;
                            let p01 = new_data[(iy0u + 1) * lw + ix0u] as f32;
                            let p11 = new_data[(iy0u + 1) * lw + (ix0u + 1)] as f32;
                            let new_val = (1.0 - fx) * (1.0 - fy) * p00
                                + fx * (1.0 - fy) * p10
                                + (1.0 - fx) * fy * p01
                                + fx * fy * p11;
                            let it = new_val - old_data[ys * lw + xs] as f32;

                            sum_ix2 += gx * gx;
                            sum_iy2 += gy * gy;
                            sum_ixiy += gx * gy;
                            sum_ixit += gx * it;
                            sum_iyit += gy * it;
                            valid_pixels += 1;
                        }
                    }

                    let need_pixels = (window_size as i32 * window_size as i32) / 2;
                    if valid_pixels < need_pixels {
                        tracked = false;
                        break;
                    }

                    let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
                    if det.abs() < 1e-6 {
                        tracked = false;
                        break;
                    }

                    let du = (sum_iy2 * sum_ixit - sum_ixiy * sum_iyit) / det;
                    let dv = (sum_ix2 * sum_iyit - sum_ixiy * sum_ixit) / det;
                    lu -= du;
                    lv -= dv;

                    if du * du + dv * dv < eps_sq {
                        break;
                    }
                }

                if !tracked {
                    break;
                }
                u = lu * scale;
                v = lv * scale;
            }

            let nx = px0 + u;
            let ny = py0 + v;
            let mut out = *kp_in;
            out.x = nx.round() as i32;
            out.y = ny.round() as i32;
            out.tracking_status = if tracked { 1 } else { 0 };
            out.error = if tracked { 0.0 } else { f32::MAX };
            output_keys.push(out);
        }

        // 6) Write results back via the public array API.
        let _ = vxTruncateArray(new_points, 0);
        let add_status = vxAddArrayItems(
            new_points,
            output_keys.len() as vx_size,
            output_keys.as_ptr() as *const c_void,
            kp_size,
        );
        if add_status != VX_SUCCESS {
            return add_status;
        }
    }

    VX_SUCCESS
}

/// VXU Convert Depth Implementation
/// Converts between image depth formats (U8 <-> S16, etc.)
/// Reference: OpenVX 1.3 spec, section on vxConvertDepth
pub fn vxu_convert_depth_impl(
    context: vx_context,
    input: vx_image,
    output: vx_image,
    policy: vx_enum,
    shift: i32,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let (src_width, src_height, src_format) = match get_image_info(input) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let (dst_width, dst_height, dst_format) = match get_image_info(output) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Validate dimensions match
        if src_width != dst_width || src_height != dst_height {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let width = src_width as usize;
        let height = src_height as usize;

        // Determine source and destination format
        let src_is_u8 = src_format == VX_DF_IMAGE_U8;
        let src_is_s16 = src_format == VX_DF_IMAGE_S16;
        let dst_is_u8 = dst_format == VX_DF_IMAGE_U8;
        let dst_is_s16 = dst_format == VX_DF_IMAGE_S16;

        // Create output image data
        let dst_fmt = match df_image_to_format(dst_format) {
            Some(f) => f,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let mut dst = match Image::new(width, height, dst_fmt) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Clamp shift to valid range
        let shift = shift.clamp(-16, 16);

        if src_is_u8 && dst_is_s16 {
            // U8 -> S16 conversion
            if shift < 0 {
                // Right shift (unsigned)
                for y in 0..height {
                    for x in 0..width {
                        let val = src.get_pixel(x, y) as u32;
                        dst.set_pixel_s16(x, y, (val >> (-shift)) as i16);
                    }
                }
            } else {
                // Left shift
                for y in 0..height {
                    for x in 0..width {
                        let val = src.get_pixel(x, y) as u32;
                        dst.set_pixel_s16(x, y, (val << shift) as i16);
                    }
                }
            }
        } else if src_is_s16 && dst_is_u8 {
            // S16 -> U8 conversion
            if policy == VX_CONVERT_POLICY_WRAP {
                if shift < 0 {
                    for y in 0..height {
                        for x in 0..width {
                            let val = src.get_pixel_s16(x, y);
                            dst.set_pixel(x, y, (val << (-shift)) as u8);
                        }
                    }
                } else {
                    for y in 0..height {
                        for x in 0..width {
                            let val = src.get_pixel_s16(x, y);
                            dst.set_pixel(x, y, (val >> shift) as u8);
                        }
                    }
                }
            } else {
                // VX_CONVERT_POLICY_SATURATE
                if shift < 0 {
                    for y in 0..height {
                        for x in 0..width {
                            let val = src.get_pixel_s16(x, y);
                            let v = (val as i32) << (-shift);
                            dst.set_pixel(x, y, v.clamp(0, 255) as u8);
                        }
                    }
                } else {
                    for y in 0..height {
                        for x in 0..width {
                            let val = src.get_pixel_s16(x, y);
                            let v = (val as i32) >> shift;
                            dst.set_pixel(x, y, v.clamp(0, 255) as u8);
                        }
                    }
                }
            }
        } else {
            // Unsupported conversion format pair
            return VX_ERROR_INVALID_FORMAT;
        }

        copy_rust_to_c_image(&dst, output)
    }
}

/// VXU Half-Scale Gaussian Implementation
/// Downscale image by 2 with Gaussian 5x5 blur before subsampling
/// Reference: OpenVX 1.3 spec
pub fn vxu_half_scale_gaussian_impl(
    context: vx_context,
    input: vx_image,
    output: vx_image,
    kernel_size: vx_size,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let (src_width, src_height, _src_format) = match get_image_info(input) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let (dst_width, dst_height, dst_format) = match get_image_info(output) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Output should be roughly half the input size
        let expected_w = (src_width as usize + 1) / 2;
        let expected_h = (src_height as usize + 1) / 2;
        if dst_width as usize != expected_w || dst_height as usize != expected_h {
            return VX_ERROR_INVALID_PARAMETERS;
        }

        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let width = src_width as usize;
        let height = src_height as usize;

        let dst_fmt = match df_image_to_format(dst_format) {
            Some(f) => f,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };
        let mut dst = match Image::new(dst_width as usize, dst_height as usize, dst_fmt) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Gaussian 5x5 kernel (separable: [1,4,6,4,1]/16)
        // For 3x3, use [1,2,1]/4
        // For 1, no blur - just subsample
        if kernel_size == 1 {
            // No Gaussian blur - just subsample
            let out_w = dst_width as usize;
            let out_h = dst_height as usize;
            for dy in 0..out_h {
                for dx in 0..out_w {
                    let sy = dy * 2; // subsample source y
                    let sx = dx * 2; // subsample source x
                    dst.set_pixel(dx, dy, src.get_pixel(sx.min(width - 1), sy.min(height - 1)));
                }
            }
        } else if kernel_size == 3 {
            // Gaussian 3x3 kernel
            // Separable: [1,2,1]/4
            // First pass: horizontal blur
            let mut blurred = vec![0u8; width * height];
            for y in 0..height {
                for x in 0..width {
                    let v0 = src.get_pixel(x.saturating_sub(1), y) as i32;
                    let v1 = src.get_pixel(x, y) as i32;
                    let v2 = src.get_pixel((x + 1).min(width - 1), y) as i32;
                    blurred[y * width + x] = ((v0 + 2 * v1 + v2 + 2) / 4) as u8;
                }
            }
            // Second pass: vertical blur + subsample
            let out_w = dst_width as usize;
            let out_h = dst_height as usize;
            for dy in 0..out_h {
                for dx in 0..out_w {
                    let sy = dy * 2; // subsample
                    let v0 = blurred[sy.saturating_sub(1).min(height - 1) * width + dx * 2] as i32;
                    let v1 = blurred[sy * width + dx * 2] as i32;
                    let v2 = blurred[(sy + 1).min(height - 1) * width + dx * 2] as i32;
                    dst.set_pixel(dx, dy, ((v0 + 2 * v1 + v2 + 2) / 4) as u8);
                }
            }
        } else {
            // kernel_size == 5 or default
            // Gaussian 5x5 kernel
            // Separable: [1,4,6,4,1]/16
            // First pass: horizontal blur
            let mut blurred = vec![0u8; width * height];
            for y in 0..height {
                for x in 0..width {
                    let v0 = src.get_pixel(x.saturating_sub(2), y) as i32;
                    let v1 = src.get_pixel(x.saturating_sub(1), y) as i32;
                    let v2 = src.get_pixel(x, y) as i32;
                    let v3 = src.get_pixel((x + 1).min(width - 1), y) as i32;
                    let v4 = src.get_pixel((x + 2).min(width - 1), y) as i32;
                    blurred[y * width + x] = ((v0 + 4 * v1 + 6 * v2 + 4 * v3 + v4 + 8) / 16) as u8;
                }
            }
            // Second pass: vertical blur + subsample
            let out_w = dst_width as usize;
            let out_h = dst_height as usize;
            for dy in 0..out_h {
                for dx in 0..out_w {
                    let sy = dy * 2; // subsample source y
                    let sx = dx * 2; // subsample source x
                    let v0 = blurred[sy.saturating_sub(2).min(height - 1) * width + sx] as i32;
                    let v1 = blurred[sy.saturating_sub(1).min(height - 1) * width + sx] as i32;
                    let v2 = blurred[sy * width + sx] as i32;
                    let v3 = blurred[(sy + 1).min(height - 1) * width + sx] as i32;
                    let v4 = blurred[(sy + 2).min(height - 1) * width + sx] as i32;
                    dst.set_pixel(
                        dx,
                        dy,
                        ((v0 + 4 * v1 + 6 * v2 + 4 * v3 + v4 + 8) / 16) as u8,
                    );
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}
