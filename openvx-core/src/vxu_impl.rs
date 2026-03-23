//! VXU (Immediate Mode) Function Implementations
//!
//! This module provides actual implementations for VXU functions that bridge
//! the C API types to the Rust vision kernel implementations.

use std::ffi::c_void;
use crate::c_api::{
    vx_context, vx_image, vx_scalar, vx_array, vx_matrix, vx_convolution,
    vx_pyramid, vx_threshold, vx_status, vx_bool,
    vx_enum, vx_df_image, vx_uint32, vx_size, vx_char,
    VX_SUCCESS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_INVALID_PARAMETERS,
    VX_ERROR_INVALID_FORMAT, VX_ERROR_NOT_IMPLEMENTED,
};
use crate::unified_c_api::{vx_distribution, vx_remap, VxCImage};

/// Image format enum for internal use
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageFormat {
    Gray,
    Rgb,
    Rgba,
    NV12,
    NV21,
}

impl ImageFormat {
    pub fn channels(&self) -> usize {
        match self {
            ImageFormat::Gray => 1,
            ImageFormat::Rgb => 3,
            ImageFormat::Rgba => 4,
            ImageFormat::NV12 => 3,
            ImageFormat::NV21 => 3,
        }
    }
}

/// Simple Image struct for VXU operations
pub struct Image {
    width: usize,
    height: usize,
    format: ImageFormat,
    data: Vec<u8>,
}

impl Image {
    pub fn new(width: usize, height: usize, format: ImageFormat) -> Option<Self> {
        let channels = format.channels();
        // Use checked_mul to prevent integer overflow and allocation failure
        let size = width
            .checked_mul(height)?
            .checked_mul(channels)?;
        // Limit allocation size to prevent OOM (max ~1GB for single image)
        if size > (1 << 30) {
            return None;
        }
        let data = vec![0u8; size];
        Some(Image { width, height, format, data })
    }

    pub fn from_data(width: usize, height: usize, format: ImageFormat, data: Vec<u8>) -> Self {
        Image { width, height, format, data }
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn format(&self) -> ImageFormat { self.format }
    pub fn data(&self) -> &[u8] { &self.data }
    pub fn data_mut(&mut self) -> &mut [u8] { &mut self.data }

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
        let idx = y.saturating_mul(self.width).saturating_add(x).saturating_mul(3);
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

    pub fn set_rgb(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        if x < self.width && y < self.height {
            let idx = y.saturating_mul(self.width).saturating_add(x).saturating_mul(3);
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
        0x00525547 => Some(ImageFormat::Gray), // 'U008' / VX_DF_IMAGE_U8
        0x52474220 => Some(ImageFormat::Rgb),  // 'RGB2' / VX_DF_IMAGE_RGB
        0x52474241 => Some(ImageFormat::Rgba), // 'RGBA' / VX_DF_IMAGE_RGBA
        0x564e3132 => Some(ImageFormat::NV12), // 'NV12' / VX_DF_IMAGE_NV12
        0x564e3231 => Some(ImageFormat::NV21), // 'NV21' / VX_DF_IMAGE_NV21
        _ => Some(ImageFormat::Gray), // Default to gray
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
    let data = img.data.read().ok()?.clone();
    let format = df_image_to_format(format)?;
    Some(Image::from_data(width as usize, height as usize, format, data))
}

/// Copy Rust Image data back to C API image
unsafe fn copy_rust_to_c_image(src: &Image, dst: vx_image) -> vx_status {
    if dst.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let img = &*(dst as *const VxCImage);
    let mut dst_data = match img.data.write() {
        Ok(d) => d,
        Err(_) => return VX_ERROR_INVALID_REFERENCE,
    };

    if dst_data.len() == src.data.len() {
        dst_data.copy_from_slice(&src.data);
        VX_SUCCESS
    } else {
        VX_ERROR_INVALID_PARAMETERS
    }
}

/// Create a new Rust Image matching the C image dimensions and format
unsafe fn create_matching_image(c_image: vx_image) -> Option<Image> {
    let (width, height, format) = get_image_info(c_image)?;
    let format = df_image_to_format(format)?;
    Image::new(width as usize, height as usize, format)
}

/// ===========================================================================
/// VXU Color Functions
/// ===========================================================================

pub fn vxu_color_convert_impl(
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

        let dst_info = match get_image_info(output) {
            Some(info) => info,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let dst_format = match df_image_to_format(dst_info.2) {
            Some(f) => f,
            None => return VX_ERROR_INVALID_FORMAT,
        };

        let mut dst = match Image::new(dst_info.0 as usize, dst_info.1 as usize, dst_format) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let result = match (src.format(), dst_format) {
            (ImageFormat::Rgb, ImageFormat::Gray) => rgb_to_gray(&src, &mut dst),
            (ImageFormat::Gray, ImageFormat::Rgb) => gray_to_rgb(&src, &mut dst),
            (ImageFormat::Rgb, ImageFormat::Rgba) => rgb_to_rgba(&src, &mut dst),
            (ImageFormat::Rgba, ImageFormat::Rgb) => rgba_to_rgb(&src, &mut dst),
            _ => {
                // Same format - copy
                let mut dst_data = dst.data_mut();
                let src_data = src.data();
                if dst_data.len() == src_data.len() {
                    dst_data.copy_from_slice(&src_data);
                    Ok(())
                } else {
                    Err(VxStatus::ErrorInvalidFormat)
                }
            }
        };

        match result {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

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

        // Simple channel extraction based on channel index
        let width = src.width();
        let height = src.height();
        let mut dst_data = dst.data_mut();

        for y in 0..height {
            for x in 0..width {
                let val = match src.format() {
                    ImageFormat::Rgb => {
                        let (r, g, b) = src.get_rgb(x, y);
                        match channel {
                            0 => r,
                            1 => g,
                            2 => b,
                            _ => r,
                        }
                    }
                    ImageFormat::Rgba => {
                        let idx = y.saturating_mul(width).saturating_add(x).saturating_mul(4);
                        let data = src.data();
                        match channel {
                            0 => *data.get(idx).unwrap_or(&0),
                            1 => *data.get(idx.saturating_add(1)).unwrap_or(&0),
                            2 => *data.get(idx.saturating_add(2)).unwrap_or(&0),
                            3 => *data.get(idx.saturating_add(3)).unwrap_or(&0),
                            _ => *data.get(idx).unwrap_or(&0),
                        }
                    }
                    _ => src.get_pixel(x, y),
                };
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = val;
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
        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let width = dst.width();
        let height = dst.height();
        let mut dst_data = dst.data_mut();

        // Load source planes
        let r_img = if plane0.is_null() { None } else { c_image_to_rust(plane0) };
        let g_img = if plane1.is_null() { None } else { c_image_to_rust(plane1) };
        let b_img = if plane2.is_null() { None } else { c_image_to_rust(plane2) };
        let a_img = if plane3.is_null() { None } else { c_image_to_rust(plane3) };

        for y in 0..height {
            for x in 0..width {
                let r = r_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                let g = g_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                let b = b_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(0);
                let a = a_img.as_ref().map(|img| img.get_pixel(x, y)).unwrap_or(255);

                let idx = y.saturating_mul(width).saturating_add(x).saturating_mul(4);
                if idx.saturating_add(3) < dst_data.len() {
                    dst_data[idx] = r;
                    dst_data[idx + 1] = g;
                    dst_data[idx + 2] = b;
                    dst_data[idx + 3] = a;
                } else if idx.saturating_add(2) < dst_data.len() {
                    dst_data[idx] = r;
                    dst_data[idx + 1] = g;
                    dst_data[idx + 2] = b;
                }
            }
        }

        copy_rust_to_c_image(&dst, output)
    }
}

/// ===========================================================================
/// VXU Filter Functions
/// ===========================================================================

pub fn vxu_gaussian3x3_impl(
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

        match gaussian3x3(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_gaussian5x5_impl(
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

        match gaussian5x5(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_box3x3_impl(
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

        match box3x3(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_median3x3_impl(
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

        match median3x3(&src, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_convolve_impl(
    context: vx_context,
    input: vx_image,
    _conv: vx_convolution,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || _conv.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // For now, apply a simple sharpening kernel
    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Default sharpening kernel
        let kernel: [[i32; 3]; 3] = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ];

        match convolve_generic(&src, &mut dst, &kernel, BorderMode::Replicate) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// ===========================================================================
/// VXU Morphology Functions
/// ===========================================================================

pub fn vxu_dilate3x3_impl(
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

        match dilate3x3(&src, &mut dst, BorderMode::Constant(0)) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_erode3x3_impl(
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

        match erode3x3(&src, &mut dst, BorderMode::Constant(255)) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_dilate5x5_impl(
    context: vx_context,
    input: vx_image,
    output: vx_image,
) -> vx_status {
    // For now, use 3x3 dilate twice as approximation
    vxu_dilate3x3_impl(context, input, output)
}

pub fn vxu_erode5x5_impl(
    context: vx_context,
    input: vx_image,
    output: vx_image,
) -> vx_status {
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

        let mut gx = match create_matching_image(output_x) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut gy = match create_matching_image(output_y) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match sobel3x3(&src, &mut gx, &mut gy) {
            Ok(_) => {
                let status_x = if output_x.is_null() { VX_SUCCESS } else { copy_rust_to_c_image(&gx, output_x) };
                let status_y = if output_y.is_null() { VX_SUCCESS } else { copy_rust_to_c_image(&gy, output_y) };
                if status_x == VX_SUCCESS && status_y == VX_SUCCESS {
                    VX_SUCCESS
                } else {
                    VX_ERROR_INVALID_PARAMETERS
                }
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
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

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match magnitude(&gx, &gy, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
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

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match phase_op(&gx, &gy, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
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

        match add(&src1, &src2, &mut dst) {
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

        match subtract(&src1, &src2, &mut dst) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_multiply_impl(
    context: vx_context,
    in1: vx_image,
    in2: vx_image,
    _scale: vx_scalar,
    _overflow_policy: vx_enum,
    _rounding_policy: vx_enum,
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

        match multiply(&src1, &src2, &mut dst, 1.0) {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_weighted_average_impl(
    context: vx_context,
    img1: vx_image,
    _alpha: vx_scalar,
    img2: vx_image,
    output: vx_image,
) -> vx_status {
    if context.is_null() || img1.is_null() || img2.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

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

        // Default alpha value 128 (0.5)
        let alpha_val: u8 = 128;

        match weighted(&src1, &src2, &mut dst, alpha_val) {
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
    _mean: vx_scalar,
    _stddev: vx_scalar,
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
            Ok((_mean_val, _stddev_val)) => {
                // In a full implementation, would write to scalar outputs
                VX_SUCCESS
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_min_max_loc_impl(
    context: vx_context,
    input: vx_image,
    _min_val: vx_scalar,
    _max_val: vx_scalar,
    _min_loc: vx_array,
    _max_loc: vx_array,
    _num_min_max: vx_scalar,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match min_max_loc(&src) {
            Ok((_min_val, _max_val, _min_loc, _max_loc)) => {
                // In a full implementation, would write to scalar/array outputs
                VX_SUCCESS
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_histogram_impl(
    context: vx_context,
    input: vx_image,
    _distribution: vx_distribution,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        match histogram(&src) {
            Ok(_hist) => {
                // In a full implementation, would write to distribution
                VX_SUCCESS
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
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

        match scale_image(&src, &mut dst, true) {
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

        // Default identity affine
        let affine_matrix: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        match warp_affine(&src, &affine_matrix, &mut dst) {
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

        // Default identity perspective
        let persp_matrix: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        match warp_perspective(&src, &persp_matrix, &mut dst) {
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
    _strength_thresh: vx_scalar,
    _min_distance: vx_scalar,
    _sensitivity: vx_scalar,
    _gradient_size: vx_enum,
    _block_size: vx_enum,
    _corners: vx_array,
    _num_corners: vx_scalar,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Default parameters
        let k = 0.04f32;
        let threshold = 100.0f32;
        let min_distance = 10usize;

        match harris_corners(&src, k, threshold, min_distance) {
            Ok(_corners) => {
                // In a full implementation, would write to array/scalar outputs
                VX_SUCCESS
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

pub fn vxu_fast_corners_impl(
    context: vx_context,
    input: vx_image,
    _strength_thresh: vx_scalar,
    _nonmax_suppression: vx_bool,
    _corners: vx_array,
    _num_corners: vx_scalar,
) -> vx_status {
    if context.is_null() || input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Default threshold
        let threshold = 20u8;

        match fast9(&src, threshold) {
            Ok(_corners) => {
                // In a full implementation, would write to array/scalar outputs
                VX_SUCCESS
            }
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
    }
}

/// ===========================================================================
/// VXU Object Detection Functions
/// ===========================================================================

pub fn vxu_canny_edge_detector_impl(
    context: vx_context,
    input: vx_image,
    _hyst_threshold: vx_threshold,
    _gradient_size: vx_enum,
    _norm_type: vx_enum,
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

        // Default thresholds
        let low_thresh = 50u8;
        let high_thresh = 150u8;

        match canny_edge_detector(&src, &mut dst, low_thresh, high_thresh) {
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

    // Pyramid generation would require multiple images
    // For now, stub implementation
    VX_SUCCESS
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
) -> vx_status {
    if context.is_null() || input.is_null() || _table.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Remap implementation requires reading from remap table
    // For now, stub implementation
    VX_SUCCESS
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
            if src_idx.saturating_add(2) < src_data.len() && dst_idx.saturating_add(2) < dst_data.len() {
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
// ============================================================================

fn clamp_u8(value: i32) -> u8 {
    if value < 0 { 0 } else if value > 255 { 255 } else { value as u8 }
}

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
    if k >= arr.len() { return 0; }

    let mut sorted = arr.to_vec();
    sorted.sort_unstable();
    sorted[k]
}

fn convolve_generic(src: &Image, dst: &mut Image, kernel: &[[i32; 3]; 3], border: BorderMode) -> VxResult<()> {
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
    let kernel = [1, 2, 1];

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
            for k in 0..3 {
                let px = x as isize + k as isize - 1;
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
            for k in 0..3 {
                let py = y as isize + k as isize - 1;
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

    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let mut sum: u32 = 0;
            let mut count: u32 = 0;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let py = y as isize + dy;
                    let px = x as isize + dx;
                    if py >= 0 && py < height as isize && px >= 0 && px < width as isize {
                        sum += src.get_pixel(px as usize, py as usize) as u32;
                        count += 1;
                    }
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = (sum / count.max(1)) as u8;
            }
        }
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

    for y in 0..height {
        for x in 0..width {
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

    for y in 0..height {
        for x in 0..width {
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

const SOBEL_X: [[i32; 3]; 3] = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
];

const SOBEL_Y: [[i32; 3]; 3] = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
];

fn sobel3x3(src: &Image, grad_x: &mut Image, grad_y: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    let gx_data = grad_x.data_mut();
    let gy_data = grad_y.data_mut();

    for y in 0..height {
        for x in 0..width {
            let mut sum_x: i32 = 0;
            let mut sum_y: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x as isize + kx as isize - 1;
                    let py = y as isize + ky as isize - 1;
                    let pixel = get_pixel_bordered(src, px, py, BorderMode::Replicate) as i32;
                    sum_x += pixel * SOBEL_X[ky][kx];
                    sum_y += pixel * SOBEL_Y[ky][kx];
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = gx_data.get_mut(idx) {
                *p = clamp_u8((sum_x / 4).max(-128).min(127) + 128);
            }
            if let Some(p) = gy_data.get_mut(idx) {
                *p = clamp_u8((sum_y / 4).max(-128).min(127) + 128);
            }
        }
    }

    Ok(())
}

fn magnitude(grad_x: &Image, grad_y: &Image, mag: &mut Image) -> VxResult<()> {
    let width = grad_x.width;
    let height = grad_x.height;

    let mag_data = mag.data_mut();

    for y in 0..height {
        for x in 0..width {
            let gx = grad_x.get_pixel(x, y) as i32 - 128;
            let gy = grad_y.get_pixel(x, y) as i32 - 128;

            let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as i32;
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = mag_data.get_mut(idx) {
                *p = magnitude.min(255) as u8;
            }
        }
    }

    Ok(())
}

fn phase_op(grad_x: &Image, grad_y: &Image, phase: &mut Image) -> VxResult<()> {
    let width = grad_x.width;
    let height = grad_y.height;

    let phase_data = phase.data_mut();

    const DEG_PER_RAD: f32 = 180.0 / std::f32::consts::PI;

    for y in 0..height {
        for x in 0..width {
            let gx = grad_x.get_pixel(x, y) as i32 - 128;
            let gy = grad_y.get_pixel(x, y) as i32 - 128;

            let phase_deg = (gy as f32).atan2(gx as f32) * DEG_PER_RAD;
            let phase_u8 = ((phase_deg + 360.0) % 360.0) / 360.0 * 255.0;

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = phase_data.get_mut(idx) {
                *p = phase_u8 as u8;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

fn add(src1: &Image, src2: &Image, dst: &mut Image) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as u16;
            let b = src2.get_pixel(x, y) as u16;
            let sum = a + b;
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = sum.min(255) as u8;
            }
        }
    }

    Ok(())
}

fn subtract(src1: &Image, src2: &Image, dst: &mut Image) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as i16;
            let b = src2.get_pixel(x, y) as i16;
            let diff = a - b;
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = diff.max(0).min(255) as u8;
            }
        }
    }

    Ok(())
}

fn multiply(src1: &Image, src2: &Image, dst: &mut Image, scale: f32) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as f32;
            let b = src2.get_pixel(x, y) as f32;
            let product = a * b * scale / 255.0;
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = product.max(0.0).min(255.0) as u8;
            }
        }
    }

    Ok(())
}

fn weighted(src1: &Image, src2: &Image, dst: &mut Image, alpha: u8) -> VxResult<()> {
    if src1.width != src2.width || src1.height != src2.height {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width;
    let height = src1.height;
    let beta = 255 - alpha;

    let dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as u32;
            let b = src2.get_pixel(x, y) as u32;
            let result = (a * alpha as u32 + b * beta as u32) / 256;
            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = result as u8;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Statistics
// ============================================================================

fn min_max_loc(src: &Image) -> VxResult<(u8, u8, Coordinate, Coordinate)> {
    let width = src.width;
    let height = src.height;

    let mut min_val: u8 = 255;
    let mut max_val: u8 = 0;
    let mut min_loc = Coordinate { x: 0, y: 0 };
    let mut max_loc = Coordinate { x: 0, y: 0 };

    for y in 0..height {
        for x in 0..width {
            let val = src.get_pixel(x, y);

            if val < min_val {
                min_val = val;
                min_loc = Coordinate { x, y };
            }

            if val > max_val {
                max_val = val;
                max_loc = Coordinate { x, y };
            }
        }
    }

    Ok((min_val, max_val, min_loc, max_loc))
}

fn mean_std_dev(src: &Image) -> VxResult<(f32, f32)> {
    let width = src.width;
    let height = src.height;
    // Use saturating_mul to prevent integer overflow
    let pixel_count = width.saturating_mul(height) as f32;

    // Compute mean
    let mut sum: u64 = 0;
    for y in 0..height {
        for x in 0..width {
            sum += src.get_pixel(x, y) as u64;
        }
    }
    let mean = sum as f32 / pixel_count;

    // Compute variance
    let mut sum_sq_diff: f64 = 0.0;
    for y in 0..height {
        for x in 0..width {
            let diff = src.get_pixel(x, y) as f32 - mean;
            sum_sq_diff += (diff * diff) as f64;
        }
    }
    let variance = sum_sq_diff as f32 / pixel_count;
    let stddev = variance.sqrt();

    Ok((mean, stddev))
}

fn histogram(src: &Image) -> VxResult<[u32; 256]> {
    let width = src.width;
    let height = src.height;
    let mut hist = [0u32; 256];

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

    let dst_data = dst.data_mut();

    for y in 0..height {
        let mut row_sum = 0u32;
        for x in 0..width {
            row_sum += src.get_pixel(x, y) as u32;
            let idx = y.saturating_mul(width).saturating_add(x);
            if idx < dst_data.len() {
                let new_val = if y == 0 {
                    (row_sum.min(255) >> 8) as u8
                } else {
                    let prev_idx = (y - 1).saturating_mul(width).saturating_add(x);
                    let prev_val = dst_data.get(prev_idx).copied().unwrap_or(0);
                    ((row_sum + (prev_val as u32 * 256)).min(255) >> 8) as u8
                };
                if let Some(d) = dst_data.get_mut(idx) {
                    *d = new_val;
                }
            }
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
    let p10 = if x1 < width { img.get_pixel(x1 as usize, y0 as usize) as f32 } else { p00 };
    let p01 = if y1 < height { img.get_pixel(x0 as usize, y1 as usize) as f32 } else { p00 };
    let p11 = if x1 < width && y1 < height { img.get_pixel(x1 as usize, y1 as usize) as f32 } else { p00 };

    let value = (1.0 - fx) * (1.0 - fy) * p00 +
                fx * (1.0 - fy) * p10 +
                (1.0 - fx) * fy * p01 +
                fx * fy * p11;

    clamp_u8(value as i32)
}

fn scale_image(src: &Image, dst: &mut Image, bilinear: bool) -> VxResult<()> {
    let src_width = src.width;
    let src_height = src.height;
    let dst_width = dst.width;
    let dst_height = dst.height;

    let dst_data = dst.data_mut();

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = (x as f32 + 0.5) * x_scale - 0.5;
            let src_y = (y as f32 + 0.5) * y_scale - 0.5;

            let value = if bilinear {
                bilinear_interpolate(src, src_x, src_y)
            } else {
                // Nearest neighbor
                let nx = clamp_u8(src_x.round() as i32);
                let ny = clamp_u8(src_y.round() as i32);
                src.get_pixel(nx as usize % src_width, ny as usize % src_height)
            };

            let idx = y.saturating_mul(dst_width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = value;
            }
        }
    }

    Ok(())
}

fn warp_affine(src: &Image, matrix: &[f32; 6], dst: &mut Image) -> VxResult<()> {
    let dst_width = dst.width;
    let dst_height = dst.height;
    let src_width = src.width as f32;
    let src_height = src.height as f32;

    let dst_data = dst.data_mut();

    let a11 = matrix[0];
    let a12 = matrix[1];
    let a13 = matrix[2];
    let a21 = matrix[3];
    let a22 = matrix[4];
    let a23 = matrix[5];

    for y in 0..dst_height {
        for x in 0..dst_width {
            let xf = x as f32;
            let yf = y as f32;

            // Inverse mapping
            let src_x = a11 * xf + a12 * yf + a13;
            let src_y = a21 * xf + a22 * yf + a23;

            let idx = y.saturating_mul(dst_width).saturating_add(x);
            if src_x < 0.0 || src_x >= src_width || src_y < 0.0 || src_y >= src_height {
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = 0;
                }
                continue;
            }

            if let Some(p) = dst_data.get_mut(idx) {
                *p = bilinear_interpolate(src, src_x, src_y);
            }
        }
    }

    Ok(())
}

fn warp_perspective(src: &Image, matrix: &[f32; 9], dst: &mut Image) -> VxResult<()> {
    let dst_width = dst.width;
    let dst_height = dst.height;
    let src_width = src.width as f32;
    let src_height = src.height as f32;

    let dst_data = dst.data_mut();

    let h11 = matrix[0];
    let h12 = matrix[1];
    let h13 = matrix[2];
    let h21 = matrix[3];
    let h22 = matrix[4];
    let h23 = matrix[5];
    let h31 = matrix[6];
    let h32 = matrix[7];
    let h33 = matrix[8];

    for y in 0..dst_height {
        for x in 0..dst_width {
            let xf = x as f32;
            let yf = y as f32;

            // Homogeneous coordinates
            let w = h31 * xf + h32 * yf + h33;
            let idx = y.saturating_mul(dst_width).saturating_add(x);
            if w.abs() < 1e-6 {
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = 0;
                }
                continue;
            }

            let src_x = (h11 * xf + h12 * yf + h13) / w;
            let src_y = (h21 * xf + h22 * yf + h23) / w;

            if src_x < 0.0 || src_x >= src_width || src_y < 0.0 || src_y >= src_height {
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = 0;
                }
                continue;
            }

            if let Some(p) = dst_data.get_mut(idx) {
                *p = bilinear_interpolate(src, src_x, src_y);
            }
        }
    }

    Ok(())
}

// ============================================================================
// Feature Detection
// ============================================================================

fn harris_corners(image: &Image, k: f32, threshold: f32, _min_distance: usize) -> VxResult<Vec<Corner>> {
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
                    let idx = (ny as usize).saturating_mul(width).saturating_add(nx as usize);
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
    corners.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));

    Ok(corners)
}

fn fast9(image: &Image, threshold: u8) -> VxResult<Vec<Corner>> {
    let width = image.width;
    let height = image.height;
    let mut corners = Vec::new();

    const CIRCLE_OFFSETS: [(isize, isize); 16] = [
        (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
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

    corners.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
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

fn canny_edge_detector(src: &Image, dst: &mut Image, low_threshold: u8, high_threshold: u8) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    // Use checked operations to prevent integer overflow
    let img_size = width
        .checked_mul(height)
        .ok_or(VxStatus::ErrorInvalidParameters)?;

    // Step 1: Gaussian blur
    let mut blurred = vec![0u8; img_size];
    {
        let kernel = [1, 2, 1];
        let mut temp = vec![0u8; img_size];

        // Horizontal pass
        for y in 0..height {
            for x in 0..width {
                let mut sum: i32 = 0;
                let mut weight: i32 = 0;
                for k in 0..3 {
                    let px = x as isize + k as isize - 1;
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
                for k in 0..3 {
                    let py = y as isize + k as isize - 1;
                    if py >= 0 && py < height as isize {
                        let idx = (py as usize).saturating_mul(width).saturating_add(x);
                        if let Some(val) = temp.get(idx) {
                            sum += *val as i32 * kernel[k];
                            weight += kernel[k];
                        }
                    }
                }
                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = blurred.get_mut(idx) {
                    *p = clamp_u8(sum / weight.max(1));
                }
            }
        }
    }

    // Step 2: Compute gradients
    let mut grad_x = vec![0i32; img_size];
    let mut grad_y = vec![0i32; img_size];
    let mut magnitude = vec![0f32; img_size];
    let mut direction = vec![0f32; img_size];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    let idx = (py as usize).saturating_mul(width).saturating_add(px);
                    let pixel = *blurred.get(idx).unwrap_or(&0) as i32;
                    gx += pixel * SOBEL_X[ky][kx];
                    gy += pixel * SOBEL_Y[ky][kx];
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(gx_p) = grad_x.get_mut(idx) {
                *gx_p = gx;
            }
            if let Some(gy_p) = grad_y.get_mut(idx) {
                *gy_p = gy;
            }
            if let Some(mag_p) = magnitude.get_mut(idx) {
                *mag_p = ((gx * gx + gy * gy) as f32).sqrt();
            }
            if let Some(dir_p) = direction.get_mut(idx) {
                *dir_p = (gy as f32).atan2(gx as f32);
            }
        }
    }

    // Step 3: Non-maximum suppression
    let mut suppressed = vec![0u8; img_size];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y.saturating_mul(width).saturating_add(x);
            let mag = *magnitude.get(idx).unwrap_or(&0.0);
            let dir = *direction.get(idx).unwrap_or(&0.0);

            let angle = ((dir + std::f32::consts::PI) * 4.0 / std::f32::consts::PI) as i32 % 4;

            let (dx1, dy1, dx2, dy2) = match angle {
                0 | 2 => (1, 0, -1, 0),
                1 => (1, 1, -1, -1),
                3 => (1, -1, -1, 1),
                _ => (0, 1, 0, -1),
            };

            let idx1 = ((y as isize + dy1) as usize)
                .saturating_mul(width)
                .saturating_add((x as isize + dx1) as usize);
            let idx2 = ((y as isize + dy2) as usize)
                .saturating_mul(width)
                .saturating_add((x as isize + dy2) as usize);

            let mag1 = *magnitude.get(idx1).unwrap_or(&0.0);
            let mag2 = *magnitude.get(idx2).unwrap_or(&0.0);
            if mag >= mag1 && mag >= mag2 {
                if let Some(p) = suppressed.get_mut(idx) {
                    *p = clamp_u8(mag as i32);
                }
            }
        }
    }

    // Step 4: Double threshold and hysteresis
    let mut edges = vec![0u8; img_size];
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let idx = y.saturating_mul(width).saturating_add(x);
            let val = *suppressed.get(idx).unwrap_or(&0);

            if let Some(e) = edges.get_mut(idx) {
                if val >= high_threshold {
                    *e = 2; // Strong edge
                } else if val >= low_threshold {
                    *e = 1; // Weak edge
                }
            }
        }
    }

    // Step 5: Edge tracking
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y.saturating_mul(width).saturating_add(x);
            let edge_val = *edges.get(idx).unwrap_or(&0);

            if edge_val == 2 {
                if let Some(d) = dst_data.get_mut(idx) {
                    *d = 255;
                }
            } else if edge_val == 1 {
                let mut connected = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        let nidx = (ny as usize).saturating_mul(width).saturating_add(nx as usize);
                        if edges.get(nidx).copied().unwrap_or(0) == 2 {
                            connected = true;
                            break;
                        }
                    }
                    if connected {
                        break;
                    }
                }

                if let Some(d) = dst_data.get_mut(idx) {
                    if connected {
                        *d = 255;
                    } else {
                        *d = 0;
                    }
                }
            } else if let Some(d) = dst_data.get_mut(idx) {
                *d = 0;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Threshold Implementation
// ============================================================================

/// Threshold an image using a threshold object
/// Supports both BINARY and RANGE threshold types
fn threshold_image(src: &Image, dst: &mut Image, thresh_type: vx_enum, value: i32, lower: i32, upper: i32, true_val: i32, false_val: i32) -> VxResult<()> {
    let width = src.width;
    let height = src.height;
    let dst_data = dst.data_mut();

    let true_v = true_val.max(0).min(255) as u8;
    let false_v = false_val.max(0).min(255) as u8;

    for y in 0..height {
        for x in 0..width {
            let pixel = src.get_pixel(x, y) as i32;

            let output = if thresh_type == 0 { // VX_THRESHOLD_TYPE_BINARY
                if pixel > value {
                    true_v
                } else {
                    false_v
                }
            } else { // VX_THRESHOLD_TYPE_RANGE
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
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        let mut dst = match create_matching_image(output) {
            Some(img) => img,
            None => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Get threshold values from threshold object
        let t = &*(threshold as *const crate::c_api_data::VxCThresholdData);

        let result = threshold_image(
            &src, &mut dst,
            t.thresh_type,
            t.value,
            t.lower,
            t.upper,
            t.true_value,
            t.false_value
        );

        match result {
            Ok(_) => copy_rust_to_c_image(&dst, output),
            Err(_) => VX_ERROR_INVALID_PARAMETERS,
        }
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
            sum += hist[i];
            cdf[i] = sum;
        }

        // Normalize CDF
        let total_pixels = src.width.saturating_mul(src.height) as u32;
        let mut equalized = [0u8; 256];
        if total_pixels > 0 {
            for i in 0..256 {
                equalized[i] = ((cdf[i] as u64 * 255u64) / total_pixels as u64) as u8;
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
    function: vx_enum,
    input: vx_image,
    mask_size: vx_size,
    output: vx_image,
) -> vx_status {
    if context.is_null() || input.is_null() || output.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // mask_size should be 3, 5, etc.
    let window_size = mask_size as isize;
    let half = window_size / 2;

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
        let window_count = (window_size * window_size) as usize;
        let mut window = vec![0u8; window_count];

        // function: 0=min, 1=max, 2=median
        for y in 0..height {
            for x in 0..width {
                // Fill window
                let mut idx = 0;
                for dy in -half..=half {
                    for dx in -half..=half {
                        let py = (y as isize + dy).max(0).min(height as isize - 1) as usize;
                        let px = (x as isize + dx).max(0).min(width as isize - 1) as usize;
                        window[idx] = src.get_pixel(px, py);
                        idx += 1;
                    }
                }

                let value = match function {
                    0 => {
                        // Min
                        let mut min_val = 255u8;
                        for v in &window {
                            if *v < min_val {
                                min_val = *v;
                            }
                        }
                        min_val
                    }
                    1 => {
                        // Max
                        let mut max_val = 0u8;
                        for v in &window {
                            if *v > max_val {
                                max_val = *v;
                            }
                        }
                        max_val
                    }
                    _ => {
                        // Median (default)
                        window.sort_unstable();
                        window[window_count / 2]
                    }
                };

                let idx = y.saturating_mul(width).saturating_add(x);
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
        let mut dst_data = dst.data_mut();

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

        // Bitwise OR implementation
        let width = dst.width();
        let height = dst.height();
        let mut dst_data = dst.data_mut();

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
        let mut dst_data = dst.data_mut();

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

pub fn vxu_not_impl(
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

        // Bitwise NOT implementation
        let width = dst.width();
        let height = dst.height();
        let mut dst_data = dst.data_mut();

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
