//! Color conversion kernels

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait};
use openvx_image::{Image, ImageFormat};

/// BT.709 coefficients for YUV conversion (from OpenVX conformance tests)
const Y_COEFF_R: i32 = 54;   // 0.2126 * 256 (BT.709)
const Y_COEFF_G: i32 = 183;  // 0.7152 * 256 (BT.709)
const Y_COEFF_B: i32 = 18;   // 0.0722 * 256 (BT.709)

const U_COEFF_R: i32 = -29;  // -0.1146 * 256 (BT.709)
const U_COEFF_G: i32 = -99;  // -0.3854 * 256 (BT.709)
const U_COEFF_B: i32 = 128;  // 0.5 * 256

const V_COEFF_R: i32 = 128;  // 0.5 * 256
const V_COEFF_G: i32 = -116; // -0.4542 * 256 (BT.709)
const V_COEFF_B: i32 = -12;  // -0.0458 * 256 (BT.709)

/// ColorConvert kernel - converts between color spaces
pub struct ColorConvertKernel;

impl ColorConvertKernel {
    pub fn new() -> Self { ColorConvertKernel }
}

impl KernelTrait for ColorConvertKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.color_convert" }
    fn get_enum(&self) -> VxKernel { VxKernel::ColorConvert }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        match (src.format(), dst.format()) {
            (ImageFormat::Rgb, ImageFormat::Gray) => {
                rgb_to_gray(src, dst)?;
            }
            (ImageFormat::Gray, ImageFormat::Rgb) => {
                gray_to_rgb(src, dst)?;
            }
            (ImageFormat::Rgb, ImageFormat::Rgba) => {
                rgb_to_rgba(src, dst)?;
            }
            (ImageFormat::Rgba, ImageFormat::Rgb) => {
                rgba_to_rgb(src, dst)?;
            }
            (ImageFormat::Rgb, ImageFormat::NV12) => {
                rgb_to_nv12(src, dst)?;
            }
            (ImageFormat::NV12, ImageFormat::Rgb) => {
                nv12_to_rgb(src, dst)?;
            }
            (ImageFormat::NV12, ImageFormat::Rgba) => {
                nv12_to_rgba(src, dst)?;
            }
            (ImageFormat::Rgb, ImageFormat::IYUV) => {
                rgb_to_iyuv(src, dst)?;
            }
            (ImageFormat::IYUV, ImageFormat::Rgb) => {
                iyuv_to_rgb(src, dst)?;
            }
            (ImageFormat::IYUV, ImageFormat::Rgba) => {
                iyuv_to_rgba(src, dst)?;
            }
            (ImageFormat::Rgb, ImageFormat::YUV4) => {
                rgb_to_yuv4(src, dst)?;
            }
            (ImageFormat::YUV4, ImageFormat::Rgb) => {
                yuv4_to_rgb(src, dst)?;
            }
            (ImageFormat::YUV4, ImageFormat::Rgba) => {
                yuv4_to_rgba(src, dst)?;
            }
            (ImageFormat::IYUV, ImageFormat::NV12) => {
                iyuv_to_nv12(src, dst)?;
            }
            (ImageFormat::NV12, ImageFormat::IYUV) => {
                nv12_to_iyuv(src, dst)?;
            }
            (ImageFormat::Rgba, ImageFormat::NV12) => {
                rgbx_to_nv12(src, dst)?;
            }
            (ImageFormat::Rgba, ImageFormat::IYUV) => {
                rgbx_to_iyuv(src, dst)?;
            }
            (ImageFormat::Rgba, ImageFormat::YUV4) => {
                rgbx_to_yuv4(src, dst)?;
            }
            (ImageFormat::NV12, ImageFormat::YUV4) => {
                nv12_to_yuv4(src, dst)?;
            }
            (ImageFormat::NV21, ImageFormat::YUV4) => {
                nv21_to_yuv4(src, dst)?;
            }
            (ImageFormat::IYUV, ImageFormat::YUV4) => {
                iyuv_to_yuv4(src, dst)?;
            }
            _ => {
                // Same format, just copy
                dst.data_mut().copy_from_slice(src.data().as_ref());
            }
        }
        
        Ok(())
    }
}

/// ChannelExtract kernel
pub struct ChannelExtractKernel;

impl ChannelExtractKernel {
    pub fn new() -> Self { ChannelExtractKernel }
}

impl KernelTrait for ChannelExtractKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.channel_extract" }
    fn get_enum(&self) -> VxKernel { VxKernel::ChannelExtract }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        // Get the channel value from the scalar parameter
        let channel = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<openvx_core::VxCScalar>())
            .and_then(|s| s.get_i32())
            .unwrap_or(0);

        let dst = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

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

        Ok(())
    }
}

/// ChannelCombine kernel
pub struct ChannelCombineKernel;

impl ChannelCombineKernel {
    pub fn new() -> Self { ChannelCombineKernel }
}

impl KernelTrait for ChannelCombineKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.channel_combine" }
    fn get_enum(&self) -> VxKernel { VxKernel::ChannelCombine }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 4 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let r = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let g = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let b = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params.get(3)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        let width = dst.width();
        let height = dst.height();
        let mut dst_data = dst.data_mut();
        
        for y in 0..height {
            for x in 0..width {
                let rv = r.get_pixel(x, y);
                let gv = g.get_pixel(x, y);
                let bv = b.get_pixel(x, y);
                let idx = (y * width + x) * 3;
                dst_data[idx] = rv;
                dst_data[idx + 1] = gv;
                dst_data[idx + 2] = bv;
            }
        }
        
        Ok(())
    }
}

/// RGB to YUV conversion (BT.709)
#[inline]
fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    
    // BT.709 coefficients (from OpenVX conformance test)
    // Y = 0.2126*R + 0.7152*G + 0.0722*B
    // U = -0.1146*R - 0.3854*G + 0.5*B + 128
    // V = 0.5*R - 0.4542*G - 0.0458*B + 128
    // Note: The test adds 0.5f before casting to int for rounding
    // We add 128 before >> 8 to achieve the same effect
    let y = (((54 * r + 183 * g + 18 * b + 128) >> 8)).clamp(0, 255) as u8;
    let u = (((-29 * r - 99 * g + 128 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
    let v = (((128 * r - 116 * g - 12 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
    
    (y, u, v)
}

/// YUV to RGB conversion (BT.709)
/// R = Y + 1.5748*(V-128)
/// G = Y - 0.1873*(U-128) - 0.4681*(V-128)
/// B = Y + 1.8556*(U-128)
#[inline]
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    
    // BT.709 coefficients (from OpenVX conformance test)
    // R = Y + 1.5748*V = Y + (403*V)/256
    // G = Y - 0.1873*U - 0.4681*V = Y - (48*U + 120*V)/256
    // B = Y + 1.8556*U = Y + (475*U)/256
    // Add 128 before >> 8 for round-to-nearest (matching test's + 0.5f)
    let r = y + ((403 * v + 128) >> 8);
    let g = y - ((48 * u + 120 * v + 128) >> 8);
    let b = y + ((475 * u + 128) >> 8);
    
    (r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8)
}

/// RGB to Grayscale using BT.709 coefficients
pub fn rgb_to_gray(src: &Image, dst: &Image) -> VxResult<()> {
    if dst.format() != ImageFormat::Gray {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    
    // BT.709 coefficients: Y = 0.2126*R + 0.7152*G + 0.0722*B
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            // Add 128 for rounding before >> 8 to match test's + 0.5f
            let gray = (((54 * r as u32 + 183 * g as u32 + 18 * b as u32 + 128) >> 8)).clamp(0, 255) as u8;
            dst_data[y * width + x] = gray;
        }
    }
    
    Ok(())
}

/// Grayscale to RGB
pub fn gray_to_rgb(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::Gray || dst.format() != ImageFormat::Rgb {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    
    for y in 0..height {
        for x in 0..width {
            let gray = src.get_pixel(x, y);
            let idx = (y * width + x) * 3;
            dst_data[idx] = gray;
            dst_data[idx + 1] = gray;
            dst_data[idx + 2] = gray;
        }
    }
    
    Ok(())
}

/// RGB to RGBA
pub fn rgb_to_rgba(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::Rgb || dst.format() != ImageFormat::Rgba {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            let idx = (y * width + x) * 4;
            dst_data[idx] = r;
            dst_data[idx + 1] = g;
            dst_data[idx + 2] = b;
            dst_data[idx + 3] = 255; // Alpha
        }
    }
    
    Ok(())
}

/// RGBA to RGB
pub fn rgba_to_rgb(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::Rgba || dst.format() != ImageFormat::Rgb {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    for y in 0..height {
        for x in 0..width {
            let src_idx = (y * width + x) * 4;
            let dst_idx = (y * width + x) * 3;
            dst_data[dst_idx] = src_data[src_idx];
            dst_data[dst_idx + 1] = src_data[src_idx + 1];
            dst_data[dst_idx + 2] = src_data[src_idx + 2];
        }
    }
    
    Ok(())
}

/// RGB to NV12
pub fn rgb_to_nv12(src: &Image, dst: &Image) -> VxResult<()> {
    if dst.format() != ImageFormat::NV12 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();

    // Y plane is full size, UV plane is half height, same width (interleaved)
    let chroma_offset = width.saturating_mul(height);
    
    // First pass: compute Y for all pixels
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            let (y_val, _, _) = rgb_to_yuv(r, g, b);
            dst_data[y * width + x] = y_val;
        }
    }
    
    // Second pass: compute UV for each 2x2 block (4:2:0 subsampling)
    // Reference implementation computes U/V for each pixel, then averages
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            // Compute U and V for each of the 4 pixels in the 2x2 block
            let mut sum_u = 0i32;
            let mut sum_v = 0i32;
            let mut count = 0i32;
            
            for dy in 0..2 {
                for dx in 0..2 {
                    let py = y + dy;
                    let px = x + dx;
                    if py < height && px < width {
                        let (r, g, b) = src.get_rgb(px, py);
                        let (_, u, v) = rgb_to_yuv(r, g, b);
                        sum_u += u as i32;
                        sum_v += v as i32;
                        count += 1;
                    }
                }
            }
            
            let u_val = (sum_u / count) as u8;
            let v_val = (sum_v / count) as u8;
            
            let chroma_y = y / 2;
            // NV12: UV pairs are interleaved. Each pair takes 2 bytes.
            // For column x (0, 2, 4, ...), the UV pair starts at (x/2) * 2
            let uv_x = (x / 2) * 2;
            let uv_idx = chroma_offset + chroma_y * width + uv_x;
            if uv_idx + 1 < dst_data.len() {
                dst_data[uv_idx] = u_val;
                dst_data[uv_idx + 1] = v_val;
            }
        }
    }
    
    Ok(())
}

/// NV12 to RGB
pub fn nv12_to_rgb(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::NV12 || dst.format() != ImageFormat::Rgb {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let chroma_offset = width.saturating_mul(height);
    // NV12 uses full width as stride for UV plane (interleaved)
    let chroma_stride = width;
    
    for y in 0..height {
        for x in 0..width {
            let y_val = src_data[y * width + x];
            // For NV12, UV is at half resolution
            // U is at: chroma_offset + (y/2) * stride + (x/2) * 2
            // V is at: chroma_offset + (y/2) * stride + (x/2) * 2 + 1
            let uv_x = (x / 2) * 2;
            let uv_y = y / 2;
            let uv_idx = chroma_offset + uv_y * chroma_stride + uv_x;
            
            let u = if uv_idx < src_data.len() { 
                src_data[uv_idx]
            } else { 128 };
            let v = if uv_idx + 1 < src_data.len() { 
                src_data[uv_idx + 1]
            } else { 128 };
            
            let (r, g, b) = yuv_to_rgb(y_val, u, v);
            
            let idx = (y * width + x) * 3;
            dst_data[idx] = r;
            dst_data[idx + 1] = g;
            dst_data[idx + 2] = b;
        }
    }
    
    Ok(())
}

/// NV12 to RGBA
pub fn nv12_to_rgba(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::NV12 || dst.format() != ImageFormat::Rgba {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let chroma_offset = width.saturating_mul(height);
    // NV12 uses full width as stride for UV plane (interleaved)
    let chroma_stride = width;
    
    for y in 0..height {
        for x in 0..width {
            let y_val = src_data[y * width + x];
            // For NV12, UV is at half resolution
            // U is at: chroma_offset + (y/2) * stride + (x/2) * 2
            // V is at: chroma_offset + (y/2) * stride + (x/2) * 2 + 1
            let uv_x = (x / 2) * 2;
            let uv_y = y / 2;
            let uv_idx = chroma_offset + uv_y * chroma_stride + uv_x;
            
            let u = if uv_idx < src_data.len() { 
                src_data[uv_idx]
            } else { 128 };
            let v = if uv_idx + 1 < src_data.len() { 
                src_data[uv_idx + 1]
            } else { 128 };
            
            let (r, g, b) = yuv_to_rgb(y_val, u, v);
            
            let idx = (y * width + x) * 4;
            dst_data[idx] = r;
            dst_data[idx + 1] = g;
            dst_data[idx + 2] = b;
            dst_data[idx + 3] = 255;
        }
    }
    
    Ok(())
}

/// RGB to IYUV (I420)
pub fn rgb_to_iyuv(src: &Image, dst: &Image) -> VxResult<()> {
    if dst.format() != ImageFormat::IYUV {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    
    let y_size = width * height;
    let half_w = (width + 1) / 2;
    let half_h = (height + 1) / 2;
    let u_size = half_w * half_h;
    
    // Compute Y plane for all pixels
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            let (y_val, _, _) = rgb_to_yuv(r, g, b);
            dst_data[y * width + x] = y_val;
        }
    }
    
    // Compute U and V planes (subsampled 2x2)
    // Reference implementation computes U/V for each pixel, then averages
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let mut sum_u = 0i32;
            let mut sum_v = 0i32;
            let mut count = 0i32;
            
            for dy in 0..2 {
                for dx in 0..2 {
                    let py = y + dy;
                    let px = x + dx;
                    if py < height && px < width {
                        let (sr, sg, sb) = src.get_rgb(px, py);
                        let (_, u_val, v_val) = rgb_to_yuv(sr, sg, sb);
                        sum_u += u_val as i32;
                        sum_v += v_val as i32;
                        count += 1;
                    }
                }
            }
            
            let u_val = (sum_u / count) as u8;
            let v_val = (sum_v / count) as u8;
            
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
    
    Ok(())
}

/// IYUV to RGB
pub fn iyuv_to_rgb(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::IYUV || dst.format() != ImageFormat::Rgb {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let half_w = (width + 1) / 2;
    let half_h = (height + 1) / 2;
    let u_size = half_w * half_h;
    
    for y in 0..height {
        for x in 0..width {
            let y_val = src_data[y * width + x];
            let uv_y = y / 2;
            let uv_x = x / 2;
            let u_idx = y_size + uv_y * half_w + uv_x;
            let v_idx = y_size + u_size + uv_y * half_w + uv_x;
            
            let u = if u_idx < src_data.len() { src_data[u_idx] } else { 128 };
            let v = if v_idx < src_data.len() { src_data[v_idx] } else { 128 };
            
            let (r, g, b) = yuv_to_rgb(y_val, u, v);
            let dst_idx = (y * width + x) * 3;
            dst_data[dst_idx] = r;
            dst_data[dst_idx + 1] = g;
            dst_data[dst_idx + 2] = b;
        }
    }
    
    Ok(())
}

/// IYUV to RGBA
pub fn iyuv_to_rgba(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::IYUV || dst.format() != ImageFormat::Rgba {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let half_w = (width + 1) / 2;
    let u_size = half_w * ((height + 1) / 2);
    
    for y in 0..height {
        for x in 0..width {
            let y_val = src_data[y * width + x];
            let uv_y = y / 2;
            let uv_x = x / 2;
            let u_idx = y_size + uv_y * half_w + uv_x;
            let v_idx = y_size + u_size + uv_y * half_w + uv_x;
            
            let u = if u_idx < src_data.len() { src_data[u_idx] } else { 128 };
            let v = if v_idx < src_data.len() { src_data[v_idx] } else { 128 };
            
            let (r, g, b) = yuv_to_rgb(y_val, u, v);
            let dst_idx = (y * width + x) * 4;
            dst_data[dst_idx] = r;
            dst_data[dst_idx + 1] = g;
            dst_data[dst_idx + 2] = b;
            dst_data[dst_idx + 3] = 255;
        }
    }
    
    Ok(())
}

/// RGB to YUV4
pub fn rgb_to_yuv4(src: &Image, dst: &Image) -> VxResult<()> {
    if dst.format() != ImageFormat::YUV4 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    
    let y_size = width * height;
    
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            let (y_val, u_val, v_val) = rgb_to_yuv(r, g, b);
            
            let idx = y * width + x;
            dst_data[idx] = y_val;
            dst_data[y_size + idx] = u_val;
            dst_data[2 * y_size + idx] = v_val;
        }
    }
    
    Ok(())
}

/// YUV4 to RGB
pub fn yuv4_to_rgb(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::YUV4 || dst.format() != ImageFormat::Rgb {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let y_val = src_data[idx];
            let u = src_data[y_size + idx];
            let v = src_data[2 * y_size + idx];
            
            let (r, g, b) = yuv_to_rgb(y_val, u, v);
            let dst_idx = (y * width + x) * 3;
            dst_data[dst_idx] = r;
            dst_data[dst_idx + 1] = g;
            dst_data[dst_idx + 2] = b;
        }
    }
    
    Ok(())
}

/// YUV4 to RGBA
pub fn yuv4_to_rgba(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::YUV4 || dst.format() != ImageFormat::Rgba {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let y_val = src_data[idx];
            let u = src_data[y_size + idx];
            let v = src_data[2 * y_size + idx];
            
            let (r, g, b) = yuv_to_rgb(y_val, u, v);
            let dst_idx = (y * width + x) * 4;
            dst_data[dst_idx] = r;
            dst_data[dst_idx + 1] = g;
            dst_data[dst_idx + 2] = b;
            dst_data[dst_idx + 3] = 255;
        }
    }
    
    Ok(())
}

/// IYUV to NV12
pub fn iyuv_to_nv12(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::IYUV || dst.format() != ImageFormat::NV12 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let half_w = (width + 1) / 2;
    let half_h = (height + 1) / 2;
    let u_size = half_w * half_h;
    
    // Copy Y plane
    for y in 0..height {
        for x in 0..width {
            dst_data[y * width + x] = src_data[y * width + x];
        }
    }
    
    // Interleave U and V planes
    for y in 0..half_h {
        for x in 0..half_w {
            let u_idx = y_size + y * half_w + x;
            let v_idx = y_size + u_size + y * half_w + x;
            let uv_idx = y_size + y * width + x * 2;
            
            if uv_idx + 1 < dst_data.len() {
                dst_data[uv_idx] = src_data[u_idx];
                dst_data[uv_idx + 1] = src_data[v_idx];
            }
        }
    }
    
    Ok(())
}

/// NV12 to IYUV
pub fn nv12_to_iyuv(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::NV12 || dst.format() != ImageFormat::IYUV {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let half_w = (width + 1) / 2;
    let half_h = (height + 1) / 2;
    let u_size = half_w * half_h;
    
    // Copy Y plane
    for y in 0..height {
        for x in 0..width {
            dst_data[y * width + x] = src_data[y * width + x];
        }
    }
    
    // De-interleave UV plane
    for y in 0..half_h {
        for x in 0..half_w {
            let uv_idx = y_size + y * width + x * 2;
            let u_idx = y_size + y * half_w + x;
            let v_idx = y_size + u_size + y * half_w + x;
            
            if u_idx < dst_data.len() && uv_idx < src_data.len() {
                dst_data[u_idx] = src_data[uv_idx];
            }
            if v_idx < dst_data.len() && uv_idx + 1 < src_data.len() {
                dst_data[v_idx] = src_data[uv_idx + 1];
            }
        }
    }
    
    Ok(())
}

/// RGBX to NV12
pub fn rgbx_to_nv12(src: &Image, dst: &Image) -> VxResult<()> {
    if dst.format() != ImageFormat::NV12 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();

    // Y plane is full size, UV plane is half height, same width (interleaved)
    let chroma_offset = width.saturating_mul(height);
    
    // First pass: compute Y for all pixels
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let r = src_data[idx];
            let g = src_data[idx + 1];
            let b = src_data[idx + 2];
            let (y_val, _, _) = rgb_to_yuv(r, g, b);
            dst_data[y * width + x] = y_val;
        }
    }
    
    // Second pass: compute UV for each 2x2 block (4:2:0 subsampling)
    // Reference implementation computes U/V for each pixel, then averages
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            // Compute U and V for each of the 4 pixels in the 2x2 block
            let mut sum_u = 0i32;
            let mut sum_v = 0i32;
            let mut count = 0i32;
            
            for dy in 0..2 {
                for dx in 0..2 {
                    let py = y + dy;
                    let px = x + dx;
                    if py < height && px < width {
                        let pidx = (py * width + px) * 4;
                        let r = src_data[pidx];
                        let g = src_data[pidx + 1];
                        let b = src_data[pidx + 2];
                        let (_, u, v) = rgb_to_yuv(r, g, b);
                        sum_u += u as i32;
                        sum_v += v as i32;
                        count += 1;
                    }
                }
            }
            
            let u_val = (sum_u / count) as u8;
            let v_val = (sum_v / count) as u8;
            
            let chroma_y = y / 2;
            // NV12: UV pairs are interleaved. Each pair takes 2 bytes.
            // For column x (0, 2, 4, ...), the UV pair starts at (x/2) * 2
            let uv_x = (x / 2) * 2;
            let uv_idx = chroma_offset + chroma_y * width + uv_x;
            if uv_idx + 1 < dst_data.len() {
                dst_data[uv_idx] = u_val;
                dst_data[uv_idx + 1] = v_val;
            }
        }
    }
    
    Ok(())
}

/// RGBX to IYUV (I420)
pub fn rgbx_to_iyuv(src: &Image, dst: &Image) -> VxResult<()> {
    if dst.format() != ImageFormat::IYUV {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let half_w = (width + 1) / 2;
    let half_h = (height + 1) / 2;
    let u_size = half_w * half_h;
    
    // Compute Y plane for all pixels
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let r = src_data[idx];
            let g = src_data[idx + 1];
            let b = src_data[idx + 2];
            let (y_val, _, _) = rgb_to_yuv(r, g, b);
            dst_data[y * width + x] = y_val;
        }
    }
    
    // Compute U and V planes (subsampled 2x2)
    // Reference implementation computes U/V for each pixel, then averages
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let mut sum_u = 0i32;
            let mut sum_v = 0i32;
            let mut count = 0i32;
            
            for dy in 0..2 {
                for dx in 0..2 {
                    let py = y + dy;
                    let px = x + dx;
                    if py < height && px < width {
                        let pidx = (py * width + px) * 4;
                        let r = src_data[pidx];
                        let g = src_data[pidx + 1];
                        let b = src_data[pidx + 2];
                        let (_, u_val, v_val) = rgb_to_yuv(r, g, b);
                        sum_u += u_val as i32;
                        sum_v += v_val as i32;
                        count += 1;
                    }
                }
            }
            
            let u_val = (sum_u / count) as u8;
            let v_val = (sum_v / count) as u8;
            
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
    
    Ok(())
}

/// RGBX to YUV4
pub fn rgbx_to_yuv4(src: &Image, dst: &Image) -> VxResult<()> {
    if dst.format() != ImageFormat::YUV4 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let r = src_data[idx];
            let g = src_data[idx + 1];
            let b = src_data[idx + 2];
            let (y_val, u_val, v_val) = rgb_to_yuv(r, g, b);
            
            let plane_idx = y * width + x;
            dst_data[plane_idx] = y_val;
            dst_data[y_size + plane_idx] = u_val;
            dst_data[2 * y_size + plane_idx] = v_val;
        }
    }
    
    Ok(())
}

/// NV12 to YUV4
pub fn nv12_to_yuv4(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::NV12 || dst.format() != ImageFormat::YUV4 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let chroma_offset = y_size;
    
    // Copy Y plane and expand UV to full resolution
    for y in 0..height {
        for x in 0..width {
            let y_val = src_data[y * width + x];
            let uv_x = (x / 2) * 2;
            let uv_y = y / 2;
            let uv_idx = chroma_offset + uv_y * width + uv_x;
            
            let u = if uv_idx < src_data.len() { src_data[uv_idx] } else { 128 };
            let v = if uv_idx + 1 < src_data.len() { src_data[uv_idx + 1] } else { 128 };
            
            let plane_idx = y * width + x;
            dst_data[plane_idx] = y_val;
            dst_data[y_size + plane_idx] = u;
            dst_data[2 * y_size + plane_idx] = v;
        }
    }
    
    Ok(())
}

/// NV21 to YUV4
pub fn nv21_to_yuv4(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::NV21 || dst.format() != ImageFormat::YUV4 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let chroma_offset = y_size;
    
    // Copy Y plane and expand VU to full resolution
    for y in 0..height {
        for x in 0..width {
            let y_val = src_data[y * width + x];
            let vu_x = (x / 2) * 2;
            let vu_y = y / 2;
            let vu_idx = chroma_offset + vu_y * width + vu_x;
            
            // NV21: VU order
            let v = if vu_idx < src_data.len() { src_data[vu_idx] } else { 128 };
            let u = if vu_idx + 1 < src_data.len() { src_data[vu_idx + 1] } else { 128 };
            
            let plane_idx = y * width + x;
            dst_data[plane_idx] = y_val;
            dst_data[y_size + plane_idx] = u;
            dst_data[2 * y_size + plane_idx] = v;
        }
    }
    
    Ok(())
}

/// IYUV to YUV4
pub fn iyuv_to_yuv4(src: &Image, dst: &Image) -> VxResult<()> {
    if src.format() != ImageFormat::IYUV || dst.format() != ImageFormat::YUV4 {
        return Err(openvx_core::VxStatus::ErrorInvalidFormat);
    }
    
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    
    let y_size = width * height;
    let half_w = (width + 1) / 2;
    let half_h = (height + 1) / 2;
    let u_size = half_w * half_h;
    
    // Copy Y plane
    for y in 0..height {
        for x in 0..width {
            let plane_idx = y * width + x;
            dst_data[plane_idx] = src_data[plane_idx];
        }
    }
    
    // Expand U and V planes from subsampled to full resolution
    for y in 0..height {
        for x in 0..width {
            let uv_y = y / 2;
            let uv_x = x / 2;
            let u_idx = y_size + uv_y * half_w + uv_x;
            let v_idx = y_size + u_size + uv_y * half_w + uv_x;
            
            let u = if u_idx < src_data.len() { src_data[u_idx] } else { 128 };
            let v = if v_idx < src_data.len() { src_data[v_idx] } else { 128 };
            
            let plane_idx = y * width + x;
            dst_data[y_size + plane_idx] = u;
            dst_data[2 * y_size + plane_idx] = v;
        }
    }
    
    Ok(())
}
