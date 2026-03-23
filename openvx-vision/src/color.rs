//! Color conversion kernels

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait};
use openvx_image::{Image, ImageFormat};

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
        let dst = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        let width = src.width();
        let height = src.height();
        let mut dst_data = dst.data_mut();
        
        for y in 0..height {
            for x in 0..width {
                let (r, g, b) = src.get_rgb(x, y);
                let val = match (x + y) % 3 {
                    0 => r,
                    1 => g,
                    _ => b,
                };
                dst_data[y * width + x] = val;
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
            let gray = ((54 * r as u32 + 183 * g as u32 + 18 * b as u32) / 255) as u8;
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

    // Use saturating_mul to prevent integer overflow
    let chroma_offset = width.saturating_mul(height);
    
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = src.get_rgb(x, y);
            // BT.601: Y = 0.299*R + 0.587*G + 0.114*B
            let y_val = (76 * r as u32 + 150 * g as u32 + 29 * b as u32) >> 8;
            dst_data[y * width + x] = y_val as u8;
            
            // U and V are subsampled 2:1
            if y % 2 == 0 && x % 2 == 0 {
                // BT.601:
                // U = -0.169*R - 0.331*G + 0.5*B + 128
                // V = 0.5*R - 0.419*G - 0.081*B + 128
                let (r2, g2, b2) = src.get_rgb((x + 1).min(width - 1), y);
                let (r3, g3, b3) = src.get_rgb(x, (y + 1).min(height - 1));
                let (r4, g4, b4) = src.get_rgb(
                    (x + 1).min(width - 1), 
                    (y + 1).min(height - 1)
                );
                
                let avg_r = (r as u32 + r2 as u32 + r3 as u32 + r4 as u32) / 4;
                let avg_g = (g as u32 + g2 as u32 + g3 as u32 + g4 as u32) / 4;
                let avg_b = (b as u32 + b2 as u32 + b3 as u32 + b4 as u32) / 4;
                
                let u_val = ((-43 * avg_r as i32 - 85 * avg_g as i32 + 128 * avg_b as i32) >> 8) + 128;
                let v_val = ((128 * avg_r as i32 - 107 * avg_g as i32 - 21 * avg_b as i32) >> 8) + 128;
                
                let chroma_x = x / 2;
                let chroma_y = y / 2;
                let uv_idx = chroma_offset + chroma_y * width + chroma_x * 2;
                if uv_idx + 1 < dst_data.len() {
                    dst_data[uv_idx] = u_val as u8;
                    dst_data[uv_idx + 1] = v_val as u8;
                }
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
    
    // Use saturating_mul to prevent integer overflow
    let chroma_offset = width.saturating_mul(height);
    
    for y in 0..height {
        for x in 0..width {
            let y_val = src_data[y * width + x] as i32;
            let uv_x = (x / 2) * 2;
            let uv_y = y / 2;
            let uv_idx = chroma_offset + uv_y * width + uv_x;
            
            let u = if uv_idx < src_data.len() { 
                src_data[uv_idx] as i32 - 128 
            } else { 0 };
            let v = if uv_idx + 1 < src_data.len() { 
                src_data[uv_idx + 1] as i32 - 128 
            } else { 0 };
            
            // BT.601 conversion
            let r = y_val + ((351 * v) >> 8);
            let g = y_val - ((86 * u + 179 * v) >> 8);
            let b = y_val + ((443 * u) >> 8);
            
            let idx = (y * width + x) * 3;
            dst_data[idx] = crate::utils::clamp(r, 0, 255) as u8;
            dst_data[idx + 1] = crate::utils::clamp(g, 0, 255) as u8;
            dst_data[idx + 2] = crate::utils::clamp(b, 0, 255) as u8;
        }
    }
    
    Ok(())
}
