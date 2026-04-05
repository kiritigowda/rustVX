//! Gradient operations

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait};
use openvx_image::{Image, ImageFormat};
use crate::utils::{get_pixel_bordered, BorderMode};
use std::f32;

/// Sobel3x3 kernel
pub struct Sobel3x3Kernel;

impl Sobel3x3Kernel {
    pub fn new() -> Self { Sobel3x3Kernel }
}

impl KernelTrait for Sobel3x3Kernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.sobel3x3" }
    fn get_enum(&self) -> VxKernel { VxKernel::Sobel3x3 }
    
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
        let grad_x = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let grad_y = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        sobel3x3(src, grad_x, grad_y)?;
        Ok(())
    }
}

/// Magnitude kernel
pub struct MagnitudeKernel;

impl MagnitudeKernel {
    pub fn new() -> Self { MagnitudeKernel }
}

impl KernelTrait for MagnitudeKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.magnitude" }
    fn get_enum(&self) -> VxKernel { VxKernel::Magnitude }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let grad_x = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let grad_y = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let mag = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        magnitude(grad_x, grad_y, mag)?;
        Ok(())
    }
}

/// Phase kernel
pub struct PhaseKernel;

impl PhaseKernel {
    pub fn new() -> Self { PhaseKernel }
}

impl KernelTrait for PhaseKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.phase" }
    fn get_enum(&self) -> VxKernel { VxKernel::Phase }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let grad_x = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let grad_y = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let phase = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        phase_op(grad_x, grad_y, phase)?;
        Ok(())
    }
}

/// Sobel X kernel coefficients
const SOBEL_X: [[i32; 3]; 3] = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
];

/// Sobel Y kernel coefficients (rotated)
const SOBEL_Y: [[i32; 3]; 3] = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
];

/// Compute Sobel gradients (outputs to S16 format)
pub fn sobel3x3(src: &Image, grad_x: &Image, grad_y: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    
    // Check if output is S16 format
    let is_s16 = grad_x.format() == ImageFormat::S16;
    
    if is_s16 {
        // Output raw i16 values for S16 format
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum_x: i32 = 0;
                let mut sum_y: i32 = 0;
                
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let pixel = src.get_pixel(px, py) as i32;
                        sum_x += pixel * SOBEL_X[ky][kx];
                        sum_y += pixel * SOBEL_Y[ky][kx];
                    }
                }
                
                // Output raw i16 values (no scaling/offset)
                grad_x.set_pixel_i16(x, y, sum_x as i16);
                grad_y.set_pixel_i16(x, y, sum_y as i16);
            }
        }
    } else {
        // Original U8 output with scaling
        let mut gx_data = grad_x.data_mut();
        let mut gy_data = grad_y.data_mut();
        
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum_x: i32 = 0;
                let mut sum_y: i32 = 0;
                
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let pixel = src.get_pixel(px, py) as i32;
                        sum_x += pixel * SOBEL_X[ky][kx];
                        sum_y += pixel * SOBEL_Y[ky][kx];
                    }
                }
                
                // Scale to fit in u8 (divide by 4)
                gx_data[y * width + x] = ((sum_x / 4).max(-128).min(127) + 128) as u8;
                gy_data[y * width + x] = ((sum_y / 4).max(-128).min(127) + 128) as u8;
            }
        }
    }
    
    Ok(())
}

/// Compute gradient magnitude: sqrt(grad_x² + grad_y²)
pub fn magnitude(grad_x: &Image, grad_y: &Image, mag: &Image) -> VxResult<()> {
    let width = grad_x.width();
    let height = grad_x.height();
    
    let mut mag_data = mag.data_mut();
    
    for y in 0..height {
        for x in 0..width {
            // Convert from offset representation back to signed
            let gx = grad_x.get_pixel(x, y) as i32 - 128;
            let gy = grad_y.get_pixel(x, y) as i32 - 128;
            
            let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as i32;
            mag_data[y * width + x] = magnitude.min(255) as u8;
        }
    }
    
    Ok(())
}

/// Compute phase: atan2(grad_y, grad_x) * (180/π)
pub fn phase_op(grad_x: &Image, grad_y: &Image, phase: &Image) -> VxResult<()> {
    let width = grad_x.width();
    let height = grad_y.height();
    
    let mut phase_data = phase.data_mut();
    
    const DEG_PER_RAD: f32 = 180.0 / f32::consts::PI;
    
    for y in 0..height {
        for x in 0..width {
            // Convert from offset representation back to signed
            let gx = grad_x.get_pixel(x, y) as i32 - 128;
            let gy = grad_y.get_pixel(x, y) as i32 - 128;
            
            // Compute phase in degrees [0, 360)
            let phase_deg = (gy as f32).atan2(gx as f32) * DEG_PER_RAD;
            let phase_u8 = ((phase_deg + 360.0) % 360.0) / 360.0 * 255.0;
            
            phase_data[y * width + x] = phase_u8 as u8;
        }
    }
    
    Ok(())
}

/// Combined Sobel with magnitude and phase output
pub fn sobel3x3_full(src: &Image) -> VxResult<(Vec<i16>, Vec<i16>, Vec<f32>)> {
    let width = src.width();
    let height = src.height();
    let size = width * height;
    
    let mut grad_x = vec![0i16; size];
    let mut grad_y = vec![0i16; size];
    let mut magnitude = vec![0f32; size];
    
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
            
            let idx = y * width + x;
            grad_x[idx] = sum_x as i16;
            grad_y[idx] = sum_y as i16;
            magnitude[idx] = ((sum_x * sum_x + sum_y * sum_y) as f32).sqrt();
        }
    }
    
    Ok((grad_x, grad_y, magnitude))
}
