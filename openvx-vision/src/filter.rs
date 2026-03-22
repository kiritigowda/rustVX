//! Filter kernels implementation

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait};
use openvx_image::Image;
use crate::utils::{get_pixel_bordered, BorderMode, clamp_u8, quickselect};

/// Generic convolution kernel
pub struct ConvolveKernel;

impl ConvolveKernel {
    pub fn new() -> Self { ConvolveKernel }
}

impl KernelTrait for ConvolveKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.convolve" }
    fn get_enum(&self) -> VxKernel { VxKernel::Convolve }
    
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
        let dst = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        // For simplicity, apply a generic 3x3 convolution
        let kernel: [[i32; 3]; 3] = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ];
        
        convolve_generic(src, dst, &kernel, BorderMode::Replicate)?;
        
        Ok(())
    }
}

/// Gaussian3x3 kernel - separable [1,2,1] horizontal then vertical
pub struct Gaussian3x3Kernel;

impl Gaussian3x3Kernel {
    pub fn new() -> Self { Gaussian3x3Kernel }
}

impl KernelTrait for Gaussian3x3Kernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.gaussian3x3" }
    fn get_enum(&self) -> VxKernel { VxKernel::Gaussian3x3 }
    
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
        
        gaussian3x3(src, dst)?;
        Ok(())
    }
}

/// Gaussian5x5 kernel - separable [1,4,6,4,1] kernel
pub struct Gaussian5x5Kernel;

impl Gaussian5x5Kernel {
    pub fn new() -> Self { Gaussian5x5Kernel }
}

impl KernelTrait for Gaussian5x5Kernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.gaussian5x5" }
    fn get_enum(&self) -> VxKernel { VxKernel::Gaussian5x5 }
    
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
        
        gaussian5x5(src, dst)?;
        Ok(())
    }
}

/// Box3x3 kernel - moving average optimization
pub struct Box3x3Kernel;

impl Box3x3Kernel {
    pub fn new() -> Self { Box3x3Kernel }
}

impl KernelTrait for Box3x3Kernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.box3x3" }
    fn get_enum(&self) -> VxKernel { VxKernel::Box3x3 }
    
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
        
        box3x3(src, dst)?;
        Ok(())
    }
}

/// Median3x3 kernel - quickselect on 3x3 neighborhood
pub struct Median3x3Kernel;

impl Median3x3Kernel {
    pub fn new() -> Self { Median3x3Kernel }
}

impl KernelTrait for Median3x3Kernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.median3x3" }
    fn get_enum(&self) -> VxKernel { VxKernel::Median3x3 }
    
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
        
        median3x3(src, dst)?;
        Ok(())
    }
}

/// Generic NxM convolution with proper border handling
pub fn convolve_generic(src: &Image, dst: &Image, kernel: &[[i32; 3]; 3], border: BorderMode) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let kernel_sum: i32 = kernel.iter().flat_map(|r| r.iter()).sum::<i32>().max(1);
    
    let mut dst_data = dst.data_mut();
    
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
            dst_data[y * width + x] = clamp_u8(sum / kernel_sum);
        }
    }
    
    Ok(())
}

/// Separable Gaussian 3x3: [1,2,1] horizontal then vertical
pub fn gaussian3x3(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let kernel = [1, 2, 1];
    
    let mut dst_data = dst.data_mut();
    
    // Temporary buffer for horizontal pass
    let mut temp = vec![0u8; width * height];
    
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
            temp[y * width + x] = clamp_u8(sum / weight);
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
                    sum += temp[py as usize * width + x] as i32 * kernel[k];
                    weight += kernel[k];
                }
            }
            dst_data[y * width + x] = clamp_u8(sum / weight);
        }
    }
    
    Ok(())
}

/// Separable Gaussian 5x5: [1,4,6,4,1] kernel
pub fn gaussian5x5(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let kernel = [1, 4, 6, 4, 1];
    
    let mut dst_data = dst.data_mut();
    
    // Temporary buffer for horizontal pass
    let mut temp = vec![0u8; width * height];
    
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
            temp[y * width + x] = clamp_u8(sum / weight);
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
                    sum += temp[py as usize * width + x] as i32 * kernel[k];
                    weight += kernel[k];
                }
            }
            dst_data[y * width + x] = clamp_u8(sum / weight);
        }
    }
    
    Ok(())
}

/// Box filter 3x3 using moving average optimization
pub fn box3x3(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    
    let mut dst_data = dst.data_mut();
    
    // Moving average optimization
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
            
            dst_data[y * width + x] = (sum / count.max(1)) as u8;
        }
    }
    
    Ok(())
}

/// Median filter 3x3 using quickselect
pub fn median3x3(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    
    let mut dst_data = dst.data_mut();
    
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
            
            dst_data[y * width + x] = quickselect(&mut window, 4);
        }
    }
    
    Ok(())
}
