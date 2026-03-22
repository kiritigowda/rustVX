//! Arithmetic operations

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait};
use openvx_image::Image;

/// Add kernel - pixel-wise addition with overflow policy
pub struct AddKernel;

impl AddKernel {
    pub fn new() -> Self { AddKernel }
}

impl KernelTrait for AddKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.add" }
    fn get_enum(&self) -> VxKernel { VxKernel::Add }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        add(src1, src2, dst)?;
        Ok(())
    }
}

/// Subtract kernel
pub struct SubtractKernel;

impl SubtractKernel {
    pub fn new() -> Self { SubtractKernel }
}

impl KernelTrait for SubtractKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.subtract" }
    fn get_enum(&self) -> VxKernel { VxKernel::Subtract }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        subtract(src1, src2, dst)?;
        Ok(())
    }
}

/// Multiply kernel - with scale factor support
pub struct MultiplyKernel;

impl MultiplyKernel {
    pub fn new() -> Self { MultiplyKernel }
}

impl KernelTrait for MultiplyKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.multiply" }
    fn get_enum(&self) -> VxKernel { VxKernel::Multiply }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        multiply(src1, src2, dst, 1.0)?;
        Ok(())
    }
}

/// WeightedAverage kernel - (src1 * w1 + src2 * w2) / 256
pub struct WeightedAverageKernel;

impl WeightedAverageKernel {
    pub fn new() -> Self { WeightedAverageKernel }
}

impl KernelTrait for WeightedAverageKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.weighted_average" }
    fn get_enum(&self) -> VxKernel { VxKernel::WeightedAverage }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params.get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params.get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params.get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        
        weighted(src1, src2, dst, 128)?;
        Ok(())
    }
}

/// Pixel-wise addition with saturation
pub fn add(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    
    let mut dst_data = dst.data_mut();
    
    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as u16;
            let b = src2.get_pixel(x, y) as u16;
            let sum = a + b;
            dst_data[y * width + x] = sum.min(255) as u8;
        }
    }
    
    Ok(())
}

/// Pixel-wise subtraction with saturation
pub fn subtract(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    
    let mut dst_data = dst.data_mut();
    
    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as i16;
            let b = src2.get_pixel(x, y) as i16;
            let diff = a - b;
            dst_data[y * width + x] = diff.max(0).min(255) as u8;
        }
    }
    
    Ok(())
}

/// Pixel-wise multiplication with scale factor
pub fn multiply(src1: &Image, src2: &Image, dst: &Image, scale: f32) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    
    let mut dst_data = dst.data_mut();
    
    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as f32;
            let b = src2.get_pixel(x, y) as f32;
            let product = a * b * scale / 255.0;
            dst_data[y * width + x] = product.max(0.0).min(255.0) as u8;
        }
    }
    
    Ok(())
}

/// Weighted average: (src1 * w1 + src2 * w2) / 256
pub fn weighted(src1: &Image, src2: &Image, dst: &Image, alpha: u8) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    let beta = 255 - alpha;
    
    let mut dst_data = dst.data_mut();
    
    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as u32;
            let b = src2.get_pixel(x, y) as u32;
            let result = (a * alpha as u32 + b * beta as u32) / 256;
            dst_data[y * width + x] = result as u8;
        }
    }
    
    Ok(())
}
