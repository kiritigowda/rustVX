//! SIMD-optimized arithmetic operations
//!
//! Uses platform-specific SIMD when available (SSE2/AVX2/NEON)

use openvx_core::{VxResult, VxStatus};
use openvx_image::Image;

/// SIMD-optimized pixel-wise addition with saturation
#[cfg(feature = "simd")]
pub fn add_images_simd(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    let len = width * height;
    
    let src1_data = src1.data();
    let src2_data = src2.data();
    let mut dst_data = dst.data_mut();
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use crate::x86_64_simd;
        x86_64_simd::add_images_sat(
            src1_data.as_ptr(),
            src2_data.as_ptr(),
            dst_data.as_mut_ptr(),
            len
        );
    }
    
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use crate::aarch64_simd;
        aarch64_simd::add_images_sat(
            src1_data.as_ptr(),
            src2_data.as_ptr(),
            dst_data.as_mut_ptr(),
            len
        );
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        crate::simd_utils::scalar::add_images_sat_scalar(
            &src1_data,
            &src2_data,
            &mut dst_data
        );
    }
    
    Ok(())
}

/// SIMD-optimized pixel-wise subtraction with saturation
#[cfg(feature = "simd")]
pub fn subtract_images_simd(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    let len = width * height;
    
    let src1_data = src1.data();
    let src2_data = src2.data();
    let mut dst_data = dst.data_mut();
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use crate::x86_64_simd;
        x86_64_simd::sub_images_sat(
            src1_data.as_ptr(),
            src2_data.as_ptr(),
            dst_data.as_mut_ptr(),
            len
        );
    }
    
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use crate::aarch64_simd;
        aarch64_simd::sub_images_sat(
            src1_data.as_ptr(),
            src2_data.as_ptr(),
            dst_data.as_mut_ptr(),
            len
        );
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        crate::simd_utils::scalar::sub_images_sat_scalar(
            &src1_data,
            &src2_data,
            &mut dst_data
        );
    }
    
    Ok(())
}

/// SIMD-optimized weighted average
#[cfg(feature = "simd")]
pub fn weighted_avg_simd(src1: &Image, src2: &Image, dst: &Image, alpha: u8) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    let len = width * height;
    
    let src1_data = src1.data();
    let src2_data = src2.data();
    let mut dst_data = dst.data_mut();
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use crate::x86_64_simd;
        x86_64_simd::weighted_avg(
            src1_data.as_ptr(),
            src2_data.as_ptr(),
            dst_data.as_mut_ptr(),
            len,
            alpha
        );
    }
    
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use crate::aarch64_simd;
        aarch64_simd::weighted_avg(
            src1_data.as_ptr(),
            src2_data.as_ptr(),
            dst_data.as_mut_ptr(),
            len,
            alpha
        );
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        crate::simd_utils::scalar::weighted_avg_scalar(
            &src1_data,
            &src2_data,
            &mut dst_data,
            alpha
        );
    }
    
    Ok(())
}

/// SIMD-optimized pixel-wise multiplication (with scale factor)
pub fn multiply_images_simd(src1: &Image, src2: &Image, dst: &Image, scale: f32) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    
    let width = src1.width();
    let height = src1.height();
    
    let src1_data = src1.data();
    let src2_data = src2.data();
    let mut dst_data = dst.data_mut();
    
    // Multiplication is harder to vectorize with fixed-point scale
    // For now, use optimized scalar with possible partial SIMD
    
    #[cfg(feature = "simd")]
    {
        // Convert scale to fixed-point for faster computation
        let scale_q8 = (scale * 256.0) as i32;
        
        // Process in SIMD chunks if available
        // (implementation would use 16-bit widening multiply)
    }
    
    // Scalar fallback
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let a = src1_data[idx] as f32;
            let b = src2_data[idx] as f32;
            let product = a * b * scale / 255.0;
            dst_data[idx] = product.max(0.0).min(255.0) as u8;
        }
    }
    
    Ok(())
}

/// Auto-detect and use best available implementation for addition
pub fn add_images_auto(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    #[cfg(feature = "simd")]
    {
        add_images_simd(src1, src2, dst)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::arithmetic::add(src1, src2, dst)
    }
}

/// Auto-detect and use best available implementation for subtraction
pub fn subtract_images_auto(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    #[cfg(feature = "simd")]
    {
        subtract_images_simd(src1, src2, dst)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::arithmetic::subtract(src1, src2, dst)
    }
}

/// Auto-detect and use best available implementation for weighted average
pub fn weighted_avg_auto(src1: &Image, src2: &Image, dst: &Image, alpha: u8) -> VxResult<()> {
    #[cfg(feature = "simd")]
    {
        weighted_avg_simd(src1, src2, dst, alpha)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::arithmetic::weighted(src1, src2, dst, alpha)
    }
}

/// Auto-detect and use best available implementation for multiplication
pub fn multiply_images_auto(src1: &Image, src2: &Image, dst: &Image, scale: f32) -> VxResult<()> {
    #[cfg(feature = "simd")]
    {
        multiply_images_simd(src1, src2, dst, scale)
    }
    #[cfg(not(feature = "simd"))]
    {
        crate::arithmetic::multiply(src1, src2, dst, scale)
    }
}
