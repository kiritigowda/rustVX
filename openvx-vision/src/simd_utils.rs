//! SIMD Utilities for Vision Kernels
//!
//! Provides platform-optimized SIMD operations for x86_64 (SSE2/AVX2) and aarch64 (NEON)

#![cfg_attr(feature = "simd", allow(unused))]

/// Check if SIMD is available at runtime
#[inline]
pub fn is_simd_available() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("sse2")
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // NEON is mandatory on aarch64
        true
    }
    #[cfg(not(feature = "simd"))]
    {
        false
    }
}

/// Check if AVX2 is available (x86_64 only)
#[inline]
pub fn is_avx2_available() -> bool {
    #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(all(feature = "avx2", target_arch = "x86_64")))]
    {
        false
    }
}

/// SIMD lane width for different operations
pub const SIMD_U8_LANES: usize = 16; // 128-bit processes 16 u8 values
pub const SIMD_U16_LANES: usize = 8; // 128-bit processes 8 u16 values
pub const SIMD_I16_LANES: usize = 8; // 128-bit processes 8 i16 values
pub const SIMD_F32_LANES: usize = 4; // 128-bit processes 4 f32 values

/// AVX2 lane widths (256-bit)
pub const AVX2_U8_LANES: usize = 32; // 256-bit processes 32 u8 values
pub const AVX2_U16_LANES: usize = 16; // 256-bit processes 16 u16 values
pub const AVX2_I16_LANES: usize = 16; // 256-bit processes 16 i16 values
pub const AVX2_F32_LANES: usize = 8; // 256-bit processes 8 f32 values

/// Helper to round up to nearest multiple
#[inline]
pub const fn round_up(n: usize, multiple: usize) -> usize {
    ((n + multiple - 1) / multiple) * multiple
}

/// Helper to get number of complete SIMD chunks
#[inline]
pub const fn simd_chunks(len: usize, lanes: usize) -> usize {
    len / lanes
}

/// Helper to get remaining elements after SIMD chunks
#[inline]
pub const fn simd_remainder(len: usize, lanes: usize) -> usize {
    len % lanes
}

/// Scalar fallback implementations (always available)
pub mod scalar {

    /// Add images with saturation (scalar version)
    pub fn add_images_sat_scalar(src1: &[u8], src2: &[u8], dst: &mut [u8]) {
        for i in 0..src1.len().min(src2.len()).min(dst.len()) {
            let sum = src1[i] as u16 + src2[i] as u16;
            dst[i] = sum.min(255) as u8;
        }
    }

    /// Subtract images with saturation (scalar version)
    pub fn sub_images_sat_scalar(src1: &[u8], src2: &[u8], dst: &mut [u8]) {
        for i in 0..src1.len().min(src2.len()).min(dst.len()) {
            let diff = src1[i] as i16 - src2[i] as i16;
            dst[i] = diff.max(0).min(255) as u8;
        }
    }

    /// Weighted average of images (scalar version)
    pub fn weighted_avg_scalar(src1: &[u8], src2: &[u8], dst: &mut [u8], alpha: u8) {
        let beta = 255 - alpha;
        for i in 0..src1.len().min(src2.len()).min(dst.len()) {
            let a = src1[i] as u32;
            let b = src2[i] as u32;
            dst[i] = ((a * alpha as u32 + b * beta as u32) / 256) as u8;
        }
    }

    /// RGB to grayscale using BT.709 coefficients (scalar version)
    pub fn rgb_to_gray_scalar(src: &[u8], dst: &mut [u8], num_pixels: usize) {
        // BT.709: Y = 0.2126*R + 0.7152*G + 0.0722*B
        // Using fixed-point: Y = (54*R + 183*G + 18*B) / 255
        for i in 0..num_pixels {
            let r = src[i * 3] as u32;
            let g = src[i * 3 + 1] as u32;
            let b = src[i * 3 + 2] as u32;
            dst[i] = ((54 * r + 183 * g + 18 * b) / 255) as u8;
        }
    }

    /// RGB to YUV using BT.601 coefficients (scalar version)
    pub fn rgb_to_yuv_scalar(
        src: &[u8],
        y_plane: &mut [u8],
        u_plane: &mut [u8],
        v_plane: &mut [u8],
        num_pixels: usize,
    ) {
        for i in 0..num_pixels {
            let r = src[i * 3] as i32;
            let g = src[i * 3 + 1] as i32;
            let b = src[i * 3 + 2] as i32;

            // BT.601
            y_plane[i] = (((76 * r + 150 * g + 29 * b) >> 8) + 128).min(255).max(0) as u8;
            u_plane[i] = (((-43 * r - 85 * g + 128 * b) >> 8) + 128).min(255).max(0) as u8;
            v_plane[i] = (((128 * r - 107 * g - 21 * b) >> 8) + 128).min(255).max(0) as u8;
        }
    }

    /// Gaussian 3x3 horizontal pass (scalar version)
    pub fn gaussian_h3_scalar(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
        const KERNEL: [i32; 3] = [1, 2, 1];

        for y in 0..height {
            for x in 0..width {
                let mut sum: i32 = 0;
                let mut weight: i32 = 0;
                for k in 0..3 {
                    let px = x as isize + k as isize - 1;
                    if px >= 0 && px < width as isize {
                        sum += src[y * width + px as usize] as i32 * KERNEL[k];
                        weight += KERNEL[k];
                    }
                }
                dst[y * width + x] = (sum / weight.max(1)).min(255).max(0) as u8;
            }
        }
    }

    /// Gaussian 3x3 vertical pass (scalar version)
    pub fn gaussian_v3_scalar(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
        const KERNEL: [i32; 3] = [1, 2, 1];

        for y in 0..height {
            for x in 0..width {
                let mut sum: i32 = 0;
                let mut weight: i32 = 0;
                for k in 0..3 {
                    let py = y as isize + k as isize - 1;
                    if py >= 0 && py < height as isize {
                        sum += src[py as usize * width + x] as i32 * KERNEL[k];
                        weight += KERNEL[k];
                    }
                }
                dst[y * width + x] = (sum / weight.max(1)).min(255).max(0) as u8;
            }
        }
    }

    /// Sobel X pass (scalar version)
    pub fn sobel_x_scalar(src: &[u8], dst: &mut [i16], width: usize, height: usize) {
        const KERNEL: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];

        for y in 0..height {
            for x in 0..width {
                let mut sum: i32 = 0;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x as isize + kx as isize - 1;
                        let py = y as isize + ky as isize - 1;
                        let pixel =
                            if px >= 0 && px < width as isize && py >= 0 && py < height as isize {
                                src[py as usize * width + px as usize]
                            } else {
                                src[y * width + x] // replicate border
                            };
                        sum += pixel as i32 * KERNEL[ky][kx];
                    }
                }
                dst[y * width + x] = sum as i16;
            }
        }
    }

    /// Sobel Y pass (scalar version)
    pub fn sobel_y_scalar(src: &[u8], dst: &mut [i16], width: usize, height: usize) {
        const KERNEL: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

        for y in 0..height {
            for x in 0..width {
                let mut sum: i32 = 0;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x as isize + kx as isize - 1;
                        let py = y as isize + ky as isize - 1;
                        let pixel =
                            if px >= 0 && px < width as isize && py >= 0 && py < height as isize {
                                src[py as usize * width + px as usize]
                            } else {
                                src[y * width + x] // replicate border
                            };
                        sum += pixel as i32 * KERNEL[ky][kx];
                    }
                }
                dst[y * width + x] = sum as i16;
            }
        }
    }
}

/// Re-export scalar fallbacks
pub use scalar::*;
