//! aarch64 NEON SIMD implementations
//!
//! Uses core::arch intrinsics for ARM NEON optimization

#![cfg(all(feature = "simd", target_arch = "aarch64"))]

use core::arch::aarch64::*;

/// NEON implementation for u8 operations
pub mod neon {
    use super::*;

    /// Add two slices element-wise with saturation
    #[target_feature(enable = "neon")]
    pub unsafe fn add_images_sat_neon(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            let a = vld1q_u8(src1.add(i * 16));
            let b = vld1q_u8(src2.add(i * 16));
            let sum = vqaddq_u8(a, b);
            vst1q_u8(dst.add(i * 16), sum);
        }

        // Handle remainder
        for i in (len - remainder)..len {
            *dst.add(i) = (*src1.add(i)).saturating_add(*src2.add(i));
        }
    }

    /// Subtract two slices element-wise with saturation
    #[target_feature(enable = "neon")]
    pub unsafe fn sub_images_sat_neon(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            let a = vld1q_u8(src1.add(i * 16));
            let b = vld1q_u8(src2.add(i * 16));
            let diff = vqsubq_u8(a, b);
            vst1q_u8(dst.add(i * 16), diff);
        }

        // Handle remainder
        for i in (len - remainder)..len {
            *dst.add(i) = (*src1.add(i)).saturating_sub(*src2.add(i));
        }
    }

    /// Weighted average of two images
    #[target_feature(enable = "neon")]
    pub unsafe fn weighted_avg_neon(
        src1: *const u8,
        src2: *const u8,
        dst: *mut u8,
        len: usize,
        alpha: u8,
    ) {
        let beta = 255 - alpha;

        // Create coefficient vectors
        let alpha_vec = vmovl_u8(vdup_n_u8(alpha));
        let beta_vec = vmovl_u8(vdup_n_u8(beta));

        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            // Load 16 u8 values from each source
            let a = vld1q_u8(src1.add(i * 16));
            let b = vld1q_u8(src2.add(i * 16));

            // Split into low and high halves for u16 operations
            let a_low = vmovl_u8(vget_low_u8(a));
            let a_high = vmovl_u8(vget_high_u8(a));
            let b_low = vmovl_u8(vget_low_u8(b));
            let b_high = vmovl_u8(vget_high_u8(b));

            // Multiply and accumulate: (a * alpha + b * beta) / 256
            let result_low = vshrq_n_u16(
                vaddq_u16(
                    vmulq_n_u16(a_low, alpha as u16),
                    vmulq_n_u16(b_low, beta as u16),
                ),
                8,
            );
            let result_high = vshrq_n_u16(
                vaddq_u16(
                    vmulq_n_u16(a_high, alpha as u16),
                    vmulq_n_u16(b_high, beta as u16),
                ),
                8,
            );

            // Combine back to u8
            let result = vcombine_u8(vmovn_u16(result_low), vmovn_u16(result_high));
            vst1q_u8(dst.add(i * 16), result);
        }

        // Handle remainder
        for i in (len - remainder)..len {
            let a = *src1.add(i) as u32;
            let b = *src2.add(i) as u32;
            *dst.add(i) = ((a * alpha as u32 + b * beta as u32) / 256) as u8;
        }
    }

    /// Horizontal Gaussian 3x3 pass ([1,2,1] kernel)
    #[target_feature(enable = "neon")]
    pub unsafe fn gaussian_h3_neon(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        for y in 0..height {
            let row_offset = y * width;

            // Handle left edge
            *dst.add(row_offset) = *src.add(row_offset);

            // Process middle pixels in chunks of 16
            let mut x = 1;
            while x + 15 < width - 1 {
                // Load 16 pixels from previous, current, and next positions
                let prev = vld1q_u8(src.add(row_offset + x - 1));
                let curr = vld1q_u8(src.add(row_offset + x));
                let next = vld1q_u8(src.add(row_offset + x + 1));

                // Widen to u16 for arithmetic
                let prev_low = vmovl_u8(vget_low_u8(prev));
                let prev_high = vmovl_u8(vget_high_u8(prev));
                let curr_low = vmovl_u8(vget_low_u8(curr));
                let curr_high = vmovl_u8(vget_high_u8(curr));
                let next_low = vmovl_u8(vget_low_u8(next));
                let next_high = vmovl_u8(vget_high_u8(next));

                // result = (prev + curr * 2 + next) / 4
                let sum_low = vaddq_u16(vaddq_u16(prev_low, next_low), vshlq_n_u16(curr_low, 1));
                let sum_high =
                    vaddq_u16(vaddq_u16(prev_high, next_high), vshlq_n_u16(curr_high, 1));

                // Divide by 4
                let result_low = vshrq_n_u16(sum_low, 2);
                let result_high = vshrq_n_u16(sum_high, 2);

                // Narrow back to u8
                let result = vcombine_u8(vmovn_u16(result_low), vmovn_u16(result_high));
                vst1q_u8(dst.add(row_offset + x), result);

                x += 16;
            }

            // Handle remaining pixels
            while x < width - 1 {
                let p0 = *src.add(row_offset + x - 1) as u16;
                let p1 = *src.add(row_offset + x) as u16;
                let p2 = *src.add(row_offset + x + 1) as u16;
                *dst.add(row_offset + x) = ((p0 + p1 * 2 + p2) / 4) as u8;
                x += 1;
            }

            // Handle right edge
            if width > 1 {
                *dst.add(row_offset + width - 1) = *src.add(row_offset + width - 1);
            }
        }
    }

    /// Vertical Gaussian 3x3 pass ([1,2,1] kernel)
    #[target_feature(enable = "neon")]
    pub unsafe fn gaussian_v3_neon(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        // Handle top edge
        for x in 0..width {
            *dst.add(x) = *src.add(x);
        }

        let chunks = width / 16;
        let remainder = width % 16;

        // Process middle rows
        for y in 1..height - 1 {
            let prev_row = (y - 1) * width;
            let curr_row = y * width;
            let next_row = (y + 1) * width;

            // Process in chunks of 16
            for chunk in 0..chunks {
                let offset = chunk * 16;
                let prev = vld1q_u8(src.add(prev_row + offset));
                let curr = vld1q_u8(src.add(curr_row + offset));
                let next = vld1q_u8(src.add(next_row + offset));

                // Widen to u16
                let prev_low = vmovl_u8(vget_low_u8(prev));
                let prev_high = vmovl_u8(vget_high_u8(prev));
                let curr_low = vmovl_u8(vget_low_u8(curr));
                let curr_high = vmovl_u8(vget_high_u8(curr));
                let next_low = vmovl_u8(vget_low_u8(next));
                let next_high = vmovl_u8(vget_high_u8(next));

                // result = (prev + curr * 2 + next) / 4
                let sum_low = vaddq_u16(vaddq_u16(prev_low, next_low), vshlq_n_u16(curr_low, 1));
                let sum_high =
                    vaddq_u16(vaddq_u16(prev_high, next_high), vshlq_n_u16(curr_high, 1));

                // Divide by 4
                let result_low = vshrq_n_u16(sum_low, 2);
                let result_high = vshrq_n_u16(sum_high, 2);

                // Narrow and store
                let result = vcombine_u8(vmovn_u16(result_low), vmovn_u16(result_high));
                vst1q_u8(dst.add(curr_row + offset), result);
            }

            // Handle remainder
            let start = chunks * 16;
            for x in start..width {
                let p0 = *src.add(prev_row + x) as u16;
                let p1 = *src.add(curr_row + x) as u16;
                let p2 = *src.add(next_row + x) as u16;
                *dst.add(curr_row + x) = ((p0 + p1 * 2 + p2) / 4) as u8;
            }
        }

        // Handle bottom edge
        if height > 1 {
            for x in 0..width {
                *dst.add((height - 1) * width + x) = *src.add((height - 1) * width + x);
            }
        }
    }

    /// RGB to Grayscale conversion using BT.709
    /// NEON can process 8 RGB pixels at a time efficiently
    #[target_feature(enable = "neon")]
    pub unsafe fn rgb_to_gray_neon(src: *const u8, dst: *mut u8, num_pixels: usize) {
        // BT.709 coefficients: Y = 0.2126*R + 0.7152*G + 0.0722*B
        // Using Q8 fixed-point: Y = (54*R + 183*G + 18*B) / 255

        // Process 8 pixels at a time
        let chunks = num_pixels / 8;

        for i in 0..chunks {
            let offset = i * 24; // 8 pixels * 3 bytes

            // Load 24 bytes (8 RGB pixels)
            // NEON doesn't have a nice 3-byte interleaved load, so we'll:
            // Load data and use table lookup or shuffle

            // Alternative: use scalar for now (complex shuffle pattern)
            // In production, would use vld3_u8 for interleaved loads
            for j in 0..8 {
                let r = *src.add(offset + j * 3) as u32;
                let g = *src.add(offset + j * 3 + 1) as u32;
                let b = *src.add(offset + j * 3 + 2) as u32;
                *dst.add(i * 8 + j) = ((54 * r + 183 * g + 18 * b) / 255) as u8;
            }
        }

        // Handle remaining pixels
        let start = chunks * 8;
        for i in start..num_pixels {
            let r = *src.add(i * 3) as u32;
            let g = *src.add(i * 3 + 1) as u32;
            let b = *src.add(i * 3 + 2) as u32;
            *dst.add(i) = ((54 * r + 183 * g + 18 * b) / 255) as u8;
        }
    }

    /// Box filter 3x3 horizontal
    #[target_feature(enable = "neon")]
    pub unsafe fn box_h3_neon(src: *const u8, dst: *mut u8, width: usize) {
        let mut x = 1;
        while x + 15 < width - 1 {
            let prev = vld1q_u8(src.add(x - 1));
            let curr = vld1q_u8(src.add(x));
            let next = vld1q_u8(src.add(x + 1));

            // Widen to u16
            let prev_low = vmovl_u8(vget_low_u8(prev));
            let prev_high = vmovl_u8(vget_high_u8(prev));
            let curr_low = vmovl_u8(vget_low_u8(curr));
            let curr_high = vmovl_u8(vget_high_u8(curr));
            let next_low = vmovl_u8(vget_low_u8(next));
            let next_high = vmovl_u8(vget_high_u8(next));

            // Sum: prev + curr + next
            let sum_low = vaddq_u16(vaddq_u16(prev_low, curr_low), next_low);
            let sum_high = vaddq_u16(vaddq_u16(prev_high, curr_high), next_high);

            // Approximate /3 using multiply-shift
            // (sum * 0xAA) >> 9
            let result_low = vshrq_n_u16(vmulq_n_u16(sum_low, 0xAA), 9);
            let result_high = vshrq_n_u16(vmulq_n_u16(sum_high, 0xAA), 9);

            // Narrow and store
            let result = vcombine_u8(vmovn_u16(result_low), vmovn_u16(result_high));
            vst1q_u8(dst.add(x), result);

            x += 16;
        }

        // Handle remaining pixels
        while x < width - 1 {
            let sum = *src.add(x - 1) as u16 + *src.add(x) as u16 + *src.add(x + 1) as u16;
            *dst.add(x) = (sum / 3) as u8;
            x += 1;
        }
    }
}

// Re-export NEON functions
pub use neon::*;

/// Add images with saturation (NEON version)
#[inline]
pub unsafe fn add_images_sat(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
    neon::add_images_sat_neon(src1, src2, dst, len);
}

/// Subtract images with saturation (NEON version)
#[inline]
pub unsafe fn sub_images_sat(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
    neon::sub_images_sat_neon(src1, src2, dst, len);
}

/// Weighted average (NEON version)
#[inline]
pub unsafe fn weighted_avg(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize, alpha: u8) {
    neon::weighted_avg_neon(src1, src2, dst, len, alpha);
}

/// Gaussian horizontal 3x3 (NEON version)
#[inline]
pub unsafe fn gaussian_h3(src: *const u8, dst: *mut u8, width: usize, height: usize) {
    neon::gaussian_h3_neon(src, dst, width, height);
}

/// Gaussian vertical 3x3 (NEON version)
#[inline]
pub unsafe fn gaussian_v3(src: *const u8, dst: *mut u8, width: usize, height: usize) {
    neon::gaussian_v3_neon(src, dst, width, height);
}
