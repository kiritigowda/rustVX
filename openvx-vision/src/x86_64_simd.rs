//! x86_64 SSE2/AVX2 SIMD implementations
//!
//! Uses core::arch intrinsics for x86_64 optimization

#![cfg(all(feature = "simd", target_arch = "x86_64"))]

use core::arch::x86_64::*;

/// SSE2 implementation for u8 operations
pub mod sse2 {
    use super::*;

    /// Add two slices element-wise with saturation
    #[target_feature(enable = "sse2")]
    pub unsafe fn add_images_sat_sse2(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            let a = _mm_loadu_si128(src1.add(i * 16) as *const __m128i);
            let b = _mm_loadu_si128(src2.add(i * 16) as *const __m128i);
            let sum = _mm_adds_epu8(a, b);
            _mm_storeu_si128(dst.add(i * 16) as *mut __m128i, sum);
        }

        // Handle remainder
        for i in (len - remainder)..len {
            *dst.add(i) = (*src1.add(i)).saturating_add(*src2.add(i));
        }
    }

    /// Subtract two slices element-wise with saturation
    #[target_feature(enable = "sse2")]
    pub unsafe fn sub_images_sat_sse2(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            let a = _mm_loadu_si128(src1.add(i * 16) as *const __m128i);
            let b = _mm_loadu_si128(src2.add(i * 16) as *const __m128i);
            let diff = _mm_subs_epu8(a, b);
            _mm_storeu_si128(dst.add(i * 16) as *mut __m128i, diff);
        }

        // Handle remainder
        for i in (len - remainder)..len {
            *dst.add(i) = (*src1.add(i)).saturating_sub(*src2.add(i));
        }
    }

    /// Weighted average of two images
    #[target_feature(enable = "sse2")]
    pub unsafe fn weighted_avg_sse2(
        src1: *const u8,
        src2: *const u8,
        dst: *mut u8,
        len: usize,
        alpha: u8,
    ) {
        let beta = 255 - alpha;
        let alpha_u16 = _mm_set1_epi16(alpha as i16);
        let beta_u16 = _mm_set1_epi16(beta as i16);

        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            // Load 16 u8 values from each source
            let a = _mm_loadu_si128(src1.add(i * 16) as *const __m128i);
            let b = _mm_loadu_si128(src2.add(i * 16) as *const __m128i);

            // Unpack to u16 (widening)
            let a_lo = _mm_unpacklo_epi8(a, _mm_setzero_si128());
            let a_hi = _mm_unpackhi_epi8(a, _mm_setzero_si128());
            let b_lo = _mm_unpacklo_epi8(b, _mm_setzero_si128());
            let b_hi = _mm_unpackhi_epi8(b, _mm_setzero_si128());

            // Multiply and accumulate: (a * alpha + b * beta) / 256
            let result_lo = _mm_srli_epi16(
                _mm_add_epi16(
                    _mm_mullo_epi16(a_lo, alpha_u16),
                    _mm_mullo_epi16(b_lo, beta_u16),
                ),
                8,
            );
            let result_hi = _mm_srli_epi16(
                _mm_add_epi16(
                    _mm_mullo_epi16(a_hi, alpha_u16),
                    _mm_mullo_epi16(b_hi, beta_u16),
                ),
                8,
            );

            // Pack back to u8
            let result = _mm_packus_epi16(result_lo, result_hi);
            _mm_storeu_si128(dst.add(i * 16) as *mut __m128i, result);
        }

        // Handle remainder
        for i in (len - remainder)..len {
            let a = *src1.add(i) as u32;
            let b = *src2.add(i) as u32;
            *dst.add(i) = ((a * alpha as u32 + b * beta as u32) / 256) as u8;
        }
    }

    /// Horizontal Gaussian 3x3 pass ([1,2,1] kernel)
    /// Processes 16 pixels at a time with proper border handling
    #[target_feature(enable = "sse2")]
    pub unsafe fn gaussian_h3_sse2(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        // For each row
        for y in 0..height {
            let row_offset = y * width;

            // Handle left edge (first pixel) specially
            *dst.add(row_offset) = *src.add(row_offset);

            // Process middle pixels in chunks of 16
            let mut x = 1;
            while x + 15 < width - 1 {
                // Load previous, current, and next pixels
                let prev = _mm_loadu_si128(src.add(row_offset + x - 1) as *const __m128i);
                let curr = _mm_loadu_si128(src.add(row_offset + x) as *const __m128i);
                let next = _mm_loadu_si128(src.add(row_offset + x + 1) as *const __m128i);

                // Widen to u16 for arithmetic
                let prev_lo = _mm_unpacklo_epi8(prev, _mm_setzero_si128());
                let prev_hi = _mm_unpackhi_epi8(prev, _mm_setzero_si128());
                let curr_lo = _mm_unpacklo_epi8(curr, _mm_setzero_si128());
                let curr_hi = _mm_unpackhi_epi8(curr, _mm_setzero_si128());
                let next_lo = _mm_unpacklo_epi8(next, _mm_setzero_si128());
                let next_hi = _mm_unpackhi_epi8(next, _mm_setzero_si128());

                // result = (prev * 1 + curr * 2 + next * 1) / 4
                let sum_lo = _mm_add_epi16(
                    _mm_add_epi16(prev_lo, next_lo),
                    _mm_add_epi16(curr_lo, curr_lo),
                );
                let sum_hi = _mm_add_epi16(
                    _mm_add_epi16(prev_hi, next_hi),
                    _mm_add_epi16(curr_hi, curr_hi),
                );

                // Divide by 4 (right shift by 2)
                let result_lo = _mm_srli_epi16(sum_lo, 2);
                let result_hi = _mm_srli_epi16(sum_hi, 2);

                // Pack back to u8
                let result = _mm_packus_epi16(result_lo, result_hi);
                _mm_storeu_si128(dst.add(row_offset + x) as *mut __m128i, result);

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

            // Handle right edge (last pixel) specially
            if width > 1 {
                *dst.add(row_offset + width - 1) = *src.add(row_offset + width - 1);
            }
        }
    }

    /// Vertical Gaussian 3x3 pass ([1,2,1] kernel)
    #[target_feature(enable = "sse2")]
    pub unsafe fn gaussian_v3_sse2(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        // Handle top edge
        for x in 0..width {
            *dst.add(x) = *src.add(x);
        }

        // Process middle rows
        let chunks = width / 16;

        for y in 1..height - 1 {
            let prev_row = (y - 1) * width;
            let curr_row = y * width;
            let next_row = (y + 1) * width;

            // Process in chunks of 16
            for chunk in 0..chunks {
                let offset = chunk * 16;
                let prev = _mm_loadu_si128(src.add(prev_row + offset) as *const __m128i);
                let curr = _mm_loadu_si128(src.add(curr_row + offset) as *const __m128i);
                let next = _mm_loadu_si128(src.add(next_row + offset) as *const __m128i);

                // Widen to u16
                let prev_lo = _mm_unpacklo_epi8(prev, _mm_setzero_si128());
                let prev_hi = _mm_unpackhi_epi8(prev, _mm_setzero_si128());
                let curr_lo = _mm_unpacklo_epi8(curr, _mm_setzero_si128());
                let curr_hi = _mm_unpackhi_epi8(curr, _mm_setzero_si128());
                let next_lo = _mm_unpacklo_epi8(next, _mm_setzero_si128());
                let next_hi = _mm_unpackhi_epi8(next, _mm_setzero_si128());

                // result = (prev + curr * 2 + next) / 4
                let sum_lo = _mm_add_epi16(
                    _mm_add_epi16(prev_lo, next_lo),
                    _mm_add_epi16(curr_lo, curr_lo),
                );
                let sum_hi = _mm_add_epi16(
                    _mm_add_epi16(prev_hi, next_hi),
                    _mm_add_epi16(curr_hi, curr_hi),
                );

                // Divide by 4
                let result_lo = _mm_srli_epi16(sum_lo, 2);
                let result_hi = _mm_srli_epi16(sum_hi, 2);

                // Pack and store
                let result = _mm_packus_epi16(result_lo, result_hi);
                _mm_storeu_si128(dst.add(curr_row + offset) as *mut __m128i, result);
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
    ///
    /// SSE2 is a poor fit for RGB de-interleaving (proper vectorisation
    /// wants SSSE3 `_mm_shuffle_epi8` or AVX2). Until the SSSE3/AVX2
    /// path lands, this stays scalar with Q8 fixed-point coefficients —
    /// `Y = (54*R + 183*G + 18*B) / 255` per BT.709.
    #[target_feature(enable = "sse2")]
    pub unsafe fn rgb_to_gray_sse2(src: *const u8, dst: *mut u8, num_pixels: usize) {
        for i in 0..num_pixels {
            let r = *src.add(i * 3) as u32;
            let g = *src.add(i * 3 + 1) as u32;
            let b = *src.add(i * 3 + 2) as u32;
            *dst.add(i) = ((54 * r + 183 * g + 18 * b) / 255) as u8;
        }
    }

    /// Box filter 3x3 horizontal (moving average)
    /// Box filter 3x3 horizontal pass — writes u16 sums (no division)
    /// dst must be a u16 buffer of length >= width
    #[target_feature(enable = "sse2")]
    pub unsafe fn box_h3_sum_sse2(src: *const u8, dst: *mut u16, width: usize) {
        if width == 0 { return; }
        *dst.add(0) = (*src.add(0) as u16) * 2 + *src.add(1) as u16;
        if width == 1 { return; }
        let mut x = 1;
        while x + 15 < width - 1 {
            let prev = _mm_loadu_si128(src.add(x - 1) as *const __m128i);
            let curr = _mm_loadu_si128(src.add(x) as *const __m128i);
            let next = _mm_loadu_si128(src.add(x + 1) as *const __m128i);
            let prev_lo = _mm_unpacklo_epi8(prev, _mm_setzero_si128());
            let prev_hi = _mm_unpackhi_epi8(prev, _mm_setzero_si128());
            let curr_lo = _mm_unpacklo_epi8(curr, _mm_setzero_si128());
            let curr_hi = _mm_unpackhi_epi8(curr, _mm_setzero_si128());
            let next_lo = _mm_unpacklo_epi8(next, _mm_setzero_si128());
            let next_hi = _mm_unpackhi_epi8(next, _mm_setzero_si128());
            let sum_lo = _mm_add_epi16(_mm_add_epi16(prev_lo, curr_lo), next_lo);
            let sum_hi = _mm_add_epi16(_mm_add_epi16(prev_hi, curr_hi), next_hi);
            _mm_storeu_si128(dst.add(x) as *mut __m128i, sum_lo);
            _mm_storeu_si128(dst.add(x + 8) as *mut __m128i, sum_hi);
            x += 16;
        }
        while x < width - 1 {
            *dst.add(x) = *src.add(x - 1) as u16 + *src.add(x) as u16 + *src.add(x + 1) as u16;
            x += 1;
        }
        *dst.add(width - 1) = *src.add(width - 2) as u16 + *src.add(width - 1) as u16 * 2;
    }

    /// Box filter 3x3 vertical pass — reads u16 horizontal sums, writes u8 result /9
    #[target_feature(enable = "sse2")]
    pub unsafe fn box_v3_sse2(src: *const u16, dst: *mut u8, width: usize, height: usize) {
        if height == 0 || width == 0 { return; }
        let recip9 = _mm_set1_epi16(7282);
        for x in 0..width {
            let sum = *src.add(x) * 2 + *src.add(width + x);
            *dst.add(x) = (sum / 9) as u8;
        }
        if height == 1 { return; }
        for y in 1..height - 1 {
            let prev_row = (y - 1) * width;
            let curr_row = y * width;
            let next_row = (y + 1) * width;
            let mut x = 0;
            while x + 15 < width {
                let prev_lo = _mm_loadu_si128(src.add(prev_row + x) as *const __m128i);
                let prev_hi = _mm_loadu_si128(src.add(prev_row + x + 8) as *const __m128i);
                let curr_lo = _mm_loadu_si128(src.add(curr_row + x) as *const __m128i);
                let curr_hi = _mm_loadu_si128(src.add(curr_row + x + 8) as *const __m128i);
                let next_lo = _mm_loadu_si128(src.add(next_row + x) as *const __m128i);
                let next_hi = _mm_loadu_si128(src.add(next_row + x + 8) as *const __m128i);
                let sum_lo = _mm_add_epi16(_mm_add_epi16(prev_lo, curr_lo), next_lo);
                let sum_hi = _mm_add_epi16(_mm_add_epi16(prev_hi, curr_hi), next_hi);
                let div_lo = _mm_mulhi_epi16(sum_lo, recip9);
                let div_hi = _mm_mulhi_epi16(sum_hi, recip9);
                let result = _mm_packus_epi16(div_lo, div_hi);
                _mm_storeu_si128(dst.add(curr_row + x) as *mut __m128i, result);
                x += 16;
            }
            while x < width {
                let sum = *src.add(prev_row + x) + *src.add(curr_row + x) + *src.add(next_row + x);
                *dst.add(curr_row + x) = (sum / 9) as u8;
                x += 1;
            }
        }
        let last = (height - 1) * width;
        for x in 0..width {
            let sum = *src.add(last - width + x) + *src.add(last + x) * 2;
            *dst.add(last + x) = (sum / 9) as u8;
        }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn box_h3_sse2(src: *const u8, dst: *mut u8, width: usize) {
        // Process middle pixels in chunks
        let mut x = 1;
        while x + 15 < width - 1 {
            let prev = _mm_loadu_si128(src.add(x - 1) as *const __m128i);
            let curr = _mm_loadu_si128(src.add(x) as *const __m128i);
            let next = _mm_loadu_si128(src.add(x + 1) as *const __m128i);

            // Widen to u16
            let prev_lo = _mm_unpacklo_epi8(prev, _mm_setzero_si128());
            let prev_hi = _mm_unpackhi_epi8(prev, _mm_setzero_si128());
            let curr_lo = _mm_unpacklo_epi8(curr, _mm_setzero_si128());
            let curr_hi = _mm_unpackhi_epi8(curr, _mm_setzero_si128());
            let next_lo = _mm_unpacklo_epi8(next, _mm_setzero_si128());
            let next_hi = _mm_unpackhi_epi8(next, _mm_setzero_si128());

            // Average: (prev + curr + next) / 3
            let sum_lo = _mm_add_epi16(_mm_add_epi16(prev_lo, curr_lo), next_lo);
            let sum_hi = _mm_add_epi16(_mm_add_epi16(prev_hi, curr_hi), next_hi);

            // Divide by 3 using multiply by reciprocal: (sum * 0xAAAB) >> 17
            // 0xAAAB is the Q17 fixed-point representation of 1/3
            // Cast u16 to i16 for _mm_set1_epi16
            let recip = 0xAAABu16 as i16;
            let result_lo = _mm_srli_epi16(_mm_mulhi_epu16(sum_lo, _mm_set1_epi16(recip)), 1);
            let result_hi = _mm_srli_epi16(_mm_mulhi_epu16(sum_hi, _mm_set1_epi16(recip)), 1);

            // Pack and store
            let result = _mm_packus_epi16(result_lo, result_hi);
            _mm_storeu_si128(dst.add(x) as *mut __m128i, result);

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

/// AVX2 implementations (256-bit vectors)
pub mod avx2 {
    use super::*;

    /// Add two slices element-wise with saturation
    #[target_feature(enable = "avx2")]
    pub unsafe fn add_images_sat_avx2(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
        let chunks = len / 32;

        for i in 0..chunks {
            let a = _mm256_loadu_si256(src1.add(i * 32) as *const __m256i);
            let b = _mm256_loadu_si256(src2.add(i * 32) as *const __m256i);
            let sum = _mm256_adds_epu8(a, b);
            _mm256_storeu_si256(dst.add(i * 32) as *mut __m256i, sum);
        }

        // Handle remainder with SSE2
        let start = chunks * 32;
        for i in start..len {
            *dst.add(i) = (*src1.add(i)).saturating_add(*src2.add(i));
        }
    }

    /// Subtract two slices element-wise with saturation
    #[target_feature(enable = "avx2")]
    pub unsafe fn sub_images_sat_avx2(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
        let chunks = len / 32;

        for i in 0..chunks {
            let a = _mm256_loadu_si256(src1.add(i * 32) as *const __m256i);
            let b = _mm256_loadu_si256(src2.add(i * 32) as *const __m256i);
            let diff = _mm256_subs_epu8(a, b);
            _mm256_storeu_si256(dst.add(i * 32) as *mut __m256i, diff);
        }

        // Handle remainder
        let start = chunks * 32;
        for i in start..len {
            *dst.add(i) = (*src1.add(i)).saturating_sub(*src2.add(i));
        }
    }

    /// Weighted average of two images
    #[target_feature(enable = "avx2")]
    pub unsafe fn weighted_avg_avx2(
        src1: *const u8,
        src2: *const u8,
        dst: *mut u8,
        len: usize,
        alpha: u8,
    ) {
        let beta = 255 - alpha;
        let alpha_u16 = _mm256_set1_epi16(alpha as i16);
        let beta_u16 = _mm256_set1_epi16(beta as i16);

        let chunks = len / 32;

        for i in 0..chunks {
            // Load 32 u8 values from each source
            let a = _mm256_loadu_si256(src1.add(i * 32) as *const __m256i);
            let b = _mm256_loadu_si256(src2.add(i * 32) as *const __m256i);

            // Unpack to u16 - first 16 elements
            let a_lo = _mm256_unpacklo_epi8(a, _mm256_setzero_si256());
            let a_hi = _mm256_unpackhi_epi8(a, _mm256_setzero_si256());
            let b_lo = _mm256_unpacklo_epi8(b, _mm256_setzero_si256());
            let b_hi = _mm256_unpackhi_epi8(b, _mm256_setzero_si256());

            // Multiply and accumulate
            let result_lo = _mm256_srli_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(a_lo, alpha_u16),
                    _mm256_mullo_epi16(b_lo, beta_u16),
                ),
                8,
            );
            let result_hi = _mm256_srli_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(a_hi, alpha_u16),
                    _mm256_mullo_epi16(b_hi, beta_u16),
                ),
                8,
            );

            // Pack back to u8
            let result = _mm256_packus_epi16(result_lo, result_hi);
            // Fix lane ordering after pack
            let result = _mm256_permute4x64_epi64(result, mm_shuffle(3, 1, 2, 0));
            _mm256_storeu_si256(dst.add(i * 32) as *mut __m256i, result);
        }

        // Handle remainder
        let start = chunks * 32;
        for i in start..len {
            let a = *src1.add(i) as u32;
            let b = *src2.add(i) as u32;
            *dst.add(i) = ((a * alpha as u32 + b * beta as u32) / 256) as u8;
        }
    }

    /// Horizontal Gaussian 3x3 pass with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn gaussian_h3_avx2(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        for y in 0..height {
            let row_offset = y * width;

            // Handle left edge
            *dst.add(row_offset) = *src.add(row_offset);

            // Process middle in chunks of 32
            let mut x = 1;
            while x + 31 < width - 1 {
                // Load 32 pixels from prev, curr, next
                let prev = _mm256_loadu_si256(src.add(row_offset + x - 1) as *const __m256i);
                let curr = _mm256_loadu_si256(src.add(row_offset + x) as *const __m256i);
                let next = _mm256_loadu_si256(src.add(row_offset + x + 1) as *const __m256i);

                // Unpack to u16
                let prev_lo = _mm256_unpacklo_epi8(prev, _mm256_setzero_si256());
                let prev_hi = _mm256_unpackhi_epi8(prev, _mm256_setzero_si256());
                let curr_lo = _mm256_unpacklo_epi8(curr, _mm256_setzero_si256());
                let curr_hi = _mm256_unpackhi_epi8(curr, _mm256_setzero_si256());
                let next_lo = _mm256_unpacklo_epi8(next, _mm256_setzero_si256());
                let next_hi = _mm256_unpackhi_epi8(next, _mm256_setzero_si256());

                // result = (prev + curr * 2 + next) / 4
                let sum_lo = _mm256_add_epi16(
                    _mm256_add_epi16(prev_lo, next_lo),
                    _mm256_add_epi16(curr_lo, curr_lo),
                );
                let sum_hi = _mm256_add_epi16(
                    _mm256_add_epi16(prev_hi, next_hi),
                    _mm256_add_epi16(curr_hi, curr_hi),
                );

                // Divide by 4
                let result_lo = _mm256_srli_epi16(sum_lo, 2);
                let result_hi = _mm256_srli_epi16(sum_hi, 2);

                // Pack back to u8
                let result = _mm256_packus_epi16(result_lo, result_hi);
                let result = _mm256_permute4x64_epi64(result, mm_shuffle(3, 1, 2, 0));
                _mm256_storeu_si256(dst.add(row_offset + x) as *mut __m256i, result);

                x += 32;
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

    /// Vertical Gaussian 3x3 pass with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn gaussian_v3_avx2(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        // Handle top edge
        for x in 0..width {
            *dst.add(x) = *src.add(x);
        }

        let chunks = width / 32;

        // Process middle rows
        for y in 1..height - 1 {
            let prev_row = (y - 1) * width;
            let curr_row = y * width;
            let next_row = (y + 1) * width;

            // Process chunks of 32
            for chunk in 0..chunks {
                let offset = chunk * 32;
                let prev = _mm256_loadu_si256(src.add(prev_row + offset) as *const __m256i);
                let curr = _mm256_loadu_si256(src.add(curr_row + offset) as *const __m256i);
                let next = _mm256_loadu_si256(src.add(next_row + offset) as *const __m256i);

                // Unpack to u16
                let prev_lo = _mm256_unpacklo_epi8(prev, _mm256_setzero_si256());
                let prev_hi = _mm256_unpackhi_epi8(prev, _mm256_setzero_si256());
                let curr_lo = _mm256_unpacklo_epi8(curr, _mm256_setzero_si256());
                let curr_hi = _mm256_unpackhi_epi8(curr, _mm256_setzero_si256());
                let next_lo = _mm256_unpacklo_epi8(next, _mm256_setzero_si256());
                let next_hi = _mm256_unpackhi_epi8(next, _mm256_setzero_si256());

                // result = (prev + curr * 2 + next) / 4
                let sum_lo = _mm256_add_epi16(
                    _mm256_add_epi16(prev_lo, next_lo),
                    _mm256_add_epi16(curr_lo, curr_lo),
                );
                let sum_hi = _mm256_add_epi16(
                    _mm256_add_epi16(prev_hi, next_hi),
                    _mm256_add_epi16(curr_hi, curr_hi),
                );

                let result_lo = _mm256_srli_epi16(sum_lo, 2);
                let result_hi = _mm256_srli_epi16(sum_hi, 2);

                // Pack and store
                let result = _mm256_packus_epi16(result_lo, result_hi);
                let result = _mm256_permute4x64_epi64(result, mm_shuffle(3, 1, 2, 0));
                _mm256_storeu_si256(dst.add(curr_row + offset) as *mut __m256i, result);
            }

            // Handle remainder
            let start = chunks * 32;
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
    /// Box filter 3x3 horizontal pass with AVX2 — writes u16 sums (no division)
    #[target_feature(enable = "avx2")]
    pub unsafe fn box_h3_sum_avx2(src: *const u8, dst: *mut u16, width: usize) {
        if width == 0 { return; }
        *dst.add(0) = (*src.add(0) as u16) * 2 + *src.add(1) as u16;
        if width == 1 { return; }
        let mut x = 1;
        while x + 31 < width - 1 {
            let prev = _mm256_loadu_si256(src.add(x - 1) as *const __m256i);
            let curr = _mm256_loadu_si256(src.add(x) as *const __m256i);
            let next = _mm256_loadu_si256(src.add(x + 1) as *const __m256i);
            let prev_lo = _mm256_unpacklo_epi8(prev, _mm256_setzero_si256());
            let prev_hi = _mm256_unpackhi_epi8(prev, _mm256_setzero_si256());
            let curr_lo = _mm256_unpacklo_epi8(curr, _mm256_setzero_si256());
            let curr_hi = _mm256_unpackhi_epi8(curr, _mm256_setzero_si256());
            let next_lo = _mm256_unpacklo_epi8(next, _mm256_setzero_si256());
            let next_hi = _mm256_unpackhi_epi8(next, _mm256_setzero_si256());
            let sum_lo = _mm256_add_epi16(_mm256_add_epi16(prev_lo, curr_lo), next_lo);
            let sum_hi = _mm256_add_epi16(_mm256_add_epi16(prev_hi, curr_hi), next_hi);
            _mm256_storeu_si256(dst.add(x) as *mut __m256i, sum_lo);
            _mm256_storeu_si256(dst.add(x + 16) as *mut __m256i, sum_hi);
            x += 32;
        }
        while x < width - 1 {
            *dst.add(x) = *src.add(x - 1) as u16 + *src.add(x) as u16 + *src.add(x + 1) as u16;
            x += 1;
        }
        *dst.add(width - 1) = *src.add(width - 2) as u16 + *src.add(width - 1) as u16 * 2;
    }

    /// Box filter 3x3 vertical pass with AVX2 — reads u16 horizontal sums, writes u8 result /9
    #[target_feature(enable = "avx2")]
    pub unsafe fn box_v3_avx2(src: *const u16, dst: *mut u8, width: usize, height: usize) {
        if height == 0 || width == 0 { return; }
        let recip9 = _mm256_set1_epi16(7282);
        for x in 0..width {
            let sum = *src.add(x) * 2 + *src.add(width + x);
            *dst.add(x) = (sum / 9) as u8;
        }
        if height == 1 { return; }
        for y in 1..height - 1 {
            let prev_row = (y - 1) * width;
            let curr_row = y * width;
            let next_row = (y + 1) * width;
            let mut x = 0;
            while x + 31 < width {
                let prev_lo = _mm256_loadu_si256(src.add(prev_row + x) as *const __m256i);
                let prev_hi = _mm256_loadu_si256(src.add(prev_row + x + 16) as *const __m256i);
                let curr_lo = _mm256_loadu_si256(src.add(curr_row + x) as *const __m256i);
                let curr_hi = _mm256_loadu_si256(src.add(curr_row + x + 16) as *const __m256i);
                let next_lo = _mm256_loadu_si256(src.add(next_row + x) as *const __m256i);
                let next_hi = _mm256_loadu_si256(src.add(next_row + x + 16) as *const __m256i);
                let sum_lo = _mm256_add_epi16(_mm256_add_epi16(prev_lo, curr_lo), next_lo);
                let sum_hi = _mm256_add_epi16(_mm256_add_epi16(prev_hi, curr_hi), next_hi);
                let div_lo = _mm256_mulhi_epi16(sum_lo, recip9);
                let div_hi = _mm256_mulhi_epi16(sum_hi, recip9);
                let packed = _mm256_packus_epi16(div_lo, div_hi);
                let result = _mm256_permute4x64_epi64(packed, mm_shuffle(3, 1, 2, 0));
                _mm256_storeu_si256(dst.add(curr_row + x) as *mut __m256i, result);
                x += 32;
            }
            while x < width {
                let sum = *src.add(prev_row + x) + *src.add(curr_row + x) + *src.add(next_row + x);
                *dst.add(curr_row + x) = (sum / 9) as u8;
                x += 1;
            }
        }
        let last = (height - 1) * width;
        for x in 0..width {
            let sum = *src.add(last - width + x) + *src.add(last + x) * 2;
            *dst.add(last + x) = (sum / 9) as u8;
        }
    }

}

// Re-export commonly used functions based on detected features

/// Add images with saturation (auto-selects SSE2/AVX2)
#[inline]
pub unsafe fn add_images_sat(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
    if is_x86_feature_detected!("avx2") {
        avx2::add_images_sat_avx2(src1, src2, dst, len);
    } else if is_x86_feature_detected!("sse2") {
        sse2::add_images_sat_sse2(src1, src2, dst, len);
    } else {
        // Fallback
        for i in 0..len {
            *dst.add(i) = (*src1.add(i)).saturating_add(*src2.add(i));
        }
    }
}

/// Subtract images with saturation (auto-selects SSE2/AVX2)
#[inline]
pub unsafe fn sub_images_sat(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize) {
    if is_x86_feature_detected!("avx2") {
        avx2::sub_images_sat_avx2(src1, src2, dst, len);
    } else if is_x86_feature_detected!("sse2") {
        sse2::sub_images_sat_sse2(src1, src2, dst, len);
    } else {
        for i in 0..len {
            *dst.add(i) = (*src1.add(i)).saturating_sub(*src2.add(i));
        }
    }
}

/// Weighted average (auto-selects SSE2/AVX2)
#[inline]
pub unsafe fn weighted_avg(src1: *const u8, src2: *const u8, dst: *mut u8, len: usize, alpha: u8) {
    if is_x86_feature_detected!("avx2") {
        avx2::weighted_avg_avx2(src1, src2, dst, len, alpha);
    } else if is_x86_feature_detected!("sse2") {
        sse2::weighted_avg_sse2(src1, src2, dst, len, alpha);
    } else {
        let beta = 255 - alpha;
        for i in 0..len {
            let a = *src1.add(i) as u32;
            let b = *src2.add(i) as u32;
            *dst.add(i) = ((a * alpha as u32 + b * beta as u32) / 256) as u8;
        }
    }
}

/// Gaussian horizontal 3x3 (auto-selects SSE2/AVX2)
#[inline]
pub unsafe fn gaussian_h3(src: *const u8, dst: *mut u8, width: usize, height: usize) {
    if is_x86_feature_detected!("avx2") {
        avx2::gaussian_h3_avx2(src, dst, width, height);
    } else if is_x86_feature_detected!("sse2") {
        sse2::gaussian_h3_sse2(src, dst, width, height);
    } else {
        // Scalar fallback
        for y in 0..height {
            let row_offset = y * width;
            *dst.add(row_offset) = *src.add(row_offset);
            for x in 1..width - 1 {
                let p0 = *src.add(row_offset + x - 1) as u16;
                let p1 = *src.add(row_offset + x) as u16;
                let p2 = *src.add(row_offset + x + 1) as u16;
                *dst.add(row_offset + x) = ((p0 + p1 * 2 + p2) / 4) as u8;
            }
            if width > 1 {
                *dst.add(row_offset + width - 1) = *src.add(row_offset + width - 1);
            }
        }
    }
}

/// Gaussian vertical 3x3 (auto-selects SSE2/AVX2)
#[inline]
pub unsafe fn gaussian_v3(src: *const u8, dst: *mut u8, width: usize, height: usize) {
    if is_x86_feature_detected!("avx2") {
        avx2::gaussian_v3_avx2(src, dst, width, height);
    } else if is_x86_feature_detected!("sse2") {
        sse2::gaussian_v3_sse2(src, dst, width, height);
    } else {
        // Scalar fallback
        for x in 0..width {
            *dst.add(x) = *src.add(x);
        }
        for y in 1..height - 1 {
            let prev_row = (y - 1) * width;
            let curr_row = y * width;
            let next_row = (y + 1) * width;
            for x in 0..width {
                let p0 = *src.add(prev_row + x) as u16;
                let p1 = *src.add(curr_row + x) as u16;
                let p2 = *src.add(next_row + x) as u16;
                *dst.add(curr_row + x) = ((p0 + p1 * 2 + p2) / 4) as u8;
            }
        }
        if height > 1 {
            for x in 0..width {
                *dst.add((height - 1) * width + x) = *src.add((height - 1) * width + x);
            }
        }
    }
}

/// Box filter horizontal 3x3 (auto-selects AVX2/SSE2)
/// Writes u16 sums into dst; caller must divide by 9 after vertical pass.
#[inline]
pub unsafe fn box_h3(src: *const u8, dst: *mut u16, width: usize, height: usize) {
    if is_x86_feature_detected!("avx2") {
        for y in 0..height {
            avx2::box_h3_sum_avx2(src.add(y * width), dst.add(y * width), width);
        }
    } else if is_x86_feature_detected!("sse2") {
        for y in 0..height {
            sse2::box_h3_sum_sse2(src.add(y * width), dst.add(y * width), width);
        }
    } else {
        for y in 0..height {
            let row_src = src.add(y * width);
            let row_dst = dst.add(y * width);
            if width == 1 {
                *row_dst.add(0) = *row_src.add(0) as u16 * 3;
            } else {
                *row_dst.add(0) = *row_src.add(0) as u16 * 2 + *row_src.add(1) as u16;
                if width == 2 {
                    *row_dst.add(1) = *row_src.add(0) as u16 + *row_src.add(1) as u16 * 2;
                } else {
                    let mut sum = *row_src.add(0) as u16 + *row_src.add(1) as u16 + *row_src.add(2) as u16;
                    *row_dst.add(1) = sum;
                    for x in 2..width - 1 {
                        sum += *row_src.add(x + 1) as u16 - *row_src.add(x - 2) as u16;
                        *row_dst.add(x) = sum;
                    }
                    *row_dst.add(width - 1) = *row_src.add(width - 2) as u16 + *row_src.add(width - 1) as u16 * 2;
                }
            }
        }
    }
}

/// Box filter vertical 3x3 (auto-selects AVX2/SSE2)
/// Reads u16 horizontal sums, writes u8 result divided by 9.
#[inline]
pub unsafe fn box_v3(src: *const u16, dst: *mut u8, width: usize, height: usize) {
    if is_x86_feature_detected!("avx2") {
        avx2::box_v3_avx2(src, dst, width, height);
    } else if is_x86_feature_detected!("sse2") {
        sse2::box_v3_sse2(src, dst, width, height);
    } else {
        for x in 0..width {
            let sum = *src.add(x) * 2 + *src.add(width + x);
            *dst.add(x) = (sum / 9) as u8;
        }
        for y in 1..height - 1 {
            for x in 0..width {
                let sum = *src.add((y - 1) * width + x) + *src.add(y * width + x) + *src.add((y + 1) * width + x);
                *dst.add(y * width + x) = (sum / 9) as u8;
            }
        }
        let last = (height - 1) * width;
        for x in 0..width {
            let sum = *src.add(last - width + x) + *src.add(last + x) * 2;
            *dst.add(last + x) = (sum / 9) as u8;
        }
    }
}

#[inline]
const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
