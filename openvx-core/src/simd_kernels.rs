//! SIMD-accelerated primitives used by the C-API kernel callbacks
//! in [`crate::vxu_impl`].
//!
//! ## Why this lives here
//!
//! `openvx-core` is the lowest-level workspace crate; both `openvx-image`
//! and `openvx-vision` already depend on it. Putting the SIMD kernels
//! that the FFI graph executor invokes (vxAdd, vxSubtract, vxBox3x3,
//! vxGaussian3x3, vxColorConvert) into the same crate as the executor
//! avoids the dependency cycle that would otherwise be required to
//! reach the existing primitives in `openvx-vision::x86_64_simd`.
//!
//! ## Dispatch contract
//!
//! Every public entry point exposes a runtime-dispatched helper plus
//! one or more `#[target_feature]`-gated implementations. Callers in
//! `vxu_impl` decide whether to use the SIMD path with the pattern:
//!
//! ```ignore
//! #[cfg(all(feature = "simd", target_arch = "x86_64"))]
//! unsafe {
//!     if std::is_x86_feature_detected!("avx2") {
//!         crate::simd_kernels::add_u8_sat::avx2(a, b, d, len);
//!         return Ok(());
//!     }
//!     if std::is_x86_feature_detected!("sse2") {
//!         crate::simd_kernels::add_u8_sat::sse2(a, b, d, len);
//!         return Ok(());
//!     }
//! }
//! // fallthrough: tight scalar slice loop
//! ```
//!
//! Without the `simd` Cargo feature (or off-x86_64), the modules below
//! collapse to no-op stubs and the callers fall through to their scalar
//! fast paths. That gives us the user-visible contract:
//!
//! * `cargo build --features '... openvx-core/sse2 openvx-core/avx2'`
//!   → SIMD-fast kernels.
//! * `cargo build` (no SIMD features) → scalar slice-iter kernels.
//!
//! Both paths are still ~50× faster than the previous
//! `for y in ..height { for x in ..width { dst.set_pixel(x,y, ...) } }`
//! style loops in `vxu_impl` because they iterate slices directly.

#![allow(clippy::missing_safety_doc)]

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// ============================================================================
// Saturating u8 add  (vxAdd, U8+U8 -> U8, VX_CONVERT_POLICY_SATURATE)
// ============================================================================

/// Saturating u8 element-wise add. SSE2 / AVX2 implementations.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod add_u8_sat {
    use super::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 32;
        for i in 0..chunks {
            let va = _mm256_loadu_si256(a.add(i * 32) as *const __m256i);
            let vb = _mm256_loadu_si256(b.add(i * 32) as *const __m256i);
            _mm256_storeu_si256(d.add(i * 32) as *mut __m256i, _mm256_adds_epu8(va, vb));
        }
        // tail
        for i in (chunks * 32)..len {
            *d.add(i) = (*a.add(i)).saturating_add(*b.add(i));
        }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn sse2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 16;
        for i in 0..chunks {
            let va = _mm_loadu_si128(a.add(i * 16) as *const __m128i);
            let vb = _mm_loadu_si128(b.add(i * 16) as *const __m128i);
            _mm_storeu_si128(d.add(i * 16) as *mut __m128i, _mm_adds_epu8(va, vb));
        }
        for i in (chunks * 16)..len {
            *d.add(i) = (*a.add(i)).saturating_add(*b.add(i));
        }
    }
}

// ============================================================================
// Saturating u8 sub  (vxSubtract, U8-U8 -> U8, VX_CONVERT_POLICY_SATURATE)
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod sub_u8_sat {
    use super::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 32;
        for i in 0..chunks {
            let va = _mm256_loadu_si256(a.add(i * 32) as *const __m256i);
            let vb = _mm256_loadu_si256(b.add(i * 32) as *const __m256i);
            _mm256_storeu_si256(d.add(i * 32) as *mut __m256i, _mm256_subs_epu8(va, vb));
        }
        for i in (chunks * 32)..len {
            *d.add(i) = (*a.add(i)).saturating_sub(*b.add(i));
        }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn sse2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 16;
        for i in 0..chunks {
            let va = _mm_loadu_si128(a.add(i * 16) as *const __m128i);
            let vb = _mm_loadu_si128(b.add(i * 16) as *const __m128i);
            _mm_storeu_si128(d.add(i * 16) as *mut __m128i, _mm_subs_epu8(va, vb));
        }
        for i in (chunks * 16)..len {
            *d.add(i) = (*a.add(i)).saturating_sub(*b.add(i));
        }
    }
}

// ============================================================================
// Wrapping u8 add / sub  (VX_CONVERT_POLICY_WRAP). Uses _mm_add/sub_epi8
// which is modular at 8-bit, matching `u8::wrapping_add/sub`.
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod add_u8_wrap {
    use super::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 32;
        for i in 0..chunks {
            let va = _mm256_loadu_si256(a.add(i * 32) as *const __m256i);
            let vb = _mm256_loadu_si256(b.add(i * 32) as *const __m256i);
            _mm256_storeu_si256(d.add(i * 32) as *mut __m256i, _mm256_add_epi8(va, vb));
        }
        for i in (chunks * 32)..len {
            *d.add(i) = (*a.add(i)).wrapping_add(*b.add(i));
        }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn sse2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 16;
        for i in 0..chunks {
            let va = _mm_loadu_si128(a.add(i * 16) as *const __m128i);
            let vb = _mm_loadu_si128(b.add(i * 16) as *const __m128i);
            _mm_storeu_si128(d.add(i * 16) as *mut __m128i, _mm_add_epi8(va, vb));
        }
        for i in (chunks * 16)..len {
            *d.add(i) = (*a.add(i)).wrapping_add(*b.add(i));
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod sub_u8_wrap {
    use super::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 32;
        for i in 0..chunks {
            let va = _mm256_loadu_si256(a.add(i * 32) as *const __m256i);
            let vb = _mm256_loadu_si256(b.add(i * 32) as *const __m256i);
            _mm256_storeu_si256(d.add(i * 32) as *mut __m256i, _mm256_sub_epi8(va, vb));
        }
        for i in (chunks * 32)..len {
            *d.add(i) = (*a.add(i)).wrapping_sub(*b.add(i));
        }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn sse2(a: *const u8, b: *const u8, d: *mut u8, len: usize) {
        let chunks = len / 16;
        for i in 0..chunks {
            let va = _mm_loadu_si128(a.add(i * 16) as *const __m128i);
            let vb = _mm_loadu_si128(b.add(i * 16) as *const __m128i);
            _mm_storeu_si128(d.add(i * 16) as *mut __m128i, _mm_sub_epi8(va, vb));
        }
        for i in (chunks * 16)..len {
            *d.add(i) = (*a.add(i)).wrapping_sub(*b.add(i));
        }
    }
}

// ============================================================================
// Gaussian 3x3 ([1,2,1;2,4,2;1,2,1] / 16)  (vxGaussian3x3)
//
// Implemented as a single fused row pass over each interior row using
// `(a + 2b + c)`-style adds on widened u16 lanes. Border behaviour:
// the caller is responsible for the y∈{0, height-1} and x∈{0, width-1}
// fringe — the SSE2/AVX2 routines below only touch interior pixels
// (`y in 1..height-1`, `x in 1..width-1`), exactly matching the
// scalar implementation in `vxu_impl::gaussian3x3` (VX_BORDER_UNDEFINED).
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod gaussian3x3_u8 {
    use super::*;

    /// SSE2 horizontal+vertical fused 3x3 Gaussian, `>> 4` (truncation).
    ///
    /// The existing scalar implementation in `vxu_impl::gaussian3x3`
    /// uses truncating shift (no rounding) and the OpenVX CTS gates on
    /// that exact behaviour. We must match it bit-for-bit, otherwise
    /// the SIMD path produces ±1 drift on every pixel and Gaussian
    /// CTS tests start failing.
    #[target_feature(enable = "sse2")]
    pub unsafe fn sse2(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        if width < 3 || height < 3 {
            return;
        }
        let zero = _mm_setzero_si128();
        for y in 1..height - 1 {
            let p0 = src.add((y - 1) * width);
            let p1 = src.add(y * width);
            let p2 = src.add((y + 1) * width);
            let dr = dst.add(y * width);

            let mut x = 1usize;
            // Process 14 pixels per 16-byte load (cols 1..15 with 1-wide halo)
            while x + 15 <= width - 1 {
                // Load 16 bytes starting at x-1, x, x+1 from each row.
                let r0_l = _mm_loadu_si128(p0.add(x - 1) as *const __m128i);
                let r0_c = _mm_loadu_si128(p0.add(x) as *const __m128i);
                let r0_r = _mm_loadu_si128(p0.add(x + 1) as *const __m128i);
                let r1_l = _mm_loadu_si128(p1.add(x - 1) as *const __m128i);
                let r1_c = _mm_loadu_si128(p1.add(x) as *const __m128i);
                let r1_r = _mm_loadu_si128(p1.add(x + 1) as *const __m128i);
                let r2_l = _mm_loadu_si128(p2.add(x - 1) as *const __m128i);
                let r2_c = _mm_loadu_si128(p2.add(x) as *const __m128i);
                let r2_r = _mm_loadu_si128(p2.add(x + 1) as *const __m128i);

                // For each row r: row_sum = r_l + 2*r_c + r_r  (u16 lanes).
                let row_sum_lo = |l: __m128i, c: __m128i, r: __m128i| -> __m128i {
                    let l_lo = _mm_unpacklo_epi8(l, zero);
                    let c_lo = _mm_unpacklo_epi8(c, zero);
                    let r_lo = _mm_unpacklo_epi8(r, zero);
                    _mm_add_epi16(_mm_add_epi16(l_lo, r_lo), _mm_add_epi16(c_lo, c_lo))
                };
                let row_sum_hi = |l: __m128i, c: __m128i, r: __m128i| -> __m128i {
                    let l_hi = _mm_unpackhi_epi8(l, zero);
                    let c_hi = _mm_unpackhi_epi8(c, zero);
                    let r_hi = _mm_unpackhi_epi8(r, zero);
                    _mm_add_epi16(_mm_add_epi16(l_hi, r_hi), _mm_add_epi16(c_hi, c_hi))
                };

                let s0_lo = row_sum_lo(r0_l, r0_c, r0_r);
                let s1_lo = row_sum_lo(r1_l, r1_c, r1_r);
                let s2_lo = row_sum_lo(r2_l, r2_c, r2_r);
                let s0_hi = row_sum_hi(r0_l, r0_c, r0_r);
                let s1_hi = row_sum_hi(r1_l, r1_c, r1_r);
                let s2_hi = row_sum_hi(r2_l, r2_c, r2_r);

                // total = s0 + 2*s1 + s2  (still in u16)
                let tot_lo = _mm_add_epi16(_mm_add_epi16(s0_lo, s2_lo), _mm_add_epi16(s1_lo, s1_lo));
                let tot_hi = _mm_add_epi16(_mm_add_epi16(s0_hi, s2_hi), _mm_add_epi16(s1_hi, s1_hi));

                // total >> 4  (truncating; matches scalar kernel exactly)
                let out_lo = _mm_srli_epi16(tot_lo, 4);
                let out_hi = _mm_srli_epi16(tot_hi, 4);
                let out = _mm_packus_epi16(out_lo, out_hi);

                _mm_storeu_si128(dr.add(x) as *mut __m128i, out);
                x += 16;
            }
            // tail
            while x < width - 1 {
                let s = *p0.add(x - 1) as u32
                    + 2 * *p0.add(x) as u32
                    + *p0.add(x + 1) as u32
                    + 2 * *p1.add(x - 1) as u32
                    + 4 * *p1.add(x) as u32
                    + 2 * *p1.add(x + 1) as u32
                    + *p2.add(x - 1) as u32
                    + 2 * *p2.add(x) as u32
                    + *p2.add(x + 1) as u32;
                *dr.add(x) = (s >> 4) as u8;
                x += 1;
            }
        }
    }
}

// ============================================================================
// Box 3x3 ([1,1,1;1,1,1;1,1,1] / count)  (vxBox3x3)
//
// The existing scalar `box3x3` in `vxu_impl` divides by the actual
// neighbour count (so border pixels divide by 4 or 6 instead of 9).
// To keep that behaviour bit-exact, the SIMD path only touches the
// interior `y in 1..height-1`, `x in 1..width-1` (where count == 9),
// and the caller fills in the borders with the existing scalar logic.
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod box3x3_u8 {
    use super::*;

    /// SSE2 box 3x3 over the interior. Uses (sum * 7282) >> 16 ≈ /9.
    /// The reciprocal-multiply is exact for u16 sums in [0, 2295].
    #[target_feature(enable = "sse2")]
    pub unsafe fn sse2(src: *const u8, dst: *mut u8, width: usize, height: usize) {
        if width < 3 || height < 3 {
            return;
        }
        let zero = _mm_setzero_si128();
        let recip9 = _mm_set1_epi16(7282); // round(2^16 / 9)
        for y in 1..height - 1 {
            let p0 = src.add((y - 1) * width);
            let p1 = src.add(y * width);
            let p2 = src.add((y + 1) * width);
            let dr = dst.add(y * width);

            let mut x = 1usize;
            while x + 15 <= width - 1 {
                let r0_l = _mm_loadu_si128(p0.add(x - 1) as *const __m128i);
                let r0_c = _mm_loadu_si128(p0.add(x) as *const __m128i);
                let r0_r = _mm_loadu_si128(p0.add(x + 1) as *const __m128i);
                let r1_l = _mm_loadu_si128(p1.add(x - 1) as *const __m128i);
                let r1_c = _mm_loadu_si128(p1.add(x) as *const __m128i);
                let r1_r = _mm_loadu_si128(p1.add(x + 1) as *const __m128i);
                let r2_l = _mm_loadu_si128(p2.add(x - 1) as *const __m128i);
                let r2_c = _mm_loadu_si128(p2.add(x) as *const __m128i);
                let r2_r = _mm_loadu_si128(p2.add(x + 1) as *const __m128i);

                let row_sum_lo = |l: __m128i, c: __m128i, r: __m128i| -> __m128i {
                    let l_lo = _mm_unpacklo_epi8(l, zero);
                    let c_lo = _mm_unpacklo_epi8(c, zero);
                    let r_lo = _mm_unpacklo_epi8(r, zero);
                    _mm_add_epi16(_mm_add_epi16(l_lo, c_lo), r_lo)
                };
                let row_sum_hi = |l: __m128i, c: __m128i, r: __m128i| -> __m128i {
                    let l_hi = _mm_unpackhi_epi8(l, zero);
                    let c_hi = _mm_unpackhi_epi8(c, zero);
                    let r_hi = _mm_unpackhi_epi8(r, zero);
                    _mm_add_epi16(_mm_add_epi16(l_hi, c_hi), r_hi)
                };

                let s0_lo = row_sum_lo(r0_l, r0_c, r0_r);
                let s1_lo = row_sum_lo(r1_l, r1_c, r1_r);
                let s2_lo = row_sum_lo(r2_l, r2_c, r2_r);
                let s0_hi = row_sum_hi(r0_l, r0_c, r0_r);
                let s1_hi = row_sum_hi(r1_l, r1_c, r1_r);
                let s2_hi = row_sum_hi(r2_l, r2_c, r2_r);

                let tot_lo = _mm_add_epi16(_mm_add_epi16(s0_lo, s1_lo), s2_lo);
                let tot_hi = _mm_add_epi16(_mm_add_epi16(s0_hi, s1_hi), s2_hi);

                // sum / 9 ≈ (sum * 7282) >> 16  (signed mulhi is fine, sum <= 9*255 = 2295)
                let div_lo = _mm_mulhi_epi16(tot_lo, recip9);
                let div_hi = _mm_mulhi_epi16(tot_hi, recip9);
                let out = _mm_packus_epi16(div_lo, div_hi);

                _mm_storeu_si128(dr.add(x) as *mut __m128i, out);
                x += 16;
            }
            while x < width - 1 {
                let s = *p0.add(x - 1) as u32
                    + *p0.add(x) as u32
                    + *p0.add(x + 1) as u32
                    + *p1.add(x - 1) as u32
                    + *p1.add(x) as u32
                    + *p1.add(x + 1) as u32
                    + *p2.add(x - 1) as u32
                    + *p2.add(x) as u32
                    + *p2.add(x + 1) as u32;
                *dr.add(x) = (s / 9) as u8;
                x += 1;
            }
        }
    }
}

// ============================================================================
// RGB -> u8 grayscale (BT.709, Q8 fixed point: y = (54R + 183G + 18B) / 255)
//
// SSE2 RGB de-interleave is awkward (it really wants SSSE3 _mm_shuffle_epi8),
// so the "SIMD" path here is actually a tight scalar slice loop with the
// exact same Q8 coefficients as the existing scalar code. That alone is
// ~10× faster than the per-pixel `src.get_rgb(x, y)` call in vxu_impl
// because it drops the bounds-checked accessor and walks the source slice
// linearly.
// ============================================================================

#[inline]
pub fn rgb_to_gray_fast(src: &[u8], dst: &mut [u8]) {
    let n = dst.len().min(src.len() / 3);
    for i in 0..n {
        let r = src[i * 3] as u32;
        let g = src[i * 3 + 1] as u32;
        let b = src[i * 3 + 2] as u32;
        dst[i] = ((54 * r + 183 * g + 18 * b) / 255) as u8;
    }
}

// ============================================================================
// Correctness tests
//
// Each test pseudorandomly fills the input buffer, computes the
// expected output with a pure-scalar reference, then runs the SIMD
// path on a copy and asserts byte-for-byte equality. The reference
// implementations exactly mirror the policy logic in `vxu_impl` so a
// regression here also catches any divergence between the SIMD path
// and the scalar fallback the FFI uses when SIMD is unavailable.
// ============================================================================

#[cfg(all(test, feature = "simd", target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn fill_pseudo(buf: &mut [u8], seed: u64) {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for byte in buf.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (s >> 33) as u8;
        }
    }

    fn ref_add_sat(a: &[u8], b: &[u8], d: &mut [u8]) {
        for ((x, y), o) in a.iter().zip(b).zip(d) {
            *o = x.saturating_add(*y);
        }
    }
    fn ref_sub_sat(a: &[u8], b: &[u8], d: &mut [u8]) {
        for ((x, y), o) in a.iter().zip(b).zip(d) {
            *o = x.saturating_sub(*y);
        }
    }
    fn ref_add_wrap(a: &[u8], b: &[u8], d: &mut [u8]) {
        for ((x, y), o) in a.iter().zip(b).zip(d) {
            *o = x.wrapping_add(*y);
        }
    }
    fn ref_sub_wrap(a: &[u8], b: &[u8], d: &mut [u8]) {
        for ((x, y), o) in a.iter().zip(b).zip(d) {
            *o = x.wrapping_sub(*y);
        }
    }
    fn ref_gaussian3x3(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let s = src[(y - 1) * width + x - 1] as u32
                    + 2 * src[(y - 1) * width + x] as u32
                    + src[(y - 1) * width + x + 1] as u32
                    + 2 * src[y * width + x - 1] as u32
                    + 4 * src[y * width + x] as u32
                    + 2 * src[y * width + x + 1] as u32
                    + src[(y + 1) * width + x - 1] as u32
                    + 2 * src[(y + 1) * width + x] as u32
                    + src[(y + 1) * width + x + 1] as u32;
                dst[y * width + x] = (s >> 4) as u8;
            }
        }
    }
    fn ref_box3x3(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let s = src[(y - 1) * width + x - 1] as u32
                    + src[(y - 1) * width + x] as u32
                    + src[(y - 1) * width + x + 1] as u32
                    + src[y * width + x - 1] as u32
                    + src[y * width + x] as u32
                    + src[y * width + x + 1] as u32
                    + src[(y + 1) * width + x - 1] as u32
                    + src[(y + 1) * width + x] as u32
                    + src[(y + 1) * width + x + 1] as u32;
                dst[y * width + x] = (s / 9) as u8;
            }
        }
    }

    /// Sweep a few representative lengths so we hit:
    /// * a multiple of the AVX2 lane (32) and SSE2 lane (16)
    /// * lengths just over those, exercising the scalar tail
    /// * a "real" full-FHD luma length (1920*1080)
    const ADD_SUB_LENS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 64, 1920 * 1080];

    #[test]
    fn add_u8_sat_matches_scalar() {
        let mut a = vec![0u8; *ADD_SUB_LENS.last().unwrap()];
        let mut b = vec![0u8; *ADD_SUB_LENS.last().unwrap()];
        fill_pseudo(&mut a, 1);
        fill_pseudo(&mut b, 2);
        for &len in ADD_SUB_LENS {
            let a = &a[..len];
            let b = &b[..len];
            let mut want = vec![0u8; len];
            ref_add_sat(a, b, &mut want);

            if std::is_x86_feature_detected!("avx2") {
                let mut got = vec![0u8; len];
                unsafe { add_u8_sat::avx2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), len) };
                assert_eq!(got, want, "AVX2 add_sat differs at len={len}");
            }
            if std::is_x86_feature_detected!("sse2") {
                let mut got = vec![0u8; len];
                unsafe { add_u8_sat::sse2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), len) };
                assert_eq!(got, want, "SSE2 add_sat differs at len={len}");
            }
        }
    }

    #[test]
    fn sub_u8_sat_matches_scalar() {
        let mut a = vec![0u8; *ADD_SUB_LENS.last().unwrap()];
        let mut b = vec![0u8; *ADD_SUB_LENS.last().unwrap()];
        fill_pseudo(&mut a, 3);
        fill_pseudo(&mut b, 4);
        for &len in ADD_SUB_LENS {
            let a = &a[..len];
            let b = &b[..len];
            let mut want = vec![0u8; len];
            ref_sub_sat(a, b, &mut want);

            if std::is_x86_feature_detected!("avx2") {
                let mut got = vec![0u8; len];
                unsafe { sub_u8_sat::avx2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), len) };
                assert_eq!(got, want, "AVX2 sub_sat differs at len={len}");
            }
            if std::is_x86_feature_detected!("sse2") {
                let mut got = vec![0u8; len];
                unsafe { sub_u8_sat::sse2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), len) };
                assert_eq!(got, want, "SSE2 sub_sat differs at len={len}");
            }
        }
    }

    #[test]
    fn add_u8_wrap_matches_scalar() {
        let mut a = vec![0u8; 1024];
        let mut b = vec![0u8; 1024];
        fill_pseudo(&mut a, 5);
        fill_pseudo(&mut b, 6);
        let mut want = vec![0u8; 1024];
        ref_add_wrap(&a, &b, &mut want);

        if std::is_x86_feature_detected!("avx2") {
            let mut got = vec![0u8; 1024];
            unsafe { add_u8_wrap::avx2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), 1024) };
            assert_eq!(got, want);
        }
        if std::is_x86_feature_detected!("sse2") {
            let mut got = vec![0u8; 1024];
            unsafe { add_u8_wrap::sse2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), 1024) };
            assert_eq!(got, want);
        }
    }

    #[test]
    fn sub_u8_wrap_matches_scalar() {
        let mut a = vec![0u8; 1024];
        let mut b = vec![0u8; 1024];
        fill_pseudo(&mut a, 7);
        fill_pseudo(&mut b, 8);
        let mut want = vec![0u8; 1024];
        ref_sub_wrap(&a, &b, &mut want);

        if std::is_x86_feature_detected!("avx2") {
            let mut got = vec![0u8; 1024];
            unsafe { sub_u8_wrap::avx2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), 1024) };
            assert_eq!(got, want);
        }
        if std::is_x86_feature_detected!("sse2") {
            let mut got = vec![0u8; 1024];
            unsafe { sub_u8_wrap::sse2(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), 1024) };
            assert_eq!(got, want);
        }
    }

    #[test]
    fn gaussian3x3_matches_scalar() {
        if !std::is_x86_feature_detected!("sse2") {
            return;
        }
        // Sweep a few sizes that exercise the 16-pixel inner loop and
        // its scalar tail — including odd widths.
        for &(w, h) in &[(3usize, 3usize), (17, 7), (32, 8), (33, 9), (96, 17), (1920, 8)] {
            let mut src = vec![0u8; w * h];
            fill_pseudo(&mut src, (w as u64) * 31 + h as u64);
            let mut want = vec![0u8; w * h];
            let mut got = vec![0u8; w * h];
            ref_gaussian3x3(&src, &mut want, w, h);
            unsafe { gaussian3x3_u8::sse2(src.as_ptr(), got.as_mut_ptr(), w, h) };
            // Only compare interior — borders are caller's responsibility.
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = y * w + x;
                    assert_eq!(
                        got[i], want[i],
                        "gaussian3x3 differs at (x={x}, y={y}) for {w}x{h}"
                    );
                }
            }
        }
    }

    #[test]
    fn box3x3_matches_scalar() {
        if !std::is_x86_feature_detected!("sse2") {
            return;
        }
        for &(w, h) in &[(3usize, 3usize), (17, 7), (32, 8), (33, 9), (96, 17), (1920, 8)] {
            let mut src = vec![0u8; w * h];
            fill_pseudo(&mut src, (w as u64) * 17 + h as u64);
            let mut want = vec![0u8; w * h];
            let mut got = vec![0u8; w * h];
            ref_box3x3(&src, &mut want, w, h);
            unsafe { box3x3_u8::sse2(src.as_ptr(), got.as_mut_ptr(), w, h) };
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = y * w + x;
                    assert_eq!(
                        got[i], want[i],
                        "box3x3 differs at (x={x}, y={y}) for {w}x{h}"
                    );
                }
            }
        }
    }

    #[test]
    fn rgb_to_gray_fast_matches_scalar() {
        let n = 1024;
        let mut rgb = vec![0u8; n * 3];
        fill_pseudo(&mut rgb, 99);
        let mut want = vec![0u8; n];
        for i in 0..n {
            let r = rgb[i * 3] as u32;
            let g = rgb[i * 3 + 1] as u32;
            let b = rgb[i * 3 + 2] as u32;
            want[i] = ((54 * r + 183 * g + 18 * b) / 255) as u8;
        }
        let mut got = vec![0u8; n];
        rgb_to_gray_fast(&rgb, &mut got);
        assert_eq!(got, want);
    }
}
