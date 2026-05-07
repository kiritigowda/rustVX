//! Parallel processing implementations using Rayon
//!
//! Provides multi-threaded versions of vision kernels

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use openvx_core::{VxResult, VxStatus};
use openvx_image::Image;

/// Parallel Gaussian 3x3 filter
#[cfg(feature = "parallel")]
pub fn gaussian3x3_parallel(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    if width < 3 || height < 3 {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let src_data = src.data();
    // Use saturating_mul to prevent integer overflow
    let temp_size = width.saturating_mul(height);
    let mut temp_buffer = vec![0u8; temp_size];
    let mut dst_data = dst.data_mut();

    // Horizontal pass - parallel by rows
    temp_buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let row_offset = y * width;

            // Handle left edge
            row[0] = src_data[row_offset];

            // Middle pixels
            for x in 1..width - 1 {
                let p0 = src_data[row_offset + x - 1] as u16;
                let p1 = src_data[row_offset + x] as u16;
                let p2 = src_data[row_offset + x + 1] as u16;
                row[x] = ((p0 + p1 * 2 + p2) / 4) as u8;
            }

            // Handle right edge
            if width > 1 {
                row[width - 1] = src_data[row_offset + width - 1];
            }
        });

    // Vertical pass - parallel by rows
    dst_data
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            if y == 0 || y == height - 1 {
                // Top/bottom edge - just copy
                let row_offset = y * width;
                for x in 0..width {
                    row[x] = temp_buffer[row_offset + x];
                }
            } else {
                // Middle rows
                let prev_row = (y - 1) * width;
                let curr_row = y * width;
                let next_row = (y + 1) * width;

                for x in 0..width {
                    let p0 = temp_buffer[prev_row + x] as u16;
                    let p1 = temp_buffer[curr_row + x] as u16;
                    let p2 = temp_buffer[next_row + x] as u16;
                    row[x] = ((p0 + p1 * 2 + p2) / 4) as u8;
                }
            }
        });

    Ok(())
}

/// Parallel Gaussian 5x5 filter
#[cfg(feature = "parallel")]
pub fn gaussian5x5_parallel(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    if width < 5 || height < 5 {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let src_data = src.data();
    // Use saturating_mul to prevent integer overflow
    let temp_size = width.saturating_mul(height);
    let mut temp_buffer = vec![0u8; temp_size];
    let mut dst_data = dst.data_mut();

    const KERNEL: [i32; 5] = [1, 4, 6, 4, 1];

    // Horizontal pass - parallel by rows
    temp_buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mut sum: i32 = 0;
                let mut weight: i32 = 0;
                for k in 0..5 {
                    let px = x as isize + k as isize - 2;
                    if px >= 0 && px < width as isize {
                        sum += src_data[y * width + px as usize] as i32 * KERNEL[k];
                        weight += KERNEL[k];
                    }
                }
                row[x] = (sum / weight.max(1)).min(255).max(0) as u8;
            }
        });

    // Vertical pass - parallel by rows
    dst_data
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mut sum: i32 = 0;
                let mut weight: i32 = 0;
                for k in 0..5 {
                    let py = y as isize + k as isize - 2;
                    if py >= 0 && py < height as isize {
                        sum += temp_buffer[py as usize * width + x] as i32 * KERNEL[k];
                        weight += KERNEL[k];
                    }
                }
                row[x] = (sum / weight.max(1)).min(255).max(0) as u8;
            }
        });

    Ok(())
}

/// Parallel Box 3x3 filter
#[cfg(feature = "parallel")]
pub fn box3x3_parallel(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    if width < 3 || height < 3 {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    let src_data = src.data();
    // Use saturating_mul to prevent integer overflow
    let temp_size = width.saturating_mul(height);
    let mut temp_buffer = vec![0u8; temp_size];

    // Horizontal pass with parallel rows
    temp_buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            // Initialize sliding window sum
            let mut window_sum = (src_data[y * width] as u32 + src_data[y * width + 1] as u32) * 2
                + src_data[y * width + 2] as u32;

            for x in 1..width - 1 {
                row[x] = (window_sum / 3) as u8;

                if x + 2 < width {
                    window_sum = window_sum + src_data[y * width + x + 2] as u32
                        - src_data[y * width + x - 1] as u32;
                }
            }

            // Edges
            row[0] = ((src_data[y * width] as u16 + src_data[y * width + 1] as u16) / 2) as u8;
            row[width - 1] = ((src_data[y * width + width - 2] as u16
                + src_data[y * width + width - 1] as u16)
                / 2) as u8;
        });

    // Vertical pass with parallel rows
    let mut dst_data = dst.data_mut();
    dst_data
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                // Simple vertical average
                let mut sum = temp_buffer[y * width + x] as u32;
                let mut count = 1u32;

                if y > 0 {
                    sum += temp_buffer[(y - 1) * width + x] as u32;
                    count += 1;
                }
                if y + 1 < height {
                    sum += temp_buffer[(y + 1) * width + x] as u32;
                    count += 1;
                }

                row[x] = (sum / count) as u8;
            }
        });

    Ok(())
}

/// Parallel Sobel gradient computation
#[cfg(feature = "parallel")]
pub fn sobel3x3_parallel(src: &Image, grad_x: &mut [i16], grad_y: &mut [i16]) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    if grad_x.len() < width * height || grad_y.len() < width * height {
        return Err(VxStatus::ErrorInvalidParameters);
    }

    let src_data = src.data();

    // Sobel kernels
    const SOBEL_X: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];

    const SOBEL_Y: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    // Parallel by rows
    grad_x
        .par_chunks_mut(width)
        .zip(grad_y.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (gx_row, gy_row))| {
            if y == 0 || y == height - 1 {
                // Zero edges
                for x in 0..width {
                    gx_row[x] = 0;
                    gy_row[x] = 0;
                }
            } else {
                for x in 0..width {
                    if x == 0 || x == width - 1 {
                        gx_row[x] = 0;
                        gy_row[x] = 0;
                    } else {
                        // Sobel X
                        let mut sum_x: i32 = 0;
                        let mut sum_y: i32 = 0;

                        for ky in 0..3 {
                            for kx in 0..3 {
                                let px = x + kx - 1;
                                let py = y + ky - 1;
                                let pixel = src_data[py * width + px] as i32;
                                sum_x += pixel * SOBEL_X[ky][kx];
                                sum_y += pixel * SOBEL_Y[ky][kx];
                            }
                        }

                        gx_row[x] = sum_x as i16;
                        gy_row[x] = sum_y as i16;
                    }
                }
            }
        });

    Ok(())
}

/// Parallel RGB to Grayscale conversion
#[cfg(feature = "parallel")]
pub fn rgb_to_gray_parallel(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    let src_data = src.data();
    let mut dst_data = dst.data_mut();

    // Parallel by rows
    dst_data
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let r = src_data[idx] as u32;
                let g = src_data[idx + 1] as u32;
                let b = src_data[idx + 2] as u32;
                // BT.709: Y = 0.2126*R + 0.7152*G + 0.0722*B
                row[x] = ((54 * r + 183 * g + 18 * b) / 255) as u8;
            }
        });

    Ok(())
}

/// Parallel pixel-wise addition
#[cfg(feature = "parallel")]
pub fn add_images_parallel(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    let width = src1.width();
    let height = src1.height();

    let src1_data = src1.data();
    let src2_data = src2.data();
    let mut dst_data = dst.data_mut();

    // Parallel by rows with SIMD chunks
    dst_data
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let src1_row = &src1_data[y * width..(y + 1) * width];
            let src2_row = &src2_data[y * width..(y + 1) * width];

            for x in 0..width {
                let sum = src1_row[x] as u16 + src2_row[x] as u16;
                row[x] = sum.min(255) as u8;
            }
        });

    Ok(())
}

/// Parallel pixel-wise subtraction
#[cfg(feature = "parallel")]
pub fn subtract_images_parallel(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    let width = src1.width();
    let height = src1.height();

    let src1_data = src1.data();
    let src2_data = src2.data();
    let mut dst_data = dst.data_mut();

    // Parallel by rows
    dst_data
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let src1_row = &src1_data[y * width..(y + 1) * width];
            let src2_row = &src2_data[y * width..(y + 1) * width];

            for x in 0..width {
                let diff = src1_row[x] as i16 - src2_row[x] as i16;
                row[x] = diff.max(0).min(255) as u8;
            }
        });

    Ok(())
}

/// Auto-detect best parallel implementation for Gaussian 3x3
pub fn gaussian3x3_auto_parallel(src: &Image, dst: &Image) -> VxResult<()> {
    #[cfg(feature = "parallel")]
    {
        gaussian3x3_parallel(src, dst)
    }
    #[cfg(not(feature = "parallel"))]
    {
        crate::filter::gaussian3x3(src, dst)
    }
}

/// Auto-detect best parallel implementation for Gaussian 5x5
pub fn gaussian5x5_auto_parallel(src: &Image, dst: &Image) -> VxResult<()> {
    #[cfg(feature = "parallel")]
    {
        gaussian5x5_parallel(src, dst)
    }
    #[cfg(not(feature = "parallel"))]
    {
        crate::filter::gaussian5x5(src, dst)
    }
}

/// Auto-detect best parallel implementation for RGB to Gray
pub fn rgb_to_gray_auto_parallel(src: &Image, dst: &Image) -> VxResult<()> {
    #[cfg(feature = "parallel")]
    {
        rgb_to_gray_parallel(src, dst)
    }
    #[cfg(not(feature = "parallel"))]
    {
        crate::color::rgb_to_gray(src, dst)
    }
}

/// Auto-detect best parallel implementation for addition
pub fn add_images_auto_parallel(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    #[cfg(feature = "parallel")]
    {
        add_images_parallel(src1, src2, dst)
    }
    #[cfg(not(feature = "parallel"))]
    {
        crate::arithmetic::add(src1, src2, dst)
    }
}

/// Auto-detect best parallel implementation for subtraction
pub fn subtract_images_auto_parallel(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    #[cfg(feature = "parallel")]
    {
        subtract_images_parallel(src1, src2, dst)
    }
    #[cfg(not(feature = "parallel"))]
    {
        crate::arithmetic::subtract(src1, src2, dst)
    }
}
