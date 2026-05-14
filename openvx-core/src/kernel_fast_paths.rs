use crate::vxu_impl::{Image, ImageFormat};
use crate::VxStatus;

/// Optimised ChannelCombine for RGB (interleaved R-G-B).
pub fn channel_combine_rgb(
    r: &Image,
    g: &Image,
    b: &Image,
    dst: &mut Image,
) -> Result<(), VxStatus> {
    let width = dst.width();
    let height = dst.height();
    let pixels = width * height;

    let r_data = r.data();
    let g_data = g.data();
    let b_data = b.data();
    let dst_data = dst.data_mut();

    if r_data.len() < pixels || g_data.len() < pixels || b_data.len() < pixels {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    if dst_data.len() < pixels * 3 {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    for y in 0..height {
        let src_base = y * width;
        let dst_base = y * width * 3;
        for x in 0..width {
            let s = src_base + x;
            let d = dst_base + x * 3;
            dst_data[d] = r_data[s];
            dst_data[d + 1] = g_data[s];
            dst_data[d + 2] = b_data[s];
        }
    }
    Ok(())
}

/// Optimised ChannelCombine for RGBX (interleaved R-G-B-X).
pub fn channel_combine_rgbx(
    r: &Image,
    g: &Image,
    b: &Image,
    a: &Image,
    dst: &mut Image,
) -> Result<(), VxStatus> {
    let width = dst.width();
    let height = dst.height();
    let pixels = width * height;

    let r_data = r.data();
    let g_data = g.data();
    let b_data = b.data();
    let a_data = a.data();
    let dst_data = dst.data_mut();

    if r_data.len() < pixels || g_data.len() < pixels || b_data.len() < pixels || a_data.len() < pixels {
        return Err(VxStatus::ErrorInvalidDimension);
    }
    if dst_data.len() < pixels * 4 {
        return Err(VxStatus::ErrorInvalidDimension);
    }

    for y in 0..height {
        let src_base = y * width;
        let dst_base = y * width * 4;
        for x in 0..width {
            let s = src_base + x;
            let d = dst_base + x * 4;
            dst_data[d] = r_data[s];
            dst_data[d + 1] = g_data[s];
            dst_data[d + 2] = b_data[s];
            dst_data[d + 3] = a_data[s];
        }
    }
    Ok(())
}

/// Fast nearest-neighbor scale with precomputed source positions.
pub fn scale_image_nearest(
    src_data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_data: &mut [u8],
    dst_width: usize,
    dst_height: usize,
) {
    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    let mut src_x_pos = vec![0usize; dst_width];
    for x in 0..dst_width {
        let src_x = (x as f32 + 0.5) * x_scale - 0.5;
        src_x_pos[x] = (src_x.round() as isize).max(0).min(src_width as isize - 1) as usize;
    }

    for y in 0..dst_height {
        let src_y = (y as f32 + 0.5) * y_scale - 0.5;
        let src_y_clamped = (src_y.round() as isize).max(0).min(src_height as isize - 1) as usize;
        let src_row_base = src_y_clamped * src_width;
        let dst_row_base = y * dst_width;

        for x in 0..dst_width {
            dst_data[dst_row_base + x] = src_data[src_row_base + src_x_pos[x]];
        }
    }
}

/// Fast bilinear scale with precomputed fixed-point weights (Q8).
pub fn scale_image_bilinear(
    src_data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_data: &mut [u8],
    dst_width: usize,
    dst_height: usize,
) {
    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    const Q8: i32 = 8;
    const Q8_SCALE: i32 = 1 << Q8;

    let mut x0s = vec![0usize; dst_width];
    let mut x1s = vec![0usize; dst_width];
    let mut wxs = vec![0i32; dst_width];

    for x in 0..dst_width {
        let src_x = (x as f32 + 0.5) * x_scale - 0.5;
        let x0 = src_x.floor() as i32;
        let fx = src_x - x0 as f32;

        let x0_clamped = x0.max(0).min(src_width as i32 - 1) as usize;
        let x1_clamped = (x0 + 1).max(0).min(src_width as i32 - 1) as usize;
        let wx = (fx * Q8_SCALE as f32) as i32;

        x0s[x] = x0_clamped;
        x1s[x] = x1_clamped;
        wxs[x] = wx;
    }

    for y in 0..dst_height {
        let src_y = (y as f32 + 0.5) * y_scale - 0.5;
        let y0 = src_y.floor() as i32;
        let fy = src_y - y0 as f32;
        let wy = (fy * Q8_SCALE as f32) as i32;

        let y0_clamped = y0.max(0).min(src_height as i32 - 1) as usize;
        let y1_clamped = (y0 + 1).max(0).min(src_height as i32 - 1) as usize;

        let row0_base = y0_clamped * src_width;
        let row1_base = y1_clamped * src_width;
        let dst_row_base = y * dst_width;

        let wy_neg = Q8_SCALE - wy;

        for x in 0..dst_width {
            let wx = wxs[x];
            let wx_neg = Q8_SCALE - wx;

            let p00 = src_data[row0_base + x0s[x]] as i32;
            let p10 = src_data[row0_base + x1s[x]] as i32;
            let p01 = src_data[row1_base + x0s[x]] as i32;
            let p11 = src_data[row1_base + x1s[x]] as i32;

            let top = wx_neg * p00 + wx * p10;
            let bot = wx_neg * p01 + wx * p11;
            let val = (wy_neg * top + wy * bot) >> (Q8 + Q8);

            dst_data[dst_row_base + x] = val.min(255).max(0) as u8;
        }
    }
}

/// Optimised 3×3 convolution for U8 images with Undefined border.
pub fn convolve_3x3_u8_undefined(
    src_data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_data: &mut [u8],
    coeff: &[i16; 9],
    shift: u32,
) {
    // --- Inner region (no bounds checks) ---
    for y in 1..src_height - 1 {
        let row_m1 = &src_data[(y - 1) * src_width..y * src_width];
        let row_0  = &src_data[y * src_width..(y + 1) * src_width];
        let row_p1 = &src_data[(y + 1) * src_width..(y + 2) * src_width];
        let dst_row = &mut dst_data[y * src_width..(y + 1) * src_width];

        for x in 1..src_width - 1 {
            let sum = coeff[0] * row_m1[x - 1] as i16
                    + coeff[1] * row_m1[x] as i16
                    + coeff[2] * row_m1[x + 1] as i16
                    + coeff[3] * row_0[x - 1] as i16
                    + coeff[4] * row_0[x] as i16
                    + coeff[5] * row_0[x + 1] as i16
                    + coeff[6] * row_p1[x - 1] as i16
                    + coeff[7] * row_p1[x] as i16
                    + coeff[8] * row_p1[x + 1] as i16;
            let shifted = sum >> shift;
            dst_row[x] = shifted.clamp(0, 255) as u8;
        }
    }

    // --- Edge pixels with replicate border ---
    let xlast = src_width - 1;
    let ylast = src_height - 1;

    for y in 0..src_height {
        for x in [0, xlast] {
            if y > 0 && y < ylast && x > 0 && x < xlast { continue; }

            let y0 = if y == 0 { 0 } else { y - 1 };
            let y1 = y;
            let y2 = if y == ylast { ylast } else { y + 1 };
            let x0 = if x == 0 { 0 } else { x - 1 };
            let x1 = x;
            let x2 = if x == xlast { xlast } else { x + 1 };

            let sum = coeff[0] * src_data[y0 * src_width + x0] as i16
                    + coeff[1] * src_data[y0 * src_width + x1] as i16
                    + coeff[2] * src_data[y0 * src_width + x2] as i16
                    + coeff[3] * src_data[y1 * src_width + x0] as i16
                    + coeff[4] * src_data[y1 * src_width + x1] as i16
                    + coeff[5] * src_data[y1 * src_width + x2] as i16
                    + coeff[6] * src_data[y2 * src_width + x0] as i16
                    + coeff[7] * src_data[y2 * src_width + x1] as i16
                    + coeff[8] * src_data[y2 * src_width + x2] as i16;
            let shifted = sum >> shift;
            dst_data[y * src_width + x] = shifted.clamp(0, 255) as u8;
        }
    }
}