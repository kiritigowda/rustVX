
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

/// Fast phase computation from S16 gradient images.
/// Uses f32 atan2 (faster than f64 on x86_64) with direct slice access.
pub fn phase_s16_fast(
    gx_data: &[u8],
    gy_data: &[u8],
    phase_data: &mut [u8],
    pixels: usize,
) {
    let scale = 256.0f32 / (std::f32::consts::PI * 2.0);

    for i in 0..pixels {
        let gx = i16::from_le_bytes([gx_data[i * 2], gx_data[i * 2 + 1]]) as f32;
        let gy = i16::from_le_bytes([gy_data[i * 2], gy_data[i * 2 + 1]]) as f32;

        let mut val = gy.atan2(gx) * scale;
        if val < 0.0 {
            val += 256.0;
        }
        let mut ival = (val + 0.5).floor() as i32;
        if ival >= 256 {
            ival -= 256;
        }
        phase_data[i] = ival.clamp(0, 255) as u8;
    }
}

/// Enum for non-linear filter mode
pub enum NonLinearMode {
    Min,
    Max,
    Median,
}

/// Fast 3x3 non-linear filter for all-ones mask with replicate border
pub fn nonlinear_filter_3x3(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    mode: NonLinearMode,
) {
    match mode {
        NonLinearMode::Min => nonlinear_3x3_min(src, dst, w, h),
        NonLinearMode::Max => nonlinear_3x3_max(src, dst, w, h),
        NonLinearMode::Median => nonlinear_3x3_median(src, dst, w, h),
    }
}

fn nonlinear_3x3_min(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    for y in 1..h - 1 {
        let row_m1 = &src[(y - 1) * w..y * w];
        let row_0  = &src[y * w..(y + 1) * w];
        let row_p1 = &src[(y + 1) * w..(y + 2) * w];
        let dst_row = &mut dst[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let mut min_val = row_m1[x - 1];
            min_val = min_val.min(row_m1[x]);
            min_val = min_val.min(row_m1[x + 1]);
            min_val = min_val.min(row_0[x - 1]);
            min_val = min_val.min(row_0[x]);
            min_val = min_val.min(row_0[x + 1]);
            min_val = min_val.min(row_p1[x - 1]);
            min_val = min_val.min(row_p1[x]);
            min_val = min_val.min(row_p1[x + 1]);
            dst_row[x] = min_val;
        }
    }
    let xlast = w - 1; let ylast = h - 1;
    for y in 0..h {
        for x in [0, xlast] {
            if y > 0 && y < ylast && x > 0 && x < xlast { continue; }
            let y0 = if y == 0 { 0 } else { y - 1 };
            let y1 = y;
            let y2 = if y == ylast { ylast } else { y + 1 };
            let x0 = if x == 0 { 0 } else { x - 1 };
            let x1 = x;
            let x2 = if x == xlast { xlast } else { x + 1 };
            let mut min_val = src[y0 * w + x0];
            min_val = min_val.min(src[y0 * w + x1]);
            min_val = min_val.min(src[y0 * w + x2]);
            min_val = min_val.min(src[y1 * w + x0]);
            min_val = min_val.min(src[y1 * w + x1]);
            min_val = min_val.min(src[y1 * w + x2]);
            min_val = min_val.min(src[y2 * w + x0]);
            min_val = min_val.min(src[y2 * w + x1]);
            min_val = min_val.min(src[y2 * w + x2]);
            dst[y * w + x] = min_val;
        }
    }
}

fn nonlinear_3x3_max(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    for y in 1..h - 1 {
        let row_m1 = &src[(y - 1) * w..y * w];
        let row_0  = &src[y * w..(y + 1) * w];
        let row_p1 = &src[(y + 1) * w..(y + 2) * w];
        let dst_row = &mut dst[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let mut max_val = row_m1[x - 1];
            max_val = max_val.max(row_m1[x]);
            max_val = max_val.max(row_m1[x + 1]);
            max_val = max_val.max(row_0[x - 1]);
            max_val = max_val.max(row_0[x]);
            max_val = max_val.max(row_0[x + 1]);
            max_val = max_val.max(row_p1[x - 1]);
            max_val = max_val.max(row_p1[x]);
            max_val = max_val.max(row_p1[x + 1]);
            dst_row[x] = max_val;
        }
    }
    let xlast = w - 1; let ylast = h - 1;
    for y in 0..h {
        for x in [0, xlast] {
            if y > 0 && y < ylast && x > 0 && x < xlast { continue; }
            let y0 = if y == 0 { 0 } else { y - 1 };
            let y1 = y;
            let y2 = if y == ylast { ylast } else { y + 1 };
            let x0 = if x == 0 { 0 } else { x - 1 };
            let x1 = x;
            let x2 = if x == xlast { xlast } else { x + 1 };
            let mut max_val = src[y0 * w + x0];
            max_val = max_val.max(src[y0 * w + x1]);
            max_val = max_val.max(src[y0 * w + x2]);
            max_val = max_val.max(src[y1 * w + x0]);
            max_val = max_val.max(src[y1 * w + x1]);
            max_val = max_val.max(src[y1 * w + x2]);
            max_val = max_val.max(src[y2 * w + x0]);
            max_val = max_val.max(src[y2 * w + x1]);
            max_val = max_val.max(src[y2 * w + x2]);
            dst[y * w + x] = max_val;
        }
    }
}

fn nonlinear_3x3_median(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    for y in 1..h - 1 {
        let row_m1 = &src[(y - 1) * w..y * w];
        let row_0  = &src[y * w..(y + 1) * w];
        let row_p1 = &src[(y + 1) * w..(y + 2) * w];
        let dst_row = &mut dst[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let v = [
                row_m1[x - 1], row_m1[x], row_m1[x + 1],
                row_0[x - 1],  row_0[x],  row_0[x + 1],
                row_p1[x - 1], row_p1[x], row_p1[x + 1],
            ];
            dst_row[x] = median9(v);
        }
    }
    let xlast = w - 1; let ylast = h - 1;
    for y in 0..h {
        for x in [0, xlast] {
            if y > 0 && y < ylast && x > 0 && x < xlast { continue; }
            let y0 = if y == 0 { 0 } else { y - 1 };
            let y1 = y;
            let y2 = if y == ylast { ylast } else { y + 1 };
            let x0 = if x == 0 { 0 } else { x - 1 };
            let x1 = x;
            let x2 = if x == xlast { xlast } else { x + 1 };
            let v = [
                src[y0 * w + x0], src[y0 * w + x1], src[y0 * w + x2],
                src[y1 * w + x0], src[y1 * w + x1], src[y1 * w + x2],
                src[y2 * w + x0], src[y2 * w + x1], src[y2 * w + x2],
            ];
            dst[y * w + x] = median9(v);
        }
    }
}
