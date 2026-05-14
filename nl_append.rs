

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

/// Median of 9 elements using unrolled insertion sort.
#[inline(always)]
fn median9(mut v: [u8; 9]) -> u8 {
    if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
    if v[2] < v[1] { let t = v[2]; v[2] = v[1]; v[1] = t;
        if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
    }
    if v[3] < v[2] { let t = v[3]; v[3] = v[2]; v[2] = t;
        if v[2] < v[1] { let t = v[2]; v[2] = v[1]; v[1] = t;
            if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
        }
    }
    if v[4] < v[3] { let t = v[4]; v[4] = v[3]; v[3] = t;
        if v[3] < v[2] { let t = v[3]; v[3] = v[2]; v[2] = t;
            if v[2] < v[1] { let t = v[2]; v[2] = v[1]; v[1] = t;
                if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
            }
        }
    }
    v[4]
}
