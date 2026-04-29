pub fn vxu_harris_corners_impl(
    context: vx_context,
    input: vx_image,
    strength_thresh: vx_scalar,
    min_distance: vx_scalar,
    sensitivity: vx_scalar,
    gradient_size: vx_enum,
    block_size: vx_enum,
    corners: vx_array,
    num_corners: vx_scalar,
) -> vx_status {
    // Validate all required parameters with null checks
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if input.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Read scalar parameters
    let threshold: f32 = if !strength_thresh.is_null() {
        let mut val: f32 = 0.0;
        let status = crate::c_api_data::vxCopyScalarData(
            strength_thresh,
            &mut val as *mut f32 as *mut c_void,
            0x11001, 0x0
        );
        if status == VX_SUCCESS { val } else { 0.001 }
    } else { 0.001 };

    let min_dist: f32 = if !min_distance.is_null() {
        let mut val: f32 = 0.0;
        let status = crate::c_api_data::vxCopyScalarData(
            min_distance,
            &mut val as *mut f32 as *mut c_void,
            0x11001, 0x0
        );
        if status == VX_SUCCESS { val } else { 3.0 }
    } else { 3.0 };

    let k: f32 = if !sensitivity.is_null() {
        let mut val: f32 = 0.0;
        let status = crate::c_api_data::vxCopyScalarData(
            sensitivity,
            &mut val as *mut f32 as *mut c_void,
            0x11001, 0x0
        );
        if status == VX_SUCCESS { val } else { 0.04 }
    } else { 0.04 };

    let gs = gradient_size as usize;
    let bs = block_size as usize;
    // Valid gradient sizes: 3, 5, 7
    let gs = if gs == 3 || gs == 5 || gs == 7 { gs } else { 3 };
    // Valid block sizes: 3, 5, 7
    let bs = if bs == 3 || bs == 5 || bs == 7 { bs } else { 3 };

    unsafe {
        let src = match c_image_to_rust(input) {
            Some(img) => img,
            None => {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        };

        let width = src.width();
        let height = src.height();

        if width < 3 || height < 3 {
            // Image too small for Harris corners
            if !corners.is_null() {
                let arr = &*(corners as *const crate::unified_c_api::VxCArray);
                if let Ok(mut arr_data) = arr.items.write() {
                    arr_data.clear();
                }
            }
            if !num_corners.is_null() {
                let num: usize = 0;
                crate::c_api_data::vxCopyScalarData(
                    num_corners,
                    &num as *const usize as *mut c_void,
                    0x11002, 0x0
                );
            }
            return VX_SUCCESS;
        }

        // Get image data as flat u8 array
        let img_data = src.data();

        // Normalization factor matching MIVisionX reference:
        // For 3x3: div_factor = 1 (gradients already small)
        // For larger kernels, scale by 2^(gs-1) to match the reference's separation
        let div_factor: f32 = match gs {
            3 => 1.0,
            5 => 16.0,  // 2^4
            7 => 64.0,  // 2^6
            _ => 1.0,
        };

        // Compute GxGx, GxGy, GyGy structure tensor components
        // using separable Sobel filters (horizontal then vertical pass)
        let mut gxy = vec![GxyComponent { ixx: 0.0f32, ixy: 0.0f32, iyy: 0.0f32 }; width * height];

        match gs {
            3 => harris_sobel_3x3(&img_data, width, height, &mut gxy, div_factor),
            5 => harris_sobel_5x5(&img_data, width, height, &mut gxy, div_factor),
            7 => harris_sobel_7x7(&img_data, width, height, &mut gxy, div_factor),
            _ => harris_sobel_3x3(&img_data, width, height, &mut gxy, div_factor),
        }

        // Compute Harris response using sliding window accumulation over blockSize
        let half_block = bs / 2;
        let block_area = (bs * bs) as f32; // normalization by window size
        let mut responses = vec![0.0f32; width * height];

        // Sliding window: first compute column sums, then slide horizontally
        // This reduces O(W*H*B^2) to O(W*H*B)
        let mut col_sums_ixx = vec![0.0f32; width];
        let mut col_sums_ixy = vec![0.0f32; width];
        let mut col_sums_iyy = vec![0.0f32; width];

        for y in half_block..height - half_block {
            // Initialize column sums for this row
            // Each col_sum[c] = sum of gxy[y-half_block..=y+half_block][c].component
            col_sums_ixx.fill(0.0);
            col_sums_ixy.fill(0.0);
            col_sums_iyy.fill(0.0);

            for row in y - half_block..=y + half_block {
                let row_off = row * width;
                for col in half_block..width - half_block {
                    col_sums_ixx[col] += gxy[row_off + col].ixx;
                    col_sums_ixy[col] += gxy[row_off + col].ixy;
                    col_sums_iyy[col] += gxy[row_off + col].iyy;
                }
            }

            // Now slide horizontally across the row
            // Initialize window sum from first position
            let mut win_ixx = 0.0f32;
            let mut win_ixy = 0.0f32;
            let mut win_iyy = 0.0f32;
            for col in half_block..half_block + bs {
                win_ixx += col_sums_ixx[col];
                win_ixy += col_sums_ixy[col];
                win_iyy += col_sums_iyy[col];
            }

            // First position
            let x = half_block;
            let det = win_ixx * win_iyy - win_ixy * win_ixy;
            let trace = win_ixx + win_iyy;
            let mc = det - k * trace * trace;
            responses[y * width + x] = mc;

            // Slide the window right
            for x in (half_block + 1)..width - half_block {
                // Subtract leftmost column, add new rightmost column
                let left_col = x - 1 - half_block;
                let right_col = x + half_block;
                win_ixx += col_sums_ixx[right_col] - col_sums_ixx[left_col];
                win_ixy += col_sums_ixy[right_col] - col_sums_ixy[left_col];
                win_iyy += col_sums_iyy[right_col] - col_sums_iyy[left_col];

                let det = win_ixx * win_iyy - win_ixy * win_ixy;
                let trace = win_ixx + win_iyy;
                let mc = det - k * trace * trace;
                responses[y * width + x] = mc;
            }
        }

        // Normalize responses by block_area
        // This matches the MIVisionX reference which divides by the window size
        for r in responses.iter_mut() {
            *r /= block_area;
        }

        // Non-maximum suppression with min_distance using grid-based approach
        let radius = min_dist as i32;
        let radius_sq = (min_dist * min_dist) as f32;

        // Phase 1: Find all local maxima above threshold (3x3 NMS)
        let mut candidates: Vec<(i32, i32, f32)> = Vec::new();
        for y in 1..(height as i32 - 1) {
            for x in 1..(width as i32 - 1) {
                let idx = (y as usize) * width + (x as usize);
                let r = responses[idx];
                if r <= threshold {
                    continue;
                }
                // Check 3x3 neighborhood for local max (strictly greater, not equal)
                let mut is_max = true;
                'nms: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            if responses[(ny as usize) * width + (nx as usize)] >= r {
                                is_max = false;
                                break 'nms;
                            }
                        }
                    }
                }
                if is_max {
                    candidates.push((x, y, r));
                }
            }
        }

        // Sort by strength descending
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Phase 2: Grid-based NMS with min_distance radius check
        let mut corner_list: Vec<(i32, i32, f32)> = Vec::new();
        if radius <= 0 || radius_sq <= 0.0 {
            // No distance constraint, keep all
            corner_list = candidates;
        } else {
            // Use a grid for efficient proximity checking
            let cell_size = (radius as usize).max(1);
            let grid_w = (width + cell_size - 1) / cell_size;
            let grid_h = (height + cell_size - 1) / cell_size;
            // grid stores (x, y) of the placed corner in each cell
            let mut grid: Vec<(i32, i32)> = vec![(-1i32, -1i32); grid_w * grid_h];

            for &(x, y, strength) in &candidates {
                let cx = (x as usize) / cell_size;
                let cy = (y as usize) / cell_size;

                // Check this cell and neighboring cells (within radius)
                let mut too_close = false;
                let search_range = 2; // corners can only be in adjacent cells
                for gy in cy.saturating_sub(search_range)..=(cy + search_range).min(grid_h - 1) {
                    for gx in cx.saturating_sub(search_range)..=(cx + search_range).min(grid_w - 1) {
                        let (px, py) = grid[gy * grid_w + gx];
                        if px >= 0 {
                            let dx = x - px;
                            let dy = y - py;
                            if (dx * dx + dy * dy) as f32 <= radius_sq {
                                too_close = true;
                                break;
                            }
                        }
                    }
                    if too_close { break; }
                }

                if !too_close {
                    corner_list.push((x, y, strength));
                    grid[cy * grid_w + cx] = (x, y);
                }
            }
        }

        // Write corners to output array
        if !corners.is_null() {
            let arr = &*(corners as *const crate::unified_c_api::VxCArray);
            let mut arr_data = match arr.items.write() {
                Ok(d) => d,
                Err(_) => return VX_ERROR_INVALID_PARAMETERS,
            };

            let keypoint_size = std::mem::size_of::<vx_keypoint_t>();
            let output_size = corner_list.len() * keypoint_size;
            if arr_data.len() < output_size {
                arr_data.resize(output_size, 0);
            }

            for (i, &(x, y, strength)) in corner_list.iter().enumerate() {
                let offset = i * keypoint_size;
                if offset + keypoint_size <= arr_data.len() {
                    let kp = vx_keypoint_t {
                        x,
                        y,
                        strength,
                        scale: 0.0,
                        orientation: 0.0,
                        tracking_status: 1,
                        error: 0.0,
                    };
                    let kp_ptr = arr_data.as_mut_ptr().add(offset) as *mut vx_keypoint_t;
                    *kp_ptr = kp;
                }
            }
            // Zero out remaining data
            let end = (corner_list.len() * keypoint_size).min(arr_data.len());
            for i in (corner_list.len() * keypoint_size)..end {
                arr_data[i] = 0;
            }
        }

        // Write num_corners to scalar
        if !num_corners.is_null() {
            let num = corner_list.len() as usize;
            crate::c_api_data::vxCopyScalarData(
                num_corners,
                &num as *const usize as *mut c_void,
                0x11002, // VX_WRITE_ONLY
                0x0
            );
        }

        VX_SUCCESS
    }
}

/// Structure tensor component per pixel
#[derive(Clone, Copy, Default)]
struct GxyComponent {
    ixx: f32,
    ixy: f32,
    iyy: f32,
}

/// Separable Sobel 3x3 + structure tensor computation
/// Gx: horizontal = [-1, 0, 1], vertical = [1, 2, 1]
/// Gy: horizontal = [1, 2, 1], vertical = [-1, 0, 1]
/// Computes Gx*Gx, Gx*Gy, Gy*Gy per pixel
fn harris_sobel_3x3(img_data: &[u8], width: usize, height: usize, gxy: &mut [GxyComponent], div_factor: f32) {
    let inv_df = 1.0 / div_factor;

    for y in 1..height - 1 {
        let row_m = (y - 1) * width;
        let row_c = y * width;
        let row_p = (y + 1) * width;

        for x in 1..width - 1 {
            // Separable Sobel 3x3:
            // Gx = (I[y+1][x+1] - I[y+1][x-1]) + 2*(I[y][x+1] - I[y][x-1]) + (I[y-1][x+1] - I[y-1][x-1])
            // Gy = (I[y+1][x+1] + 2*I[y+1][x] + I[y+1][x-1]) - (I[y-1][x+1] + 2*I[y-1][x] + I[y-1][x-1])
            let p_m_l = img_data[row_m + x - 1] as i32;
            let p_m_c = img_data[row_m + x] as i32;
            let p_m_r = img_data[row_m + x + 1] as i32;
            let p_c_l = img_data[row_c + x - 1] as i32;
            let p_c_r = img_data[row_c + x + 1] as i32;
            let p_p_l = img_data[row_p + x - 1] as i32;
            let p_p_c = img_data[row_p + x] as i32;
            let p_p_r = img_data[row_p + x + 1] as i32;

            let gx: i32 = (p_p_r - p_p_l) + 2 * (p_c_r - p_c_l) + (p_m_r - p_m_l);
            let gy: i32 = (p_p_r + 2 * p_p_c + p_p_l) - (p_m_r + 2 * p_m_c + p_m_l);

            let gxf = gx as f32 * inv_df;
            let gyf = gy as f32 * inv_df;

            let idx = y * width + x;
            gxy[idx].ixx = gxf * gxf;
            gxy[idx].ixy = gxf * gyf;
            gxy[idx].iyy = gyf * gyf;
        }
    }
}

/// 5x5 Sobel + structure tensor computation
fn harris_sobel_5x5(img_data: &[u8], width: usize, height: usize, gxy: &mut [GxyComponent], div_factor: f32) {
    let inv_df = 1.0 / div_factor;

    // 5x5 Sobel kernels (from OpenVX spec)
    let gx_kernel: [[i32; 5]; 5] = [
        [-1, -2,  0,  2,  1],
        [-4, -8,  0,  8,  4],
        [-6,-12,  0, 12,  6],
        [-4, -8,  0,  8,  4],
        [-1, -2,  0,  2,  1],
    ];
    let gy_kernel: [[i32; 5]; 5] = [
        [-1, -4, -6, -4, -1],
        [-2, -8,-12, -8, -2],
        [ 0,  0,  0,  0,  0],
        [ 2,  8, 12,  8,  2],
        [ 1,  4,  6,  4,  1],
    ];

    for y in 2..height - 2 {
        for x in 2..width - 2 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;
            for ky in 0..5 {
                let row = (y + ky - 2) * width;
                for kx in 0..5 {
                    let px = row + x + kx - 2;
                    let p = img_data[px] as i32;
                    gx += p * gx_kernel[ky][kx];
                    gy += p * gy_kernel[ky][kx];
                }
            }

            // Divide by 2^(5-1) = 16 for normalization
            let gxf = (gx >> 4) as f32 * inv_df;
            let gyf = (gy >> 4) as f32 * inv_df;

            let idx = y * width + x;
            gxy[idx].ixx = gxf * gxf;
            gxy[idx].ixy = gxf * gyf;
            gxy[idx].iyy = gyf * gyf;
        }
    }
}

/// 7x7 Sobel + structure tensor computation
fn harris_sobel_7x7(img_data: &[u8], width: usize, height: usize, gxy: &mut [GxyComponent], div_factor: f32) {
    let inv_df = 1.0 / div_factor;

    // 7x7 Sobel kernels
    let gx_kernel: [[i32; 7]; 7] = [
        [-1, -4, -5,  0,  5,  4,  1],
        [-6,-24,-30,  0, 30, 24,  6],
        [-15,-60,-75, 0, 75, 60, 15],
        [-20,-80,-100,0,100, 80, 20],
        [-15,-60,-75, 0, 75, 60, 15],
        [ -6,-24,-30,  0, 30, 24,  6],
        [ -1, -4, -5,  0,  5,  4,  1],
    ];
    let gy_kernel: [[i32; 7]; 7] = [
        [-1, -6,-15,-20,-15, -6, -1],
        [-4,-24,-60,-80,-60,-24, -4],
        [-5,-30,-75,-100,-75,-30, -5],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 5, 30, 75,100, 75, 30,  5],
        [ 4, 24, 60, 80, 60, 24,  4],
        [ 1,  6, 15, 20, 15,  6,  1],
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;
            for ky in 0..7 {
                let row = (y + ky - 3) * width;
                for kx in 0..7 {
                    let px = row + x + kx - 3;
                    let p = img_data[px] as i32;
                    gx += p * gx_kernel[ky][kx];
                    gy += p * gy_kernel[ky][kx];
                }
            }

            // Divide by 2^(7-1) = 64 for normalization
            let gxf = (gx >> 6) as f32 * inv_df;
            let gyf = (gy >> 6) as f32 * inv_df;

            let idx = y * width + x;
            gxy[idx].ixx = gxf * gxf;
            gxy[idx].ixy = gxf * gyf;
            gxy[idx].iyy = gyf * gyf;
        }
    }
}