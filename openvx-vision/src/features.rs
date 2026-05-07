//! Feature detection

use openvx_core::{Context, KernelTrait, Referenceable, VxKernel, VxResult};
use openvx_image::Image;

/// HarrisCorners kernel
pub struct HarrisCornersKernel;

impl HarrisCornersKernel {
    pub fn new() -> Self {
        HarrisCornersKernel
    }
}

impl KernelTrait for HarrisCornersKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.harris_corners"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::HarrisCorners
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 6 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        // Simplified: just run Harris corner detection
        let k = 0.04f32; // Harris sensitivity
        let threshold = 100.0f32; // Strength threshold
        let _corners = harris_corners(src, k, threshold, 10)?;

        Ok(())
    }
}

/// FASTCorners kernel
pub struct FASTCornersKernel;

impl FASTCornersKernel {
    pub fn new() -> Self {
        FASTCornersKernel
    }
}

impl KernelTrait for FASTCornersKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.fast_corners"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::FASTCorners
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 4 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        let threshold = 20u8; // Default threshold
        let _corners = fast9(src, threshold)?;

        Ok(())
    }
}

/// Harris corner detector
/// 1. Compute Ix, Iy with Sobel
/// 2. Compute structure tensor [Ix², Ixy; Ixy, Iy²]
/// 3. Compute corner response = det - k*trace²
/// 4. Non-max suppression
/// 5. Sort by response
pub fn harris_corners(
    image: &Image,
    k: f32,
    threshold: f32,
    _min_distance: usize,
) -> VxResult<Vec<Corner>> {
    let width = image.width();
    let height = image.height();
    // Use saturating_mul to prevent integer overflow
    let response_size = width.saturating_mul(height);
    let mut responses = vec![0f32; response_size];

    // Compute gradients using Sobel
    let (grad_x, grad_y) = compute_gradients_sobel(image)?;

    // Compute structure tensor and corner response
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut ixx: f32 = 0.0;
            let mut iyy: f32 = 0.0;
            let mut ixy: f32 = 0.0;

            // Sum over 3x3 window
            for wy in -1..=1 {
                for wx in -1..=1 {
                    let idx = ((y as isize + wy) as usize) * width + ((x as isize + wx) as usize);
                    let ix = grad_x[idx] as f32;
                    let iy = grad_y[idx] as f32;

                    ixx += ix * ix;
                    iyy += iy * iy;
                    ixy += ix * iy;
                }
            }

            // Harris corner response: R = det(M) - k * trace(M)²
            let det = ixx * iyy - ixy * ixy;
            let trace = ixx + iyy;
            let response = det - k * trace * trace;

            responses[y * width + x] = response;
        }
    }

    // Non-maximum suppression
    let mut corners = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let response = responses[y * width + x];

            if response < threshold {
                continue;
            }

            // Check if local maximum
            let mut is_max = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if responses[ny as usize * width + nx as usize] > response {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }

            if is_max {
                corners.push(Corner {
                    x,
                    y,
                    strength: response,
                });
            }
        }
    }

    // Sort by strength (descending)
    corners.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(corners)
}

/// FAST-9 corner detector
/// Bresenham circle of radius 3 (16 points)
/// A pixel is a corner if there are 9 contiguous pixels on the circle
/// that are all brighter than center + threshold or darker than center - threshold
pub fn fast9(image: &Image, threshold: u8) -> VxResult<Vec<Corner>> {
    let width = image.width();
    let height = image.height();
    let mut corners = Vec::new();

    // Bresenham circle of radius 3 (16 points around center)
    const CIRCLE_OFFSETS: [(isize, isize); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let center = image.get_pixel(x, y);
            let high = center.saturating_add(threshold);
            let low = center.saturating_sub(threshold);

            // Sample circle
            let mut circle = [0u8; 16];
            for (i, (dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
                let px = (x as isize + dx) as usize;
                let py = (y as isize + dy) as usize;
                circle[i] = image.get_pixel(px, py);
            }

            // Check for 9 contiguous brighter or darker pixels
            let mut is_corner = false;

            // Try all starting positions
            for start in 0..16 {
                // Check brighter
                let mut brighter_count = 0;
                let mut darker_count = 0;

                for i in 0..16 {
                    let idx = (start + i) % 16;
                    if circle[idx] > high {
                        brighter_count += 1;
                        darker_count = 0;
                    } else if circle[idx] < low {
                        darker_count += 1;
                        brighter_count = 0;
                    } else {
                        brighter_count = 0;
                        darker_count = 0;
                    }

                    if brighter_count >= 9 || darker_count >= 9 {
                        is_corner = true;
                        break;
                    }
                }

                if is_corner {
                    break;
                }
            }

            if is_corner {
                // Score using segment test
                let score = compute_fast_score(&circle, center, threshold);
                corners.push(Corner {
                    x,
                    y,
                    strength: score as f32,
                });
            }
        }
    }

    // Sort by strength
    corners.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(corners)
}

/// FAST-12 corner detector (stricter version)
pub fn fast12(image: &Image, threshold: u8) -> VxResult<Vec<Corner>> {
    let width = image.width();
    let height = image.height();
    let mut corners = Vec::new();

    // Bresenham circle of radius 3 (16 points)
    const CIRCLE_OFFSETS: [(isize, isize); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let center = image.get_pixel(x, y);
            let high = center.saturating_add(threshold);
            let low = center.saturating_sub(threshold);

            // Sample circle
            let mut circle = [0u8; 16];
            for (i, (dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
                let px = (x as isize + dx) as usize;
                let py = (y as isize + dy) as usize;
                circle[i] = image.get_pixel(px, py);
            }

            // Check for 12 contiguous brighter or darker pixels
            let mut is_corner = false;

            for start in 0..16 {
                let mut brighter_count = 0;
                let mut darker_count = 0;

                for i in 0..16 {
                    let idx = (start + i) % 16;
                    if circle[idx] > high {
                        brighter_count += 1;
                        darker_count = 0;
                    } else if circle[idx] < low {
                        darker_count += 1;
                        brighter_count = 0;
                    } else {
                        brighter_count = 0;
                        darker_count = 0;
                    }

                    if brighter_count >= 12 || darker_count >= 12 {
                        is_corner = true;
                        break;
                    }
                }

                if is_corner {
                    break;
                }
            }

            if is_corner {
                let score = compute_fast_score(&circle, center, threshold);
                corners.push(Corner {
                    x,
                    y,
                    strength: score as f32,
                });
            }
        }
    }

    corners.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(corners)
}

/// Compute FAST score (sum of absolute differences)
fn compute_fast_score(circle: &[u8; 16], center: u8, threshold: u8) -> u16 {
    let mut score: u16 = 0;
    for &p in circle.iter() {
        let diff = if p > center {
            (p - center) as u16
        } else {
            (center - p) as u16
        };
        if diff > threshold as u16 {
            score += diff;
        }
    }
    score
}

/// Compute gradients using Sobel operator
fn compute_gradients_sobel(image: &Image) -> VxResult<(Vec<f32>, Vec<f32>)> {
    let width = image.width();
    let height = image.height();
    // Use saturating_mul to prevent integer overflow
    let gradient_size = width.saturating_mul(height);
    let mut grad_x = vec![0f32; gradient_size];
    let mut grad_y = vec![0f32; gradient_size];

    // Sobel kernels
    const SOBEL_X: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const SOBEL_Y: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    let pixel = image.get_pixel(px, py) as i32;
                    gx += pixel * SOBEL_X[ky][kx];
                    gy += pixel * SOBEL_Y[ky][kx];
                }
            }

            let idx = y * width + x;
            grad_x[idx] = gx as f32 / 4.0; // Scale for normalized response
            grad_y[idx] = gy as f32 / 4.0;
        }
    }

    Ok((grad_x, grad_y))
}

/// Corner structure
#[derive(Debug, Clone, Copy)]
pub struct Corner {
    pub x: usize,
    pub y: usize,
    pub strength: f32,
}

/// Non-maximum suppression for corners
pub fn non_max_suppression(corners: &mut Vec<Corner>, min_distance: usize) {
    let mut i = 0;
    while i < corners.len() {
        let (cx, cy) = (corners[i].x, corners[i].y);

        // Remove nearby corners with lower strength
        corners.retain(|c| {
            let dx = (c.x as isize - cx as isize).abs() as usize;
            let dy = (c.y as isize - cy as isize).abs() as usize;
            if dx < min_distance && dy < min_distance {
                // Keep if it's the current corner
                c.x == cx && c.y == cy
            } else {
                true
            }
        });

        i += 1;
    }
}
