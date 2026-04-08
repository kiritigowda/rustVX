<<<<<<< HEAD
//! Optical Flow implementation

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait};
use openvx_image::{Image, ImageFormat};
=======
//! Optical Flow implementation - Lucas-Kanade Pyramidal Optical Flow
//!
//! Implements the pyramidal Lucas-Kanade optical flow algorithm as specified
//! in OpenVX 1.3.1. This algorithm tracks feature points across two images
//! using a multi-resolution pyramid approach with iterative refinement.

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait, VxStatus};
use openvx_image::Image;
use crate::utils::{get_pixel_bordered, BorderMode, clamp_u8};
>>>>>>> origin/master

/// OpticalFlowPyrLK kernel - Lucas-Kanade pyramidal optical flow
pub struct OpticalFlowPyrLKKernel;

impl OpticalFlowPyrLKKernel {
    pub fn new() -> Self { OpticalFlowPyrLKKernel }
}

impl KernelTrait for OpticalFlowPyrLKKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.optical_flow_pyr_lk" }
    fn get_enum(&self) -> VxKernel { VxKernel::OpticalFlowPyrLK }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
<<<<<<< HEAD
        if params.len() < 5 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
=======
        // optical_flow_pyr_lk expects 7 parameters:
        // 0: old_pyramid (pyramid)
        // 1: new_pyramid (pyramid)  
        // 2: prev_pts (array of keypoints)
        // 3: new_pts_estimates (array of keypoints, optional)
        // 4: new_pts (output array of keypoints)
        // 5: termination (termination criteria)
        // 6: num_iterations (scalar)
        // 7: epsilon (scalar)
        // 8: window_dimension (scalar)
        // 9: use_initial_estimate (optional scalar)
        if params.len() < 5 {
            return Err(VxStatus::ErrorInvalidParameters);
>>>>>>> origin/master
        }
        Ok(())
    }
    
    fn execute(&self, _params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
<<<<<<< HEAD
        // Simplified: just run successfully
=======
        // The actual execution is handled by the VXU implementation
        // which works with C API types directly
>>>>>>> origin/master
        Ok(())
    }
}

<<<<<<< HEAD
/// Build a Gaussian pyramid (5 levels with 5x5 kernel)
pub fn build_gaussian_pyramid(image: &Image, levels: usize) -> VxResult<Vec<Image>> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(Image::new(image.width(), image.height(), ImageFormat::Gray));
    
    // Copy first level
    let src_data = image.data();
    {
        let mut dst_data = pyramid[0].data_mut();
        dst_data.copy_from_slice(src_data.as_ref());
    }
    
    // Build subsequent levels
    let mut prev_width = image.width();
    let mut prev_height = image.height();
    
    for i in 1..levels {
        let width = (prev_width + 1) / 2;
        let height = (prev_height + 1) / 2;
        pyramid.push(Image::new(width, height, ImageFormat::Gray));
        
        // Apply Gaussian blur then downsample
        let temp = Image::new(prev_width, prev_height, ImageFormat::Gray);
        crate::filter::gaussian5x5(&pyramid[i - 1], &temp)?;
        downsample(&temp, &pyramid[i])?;
        
        prev_width = width;
        prev_height = height;
    }
    
    Ok(pyramid)
}

/// Downsample by 2x using averaging
fn downsample(src: &Image, dst: &Image) -> VxResult<()> {
    let src_width = src.width();
    let src_height = src.height();
    let dst_width = dst.width();
    let dst_height = dst.height();
    
    let mut dst_data = dst.data_mut();
    
    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = x * 2;
            let src_y = y * 2;
=======
/// Gaussian pyramid level structure
#[derive(Debug, Clone)]
pub struct PyramidLevel {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Build a Gaussian pyramid with the specified number of levels
/// Uses the same Gaussian kernel as Gaussian5x5: [1,4,6,4,1]/16
pub fn build_gaussian_pyramid(image: &[u8], width: usize, height: usize, levels: usize) -> Vec<PyramidLevel> {
    let mut pyramid = Vec::with_capacity(levels);
    
    // Level 0: Original image
    pyramid.push(PyramidLevel {
        width,
        height,
        data: image.to_vec(),
    });
    
    // Build subsequent levels
    for level in 1..levels {
        let prev = &pyramid[level - 1];
        let new_width = (prev.width + 1) / 2;
        let new_height = (prev.height + 1) / 2;
        
        // Apply Gaussian blur then downsample
        let blurred = gaussian5x5_blur(&prev.data, prev.width, prev.height);
        let downsampled = downsample_by_2(&blurred, prev.width, prev.height);
        
        pyramid.push(PyramidLevel {
            width: new_width,
            height: new_height,
            data: downsampled,
        });
    }
    
    pyramid
}

/// Apply 5x5 Gaussian blur (separable kernel [1,4,6,4,1]/16)
fn gaussian5x5_blur(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let kernel = [1, 4, 6, 4, 1];
    let mut temp = vec![0u8; width * height];
    let mut result = vec![0u8; width * height];
    
    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut sum: i32 = 0;
            let mut weight: i32 = 0;
            for k in 0..5 {
                let px = x as isize + k as isize - 2;
                if px >= 0 && px < width as isize {
                    sum += data[y * width + px as usize] as i32 * kernel[k];
                    weight += kernel[k];
                }
            }
            temp[y * width + x] = clamp_u8(sum / weight.max(1));
        }
    }
    
    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum: i32 = 0;
            let mut weight: i32 = 0;
            for k in 0..5 {
                let py = y as isize + k as isize - 2;
                if py >= 0 && py < height as isize {
                    sum += temp[py as usize * width + x] as i32 * kernel[k];
                    weight += kernel[k];
                }
            }
            result[y * width + x] = clamp_u8(sum / weight.max(1));
        }
    }
    
    result
}

/// Downsample image by 2x using averaging
fn downsample_by_2(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let new_width = (width + 1) / 2;
    let new_height = (height + 1) / 2;
    let mut result = vec![0u8; new_width * new_height];
    
    for y in 0..new_height {
        for x in 0..new_width {
            let src_y = y * 2;
            let src_x = x * 2;
>>>>>>> origin/master
            
            // Average 2x2 block
            let mut sum = 0u32;
            let mut count = 0u32;
            
            for dy in 0..2 {
                for dx in 0..2 {
<<<<<<< HEAD
                    let sx = src_x + dx;
                    let sy = src_y + dy;
                    if sx < src_width && sy < src_height {
                        sum += src.get_pixel(sx, sy) as u32;
=======
                    let sy = src_y + dy;
                    let sx = src_x + dx;
                    if sx < width && sy < height {
                        sum += data[sy * width + sx] as u32;
>>>>>>> origin/master
                        count += 1;
                    }
                }
            }
            
<<<<<<< HEAD
            dst_data[y * dst_width + x] = (sum / count.max(1)) as u8;
        }
    }
    
    Ok(())
}

/// Lucas-Kanade optical flow at a single pyramid level
pub fn lucas_kanade(
    prev: &Image,
    next: &Image,
    points: &[(f32, f32)],
    window_size: usize,
    max_iter: usize,
) -> VxResult<Vec<(f32, f32)>> {
    let width = prev.width();
    let height = prev.height();
    let half_window = (window_size / 2) as isize;
    
    let mut flow = Vec::with_capacity(points.len());
    
    for &(px, py) in points {
        let mut u: f32 = 0.0;
        let mut v: f32 = 0.0;
        
        // Iterative refinement
        for _ in 0..max_iter {
=======
            result[y * new_width + x] = (sum / count.max(1)) as u8;
        }
    }
    
    result
}

/// Lucas-Kanade optical flow at a single pyramid level
/// Solves the 2x2 linear system using Cramer's rule
pub fn lucas_kanade_single_level(
    prev_level: &PyramidLevel,
    next_level: &PyramidLevel,
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iter: usize,
    epsilon: f32,
    use_initial_estimate: bool,
    initial_flow: &[(f32, f32)],
) -> (Vec<(f32, f32)>, Vec<bool>) {
    let half_window = (window_size / 2) as isize;
    let width = prev_level.width;
    let height = prev_level.height;
    
    let mut flow = Vec::with_capacity(prev_points.len());
    let mut status = Vec::with_capacity(prev_points.len());
    
    for (i, &(px, py)) in prev_points.iter().enumerate() {
        // Initialize flow vector
        let (mut u, mut v) = if use_initial_estimate && i < initial_flow.len() {
            initial_flow[i]
        } else {
            (0.0, 0.0)
        };
        
        let mut converged = false;
        let mut valid = true;
        
        // Iterative refinement
        for _iter in 0..max_iter {
>>>>>>> origin/master
            let mut sum_ix2: f32 = 0.0;
            let mut sum_iy2: f32 = 0.0;
            let mut sum_ixiy: f32 = 0.0;
            let mut sum_ixit: f32 = 0.0;
            let mut sum_iyit: f32 = 0.0;
<<<<<<< HEAD
            
            // Compute structure tensor and temporal derivatives
=======
            let mut valid_pixels = 0;
            
            // Compute spatial gradients and temporal gradient
>>>>>>> origin/master
            for wy in -half_window..=half_window {
                for wx in -half_window..=half_window {
                    let x = px as isize + wx;
                    let y = py as isize + wy;
                    
                    if x < 0 || x >= width as isize || y < 0 || y >= height as isize {
                        continue;
                    }
                    
<<<<<<< HEAD
                    // Spatial gradients (using simple finite differences)
                    let ix = compute_grad_x(prev, x as usize, y as usize);
                    let iy = compute_grad_y(prev, x as usize, y as usize);
                    
                    // Temporal gradient (frame difference)
                    let curr_x = (x as f32 + u).max(0.0).min(width as f32 - 1.0) as usize;
                    let curr_y = (y as f32 + v).max(0.0).min(height as f32 - 1.0) as usize;
                    let it = next.get_pixel(curr_x, curr_y) as f32 - prev.get_pixel(x as usize, y as usize) as f32;
=======
                    let x = x as usize;
                    let y = y as usize;
                    valid_pixels += 1;
                    
                    // Compute spatial gradients using Sobel-like 3x3
                    let ix = compute_grad_x(&prev_level.data, x, y, width, height);
                    let iy = compute_grad_y(&prev_level.data, x, y, width, height);
                    
                    // Compute temporal gradient (frame difference)
                    let curr_x = ((x as f32 + u).max(0.0).min((width - 1) as f32)) as usize;
                    let curr_y = ((y as f32 + v).max(0.0).min((height - 1) as f32)) as usize;
                    
                    let prev_val = prev_level.data[y * width + x] as f32;
                    let next_val = next_level.data[curr_y * width + curr_x] as f32;
                    let it = next_val - prev_val;
>>>>>>> origin/master
                    
                    sum_ix2 += ix * ix;
                    sum_iy2 += iy * iy;
                    sum_ixiy += ix * iy;
                    sum_ixit += ix * it;
                    sum_iyit += iy * it;
                }
            }
            
<<<<<<< HEAD
            // Solve 2x2 system using Cramer's rule
            let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
            if det.abs() < 1e-6 {
                break; // Singular matrix
=======
            // Check if we have enough valid pixels in the window
            if valid_pixels < window_size * window_size / 2 {
                valid = false;
                break;
            }
            
            // Solve 2x2 system [sum_ix2 sum_ixiy; sum_ixiy sum_iy2] * [du; dv] = -[sum_ixit; sum_iyit]
            // Using Cramer's rule
            let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
            
            // Check for singular matrix (ill-conditioned)
            if det.abs() < 1e-6 {
                valid = false;
                break;
>>>>>>> origin/master
            }
            
            let du = (sum_iy2 * sum_ixit - sum_ixiy * sum_iyit) / det;
            let dv = (sum_ix2 * sum_iyit - sum_ixiy * sum_ixit) / det;
            
<<<<<<< HEAD
            u -= du;
            v -= dv;
            
            if du.abs() < 0.01 && dv.abs() < 0.01 {
                break; // Converged
=======
            // Update flow
            u -= du;
            v -= dv;
            
            // Check convergence
            if du * du + dv * dv < epsilon * epsilon {
                converged = true;
                break;
>>>>>>> origin/master
            }
        }
        
        flow.push((u, v));
<<<<<<< HEAD
    }
    
    Ok(flow)
}

/// Pyramidal Lucas-Kanade optical flow
pub fn optical_flow_pyr_lk(
    prev_pyramid: &[Image],
    next_pyramid: &[Image],
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iter: usize,
) -> VxResult<Vec<(f32, f32)>> {
    let levels = prev_pyramid.len();
    let mut flow: Vec<(f32, f32)> = vec![(0.0, 0.0); prev_points.len()];
    
    // Coarse-to-fine refinement
    for level in (0..levels).rev() {
        // Scale points for this level
        let scale = 1u32 << level;
        let level_points: Vec<(f32, f32)> = prev_points
            .iter()
            .map(|(x, y)| (x / scale as f32, y / scale as f32))
            .collect();
        
        // Add previous level's flow
        let level_points_with_flow: Vec<(f32, f32)> = level_points
            .iter()
            .enumerate()
            .map(|(i, (x, y))| (x + flow[i].0 / scale as f32, y + flow[i].1 / scale as f32))
            .collect();
        
        // Compute optical flow at this level
        let level_flow = lucas_kanade(
            &prev_pyramid[level],
            &next_pyramid[level],
            &level_points_with_flow,
            window_size,
            max_iter,
        )?;
        
        // Accumulate flow
        if level < levels - 1 {
            for i in 0..flow.len() {
                flow[i].0 += level_flow[i].0 * scale as f32;
                flow[i].1 += level_flow[i].1 * scale as f32;
            }
        } else {
            flow = level_flow;
        }
    }
    
    Ok(flow)
}

/// Compute x gradient using central differences
fn compute_grad_x(img: &Image, x: usize, y: usize) -> f32 {
    let width = img.width();
    let left = if x > 0 { img.get_pixel(x - 1, y) } else { img.get_pixel(x, y) };
    let right = if x + 1 < width { img.get_pixel(x + 1, y) } else { img.get_pixel(x, y) };
    (right as f32 - left as f32) / 2.0
}

/// Compute y gradient using central differences
fn compute_grad_y(img: &Image, x: usize, y: usize) -> f32 {
    let height = img.height();
    let top = if y > 0 { img.get_pixel(x, y - 1) } else { img.get_pixel(x, y) };
    let bottom = if y + 1 < height { img.get_pixel(x, y + 1) } else { img.get_pixel(x, y) };
    (bottom as f32 - top as f32) / 2.0
}
=======
        status.push(valid);
    }
    
    (flow, status)
}

/// Pyramidal Lucas-Kanade optical flow
/// Implements the coarse-to-fine refinement approach
pub fn optical_flow_pyr_lk_core(
    prev_pyramid: &[PyramidLevel],
    next_pyramid: &[PyramidLevel],
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iter: usize,
    epsilon: f32,
    use_initial_estimate: bool,
    initial_flow: &[(f32, f32)],
) -> (Vec<(f32, f32)>, Vec<bool>) {
    let levels = prev_pyramid.len();
    let num_points = prev_points.len();
    
    // Initialize flow vectors
    let mut flow: Vec<(f32, f32)> = if use_initial_estimate && initial_flow.len() >= num_points {
        initial_flow[..num_points].to_vec()
    } else {
        vec![(0.0, 0.0); num_points]
    };
    
    let mut final_status = vec![true; num_points];
    
    // Coarse-to-fine refinement (start from top of pyramid)
    for level in (0..levels).rev() {
        let level_scale = 1u32 << level; // 2^level
        let scale = level_scale as f32;
        
        // Scale points for this level
        let level_points: Vec<(f32, f32)> = prev_points
            .iter()
            .map(|(x, y)| (x / scale, y / scale))
            .collect();
        
        // Scale accumulated flow from previous levels
        let level_initial_flow: Vec<(f32, f32)> = flow
            .iter()
            .map(|(u, v)| (u / scale, v / scale))
            .collect();
        
        // Use initial estimate if not at the coarsest level
        let use_estimate = level < levels - 1 || use_initial_estimate;
        
        // Compute optical flow at this level
        let (level_flow, level_status) = lucas_kanade_single_level(
            &prev_pyramid[level],
            &next_pyramid[level],
            &level_points,
            window_size,
            max_iter,
            epsilon,
            use_estimate,
            &level_initial_flow,
        );
        
        // Accumulate flow and update status
        for i in 0..num_points {
            if level == levels - 1 {
                // At the finest level, just use the computed flow
                flow[i] = (level_flow[i].0 * scale, level_flow[i].1 * scale);
            } else {
                // Propagate flow to next finer level
                flow[i].0 += level_flow[i].0 * scale;
                flow[i].1 += level_flow[i].1 * scale;
            }
            final_status[i] = final_status[i] && level_status[i];
        }
    }
    
    (flow, final_status)
}

/// Compute x gradient using central differences with proper border handling
fn compute_grad_x(data: &[u8], x: usize, y: usize, width: usize, _height: usize) -> f32 {
    let left = if x > 0 {
        data[y * width + (x - 1)]
    } else {
        data[y * width + x]
    };
    let right = if x + 1 < width {
        data[y * width + (x + 1)]
    } else {
        data[y * width + x]
    };
    (right as f32 - left as f32) / 2.0
}

/// Compute y gradient using central differences with proper border handling
fn compute_grad_y(data: &[u8], x: usize, y: usize, width: usize, height: usize) -> f32 {
    let top = if y > 0 {
        data[(y - 1) * width + x]
    } else {
        data[y * width + x]
    };
    let bottom = if y + 1 < height {
        data[(y + 1) * width + x]
    } else {
        data[y * width + x]
    };
    (bottom as f32 - top as f32) / 2.0
}

/// Keypoint structure for optical flow
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VxKeypoint {
    pub x: f32,
    pub y: f32,
    pub strength: f32,
    pub scale: f32,
    pub orientation: f32,
    pub error: f32,
}

/// Termination criteria for optical flow
pub enum TerminationCriteria {
    /// Stop after fixed number of iterations
    Iterations(u32),
    /// Stop when movement is less than epsilon
    Epsilon(f32),
    /// Stop when either condition is met
    IterationsOrEpsilon(u32, f32),
}

/// VXU Optical Flow implementation
/// This is the C API bridge function
pub fn vxu_optical_flow_pyr_lk_impl(
    prev_image: &[u8],
    next_image: &[u8],
    width: usize,
    height: usize,
    prev_points: &[(f32, f32)],
    num_levels: usize,
    window_size: usize,
    max_iter: usize,
    epsilon: f32,
    use_initial_estimate: bool,
    initial_estimates: &[(f32, f32)],
) -> (Vec<(f32, f32)>, Vec<bool>) {
    // Build pyramids for both images
    let prev_pyramid = build_gaussian_pyramid(prev_image, width, height, num_levels);
    let next_pyramid = build_gaussian_pyramid(next_image, width, height, num_levels);
    
    // Compute optical flow
    optical_flow_pyr_lk_core(
        &prev_pyramid,
        &next_pyramid,
        prev_points,
        window_size,
        max_iter,
        epsilon,
        use_initial_estimate,
        initial_estimates,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyramid_construction() {
        let width = 64;
        let height = 64;
        let data = vec![128u8; width * height];
        let pyramid = build_gaussian_pyramid(&data, width, height, 3);
        
        assert_eq!(pyramid.len(), 3);
        assert_eq!(pyramid[0].width, 64);
        assert_eq!(pyramid[0].height, 64);
        assert_eq!(pyramid[1].width, 32);
        assert_eq!(pyramid[1].height, 32);
        assert_eq!(pyramid[2].width, 16);
        assert_eq!(pyramid[2].height, 16);
    }

    #[test]
    fn test_downsample() {
        let width = 4;
        let height = 4;
        // 4x4 checkerboard
        let data = vec![
            0, 255, 0, 255,
            255, 0, 255, 0,
            0, 255, 0, 255,
            255, 0, 255, 0,
        ];
        
        let result = downsample_by_2(&data, width, height);
        assert_eq!(result.len(), 4); // 2x2
        // Each 2x2 block averages to ~128
        assert!(result[0] > 100 && result[0] < 156);
    }

    #[test]
    fn test_gaussian_blur() {
        let width = 8;
        let height = 8;
        let data = vec![128u8; width * height];
        let blurred = gaussian5x5_blur(&data, width, height);
        
        // Constant image should remain constant
        assert_eq!(blurred.len(), width * height);
        assert!(blurred.iter().all(|&v| v == 128));
    }
}
>>>>>>> origin/master
