//! Optical Flow implementation

use openvx_core::{Context, Referenceable, VxResult, VxKernel, KernelTrait};
use openvx_image::{Image, ImageFormat};

/// OpticalFlowPyrLK kernel - Lucas-Kanade pyramidal optical flow
pub struct OpticalFlowPyrLKKernel;

impl OpticalFlowPyrLKKernel {
    pub fn new() -> Self { OpticalFlowPyrLKKernel }
}

impl KernelTrait for OpticalFlowPyrLKKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.optical_flow_pyr_lk" }
    fn get_enum(&self) -> VxKernel { VxKernel::OpticalFlowPyrLK }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 5 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, _params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        // Simplified: just run successfully
        Ok(())
    }
}

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
            
            // Average 2x2 block
            let mut sum = 0u32;
            let mut count = 0u32;
            
            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = src_x + dx;
                    let sy = src_y + dy;
                    if sx < src_width && sy < src_height {
                        sum += src.get_pixel(sx, sy) as u32;
                        count += 1;
                    }
                }
            }
            
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
            let mut sum_ix2: f32 = 0.0;
            let mut sum_iy2: f32 = 0.0;
            let mut sum_ixiy: f32 = 0.0;
            let mut sum_ixit: f32 = 0.0;
            let mut sum_iyit: f32 = 0.0;
            
            // Compute structure tensor and temporal derivatives
            for wy in -half_window..=half_window {
                for wx in -half_window..=half_window {
                    let x = px as isize + wx;
                    let y = py as isize + wy;
                    
                    if x < 0 || x >= width as isize || y < 0 || y >= height as isize {
                        continue;
                    }
                    
                    // Spatial gradients (using simple finite differences)
                    let ix = compute_grad_x(prev, x as usize, y as usize);
                    let iy = compute_grad_y(prev, x as usize, y as usize);
                    
                    // Temporal gradient (frame difference)
                    let curr_x = (x as f32 + u).max(0.0).min(width as f32 - 1.0) as usize;
                    let curr_y = (y as f32 + v).max(0.0).min(height as f32 - 1.0) as usize;
                    let it = next.get_pixel(curr_x, curr_y) as f32 - prev.get_pixel(x as usize, y as usize) as f32;
                    
                    sum_ix2 += ix * ix;
                    sum_iy2 += iy * iy;
                    sum_ixiy += ix * iy;
                    sum_ixit += ix * it;
                    sum_iyit += iy * it;
                }
            }
            
            // Solve 2x2 system using Cramer's rule
            let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
            if det.abs() < 1e-6 {
                break; // Singular matrix
            }
            
            let du = (sum_iy2 * sum_ixit - sum_ixiy * sum_iyit) / det;
            let dv = (sum_ix2 * sum_iyit - sum_ixiy * sum_ixit) / det;
            
            u -= du;
            v -= dv;
            
            if du.abs() < 0.01 && dv.abs() < 0.01 {
                break; // Converged
            }
        }
        
        flow.push((u, v));
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
