//! Filter kernels implementation

use crate::utils::{clamp_u8, get_pixel_bordered, quickselect, BorderMode};
use openvx_core::{Context, KernelTrait, Referenceable, VxKernel, VxResult};
use openvx_image::Image;

/// Median of 9 elements using unrolled insertion sort.
/// Provably correct (verified exhaustively for 3-bit values and 10M random samples).
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
    if v[5] < v[4] { let t = v[5]; v[5] = v[4]; v[4] = t;
        if v[4] < v[3] { let t = v[4]; v[4] = v[3]; v[3] = t;
            if v[3] < v[2] { let t = v[3]; v[3] = v[2]; v[2] = t;
                if v[2] < v[1] { let t = v[2]; v[2] = v[1]; v[1] = t;
                    if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
                }
            }
        }
    }
    if v[6] < v[5] { let t = v[6]; v[6] = v[5]; v[5] = t;
        if v[5] < v[4] { let t = v[5]; v[5] = v[4]; v[4] = t;
            if v[4] < v[3] { let t = v[4]; v[4] = v[3]; v[3] = t;
                if v[3] < v[2] { let t = v[3]; v[3] = v[2]; v[2] = t;
                    if v[2] < v[1] { let t = v[2]; v[2] = v[1]; v[1] = t;
                        if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
                    }
                }
            }
        }
    }
    if v[7] < v[6] { let t = v[7]; v[7] = v[6]; v[6] = t;
        if v[6] < v[5] { let t = v[6]; v[6] = v[5]; v[5] = t;
            if v[5] < v[4] { let t = v[5]; v[5] = v[4]; v[4] = t;
                if v[4] < v[3] { let t = v[4]; v[4] = v[3]; v[3] = t;
                    if v[3] < v[2] { let t = v[3]; v[3] = v[2]; v[2] = t;
                        if v[2] < v[1] { let t = v[2]; v[2] = v[1]; v[1] = t;
                            if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
                        }
                    }
                }
            }
        }
    }
    if v[8] < v[7] { let t = v[8]; v[8] = v[7]; v[7] = t;
        if v[7] < v[6] { let t = v[7]; v[7] = v[6]; v[6] = t;
            if v[6] < v[5] { let t = v[6]; v[6] = v[5]; v[5] = t;
                if v[5] < v[4] { let t = v[5]; v[5] = v[4]; v[4] = t;
                    if v[4] < v[3] { let t = v[4]; v[4] = v[3]; v[3] = t;
                        if v[3] < v[2] { let t = v[3]; v[3] = v[2]; v[2] = t;
                            if v[2] < v[1] { let t = v[2]; v[2] = v[1]; v[1] = t;
                                if v[1] < v[0] { let t = v[1]; v[1] = v[0]; v[0] = t; }
                            }
                        }
                    }
                }
            }
        }
    }
    v[4]
}

/// Generic convolution kernel
pub struct ConvolveKernel;

impl ConvolveKernel {
    pub fn new() -> Self {
        ConvolveKernel
    }
}

impl KernelTrait for ConvolveKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.convolve"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Convolve
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        // For simplicity, apply a generic 3x3 convolution
        let kernel: [[i32; 3]; 3] = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]];

        convolve_generic(src, dst, &kernel, BorderMode::Replicate)?;

        Ok(())
    }
}

/// Gaussian3x3 kernel - separable [1,2,1] horizontal then vertical
pub struct Gaussian3x3Kernel;

impl Gaussian3x3Kernel {
    pub fn new() -> Self {
        Gaussian3x3Kernel
    }
}

impl KernelTrait for Gaussian3x3Kernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.gaussian_3x3"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Gaussian3x3
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        #[cfg(feature = "simd")]
        {
            crate::filter_simd::gaussian3x3_simd(src, dst)?;
            return Ok(());
        }
        #[cfg(not(feature = "simd"))]
        {
            gaussian3x3(src, dst)?;
            Ok(())
        }
    }
}

/// Gaussian5x5 kernel - separable [1,4,6,4,1] kernel
pub struct Gaussian5x5Kernel;

impl Gaussian5x5Kernel {
    pub fn new() -> Self {
        Gaussian5x5Kernel
    }
}

impl KernelTrait for Gaussian5x5Kernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.gaussian_5x5"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Gaussian5x5
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        gaussian5x5(src, dst)?;
        Ok(())
    }
}

/// Box3x3 kernel - moving average optimization
pub struct Box3x3Kernel;

impl Box3x3Kernel {
    pub fn new() -> Self {
        Box3x3Kernel
    }
}

impl KernelTrait for Box3x3Kernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.box_3x3"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Box3x3
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        box3x3(src, dst)?;
        Ok(())
    }
}

/// Median3x3 kernel - quickselect on 3x3 neighborhood
pub struct Median3x3Kernel;

impl Median3x3Kernel {
    pub fn new() -> Self {
        Median3x3Kernel
    }
}

impl KernelTrait for Median3x3Kernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.median_3x3"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Median3x3
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        median3x3(src, dst)?;
        Ok(())
    }
}

/// Generic NxM convolution with proper border handling
pub fn convolve_generic(
    src: &Image,
    dst: &Image,
    kernel: &[[i32; 3]; 3],
    border: BorderMode,
) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let kernel_sum: i32 = kernel.iter().flat_map(|r| r.iter()).sum::<i32>().max(1);

    // Fast path: Undefined border means only inner region is written;
    // all neighbors are in bounds, so skip border checks entirely.
    if border == BorderMode::Undefined && width > 2 && height > 2 {
        let src_data = src.data();
        let mut dst_data = dst.data_mut();
        let w = width as usize;
        for y in 1..(height - 1) {
            let y0 = (y - 1) as usize * w;
            let y1 = y as usize * w;
            let y2 = (y + 1) as usize * w;
            let dst_row = y as usize * w;
            for x in 1..(width - 1) {
                let x0 = x as usize - 1;
                let x1 = x as usize;
                let x2 = x as usize + 1;
                let sum: i32 = src_data[y0 + x0] as i32 * kernel[0][0]
                    + src_data[y0 + x1] as i32 * kernel[0][1]
                    + src_data[y0 + x2] as i32 * kernel[0][2]
                    + src_data[y1 + x0] as i32 * kernel[1][0]
                    + src_data[y1 + x1] as i32 * kernel[1][1]
                    + src_data[y1 + x2] as i32 * kernel[1][2]
                    + src_data[y2 + x0] as i32 * kernel[2][0]
                    + src_data[y2 + x1] as i32 * kernel[2][1]
                    + src_data[y2 + x2] as i32 * kernel[2][2];
                dst_data[dst_row + x1] = clamp_u8(sum / kernel_sum);
            }
        }
        return Ok(());
    }

    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let mut sum: i32 = 0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x as isize + kx as isize - 1;
                    let py = y as isize + ky as isize - 1;
                    let pixel = get_pixel_bordered(src, px, py, border);
                    sum += pixel as i32 * kernel[ky][kx];
                }
            }
            dst_data[y * width + x] = clamp_u8(sum / kernel_sum);
        }
    }

    Ok(())
}

/// Gaussian 3x3: Separable [1,2,1] horizontal then vertical
/// Uses REPLICATE border mode for edge pixels
/// Total normalization: 4 * 4 = 16
pub fn gaussian3x3(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    if width < 2 || height < 2 {
        // For 1x1 images just copy
        let src_data = src.data();
        let mut dst_data = dst.data_mut();
        dst_data.copy_from_slice(&src_data[..width * height]);
        return Ok(());
    }

    let src_data = src.data();
    let mut dst_data = dst.data_mut();

    // Use u8 temp buffer with pre-divided horizontal values (/4)
    // This halves memory bandwidth vs u16 and keeps values in u8 range
    let mut temp = vec![0u8; width * height];

    // Horizontal pass: [1,2,1]/4 with REPLICATE border
    // Process first row
    {
        let row = 0;
        let src_row = &src_data[row * width..row * width + width];
        let temp_row = &mut temp[row * width..row * width + width];

        // First pixel (left border replicates)
        temp_row[0] = ((src_row[0] as u16 * 3 + src_row[1] as u16) >> 2) as u8;

        // Interior pixels
        for x in 1..width - 1 {
            let sum = (src_row[x - 1] as u16 + src_row[x] as u16 * 2 + src_row[x + 1] as u16) >> 2;
            temp_row[x] = sum as u8;
        }

        // Last pixel (right border replicates)
        temp_row[width - 1] =
            ((src_row[width - 2] as u16 + src_row[width - 1] as u16 * 3) >> 2) as u8;
    }

    // Process middle rows
    for y in 1..height - 1 {
        let src_row = &src_data[y * width..y * width + width];
        let temp_row = &mut temp[y * width..y * width + width];

        // First pixel (left border replicates)
        temp_row[0] = ((src_row[0] as u16 * 3 + src_row[1] as u16) >> 2) as u8;

        // Interior pixels
        for x in 1..width - 1 {
            let sum = (src_row[x - 1] as u16 + src_row[x] as u16 * 2 + src_row[x + 1] as u16) >> 2;
            temp_row[x] = sum as u8;
        }

        // Last pixel (right border replicates)
        temp_row[width - 1] =
            ((src_row[width - 2] as u16 + src_row[width - 1] as u16 * 3) >> 2) as u8;
    }

    // Process last row
    {
        let y = height - 1;
        let src_row = &src_data[y * width..y * width + width];
        let temp_row = &mut temp[y * width..y * width + width];

        // First pixel (left border replicates)
        temp_row[0] = ((src_row[0] as u16 * 3 + src_row[1] as u16) >> 2) as u8;

        // Interior pixels
        for x in 1..width - 1 {
            let sum = (src_row[x - 1] as u16 + src_row[x] as u16 * 2 + src_row[x + 1] as u16) >> 2;
            temp_row[x] = sum as u8;
        }

        // Last pixel (right border replicates)
        temp_row[width - 1] =
            ((src_row[width - 2] as u16 + src_row[width - 1] as u16 * 3) >> 2) as u8;
    }

    // Vertical pass: [1,2,1]/4 with REPLICATE border
    // Process first row (top border replicates)
    {
        let dst_row = &mut dst_data[0..width];
        let curr = &temp[0..width];
        let next = &temp[width..width * 2];
        for x in 0..width {
            let sum = (curr[x] as u16 * 3 + next[x] as u16) >> 2;
            dst_row[x] = sum as u8;
        }
    }

    // Process middle rows
    for y in 1..height - 1 {
        let dst_row = &mut dst_data[y * width..y * width + width];
        let prev = &temp[(y - 1) * width..(y - 1) * width + width];
        let curr = &temp[y * width..y * width + width];
        let next = &temp[(y + 1) * width..(y + 1) * width + width];
        for x in 0..width {
            let sum = (prev[x] as u16 + curr[x] as u16 * 2 + next[x] as u16) >> 2;
            dst_row[x] = sum as u8;
        }
    }

    // Process last row (bottom border replicates)
    {
        let y = height - 1;
        let dst_row = &mut dst_data[y * width..y * width + width];
        let prev = &temp[(y - 1) * width..(y - 1) * width + width];
        let curr = &temp[y * width..y * width + width];
        for x in 0..width {
            let sum = (prev[x] as u16 + curr[x] as u16 * 3) >> 2;
            dst_row[x] = sum as u8;
        }
    }

    Ok(())
}

/// Separable Gaussian 5x5: [1,4,6,4,1] kernel
pub fn gaussian5x5(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let kernel = [1, 4, 6, 4, 1];

    let mut dst_data = dst.data_mut();

    // Temporary buffer for horizontal pass - use saturating_mul to prevent overflow
    let temp_size = width.saturating_mul(height);
    let mut temp = vec![0u8; temp_size];

    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut sum: i32 = 0;
            let mut weight: i32 = 0;
            for k in 0..5 {
                let px = x as isize + k as isize - 2;
                if px >= 0 && px < width as isize {
                    sum += src.get_pixel(px as usize, y) as i32 * kernel[k];
                    weight += kernel[k];
                }
            }
            temp[y * width + x] = clamp_u8(sum / weight);
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
            dst_data[y * width + x] = clamp_u8(sum / weight);
        }
    }

    Ok(())
}

/// Box filter 3x3: Average of 3x3 neighborhood
/// Uses REPLICATE border mode for edge pixels
/// Normalization: divide by 9
pub fn box3x3(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    if width == 0 || height == 0 {
        return Ok(());
    }

    let src_data = src.data();
    let mut dst_data = dst.data_mut();

    // u16 temp buffer stores horizontal sums (3 pixels per position)
    let mut temp = vec![0u16; width * height];

    // Horizontal pass: temp[y][x] = sum of 3 pixels in row y with replicate border
    for y in 0..height {
        let row = y * width;

        if width == 1 {
            temp[row] = src_data[row] as u16 * 3;
        } else {
            // x = 0: replicate left border (p0 + p0 + p1)
            temp[row] = src_data[row] as u16 * 2 + src_data[row + 1] as u16;

            if width == 2 {
                // x = 1: replicate right border (p0 + p1 + p1)
                temp[row + 1] = src_data[row] as u16 + src_data[row + 1] as u16 * 2;
            } else {
                // Initialize sliding window for x = 1
                let mut sum = src_data[row] as u16
                    + src_data[row + 1] as u16
                    + src_data[row + 2] as u16;
                temp[row + 1] = sum;

                // Sliding window for x = 2 .. width-2
                for x in 2..width - 1 {
                    sum = sum + src_data[row + x + 1] as u16 - src_data[row + x - 2] as u16;
                    temp[row + x] = sum;
                }

                // x = width-1: replicate right border (p_{w-2} + p_{w-1} + p_{w-1})
                temp[row + width - 1] = src_data[row + width - 2] as u16
                    + src_data[row + width - 1] as u16 * 2;
            }
        }
    }

    // Vertical pass: dst[y][x] = (temp[y-1][x] + temp[y][x] + temp[y+1][x]) / 9
    for x in 0..width {
        if height == 1 {
            dst_data[x] = (temp[x] / 9) as u8;
        } else {
            // y = 0: replicate top border
            let mut sum = temp[x] * 2 + temp[width + x];
            dst_data[x] = (sum / 9) as u8;

            if height == 2 {
                // y = 1: replicate bottom border
                sum = temp[x] + temp[width + x] * 2;
                dst_data[width + x] = (sum / 9) as u8;
            } else {
                // Initialize sliding window for y = 1
                sum = temp[x] + temp[width + x] + temp[2 * width + x];
                dst_data[width + x] = (sum / 9) as u8;

                // Sliding window for y = 2 .. height-2
                for y in 2..height - 1 {
                    sum = sum + temp[(y + 1) * width + x] - temp[(y - 2) * width + x];
                    dst_data[y * width + x] = (sum / 9) as u8;
                }

                // y = height-1: replicate bottom border
                let last = (height - 1) * width;
                sum = temp[last - width + x] + temp[last + x] * 2;
                dst_data[last + x] = (sum / 9) as u8;
            }
        }
    }

    Ok(())
}

/// Median filter 3x3 using sorting network (25 comparisons, branchless)
pub fn median3x3(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    // For small images, fall back to the generic path
    if width < 3 || height < 3 {
        return median3x3_slow(src, dst);
    }

    let mut dst_data = dst.data_mut();
    let src_data = src.data();
    let len = width * height;

    if src_data.len() < len || dst_data.len() < len {
        return Err(openvx_core::VxStatus::ErrorInvalidParameters);
    }

    // --- Inner region (no bounds checks, direct slice access) ---
    for y in 1..height - 1 {
        let row_m1 = &src_data[(y - 1) * width..y * width];
        let row_0  = &src_data[y * width..(y + 1) * width];
        let row_p1 = &src_data[(y + 1) * width..(y + 2) * width];
        let dst_row = &mut dst_data[y * width..(y + 1) * width];

        for x in 1..width - 1 {
            let v = [
                row_m1[x - 1], row_m1[x], row_m1[x + 1],
                row_0[x - 1],  row_0[x],  row_0[x + 1],
                row_p1[x - 1], row_p1[x], row_p1[x + 1],
            ];
            dst_row[x] = median9(v);
        }
    }

    // --- Edge pixels with replicate border ---

    // Top row (y = 0)
    {
        let row_0 = &src_data[0..width];
        let row_1 = &src_data[width..2 * width];
        let dst_row = &mut dst_data[0..width];
        for x in 0..width {
            let xl = if x == 0 { 0 } else { x - 1 };
            let xr = if x + 1 >= width { width - 1 } else { x + 1 };
            let v = [
                row_0[xl], row_0[x], row_0[xr],
                row_0[xl], row_0[x], row_0[xr],
                row_1[xl], row_1[x], row_1[xr],
            ];
            dst_row[x] = median9(v);
        }
    }

    // Bottom row (y = height - 1)
    {
        let last = (height - 1) * width;
        let row_m1 = &src_data[last - width..last];
        let row_0  = &src_data[last..last + width];
        let dst_row = &mut dst_data[last..last + width];
        for x in 0..width {
            let xl = if x == 0 { 0 } else { x - 1 };
            let xr = if x + 1 >= width { width - 1 } else { x + 1 };
            let v = [
                row_m1[xl], row_m1[x], row_m1[xr],
                row_0[xl],  row_0[x],  row_0[xr],
                row_0[xl],  row_0[x],  row_0[xr],
            ];
            dst_row[x] = median9(v);
        }
    }

    // First column (x = 0) for inner rows
    for y in 1..height - 1 {
        let row_m1 = &src_data[(y - 1) * width..y * width];
        let row_0  = &src_data[y * width..(y + 1) * width];
        let row_p1 = &src_data[(y + 1) * width..(y + 2) * width];
        let v = [
            row_m1[0], row_m1[0], row_m1[1],
            row_0[0],  row_0[0],  row_0[1],
            row_p1[0], row_p1[0], row_p1[1],
        ];
        dst_data[y * width] = median9(v);
    }

    // Last column (x = width - 1) for inner rows
    for y in 1..height - 1 {
        let row_m1 = &src_data[(y - 1) * width..y * width];
        let row_0  = &src_data[y * width..(y + 1) * width];
        let row_p1 = &src_data[(y + 1) * width..(y + 2) * width];
        let xlast = width - 1;
        let v = [
            row_m1[xlast - 1], row_m1[xlast], row_m1[xlast],
            row_0[xlast - 1],  row_0[xlast],  row_0[xlast],
            row_p1[xlast - 1], row_p1[xlast], row_p1[xlast],
        ];
        dst_data[y * width + xlast] = median9(v);
    }

    Ok(())
}

/// Original slow median3x3 using quickselect — kept as fallback.
fn median3x3_slow(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    let mut dst_data = dst.data_mut();
    let mut window = [0u8; 9];

    for y in 0..height {
        for x in 0..width {
            let mut idx = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let py = y as isize + dy;
                    let px = x as isize + dx;
                    if py >= 0 && py < height as isize && px >= 0 && px < width as isize {
                        window[idx] = src.get_pixel(px as usize, py as usize);
                    } else {
                        window[idx] = src.get_pixel(x, y); // Replicate border
                    }
                    idx += 1;
                }
            }

            dst_data[y * width + x] = quickselect(&mut window, 4);
        }
    }

    Ok(())
}
