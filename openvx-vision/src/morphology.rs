//! Morphological operations

use crate::utils::{get_pixel_bordered, BorderMode};
use openvx_core::{Context, KernelTrait, Referenceable, VxKernel, VxResult};
use openvx_image::{Image, ImageFormat};

/// Dilate3x3 kernel - max of 3x3 neighborhood
pub struct Dilate3x3Kernel;

impl Dilate3x3Kernel {
    pub fn new() -> Self {
        Dilate3x3Kernel
    }
}

impl KernelTrait for Dilate3x3Kernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.dilate3x3"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Dilate3x3
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

        dilate3x3(src, dst, BorderMode::Constant(0))?;
        Ok(())
    }
}

/// Erode3x3 kernel - min of 3x3 neighborhood
pub struct Erode3x3Kernel;

impl Erode3x3Kernel {
    pub fn new() -> Self {
        Erode3x3Kernel
    }
}

impl KernelTrait for Erode3x3Kernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.erode3x3"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Erode3x3
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

        erode3x3(src, dst, BorderMode::Constant(255))?;
        Ok(())
    }
}

/// Dilate operation - maximum of 3x3 neighborhood
pub fn dilate3x3(src: &Image, dst: &Image, border: BorderMode) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

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
                let mut max_val = src_data[y0 + x0];
                max_val = max_val.max(src_data[y0 + x1]);
                max_val = max_val.max(src_data[y0 + x2]);
                max_val = max_val.max(src_data[y1 + x0]);
                max_val = max_val.max(src_data[y1 + x1]);
                max_val = max_val.max(src_data[y1 + x2]);
                max_val = max_val.max(src_data[y2 + x0]);
                max_val = max_val.max(src_data[y2 + x1]);
                max_val = max_val.max(src_data[y2 + x2]);
                dst_data[dst_row + x1] = max_val;
            }
        }
        return Ok(());
    }

    let mut dst_data = dst.data_mut();

    // For VX_BORDER_UNDEFINED, only process the inner region
    // (exclude 1-pixel border where neighborhood is incomplete)
    let (start_y, end_y, start_x, end_x) = match border {
        BorderMode::Undefined => (1, height.saturating_sub(1), 1, width.saturating_sub(1)),
        _ => (0, height, 0, width),
    };

    for y in start_y..end_y {
        for x in start_x..end_x {
            let mut max_val: u8 = 0;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    let val = get_pixel_bordered(src, px, py, border);
                    max_val = max_val.max(val);
                }
            }

            dst_data[y * width + x] = max_val;
        }
    }

    Ok(())
}

/// Erode operation - minimum of 3x3 neighborhood
pub fn erode3x3(src: &Image, dst: &Image, border: BorderMode) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

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
                let mut min_val = src_data[y0 + x0];
                min_val = min_val.min(src_data[y0 + x1]);
                min_val = min_val.min(src_data[y0 + x2]);
                min_val = min_val.min(src_data[y1 + x0]);
                min_val = min_val.min(src_data[y1 + x1]);
                min_val = min_val.min(src_data[y1 + x2]);
                min_val = min_val.min(src_data[y2 + x0]);
                min_val = min_val.min(src_data[y2 + x1]);
                min_val = min_val.min(src_data[y2 + x2]);
                dst_data[dst_row + x1] = min_val;
            }
        }
        return Ok(());
    }

    let mut dst_data = dst.data_mut();

    // For VX_BORDER_UNDEFINED, only process the inner region
    // (exclude 1-pixel border where neighborhood is incomplete)
    let (start_y, end_y, start_x, end_x) = match border {
        BorderMode::Undefined => (1, height.saturating_sub(1), 1, width.saturating_sub(1)),
        _ => (0, height, 0, width),
    };

    for y in start_y..end_y {
        for x in start_x..end_x {
            let mut min_val: u8 = 255;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    let val = get_pixel_bordered(src, px, py, border);
                    min_val = min_val.min(val);
                }
            }

            dst_data[y * width + x] = min_val;
        }
    }

    Ok(())
}

/// Opening: erosion followed by dilation
pub fn opening3x3(src: &Image, dst: &Image, border: BorderMode) -> VxResult<()> {
    // Create temporary image
    let temp = Image::new(src.width(), src.height(), ImageFormat::Gray);
    erode3x3(src, &temp, border)?;
    dilate3x3(&temp, dst, border)?;
    Ok(())
}

/// Closing: dilation followed by erosion
pub fn closing3x3(src: &Image, dst: &Image, border: BorderMode) -> VxResult<()> {
    // Create temporary image
    let temp = Image::new(src.width(), src.height(), ImageFormat::Gray);
    dilate3x3(src, &temp, border)?;
    erode3x3(&temp, dst, border)?;
    Ok(())
}

/// Morphological gradient: dilate - erode
pub fn morphological_gradient(src: &Image, dst: &Image, border: BorderMode) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let mut min_val: u8 = 255;
            let mut max_val: u8 = 0;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    let val = get_pixel_bordered(src, px, py, border);
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }

            dst_data[y * width + x] = max_val.saturating_sub(min_val);
        }
    }

    Ok(())
}
