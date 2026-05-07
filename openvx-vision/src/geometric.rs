//! Geometric operations

use crate::utils::{bilinear_interpolate, nearest_interpolate};
use openvx_core::{Context, KernelTrait, Referenceable, VxKernel, VxResult};
use openvx_image::Image;

/// ScaleImage kernel - bilinear/nearest interpolation
pub struct ScaleImageKernel;

impl ScaleImageKernel {
    pub fn new() -> Self {
        ScaleImageKernel
    }
}

impl KernelTrait for ScaleImageKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.scale_image"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::ScaleImage
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

        scale_image(src, dst, true)?;
        Ok(())
    }
}

/// WarpAffine kernel - 2x3 matrix transformation
pub struct WarpAffineKernel;

impl WarpAffineKernel {
    pub fn new() -> Self {
        WarpAffineKernel
    }
}

impl KernelTrait for WarpAffineKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.warp_affine"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::WarpAffine
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let matrix_ref = params
            .get(1)
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        // Extract matrix data from the matrix reference
        let matrix: [f32; 6] = if let Some(matrix_data) = matrix_ref
            .as_any()
            .downcast_ref::<openvx_core::c_api_data::VxCMatrixData>(
        ) {
            matrix_data
                .as_f32_slice()
                .and_then(|v| v.try_into().ok())
                .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        } else {
            // Default identity affine transform
            [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]
        };

        warp_affine(src, &matrix, dst)?;
        Ok(())
    }
}

/// WarpPerspective kernel - 3x3 matrix transformation
pub struct WarpPerspectiveKernel;

impl WarpPerspectiveKernel {
    pub fn new() -> Self {
        WarpPerspectiveKernel
    }
}

impl KernelTrait for WarpPerspectiveKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.warp_perspective"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::WarpPerspective
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let matrix_ref = params
            .get(1)
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        // Extract matrix data from the matrix reference
        let matrix: [f32; 9] = if let Some(matrix_data) = matrix_ref
            .as_any()
            .downcast_ref::<openvx_core::c_api_data::VxCMatrixData>(
        ) {
            if matrix_data.data_type != 0x00A || matrix_data.rows != 3 || matrix_data.columns != 3 {
                return Err(openvx_core::VxStatus::ErrorInvalidParameters);
            }
            matrix_data
                .as_f32_slice()
                .and_then(|v| v.try_into().ok())
                .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        } else {
            // Default identity perspective transform if matrix not provided
            [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        };

        warp_perspective(src, &matrix, dst)?;
        Ok(())
    }
}

/// Scale image using bilinear or nearest neighbor interpolation
pub fn scale_image(src: &Image, dst: &Image, bilinear: bool) -> VxResult<()> {
    let src_width = src.width();
    let src_height = src.height();
    let dst_width = dst.width();
    let dst_height = dst.height();

    let mut dst_data = dst.data_mut();

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = (x as f32 + 0.5) * x_scale - 0.5;
            let src_y = (y as f32 + 0.5) * y_scale - 0.5;

            let value = if bilinear {
                bilinear_interpolate(src, src_x, src_y)
            } else {
                nearest_interpolate(src, src_x, src_y)
            };

            dst_data[y * dst_width + x] = value;
        }
    }

    Ok(())
}

/// Warp affine: dst(x,y) = src(A*[x,y,1])
/// CTS stores affine matrix as mat[col][row] with col\u{220}\\{0,1,2\}, row\u{220}\\{0,1\}
/// Flat layout: m[col*2 + row], so:
///   m[0]=col0_row0 (x-coeff of x), m[1]=col0_row1 (y-coeff of x)
///   m[2]=col1_row0 (x-coeff of y), m[3]=col1_row1 (y-coeff of y)
///   m[4]=col2_row0 (x-translation),  m[5]=col2_row1 (y-translation)
/// CTS reference: x0 = m[0]*x + m[2]*y + m[4]
///               y0 = m[1]*x + m[3]*y + m[5]
pub fn warp_affine(src: &Image, matrix: &[f32; 6], dst: &Image) -> VxResult<()> {
    let dst_width = dst.width();
    let dst_height = dst.height();
    let src_width = src.width() as f32;
    let src_height = src.height() as f32;

    let mut dst_data = dst.data_mut();

    // Correct CTS column-major layout
    let a11 = matrix[0]; // x-coeff of x
    let a12 = matrix[2]; // x-coeff of y
    let a13 = matrix[4]; // x-translation
    let a21 = matrix[1]; // y-coeff of x
    let a22 = matrix[3]; // y-coeff of y
    let a23 = matrix[5]; // y-translation

    for y in 0..dst_height {
        for x in 0..dst_width {
            let xf = x as f32;
            let yf = y as f32;

            // Inverse mapping: find source coordinate for this destination pixel
            let src_x = a11 * xf + a12 * yf + a13;
            let src_y = a21 * xf + a22 * yf + a23;

            // Check bounds
            if src_x < 0.0 || src_x >= src_width || src_y < 0.0 || src_y >= src_height {
                dst_data[y * dst_width + x] = 0;
                continue;
            }

            dst_data[y * dst_width + x] = bilinear_interpolate(src, src_x, src_y);
        }
    }

    Ok(())
}

/// Warp perspective: dst(x,y) = src(H*[x,y,1] / w)
/// matrix is 3x3 in COLUMN-MAJOR order (OpenVX standard):
///   m[0]=h00, m[1]=h10, m[2]=h20 (first column)
///   m[3]=h01, m[4]=h11, m[5]=h21 (second column)
///   m[6]=h02, m[7]=h12, m[8]=h22 (third column)
pub fn warp_perspective(src: &Image, matrix: &[f32; 9], dst: &Image) -> VxResult<()> {
    let dst_width = dst.width();
    let dst_height = dst.height();
    let src_width = src.width() as f32;
    let src_height = src.height() as f32;

    let mut dst_data = dst.data_mut();

    // Column-major access pattern as used by OpenVX test:
    // x_h = m[0]*x + m[3]*y + m[6]
    // y_h = m[1]*x + m[4]*y + m[7]
    // w_h = m[2]*x + m[5]*y + m[8]
    let h00 = matrix[0]; // m[0,0] in row 0, col 0
    let h10 = matrix[1]; // m[1,0] in row 1, col 0
    let h20 = matrix[2]; // m[2,0] in row 2, col 0
    let h01 = matrix[3]; // m[0,1] in row 0, col 1
    let h11 = matrix[4]; // m[1,1] in row 1, col 1
    let h21 = matrix[5]; // m[2,1] in row 2, col 1
    let h02 = matrix[6]; // m[0,2] in row 0, col 2
    let h12 = matrix[7]; // m[1,2] in row 1, col 2
    let h22 = matrix[8]; // m[2,2] in row 2, col 2

    for y in 0..dst_height {
        for x in 0..dst_width {
            let xf = x as f32;
            let yf = y as f32;

            // Homogeneous coordinates using column-major access
            let x_h = h00 * xf + h01 * yf + h02;
            let y_h = h10 * xf + h11 * yf + h12;
            let w_h = h20 * xf + h21 * yf + h22;

            if w_h.abs() < 1e-6 {
                dst_data[y * dst_width + x] = 0;
                continue;
            }

            let src_x = x_h / w_h;
            let src_y = y_h / w_h;

            // Check bounds
            if src_x < 0.0 || src_x >= src_width || src_y < 0.0 || src_y >= src_height {
                dst_data[y * dst_width + x] = 0;
                continue;
            }

            dst_data[y * dst_width + x] = bilinear_interpolate(src, src_x, src_y);
        }
    }

    Ok(())
}

/// Remap: generic coordinate mapping
pub fn remap(src: &Image, map_x: &[f32], map_y: &[f32], dst: &Image) -> VxResult<()> {
    let dst_width = dst.width();
    let dst_height = dst.height();

    let mut dst_data = dst.data_mut();

    for y in 0..dst_height {
        for x in 0..dst_width {
            let idx = y * dst_width + x;
            let src_x = map_x[idx];
            let src_y = map_y[idx];

            dst_data[idx] = bilinear_interpolate(src, src_x, src_y);
        }
    }

    Ok(())
}
