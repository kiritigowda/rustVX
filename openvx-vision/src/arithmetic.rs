//! Arithmetic operations

use openvx_core::{Context, KernelTrait, Referenceable, VxKernel, VxResult};
use openvx_image::Image;

/// Add kernel - pixel-wise addition with overflow policy
pub struct AddKernel;

impl AddKernel {
    pub fn new() -> Self {
        AddKernel
    }
}

impl KernelTrait for AddKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.add"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Add
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        add(src1, src2, dst)?;
        Ok(())
    }
}

/// Subtract kernel
pub struct SubtractKernel;

impl SubtractKernel {
    pub fn new() -> Self {
        SubtractKernel
    }
}

impl KernelTrait for SubtractKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.subtract"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Subtract
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        subtract(src1, src2, dst)?;
        Ok(())
    }
}

/// Multiply kernel - with scale factor support
pub struct MultiplyKernel;

impl MultiplyKernel {
    pub fn new() -> Self {
        MultiplyKernel
    }
}

impl KernelTrait for MultiplyKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.multiply"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Multiply
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        multiply(src1, src2, dst, 1.0)?;
        Ok(())
    }
}

/// WeightedAverage kernel - (src1 * w1 + src2 * w2) / 256
pub struct WeightedAverageKernel;

impl WeightedAverageKernel {
    pub fn new() -> Self {
        WeightedAverageKernel
    }
}

impl KernelTrait for WeightedAverageKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.weighted_average"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::WeightedAverage
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        weighted(src1, src2, dst, 0.5)?;
        Ok(())
    }
}

/// Pixel-wise addition with saturation
pub fn add(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();

    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as u16;
            let b = src2.get_pixel(x, y) as u16;
            let sum = a + b;
            dst_data[y * width + x] = sum.min(255) as u8;
        }
    }

    Ok(())
}

/// Pixel-wise subtraction with saturation
pub fn subtract(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();

    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as i16;
            let b = src2.get_pixel(x, y) as i16;
            let diff = a - b;
            dst_data[y * width + x] = diff.max(0).min(255) as u8;
        }
    }

    Ok(())
}

/// Pixel-wise multiplication with scale factor
pub fn multiply(src1: &Image, src2: &Image, dst: &Image, scale: f32) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();

    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as f32;
            let b = src2.get_pixel(x, y) as f32;
            let product = a * b * scale / 255.0;
            dst_data[y * width + x] = product.max(0.0).min(255.0) as u8;
        }
    }

    Ok(())
}

/// Weighted average: (src1 * w1 + src2 * w2) / 256
pub fn weighted(src1: &Image, src2: &Image, dst: &Image, alpha_f32: f32) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();
    let alpha_w = alpha_f32;
    let beta_w = 1.0 - alpha_f32;

    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y) as f32;
            let b = src2.get_pixel(x, y) as f32;
            let result = alpha_w * a + beta_w * b;
            // OpenVX spec: truncate towards zero (C-style cast)
            dst_data[y * width + x] = result as i32 as u8;
        }
    }

    Ok(())
}

/// Bitwise AND between two images
pub fn and(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y);
            let b = src2.get_pixel(x, y);
            dst_data[y * width + x] = a & b;
        }
    }

    Ok(())
}

/// Bitwise OR between two images
pub fn or(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y);
            let b = src2.get_pixel(x, y);
            dst_data[y * width + x] = a | b;
        }
    }

    Ok(())
}

/// Bitwise XOR between two images
pub fn xor(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y);
            let b = src2.get_pixel(x, y);
            dst_data[y * width + x] = a ^ b;
        }
    }

    Ok(())
}

/// Pixel-wise minimum (Enhanced Vision: `vxMin`).
///
/// `src1`, `src2`, and `dst` must share dimensions; per the OpenVX 1.3 spec
/// the format must also match (`VX_DF_IMAGE_U8` only is supported by this
/// path — the immediate-mode `vxuMin` handles `VX_DF_IMAGE_S16` directly via
/// `openvx_core::vxu_impl::vxu_min_impl`).
pub fn min_image(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y);
            let b = src2.get_pixel(x, y);
            dst_data[y * width + x] = a.min(b);
        }
    }

    Ok(())
}

/// Pixel-wise maximum (Enhanced Vision: `vxMax`). See `min_image` for the
/// dimension/format contract.
pub fn max_image(src1: &Image, src2: &Image, dst: &Image) -> VxResult<()> {
    if src1.width() != src2.width() || src1.height() != src2.height() {
        return Err(openvx_core::VxStatus::ErrorInvalidDimension);
    }

    let width = src1.width();
    let height = src1.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src1.get_pixel(x, y);
            let b = src2.get_pixel(x, y);
            dst_data[y * width + x] = a.max(b);
        }
    }

    Ok(())
}

/// `MinKernel` — Enhanced Vision pixel-wise minimum kernel.
pub struct MinKernel;

impl MinKernel {
    pub fn new() -> Self {
        MinKernel
    }
}

impl KernelTrait for MinKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.min"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Min
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        min_image(src1, src2, dst)?;
        Ok(())
    }
}

/// `MaxKernel` — Enhanced Vision pixel-wise maximum kernel.
pub struct MaxKernel;

impl MaxKernel {
    pub fn new() -> Self {
        MaxKernel
    }
}

impl KernelTrait for MaxKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.max"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Max
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        max_image(src1, src2, dst)?;
        Ok(())
    }
}

/// Bitwise NOT (complement) of an image
pub fn not(src: &Image, dst: &Image) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let a = src.get_pixel(x, y);
            dst_data[y * width + x] = !a;
        }
    }

    Ok(())
}

/// And kernel - bitwise AND between two images
pub struct AndKernel;

impl AndKernel {
    pub fn new() -> Self {
        AndKernel
    }
}

impl KernelTrait for AndKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.and"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::And
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        and(src1, src2, dst)?;
        Ok(())
    }
}

/// Or kernel - bitwise OR between two images
pub struct OrKernel;

impl OrKernel {
    pub fn new() -> Self {
        OrKernel
    }
}

impl KernelTrait for OrKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.or"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Or
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        or(src1, src2, dst)?;
        Ok(())
    }
}

/// Xor kernel - bitwise XOR between two images
pub struct XorKernel;

impl XorKernel {
    pub fn new() -> Self {
        XorKernel
    }
}

impl KernelTrait for XorKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.xor"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Xor
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src1 = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let src2 = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        xor(src1, src2, dst)?;
        Ok(())
    }
}

/// Not kernel - bitwise NOT of an image
pub struct NotKernel;

impl NotKernel {
    pub fn new() -> Self {
        NotKernel
    }
}

impl KernelTrait for NotKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.not"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Not
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

        not(src, dst)?;
        Ok(())
    }
}
