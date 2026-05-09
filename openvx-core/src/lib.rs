//! OpenVX Core Framework

pub mod c_api;
pub mod c_api_data;
pub mod context;
pub mod reference;
pub mod simd_kernels;
pub mod types;
pub mod unified_c_api;
pub mod vxu_impl;

pub use c_api::vx_status;
pub use context::{Context, KernelTrait};
pub use reference::{Ref, Referenceable};
pub use types::{VxBorderMode, VxError, VxImageFormat, VxKernel, VxResult, VxStatus, VxType};
pub use unified_c_api::VxCScalar;
