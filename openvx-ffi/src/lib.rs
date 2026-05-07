//! OpenVX FFI - C API Wrapper
//!
//! This crate exports all OpenVX C API functions as a dynamic library.
//! Re-exports from openvx-core to create a unified C API.

#![allow(ambiguous_glob_reexports)]

// Re-export all C API functions from openvx-core (includes image, buffer, etc.)
pub use openvx_core::c_api::*;
pub use openvx_core::c_api_data::*;
pub use openvx_core::unified_c_api::*;

// Re-export from buffer and image crates to ensure symbols are included
pub use openvx_buffer::c_api::*;
pub use openvx_image::c_api::*;
