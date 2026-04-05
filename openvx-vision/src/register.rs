//! Vision kernel registration in global KERNELS registry
//!
//! This module provides functions to register vision kernels in the global
//! KERNELS registry used by the C API (vxGetKernelByName, vxQueryKernel, etc.)

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::kernel_enums::VISION_KERNELS;

/// Register all vision kernels in the global KERNELS registry
///
/// This function should be called once during context creation to ensure
/// that vision kernels are accessible via vxGetKernelByName and vxQueryKernel.
pub fn register_vision_kernels_in_global_registry() {
    // Use the unified_c_api's KERNELS registry
    use openvx_core::unified_c_api::KERNELS;
    
    if let Ok(mut kernels) = KERNELS.lock() {
        // Register vision kernels using their correct enum values as keys
        // This ensures vxGetKernelByEnum returns the correct kernel
        for (name, kernel_enum, _num_params) in VISION_KERNELS.iter() {
            let kernel_id = *kernel_enum as u64;
            
            // Only register if not already present
            if !kernels.contains_key(&kernel_id) {
                let kernel_data = Arc::new(openvx_core::unified_c_api::VxCKernel::new(
                    *kernel_enum,
                    name.to_string(),
                ));
                
                kernels.insert(kernel_id, kernel_data);
            }
        }
    }
}

/// Check if a kernel name is a vision kernel
pub fn is_vision_kernel(name: &str) -> bool {
    VISION_KERNELS.iter().any(|(n, _, _)| *n == name)
}

/// Get the enum value for a vision kernel by name
pub fn get_vision_kernel_enum(name: &str) -> Option<i32> {
    VISION_KERNELS.iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, e, _)| *e)
}

/// Get the number of parameters for a vision kernel by name
pub fn get_vision_kernel_num_params(name: &str) -> Option<u32> {
    VISION_KERNELS.iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, _, p)| *p)
}
