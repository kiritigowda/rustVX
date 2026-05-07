//! Context

#![allow(dead_code)]

use crate::reference::{ReferenceTable, Referenceable};
use crate::types::{VxResult, VxType};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Context for OpenVX operations
pub struct Context {
    id: u32,
    reference_table: ReferenceTable,
    vendor_id: u32,
    version: (u32, u32),
    implementation: String,
}

impl Context {
    /// Create a new OpenVX context
    pub fn new() -> VxResult<Arc<Self>> {
        static CONTEXT_ID: AtomicU32 = AtomicU32::new(1);

        let context = Arc::new(Context {
            id: CONTEXT_ID.fetch_add(1, Ordering::SeqCst),
            reference_table: ReferenceTable::new(),
            vendor_id: 0xFFFF,
            version: (1, 3),
            implementation: "OpenVX Rust Implementation".to_string(),
        });

        Ok(context)
    }

    pub fn get_id(&self) -> u32 {
        self.id
    }

    pub fn generate_reference_id(&self) -> u64 {
        self.reference_table.generate_id()
    }

    pub fn get_num_unique_kernels(&self) -> u32 {
        0
    }

    pub fn register_kernel(&self, _kernel: Box<dyn KernelTrait>) -> VxResult<()> {
        Ok(())
    }
}

impl Referenceable for Context {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn get_type(&self) -> VxType {
        VxType::Context
    }

    fn get_reference_count(&self) -> usize {
        1
    }

    fn retain(&self) {}
    fn release(&self) -> usize {
        1
    }

    fn get_context_id(&self) -> u32 {
        self.id
    }

    fn get_id(&self) -> u64 {
        self.id as u64
    }

    fn query_attribute(&self, _attribute: u32, _value: &mut [u8]) -> VxResult<()> {
        Ok(())
    }
}

/// Kernel trait
pub trait KernelTrait: Send + Sync {
    fn get_name(&self) -> &str;
    fn get_enum(&self) -> crate::types::VxKernel;
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()>;
    fn execute(&self, params: &[&dyn Referenceable], context: &Context) -> VxResult<()>;
}
