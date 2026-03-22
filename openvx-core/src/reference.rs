//! Reference trait

use crate::types::{VxType, VxResult};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Trait for all OpenVX referenceable objects
pub trait Referenceable: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    
    /// Get the type of this reference
    fn get_type(&self) -> VxType;
    
    /// Get the reference count
    fn get_reference_count(&self) -> usize;
    
    /// Increment reference count
    fn retain(&self);
    
    /// Decrement reference count
    fn release(&self) -> usize;
    
    /// Get the context ID associated with this reference
    fn get_context_id(&self) -> u32;
    
    /// Get the unique ID of this reference
    fn get_id(&self) -> u64;
    
    /// Query an attribute from this reference
    fn query_attribute(&self, attribute: u32, value: &mut [u8]) -> VxResult<()>;
}

/// Reference counting wrapper
pub struct Ref<T: Referenceable> {
    inner: std::sync::Arc<T>,
}

impl<T: Referenceable> Ref<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: std::sync::Arc::new(inner),
        }
    }
    
    pub fn from_arc(arc: std::sync::Arc<T>) -> Self {
        Self { inner: arc }
    }
    
    pub fn clone_arc(&self) -> std::sync::Arc<T> {
        self.inner.clone()
    }
    
    pub fn strong_count(&self) -> usize {
        std::sync::Arc::strong_count(&self.inner)
    }
    
    pub fn get_ref(&self) -> std::sync::Arc<T> {
        self.inner.clone()
    }
}

impl<T: Referenceable> Clone for Ref<T> {
    fn clone(&self) -> Self {
        self.inner.retain();
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Referenceable> std::ops::Deref for Ref<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<T: Referenceable + std::fmt::Debug> std::fmt::Debug for Ref<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ref").field("inner", &*self.inner).finish()
    }
}

/// Reference table
pub struct ReferenceTable {
    next_id: AtomicUsize,
}

impl ReferenceTable {
    pub fn new() -> Self {
        Self {
            next_id: AtomicUsize::new(1),
        }
    }
    
    pub fn generate_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst) as u64
    }
}

impl Default for ReferenceTable {
    fn default() -> Self {
        Self::new()
    }
}
