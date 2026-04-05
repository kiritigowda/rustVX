# Final Reference Counting Fix

## Problem

vxRetainReference returns vx_status but OpenVX spec requires vx_uint32 (the count).

## Current Code (WRONG):
```rust
pub extern "C" fn vxRetainReference(_ref_: vx_reference) -> vx_status {
    // ... increments count ...
    return VX_SUCCESS;  // WRONG - should return count
}
```

## Fixed Code:
```rust
#[no_mangle]
pub extern "C" fn vxRetainReference(_ref_: vx_reference) -> vx_uint32 {
    if _ref_.is_null() {
        return 0;
    }
    let addr = _ref_ as usize;
    if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
        if let Some(count) = counts.get(&addr) {
            let new_count = count.fetch_add(1, Ordering::SeqCst) + 1;
            return new_count as vx_uint32;
        }
    }
    0
}
```

## Also Fix vxReleaseReference:

Current returns vx_status, should return vx_uint32:
```rust
#[no_mangle]
pub extern "C" fn vxReleaseReference(ref_: *mut vx_reference) -> vx_uint32 {
    if ref_.is_null() || (*ref_).is_null() {
        return 0;
    }
    
    let addr = *ref_ as usize;
    let mut new_count = 0;
    
    if let Ok(counts) = REFERENCE_COUNTS.lock() {
        if let Some(count) = counts.get(&addr) {
            let current = count.load(Ordering::SeqCst);
            if current > 0 {
                new_count = current - 1;
                count.store(new_count, Ordering::SeqCst);
            }
        }
    }
    
    *ref_ = std::ptr::null_mut();
    new_count as vx_uint32
}
```

## Files to Modify:
- openvx-core/src/c_api.rs
  - Change vxRetainReference return type: vx_status -> vx_uint32
  - Change vxReleaseReference return type: vx_status -> vx_uint32
  - Update implementations to return count, not VX_SUCCESS
