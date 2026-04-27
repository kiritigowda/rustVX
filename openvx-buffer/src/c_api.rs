//! C API for OpenVX Buffer and Array

#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::sync::{RwLock, atomic::AtomicUsize};
use std::collections::HashMap;
use openvx_core::c_api::{
    vx_context, vx_graph, vx_array, vx_status, vx_enum, vx_size, vx_uint32, vx_map_id, vx_int32,
    vx_reference,
    VX_SUCCESS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_INVALID_PARAMETERS,
    VX_ERROR_NOT_IMPLEMENTED,
    VX_TYPE_CHAR, VX_TYPE_UINT8, VX_TYPE_INT8, VX_TYPE_UINT16, VX_TYPE_INT16,
    VX_TYPE_UINT32, VX_TYPE_INT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT64,
    VX_TYPE_INT64, VX_TYPE_UINT64,
    VX_TYPE_BOOL, VX_TYPE_ENUM, VX_TYPE_SIZE, VX_TYPE_DF_IMAGE,
    VX_TYPE_RECTANGLE, VX_TYPE_KEYPOINT, VX_TYPE_COORDINATES2D, VX_TYPE_COORDINATES3D,
    VX_ARRAY_CAPACITY, VX_ARRAY_ITEMTYPE, VX_ARRAY_NUMITEMS, VX_ARRAY_ITEMSIZE,
    VX_MEMORY_TYPE_HOST, VX_MEMORY_TYPE_NONE,
    vx_bool, vx_df_image, vx_rectangle_t, vx_keypoint_t, vx_coordinates2d_t, vx_coordinates3d_t,
    VX_READ_ONLY, VX_WRITE_ONLY,
};
use openvx_core::unified_c_api::{vx_distribution, vxCreateDistribution, REFERENCE_COUNTS, REFERENCE_TYPES, USER_STRUCTS};
use openvx_core::unified_c_api::{VX_TYPE_ARRAY};
use openvx_core::c_api::vxGetContext;

/// Array struct for C API
pub struct VxCArray {
    item_type: vx_enum,
    item_size: vx_size,
    capacity: vx_size,
    num_items: RwLock<vx_size>,
    data: RwLock<Vec<u8>>,
    context: vx_context,
    mapped_ranges: RwLock<HashMap<vx_map_id, (vx_size, vx_size, Vec<u8>)>>,
    is_virtual: bool,
}

impl VxCArray {
    fn type_to_size(item_type: vx_enum) -> vx_size {
        // Check if it's a user-defined struct type
        if item_type >= 0x100 {
            // Look up in USER_STRUCTS registry
            if let Ok(structs) = USER_STRUCTS.lock() {
                if let Some((_, size)) = structs.get(&item_type) {
                    return *size;
                }
            }
            // Default to 16 bytes for unknown user structs
            return 16;
        }
        
        match item_type {
            VX_TYPE_CHAR | VX_TYPE_UINT8 | VX_TYPE_INT8 => 1,
            VX_TYPE_UINT16 | VX_TYPE_INT16 => 2,
            VX_TYPE_UINT32 | VX_TYPE_INT32 | VX_TYPE_FLOAT32 | VX_TYPE_ENUM => 4,
            VX_TYPE_UINT64 | VX_TYPE_INT64 | VX_TYPE_FLOAT64 | VX_TYPE_SIZE => 8,
            VX_TYPE_BOOL => std::mem::size_of::<vx_bool>(),
            VX_TYPE_DF_IMAGE => std::mem::size_of::<vx_df_image>(),
            VX_TYPE_RECTANGLE => std::mem::size_of::<vx_rectangle_t>(),
            VX_TYPE_KEYPOINT => std::mem::size_of::<vx_keypoint_t>(),
            VX_TYPE_COORDINATES2D => std::mem::size_of::<vx_coordinates2d_t>(),
            VX_TYPE_COORDINATES3D => std::mem::size_of::<vx_coordinates3d_t>(),
            _ => 1,
        }
    }
}

// Internal storage for arrays
use std::sync::atomic::{Ordering};
static ARRAY_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Create an array
#[no_mangle]
pub extern "C" fn vxCreateArray(
    context: vx_context,
    item_type: vx_enum,
    capacity: vx_size,
) -> vx_array {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    if capacity == 0 {
        return std::ptr::null_mut();
    }

    let item_size = VxCArray::type_to_size(item_type);
    let total_size = capacity
        .checked_mul(item_size)
        .and_then(|s| s.try_into().ok())
        .unwrap_or(0);
    if total_size == 0 && capacity > 0 {
        return std::ptr::null_mut();
    }

    let array = Box::new(VxCArray {
        item_type,
        item_size,
        capacity,
        num_items: RwLock::new(0),
        data: RwLock::new(vec![0u8; total_size]),
        context,
        mapped_ranges: RwLock::new(HashMap::new()),
        is_virtual: false,
    });

    let array_ptr = Box::into_raw(array) as vx_array;
    
    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(array_ptr as usize, AtomicUsize::new(1));
        }
    }
    
    // Register in REFERENCE_TYPES for type detection
    unsafe {
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(array_ptr as usize, VX_TYPE_ARRAY);
        }
    }
    
    array_ptr
}

/// Add items to array
#[no_mangle]
pub extern "C" fn vxAddArrayItems(
    arr: vx_array,
    count: vx_size,
    ptr: *const c_void,
    stride: vx_size,
) -> vx_status {
    if arr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() && count > 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let array = unsafe { &*(arr as *const VxCArray) };
    let mut num_items = array.num_items.write().unwrap();

    // Check capacity
    if *num_items + count > array.capacity {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    if count == 0 {
        return VX_SUCCESS;
    }

    let mut data = array.data.write().unwrap();
    let dest_offset = num_items.checked_mul(array.item_size).unwrap_or(0);
    if dest_offset + count.checked_mul(array.item_size).unwrap_or(0) > data.len() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    let src = ptr as *const u8;

    unsafe {
        if stride == array.item_size || stride == 0 {
            // Contiguous copy
            let copy_size = count.checked_mul(array.item_size).unwrap_or(0);
            let src_slice = std::slice::from_raw_parts(src, copy_size);
            data[dest_offset..dest_offset + copy_size].copy_from_slice(src_slice);
        } else {
            // Strided copy
            for i in 0..count {
                let src_offset = i * stride;
                let dest_idx = dest_offset + i * array.item_size;
                std::ptr::copy_nonoverlapping(
                    src.add(src_offset),
                    data.as_mut_ptr().add(dest_idx),
                    array.item_size,
                );
            }
        }
    }

    *num_items += count;
    VX_SUCCESS
}

/// Truncate array
#[no_mangle]
pub extern "C" fn vxTruncateArray(
    arr: vx_array,
    new_num_items: vx_size,
) -> vx_status {
    if arr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let array = unsafe { &*(arr as *const VxCArray) };
    let mut num_items = array.num_items.write().unwrap();

    if new_num_items > *num_items {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    *num_items = new_num_items;
    VX_SUCCESS
}

/// Query array attributes
#[no_mangle]
pub extern "C" fn vxQueryArray(
    arr: vx_array,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if arr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() || size == 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let array = unsafe { &*(arr as *const VxCArray) };

    unsafe {
        match attribute {
            VX_ARRAY_CAPACITY => {
                if size != std::mem::size_of::<vx_size>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_size) = array.capacity;
            }
            VX_ARRAY_ITEMTYPE => {
                if size < std::mem::size_of::<vx_enum>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_enum) = array.item_type;
            }
            VX_ARRAY_NUMITEMS => {
                if size != std::mem::size_of::<vx_size>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_size) = *array.num_items.read().unwrap();
            }
            VX_ARRAY_ITEMSIZE => {
                if size != std::mem::size_of::<vx_size>() {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
                *(ptr as *mut vx_size) = array.item_size;
            }
            _ => return VX_ERROR_NOT_IMPLEMENTED,
        }
    }

    VX_SUCCESS
}

/// Create a virtual array (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualArray(
    graph: vx_graph,
    item_type: vx_enum,
    capacity: vx_size,
) -> vx_array {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    
    // Get the context from the graph
    let context = vxGetContext(graph as vx_reference);
    if context.is_null() {
        return std::ptr::null_mut();
    }
    
    // Virtual arrays can have capacity 0 (unspecified), so default to something reasonable
    let actual_capacity = if capacity == 0 { 1024 } else { capacity };
    if actual_capacity == 0 {
        return std::ptr::null_mut();
    }

    let item_size = VxCArray::type_to_size(item_type);
    let total_size = actual_capacity
        .checked_mul(item_size)
        .and_then(|s| s.try_into().ok())
        .unwrap_or(0);
    if total_size == 0 && actual_capacity > 0 {
        return std::ptr::null_mut();
    }

    let array = Box::new(VxCArray {
        item_type,
        item_size,
        capacity: actual_capacity,
        num_items: RwLock::new(0),
        data: RwLock::new(vec![0u8; total_size]),
        context,
        mapped_ranges: RwLock::new(HashMap::new()),
        is_virtual: true,
    });

    let array_ptr = Box::into_raw(array) as vx_array;
    
    // Register in reference counting
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(array_ptr as usize, AtomicUsize::new(1));
        }
    }
    
    // Register in REFERENCE_TYPES for type detection
    unsafe {
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(array_ptr as usize, VX_TYPE_ARRAY);
        }
    }
    
    array_ptr
}

/// Create a virtual distribution (for graph intermediate results)
#[no_mangle]
pub extern "C" fn vxCreateVirtualDistribution(
    graph: vx_graph,
    num_bins: vx_size,
    offset: vx_int32,
    range: vx_uint32,
) -> vx_distribution {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    // Virtual distributions are created like regular ones
    vxCreateDistribution(graph as vx_context, num_bins, offset as u32, range)
}

/// Release array
#[no_mangle]
pub extern "C" fn vxReleaseArray(arr: *mut vx_array) -> vx_status {
    if arr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*arr).is_null() {
            let addr = *arr as usize;
            
            // Check reference count before freeing
            let should_free = if let Ok(counts) = REFERENCE_COUNTS.lock() {
                if let Some(cnt) = counts.get(&addr) {
                    let current = cnt.load(std::sync::atomic::Ordering::SeqCst);
                    if current > 1 {
                        cnt.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            } else {
                false
            };
            
            if should_free {
                // Remove from reference counts and types
                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                    counts.remove(&addr);
                }
                if let Ok(mut types) = REFERENCE_TYPES.lock() {
                    types.remove(&addr);
                }
                
                let _ = Box::from_raw(*arr as *mut VxCArray);
            }
            
            *arr = std::ptr::null_mut();
        } else {
        }
    }

    VX_SUCCESS
}

/// Map array range for CPU access
#[no_mangle]
pub extern "C" fn vxMapArrayRange(
    arr: vx_array,
    start: vx_size,
    end: vx_size,
    map_id: *mut vx_map_id,
    stride: *mut vx_size,
    ptr: *mut *mut c_void,
    _usage: vx_enum,
    mem_type: vx_enum,
    _flags: vx_uint32,
) -> vx_status {
    if arr.is_null() || map_id.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    if mem_type != VX_MEMORY_TYPE_HOST && mem_type != VX_MEMORY_TYPE_NONE {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let array = unsafe { &*(arr as *const VxCArray) };
    let num_items = *array.num_items.read().unwrap();

    if end > num_items {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if start >= end {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    // Copy data to a temporary buffer for mapping
    let data = array.data.read().unwrap();

    let range_size = end.checked_sub(start)
        .and_then(|len| len.checked_mul(array.item_size))
        .unwrap_or(0);
    if range_size == 0 && start < end {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    let offset = start.checked_mul(array.item_size).unwrap_or(0);
    if offset.saturating_add(range_size) > data.len() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    // Allocate stable heap buffer
    let mut mapped_data: Vec<u8> = vec![0u8; range_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(offset),
            mapped_data.as_mut_ptr(),
            range_size,
        );
    }

    let id = ARRAY_ID_COUNTER.fetch_add(1, Ordering::SeqCst) as vx_map_id;

    // Get pointer - this stays valid as long as the Vec isn't dropped
    // The Vec is moved into mapped_ranges HashMap which keeps it alive
    // until vxUnmapArrayRange removes it
    let data_ptr = mapped_data.as_mut_ptr();

    {
        let mut mapped = array.mapped_ranges.write().unwrap();
        mapped.insert(id, (start, end, mapped_data));
    }

    unsafe {
        *map_id = id;
        *ptr = data_ptr as *mut c_void;
        if !stride.is_null() {
            *stride = array.item_size;
        }
    }

    VX_SUCCESS
}

/// Unmap previously mapped range
#[no_mangle]
pub extern "C" fn vxUnmapArrayRange(arr: vx_array, map_id: vx_map_id) -> vx_status {
    if arr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let array = unsafe { &*(arr as *const VxCArray) };
    let mut mapped = array.mapped_ranges.write().unwrap();

    if let Some((start, end, data)) = mapped.remove(&map_id) {
        // Copy data back if it was a write
        let range_size = end.checked_sub(start)
            .and_then(|len| len.checked_mul(array.item_size))
            .unwrap_or(0);
        let offset = start.checked_mul(array.item_size).unwrap_or(0);
        
        let mut array_data = array.data.write().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                array_data.as_mut_ptr().add(offset),
                range_size,
            );
        }
        // data Vec is automatically dropped when it goes out of scope
    }

    VX_SUCCESS
}

/// Copy array range data
#[no_mangle]
pub extern "C" fn vxCopyArrayRange(
    arr: vx_array,
    range_start: vx_size,
    range_end: vx_size,
    user_stride: vx_size,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    if arr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST && user_mem_type != 0x0 {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    // Use map/unmap approach
    let mut map_id: vx_map_id = 0;
    let mut stride: vx_size = 0;
    let mut ptr: *mut c_void = std::ptr::null_mut();

    let map_usage = match usage {
        VX_READ_ONLY | 0x1 => VX_READ_ONLY,
        VX_WRITE_ONLY | 0x2 => VX_WRITE_ONLY,
        _ => return VX_ERROR_INVALID_PARAMETERS,
    };

    let status = vxMapArrayRange(
        arr, range_start, range_end,
        &mut map_id, &mut stride, &mut ptr,
        map_usage, VX_MEMORY_TYPE_HOST, 0,
    );
    if status != VX_SUCCESS {
        return status;
    }

    let array = unsafe { &*(arr as *const VxCArray) };
    let item_size = array.item_size;
    let count = range_end - range_start;

    unsafe {
        match usage {
            VX_READ_ONLY | 0x1 => {
                if user_stride == item_size || user_stride == 0 {
                    let copy_size = count * item_size;
                    std::ptr::copy_nonoverlapping(ptr as *const u8, user_ptr as *mut u8, copy_size);
                } else {
                    for i in 0..count {
                        std::ptr::copy_nonoverlapping(
                            (ptr as *const u8).add(i * stride),
                            (user_ptr as *mut u8).add(i * user_stride),
                            item_size,
                        );
                    }
                }
            }
            VX_WRITE_ONLY | 0x2 => {
                if user_stride == item_size || user_stride == 0 {
                    let copy_size = count * item_size;
                    std::ptr::copy_nonoverlapping(user_ptr as *const u8, ptr as *mut u8, copy_size);
                } else {
                    for i in 0..count {
                        std::ptr::copy_nonoverlapping(
                            (user_ptr as *const u8).add(i * user_stride),
                            (ptr as *mut u8).add(i * stride),
                            item_size,
                        );
                    }
                }
            }
            _ => return VX_ERROR_INVALID_PARAMETERS,
        }
    }

    vxUnmapArrayRange(arr, map_id);
    VX_SUCCESS
}
