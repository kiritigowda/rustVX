//! C API for OpenVX Buffer and Array

use std::ffi::c_void;
use std::sync::RwLock;
use std::collections::HashMap;
use openvx_core::c_api::{
    vx_context, vx_array, vx_status, vx_enum, vx_size, vx_uint32, vx_map_id,
    VX_SUCCESS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_INVALID_PARAMETERS,
    VX_ERROR_NOT_IMPLEMENTED,
    VX_TYPE_UINT8, VX_TYPE_INT8, VX_TYPE_UINT16, VX_TYPE_INT16,
    VX_TYPE_UINT32, VX_TYPE_INT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT64,
    VX_ARRAY_CAPACITY, VX_ARRAY_ITEMTYPE, VX_ARRAY_NUMITEMS, VX_ARRAY_ITEMSIZE,
    VX_READ_ONLY, VX_WRITE_ONLY, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, VX_MEMORY_TYPE_NONE,
};

/// Array struct for C API
pub struct VxCArray {
    item_type: vx_enum,
    item_size: vx_size,
    capacity: vx_size,
    num_items: RwLock<vx_size>,
    data: RwLock<Vec<u8>>,
    context: vx_context,
    mapped_ranges: RwLock<HashMap<vx_map_id, (vx_size, vx_size, Vec<u8>)>>,
}

impl VxCArray {
    fn type_to_size(item_type: vx_enum) -> vx_size {
        match item_type {
            VX_TYPE_UINT8 | VX_TYPE_INT8 => 1,
            VX_TYPE_UINT16 | VX_TYPE_INT16 => 2,
            VX_TYPE_UINT32 | VX_TYPE_INT32 | VX_TYPE_FLOAT32 => 4,
            VX_TYPE_FLOAT64 => 8,
            _ => 1,
        }
    }
}

// Internal storage for arrays
use std::sync::atomic::{AtomicUsize, Ordering};
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
    let total_size = capacity * item_size;

    let array = Box::new(VxCArray {
        item_type,
        item_size,
        capacity,
        num_items: RwLock::new(0),
        data: RwLock::new(vec![0u8; total_size]),
        context,
        mapped_ranges: RwLock::new(HashMap::new()),
    });

    Box::into_raw(array) as vx_array
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
    let dest_offset = *num_items * array.item_size;
    let src = ptr as *const u8;

    unsafe {
        if stride == array.item_size || stride == 0 {
            // Contiguous copy
            let src_slice = std::slice::from_raw_parts(src, count * array.item_size);
            data[dest_offset..dest_offset + count * array.item_size].copy_from_slice(src_slice);
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
                if size != std::mem::size_of::<vx_enum>() {
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

/// Release array
#[no_mangle]
pub extern "C" fn vxReleaseArray(arr: *mut vx_array) -> vx_status {
    if arr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    unsafe {
        if !(*arr).is_null() {
            let _ = Box::from_raw(*arr as *mut VxCArray);
            *arr = std::ptr::null_mut();
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

    let range_size = (end - start) * array.item_size;
    let offset = start * array.item_size;

    // Copy data to a temporary buffer for mapping
    let data = array.data.read().unwrap();
    let mut mapped_data = vec![0u8; range_size];
    
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(offset),
            mapped_data.as_mut_ptr(),
            range_size,
        );
    }

    let id = ARRAY_ID_COUNTER.fetch_add(1, Ordering::SeqCst) as vx_map_id;

    {
        let mut mapped = array.mapped_ranges.write().unwrap();
        mapped.insert(id, (start, end, mapped_data.clone()));
    }

    unsafe {
        *map_id = id;
        *ptr = mapped_data.as_ptr() as *mut c_void;
        if !stride.is_null() {
            *stride = array.item_size;
        }
    }

    // Keep the data alive by storing it in the mapped ranges
    std::mem::forget(mapped_data);

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
        let range_size = (end - start) * array.item_size;
        let offset = start * array.item_size;
        
        let mut array_data = array.data.write().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                array_data.as_mut_ptr().add(offset),
                range_size,
            );
        }
        
        // Reconstruct the vec to properly drop it
        let _ = unsafe { Vec::from_raw_parts(data.as_ptr() as *mut u8, data.len(), data.len()) };
        std::mem::forget(data); // Prevent double free
    }

    VX_SUCCESS
}
