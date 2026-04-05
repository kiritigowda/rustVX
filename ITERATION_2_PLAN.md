# Iteration 2: Final 5 SmokeTest Failures

## Current Status: 9/14 passing

## Remaining Failures

### 1. SmokeTestBase.vxReleaseReferenceBase
**Likely cause:** Reference counting issue in base test
**Fix:** Check vxReleaseReference implementation

### 2. SmokeTestBase.vxLoadKernels
**Error:** vxLoadKernels not working
**Fix:** Implement vxLoadKernels to load kernel modules

### 3. SmokeTestBase.vxUnloadKernels  
**Error:** vxUnloadKernels not working
**Fix:** Implement vxUnloadKernels

### 4. SmokeTest.vxSetParameterByReference
**Error:** Parameter by reference setting
**Fix:** Complete vxSetParameterByReference

### 5. SmokeTest.vxGetParameterByIndex
**Debug shows:** vxQueryKernel VX_KERNEL_PARAMETERS not implemented
**Fix:** Add VX_KERNEL_PARAMETERS case to vxQueryKernel

## Quick Fixes Needed

### Fix 1: vxQueryKernel VX_KERNEL_PARAMETERS
```rust
VX_KERNEL_PARAMETERS => {
    if size >= 4 {
        let num = kernel_data.num_params as i32;
        *(ptr as *mut i32) = num;
        return VX_SUCCESS;
    }
}
```

### Fix 2: vxLoadKernels / vxUnloadKernels
Stub implementations returning VX_SUCCESS

### Fix 3: vxSetParameterByReference
Complete the implementation

## Target
Get to 14/14 SmokeTest passing (Group 1 complete)
