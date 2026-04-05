# Phase 3: Convolution Implementation Plan

## Target: 1,009 Tests (Convolve)

## Analysis

The **Convolve** test group has **1,009 tests** - the largest single test group in the entire CTS suite (6.6% of all tests).

## Current Status

**vxConvolve** allows custom convolution kernels (not just the fixed 3x3 filters like Gaussian/Box).

Key requirements:
- Variable kernel sizes (up to 9x9 or larger)
- Custom kernel coefficients (i16 values)
- Kernel scaling factor
- Multiple border handling modes

## Implementation Requirements

### 1. Data Structures

**VxCConvolution** (in c_api_data.rs):
```rust
pub struct VxCConvolution {
    pub rows: usize,
    pub cols: usize,
    pub scale: u32,
    pub data: RwLock<Vec<i16>>,  // Kernel coefficients
}
```

### 2. API Functions

**vxCreateConvolution** - Create convolution object:
```rust
pub extern "C" fn vxCreateConvolution(
    context: vx_context,
    columns: vx_size,
    rows: vx_size,
) -> vx_convolution
```

**vxQueryConvolution** - Query attributes:
```rust
VX_CONVOLUTION_ROWS
VX_CONVOLUTION_COLUMNS
VX_CONVOLUTION_SCALE
VX_CONVOLUTION_SIZE
```

**vxSetConvolutionAttribute** - Set scale:
```rust
VX_CONVOLUTION_SCALE
```

**vxCopyConvolutionCoefficients** - Get/Set coefficients:
```rust
// Read/write i16 coefficients
```

### 3. Convolution Algorithm

**Generic convolution for any kernel size:**
```rust
pub fn convolve(src: &Image, dst: &Image, kernel: &VxCConvolution) -> VxResult<()> {
    let k_rows = kernel.rows;
    let k_cols = kernel.cols;
    let k_center_x = k_cols / 2;
    let k_center_y = k_rows / 2;
    let scale = kernel.scale as i32;
    
    for y in 0..src.height() {
        for x in 0..src.width() {
            let mut sum: i32 = 0;
            
            // Apply kernel
            for ky in 0..k_rows {
                for kx in 0..k_cols {
                    let src_x = x as isize + kx as isize - k_center_x as isize;
                    let src_y = y as isize + ky as isize - k_center_y as isize;
                    
                    let pixel = get_pixel_bordered(src, src_x, src_y, border) as i32;
                    let coeff = kernel.data[ky * k_cols + kx] as i32;
                    
                    sum += pixel * coeff;
                }
            }
            
            // Apply scale and clamp
            if scale > 0 {
                sum = sum / scale;
            }
            
            dst.set_pixel(x, y, sum.max(0).min(255) as u8);
        }
    }
    
    Ok(())
}
```

### 4. Kernel Implementation

**ConvolveKernel** in openvx-vision/src/filter.rs or geometric.rs:
```rust
pub struct ConvolveKernel;

impl KernelTrait for ConvolveKernel {
    fn get_name(&self) -> &str { "org.khronos.openvx.custom_convolution" }
    fn get_enum(&self) -> VxKernel { VxKernel::Convolve }
    
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        // Need: input image, convolution object, output image
        if params.len() < 3 {
            return Err(VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }
    
    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params[0].as_any().downcast_ref::<Image>().ok_or(...)?;
        let conv = params[1].as_any().downcast_ref::<VxCConvolution>().ok_or(...)?;
        let dst = params[2].as_any().downcast_ref::<Image>().ok_or(...)?;
        
        convolve(src, dst, conv)?;
        Ok(())
    }
}
```

## Implementation Steps

### Step 1: [VERIFY] Check current implementation
**Agent:** convolve-analysis-agent
**Task:** Check if vxCreateConvolution and related functions exist
**Files:** openvx-core/src/c_api_data.rs, openvx-vision/src/filter.rs

### Step 2: [IMPLEMENT] Convolution Data Objects
**Agent:** convolve-data-agent
**Task:** Ensure VxCConvolution struct and all API functions work
**Files:** openvx-core/src/c_api_data.rs

### Step 3: [IMPLEMENT] Convolve Algorithm
**Agent:** convolve-algorithm-agent  
**Task:** Implement generic convolution with any kernel size
**Files:** openvx-vision/src/filter.rs or new file

### Step 4: [IMPLEMENT] Convolve Kernel
**Agent:** convolve-kernel-agent
**Task:** Implement ConvolveKernel::execute()
**Files:** openvx-vision/src/filter.rs

### Step 5: [TEST] Verify Convolve Tests
**Agent:** convolve-test-agent
**Task:** Run Convolve* tests and verify fixes
**Expected:** 900+ tests passing

## Key Implementation Details

### Kernel Coefficient Storage:
- Stored as `Vec<i16>` (signed 16-bit)
- Row-major order: data[row * cols + col]
- Center of kernel at (cols/2, rows/2)

### Border Handling:
- VX_BORDER_UNDEFINED: Skip border pixels (output = 0)
- VX_BORDER_CONSTANT: Use constant value (0)
- VX_BORDER_REPLICATE: Repeat edge pixels

### Scale Factor:
- Used to prevent overflow: result = sum / scale
- Default scale = 1 (no scaling)
- User can set via vxSetConvolutionAttribute

## Success Criteria

- **vxCreateConvolution**: Creates convolution object
- **vxQueryConvolution**: Returns correct attributes
- **vxCopyConvolutionCoefficients**: Read/write coefficients
- **ConvolveKernel::execute**: Performs convolution
- **Convolve* tests**: 900+ tests passing

## Potential Impact

**1,009 tests** - This single fix could add 6.6% to overall conformance!

## Files to Modify

1. **openvx-core/src/c_api_data.rs**
   - VxCConvolution struct (if not exists)
   - vxCreateConvolution
   - vxQueryConvolution
   - vxSetConvolutionAttribute
   - vxCopyConvolutionCoefficients
   - vxReleaseConvolution

2. **openvx-vision/src/filter.rs** (or new file)
   - convolve() function
   - ConvolveKernel implementation

3. **openvx-vision/src/lib.rs**
   - Register ConvolveKernel
