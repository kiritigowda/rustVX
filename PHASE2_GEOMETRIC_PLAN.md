# Phase 2: Geometric Transforms Implementation Plan

## Target: 2,028 Tests (Scale + WarpAffine + WarpPerspective + Remap)

## Test Groups Analysis

| Group | Tests | Current Status | Key Issue |
|-------|-------|----------------|-----------|
| **Scale** | 982 | Partial | Interpolation accuracy |
| **WarpAffine** | 305 | Partial | Matrix application |
| **WarpPerspective** | 361 | Partial | Homography transforms |
| **Remap** | 380 | Partial | Lookup table mapping |

**Total: 2,028 tests (13.3% of full suite)**

---

## Core Algorithms Required

### 1. Scale Image (Bilinear Interpolation)

**Algorithm:**
```
For each output pixel (x_out, y_out):
  x_src = x_out * (src_width / dst_width)
  y_src = y_out * (src_height / dst_height)
  
  // Bilinear interpolation
  x0 = floor(x_src), y0 = floor(y_src)
  x1 = x0 + 1, y1 = y0 + 1
  
  dx = x_src - x0
  dy = y_src - y0
  
  value = (1-dx)*(1-dy)*I(x0,y0) + dx*(1-dy)*I(x1,y0) + 
          (1-dx)*dy*I(x0,y1) + dx*dy*I(x1,y1)
```

**Constants:**
- VX_INTERPOLATION_NEAREST_NEIGHBOR = 0
- VX_INTERPOLATION_BILINEAR = 1
- VX_INTERPOLATION_AREA = 2

### 2. Warp Affine (2x3 Matrix)

**Matrix Form:**
```
[x']   [a b c] [x]
[y'] = [d e f] [y]
              [1]
```

**Algorithm:**
```
For each output pixel (x_out, y_out):
  x_src = a*x_out + b*y_out + c
  y_src = d*x_out + e*y_out + f
  
  // Sample from source using interpolation
  output(x_out, y_out) = interpolate(source, x_src, y_src)
```

### 3. Warp Perspective (3x3 Homography)

**Matrix Form:**
```
[x']   [a b c] [x]
[y'] = [d e f] [y]
[w ]   [g h i] [1]
```

**Algorithm:**
```
For each output pixel (x_out, y_out):
  x_h = a*x_out + b*y_out + c
  y_h = d*x_out + e*y_out + f
  w_h = g*x_out + h*y_out + i
  
  x_src = x_h / w_h
  y_src = y_h / w_h
  
  output(x_out, y_out) = interpolate(source, x_src, y_src)
```

### 4. Remap (Arbitrary Mapping)

**Algorithm:**
```
For each output pixel (x_out, y_out):
  x_src = map_x(x_out, y_out)
  y_src = map_y(x_out, y_out)
  
  output(x_out, y_out) = interpolate(source, x_src, y_src)
```

---

## Implementation Strategy

### Step 1: [ANALYSIS] Current Implementation Status
**Agent:** geometric-analysis-agent
**Task:**
1. Find existing Scale, WarpAffine, WarpPerspective, Remap implementations
2. Identify gaps and accuracy issues
3. Check interpolation implementations

**Deliverable:** Report on current state and specific fixes needed

### Step 2: [FIX] Scale Image
**Agent:** scale-fix-agent  
**Dependencies:** Step 1
**Task:**
1. Fix bilinear interpolation in scale_image
2. Ensure proper border handling (UNDEFINED, CONSTANT, REPLICATE)
3. Handle different interpolation modes
4. Fix output format matching

**Files:** openvx-vision/src/geometric.rs

### Step 3: [FIX] Warp Affine
**Agent:** warp-affine-agent
**Dependencies:** Step 1
**Task:**
1. Fix 2x3 matrix application
2. Ensure backward mapping (output->input)
3. Fix interpolation at boundaries
4. Handle border modes

**Files:** openvx-vision/src/geometric.rs

### Step 4: [FIX] Warp Perspective
**Agent:** warp-perspective-agent
**Dependencies:** Step 1
**Task:**
1. Fix 3x3 homography matrix application
2. Implement proper perspective division (w)
3. Ensure backward mapping
4. Fix interpolation

**Files:** openvx-vision/src/geometric.rs

### Step 5: [FIX] Remap
**Agent:** remap-fix-agent
**Dependencies:** Step 1
**Task:**
1. Fix remap table lookup
2. Handle fractional coordinates properly
3. Ensure interpolation works with arbitrary maps

**Files:** openvx-vision/src/geometric.rs

### Step 6: [INTEGRATION] Test All Geometric
**Agent:** geometric-test-agent
**Dependencies:** Steps 2-5
**Task:**
1. Build and test all geometric transforms
2. Run Scale*, WarpAffine*, WarpPerspective*, Remap* tests
3. Verify fix success

**Expected Result:** 2,000+ tests passing

---

## Key Implementation Details

### Interpolation Requirements:
- **Bilinear:** Must use fractional weights correctly
- **Nearest:** Simple rounding to nearest pixel
- **Area:** Average of source pixels covered

### Border Modes:
- **VX_BORDER_UNDEFINED:** Don't process border
- **VX_BORDER_CONSTANT:** Use constant value (0)
- **VX_BORDER_REPLICATE:** Repeat edge pixels

### Matrix Storage:
- Affine: 2x3 matrix (6 floats)
- Perspective: 3x3 matrix (9 floats)

---

## Risk Analysis

**High Risk:**
- Perspective transform accuracy (homography division)
- Complex border handling
- Interpolation accuracy at fractional positions

**Mitigation:**
- Test each fix incrementally
- Use reference implementation for comparison
- Validate with known transformations

---

## Success Criteria

- **Scale:** 900+ tests passing (was failing)
- **WarpAffine:** 280+ tests passing (was failing)
- **WarpPerspective:** 330+ tests passing (was failing)
- **Remap:** 350+ tests passing (was failing)

**Total Phase 2: 1,800+ tests passing**
