# rustVX Vision Conformance Execution Plan v2
**Date:** April 28, 2026
**Branch:** conformance/vision (from master @ 92b9b36)
**Goal:** Pass all 1160 CTS tests (178 baseline + 982 vision) consistently and achieve OpenVX 1.3 Vision Conformance

---

## Starting Point: 1160/1160 ALL PASS ✅

We achieved full baseline + vision conformance on April 27. The task now is to:
1. **Stabilize** — ensure all 1160 pass reliably (no crashes, no memory corruption)
2. **Close gaps** — fix the remaining edge cases that crash/hang on certain test configurations
3. **Prepare for submission** — clean up debug output, validate CI, ensure reproducibility

---

## Known Instabilities (from April 27 session + CTS run)

### Crash: `Image.FormatImagePatchAddress1d` with IYUV
- **Symptom:** malloc assertion failure (sysmalloc) at 256x256 IYUV
- **Root cause:** Likely buffer overflow in planar YUV image patch mapping
- **Impact:** Prevents running full CTS suite without `--filter` exclusion
- **Priority:** P0 — blocks full CI run

### Debug Output Spam
- **Symptom:** `eprintln!("DEBUG ...")` calls throughout unified_c_api.rs (8000+ line file)
- **Impact:** Slows CI, pollutes test output, could cause CI timeout
- **Priority:** P1 — should clean before submission

### Potential Memory Leaks
- **Symptom:** Global registries (NODES, GRAPHS, etc.) grow unbounded across test contexts
- **Mitigation:** Context cleanup clears registries, but cross-test leaks possible
- **Priority:** P2 — monitor with valgrind/ASan

---

## Step-by-Step Plan

### Step 1: [STABILITY] Fix IYUV Image Patch Buffer Overflow
**Dependencies:** None
**Approach:**
- Audit `vxMapImagePatch` for planar YUV formats (VX_DF_IMAGE_IYUV / NV12 / YUYV)
- The crash occurs at FormatImagePatchAddress1d with IYUV 256x256
- Check plane stride calculations: IYUV has 3 separate planes (Y, U, V) with U/V at half resolution
- Ensure buffer allocation accounts for all planes and correct stride alignment
- Add bounds checking in `vxFormatImagePatchAddress1d` / `vxFormatImagePatchAddress2d`
- Test with: `VX_TEST_DATA_PATH=... ./bin/vx_test_conformance --filter=Image.FormatImagePatchAddress1d`

**Verification:** CTS `Image.FormatImagePatchAddress1d` passes all sub-cases without crash
**Files:** `openvx-image/src/c_api.rs`, `openvx-core/src/unified_c_api.rs`

### Step 2: [STABILITY] Fix Remaining Image Test Failures
**Dependencies:** Step 1 (IYUV fix needed first)
**Approach:**
- Run full Image test group, identify remaining failures
- Known partial failures: `Image.CreateImageFromHandle`, `Image.SwapImageHandle`, `Image.vxSetImagePixelValues`
- Fix each systematically
- Verify `vxMapImagePatch` and `vxCopyImagePatch` reach 100%

**Verification:** `Image.*` tests all pass
**Files:** `openvx-image/src/c_api.rs`

### Step 3: [CLEANUP] Remove Debug eprintln Spam
**Dependencies:** None (can be done in parallel)
**Approach:**
- Replace all `eprintln!("DEBUG ...")` in `unified_c_api.rs` with `log::debug!()` or remove entirely
- Remove `eprintln!("ERROR: ...")` that are just noise, keep only genuine error paths
- Use `log::warn` for recoverable errors, `log::error` for genuine failures
- This reduces output from thousands of lines to clean pass/fail reporting

**Verification:** CTS output is clean (no DEBUG lines unless RUST_LOG=debug)
**Files:** `openvx-core/src/unified_c_api.rs`, `openvx-core/src/c_api.rs`, `openvx-image/src/c_api.rs`

### Step 4: [STABILITY] Fix vxuMultiply and Arithmetic Edge Cases
**Dependencies:** None (can be done in parallel)
**Approach:**
- `vxuMultiply` had 5/306 passing before April 27 fixes — verify all 306 now pass
- If not, debug: likely overflow handling, scale parameter, or VX_DF_IMAGE format support gaps
- Check S16/U16 format support in arithmetic operations (recent commit added some)
- Run focused tests: `--filter=vxMultiply` and `--filter=vxuMultiply`

**Verification:** vxMultiply 306/306, vxuMultiply 170/170, vxAddSub 76/76, WeightedAverage 102/102
**Files:** `openvx-vision/src/arithmetic.rs`, `openvx-core/src/vxu_impl.rs`

### Step 5: [STABILITY] Fix Scale/Warp/Remap Geometric Tests
**Dependencies:** None (can be done in parallel)
**Approach:**
- Scale was 488/982 — need to verify it's now 982/982 after April 27 fixes
- WarpAffine was 71/305 — check all interpolation modes (NEAREST, BILINEAR, AREA)
- WarpPerspective was 174/361 — same interpolation issues
- Remap was 52/380 — likely related to vxMapRemapPatch hang
- These are the highest-test-count groups; fixing them has the biggest impact

**Verification:** Scale 982/982, WarpAffine 305/305, WarpPerspective 361/361, Remap 380/380
**Files:** `openvx-vision/src/geometric.rs`, `openvx-core/src/unified_c_api.rs`

### Step 6: [STABILITY] Fix Feature Detection Tests
**Dependencies:** None (can be done in parallel)
**Approach:**
- HarrisCorners was 0/433 — likely needs corner response computation, NMS
- FastCorners was 0/24 — needs FAST corner detection implementation check
- vxCanny was 0/28 — Canny edge detector needs hysteresis thresholding
- OpticalFlowPyrLK was 1/5 — needs pyramid-based Lucas-Kanade
- Verify these all pass now after April 27 fixes

**Verification:** HarrisCorners 433/433, FastCorners 24/24, vxCanny 28/28, OptFlowPyrLK 5/5
**Files:** `openvx-vision/src/features.rs`, `openvx-vision/src/object_detection.rs`, `openvx-vision/src/optical_flow.rs`

### Step 7: [VALIDATION] Full CTS Run with Zero Crashes
**Dependencies:** Steps 1-6
**Approach:**
- Run complete CTS suite: `VX_TEST_DATA_PATH=... ./bin/vx_test_conformance --quiet`
- Must complete without crashes, hangs, or memory errors
- Verify 1160/1160 pass count
- Run 3 times to confirm reproducibility
- Check for flaky tests (pass sometimes, fail sometimes)

**Verification:** 1160/1160 three consecutive runs, zero crashes
**Files:** None (validation only)

### Step 8: [CI] Update GitHub Actions Workflow
**Dependencies:** Step 7
**Approach:**
- Update `.github/workflows/conformance.yml` to build rustVX + CTS and run full suite
- Remove any `--filter` exclusions
- Add expected pass count assertion (1160)
- Add artifact upload for test results XML
- Ensure CI runs on PRs to `conformance/vision` branch

**Verification:** CI pipeline passes with 1160/1160
**Files:** `.github/workflows/conformance.yml`

### Step 9: [CLEANUP] Remove Stale Debug Assertions, Add Error Handling
**Dependencies:** Step 3
**Approach:**
- Audit all `unwrap()`, `expect()`, and `panic!()` calls in FFI boundary code
- Replace with proper error returns (VX_ERROR_INVALID_PARAMETERS, etc.)
- Ensure no path can cause SIGABRT/SIGSEGV in production
- Add proper `vx_status` error codes throughout

**Verification:** No panics in any CTS test path
**Files:** `openvx-core/src/unified_c_api.rs`, `openvx-core/src/c_api.rs`, `openvx-core/src/c_api_data.rs`, `openvx-image/src/c_api.rs`, `openvx-buffer/src/c_api.rs`

### Step 10: [SUBMISSION] Final Review and PR
**Dependencies:** Steps 7-9
**Approach:**
- Squash or organize commits into logical groups
- Write detailed PR description with test results
- Add CHANGELOG entry
- Push to GitHub and create PR from `conformance/vision` → `master`
- Verify CI passes

**Verification:** Clean PR with passing CI, 1160/1160 CTS results
**Files:** CHANGELOG.md, README.md (if needed)

---

## Parallelism Strategy

Steps 1-6 can be partially parallelized:
- **Thread A:** Steps 1→2 (IYUV crash → full image tests)
- **Thread B:** Step 3 (debug cleanup — independent)
- **Thread C:** Steps 4→5→6 (arithmetic → geometric → features)

Step 7 requires all prior steps complete.
Steps 8-10 are sequential after Step 7.

## Risk Analysis

**P0 Risk:** IYUV buffer overflow crash blocks full CTS run. This is a memory safety bug that must be fixed first.

**P1 Risk:** Debug spam makes CI output unreadable and can cause timeouts. Easy fix but must be done.

**P2 Risk:** Flaky tests — some tests may pass locally but fail in CI due to timing or memory differences. Need reproducibility verification.

**P3 Risk:** Geometric operations (Scale 982 tests, WarpAffine 305, WarpPerspective 361) are the biggest test groups. If these regressed from the April 27 fixes, they represent the largest potential failure count.

## Rollback Plan

If any step introduces regressions:
1. Revert the specific commit
2. Re-run CTS to verify no regressions
3. Fix the issue in a separate commit before merging

The branch `conformance/vision` is based on the known-good commit `92b9b36` (1160/1160), so we always have a clean baseline to diff against.