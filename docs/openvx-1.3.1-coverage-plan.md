# rustVX → OpenVX 1.3.1 — 100% API Coverage Plan

> Status: **P2–P4 complete, P5 partial, P6–P8 pending**. Last updated 2026-05-27.
> Headers inventoried: `include/VX/*.h` (the OpenVX 1.3 + KHR-extension reference headers bundled in this repo).

---

## 1. Executive summary

rustVX currently exports **~300 of 361** OpenVX 1.3.1 `VX_API_ENTRY` functions (**~83% by name**), with all **6,786** enabled CTS tests across the **base** + **vision** + **enhanced vision** profiles passing on every push and pull request (see [Conformance Status](../README.md#conformance-status)).

**Completed phases:**
- **P2** (Base API + User Data Object, 10 functions) — **COMPLETE** via PRs #16, #18, #23, #24
- **P3** (Enhanced Vision non-tensor kernels, 14 functions) — **COMPLETE** via PRs #35, #36, #39
- **P4** (Tensor kernels, 14 functions) — **COMPLETE** via PR #40
- **P5 partial** (Control-flow nodes, 2/10 functions) — **COMPLETE** via PR #41

**Remaining work:**
- **P5** (Neural Network layers, 8 functions) — NOT STARTED
- **P6** (Optional KHR extensions, 28 functions) — NOT STARTED
- **P7** (Legacy 1.0.1 compatibility, 25 functions) — NOT STARTED
- **P8** (Cleanup of broken exports + casing aliases + docs) — NOT STARTED

---

## 2. Phase completion log

| Phase | Functions | Status | PRs | CTS Impact |
|---|---|---|---|---|
| **P2** — Base API & UDO | 10 | ✅ **COMPLETE** | #16, #18, #23, #24 | User Data Object extension (14 tests) |
| **P3** — Enhanced Vision (non-tensor) | 14 | ✅ **COMPLETE** | #35, #36, #39 | +BilateralFilter (361), +HOG (22), +LBP/MatchTemplate/NMS/Copy/HoughLinesP (84) |
| **P4** — Tensor kernels | 14 | ✅ **COMPLETE** | #40 | TensorOp (214 tests) |
| **P5a** — Control-flow nodes | 2 | ✅ **COMPLETE** | #41 | SelectNode (11) + ScalarOperationNode (175 tests) |
| **P5b** — Neural Network layers | 8 | ⏳ **PENDING** | — | NN conformance profile |
| **P6** — Optional KHR extensions | 28 | ⏳ **PENDING** | — | Extensions (non-conformance) |
| **P7** — Legacy 1.0.1 compat | 25 | ⏳ **PENDING** | — | Backwards compatibility |
| **P8** — Cleanup | — | ⏳ **PENDING** | — | Broken exports, casing aliases, docs |

**Current conformance tally:** 6,786 / 6,786 tests passing (100%) across baseline + vision + enhanced vision + user-data-object extension.

---

## 3. Current state

| Header | Spec functions | Implemented | Missing | Coverage |
|---|---:|---:|---:|---:|
| `vx_api.h`                   | 166 | **166** | 0   | **100%** |
| `vx_nodes.h`                 |  61 |  **61** | 0   | **100%** |
| `vxu.h`                      |  59 |  **59** | 0   | **100%** |
| `vx_khr_user_data_object.h`  |   7 |   **7** | 0   | **100%** |
| `vx_compatibility.h`         |  26 |   1 | 25  | 3.8%  |
| `vx_khr_nn.h`                |   8 |   0 | 8   | 0%    |
| `vx_khr_xml.h`               |   6 |   3 | 3   | 50%   |
| `vx_khr_pipelining.h`        |  12 |   0 | 12  | 0%    |
| `vx_khr_class.h`             |   3 |   0 | 3   | 0%    |
| `vx_khr_icd.h`               |   3 |   0 | 3   | 0%    |
| `vx_khr_buffer_aliasing.h`   |   2 |   0 | 2   | 0%    |
| `vx_khr_opencl.h`            |   2 |   0 | 2   | 0%    |
| `vx_import.h`                |   3 |   3 | 0   | 100%  |
| `vx_khr_ix.h` (export)       |   2 |   2 | 0   | 100%  |
| `vx_khr_import_kernel.h`     |   1 |   0 | 1   | 0%    |
| `vx_khr_opencl_interop.h`    |   1 |   0 | 1   | 0%    |
| `vx_khr_tiling.h`            |   1 |   0 | 1   | 0%    |
| **TOTAL**                    | **361** | **~300** | **~61** | **~83%** |

*Note: The 300 implemented count is approximate; the P2–P4 + P5a additions (+40 functions) were landed incrementally. A fresh re-audit of the FFI surface is recommended before declaring P5–P8 complete.*

---

## 4. Remaining API inventory (per phase)

### Phase 5b — Neural Network feature set (NN extension, **8 missing**)

The eight layer ops from `vx_khr_nn.h`. Each is a graph-mode node creator (no `vxu*` counterparts — NN ops only exist in graph mode):

- `vxActivationLayer`
- `vxConvolutionLayer`
- `vxDeconvolutionLayer`
- `vxFullyConnectedLayer`
- `vxLocalResponseNormalizationLayer`
- `vxPoolingLayer`
- `vxROIPoolingLayer`
- `vxSoftmaxLayer`

These are all tensor-in / tensor-out, so P4 (the Tensor data-type plumbing and the seven tensor kernels) is a hard dependency — **now satisfied**.

**Exit criteria:**
- All 8 layers pass their CTS sections in the NN profile.
- A small "ImageNet inference" smoke test runs end-to-end at FHD against rustVX.

### Phase 6 — Optional KHR extensions (**28 missing**)

Out of scope for OpenVX 1.3.1 conformance per se, but in scope for "100% of the API surface declared in the bundled headers". Grouped by extension:

| Extension | Functions | Status notes |
|---|---:|---|
| Pipelining / streaming (`vx_khr_pipelining.h`) | 12 | Largest sub-bucket. Requires async graph-executor refactor. |
| Classifier (`vx_khr_class.h`) | 3 | Builds on UDO (P2 — **now complete**). |
| ICD (`vx_khr_icd.h`) | 3 | Multi-implementation loader. |
| XML I/O (`vx_khr_xml.h`) | 3 | Round-trip serialisation. |
| Buffer aliasing (`vx_khr_buffer_aliasing.h`) | 2 | Memory aliasing hints. |
| OpenCL interop (`vx_khr_opencl.h` + `vx_khr_opencl_interop.h`) | 3 | **Out of scope** while rustVX is CPU-only — stub with `VX_ERROR_NOT_SUPPORTED`. |
| Import-kernel-from-URL (`vx_khr_import_kernel.h`) | 1 | Dynamic kernel modules. |
| Tiling kernels (`vx_khr_tiling.h`) | 1 | Cache-aware tiled execution. |
| **Sub-total** | **28** | |

### Phase 7 — Legacy 1.0.1 compatibility (**25 missing**)

All from `vx_compatibility.h`, gated by `#ifdef VX_1_0_1_NAMING_COMPATIBILITY`.

### Phase 8 — Cleanup (cross-cutting)

Things the original audit surfaced that are bugs or footguns *unrelated* to missing API surface:

**Broken / misleading exports (10):**

| Symbol | Behaviour today | Right answer |
|---|---|---|
| `vxCopyNode` | Always returns `NULL` | Implement or remove export |
| `vxCopyArray` | Returns success without copying | Implement using `vxMapArrayRange` + memcpy |
| `vxCopyRemap` | Returns success without copying | Implement using `vxMapRemapPatch` |
| `vxMapPyramidLevel` | Returns `-30` placeholder | Implement (delegates to per-level image map) |
| `vxCreateConvolutionFromPattern` | Always `NULL` | Implement |
| `vxCornerMinEigenValNode` | Factory exists, kernel not registered | Register kernel or remove factory |
| `vxMeanShiftNode` | Same as above | Same |
| `vxDilate5x5Node` / `vxErode5x5Node` / `vxSobel5x5Node` | No dispatch arm → `VX_ERROR_INVALID_KERNEL` | Wire up or remove; **not in 1.3.1 spec** |
| `vxAllocateImageMemory` / `vxLockImage` / `vxMapImage` / `vxUnlockImage` / `vxUnmapImage` / `vxReleaseImageMemory` | `NOT_IMPLEMENTED` / `NULL` | Move to P7 (legacy) or remove |

**Casing aliases (2):**
- `vxFastCornersNode` / `vxuFastCorners` (spec spelling) vs `vxFASTCornersNode` / `vxuFASTCorners` (current export). Add spec-spelled aliases.

**Non-spec extras (28):** Document under `docs/non-spec-exports.md`.

---

## 5. Coverage trajectory (updated)

| Milestone | Implemented | Spec | Coverage |
|---|---:|---:|---:|
| 2026-05-10 (original plan)                  | 260 | 361 | 72.0% |
| After **P2** (Base API + UDO, +10)         | 270 | 361 | 74.8% |
| After **P3** (Enhanced Vision non-tensor, +14) | 284 | 361 | 78.7% |
| After **P4** (Tensor kernels, +14)         | 298 | 361 | 82.5% |
| After **P5a** (Control-flow nodes, +2)     | 300 | 361 | 83.1% |
| After **P5b** (NN layers, +8)              | 308 | 361 | 85.3% |
| After **P6** (Optional KHR, +28)           | 336 | 361 | 93.1% |
| After **P7** (Legacy 1.0.1 compat, +25)    | 361 | 361 | **100%** |

---

## 6. Open issues — status review

| Issue | Title | Status | Action |
|---|---|---|---|
| #38 | [Bug] Multiple implementation bugs found during Enhanced Vision CTS validation | **STALE / Should close** | All listed bugs (BilateralFilter border, HOGFeatures nondeterminism, `vxCopyTensor` overflow, etc.) were fixed in PRs #35, #36, #39, #40, #41. CI passes 100%. **Close as resolved.** |
| #22 | [P2] Implement vxMapUserDataObject + vxUnmapUserDataObject | **COMPLETE** | Implemented in PR #24. CTS `UserDataObject.*` passes (14/14). **Close as resolved.** |
| #21 | [P2] Implement vxCopyUserDataObject | **COMPLETE** | Implemented in PR #24. **Close as resolved.** |
| #20 | [P2] Implement vxQueryUserDataObject | **COMPLETE** | Implemented in PR #24. **Close as resolved.** |
| #19 | [P2] Implement vx_user_data_object lifecycle (Create / Virtual / Release) | **CLOSED** | Already closed. |
| #18 | [P2] Implement vxSetGraphAttribute | **CLOSED** | Already closed. |
| #17 | [P2] Implement vxRegisterKernelLibrary | **CLOSED** | Already closed. |
| #16 | [P2] Implement vxAddLogEntry | **CLOSED** | Already closed. |

**Recommended:** Close issues #20, #21, #22, and #38. They track work that is now in `main` and passing CI.

---

## 7. Testing strategy (unchanged)

Each phase ships with:

1. **Unit tests** in the relevant crate, comparing kernel output against a known-good scalar reference.
2. **CTS jobs** extended in `.github/workflows/conformance.yml` — see the 6 functional Enhanced Vision CI groups added in PR #42.
3. **Benchmark coverage** — each new kernel with meaningful runtime cost gets an `openvx-mark` bench entry.
4. **Perf-gate participation** — benched kernels automatically join the PR-vs-main `perf-gate`.

---

## 8. Risks & open questions (updated)

| # | Risk | Status | Mitigation |
|---|---|---|---|
| R1 | Tensor plumbing partial / unstable. | **RESOLVED** | P4 landed; tensor ops pass 214/214 CTS tests. |
| R2 | NN profile is large. | **ACTIVE** | P5b is the next major phase. Budget 2–3 weeks per hard layer. |
| R3 | Pipelining requires graph-executor refactor. | **ACTIVE** | Schedule last in P6; executor is still synchronous-only. |
| R4 | OpenCL interop out of scope while CPU-only. | **ACCEPTED** | Stub with `VX_ERROR_NOT_SUPPORTED`; revisit if GPU backend added. |
| R5 | 5×5 morphology / Sobel — not in 1.3.1 spec. | **ACTIVE** | P8 should delete or wire up; do not promise conformance. |
| R6 | Conformance test data churn. | **MITIGATED** | CTS submodule pinned; bump deliberately per phase. |

---

## 9. Tracking

Phase labels in use:
- `coverage/p2-base-udo` — **retire** (all issues closed)
- `coverage/p3-enhanced-vision` — **retire**
- `coverage/p4-tensor` — **retire**
- `coverage/p5-neural-network` — **activate** for P5b
- `coverage/p6-khr-extensions` — pending
- `coverage/p7-legacy-compat` — pending
- `coverage/p8-cleanup` — pending

---

## 10. References

- OpenVX 1.3 specification: <https://registry.khronos.org/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html>
- OpenVX 1.3.1 errata: <https://registry.khronos.org/OpenVX/specs/1.3.1/html/OpenVX_Specification_1_3_1.html>
- rustVX conformance status: [`README.md`](../README.md#conformance-status)
- Khronos sample-impl bug (LaplacianPyramid Initializer hoist): <https://github.com/KhronosGroup/OpenVX-sample-impl/issues/59>
- openvx-mark LaplacianPyramid fix-up: <https://github.com/kiritigowda/openvx-mark/pull/4>

---

## Appendix A. Full missing-function list, alphabetical (updated)

### P5b — Neural Network layers (8)

```
vxActivationLayer
vxConvolutionLayer
vxDeconvolutionLayer
vxFullyConnectedLayer
vxLocalResponseNormalizationLayer
vxPoolingLayer
vxROIPoolingLayer
vxSoftmaxLayer
```

### P6 — Optional KHR (28)

```
# Pipelining (vx_khr_pipelining.h)
vxDisableEvents                       vxEnableEvents
vxEnableGraphStreaming                vxGraphParameterCheckDoneRef
vxGraphParameterDequeueDoneRef        vxGraphParameterEnqueueReadyRef
vxRegisterEvent                       vxSendUserEvent
vxSetGraphScheduleConfig              vxStartGraphStreaming
vxStopGraphStreaming                  vxWaitEvent

# Classifier (vx_khr_class.h)
vxImportClassifierModel
vxReleaseClassifierModel
vxScanClassifierNode

# ICD (vx_khr_icd.h)
vxCreateContextFromPlatform           vxIcdGetPlatforms             vxQueryPlatform

# XML I/O (vx_khr_xml.h)
vxExportToXML                         vxGetImportReferenceByIndex   vxImportFromXML

# Buffer aliasing (vx_khr_buffer_aliasing.h)
vxAliasParameterIndexHint             vxIsParameterAliased

# OpenCL (vx_khr_opencl.h + vx_khr_opencl_interop.h)
vxAddOpenCLAsBinaryKernel             vxAddOpenCLAsSourceKernel
vxCreateContextFromCL

# Misc KHR
vxImportKernelFromURL  (vx_khr_import_kernel.h)
vxAddTilingKernel      (vx_khr_tiling.h)
```

### P7 — Legacy 1.0.1 compat (25)

```
vxAccessArrayRange              vxCommitArrayRange
vxAccessDistribution            vxCommitDistribution
vxAccessImagePatch              vxCommitImagePatch
vxAccessLUT                     vxCommitLUT
vxAccumulateImageNode           vxuAccumulateImage
vxAccumulateSquareImageNode     vxuAccumulateSquareImage
vxAccumulateWeightedImageNode   vxuAccumulateWeightedImage
vxAddKernel
vxComputeImagePatchSize
vxGetRemapPoint                 vxSetRemapPoint
vxNormalizationLayer            (re-export of vx_khr_nn.h)
vxReadConvolutionCoefficients   vxWriteConvolutionCoefficients
vxReadMatrix                    vxWriteMatrix
vxReadScalarValue               vxWriteScalarValue
```
