# rustVX → OpenVX 1.3.1 — 100% API Coverage Plan

> Status: **planning**. Last updated 2026-05-10.
> Branch: `kg/plan-openvx-1.3.1-coverage`.
> Headers inventoried: `include/VX/*.h` (the OpenVX 1.3 + KHR-extension reference headers bundled in this repo).

---

## 1. Executive summary

rustVX currently exports **260 of 361** OpenVX 1.3.1 `VX_API_ENTRY` functions (**~72% by name**), with all 5 828 enabled CTS tests across the **base** + **vision** + Phase 1 of the **enhanced vision** profiles passing on every push and pull request (see [Conformance Status](../README.md#conformance-status)).

The remaining **101 functions** fall into five clearly separable buckets that map cleanly onto five future Phases:

| # | Bucket | Functions | Effort | OpenVX 1.3.1 spec position |
|---|---|---:|---|---|
| **P2** | Base-API and User Data Object gaps | **10** | S | Base — required for full conformance |
| **P3** | Enhanced Vision feature set (non-tensor kernels) | **14** | M | Enhanced Vision — opt-in feature set, already partly enabled |
| **P4** | Tensor data object + Tensor kernels | **14** | L | Required by NN feature set and Enhanced Vision tensor ops |
| **P5** | Neural-Network feature set + control-flow nodes | **10** | L | NN gates the NN conformance profile; control-flow is base |
| **P6** | Optional KHR extensions (pipelining / classifier / ICD / XML / OpenCL / etc.) | **28** | L (variable) | Optional KHR — out of scope for spec conformance, in scope for full surface |
| **P7** | OpenVX 1.0.1 legacy compatibility shims | **25** | S | `vx_compatibility.h` — explicitly opt-in via `#ifdef VX_1_0_1_NAMING_COMPATIBILITY` |
| — | **TOTAL** | **101** | — | — |

Phase 2 is the only one strictly required to pass the OpenVX 1.3.1 conformance suite for the base + vision + already-claimed feature sets. Phases 3 / 4 / 5 unlock new conformance feature sets (Enhanced Vision, Neural Network). Phase 6 broadens the surface; Phase 7 is for legacy callers.

Alongside the new work, a **cleanup phase (P8)** addresses ten known-broken or misleading rust-only exports identified during this audit.

---

## 2. Current state

| Header | Spec functions | Implemented | Missing | Coverage |
|---|---:|---:|---:|---:|
| `vx_api.h`                   | 166 | 163 | 3   | 98.2% |
| `vx_nodes.h`                 |  61 |  46 | 15  | 75.4% |
| `vxu.h`                      |  59 |  44 | 15  | 74.6% |
| `vx_compatibility.h`         |  26 |   1 | 25  | 3.8%  |
| `vx_khr_nn.h`                |   8 |   0 | 8   | 0%    |
| `vx_khr_user_data_object.h`  |   7 |   0 | 7   | 0%    |
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
| **TOTAL**                    | **361** | **260** | **101** | **72.0%** |

In addition, **52 rust-only exports** are not in any spec header; these are private helpers, casing aliases, and a handful of broken stubs that the cleanup phase (P8) addresses.

The 5 828 CTS tests rustVX passes today are gated by what's already implemented at the kernel level — the missing API surface above is what unlocks the *additional* CTS sections the suite contains for tensor / NN / classifier / pipelining / user-data / control-flow.

---

## 3. Missing API inventory (per phase)

### Phase 2 — Base API & User Data Objects (**10 missing**)

These are gaps inside the core 1.3.1 base API and the User Data Object KHR (which OpenVX 1.3.1 promoted to mandatory for the User Data feature set). They're prerequisites for the rest of the work because several enhanced-vision and NN kernels accept user-data parameters.

**`vx_api.h` (base) — 3 missing**:
- `vxAddLogEntry` — context-level logging callback hook. Adjacent to `vxRegisterLogCallback` (which IS implemented).
- `vxRegisterKernelLibrary` — used by `vxLoadKernels` to register a target-specific kernel library. rustVX's `vxLoadKernels` is already real, so this is the matching public registration entry point.
- `vxSetGraphAttribute` — write side of `vxQueryGraph` (read side is real).

**`vx_khr_user_data_object.h` — 7 missing**:
- `vxCreateUserDataObject`
- `vxCreateVirtualUserDataObject`
- `vxReleaseUserDataObject`
- `vxQueryUserDataObject`
- `vxCopyUserDataObject`
- `vxMapUserDataObject`
- `vxUnmapUserDataObject`

User Data Objects are a generic "blob" data type that nodes can consume — the spec uses them to pass classifier models, NN weights, custom kernel state, etc. Required to be on the public surface for the Classifier and NN extensions to be conformant.

### Phase 3 — Enhanced Vision kernels (non-tensor) (**14 missing**)

Six kernels missing both graph-mode and immediate-mode entry points (12), plus two kernels where rustVX has the node factory but the `vxu*` immediate-mode wrapper is missing (`vxuCopy`, `vxuHoughLinesP`). All are in the OpenVX 1.3.1 **Enhanced Vision** feature set.

| Kernel | `vx*Node` | `vxu*` | Notes |
|---|:---:|:---:|---|
| Bilateral Filter            | ❌ | ❌ | Spatial + range Gaussian filter |
| HOG Cells                   | ❌ | ❌ | Histogram of Oriented Gradients, step 1 |
| HOG Features                | ❌ | ❌ | HOG step 2 (block normalisation) |
| LBP (Local Binary Pattern)  | ❌ | ❌ | Used by classifier / texture |
| Match Template              | ❌ | ❌ | Normalised cross-correlation |
| Non-Max Suppression         | ❌ | ❌ | Required by FAST / Harris, exposed standalone in 1.3 |
| Hough Lines (probabilistic) | ✅ already implemented | ❌ | `vxuHoughLinesP` is the missing immediate-mode wrapper |
| Copy                        | ✅ already implemented | ❌ | `vxuCopy` is the missing immediate-mode wrapper |

`vxHoughLinesPNode` is wired up in `register_standard_kernels` but is one of the broken extras flagged in §4 below; P3 fixes it for real.

### Phase 4 — Tensor data type + Tensor kernels (**14 missing**)

The Tensor data object is partially scaffolded in rustVX today (`vxCreateTensor`, `vxQueryTensor`, `vxMapTensorPatch` etc. are real), but the **seven tensor kernels** that Enhanced Vision defines are not wired into the kernel dispatch — neither `vx*Node` factories nor `vxu*` immediate-mode entry points.

| Tensor kernel | `vx*Node` | `vxu*` |
|---|:---:|:---:|
| Tensor Add                   | ❌ | ❌ |
| Tensor Subtract              | ❌ | ❌ |
| Tensor Multiply              | ❌ | ❌ |
| Tensor Convert Bit-Depth     | ❌ | ❌ |
| Tensor Matrix Multiply       | ❌ | ❌ |
| Tensor Table Lookup          | ❌ | ❌ |
| Tensor Transpose             | ❌ | ❌ |

### Phase 5 — Neural Network feature set (NN extension, **8 missing**)

The eight layer ops from `vx_khr_nn.h`. Each is a graph-mode node creator (no `vxu*` counterparts — NN ops only exist in graph mode):

- `vxActivationLayer`
- `vxConvolutionLayer`
- `vxDeconvolutionLayer`
- `vxFullyConnectedLayer`
- `vxLocalResponseNormalizationLayer`
- `vxPoolingLayer`
- `vxROIPoolingLayer`
- `vxSoftmaxLayer`

These are all tensor-in / tensor-out, so P4 (the Tensor data-type plumbing and the seven tensor kernels) is a hard dependency.

**Two control-flow kernels** also land here because they share the "tensor / scalar parameter pack" plumbing pattern:
- `vxScalarOperationNode` — scalar arithmetic on `vx_scalar` (no `vxu*` in spec)
- `vxSelectNode` — predicated copy (no `vxu*` in spec)

Both are part of OpenVX 1.3.1 **base** (control-flow group), but they're easiest to land alongside the NN work because they exercise the same generic-parameter machinery.

### Phase 6 — Optional KHR extensions (**28 missing**)

Out of scope for OpenVX 1.3.1 conformance per se, but in scope for "100% of the API surface declared in the bundled headers". Grouped by extension:

| Extension | Functions | Status notes |
|---|---:|---|
| Pipelining / streaming (`vx_khr_pipelining.h`) | 12 | Largest sub-bucket. Adds `vxEnableGraphStreaming`, `vxGraphParameterEnqueueReadyRef`, etc. Requires substantial graph-executor surgery to support async/streaming. |
| Classifier (`vx_khr_class.h`) | 3 | `vxImportClassifierModel`, `vxReleaseClassifierModel`, `vxScanClassifierNode`. Builds on UDO (P2). |
| ICD (`vx_khr_icd.h`) | 3 | OpenVX installable client driver — multi-implementation loader. Useful only when multiple OpenVX vendors are present on a system. |
| XML I/O (`vx_khr_xml.h`) | 3 | `vxExportToXML`, `vxImportFromXML`, `vxGetImportReferenceByIndex`. Round-trip serialisation. |
| Buffer aliasing (`vx_khr_buffer_aliasing.h`) | 2 | Memory aliasing hints between parameters. |
| OpenCL interop (`vx_khr_opencl.h` + `vx_khr_opencl_interop.h`) | 3 | `vxAddOpenCLAsBinaryKernel`, `vxAddOpenCLAsSourceKernel`, `vxCreateContextFromCL`. Out of scope while rustVX is CPU-only. |
| Import-kernel-from-URL (`vx_khr_import_kernel.h`) | 1 | `vxImportKernelFromURL` for dynamic kernel modules. |
| Tiling kernels (`vx_khr_tiling.h`) | 1 | `vxAddTilingKernel` for cache-aware tiled kernel execution. |
| **Sub-total** | **28** | |

Each of these is an independently shippable sub-PR. They can be parallelised once P2–P5 are done.

### Phase 7 — Legacy 1.0.1 compatibility (**25 missing**)

All from `vx_compatibility.h`, gated by `#ifdef VX_1_0_1_NAMING_COMPATIBILITY`. The OpenVX spec considers these *deprecated* — they exist so older OpenVX 1.0.1 application code can link against a 1.3 implementation unchanged.

Two sub-groups:

**Legacy accessor pattern** (16): `vxAccessImagePatch` / `vxCommitImagePatch`, `vxAccessArrayRange` / `vxCommitArrayRange`, `vxAccessDistribution` / `vxCommitDistribution`, `vxAccessLUT` / `vxCommitLUT`, `vxReadConvolutionCoefficients` / `vxWriteConvolutionCoefficients`, `vxReadMatrix` / `vxWriteMatrix`, `vxReadScalarValue` / `vxWriteScalarValue`, `vxComputeImagePatchSize`, `vxAddKernel`. These can mostly be implemented as thin wrappers around the existing 1.3 `vxMap*Patch` / `vxCopy*` APIs.

**Legacy Accumulate-* kernels** (6): `vxAccumulateImageNode`, `vxAccumulateWeightedImageNode`, `vxAccumulateSquareImageNode` and their three `vxu*` counterparts. These were vision kernels in 1.0.1; in 1.3.1 their behaviour is achievable with `vxMultiplyNode` + `vxAddNode` graphs, so the compatibility shim is straightforward.

**Misc** (3): `vxGetRemapPoint` / `vxSetRemapPoint` (re-imagined as `vxMapRemapPatch` / `vxCopyRemapPatch` in 1.3), `vxNormalizationLayer` (re-exported under `vx_khr_nn.h` in modern installs).

---

## 4. Cleanup items (cross-cutting Phase 8)

Things the audit surfaced that are bugs or footguns *unrelated* to missing API surface:

### 4.1 Broken / misleading exports (10)

| Symbol | Behaviour today | Right answer |
|---|---|---|
| `vxCopyNode` | Always returns `NULL` | Implement (graph-graph node copy) or remove the export and document gap |
| `vxCopyArray` | Returns success without copying | Implement using `vxMapArrayRange` + memcpy |
| `vxCopyRemap` | Returns success without copying | Implement using existing `vxMapRemapPatch` |
| `vxMapPyramidLevel` | Returns `-30` placeholder | Implement (delegates to per-level image map) |
| `vxCreateConvolutionFromPattern` | Always `NULL` | Implement |
| `vxCornerMinEigenValNode` | Factory exists, kernel not registered → empty node | Either register the kernel or remove the factory |
| `vxMeanShiftNode` | Same as above | Same |
| `vxDilate5x5Node` / `vxErode5x5Node` / `vxSobel5x5Node` | Factory + kernel registered for 5×5 names, but **no `dispatch_kernel_with_border_impl` arm** for them → `vxProcessGraph` returns `VX_ERROR_INVALID_KERNEL` | Either wire up dispatch (5×5 separable filter) or remove the factories. 5×5 variants are **not in 1.3.1 spec** — likely brought over from earlier 1.2 work. |
| `vxAllocateImageMemory` / `vxLockImage` / `vxMapImage` / `vxUnlockImage` / `vxUnmapImage` / `vxReleaseImageMemory` | All `NOT_IMPLEMENTED` / `NULL` returns | These are not 1.3.1 spec; they look like 1.0.1 patches that escaped through the compatibility refactor. Either move to P7 (legacy compat) or remove. |

### 4.2 Casing aliases (2)

rustVX exports `vxFASTCornersNode` / `vxuFASTCorners` (uppercase `FAST`). The 1.3.1 spec spelling is `vxFastCornersNode` / `vxuFastCorners` (camelCase). **Action**: add the spec-spelled aliases that delegate to the existing implementations; keep the uppercase variants for backwards compatibility with any in-tree callers. Costs ~6 lines per pair.

### 4.3 Non-spec extras worth keeping (28)

Helpers that rustVX uses internally and exports for convenience but aren't in any spec header: `vxCloneImage`, `vxCopyImage`, `vxCopyImagePlane`, `vxCopyPyramid`, `vxCopyScalarData`, `vxCopyTensor`, `vxCopyThreshold`, `vxQueryThresholdData`, `vxMoveArrayRange`, `vxRegisterPyramidLevelImage`, `vxSetObjectArrayItem`, `vxQueryGraphParameterAttribute`, `vxSetGraphParameterAttribute`, `vxQueryParameterFull`, `vxAccessDelayElement`, `vxCommitDelayElement`, `vxEnumerateTargets`, `vxQueryTarget`, `vxQueryTargetMetric`, `vxSetMatrixAttribute`, `vxComputeImagePattern`, `vxCreateImageFromROIH`, `vxCreateMetaFormat`, `vxCreateThresholdForImageUnified`, `vxCreateUniformImageFromHandle`, `vxGetImagePlaneCount`. Keep, but document under `docs/non-spec-exports.md` so downstream users know they're rustVX-private and unstable.

---

## 5. Phased rollout

### Phase 2 — Base API & UDO (target: **260 → 270 functions implemented**)

**Scope**: 10 functions.
**Effort**: S — couple of days. No new data types, no new kernel math.
**Dependencies**: none.
**Exit criteria**:
- All 10 functions present and pass round-trip tests (`create → query → map / set / get → release`).
- `cargo test --workspace` green, CTS conformance green, perf-gate green.
- Add CTS sections gated on these APIs to the matrix (`UserDataObject`, `LogEntry` smoke).

**Sub-tasks**:
1. `vxAddLogEntry` + `vxRegisterLogCallback` integration test that round-trips a log entry through the context's callback.
2. `vxSetGraphAttribute` mirror of `vxQueryGraph` — straightforward dispatch.
3. `vxRegisterKernelLibrary` paired with the existing `vxLoadKernels`.
4. `vx_khr_user_data_object.h` data type implementation: 7 functions + `vx_user_data_object` opaque handle, internal `Vec<u8>` storage, dirty-tracking for graph re-execution.

### Phase 3 — Enhanced Vision (non-tensor) kernels (target: **270 → 284**)

**Scope**: 14 functions (6 paired `Node`/`vxu*` + 2 single-mode catch-ups).
**Effort**: M — kernels with varying complexity. Bilateral and HOG are the hardest.
**Dependencies**: P2 (for kernels that consume UDO config — bilateral filter, HOG params).
**Exit criteria**:
- Each kernel passes its dedicated CTS section (extend the `enhanced-vision` workflow job to include them — currently filtered to `Min.*:Max.*` only).
- Both graph-mode and immediate-mode variants where the spec defines both.

**Per-kernel sub-tasks** (one PR each):
1. `vxuCopy` (already-paired with `vxCopyNode`; trivial slice memcpy).
2. `vxNonMaxSuppression` / `vxuNonMaxSuppression`.
3. `vxBilateralFilter` / `vxuBilateralFilter`.
4. `vxMatchTemplate` / `vxuMatchTemplate`.
5. `vxHoughLinesP` / `vxuHoughLinesP` (rust already has the node factory but it's in the "broken extras" — wire dispatch).
6. `vxLBP` / `vxuLBP`.
7. `vxHOGCells` / `vxuHOGCells`.
8. `vxHOGFeatures` / `vxuHOGFeatures`.

### Phase 4 — Tensor kernels (target: **284 → 298**)

**Scope**: 14 functions (7 paired tensor kernels).
**Effort**: L — tensor data plumbing exists but is partial; SIMD optimisation desired.
**Dependencies**: P2 (UDO sometimes used as bias/scale config); finish polishing existing `vx_tensor` handle.
**Exit criteria**:
- All seven tensor kernels pass their CTS sections.
- Tensor kernels have AVX2 SIMD paths in `openvx-core::simd_kernels` for the U8/S16/F32 cases the CTS exercises.
- `perf-gate` job validates no regression on existing kernels (P4 touches widely-shared tensor plumbing).

**Per-kernel sub-tasks**:
1. `vxTensorAdd` / `vxuTensorAdd` (start here — simplest).
2. `vxTensorSubtract` / `vxuTensorSubtract`.
3. `vxTensorMultiply` / `vxuTensorMultiply`.
4. `vxTensorTableLookup` / `vxuTensorTableLookup`.
5. `vxTensorConvertDepth` / `vxuTensorConvertDepth`.
6. `vxTensorTranspose` / `vxuTensorTranspose`.
7. `vxTensorMatrixMultiply` / `vxuTensorMatrixMultiply` (the big one; foundation for FullyConnectedLayer in P5).

### Phase 5 — Neural Network feature set + control-flow nodes (target: **298 → 308**)

**Scope**: 10 functions (8 NN layers + 2 control-flow nodes).
**Effort**: L — NN layers compose tensor primitives but each has its own parameter pack.
**Dependencies**: P4 (every NN layer is tensor-in / tensor-out).
**Exit criteria**:
- All 8 layers pass their CTS sections in the NN profile.
- A small "ImageNet inference" smoke test (e.g., the spec's referenced Caffe-style sample net) runs end-to-end at FHD against rustVX.
- The benchmark suite extends with a "Neural Network" category.

**Per-layer sub-tasks**:
1. `vxActivationLayer` (ReLU / sigmoid / tanh / etc. — element-wise; simplest).
2. `vxConvolutionLayer` (the heavy lifter; can reuse openvx-core's existing 2D convolution primitives).
3. `vxFullyConnectedLayer` (tensor matrix multiply — built on P4's last sub-task).
4. `vxPoolingLayer` (max / average pooling — straightforward).
5. `vxSoftmaxLayer` (output normalisation).
6. `vxLocalResponseNormalizationLayer` (cross-channel normalisation).
7. `vxROIPoolingLayer`.
8. `vxDeconvolutionLayer` (transposed convolution).
9. `vxScalarOperationNode` (base API, easiest control-flow op).
10. `vxSelectNode` (predicated copy).

### Phase 6 — Optional KHR extensions (target: **308 → 336**)

**Scope**: 28 functions across 8 extension headers.
**Effort**: variable; pipelining is L, the rest are S–M.
**Dependencies**: P2 (UDO needed for classifier).
**Exit criteria**: per-extension acceptance:

| Sub-PR | Functions | Acceptance |
|---|---|---|
| 6a Pipelining | 12 | Graph streaming smoke test ends to end |
| 6b Classifier | 3 | Round-trip import/release/scan with a tiny stub model |
| 6c XML I/O | 3 | Round-trip `vxImportFromXML(vxExportToXML(g))` for a known graph |
| 6d ICD | 3 | Multi-implementation loader works against rustVX as the single registered impl |
| 6e Buffer aliasing | 2 | Aliasing hint accepted; graph executor honours it in the simple case |
| 6f Import kernel from URL | 1 | Loads a `.so` from `file://` and registers its kernels |
| 6g Tiling kernels | 1 | Registers a tiling kernel and `vxProcessGraph` exercises it |
| 6h OpenCL interop | 3 | Out of scope while rustVX is CPU-only — stub with `VX_ERROR_NOT_SUPPORTED` and document why |

Sub-PRs can land in any order once P5 is in.

### Phase 7 — Legacy 1.0.1 compatibility (target: **336 → 361** — 100% of bundled headers)

**Scope**: 25 functions under `#ifdef VX_1_0_1_NAMING_COMPATIBILITY`.
**Effort**: S — almost all are thin wrappers over already-implemented 1.3 APIs.
**Dependencies**: nothing.
**Exit criteria**:
- Every function in `vx_compatibility.h` has an export.
- An integration test that flips on `VX_1_0_1_NAMING_COMPATIBILITY` and exercises the old `vxAccess` / `vxCommit` pattern round-trips correctly.
- Document under `docs/legacy-1.0.1.md` that these are deprecated and the modern API is preferred.

### Phase 8 — Cleanup (cross-cutting, run alongside P2–P7)

**Scope**: §4.1 (10 broken exports) + §4.2 (2 casing aliases) + §4.3 (28 helpers documented).
**Effort**: S — each item is 5–50 lines.
**Dependencies**: none.
**Exit criteria**:
- No exported function returns "fake success" or a placeholder error code; every export either does its job or explicitly `VX_ERROR_NOT_SUPPORTED`s with a documented reason.
- `docs/non-spec-exports.md` exists.
- The CI workflow runs `cargo test --workspace -- --include-ignored` so previously-flagged broken stubs now have round-trip tests.

---

## 6. Coverage trajectory

| Milestone | Implemented | Spec | Coverage |
|---|---:|---:|---:|
| Today                                      | 260 | 361 | 72.0% |
| After **P2** (Base API + UDO, +10)         | 270 | 361 | 74.8% |
| After **P3** (Enhanced Vision non-tensor, +14) | 284 | 361 | 78.7% |
| After **P4** (Tensor kernels, +14)         | 298 | 361 | 82.5% |
| After **P5** (NN + control-flow, +10)      | 308 | 361 | 85.3% |
| After **P6** (Optional KHR, +28)           | 336 | 361 | 93.1% |
| After **P7** (Legacy 1.0.1 compat, +25)    | 361 | 361 | **100%** |

Sanity check: `260 + 10 + 14 + 14 + 10 + 28 + 25 = 361` ✓.

P2 unlocks one new CTS feature set; P3 + P4 + P5 together unlock the entire Enhanced Vision + Neural Network profiles. P6 + P7 land 100% surface coverage but don't add new conformance profiles.

---

## 7. Testing strategy per phase

Each phase ships with:

1. **Unit tests** in the relevant `openvx-vision` / `openvx-image` / `openvx-core` crate, comparing kernel output against a known-good scalar reference (the existing pattern from `openvx-core::simd_kernels::tests`).
2. **CTS jobs** extended in `.github/workflows/conformance.yml` to include the new test sections — see existing per-feature-set jobs (`vision-color`, `vision-filters`, …, `enhanced-vision`). Each new phase typically adds one or two job filter expressions.
3. **Benchmark coverage** — each new kernel that has a meaningful runtime cost gets a corresponding `openvx-mark` bench entry (file an upstream PR against `kiritigowda/openvx-mark` if the kernel isn't in the bench harness yet).
4. **Perf-gate participation** — once a kernel is benched, it automatically becomes part of the PR-vs-main `perf-gate` job, so any P3+ kernel-level perf regression caught from there blocks merge.

---

## 8. Risks & open questions

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Tensor plumbing partial / unstable.** P4 will surface latent bugs in the existing `vx_tensor` handle code. | Land a "Tensor data type hardening" mini-PR before starting P4 (test coverage on `vxCreateTensorFromHandle`, `vxSwapTensorHandle`, etc.). |
| R2 | **NN profile is large.** Even with the layers reduced to tensor primitives, each layer has its own parameter pack and special-casing. | Defer to P5 only after the tensor work is solid; budget 2–3 weeks per layer for the harder ones. |
| R3 | **Pipelining requires graph-executor refactor.** Today the graph executor is synchronous `vxProcessGraph`-only. Streaming requires lifecycle hooks for enqueue/dequeue + an event queue. | Schedule pipelining (6a) last in P6, after the easier KHR extensions are done — gives time to design the executor refactor. |
| R4 | **OpenCL interop out of scope** while rustVX is CPU-only. | Stub the three OpenCL functions with `VX_ERROR_NOT_SUPPORTED` and document the limitation; revisit if rustVX ever adds a GPU back-end. |
| R5 | **5x5 morphology / Sobel — are they in 1.3.1?** The bundled headers don't declare them. They exist in rustVX as broken extras (§4.1). | P8 either deletes them or wires up the dispatch arm; do not promise 1.3.1 conformance for them. |
| R6 | **Conformance test data churn.** CTS sections we don't run today may have different golden image versions in newer CTS releases. | Pin the CTS submodule SHA per phase; bump deliberately and capture in the phase's PR description. |

---

## 9. Tracking

Each phase becomes one GitHub label / project column:

- `coverage/p2-base-udo`
- `coverage/p3-enhanced-vision`
- `coverage/p4-tensor`
- `coverage/p5-neural-network`
- `coverage/p6-khr-extensions`
- `coverage/p7-legacy-compat`
- `coverage/p8-cleanup`

Within each label, one issue per missing function. The lists in §3 of this doc are the source of truth for issue creation. Suggested issue title format: `[P3] Implement vxBilateralFilter / vxuBilateralFilter`.

A `docs/openvx-coverage-status.md` can be generated by a small script that reads `include/VX/*.h` + the workspace `pub extern "C"` exports, identical to the script that produced the §2 table — running that on every push to `main` and committing the result keeps the percentage in the README badge live.

---

## 10. References

- OpenVX 1.3 specification: <https://registry.khronos.org/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html>
- OpenVX 1.3.1 errata: <https://registry.khronos.org/OpenVX/specs/1.3.1/html/OpenVX_Specification_1_3_1.html>
- rustVX conformance status: [`README.md`](../README.md#conformance-status)
- Khronos sample-impl bug we filed during this audit (LaplacianPyramid Initializer hoist): <https://github.com/KhronosGroup/OpenVX-sample-impl/issues/59>
- openvx-mark LaplacianPyramid fix-up: <https://github.com/kiritigowda/openvx-mark/pull/4>

---

## Appendix A. Full missing-function list, alphabetical

For convenience — the 101 missing functions grouped per phase.

### P2 — Base API & UDO (10)

```
vxAddLogEntry
vxCopyUserDataObject
vxCreateUserDataObject
vxCreateVirtualUserDataObject
vxMapUserDataObject
vxQueryUserDataObject
vxRegisterKernelLibrary
vxReleaseUserDataObject
vxSetGraphAttribute
vxUnmapUserDataObject
```

### P3 — Enhanced Vision non-tensor (12)

```
vxBilateralFilterNode      vxuBilateralFilter
vxHOGCellsNode             vxuHOGCells
vxHOGFeaturesNode          vxuHOGFeatures
vxLBPNode                  vxuLBP
vxMatchTemplateNode        vxuMatchTemplate
vxNonMaxSuppressionNode    vxuNonMaxSuppression
                           vxuCopy
                           vxuHoughLinesP
```

### P4 — Tensor kernels (14)

```
vxTensorAddNode               vxuTensorAdd
vxTensorConvertDepthNode      vxuTensorConvertDepth
vxTensorMatrixMultiplyNode    vxuTensorMatrixMultiply
vxTensorMultiplyNode          vxuTensorMultiply
vxTensorSubtractNode          vxuTensorSubtract
vxTensorTableLookupNode       vxuTensorTableLookup
vxTensorTransposeNode         vxuTensorTranspose
```

### P5 — NN + control flow (10)

```
vxActivationLayer
vxConvolutionLayer
vxDeconvolutionLayer
vxFullyConnectedLayer
vxLocalResponseNormalizationLayer
vxPoolingLayer
vxROIPoolingLayer
vxSoftmaxLayer
vxScalarOperationNode
vxSelectNode
```

### P6 — Optional KHR (31)

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
