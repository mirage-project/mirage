# TPU Pallas Backend Changes

This document summarizes the changes made on branch `tpu`, why they were needed, and what was verified on a real TPU VM.

## 1. Added a standalone Pallas TPU backend

### New files

- `include/mirage/pallas_transpiler/transpile.h`
- `src/pallas_transpiler/transpile.cc`
- `include/mirage/transpiler/graph_normalizer.h`
- `src/backend_transpiler/graph_normalizer.cc`

### Updated files

- `src/nki_transpiler/transpile.cc`
- `python/mirage/_cython/CCore.pxd`
- `python/mirage/_cython/core.pyx`
- `python/mirage/kernel.py`

### What changed

- Added a new `pallas_transpiler` backend that emits Python/JAX/Pallas code.
- Added a shared graph-normalization pass used by NKI and Pallas to:
  - clone Mirage graphs into a fresh graph for backend lowering,
  - decompose `TB_RMS_NORM_OP`,
  - lower supported `TB_FORLOOP_ACCUM_*` variants into simpler primitive ops,
  - preserve input/output structure and guid mappings needed by codegen.
- Wired `generate_pallas_program(...)` through the Python/Cython layer.
- Added Pallas runtime integration so generated code can be imported and executed through the Mirage Python API.

### Why this was necessary

Mirage already had backend-specific transpilers. TPU support needed the same shape of integration:

- a dedicated backend entrypoint,
- a Python-visible transpilation API,
- and a runtime path that could execute generated Pallas kernels on TPU.

The shared graph normalizer was needed so backend-independent lowering logic did not have to be duplicated again inside the Pallas backend.

## 2. Made Mirage buildable and importable on TPU-only systems

### New file

- `src/backend_transpiler/transpile_stub.cc`

### Updated files

- `setup.py`
- `include/mirage/persistent_kernel/runtime_header.h`
- `src/kernel/device_memory_manager.cc`
- `src/kernel/triton_code_gen.cc`
- `python/mirage/__init__.py`
- `python/mirage/utils.py`

### What changed

- Made non-CUDA builds stop depending on CUDA-only headers and symbols.
- Added non-CUDA stubs for legacy backend entrypoints that the Python extension still expects.
- Made Python imports tolerate TPU/JAX-only environments where `torch` is absent.
- Guarded optional imports that were not relevant to the TPU path.

### Why this was necessary

Before any Pallas code could be debugged, Mirage had to:

- build with `USE_CUDA OFF`,
- import without a CUDA runtime,
- and avoid failing on unconditional torch- or CUDA-specific imports.

Without these compatibility fixes, TPU testing failed before reaching any actual Pallas backend logic.

## 3. Added Pallas backend tests and TPU demos

### New files

- `tests/python/test_pallas_backend.py`
- `demo/pallas_rms_norm.py`

### What changed

- Added Python tests for:
  - successful Pallas code generation on a supported graph,
  - descriptive rejection of unsupported threadblock operators.
- Added a TPU demo for RMSNorm that now uses the proper user path:
  - build a normal KN graph,
  - call `superoptimize(..., backend="pallas")`,
  - transpile the selected graph,
  - execute it on TPU,
  - compare against a JAX reference.

### Why this was necessary

The backend needed both:

- a stable regression surface for codegen and API behavior,
- and an end-to-end TPU validation path that exercised the actual `superoptimize` flow rather than a hand-built customized graph.

## 4. Fixed the Pallas superoptimize candidate-selection logic

### Updated file

- `python/mirage/kernel.py`

### What changed

- `superoptimize(..., backend="pallas")` no longer returns `all_graphs[0]` blindly.
- It now:
  - raises a clear error if search returns no candidates,
  - iterates through the candidate graphs,
  - calls `generate_pallas_program(...)` on each one,
  - selects the first graph that the Pallas transpiler accepts,
  - raises a clear error if none of the candidates are transpileable.

### Why this was necessary

The original implementation was only a placeholder. It treated the first search result as valid without checking whether the Pallas backend could lower it. That was not defensible for a new backend and made failures difficult to interpret.

## 5. Fixed real search issues blocking RMSNorm on Pallas

### Updated files

- `src/search/search_c.cc`
- `python/mirage/kernel.py`

### What changed

- Added backend-specific search defaults for Pallas:
  - include `forloop_range = 1` when the user does not provide `franges`,
  - include `grid_dim = (1, 1, 1)` when the user does not provide `griddims`.
- Made the user-facing `superoptimize(..., backend="pallas")` path use the formal verifier by default.

### Why this was necessary

Two separate search problems showed up during TPU validation of RMSNorm:

1. `forloop_range = 1` was not in the default search space, even though Mirage often needs a no-op accum stage to make threadblock outputs legal.
2. The non-CUDA search heuristics only proposed large grid sizes, so a small single-chip example like the RMSNorm demo had no viable customized-op grid candidate at all.

Adding `frange = 1` and `grid_dim = (1, 1, 1)` for Pallas fixed the search space so RMSNorm candidates could actually be discovered.

The formal-verifier default was added because the probabilistic verifier still crashes on this Pallas RMSNorm search space. The formal path is the one that was verified successfully end to end on TPU.

## 6. Added missing RMSNorm fingerprint support for customized threadblock graphs

### Updated file

- `src/kernel/customized.cc`

### What changed

- Added a missing `TB_RMS_NORM_OP` fingerprint implementation in `KNCustomizedOp::fingerprint`.

### Why this was necessary

Even after search began exploring the right graph shapes, RMSNorm still needed threadblock-level fingerprint support in the customized-kernel verifier path. Without this case, RMSNorm-containing customized graphs could not be checked correctly by the probabilistic verifier.

This fix was necessary, but it was not sufficient by itself to make the probabilistic verifier path stable. The direct probabilistic Pallas search path still segfaults for the RMSNorm case.

## 7. TPU validation results

The following behaviors were verified on the TPU VM:

- Mirage builds and imports on a TPU-only setup.
- `generate_pallas_program(...)` works for supported graphs.
- `superoptimize(..., backend="pallas")` now finds RMSNorm candidate graphs for the demo case.
- A real superoptimized RMSNorm graph found through the formal verifier transpiles successfully with the current Pallas backend.
- `demo/pallas_rms_norm.py` now runs through the intended path and completes successfully on TPU.

Observed TPU result for `demo/pallas_rms_norm.py`:

- discovered `13` candidate muGraphs
- output shape `(8, 8)`
- `max_abs_err = 1.0`
- `output_sum = 69767.5`

The non-blocking warning about `qwen3.builder` missing `safetensors` is still present on the TPU VM, but it does not affect the Pallas backend flow.

## 8. What is still unresolved

The current Pallas path is substantially better than the original version, but it is not complete.

Known remaining issues:

- The probabilistic verifier path for `mi.search(..., backend="pallas", is_formal_verified=False)` still segfaults on the RMSNorm search space.
- The current Pallas superoptimize flow relies on the formal verifier by default for correctness and stability.
- The transpiler has been validated on the RMSNorm path through normalized primitive ops, not as a direct lowering of raw `TB_RMS_NORM_OP`.
- The backend remains intentionally narrow:
  - single-chip oriented,
  - heuristic-driven,
  - and limited to the first supported subset of Mirage ops.

Still out of scope for this work:

- distributed TPU execution,
- topology-aware scheduling,
- remote DMA and collectives,
- broad full-graph Pallas coverage across all Mirage operators,
- and a complete fix for the probabilistic verifier on the new search space.
