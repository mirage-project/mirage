# TPU Pallas Backend Changes

This document summarizes the changes made on branch `tpu` to bring Mirage closer to a usable TPU backend through JAX/Pallas, and explains why each change was necessary.

## 1. Added an initial Pallas TPU backend

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

- Added a new standalone `pallas_transpiler` backend.
- Added a shared graph-normalization pass used to:
  - clone Mirage graphs into a fresh graph for backend lowering,
  - decompose `TB_RMS_NORM_OP`,
  - lower supported `TB_FORLOOP_ACCUM_*` variants into primitive ops,
  - preserve input/output graph structure for codegen.
- Switched the NKI transpiler to use the shared graph-normalization logic instead of keeping its own copy of that lowering path.
- Added Python/Cython bindings for Pallas transpilation so Python code can call `generate_pallas_program(...)`.
- Added Pallas runtime integration in `python/mirage/kernel.py`:
  - `backend="pallas"` dispatch,
  - dynamic compilation of generated Python,
  - JAX-based execution on TPU devices,
  - simple backend selection behavior in `superoptimize(..., backend="pallas")`.

### Why this was necessary

The repo already had backend-specific transpilers, but nothing for TPU/Pallas. The new backend needed:

- a backend-specific transpiler entrypoint,
- Python exposure so it could be called from Mirage’s public API,
- runtime integration so generated Pallas programs could actually execute,
- and shared graph preprocessing so the same backend-independent lowering logic did not have to be reimplemented in both NKI and Pallas.

Without the normalization pass, the Pallas backend would have had to directly handle composite threadblock operators like RMSNorm and complex accumulators. That would have made the initial TPU backend much more fragile and much harder to debug.

## 2. Made Mirage usable on TPU-only and non-CUDA installs

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

- Made `setup.py` stop linking CUDA headers and libraries when the selected backend is not CUDA.
- Guarded CUDA-only runtime types in `runtime_header.h` so non-CUDA builds can still compile the shared headers.
- Added a non-CUDA stub for the legacy CUDA transpiler entrypoint so the Python extension can still link on TPU-only builds.
- Added a non-CUDA stub for `cython_set_gpu_device_id(...)` so importing `mirage.core` does not fail on TPU-only installs.
- Added a non-CUDA stub for Triton codegen so the extension does not contain an undefined symbol for `generate_triton_program(...)`.
- Made Python imports more tolerant of TPU-only environments:
  - made `torch`-dependent helpers optional,
  - guarded MPK imports behind `ModuleNotFoundError`,
  - preserved runtime errors when CUDA- or torch-specific features are actually used.

### Why this was necessary

The original codebase assumed a CUDA environment in several places:

- build logic,
- runtime headers,
- extension symbols,
- and Python imports.

That meant the Mirage package could not even be imported cleanly on the TPU VM, even before exercising any Pallas logic. Several TPU debugging sessions failed first on missing CUDA symbols or unconditional PyTorch imports rather than on actual backend issues.

These compatibility fixes were required to:

- build Mirage on a TPU VM with `USE_CUDA OFF`,
- import `mirage` without installing the full CUDA/PyTorch stack,
- and make TPU testing fail for real Pallas backend reasons instead of basic packaging or linking issues.

## 3. Added validation for the new backend

### New files

- `tests/python/test_pallas_backend.py`
- `demo/pallas_rms_norm.py`

### What changed

- Added Python tests for:
  - successful Pallas code generation on a supported graph,
  - descriptive error reporting on unsupported threadblock ops.
- Added a working Pallas RMSNorm demo that:
  - builds an explicit customized threadblock RMSNorm graph,
  - uses the Pallas backend directly instead of relying on `superoptimize(..., backend="pallas")`,
  - runs the generated program,
  - and compares the result against a JAX reference.

### Why this was necessary

The backend needed a concrete validation path beyond just generating code.

The test file provides a stable regression surface for:

- the Python API,
- generated-code structure,
- and unsupported-op reporting.

The RMSNorm demo was especially important because it exposed two real issues during TPU validation:

1. The original demo depended on `superoptimize(..., backend="pallas")` to discover a suitable graph, but that path was not producing one for the RMSNorm example.
2. Threadblock outputs in Mirage are expected to come after an accumulation op. A direct `TB_RMS_NORM_OP -> TB_OUTPUT_OP` path was not valid in practice, so the demo had to insert a no-op `forloop_accum` when `forloop_range=1`.

The final demo reflects the actual supported Pallas path instead of relying on behavior the backend does not yet implement.

## 4. Bugs found and fixed during TPU validation

These issues were not part of the initial design, but were discovered while testing on a real TPU VM.

### Pallas output-shape codegen bug

In `src/pallas_transpiler/transpile.cc`, the generated `out_shape` originally used an output tensor symbol before that symbol existed. This produced invalid generated Python during execution.

This was fixed by emitting the output dtype from DTensor metadata directly instead of referencing the not-yet-created output variable.

### Optional PyTorch import issues

The TPU VM did not have `torch` installed. Mirage initially imported `torch` too early and too broadly, which prevented use of the JAX/Pallas path even though the TPU backend itself does not require torch.

This was fixed by:

- making top-level torch imports optional where possible,
- raising clear runtime errors only when torch-specific functionality is actually used.

### Cython runtime annotation issue

`python/mirage/_cython/core.pyx` contained self-referential annotations inside the `dtype` class that broke module initialization in the TPU environment. Those annotations were removed so the module could import reliably.

## 5. Outcome

At the end of this work:

- Mirage has an initial runnable Pallas TPU backend.
- The backend is exposed through the Python API.
- Mirage can be built and imported on a TPU VM without a CUDA runtime.
- The Pallas backend has regression tests.
- The Pallas RMSNorm demo runs successfully through the supported customized-kernel path.

## 6. Scope limits that still remain

This work does not make the TPU backend fully feature-complete. The current implementation is still intentionally narrow:

- single-chip oriented,
- heuristic-based,
- and limited to the subset of Mirage ops that were targeted for the first milestone.

In particular, the following areas are still intentionally out of scope or only partially supported:

- distributed TPU execution,
- topology-aware scheduling,
- remote DMA and collectives,
- broad `superoptimize(..., backend="pallas")` coverage,
- and full support for every existing Mirage kernel and threadblock operator.
