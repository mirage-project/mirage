# TPU Pallas Backend Changes

This document summarizes the current Pallas TPU work on branch `tpu`, why the changes were needed, and what was verified on the TPU VM.

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
- Added a shared graph-normalization pass and moved backend-independent lowering into it.
- Switched the NKI transpiler to use the shared graph normalizer instead of carrying its own copy of that logic.
- Wired `generate_pallas_program(...)` through the Python/Cython layer.
- Added Python runtime integration so generated Pallas code can be imported and executed through Mirage.

### Why this was necessary

TPU support needed the same basic integration shape as the existing backends:

- a backend-specific transpiler,
- Python bindings,
- and a runtime path that can execute generated kernels.

The graph normalizer was important for two reasons:

- Pallas and NKI both needed the same backend-independent decomposition and graph rebuild logic.
- Keeping that logic shared prevents the backend implementations from drifting apart.

The shared normalizer now handles the graph cloning and primitive lowering path that both backends need, including RMSNorm and supported forloop-accum rewrites.

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

- Removed non-essential CUDA assumptions from the TPU build path.
- Added non-CUDA stubs for legacy extension symbols that still need to link.
- Made Python imports tolerate JAX-only / TPU-only environments where `torch` is absent.
- Guarded optional imports that are unrelated to the Pallas path.

### Why this was necessary

Before backend debugging could even start, Mirage had to:

- build with `USE_CUDA OFF`,
- import without CUDA,
- and avoid crashing on TPU VMs because of unconditional CUDA- or torch-specific imports.

Without these fixes, TPU validation failed before any Pallas logic ran.

## 3. Added tests and a real superoptimize-based RMSNorm demo

### New files

- `tests/python/test_pallas_backend.py`
- `demo/pallas_rms_norm.py`

### What changed

- Added Python tests for supported Pallas codegen and for clear rejection of unsupported threadblock ops.
- Reworked the RMSNorm demo so it uses the real user path:
  - build a normal KN graph,
  - call `superoptimize(..., backend="pallas")`,
  - transpile the selected graph,
  - execute it on TPU,
  - compare against a JAX reference.

### Why this was necessary

The backend needed:

- regression coverage for codegen and API behavior,
- and a TPU demo that exercises the actual search + transpile + execute path.

This replaces the earlier temporary demo shape that bypassed the real superoptimize flow.

## 4. Fixed Pallas candidate selection in superoptimize

### Updated file

- `python/mirage/kernel.py`

### What changed

- `superoptimize(..., backend="pallas")` no longer returns the first search result blindly.
- It now:
  - errors clearly if search returns no candidates,
  - iterates over the candidate graphs,
  - runs `generate_pallas_program(...)` on each one,
  - selects the first graph that the Pallas transpiler accepts,
  - errors clearly if no candidate is transpileable.

### Why this was necessary

The first-result behavior was only a placeholder. A new backend needs to validate that the selected graph is actually lowerable before handing it back to the user.

## 5. Fixed search-space issues that blocked small Pallas RMSNorm examples

### Updated file

- `src/search/search_c.cc`

### What changed

- Added backend-specific default search candidates for Pallas when the user does not override them:
  - `forloop_range = 1`
  - `grid_dim = (1, 1, 1)`

### Why this was necessary

Two real search issues showed up during TPU validation:

1. Mirage often needs a no-op accum stage with `forloop_range = 1` to make threadblock outputs legal.
2. The non-CUDA search heuristics only proposed large grids, which meant small single-chip TPU examples had no viable customized-op grid candidate.

Adding these defaults made the RMSNorm demo searchable through the normal Pallas path.

## 6. Made the probabilistic verifier work on the pre-normalized search graph

### Updated files

- `src/kernel/customized.cc`
- `src/kernel/element_unary.cc`
- `src/kernel/element_binary.cc`
- `src/kernel/reduction.cc`

### What changed

- Added missing CPU fingerprint implementations for KN unary, KN binary, and KN reduction operators used by search.
- Expanded customized-kernel fingerprinting to cover the TB primitives that search can generate directly, including:
  - unary primitives such as `TB_SQUARE_OP`, `TB_SQRT_OP`, `TB_MUL_SCALAR_OP`,
  - binary primitives such as `TB_ADD_OP`, `TB_MUL_OP`, `TB_DIV_OP`, `TB_POW_OP`,
  - reduction primitives such as `TB_REDUCTION_*`,
  - forloop reduction-accum variants,
  - and `TB_RMS_NORM_OP`.
- Fixed undefined behavior in `KNCustomizedOp::fingerprint()` by returning `true` instead of falling off the end of the function.

### Why this was necessary

Search verification happens on the pre-normalized graph, not the backend-normalized Pallas graph.

That means the probabilistic verifier must be able to fingerprint the primitive KN/TB operators that search generates directly. Before these fixes:

- several KN fingerprint paths were still stubs,
- customized-kernel fingerprinting only handled a narrow subset of TB ops,
- and the customized fingerprint function had a missing return.

Those gaps were the reason the direct probabilistic Pallas search path was crashing.

## 7. TPU validation results

The following behaviors were verified on the TPU VM:

- Mirage builds and imports on a TPU-only setup.
- `generate_pallas_program(...)` works for supported graphs.
- `mi.search(..., backend="pallas", is_formal_verified=False)` now works for the RMSNorm demo graph.
- `superoptimize(..., backend="pallas")` finds RMSNorm candidate graphs through the normal default verifier path.
- `demo/pallas_rms_norm.py` runs end to end on TPU through search, transpile, and execute.

Observed TPU result for `demo/pallas_rms_norm.py`:

- discovered `13` candidate muGraphs
- output shape `(8, 8)`
- `max_abs_err = 1.0`
- `output_sum = 69767.5`

The non-blocking warning about `qwen3.builder` missing `safetensors` is still present on the TPU VM, but it does not affect the Pallas backend flow.

## 8. What remains out of scope

The current backend is still intentionally narrow:

- single-chip oriented,
- heuristic-driven,
- and limited to the initial supported subset of Mirage ops.

Still out of scope for this work:

- distributed TPU execution,
- topology-aware scheduling,
- remote DMA and collectives,
- broad full-graph Pallas coverage across all Mirage operators.
