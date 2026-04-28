---
name: test-mode
description: Guide for using MPK test mode to unit-test individual layers or multi-layer pipelines through the full compilation pipeline. Use when writing layer tests, debugging kernel output, or validating a new task end-to-end.
---

# MPK Test Mode

Test mode compiles and runs an MPK task graph **exactly once** and exits. It exercises the full pipeline — Python layer API, task registration, C++ code generation, nvcc compilation, runtime dispatch, and the persistent runtime's metadata setup (`init_kernel` + `prepare_next_batch`) — making it the primary tool for validating that a new layer or task works end-to-end.

Test mode is selected by setting `params["test_mode"] = True` at construction time. Internally this defines `-DMPK_TEST_MODE` for the launcher build, which:
1. Auto-allocates any meta tensors the test author didn't pass (so paged-attention / embedding / sampling layers see valid `qo_indptr_buffer`, `paged_kv_*`, `input_tokens`, etc.).
2. Lets `init_request_resources()` and `prepare_next_batch` run normally — the same code paths production uses.
3. Forces `prepare_next_batch`'s always-finalize shortcut on iter 1, which returns false and terminates the scheduler after exactly one task-graph pass.

## Quick Start

```python
import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# 1. Configure
num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
params = PersistentKernel.get_default_init_parameters()
params["test_mode"] = True
params["num_workers"] = num_workers
params["num_local_schedulers"] = num_schedulers
pk = PersistentKernel(**params)

# 2. Create tensors and attach (both inputs AND outputs)
x   = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
w   = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
out = torch.zeros(16, 4096, dtype=torch.bfloat16, device="cuda")

x_dt   = pk.attach_input(x, name="x")
w_dt   = pk.attach_input(w, name="w")
out_dt = pk.attach_input(out, name="out")

# 3. Build layer(s)
block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
pk.rmsnorm_layer(input=x_dt, weight=w_dt, output=out_dt,
                 grid_dim=(16, 1, 1), block_dim=block_dim)

# 4. Compile and run — same call as production; MPK_TEST_MODE is baked into the .so
pk.compile(output_dir="./test_output")   # saves .cu and .json for debugging
pk()
torch.cuda.synchronize()

# 5. Compare — `out` tensor is now modified in-place
ref = torch_rmsnorm(x, w)
print("Max diff:", (out - ref).abs().max().item())

# 6. Cleanup
pk.finalize()
```

## What Test Mode Actually Does

The launcher's first scheduler event is always `EVENT_END_OF_TASK_GRAPH`, so `prepare_next_batch` fires *before* iter 0. Concretely:

```
init_kernel:           zero step / request_ids / qo_indptr / paged_kv_indptr; seed page_queue
1st END_OF_TASK_GRAPH (iter_num=0): prepare_next_batch fills meta tensors for iter 0 → returns true
iter 0:                the test — layer-under-test runs with valid meta tensors
2nd END_OF_TASK_GRAPH (iter_num=1): prepare_next_batch finalizes (MPK_TEST_MODE always-finalize)
                                    → no new prefills (next_request_id == total_num_requests)
                                    → returns false → terminate
```

So tests of meta-tensor-dependent layers (paged attention, MoE routing, embedding, sampling) work — they read the values that `prepare_next_batch` wrote.

## API Reference

### `PersistentKernel.get_default_init_parameters()` (classmethod)

Returns a dict with safe defaults for test mode. You **must** set `params["test_mode"] = True` — it is not in the defaults.

Commonly overridden keys:

| Key | Default | When to override |
|---|---|---|
| `test_mode` | (not present) | **Always** set to `True` |
| `num_workers` | 1 | Set from `mirage.get_configurations_from_gpu(0)` |
| `num_local_schedulers` | 4 | Set from `mirage.get_configurations_from_gpu(0)` |
| `max_num_batched_tokens` | 1 | Set to your test's batch size if the task kernel uses this compile-time constant |
| `max_num_batched_requests` | 1 | Same as above |
| `max_num_pages` / `page_size` / `max_seq_length` | 1 | Bump these so `prepare_next_batch` can fit your prefill (`max_num_pages * page_size >= prompt_length`) |
| `world_size` / `mpi_rank` | 1 / 0 | For multi-GPU tests; set from `mpi4py.MPI.COMM_WORLD` |
| `use_cutlass_kernel` | False | Set `True` if your layer uses CUTLASS-based kernels |
| `meta_tensors` | `{}` | Auto-defaulted; **override only the entries that drive your test scenario** (typically `prompt_lengths` and/or `tokens`) — see "Meta-Tensor Defaults" below |

### `mirage.get_configurations_from_gpu(rank)`

Returns `(num_workers, num_schedulers)` tuned for the GPU at the given rank. Always use this rather than hardcoding — the values depend on SM count and architecture.

### `pk.attach_input(tensor, name)`

Registers a PyTorch CUDA tensor with the computation graph. Returns a `DTensor` for use in layer calls.

- Call for **every** tensor — inputs, weights, AND outputs.
- Output tensors are modified **in-place** when the kernel runs.
- Tensor must be **contiguous** (row-major / C-order).
- Name must be **unique** across all attached tensors.

### `pk.compile(output_dir=None)`

Generates CUDA code, compiles with nvcc, loads the resulting `.so` module.

- **Set `output_dir`** to save `test_rank0.cu` and `task_graph_rank0.json` — essential for debugging compilation errors or incorrect results.
- Compilation can be slow (1–10+ minutes) depending on which task kernels are instantiated.

### `pk()` — Launch the kernel

Same call as production. In test mode the launcher was compiled with `-DMPK_TEST_MODE` so it terminates after one task-graph pass. The previous `pk.run_test_mode()` method has been removed; use `pk()` directly.

- Must be called **after** `compile()`.
- **Does not synchronize** — call `torch.cuda.synchronize()` before reading output tensors.
- Optional `default_stream=stream` kwarg if you don't want the current stream.
- Profiler trace export: pass `params["profiler_tensor"] = torch.zeros(N, dtype=torch.uint64, device="cuda")` and optional `params["trace_name"]` before `compile()`. After `pk()` returns, a `<trace_name>.perfetto-trace` file is written.

### `pk.finalize()`

Frees GPU resources (queues, events, task/event storage). Call when done.

## Meta-Tensor Defaults

Test mode auto-allocates any of the 10 meta tensors that you don't pass:

| Key | Default shape | Default dtype | Default content |
|---|---|---|---|
| `tokens` | `(1, max_seq_length)` | `int64` | zeros |
| `step` | `(total_num_requests,)` | `int32` | zeros |
| `prompt_lengths` | `(total_num_requests,)` | `int32` | filled with `max_num_batched_tokens` |
| `input_tokens` | `(max_num_batched_tokens,)` | `int64` | zeros (filled by `prepare_next_batch`) |
| `output_tokens` | `(max_num_batched_tokens,)` | `int64` | zeros |
| `num_new_tokens` | `(1,)` | `int32` | zeros |
| `qo_indptr_buffer` | `(max_num_batched_requests + 1,)` | `int32` | zeros (filled by `prepare_next_batch`) |
| `paged_kv_indptr_buffer` | `(max_num_batched_requests + 1,)` | `int32` | zeros (filled by `prepare_next_batch`) |
| `paged_kv_indices_buffer` | `(max_num_pages,)` | `int32` | zeros (filled by `prepare_next_batch`) |
| `paged_kv_last_page_len_buffer` | `(max_num_batched_requests,)` | `int32` | zeros (filled by `prepare_next_batch`) |

`total_num_requests` is derived from `tokens.shape[0]` (defaults to 1).

**Override only what your test scenario requires.** Typical patterns:

```python
# Single prefill of length N (controls qo_indptr_buffer / paged_kv_* via prepare_next_batch)
params["meta_tensors"] = {
    "prompt_lengths": torch.tensor([N], dtype=torch.int32, device="cuda"),
}

# Specific prompt content (e.g. for embedding-layer tests that read input_tokens)
params["meta_tensors"] = {
    "prompt_lengths": torch.tensor([N], dtype=torch.int32, device="cuda"),
    "tokens": torch.tensor([[101, 7592, 2088, ...]], dtype=torch.int64, device="cuda"),
}

# Multi-request batch — total_num_requests inferred from tokens.shape[0]
params["meta_tensors"] = {
    "tokens": torch.zeros((4, max_seq_length), dtype=torch.int64, device="cuda"),
    "prompt_lengths": torch.tensor([16, 8, 32, 4], dtype=torch.int32, device="cuda"),
}
```

The shape/dtype assertions that production runs through (e.g. `tokens.shape[1] == max_seq_length`, `prompt_lengths.dtype == int32`) all run in test mode too — defaults satisfy them by construction; user overrides will fail loudly if they don't match.

## Multi-Layer Pipeline Example

Multiple layers can be chained with intermediate tensors. From the Qwen3 dense MLP pattern:

```python
# Gate+Up linear → SiLU-Mul → Down+Residual

# Attach weights separately, then shuffle for interleaved gate/up layout
w_gate_dt = pk.attach_input(w_gate, name="w_gate")
w_up_dt   = pk.attach_input(w_up, name="w_up")
w_gatedup_dt = pk.shuffle_tensors(
    inputs=[w_gate_dt, w_up_dt],
    shuffled_dim=0,
    num_groups=num_tasks // 2,
    name="w_gatedup",
)

# Layer 1: Gate+Up fused linear
pk.linear_layer(input=input_dt, weight=w_gatedup_dt, output=mlp_mid_dt,
                grid_dim=(num_tasks, 1, 1), block_dim=block_dim)

# Layer 2: SiLU activation * element-wise multiply
pk.silu_mul_layer(input=mlp_mid_dt, output=silu_out_dt,
                  grid_dim=(num_tasks // 2, 1, 1), block_dim=block_dim)

# Layer 3: Down projection + residual add
pk.linear_with_residual_layer(input=silu_out_dt, weight=w_down_dt,
                              residual=residual_dt, output=mlp_out_dt,
                              grid_dim=(hidden_size // 64, 1, 1), block_dim=block_dim)
```

**Key pattern:** intermediate tensors (`mlp_mid`, `silu_out`) are pre-allocated and attached via `attach_input` so they can be inspected after execution if needed. For a runnable multi-task test see `tests/runtime_python/test_mode/test_diamond_fork_join_testmode.py`.

## Multi-GPU Tests

Test mode supports `world_size > 1`. Each rank is independent — auto-defaults are deterministic functions of kernel params, so they produce identical values on every rank.

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()
torch.cuda.set_device(rank)

params = PersistentKernel.get_default_init_parameters()
params["test_mode"] = True
params["world_size"] = world_size
params["mpi_rank"] = rank
# ... rest of setup is the same as single-GPU
pk = PersistentKernel(**params)
# ... attach + register layers (incl. NVSHMEM ops like pk.allreduce_layer)
pk.compile(output_dir=...)
pk()
torch.cuda.synchronize()
```

**Launch convention:**

```bash
LD_PRELOAD=$NVSHMEM_HOME/lib/libnvshmem_host.so \
mpirun --np 2 -x LD_PRELOAD -x LD_LIBRARY_PATH -x NVSHMEM_HOME \
    python tests/runtime_python/test_mode/test_multigpu_rmsnorm_testmode.py
```

The `LD_PRELOAD` is required so `dlopen()`-loaded launcher modules resolve `nvshmem_selected_device_transport` and other NVSHMEM-versioned symbols. This is an existing NVSHMEM 3.x quirk, unrelated to test mode.

## Constraints

- **One execution pass** — the task graph runs once and `prepare_next_batch` returns false on its second call. No multi-iteration scheduling.
- **Meta tensors auto-allocated** — pass overrides only for entries your test scenario depends on. Defaults are sized from kernel-level params (`max_num_batched_tokens`, `max_num_batched_requests`, `max_num_pages`, `max_seq_length`, etc.), so bump those if your test needs larger buffers.
- **`MPK_TEST_MODE` is a compile-time flag** — switching `test_mode` between True and False requires re-running `pk.compile()`; the same launcher .so isn't reusable across modes.

## Debugging Tips

**Compilation fails:**
- Check `<output_dir>/test_rank0.cu` for the generated code. Search for your task name in the `_execute_task()` function.
- Check `<output_dir>/task_graph_rank0.json` for the task graph. This file might be extremely long; don't read it raw — use `scripts/parse_task_graph.py`.

**Incorrect dimension splitting:**
- The MPK layers require `input_map` for each associated tensor to specify how dimensions are split across the grid. If grid/block dimensions don't divide tensor dimensions correctly, the kernel may read/write out of bounds, producing NaNs or wrong results.

**Incomplete task attributes:**
- Ensure all required attributes for each task are correctly specified in the compilation logic in `runtime.cc`. Missing/incorrect attributes cause undefined behavior or compilation errors.

**Kernel hangs / never terminates:**
- Verify `total_num_requests` is set to match the number of in-flight test requests (typically 1, derived from `tokens.shape[0]`). If `next_request_id` never reaches `total_num_requests`, `prepare_next_batch` will keep returning true and iterations will not stop.
- Verify the active `mode` is `"offline"` (the default). `MPK_TEST_MODE` is designed to layer on top of MODE_OFFLINE's `prepare_next_batch`; other modes are not supported.

**Verifying that `prepare_next_batch` actually ran:**
- After `pk()` returns, read back `pk.meta_tensors["step"][0]`. It should equal `prompt_lengths[0]` — `prepare_next_batch`'s Step 1.1 advances `step` by `num_tokens` on the second call. See `test_prepare_next_batch_testmode.py` for the canonical assertion.

## Example Test Files

| File | What it tests |
|---|---|
| `tests/runtime_python/test_mode/test_rmsnorm_testmode.py` | Single layer (RMSNorm), default meta tensors |
| `tests/runtime_python/test_mode/test_prepare_next_batch_testmode.py` | Verifies `prepare_next_batch` runs and populates metadata (asserts `step` advanced) |
| `tests/runtime_python/test_mode/test_diamond_fork_join_testmode.py` | Multi-task graph (synthetic fork+join) |
| `tests/runtime_python/test_mode/test_multigpu_rmsnorm_testmode.py` | Multi-GPU smoke test under `mpirun -n 2` |
