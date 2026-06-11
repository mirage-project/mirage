---
name: test-mode
description: Guide for using MPK test mode to unit-test individual layers or multi-layer pipelines through the full compilation pipeline. Use when writing layer tests, debugging kernel output, or validating a new task end-to-end.
---

# MPK Test Mode

Test mode compiles and runs an MPK task graph **exactly once** and exits. It exercises the full pipeline — Python layer API, task registration, C++ code generation, nvcc compilation, and runtime dispatch — making it the primary tool for validating that a new layer or task works end-to-end. It runs the graph with the **user-supplied meta tensors used verbatim**, so a test can forge an arbitrary runtime status and the kernel sees exactly the values the test provides.

Test mode is selected by setting `params["test_mode"] = True` at construction time. Internally this defines `-DMPK_TEST_MODE` for the launcher build, which:
1. Auto-allocates (as **valid zero-filled** GPU buffers) any of the 10 meta tensors the test author didn't pass, so every layer sees a non-null pointer. It does **not** populate meaningful values.
2. **Skips `init_request_resources()`** (which would reset `step`/`request_ids`/`qo_indptr_buffer`/`paged_kv_*`/the page queue) and **skips `prepare_next_batch()`** (which in production fills those buffers). The user-supplied meta tensors therefore reach the kernels **unmodified** — a forged runtime status is preserved.
3. Runs the task graph for exactly one pass, then terminates the schedulers.

> **Each test owns its meta tensors.** Because `prepare_next_batch` does not run, a layer that reads a runtime value from a meta tensor sees only what the test put there. In particular, **token-parallel** kernels read their active-token count from `qo_indptr_buffer[max_num_batched_requests]` (the last entry) — notably `silu_mul` and `argmax`. If you use one of those and leave `qo_indptr_buffer` at its zero default, the kernel sees **0 active tokens** and silently writes nothing. Attention kernels additionally read `paged_kv_indptr_buffer`/`paged_kv_indices_buffer`/`paged_kv_last_page_len_buffer`. Set whatever your layer reads in `params["meta_tensors"]`. (Static-grid layers — `rmsnorm`, `linear`, `linear_with_residual`, the `moe_*` GEMMs — don't read these and need no meta setup.)

## Required: PyTorch Reference Comparison

**Every test mode file must include a PyTorch reference implementation** that computes the same operation, and must compare the MPK output against it numerically. A test that only runs the kernel without checking correctness is not a valid test — it only proves the kernel doesn't crash.

The reference should:
- Use plain `torch` ops (or `torch.nn.functional`) to implement the same math the layer performs.
- Run on the same input tensors as the MPK kernel (cast to a higher precision like `float32` if needed for a trustworthy reference).
- Be compared with a tolerance appropriate to the dtype: bf16 typically `atol=1e-2, rtol=1e-2`; fp16 similar; fp32 much tighter.

Use `torch.testing.assert_close(out, ref, atol=..., rtol=...)` and/or print `(out - ref).abs().max()` so failures surface immediately rather than silently producing wrong numbers.

### Where the reference lives: `pytorch_reference.py`

Per-layer test_mode files **must import their PyTorch reference from `pytorch_reference.py` in the same folder**, not redefine it inline. The folder layout is `tests/runtime_python/<arch>/sm100_<layer>/`, with one `pytorch_reference.py` per folder containing one function per in-scope layer. Both the new test_mode test (`test_<layer>_testmode.py`) and the existing kernel-wrapper test (`test_<layer>.py`) import from the same file, so they stay aligned on a single canonical reference.

If `pytorch_reference.py` does not yet exist for the layer, create it. If a kernel-wrapper test already exists with an inline reference, extract that reference into `pytorch_reference.py` and refactor the kernel-wrapper test to import from it.

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

# 5. Compare against a PyTorch reference
def torch_rmsnorm(x, w, eps=1e-6):
    x_f32 = x.to(torch.float32)
    rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * rms * w.to(torch.float32)).to(x.dtype)

ref = torch_rmsnorm(x, w)
print("Max diff:", (out - ref).abs().max().item())
torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

# 6. Cleanup
pk.finalize()
```

## What Test Mode Actually Does

`prepare_kernel` seeds the first scheduler event, the task graph runs once, and the schedulers terminate after the pass. `init_request_resources()` and `prepare_next_batch()` are **not** called:

```
init / launch:  meta tensors hold exactly the values the test supplied (no reset, no fill)
one pass:       the layer(s) under test run, reading meta tensors verbatim
terminate:      schedulers stop after the single pass
```

So to test a meta-tensor-dependent layer, **forge the meta tensors yourself** to describe the runtime status you want (e.g. a decode at an arbitrary step with a specific paged-KV layout). See `tests/runtime_python/test_mode/test_forge_meta_paged_attention_testmode.py` (forges `qo_indptr_buffer` / `paged_kv_*` across multi-page and non-contiguous layouts) and `test_qwen3_mlp_testmode.py` (sets `qo_indptr_buffer` so `silu_mul` sees the right active-token count).

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
| `max_num_pages` / `page_size` / `max_seq_length` | 1 | Bump these if your kernel's compile-time constants or your forged paged-KV layout need larger buffers |
| `world_size` / `mpi_rank` | 1 / 0 | For multi-GPU tests; set from `mpi4py.MPI.COMM_WORLD` |
| `use_cutlass_kernel` | False | Set `True` if your layer uses CUTLASS-based kernels |
| `meta_tensors` | `{}` | Auto-defaulted to **zeros**; **set the entries your layer reads** (e.g. `qo_indptr_buffer` for token-parallel kernels, `paged_kv_*` for attention) — see "Meta-Tensor Defaults" below |

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
- Profiler export: pass `params["profiler_tensor"]` and optional `params["trace_name"]` before `compile()`. After `pk()` returns, both `<trace_name>.perfetto-trace` and `<trace_name>.csv` are written. See "Profiling" below.

### `pk.finalize()`

Frees GPU resources (queues, events, task/event storage). Call when done.

## Meta-Tensor Defaults

Test mode auto-allocates any of the 10 meta tensors that you don't pass:

| Key | Default shape | Default dtype | Default content |
|---|---|---|---|
| `tokens` | `(1, max_seq_length)` | `int64` | zeros |
| `step` | `(total_num_requests,)` | `int32` | zeros |
| `prompt_lengths` | `(total_num_requests,)` | `int32` | filled with `max_num_batched_tokens` |
| `input_tokens` | `(max_num_batched_tokens,)` | `int64` | zeros — **set it if your layer reads it** |
| `output_tokens` | `(max_num_batched_tokens,)` | `int64` | zeros |
| `num_new_tokens` | `(1,)` | `int32` | zeros |
| `qo_indptr_buffer` | `(max_num_batched_requests + 1,)` | `int32` | zeros — **set it if your layer reads it** |
| `paged_kv_indptr_buffer` | `(max_num_batched_requests + 1,)` | `int32` | zeros — **set it if your layer reads it** |
| `paged_kv_indices_buffer` | `(max_num_pages,)` | `int32` | zeros — **set it if your layer reads it** |
| `paged_kv_last_page_len_buffer` | `(max_num_batched_requests,)` | `int32` | zeros — **set it if your layer reads it** |

`total_num_requests` is derived from `tokens.shape[0]` (defaults to 1).

**Set exactly what your layer reads** — values are NOT derived for you (no `prepare_next_batch`). Typical patterns:

```python
# Token-parallel kernel (silu_mul / argmax): supply the active-token count in the
# LAST entry of qo_indptr_buffer. Here: one request of `batch` active tokens.
qo = torch.zeros(max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
qo[max_num_batched_requests] = batch
params["meta_tensors"] = {"qo_indptr_buffer": qo}

# Paged attention: forge the full paged-KV layout for a decode over `seq_len`
# positions laid out across physical pages `pkv_indices`
# (see test_forge_meta_paged_attention_testmode.py).
params["meta_tensors"] = {
    "qo_indptr_buffer":              torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
    "paged_kv_indptr_buffer":        torch.tensor([0, num_pages], dtype=torch.int32, device="cuda"),
    "paged_kv_indices_buffer":       pkv_indices,                # physical page ids
    "paged_kv_last_page_len_buffer": torch.tensor([last_page_len], dtype=torch.int32, device="cuda"),
    "step":                          torch.tensor([seq_len - 1], dtype=torch.int32, device="cuda"),
}
```

The shape/dtype assertions that production runs through (e.g. `tokens.shape[1] == max_seq_length`, `qo_indptr_buffer.dtype == int32`) all run in test mode too — defaults satisfy them by construction; user overrides will fail loudly if they don't match.

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

## Profiling

Test mode supports profiling because `pk()` runs the standard `__call__` path. Profiling is opt-in: pass a `profiler_tensor` and the post-run hook writes both a Perfetto trace (for human inspection) and a CSV (for programmatic queries).

```python
params = PersistentKernel.get_default_init_parameters()
params["test_mode"] = True
params["trace_name"] = "rmsnorm_smoke"           # optional; defaults to f"mirage_{mpi_rank}"
params["profiler_tensor"] = torch.zeros(
    3000 * 128, dtype=torch.uint64, device="cuda"
)
pk = PersistentKernel(**params)

# ... attach tensors, register layers, pk.compile(...) as usual ...
pk()
torch.cuda.synchronize()
# Two files now exist alongside the script:
#   rmsnorm_smoke.perfetto-trace   ← drag into ui.perfetto.dev
#   rmsnorm_smoke.csv              ← query with scripts/parse_profile.py
```

The profiler buffer must be `uint64` on CUDA. `3000 * 128` entries is the conventional size used by the demos and is plenty for short test-mode runs. Each task event consumes 2 entries (BEGIN + END); buffer overflow would surface later as a `RuntimeError("dangling BEGIN ...")` from the CSV writer.

### CSV schema

One row per fully-paired task event (and one row per `kInstant` event with `duration_ns=0`):

| Column | Meaning |
|---|---|
| `task_type_id` | `TaskType` enum value (e.g. `253`) |
| `task_type_name` | Symbolic name (e.g. `TASK_LINEAR_SM100`) |
| `block_idx`, `group_idx` | Worker that executed the event |
| `event_no` | Per-worker execution counter |
| `begin_ts`, `end_ts` | Raw 32-bit `%globaltimer_lo` values (ns, wraps every ~4.3 s) |
| `duration_ns` | `(end_ts - begin_ts) mod 2^32` |

### Querying the CSV — `scripts/parse_profile.py`

All output is JSON; errors print `{"error": "..."}` and exit 2.

```bash
# What task types ran, with event counts
python scripts/parse_profile.py rmsnorm_smoke.csv --list

# Average runtime of one task type
python scripts/parse_profile.py rmsnorm_smoke.csv TASK_RMS_NORM_HOPPER --stat avg

# Min / max / avg / median in one shot
python scripts/parse_profile.py rmsnorm_smoke.csv TASK_LINEAR_SM100 --stat all

# Numeric task-type id is also accepted
python scripts/parse_profile.py rmsnorm_smoke.csv 253 --stat min
```

Sample output:
```json
{"task_type": "TASK_LINEAR_SM100", "count": 7040, "min_ns": 11776, "max_ns": 193856, "avg_ns": 26731.58, "median_ns": 31184.0}
```

For finer-grained analysis (per-worker breakdown, percentiles, outliers), `pandas.read_csv(...)` the file directly — the schema is stable and column names speak for themselves.

## Constraints

- **One execution pass** — the task graph runs once, then the schedulers terminate. `init_request_resources()` / `prepare_next_batch()` are skipped (meta tensors are used verbatim); no multi-iteration scheduling.
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

**Output is all zeros (no crash):**
- A token-parallel kernel (e.g. `silu_mul`, `argmax`) likely read `qo_indptr_buffer[max_num_batched_requests] == 0` because the test didn't set it (test mode does not run `prepare_next_batch` to fill it). Supply `qo_indptr_buffer` with the active-token count in its last entry — see "Meta-Tensor Defaults".

**Kernel hangs / never terminates:**
- Verify the active `mode` is `"offline"` (the default); `MPK_TEST_MODE` is only wired for MODE_OFFLINE.
- The MPK runtime assumes occupying the entire GPU. If other processes are running, they can interfere with scheduling and cause hangs. Always check GPU availability before running. And if it hangs, kill and rerun on other idle GPUs.

## Example Test Files

| File | What it tests |
|---|---|
| `tests/runtime_python/test_mode/test_rmsnorm_testmode.py` | Single layer (RMSNorm), default meta tensors |
| `tests/runtime_python/test_mode/test_diamond_fork_join_testmode.py` | Multi-task graph (synthetic fork+join) |
