---
name: test-mode
description: Guide for using MPK test mode to unit-test individual layers or multi-layer pipelines through the full compilation pipeline. Use when writing layer tests, debugging kernel output, or validating a new task end-to-end.
---

# MPK Test Mode

Test mode compiles and runs an MPK task graph **exactly once**, without meta-tensors, batching, or persistent-loop iteration. It exercises the full pipeline — Python layer API, task registration, C++ code generation, nvcc compilation, and runtime dispatch — making it the primary tool for validating that a new layer or task works correctly end-to-end.

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

# 4. Compile and run
pk.compile(output_dir="./test_output")   # saves .cu and .json for debugging
pk.run_test_mode()
torch.cuda.synchronize()

# 5. Compare — `out` tensor is now modified in-place
ref = torch_rmsnorm(x, w)
print("Max diff:", (out - ref).abs().max().item())

# 6. Cleanup
pk.finalize()
```

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
| `use_cutlass_kernel` | False | Set `True` if your layer uses CUTLASS-based kernels |
| `meta_tensors` | `{}` | Some layers need stubs (e.g., `qo_indptr_buffer`) — see multi-layer example |

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

### `pk.run_test_mode()`

Launches the compiled task graph **once** on the current CUDA stream.

- Only available when `test_mode=True`.
- Must be called **after** `compile()`.
- **Does not synchronize** — call `torch.cuda.synchronize()` afterward before reading output tensors.

### `pk.finalize()`

Frees GPU resources (queues, events, task/event storage). Call when done.

## Multi-Layer Pipeline Example

`tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py` demonstrates chaining multiple layers:

```python
# Gate+Up linear → SiLU-Mul → Down+Residual  (Qwen3 dense MLP)

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

**Key pattern:** intermediate tensors (`mlp_mid`, `silu_out`) are pre-allocated and attached via `attach_input` so they can be inspected after execution if needed.

**Meta-tensor stubs:** some layers (e.g., `linear_with_residual_layer`) may reference `qo_indptr_buffer` at compile time. Pass a minimal stub:
```python
params["meta_tensors"] = {
    "qo_indptr_buffer": torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
}
```

## Constraints

- **One execution pass** — the task graph runs once and terminates. No iteration, no `prepare_next_batch`.
- **Meta-tensors optional** — pass `{}` or minimal stubs. Missing pointers become null; the kernel must not dereference them.

## Debugging Tips

**Compilation fails:**
- Check `<output_dir>/test_rank0.cu` for the generated code. Search for your task name in the `_execute_task()` function.
- Check `<output_dir>/task_graph_rank0.json` for the task graph. This file might be extremely long, dont read it in a raw fashion. Use `scripts/parse_task_graph.py` to read the task graph.

**Incorrect dimension splitting:**
- The MPK layers require `input_map` for each associated tensor to specify how dimensions are split across the grid. If the grid or block dimensions don't divide the tensor dimensions correctly, the kernel may read/write out of bounds, causing NaNs or incorrect results.

**Incomplete task attributes:**
- Ensure all required attributes for each task are correctly specified in the compilation logic, in `runtime.cc`. Missing or incorrect attributes can lead to undefined behavior or compilation errors.

## Example Test Files

| File | What it tests |
|---|---|
| `tests/runtime_python/test_mode/test_rmsnorm_testmode.py` | Single layer (RMSNorm) |
| `tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py` | Multi-layer pipeline (Qwen3 MLP) |
