---
name: add-mpk-task
description: Step-by-step guide for adding a new task implementation to Mirage Persistent Kernel (MPK). Use this when adding a new GPU operator (e.g., a new attention variant, normalization, activation) to the MPK megakernel.
---

You are helping the user add a new task to the MPK (Mirage Persistent Kernel) runtime. A "task" is a single fused GPU operation (one thread block's worth of work) that runs as a node in the megakernel's task graph.

There are **two layers** to add a new operator:

1. **Kernel + registration** (Steps 1-6): the CUDA implementation, the `TaskType` enum, the task-name → registration-function dispatch, and the code-generation function. This part is the same for every task.
2. **Python catalog module** (Step 7): an `MPKModule` subclass that owns its own `forward()` (PyTorch reference), `auto_grid_dim()` (parallelism heuristic), and `compile()` (TBGraph + `register_task` wiring). New layers live under `python/mirage/mpk/layers/<family>/<my_module>.py`.

The legacy `pk.foo_layer()` methods on `PersistentKernel` are kept for back-compat but are **not** the path forward — adding a new layer should not require editing `python/mirage/mpk/persistent_kernel.py`.

## Task Lifecycle Overview

```
Python (catalog module)
  ├── MPKModule.forward()    — PyTorch reference (for correctness oracle)
  └── MPKModule.compile()    — builds TBGraph, calls register_task
        → graph.cc (name→type dispatch)
          → task_register.cc (code generation)
            → runtime_header.h (enum)
            → tasks/{arch}/{my_task}.cuh (CUDA kernel)
              → generated _execute_task() dispatch
                → persistent_kernel.cuh (runtime scheduler)
```

## Step-by-Step: 7 Files to Touch

---

### Step 1 — `include/mirage/persistent_kernel/runtime_header.h`

Add a new value to the `TaskType` enum. If your task uses TMA descriptors, also add it to the relevant range / dispatch table in `runtime.cc` (see the existing `TASK_SM100_TMA_START_TASK` band or one of the explicit `MLA_*` / `FP8_*` `if`-branches around line 1240-1268 of `src/kernel/runtime.cc`).

---

### Step 2 — `include/mirage/persistent_kernel/tasks/{arch}/{my_task}.cuh`

Create the CUDA device function. It **must** be `__device__ __forceinline__` — the runtime calls it directly from inside `_execute_task()`, not as a kernel launch.

**Template for a simple elementwise-style task:**

```cpp
#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

// Template parameters encode compile-time specializations extracted from
// the threadblock graph (tensor dims, strides). They are filled in by
// register_my_op_task() in task_register.cc.
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void my_op_impl(
    void const *input_ptr,   // task_desc->input_ptrs[0]
    void const *weight_ptr,  // task_desc->input_ptrs[1]
    void *output_ptr,        // task_desc->output_ptrs[0]
    float eps)
{
  extern __shared__ char smem[];

  // NUM_THREADS is 128 (Ampere) or 256 (Hopper/Blackwell), defined in
  // tasks/common/worker_config.h. Your kernel MUST be correct for both.
  // Use NUM_THREADS in loops, not a hardcoded constant.

  T const *d_input  = static_cast<T const *>(input_ptr);
  T const *d_weight = static_cast<T const *>(weight_ptr);
  T       *d_output = static_cast<T *>(output_ptr);

  // ... kernel logic ...
}

} // namespace kernel
```

**Key rules for the kernel:**
- Use `NUM_THREADS` (from `common_header.cuh`), never hardcode 128 or 256.
- Use `extern __shared__ char smem[]` for shared memory; the runtime allocates it.
- The function receives raw `void*` pointers; cast them yourself.
- `task_desc->input_ptrs[i]` maps to inputs in the order they were added via `tb_graph.new_input()`.
- `task_desc->output_ptrs[i]` maps to outputs in `tb_graph.new_input()` order after inputs.
- Access `runtime_config.tokens`, `runtime_config.step`, `runtime_config.qo_indptr_buffer`, etc. for metadata.

---

### Step 3 — `include/mirage/persistent_kernel/tasks/{arch}/task_header.cuh`

Add an `#include` for your new file. The megakernel codegen uses this bundle to find every `kernel::*_kernel` symbol — forgetting to include yours produces nvcc errors like `namespace "kernel" has no member ...`.

---

### Step 4 — `include/mirage/kernel/task_register.h`

Declare the new registration function in the `TaskRegister` class:

---

### Step 5 — `src/kernel/task_register.cc`

Implement the registration function. Its job is to:
1. Read tensor dimensions from the `bgraph` (the `TBGraph` built in Python).
2. Generate a C++ code string that calls your templated kernel with those dimensions.

```cpp
int TaskRegister::register_my_op_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params) {
  assert(params.size() == 0);
  int num_inputs  = 2;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)(num_inputs + num_outputs));

  std::vector<tb::TBInputOp *> input_ops, output_ops;
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    auto *iop = static_cast<tb::TBInputOp *>(op);
    if (input_ops.size() < (size_t)num_inputs) input_ops.push_back(iop);
    else                                       output_ops.push_back(iop);
  }

  int batch_size  = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim  = output_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::my_op_impl<bfloat16, $, $>(", batch_size, hidden_dim);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-6f);");

  return register_task_variant(TASK_MY_OP, code.to_string());
}
```

**Reading tensor properties from `bgraph`:**
- `input_ops[i]->dtensor` — kernel-level DTensor for input i (global shape/strides).
- `output_ops[i]->dtensor` — kernel-level DTensor for output i.
- `output_ops[i]->output_tensors[0]` — threadblock-level STensor (may differ in dims/strides).
- `dtensor.owner_op` cast to `kn::KNInputOp *` for `input_strides`.

**Injecting runtime metadata via `code.e()`:**
- `runtime_config.tokens` / `step[i]` / `qo_indptr_buffer` — meta-tensor pointers.
- `task_desc->task_metadata.request_id` / `expert_offset` / `kv_idx` — per-task fields.

---

### Step 6 — `src/kernel/graph.cc` — `Graph::register_task()`

Add an `else if` branch mapping your task name string to the registration function:

```cpp
} else if (name == "my_op") {
  int variant_id = task_register->register_my_op_task(customized->bgraph, params);
  task_config[op] = std::make_tuple(2, 1, TASK_MY_OP, variant_id);  // (num_inputs, num_outputs, TaskType, variant_id)
}
```

Maximum: **7 inputs, 3 outputs** per task (hard limit in `runtime_header.h`).

---

### Step 7 — Add a catalog module (NEW APPROACH)

Create `python/mirage/mpk/layers/<family>/<my_module>.py`. Inherit from `MPKModule` and implement the trio: `forward()` (eager PyTorch reference), `auto_grid_dim()` (parallelism heuristic), `compile()` (registers the task on `current_pk().kn_graph`).

```python
"""<one-line summary>.

Kernel: ``include/mirage/persistent_kernel/tasks/<arch>/my_task.cuh``
Task name: ``my_op``
"""
from __future__ import annotations
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule  # adjust dot count for your subpackage

__all__ = ["MyOp"]

GridDim  = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MyOp(MPKModule):
    """<≤6-line description + __init__ arg list>."""

    def __init__(self, hidden_dim: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.hidden_dim = hidden_dim
        # any nn.Parameter weights go here; load_state_dict will populate them
        self.weight = nn.Parameter(torch.empty(hidden_dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eager reference: <one-line math>."""
        # Reference implementation; this is the correctness oracle used by tests.
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + 1e-6) * self.weight).to(x.dtype)

    def auto_grid_dim(self, x_dt: Any) -> GridDim:
        """One CTA per token row, capped at ``num_workers``."""
        from ... import context as _ctx
        pk = _ctx.current_pk()
        # Target num_workers (148 on Blackwell B200) to saturate the runtime.
        # Document any kernel-side hard constraint that bounds the grid below
        # num_workers (e.g. "one CTA per expert", "MMA-M=128 alignment").
        return (min(int(pk.num_workers), pk.max_num_batched_tokens), 1, 1)

    def compile(
        self,
        x: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register one ``my_op`` task.

        Tensor contract:
          x      : (batch_size, hidden_dim) bf16, row-major contiguous
          weight : (hidden_dim,) bf16, nn.Parameter
          output : (batch_size, hidden_dim) bf16, allocated if None

        Notes (≤2 lines): hidden_dim must be a multiple of 128 (Ampere) / 64
        (Hopper/Blackwell); slice-offset kwargs are unsupported.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if grid_dim is None:  grid_dim = self.auto_grid_dim(x)
        if block_dim is None: block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        batch = x.dim(0)
        if output is None:
            out_dt = pk.new_tensor(dims=(batch, self.hidden_dim), dtype=x.dtype,
                                   name=f"{self.prefix}my_op_out")
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(output, name=f"{self.prefix}my_op_out")
        else:
            out_dt = output

        # ----- TBGraph construction (formerly pk.my_op_layer body) -----
        assert x.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x,    (0, -1, -1), 1, True)   # partition x on dim 0 (batch)
        tb_graph.new_input(w_dt, (-1, -1, -1), 0, True)  # weight: no partition
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True) # output: partition on batch
        pk.kn_graph.customized([x, w_dt, out_dt], tb_graph)

        # Architecture-specific task-name dispatch — read each arch's .cuh.
        if 100 <= pk.target_cc < 120:
            pk.kn_graph.register_task(tb_graph, "my_op_sm100")
        elif 90 <= pk.target_cc < 100:
            pk.kn_graph.register_task(tb_graph, "my_op_hopper")
        elif 80 <= pk.target_cc < 90:
            pk.kn_graph.register_task(tb_graph, "my_op")
        else:
            raise RuntimeError(f"MyOp: unsupported cc {pk.target_cc}")
        return out_dt
```

After creating the file, re-export the class in `python/mirage/mpk/layers/<family>/__init__.py` and in `python/mirage/mpk/layers/__init__.py`.

**Why this is the new path**:
- The catalog module owns its task wiring — no edits to `persistent_kernel.py`.
- `forward()` gives a free correctness oracle for tests.
- `auto_grid_dim()` keeps callers from re-deriving the heuristic.
- Composability: model authors compose `MyOp` like any `nn.Module`.

**Variant split convention** (apply when your task has 2+ behavioral variants):
- Variants that differ by kernel (e.g. `linear_sm100` vs `linear_swapAB_hopper`, FP8 vs bf16): **split into separate classes** in the same file. Share `__init__` / `forward` / `auto_grid_dim` via a private `_MyOpBase(MPKModule)` and override only `compile()` in each subclass.
- Variants that differ only by a numeric template parameter (e.g. `tp_size in {2,4,8}`): keep as a single class with a kwarg.
- Optionally keep a back-compat factory function (`MyOp(..., variant="foo")` returns the right subclass) when you have existing callers — see `MoETopkRouting` in `python/mirage/mpk/layers/moe/routing.py` for the pattern.

**Documentation convention** (used uniformly across the catalog):
- Module docstring ≤8 lines: what + which `.cuh`.
- Class docstring ≤6 lines: what + `__init__` arg list.
- `forward()` ≤3 lines: math equation.
- `auto_grid_dim()` ≤3 lines: parallelism axis + dominating constraint.
- `compile()` ≤15 lines including the **tensor contract** block. Every input/output gets: shape (named dims), dtype, layout, special attribute (TMA alignment, slice-override semantics, etc.). The contract is mandatory.
- DELETE: historical rationale, PR refs, dates, design alternatives.

---

## Legacy path (back-compat only)

If you must add a `pk.my_op_layer(...)` method on `PersistentKernel` itself (e.g. an existing builder needs the method), you can still do so in `python/mirage/mpk/persistent_kernel.py`. The new catalog module is preferred; the legacy method should just call the catalog module under the hood once one exists. New code should not require touching `persistent_kernel.py`.

---

## Critical Constraints

### block_dim Must Match Each Kernel's Documented NUM_THREADS

The default worker `WORKER_NUM_THREADS` (defined in `include/mirage/persistent_kernel/tasks/common/worker_config.h`) is:

```
Ampere (SM80/86/89):   block_dim = (128, 1, 1)
Hopper (SM90):         block_dim = (256, 1, 1)
Blackwell (SM100):     block_dim = (256, 1, 1)
```

Defined in `include/mirage/persistent_kernel/tasks/common/worker_config.h`. A mismatch does **not** produce a compile error but silently corrupts results. `MPKModule.default_block_dim()` returns the right value based on `current_pk().target_cc`.

### TBGraph Operator Order

`bgraph.operators` is ordered exactly as `tb_graph.new_input()` was called. The first `num_inputs` entries are inputs; the remaining are outputs.

### grid_dim Sizing

`grid_dim.x * grid_dim.y * grid_dim.z` = total task instances. **Target `current_pk().num_workers`** (148 on Blackwell) to saturate the runtime. If kernel constraints force a smaller grid (e.g. `kernel requires grid.y == num_kv_heads`), document the constraint in `auto_grid_dim()`'s docstring.

### Variant Deduplication

`register_task_variant()` deduplicates by the generated code string. Two calls with the same template parameters share a `variant_id`.

### Tasks Must Be blockIdx-Agnostic

The persistent kernel runtime dispatches tasks to **arbitrary** worker thread blocks. A task CANNOT use `blockIdx.x/y/z` to determine its identity, compute batch offsets, or select experts.

**Anti-pattern — WRONG:**
```cpp
int batch_idx = blockIdx.x;  // WRONG: blockIdx is the worker ID, not the task ID
int expert_id = blockIdx.x % num_experts;  // WRONG
```

**Correct:** per-task data lives in the `TaskDesc`:
- `task_desc->input_ptrs[i]` / `output_ptrs[i]` — already point at the correct per-task slice (partitioned by grid_dim via TBGraph)
- `task_desc->task_metadata.expert_offset` / `request_id` — per-task fields set during code-gen

The runtime resolves grid coordinates → per-task pointers during task graph generation.

---

## Verification

A new task should be tested **two ways**:

1. **Catalog test** at `tests/runtime_python/layers/test_<module>.py` — uses `test_mode=True`, calls `module.forward()` for the reference, runs `pk()` for the MPK output, compares with `torch.testing.assert_close`. This is the primary correctness oracle and is cheap (one file per module).
2. **Kernel-wrapper test + benchmark** at `tests/runtime_python/{arch}/sm100_<task>/` — wraps the `__device__` kernel in a `__global__` launcher via pybind11 so you can test the CUDA kernel without going through the megakernel codegen. Useful for low-level kernel debugging and performance benchmarking.

### Catalog test (Step 7-A) — preferred for correctness

Create `tests/runtime_python/layers/test_my_op.py`:

```python
import os, sys, torch, mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.<family>.my_op_module import MyOp

def test_my_op_testmode():
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    batch, hidden = 2, 4096
    x = torch.randn(batch, hidden, dtype=dtype, device=device)
    out = torch.zeros(batch, hidden, dtype=dtype, device=device)

    m = MyOp(hidden_dim=hidden).to(device, dtype)
    ref = m.forward(x)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["max_num_batched_tokens"] = batch
    params["max_num_batched_requests"] = batch
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    with pk.compile_scope():
        m.compile(x_dt, output=out)

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("PASSED")
    pk.finalize()

if __name__ == "__main__":
    test_my_op_testmode()
```

See `/test-mode` for the full pattern, and `tests/runtime_python/layers/test_rmsnorm.py` / `test_paged_attention.py` for canonical examples.

### Kernel-wrapper test + benchmark (Steps A-C, 9) — only when kernel-level debugging is needed

For each kernel, there can be a dedicated folder in `tests/runtime_python/{arch}/sm100_<task>/` hosting:
- `runtime_kernel_wrapper.cu` — `__global__` wrapper + pybind11 binding for the `__device__` impl.
- `setup.py` — builds the wrapper with arch-specific flags.
- `test_<task>.py` — direct CUDA invocation + numerical compare against a `pytorch_reference.py`.
- `bench_<task>.py` — perf measurements over 3-4 representative shapes.

This path is heavier — only invest in it when you need to profile the kernel in isolation. The catalog test is the daily-driver.
