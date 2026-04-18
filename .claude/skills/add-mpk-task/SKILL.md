---
name: add-mpk-task
description: Step-by-step guide for adding a new task implementation to Mirage Persistent Kernel (MPK). Use this when adding a new GPU operator (e.g., a new attention variant, normalization, activation) to the MPK megakernel.
---

You are helping the user add a new task to the MPK (Mirage Persistent Kernel) runtime. A "task" is a single fused GPU operation (one thread block's worth of work) that runs as a node in the megakernel's task graph.

## Task Lifecycle Overview

A task flows through 7 files across 4 layers:

```
Python (user API)
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

Add a new value to the `TaskType` enum. Pick a number in the appropriate range:
- **100–149**: Ampere (baseline)
- **150–198**: Hopper (SM90)
- **230–298**: Blackwell (SM100)
- **300–349**: Multi-GPU

```cpp
// Example: adding TASK_MY_OP in the Ampere range
TASK_MY_OP = 122,  // pick next available number in your range
```

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

  // No __syncthreads() needed after the last store — the runtime's
  // worker loop does a __syncthreads() after _execute_task() returns.
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

Add an `#include` for your new file if the architecture's `task_header.cuh` does not already pull it in via a wildcard:

```cpp
#include "tasks/ampere/my_task.cuh"   // add this line
```

Also add your `TaskType` to the `task_type_to_name` map in `src/kernel/runtime.cc` (search for the existing map entries like `{TASK_RMS_NORM, "TASK_RMS_NORM"}`):

```cpp
{TASK_MY_OP, "TASK_MY_OP"},
```

---

### Step 4 — `include/mirage/kernel/task_register.h`

Declare the new registration function in the `TaskRegister` class:

```cpp
int register_my_op_task(threadblock::Graph const &bgraph,
                        std::vector<int> const &params);
```

---

### Step 5 — `src/kernel/task_register.cc`

Implement the registration function. Its job is to:
1. Read tensor dimensions from the `bgraph` (the `TBGraph` built in Python).
2. Generate a C++ code string that calls your templated kernel with those dimensions.

```cpp
int TaskRegister::register_my_op_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params) {
  // params is whatever you pass from Python as the third arg to register_task().
  // params.size() == 0 if you pass nothing.
  assert(params.size() == 0);

  // bgraph.operators contains (num_inputs + num_outputs) TBInputOp nodes,
  // inputs first in registration order.
  int num_inputs  = 2;  // must match tb_graph.new_input() calls for inputs
  int num_outputs = 1;  // must match tb_graph.new_input() calls for outputs
  assert(bgraph.operators.size() == (size_t)(num_inputs + num_outputs));

  std::vector<tb::TBInputOp *> input_ops, output_ops;
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    auto *iop = static_cast<tb::TBInputOp *>(op);
    if (input_ops.size() < (size_t)num_inputs)
      input_ops.push_back(iop);
    else
      output_ops.push_back(iop);
  }

  // Extract tensor dimensions from the output tensor descriptor.
  // output_tensors[0] holds the STensor (shared memory tensor) shape.
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size  = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim  = output_ops[0]->output_tensors[0].dim[1];

  // For stride of a KN-level tensor, cast through owner_op:
  // kn::KNInputOp *kn_op = static_cast<kn::KNInputOp *>(
  //     output_ops[0]->dtensor.owner_op);
  // int output_stride = static_cast<int>(kn_op->input_strides[0]);

  // Generate the code string. "$" is a placeholder replaced with the
  // corresponding argument value by CodeKeeper::e().
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::my_op_impl<bfloat16, $, $>(", batch_size, hidden_dim);
  code.e("    task_desc->input_ptrs[0],");   // input
  code.e("    task_desc->input_ptrs[1],");   // weight
  code.e("    task_desc->output_ptrs[0],");  // output
  code.e("    1e-6f);");

  // register_task_variant deduplicates: same code string → same variant_id.
  return register_task_variant(TASK_MY_OP, code.to_string());
}
```

**Reading tensor properties from `bgraph`:**
- `input_ops[i]->dtensor` — the kernel-level DTensor for input i (global shape/strides).
- `output_ops[i]->dtensor` — the kernel-level DTensor for output i.
- `output_ops[i]->output_tensors[0]` — the threadblock-level STensor (may differ in dims/strides).
- `dtensor.dim[d]`, `dtensor.num_dims` — global tensor dimensions.
- `dtensor.owner_op` — the upstream KN operator; cast to `kn::KNInputOp *` to get `input_strides`.

**Injecting runtime metadata via `code.e()`:**
- `runtime_config.tokens` — pointer to the token buffer.
- `runtime_config.step[i]` — current decode step for request i.
- `runtime_config.qo_indptr_buffer` — paged attention indptr.
- `task_desc->task_metadata.request_id` — which request this task handles.
- `task_desc->task_metadata.kv_idx` — KV cache chunk index (for split-KV).

---

### Step 6 — `src/kernel/graph.cc` — `Graph::register_task()`

Add an `else if` branch mapping your task name string to the registration function:

```cpp
} else if (name == "my_op") {
  int variant_id = task_register->register_my_op_task(customized->bgraph, params);
  // Tuple: (num_inputs, num_outputs, TaskType, variant_id)
  // num_inputs/num_outputs must match what register_my_op_task expects.
  task_config[op] = std::make_tuple(2, 1, TASK_MY_OP, variant_id);
}
```

**`task_config` tuple fields:**
1. `num_inputs` — must equal the number of `input_ops` in `register_my_op_task`
2. `num_outputs` — must equal the number of `output_ops`
3. `TaskType` — the enum value you added in Step 1
4. `variant_id` — returned by `register_task_variant()`

Maximum: **7 inputs, 3 outputs** per task (hard limit in `runtime_header.h`).

---

### Step 7 — `python/mirage/mpk/persistent_kernel.py`

Add a Python method that users call to insert your task into the computation graph:

```python
def my_op_layer(
    self,
    input: DTensor,    # first input tensor
    weight: DTensor,   # second input tensor
    output: DTensor,   # output tensor
    grid_dim: tuple,   # (num_tasks_x, num_tasks_y, num_tasks_z)
    block_dim: tuple,  # MUST be (128,1,1) for Ampere or (256,1,1) for Hopper/Blackwell
):
    assert input.num_dims == 2
    assert output.num_dims == 2

    # TBGraph partition scheme: new_input(tensor, partition, forloop_dim, is_write)
    # partition: (-1,-1,-1) = whole tensor per task (no partitioning)
    #            (0,-1,-1)  = split along dim 0 (grid_dim.x tasks)
    #            (1,-1,-1)  = split along dim 1
    # forloop_dim: dimension iterated in forloop (-1 = none, 0 = first dim, ...)
    # is_write: True if this tensor is written by the task
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(input,  (0, -1, -1), 1, True)   # input, split on dim0
    tb_graph.new_input(weight, (-1, -1, -1), 0, True)  # weight, no split
    tb_graph.new_input(output, (0, -1, -1), 1, True)   # output, split on dim0

    self.kn_graph.customized([input, weight, output], tb_graph)
    # String name must exactly match the else-if branch in graph.cc.
    # params list corresponds to params[] in register_my_op_task().
    self.kn_graph.register_task(tb_graph, "my_op", [])  # [] = no params
```

You could reference /mpk-internals skill to futher understand how this works.

---

## Critical Constraints

### block_dim Must Match WORKER_NUM_THREADS

```
Ampere (SM80/86/89):   block_dim = (128, 1, 1)
Hopper (SM90):         block_dim = (256, 1, 1)
Blackwell (SM100):     block_dim = (256, 1, 1)
```

Defined in `include/mirage/persistent_kernel/tasks/common/worker_config.h`. The worker launch configuration uses this constant — a mismatch does **not** produce a compile error but will silently corrupt results because your kernel will have different warp/thread assumptions than what the scheduler expects. Use `mi.get_configurations_from_gpu(rank)` to probe the GPU if needed. In practice, use the correct `block_dim` based on `self.target_cc >= 90`.

### TBGraph Operator Order

`bgraph.operators` is ordered exactly as `tb_graph.new_input()` was called. The first `num_inputs` entries are inputs; the remaining `num_outputs` are outputs. The split in `register_my_op_task` must match this exactly.

### grid_dim Sizing

`grid_dim.x * grid_dim.y * grid_dim.z` = total number of task instances. Each becomes one thread block assigned to one worker SM. For good load balance, make the total task count a multiple of `num_workers`. The C++ runtime does not validate this — mismatches cause load imbalance or incorrect results.

### Variant Deduplication

`register_task_variant()` deduplicates by the generated code string. Two calls with the same template parameters produce the same code string and share a `variant_id`. You don't need to manage this manually.

### Architecture-Specific Tasks

If your task only makes sense for one GPU generation (e.g., uses TMA or WGMMA), name it with a suffix (`_hopper`, `_sm100`) and guard the TBGraph building with `if self.target_cc >= 90`. See `paged_attention_layer()` vs `paged_attention_hopper()` in `persistent_kernel.py` for the pattern.

### Tasks Must Be blockIdx-Agnostic

The persistent kernel runtime dispatches tasks to **arbitrary** worker thread blocks. A task CANNOT use `blockIdx.x/y/z` to determine its identity, compute batch offsets, or select experts.

**Anti-pattern — WRONG:**
```cpp
int batch_idx = blockIdx.x;  // WRONG: blockIdx is the worker ID, not the task ID
int expert_id = blockIdx.x % num_experts;  // WRONG: same reason
```

**Correct approach:** All per-task information is in the `TaskDesc` struct passed to `_execute_task()`:
- `task_desc->input_ptrs[i]` / `task_desc->output_ptrs[i]` — already point to the correct per-task data slice (partitioned by grid_dim via TBGraph)
- `task_desc->task_metadata.expert_offset` — which expert subset this task handles
- `task_desc->task_metadata.request_id` — which request this task belongs to

The runtime handles the mapping from grid coordinates to task metadata during task graph generation. Your kernel just reads from the pointers and metadata it receives.

---

## Verification

Adding a new task requires **three parts**:
1. **Kernel correctness** (Steps A–C) — Test the CUDA kernel directly via a pybind11 wrapper
2. **Pipeline correctness** (Step 8) — Test the full Python API → code generation → runtime path via test mode
3. **Performance benchmark** (Step 9) — Measure latency/throughput across representative shapes

### Step A — Add kernel wrapper to `runtime_kernel_wrapper.cu`

The wrapper file wraps each `__device__ __forceinline__` kernel in a `__global__` launcher and exposes it via pybind11. Follow the pattern used by existing tasks (e.g., `linear_kernel_wrapper` at line ~1230):

```cpp
// 1. Add a __global__ wrapper that calls your device function
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__global__ void my_op_kernel_wrapper(void const *input_ptr,
                                     void const *weight_ptr,
                                     void *output_ptr,
                                     float eps) {
  // You could modify the input ptr for different threadblocks to mimic the real runtime
  // (e.g., add blockIdx.x * BATCH_SIZE * HIDDEN_DIM * sizeof(T) to input_ptr for batch partitioning)
  kernel::my_op_impl<T, BATCH_SIZE, HIDDEN_DIM>(input_ptr, weight_ptr, output_ptr, eps);
}

// 2. Add a launch helper that hardcodes dims and sets shared memory size
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
void launch_my_op(void const *input_ptr, void const *weight_ptr,
                  void *output_ptr, float eps) {
  dim3 grid_dim(1, 1, 1);                 // Adjust as needed for testing your op
  dim3 block_dim(128, 1, 1);              // 128 for Ampere; 256 for Hopper/Blackwell
  size_t smem_size = 3 * HIDDEN_DIM * sizeof(T) + 128;  // input + weight + output buffers

  cudaFuncSetAttribute(my_op_kernel_wrapper<T, BATCH_SIZE, HIDDEN_DIM>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  my_op_kernel_wrapper<T, BATCH_SIZE, HIDDEN_DIM>
      <<<grid_dim, block_dim, smem_size>>>(input_ptr, weight_ptr, output_ptr, eps);
  cudaDeviceSynchronize();
}

// 3. Add the Python-facing C++ function with dimension dispatch
void my_op(torch::Tensor input, torch::Tensor weight, torch::Tensor output, float eps) {
  void const *input_ptr  = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void       *output_ptr = output.data_ptr();
  int hidden_dim = input.size(1);
  // dispatch on runtime dim; add cases for each size you want to test
  if (hidden_dim == 4096) {
    launch_my_op<bfloat16, 1, 4096>(input_ptr, weight_ptr, output_ptr, eps);
  } else {
    printf("Unsupported hidden_dim: %d\n", hidden_dim);
  }
}
```

Then register it in `PYBIND11_MODULE`:
```cpp
m.def("my_op", &my_op, "My new op kernel");
```

### Step B — Rebuild the test extension

```bash
pip setup.py build_ext --inplace   # rebuilds runtime_kernel.so
```

For Blackwell-specific tasks, use the corresponding setup in `tests/runtime_python/blackwell/sm100_{task}/setup.py` instead. Arch-specific setups pass `-DMIRAGE_GRACE_BLACKWELL` and `-gencode=arch=compute_100a,code=sm_100a`.

### Step C — Write and run the test script

Create `tests/runtime_python/test_my_op.py`:

```python
import torch
import runtime_kernel

dtype  = torch.bfloat16
device = "cuda"
hidden_dim = 4096

input  = torch.randn(1, hidden_dim, dtype=dtype, device=device)
weight = torch.randn(hidden_dim,    dtype=dtype, device=device)
output = torch.empty(1, hidden_dim, dtype=dtype, device=device)

runtime_kernel.my_op(input, weight, output, eps=1e-6)

# PyTorch reference
variance = input.pow(2).mean(-1, keepdim=True)
ref = input * torch.rsqrt(variance + 1e-6) * weight

print("Max abs error:", (output - ref).abs().max().item())
print("Ratio (kernel / torch):", (output / ref).flatten()[:8])
```

Run it:
```bash
cd tests/runtime_python
python test_my_op.py
```

A ratio close to 1.0 everywhere (or max abs error within bfloat16 rounding, ~1e-2) indicates a correct implementation.

---

### Step 8 — Runtime Test with `test_mode`

After verifying the kernel in isolation (Steps A–C), test it through the full MPK compilation pipeline using test mode. This validates the Python layer method (Step 7), task registration (Steps 5–6), code generation, and runtime dispatch end-to-end.

Create `tests/runtime_python/test_mode/test_my_op_testmode.py`. See the `/test-mode` skill for the complete API guide, examples, and debugging tips.

---

### Step 9 — Performance Benchmark

Create a benchmark alongside the kernel wrapper test at `tests/runtime_python/blackwell/<task>/bench_<task>.py`. It should:

1. Define at least 3–4 representative shape configurations (small, medium, production-scale).
2. Warm up the kernel (16+ iterations).
3. Measure latency using `torch.cuda.Event(enable_timing=True)` over 100+ repetitions.
4. Report average time (ms) per configuration.
