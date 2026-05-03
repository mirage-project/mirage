---
name: add-mpk-model
description: Guide for adding a new model (e.g., Llama4, DeepSeek V3) to the MPK persistent kernel. Covers prerequisites check, demo structure, layer wiring, and testing.
---

You are helping the user add a new model to MPK. This is a context + guidelines skill (not a step-by-step recipe) because model implementations vary significantly depending on architecture (dense vs MoE, GQA vs MLA, etc.).

## Prerequisites Check

Before writing any model code, identify what the new model needs:

1. **List the model's layers**: embedding, normalization, attention (what variant?), feed-forward (dense or MoE?), output head.
2. **Check which layer methods already exist** in `python/mirage/mpk/persistent_kernel.py`. Search for `def *_layer` methods. Common ones: `embed_layer`, `rmsnorm_layer`, `linear_layer`, `silu_mul_layer`, `paged_attention_layer`, `moe_topk_softmax_routing_layer`, `moe_w13_linear_layer`, etc.
3. **For any missing layers**: use the `/add-mpk-task` skill to add them first. Each missing layer requires implementing a CUDA task kernel and Python layer method.
4. **Check the model's attention mechanism**: GQA (grouped-query attention) is standard for Qwen3/Llama. MLA (multi-latent attention) requires `mla_prefill_layer`/`mla_decode_layer`. Novel attention variants need new tasks.
5. **Check weight naming**: You'll need to map HuggingFace weight names to MPK names for the weight shard loader.

## Where Model Code Lives

Model implementations live in `demo/<model_name>/`, NOT in `python/mirage/mpk/models/`. The `models/` directory holds only base infrastructure (`GraphBuilder`, `MirageModelConfig`, model registry).

```
demo/<model_name>/
  demo.py                    # End-to-end inference demo
  models/                    # HuggingFace model files
    modeling_<model>.py      # HF model definition (for reference)
    configuration_<model>.py # HF config class
  <model>_shard_loader.py    # Weight name mapping + sharding (if multi-GPU)
```

**Reference implementations:**
- `demo/qwen3/` — Canonical dense transformer model
- `demo/deepseek_v3/` — MoE model (DeepSeek V3 with MLA + MoE)

## How to Build the Demo

The demo script follows this pattern (see `demo/qwen3/demo.py`):

```python
# 1. Parse args (model path, batching config, profiling flags)
# 2. Load model config from HuggingFace
# 3. Create MPKMetadata with runtime configuration
metadata = MPKMetadata(
    mode="offline",
    model_name="org/Model-Name",
    weight_from_model=True,
    max_num_batched_tokens=...,
    max_num_batched_requests=...,
    page_size=..., max_num_pages=...,
    # ...
)
# 4. Create MPK and build the computation graph
mpk = MPK(metadata)
mpk.build()    # Calls the model builder to wire layers
# 5. Compile the megakernel
mpk.compile()
# 6. Load request and run inference
mpk.load_new_request("Your prompt here")
mpk()
```

## Wiring Layers in the Builder

The builder (subclass of `GraphBuilder` or custom code in the demo) constructs the computation graph by calling layer methods on `PersistentKernel`:

```python
# Attach weight tensors from state dict
w_norm = pk.attach_input(state_dict["model.layers.0.input_layernorm.weight"], name="layer_0_norm")
w_qkv = pk.attach_input(qkv_weight, name="layer_0_wqkv")

# Create intermediate buffers
norm_out = pk.new_tensor(dims=(...), name="norm_out", io_category="cuda_tensor")

# Chain layer calls
pk.rmsnorm_layer(input=x, weight=w_norm, output=norm_out, grid_dim=(...), block_dim=(...))
pk.linear_layer(input=norm_out, weight=w_qkv, output=qkv_out, grid_dim=(...), block_dim=(...))
pk.paged_attention_layer(...)
# ... etc for each layer in the model
```

### grid_dim / block_dim Selection

- `block_dim`: Always `(128,1,1)` for Ampere, `(256,1,1)` for Hopper/Blackwell. Use `pk.target_cc >= 90` to choose.
- `grid_dim`: Depends on the layer. For batch-parallel layers (rmsnorm, embedding): `(max_num_batched_tokens, 1, 1)`. For linear layers, use the `grid_for_rmsnorm_linear_layer()` helper from the Qwen3 demo.

### Weight Shard Loader (Multi-GPU)

If the model uses tensor parallelism, create a shard loader mapping HuggingFace weight names to MPK names with sharding types:

```python
mapping = {
    "q_proj": {"name": "wq", "shard_type": [(ShardType.COL_PARALLEL,)]},
    "o_proj": {"name": "wo", "shard_type": [(ShardType.ROW_PARALLEL,)]},
    "input_layernorm": {"name": "attn_norm", "shard_type": [(ShardType.NONE,)]},
    # ...
}
```

See `demo/qwen3/qwen3_shard_loader.py` for the complete pattern.

## Attention Patterns

- **Standard GQA** (Llama, Qwen3): Use `paged_attention_layer` or `paged_attention_split_kv_layer`.
- **MLA** (DeepSeek V3): Use `mla_prefill_layer` / `mla_decode_layer`.
- **Novel attention**: Implement as a new task via `/add-mpk-task`.

## MoE Models

For Mixture-of-Experts models, the available layer methods are:
- `moe_topk_softmax_routing_layer` — Router (top-k gating with softmax)
- `moe_sigmoid_topk_routing_layer` — Router (sigmoid gating, DeepSeek V3 style)
- `moe_w13_linear_layer` / `moe_w13_fp8_layer` — First expert linear (gate+up fused)
- `moe_silu_mul_layer` — SiLU activation between expert linear layers
- `moe_w2_linear_layer` / `moe_w2_fp8_layer` — Second expert linear (down projection)
- `moe_mul_sum_add_layer` — Combine expert outputs with routing weights + residual

## Verification

1. **Test individual layers** using test mode before wiring the full model. See `tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py` for the canonical pattern — it tests gate+up linear, silu_mul, and down+residual individually and as a pipeline.
2. **Compile test**: `mpk.compile(output_dir="./debug_output")` to inspect generated CUDA code and task graph JSON.
3. **Correctness test**: Compare MPK output against a HuggingFace reference model on the same prompt. Outputs should match within bfloat16 tolerance (~1e-2 max abs error per token).


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


---
name: mpk-internals
description: Reference guide for the MPK compilation-to-runtime pipeline. Use when asked how MPK works internally, how compilation/code generation works, what happens at runtime, or when debugging the megakernel scheduler.
---

# MPK Internals: Compilation-to-Runtime Pipeline

This document traces the full lifecycle of an MPK megakernel from Python graph construction through CUDA compilation to persistent kernel execution.

## Pipeline Overview

```
Phase 1: Python Graph Building
  PersistentKernel.compile()
  → layer methods build KNGraph/TBGraph
  → kn_graph.generate_task_graph()
        |
        v
Phase 2: C++ Code Generation  (runtime.cc)
  Graph::generate_task_graph()
  → register_mugraph()     — builds task/event lists
  → print_task_graph()     — emits CUDA code + JSON
        |
        v
  Two artifacts:
    test.cu              — _init_persistent_kernel(), _execute_task(), Python C ext
    task_graph.json      — task descriptors, events, dependencies
        |
        v
Phase 3: CUDA Compilation
  nvcc test.cu → test.so  (Python extension module: __mirage_launcher)
        |
        v
Phase 4: Runtime Initialization
  init_persistent_kernel()
  → loads JSON, allocates GPU queues, builds RuntimeConfig
        |
        v
Phase 5: Runtime Execution
  launch_persistent_kernel()
  → prepare_kernel (reset queues)
  → worker_kernel + scheduler_kernel (persistent loop)
  → workers fetch tasks, wait on events, call _execute_task()
  → schedulers process events, enqueue tasks to workers
```

---

## Phase 1: Python Graph Building

### Key file: `python/mirage/mpk/persistent_kernel.py`

**Entry point:** `PersistentKernel.compile()`

The compilation method does the following in order:

1. **Generate task graph** — calls `self.kn_graph.generate_task_graph(num_gpus, my_gpu_id)` which bridges through Cython (`python/mirage/_cython/core.pyx`, `generate_task_graph()`) into C++. Returns `{"cuda_code": str, "json_file": str}`.

2. **Write files** — writes `test.cu` (CUDA code + HARD_CODE Python extension wrapper) and `task_graph.json` to a temp directory.

3. **Compile** — builds the nvcc command via `get_compile_command()` and calls `subprocess.check_call()`.

4. **Load module** — uses `importlib.util.spec_from_file_location()` to dynamically load the compiled `.so` as Python module `__mirage_launcher`. Extracts `init_func`, `launch_func`, `init_request_func`, `finalize_func`.

5. **Initialize runtime** — calls `init_func(...)` with meta-tensor pointers, worker/scheduler counts, and serving config.

### How layers build the graph

Each layer method (e.g., `rmsnorm_layer`, `linear_layer`, `moe_w13_fp8_layer`) does:
1. Create a `TBGraph` with `CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx)`
2. Call `tb_graph.new_input(dtensor, partition, forloop_dim, store_in_dmem)` for each input and output
3. Call `self.kn_graph.customized([tensors...], tb_graph)` to register the operator
4. Call `self.kn_graph.register_task(tb_graph, "task_name")` which dispatches to C++ `Graph::register_task()`

### HARD_CODE: the Python C extension wrapper

The `HARD_CODE` constant (top of `persistent_kernel.py`) is a C string appended to the generated CUDA code. It defines a Python extension module with four functions:
- `init_func` — parses Python args, calls C++ `init_persistent_kernel()`
- `launch_func` — takes a CUDA stream pointer, calls `launch_persistent_kernel(stream)`
- `init_request_func` — calls `init_request_resources()` (for online serving)
- `finalize_func` — calls `finalize_persistent_kernel()`

---

## Layer API: TBGraph Partition Scheme

Each layer method (e.g., `rmsnorm_layer`, `linear_layer`, `moe_w13_linear_layer`) builds a **TBGraph** that describes how the global tensors are sliced into per-task tiles. This section explains every parameter.

### `CyTBGraph` constructor

```python
tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx))
```

| Parameter | Meaning |
|---|---|
| `grid_dim` | `(x, y, z)` — number of task instances in each dimension. Total tasks = `x * y * z`. |
| `block_dim` | `(threads, 1, 1)` — threads per task. Must be `(128,1,1)` Ampere, `(256,1,1)` Hopper/Blackwell. |
| `forloop_range` | Number of forloop iterations (always **1** in MPK — see note below). |
| `reduction_dimx` | Tile size for the reduction dimension (always **64** in MPK). |

### `tb_graph.new_input()` — registering a tensor

```python
tb_graph.new_input(dtensor, input_map, forloop_dim, store_in_dmem)
```

Called for **every** tensor the task touches — both inputs and outputs. The first `num_inputs` calls register inputs; the remaining register outputs. This ordering must match `num_inputs`/`num_outputs` in `graph.cc`'s `task_config` tuple.

#### `input_map`: the partition tuple

A 3-element tuple `(mx, my, mz)` that maps **grid dimensions → tensor dimensions**:

| `input_map.x` value | Meaning |
|---|---|
| `-1` | `grid_dim.x` does **not** partition this tensor. Every task sees the full extent of every dimension. |
| `0` | `grid_dim.x` partitions **tensor dimension 0**. Task at grid position `gx` sees the slice `[gx * dim[0]/grid_x : (gx+1) * dim[0]/grid_x]` along dim 0. |
| `1` | `grid_dim.x` partitions **tensor dimension 1**. Same slicing logic on dim 1. |
| `2` | `grid_dim.x` partitions **tensor dimension 2**. |

`input_map.y` and `input_map.z` work identically for `grid_dim.y` and `grid_dim.z`.

**In short:** the value tells you *which tensor dimension* that grid axis splits. `-1` means "don't split by this grid axis."

#### `forloop_dim` (vestigial in MPK)

In the Mirage superoptimizer, `forloop_dim` and `forloop_range` together control tiled reduction loops within a TBGraph. However, **in MPK `forloop_range` is always 1**, which makes `forloop_dim` a no-op — the dimension division (`dim / 1`) and stride multiplier (`* 1`) have no effect regardless of what value you pass.

MPK task kernels handle their own internal tiling and reduction directly in CUDA (e.g., looping over the K dimension in a matmul). The TBGraph forloop mechanism is not used. You'll see various `forloop_dim` values in existing layer methods (e.g., `1`, `2`, `-1`), but they're all equivalent when `forloop_range=1`. By convention, existing code sets `forloop_dim` to the "reduction dimension" of the operation, but this is cosmetic.

#### `store_in_dmem`

- `True` — the per-task tensor slice lives in **device (global) memory**. Should be set to **True** for all MPK tensors.

### Annotated example: `moe_w13_linear_layer`

```python
def moe_w13_linear_layer(self, input, weight, moe_routing_indices,
                         moe_mask, output, grid_dim, block_dim):

    # input:              (batch_size, hidden_size)                        2D bf16
    # weight:             (num_experts, 2*intermediate_size, hidden_size)  3D bf16
    # moe_routing_indices:(num_experts, batch_size)                        2D int32
    # moe_mask:           (num_experts + 1,)                               1D int32
    # output:             (batch_size, num_experts_per_tok, 2*inter_size)  3D bf16

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))

    #                       tensor,              input_map,    forloop_dim*, store_in_dmem
    #  (* forloop_dim is vestigial in MPK — has no effect when forloop_range=1)

    tb_graph.new_input(input,               (-1, -1, -1),  1,           True)
    #  → No partition on any grid axis. Every task sees full (batch, hidden).

    tb_graph.new_input(weight,              (-1,  1, -1),  2,           True)
    #  → grid_dim.y partitions dim 1 (the 2*intermediate_size axis).
    #    Each task handles 2*inter_size / grid_dim.y rows of the weight matrix.

    tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1,           True)
    #  → No partition. Every task sees the full routing table.

    tb_graph.new_input(moe_mask,            (-1, -1, -1), -1,           True)
    #  → No partition. Every task sees the full mask.

    tb_graph.new_input(output,              (-1,  2, -1), -1,           True)
    #  → grid_dim.z partitions dim 2 (the 2*intermediate_size axis of the output).
    #    Each task writes to its slice of output columns.

    self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, output], tb_graph)
    self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm100")
```

### How partitioning connects to task pointers

At runtime, the partition tuple is resolved during task graph generation (`src/threadblock/graph.cc`). For each task instance (one grid coordinate), the code generator computes a **byte offset** from the tensor's base pointer:

```
per_task_ptr = base_ptr
             + blockIdx.x * stride_for(input_map.x)
             + blockIdx.y * stride_for(input_map.y)
             + blockIdx.z * stride_for(input_map.z)
```

These offsets are baked into the `TaskDesc` at init time (via JSON → `FullTaskDesc` → `TaskDesc`). The task kernel receives pre-offset pointers in `task_desc->input_ptrs[i]` and `task_desc->output_ptrs[i]` — this is why tasks are **blockIdx-agnostic**.

---

## Phase 2: C++ Code Generation

### Key file: `src/kernel/runtime.cc`

**Entry point:** `Graph::generate_task_graph()`

This function orchestrates all code generation:

1. **`register_mugraph()`** — walks the KNGraph operators and converts each into `FullTaskDesc` entries. For each `KN_CUSTOMIZED_OP`, it queries `task_config[op]` (a tuple of `num_inputs, num_outputs, TaskType, variant_id` set by `Graph::register_task()`) to determine the task type and variant. It also creates `EventDesc` entries for inter-task dependencies and populates `first_tasks` (the initial ready tasks).

2. **`print_task_graph()`** — generates two outputs:

   **Output 1: CUDA code** containing three generated functions:
   - `construct_task_graph()` — loads `task_graph.json` at runtime, parses it into `FullTaskDesc`/`EventDesc` vectors, and creates TMA descriptors for Hopper/Blackwell tasks.
   - `_init_persistent_kernel()` — sets up tensor pointers from `io_configs` (torch tensors, cudaMalloc buffers, shuffled tensors, NVSHMEM buffers). Called once during initialization.
   - `_execute_task()` — a giant if/else dispatcher that maps `(task_type, variant_id)` pairs to the actual kernel function calls. Each branch contains the code string generated by the corresponding `TaskRegister::register_*_task()` function.

   **Output 2: JSON task graph** — serializes all tasks, events, and dependencies (see JSON Schema section below).

### Key file: `src/kernel/graph.cc`

**`Graph::register_task()`** maps task name strings to registration functions:
```
"moe_w13_fp8_sm100" → register_moe_fp8_sm100_task() → TASK_MOE_W13_FP8_SM100
```

Each registration function (in `src/kernel/task_register.cc`) reads tensor dimensions from the TBGraph, generates a CUDA code string calling the templated kernel with those dimensions, and returns a `variant_id` via `register_task_variant()`. Same code string → same variant_id (deduplication).

---

## Phase 3: CUDA Compilation

### Key function: `get_compile_command()` in `persistent_kernel.py`

Builds the nvcc command with:
- **Includes**: Python headers, Mirage headers, CUTLASS, JSON library
- **Architecture flags**: `-gencode=arch=compute_90a,code=sm_90a` (Hopper), `compute_100a,code=sm_100a` (Blackwell)
- **Feature defines**: `-DMPK_ENABLE_TMA` (Hopper/Blackwell), `-DMIRAGE_GRACE_HOPPER` or `-DMIRAGE_GRACE_BLACKWELL`
- **Runtime defines**: `-DMODE_OFFLINE`, `-DMPK_MAX_NUM_BATCHED_REQUESTS=N`, `-DMPK_MAX_NUM_BATCHED_TOKENS=N`, `-DMPK_MAX_NUM_PAGES=N`, `-DMPK_PAGE_SIZE=N`, `-DMPK_MAX_SEQ_LENGTH=N`
- **Scheduler config**: `-DMAX_WORKER_PER_SCHEDULER=N` (computed from worker/scheduler ratio)
- **Output**: shared library (`.so`) as a Python extension module

For multi-GPU (NVSHMEM): adds `-rdc=true`, NVSHMEM/MPI includes and libraries.

---

## Phase 4: Runtime Initialization

### Key file: `include/mirage/persistent_kernel/persistent_kernel.cuh`

**`init_persistent_kernel()`** sets up the full runtime state:

1. **Meta-tensor mapping** — stores 10 meta-tensor pointers in `global_runtime_config` (step, tokens, input_tokens, output_tokens, num_new_tokens, prompt_lengths, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len).

2. **NVSHMEM init** (if multi-GPU) — calls `nvshmemx_init_attr()`, creates NVSHMEM teams for cross-GPU communication.

3. **Call generated `_init_persistent_kernel()`** — this loads the JSON task graph via `construct_task_graph()`, allocates GPU memory for intermediate tensors, and populates the `all_tasks`, `all_events`, `first_tasks` vectors.

4. **Allocate runtime queues on GPU**:
   - `worker_queues[2 * num_workers]` — per-worker task queues (local + remote). Each is a circular buffer of `TaskId` with length `per_worker_queue_len` (1024).
   - `sched_queues[num_schedulers + 1]` — per-scheduler event queues + one global broadcast queue. Circular buffers of `EventId`.
   - `worker_queue_last_ready_task_id[2 * num_workers]` — atomic counters for queue tail.
   - `sched_queue_last_ready_event_id[num_schedulers + 1]` — atomic counters for event queue tail.
   - `all_event_counters[num_events]` — atomic counters tracking how many times each event has been triggered.
   - `all_event_num_triggers[num_events]` — how many triggers each event needs before it's considered "ready".

5. **Copy task/event data to GPU** — `all_tasks`, `all_events`, `first_tasks` are copied to device memory.

6. **Set kernel attributes** — sets `cudaFuncAttributeMaxDynamicSharedMemorySize` for worker and scheduler kernels.

7. **Create streams and events** — separate CUDA streams for workers and schedulers (split mode), plus synchronization events.

8. **Call `init_request_resources()`** — launches `init_kernel` which initializes per-request state (step counters, page queues for MODE_OFFLINE/MODE_ONLINE).

---

## Phase 5: Runtime Execution

### Key file: `include/mirage/persistent_kernel/persistent_kernel.cuh`

### Launch: `launch_persistent_kernel(stream)`

1. **`prepare_kernel<<<>>>`** — resets all queue pointers and event counters to zero. Seeds the initial `EVENT_END_OF_TASK_GRAPH` event to scheduler[0], which kicks off the first iteration.

2. **Kernel launch** (two modes):
   - **Split mode** (`split_worker_scheduler = true`): launches `worker_kernel` and `scheduler_kernel` as separate kernels on separate streams. Workers get `WORKER_NUM_THREADS` threads per block; schedulers get 32 threads (1 warp). Synchronized via CUDA events. This is now the default mode.
   - **Unified mode**: launches a single `persistent_kernel` where blocks `[0, num_workers)` run `execute_worker()` and remaining blocks run `execute_scheduler()`.

### Worker loop: `execute_worker()`

Each worker thread block runs an infinite loop:

1. **Fetch tasks** — polls `worker_queue_last_ready_task_id[worker_id]` using `ld_acquire` until new tasks appear. Loads a batch of `TaskDesc` from the queue into shared memory (using `cp.async` for efficiency).

2. **Wait for dependencies** — if `task_desc->dependent_event != EVENT_INVALID_ID`, polls the event counter `all_event_counters[event_index]` until it reaches `num_triggers * iteration_num`. For NVSHMEM events, uses `nvshmem_signal_wait_until`.

3. **Execute task** — calls `_execute_task(task_desc, runtime_config)` which dispatches to the generated kernel code based on `(task_type, variant_id)`.

4. **Signal completion** — atomically increments `all_event_counters[trigger_event_index]`. If this was the final trigger for that event, enqueues the event to the appropriate scheduler's queue.

5. **Terminate** — when a `TASK_TERMINATE` task is received, the worker returns.

### Scheduler loop: `execute_scheduler()`

Each scheduler runs on a single warp (32 threads, only thread 0 active). Up to 4 schedulers can share one SM (4 warps):

1. **Fetch events** — polls `sched_queue_last_ready_event_id[sched_id]` for new events.

2. **Process event by type**:
   - `EVENT_LAUNCH_TASKS` / `EVENT_LAUNCH_MASSIVE_TASKS`: enqueue the task range `[first_task_id, last_task_id)` to worker queues in round-robin fashion.
   - `EVENT_LAUNCH_DEPENDENT_TASKS`: similar but increments `iteration_num` (for cross-iteration dependencies).
   - `EVENT_END_OF_TASK_GRAPH`: calls `prepare_next_batch()` to set up the next inference iteration (finalize previous batch, allocate KV cache pages, load new tokens). If `prepare_next_batch` returns false (no more work), calls `terminate_schedulers()`.
   - Termination event: enqueues `TASK_TERMINATE` to all workers and returns.

3. **Task assignment** — each scheduler owns a range of workers (`my_first_worker` to `my_last_worker`). It round-robins task assignments within this range, using local counters to track queue positions.

### Serving modes

`prepare_next_batch()` (defined per mode via `#ifdef`):
- **MODE_OFFLINE**: processes all requests in a fixed batch. Finishes previous tokens, allocates KV cache pages, sets up input_tokens for next step.
- **MODE_ONLINE**: supports dynamic request arrival. Checks for new requests via `next_request_id`.
- **MODE_ONLINE_NOTOKEN**: online mode without explicit token tracking.

---

## Task Graph JSON Schema

The `task_graph.json` file is the key intermediate artifact between code generation and runtime. Generated by `print_task_graph()` in `runtime.cc`, loaded by `construct_task_graph()` at init time.

The task graph JSON is very large and should never be read in a raw fashion. Always use `scripts/parse_task_graph.py` to parse and analyze it.

```json
{
  "all_tasks": [
    {
      "task_type": 0,           // TaskType enum value
      "variant_id": 0,          // code variant (same task, different dims)
      "inputs": [
        {
          "base_ptr": "tensor_name",  // matches io_configs key
          "offset": 0,                // byte offset from base
          "dims": [128, 4096],
          "strides": [4096, 1],
          "data_type": 1              // dtype enum
        }
      ],
      "outputs": [ /* same structure */ ],
      "trigger_event": 65537,   // EventId this task signals on completion
      "dependent_event": 65536, // EventId this task waits for before executing
      "request_id": -1,         // task_metadata: which request (-1 = all)
      "expert_offset": -1,      // task_metadata: MoE expert offset
      "kv_idx": -1,             // task_metadata: KV cache chunk index
      "merge_task_offset": -1,  // task_metadata: split-KV merge offset
      "task_offset": -1         // task_metadata: NVSHMEM team mapping
    }
  ],
  "all_events": [
    {
      "event_type": 0,          // EVENT_TERMINATION, EVENT_LAUNCH_TASKS, etc.
      "num_triggers": 1,        // how many task completions before this event fires
      "first_task_id": 0,       // range of tasks this event unlocks
      "last_task_id": 4
    }
  ],
  "first_tasks": [1, 2, 3]     // TaskIds ready to execute immediately
}
```

**Event types** (`runtime_header.h`):
- `EVENT_TERMINATION` (0) — terminate the kernel
- `EVENT_LAUNCH_TASKS` (1) — enqueue a range of tasks to one scheduler
- `EVENT_END_OF_TASK_GRAPH` (2) — end of one forward pass; triggers `prepare_next_batch`
- `EVENT_EMPTY` (3) — no-op
- `EVENT_LAUNCH_MASSIVE_TASKS` (4) — large task range split across all local schedulers
- `EVENT_LAUNCH_DEPENDENT_TASKS` (5) — cross-iteration dependent tasks

**TaskId encoding** (64-bit): `[iteration_num: upper 32 bits][position_index: lower 32 bits]`

**EventId encoding** (64-bit): `[nvshmem_tag: upper bits][gpu_id: middle 16 bits][event_index: lower 32 bits]`

---

## Key Data Structures

### `RuntimeConfig` (`runtime_header.h`)

Global configuration struct stored in GPU global memory. Contains:
- **Topology**: `num_workers`, `num_local_schedulers`, `num_remote_schedulers`, `num_gpus`, `my_gpu_id`
- **Queue pointers**: `worker_queues[][]`, `sched_queues[][]`, atomic tail counters
- **Task/Event storage**: `all_tasks[]`, `all_events[]`, `all_event_counters[]`, `first_tasks[]`
- **LLM metadata**: `step[]`, `tokens[]`, `input_tokens[]`, `output_tokens[]`, KV cache page management arrays
- **Execution control**: `split_worker_scheduler`, CUDA streams/events for synchronization

### `FullTaskDesc` (`runtime_header.h`)

Full task descriptor used during code generation and JSON serialization. Contains tensor descriptors with shapes/strides, event IDs, and task metadata.

### `TaskDesc` (`runtime_header.h`)

Compact runtime task descriptor (16-byte aligned). Contains only raw pointers (`input_ptrs[7]`, `output_ptrs[3]`), TMA descriptor pointers (if Hopper/Blackwell), event IDs, and task metadata. Constructed from `FullTaskDesc` at init time by resolving tensor names to GPU pointers.

### `TaskDesc::TaskMetadata` (union)

Per-task metadata packed into 8 bytes. Interpretation depends on task type:
- `expert_offset` (int) — MoE: which expert subset this task handles
- `request_id` (int16) + `kv_idx` (uint16) + `merge_task_offset` (int) — paged attention
- `task_offset` (int) — NVSHMEM team index for multi-GPU tasks

### `EventDesc` (`runtime_header.h`)

Event descriptor: `event_type`, `num_triggers` (how many completions needed), `first_task_id`/`last_task_id` (range of tasks this event unlocks).

### `TensorDesc` (`runtime_header.h`)

Tensor metadata for JSON serialization: `num_dims`, `base_ptr` (name string at codegen time, resolved to GPU pointer at init), `dim[]`, `stride[]`, `data_type`, optional TMA descriptor pointers.