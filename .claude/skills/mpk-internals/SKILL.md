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
