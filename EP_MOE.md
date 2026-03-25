# EP MoE (Expert Parallelism MoE) — Implementation Overview

## What It Does

Integrates two EP MoE kernels into the MPK (Mirage Persistent Kernel) task graph:
- **Routing**: top-k + softmax over expert logits → token-major indices/weights
- **Combine**: weighted sum of expert outputs + residual add

The **dispatch** kernel (all-to-all token scatter) is currently skipped — it requires `cooperative_groups::grid.sync()` which is incompatible with the MPK worker/scheduler split.

---

## Data Flow

Each rank owns `num_experts / world_size` experts and `batch / world_size` tokens (logical).

```
GPU 0                                          GPU 1
─────────────────────────────────────────────────────────────────────

hidden_states [batch, hidden]                  hidden_states [batch, hidden]
        │                                              │
        ▼                                              ▼
  MoE Gate Linear                               MoE Gate Linear
  router_logits [batch, num_experts]            router_logits [batch, num_experts]
        │                                              │
        ▼                                              ▼
  ┌──────────────────┐                          ┌──────────────────┐
  │   EP Routing     │                          │   EP Routing     │
  │  top-k + softmax │                          │  top-k + softmax │
  └──────────────────┘                          └──────────────────┘
  routing_indices [batch, topk]                 routing_indices [batch, topk]
  routing_weights [batch, topk]                 routing_weights [batch, topk]
  dispatch_counts [world_size]                  dispatch_counts [world_size]
        │                                              │
        ▼                                              ▼
  ┌──────────────────┐   NVLink peer write      ┌──────────────────┐
  │   EP Dispatch    │ ────────────────────────► │   EP Dispatch    │
  │  grid.sync()     │ ◄──────────────────────── │  grid.sync()     │
  │  scatter tokens  │   NVLink peer write       │  scatter tokens  │
  └──────────────────┘                          └──────────────────┘
  sync_flags[0] = 1                             sync_flags[1] = 1
  recv_buf [recv_tokens, hidden]                recv_buf [recv_tokens, hidden]
        │                                              │
        ▼                                              ▼
  Expert FFN (local experts 0..k-1)             Expert FFN (local experts k..N-1)
  expert_outputs [batch, topk, hidden]          expert_outputs [batch, topk, hidden]
        │                                              │
        ▼                                              ▼
  ┌──────────────────┐                          ┌──────────────────┐
  │   EP Combine     │                          │   EP Combine     │
  │  spin on flags   │                          │  spin on flags   │
  │  weighted sum    │                          │  weighted sum    │
  │  + residual      │                          │  + residual      │
  └──────────────────┘                          └──────────────────┘
  output [batch, hidden]                        output [batch, hidden]
```

**Note:** Dispatch is **not wired into MPK** — `grid.sync()` is incompatible with MPK's
split worker/scheduler kernel. Routing and Combine are wired into MPK via `__device__` wrappers.

---

## MPK Integration Path

```
Python: mpk.ep_moe_routing_layer() / ep_moe_combine_layer()
        │  persistent_kernel.py
        │  → customized() + register_task("ep_moe_routing_distributed" / "ep_moe_all_to_all_combine")
        ▼
  graph.cc: register_task()
        │  → task_register->register_ep_moe_routing_distributed_task()
        │  → task_register->register_ep_moe_all_to_all_combine_task()
        ▼
  task_register.cc: emits _execute_task() CUDA snippet
        │  mirage::kernel::moe_routing_device_impl<T, NUM_EXPERTS, TOPK, WORLD_SIZE>
        │  mirage::kernel::all_to_all_combine_device_impl<T, HIDDEN_DIM, TOPK, WORLD_SIZE>
        ▼
  generate_task_graph() → nvcc → .so → launch_persistent_kernel()
```

---

## Key Files

| File | Role |
|------|------|
| `include/mirage/persistent_kernel/tasks/common/moe_routing_distributed.cuh` | Routing kernel: `__global__` + `__device__` impl |
| `include/mirage/persistent_kernel/tasks/common/all_to_all_combine_task.cuh` | Combine kernel: `__global__` + `__device__` impl |
| `include/mirage/persistent_kernel/tasks/common/all_to_all_dispatch_task.cuh` | Dispatch kernel: `__global__` only (not in MPK) |
| `include/mirage/persistent_kernel/persistent_kernel.cuh` | Includes both EP MoE headers |
| `src/kernel/task_register.cc` | Code generation for `_execute_task()` |
| `src/kernel/graph.cc` | `register_task()` dispatch cases |
| `src/kernel/runtime.cc` | `task_type_to_name` entries (required by assert) |
| `python/mirage/persistent_kernel.py` | Python API: `ep_moe_routing_layer`, `ep_moe_combine_layer` |

---

## Standalone Tests

```bash
cd tests/runtime_python
conda run -n mpk python setup.py build_ext --inplace
conda run -n mpk python test_moe_routing.py        # ✓ single-GPU routing
conda run -n mpk python test_moe_combine.py        # ✓ single-GPU combine
conda run -n mpk python test_ep_moe_multigpu.py    # ✓ multi-GPU: routing + dispatch + combine (2×B200, NVLink)
```
