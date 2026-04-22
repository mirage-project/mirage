# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Mirage Persistent Kernel (MPK): a compiler + runtime that fuses an entire LLM inference pass (compute + TP communication) into a single persistent CUDA megakernel. The active branch is `mpk`, which is where almost all current development happens — the `main` branch is the legacy superoptimizer (Mirage kernel-search, still referenced by some CI targets under `src/search/`, `src/transpiler/`, etc., but not part of the MPK runtime path).

## Build

```bash
pip install -e . -v          # editable install; triggers CMake + Cython
```

CMake drives the C++ library build, Cython links `python/mirage/core.*.so` to it. The build is **not fully incremental** through Python:
- Editing `src/**/*.cc` or `include/mirage/**/*.h` usually rebuilds via `pip install -e . -v` rerun, but the Cython `.so` can go stale. If Python behavior doesn't reflect a C++ change, `rm python/mirage/core.*.so && touch python/mirage/_cython/*.pyx` then re-run the install.
- Editing `.cuh` files under `include/mirage/persistent_kernel/tasks/` does **not** require a library rebuild — those are compiled by `nvcc` at *runtime* when the megakernel is JIT-assembled (see "Runtime compilation" below).

## Test / lint

```bash
bash scripts/format.sh       # clang-format-15 (downloads binary to .tools/). CI enforces this.
pytest tests/python/         # legacy Mirage search/transpiler tests
pytest tests/runtime_python/ # MPK runtime tests (test_mode/ is the subset that runs without a real GPU)
```

Single test:
```bash
pytest tests/runtime_python/test_mode/test_mxfp4_moe_pipeline_testmode.py -s
```

CI workflows in `.github/workflows/`: `code-format.yml` (clang-format-15), `build-test.yml` (library import on CPU), `ci-tests-qwen3.yml` (GPU smoke), `gpu-tests.yml`. Run `bash scripts/format.sh` before pushing or CI will flag the PR.

## Running demos

The two actively-maintained demos are `demo/qwen3/demo.py` and `demo/deepseek_v3/demo.py`. DeepSeek V3 has its own `demo/deepseek_v3/readme.md` with full env setup (NVSHMEM, MPI, venv path) and CLI flag reference — **read it before touching the demo**; the CLI surface is large and flag names are non-obvious.

Minimal single-GPU (TP=1) DeepSeek run:
```bash
python demo/deepseek_v3/demo.py --model-path /path/to/DeepSeek-V3 \
  --use-mirage --correctness --layers 3 --max-num-batched-tokens 1 \
  --max-seq-length 512 --max-new-tokens 1
```

Multi-GPU requires `mpirun -np $TP` with NVSHMEM + MPI env vars exported; the DeepSeek readme has the full incantation.

## Architecture: the persistent-kernel compile path

MPK compiles a model by building a **task graph** (Python) which is then lowered to a single CUDA megakernel (runtime codegen + `nvcc`). The flow has three layers:

**1. Builder (Python, `python/mirage/mpk/models/<model>/builder.py`)**
Each supported model has a builder that composes the graph by calling fused-op methods on a `PersistentKernel` (`python/mirage/mpk/persistent_kernel.py`). Each method (`rmsnorm_linear_layer`, `mla_decode_layer`, `mla_kv_gather_layer`, `moe_linear_layer`, `allreduce_layer`, etc.) registers a **task type name + params + tb_graph + grid/block dims + input/output tensor bindings** onto `self.kn_graph`. Crucially, the builder is also where **tensor lifetimes and buffer aliasing** are decided — allocations are either `mpk.new_tensor(...)` (lives in the megakernel's shared buffer pool) or `mpk.attach_input(torch_tensor=...)` (external PyTorch tensor bound by pointer). TP sharding rules live in the builder too, as regex→(shard_dim|None) lists applied to the HuggingFace `state_dict` keys.

**2. Graph registration (C++, `src/kernel/graph.cc::Graph::register_task`)**
Maps each task type name to a `TASK_*` enum + input/output tuple + `register_*_task` function. The tuple `(num_inputs, num_outputs, TASK_ENUM, variant_id)` tells the runtime scheduler how many of the Python `new_input(...)` tensors go into `task_desc->input_ptrs` vs `task_desc->output_ptrs`. Mismatching this against the Python-side `tb_graph.new_input` count is a common footgun — it shows up as "Invalid __global__ read" once the kernel runs.

**3. Task registration / codegen (C++, `src/kernel/task_register.cc`)**
Each `register_<task>_task` function emits a C++ snippet (via `CodeKeeper::e(...)`) that gets baked into the per-worker dispatch switch in the generated megakernel. The snippet calls into a corresponding device function in `include/mirage/persistent_kernel/tasks/blackwell/*.cuh` (SM100 = B200) or `tasks/hopper/*.cuh`. Runtime values like sequence length come from `runtime_config.paged_kv_indptr_buffer[...]` — a common pattern is to compute `S_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + paged_kv_last_page_len[bi]` inside the snippet so the kernel sees the true KV length this iteration (this is what makes chunked prefill work with a statically-compiled task graph).

**4. Runtime compilation (`src/kernel/runtime.cc::print_task_graph`)**
Walks the task graph, writes a `test.cu` file (the whole megakernel: scheduler + all worker dispatch code), and invokes `nvcc -gencode=arch=compute_100a,code=sm_100a ...`. The `MPK_MAX_NUM_BATCHED_TOKENS`, `MPK_MAX_NUM_BATCHED_REQUESTS`, `MPK_MAX_SEQ_LENGTH`, `MPK_PAGE_SIZE`, `MPK_MAX_NUM_PAGES` compile-time constants are set from the builder's constructor args and bake into every task. The resulting `.so` is `dlopen`ed and called via `init_persistent_kernel` / the kernel entry point.

**Task dispatch at runtime (`persistent_kernel.cuh::execute_worker`)**: the scheduler (`scheduler_kernel`) pushes `TaskDesc` records into per-worker queues. Each worker is 128 or 256 threads on one SM. Task metadata (`request_id`, `kv_idx`, etc.) is set by the scheduler based on `task_type` — see the switch in `runtime.cc::register_mugraph` (e.g., `TASK_MLA_KV_GATHER_SM100` gets `request_id = bid.y`).

## Key invariants / footguns

- **`grid_dim` must match the real parallelism the kernel expects.** If the kernel indexes by `request_id = blockIdx.y` and `kv_idx = blockIdx.x`, the Python-side `grid_dim` and the input `dim_maps` (3rd arg to `tb_graph.new_input`) have to line up. Wrong `dim_maps` silently produces wrong pointer offsets per task.
- **Output tensors are frequently passed via `new_input(store_in_dmem=True)`** rather than as a separate output — that's the "MPK convention". The task-register tuple must then say `(N+1, 0, ...)` not `(N, 1, ...)`, and the codegen reads `input_ptrs[N]` for the output pointer. `mla_decode` and `mla_prefill` both follow this pattern.
- **Dynamic shared-memory budget per worker is ~205 KB** (B200 SM100a total 227 KB minus static overhead). Kernels that tile beyond this hit `Invalid __shared__ write` at runtime (not compile time). Reduce pipeline stages before widening tile shapes.
- **`-rdc=true` is the default for every NVSHMEM build, including B200 SM100a.** Older notes claiming it hangs SM100a were diagnosed on a pre-CUDA-13.2 toolchain where `libnvshmem_device.a` inflated registers 166→255; gone on CUDA 13.2 + NVSHMEM 3.6.5 (verified 2026-04-22). `MPK_RDC_FALSE=1` on Blackwell falls back to the old self-contained-allreduce path (hand-rolled `nvshmemi_device_state_d` + `nvshmemid_hostlib_init_attr` callback) for regression isolation.
- **MLA TP decode kernels (`mla_mtp_decode_tp{2,4,8}_sm100.cuh`)** issue `tcgen05.alloc` from warp 0 and use a named barrier `bar.sync 1, 128` for the active-thread sync — do not replace with `__syncthreads()`, CuTe requires the same warp to alloc/dealloc and Independent Thread Scheduling otherwise causes warp drift. If you add a new MLA TP variant, match this pattern.
- **Weight layout for MLA absorbed-Q**: `q_b_proj` (post-absorption) is `[H * 576, q_lora]` where each head is `[nope(512) | pe(64)]`. The decode kernel consumes the fused 576-d form; the prefill kernel wants `q_b_nope` and `q_b_pe` as separate `[H*512, q_lora]` and `[H*64, q_lora]`. `demo.py`'s Phase-1/Phase-2 absorption code is what splits them at load time.
- **TP sharding rules live in two places** — `builder.py`'s `SHARD_RULES` regex list *and* `demo.py`'s model-conversion pass. When adding a new parallel linear, update both.

## Paths you'll usually touch together

- Add a task type: `runtime_header.h` (enum) + `task_register.cc` (`register_*`) + `graph.cc` (dispatch) + `runtime.cc` (`task_type_to_name` + request_id/kv_idx handler if non-default) + corresponding `.cuh` in `tasks/blackwell/` + Python wrapper in `persistent_kernel.py` + builder call site.
- Add a model: `python/mirage/mpk/models/<name>/builder.py`, register in `model_registry.py`, add a demo script under `demo/<name>/`.

## In-tree references worth reading

- `demo/deepseek_v3/readme.md` — full DeepSeek run reference, env vars, known limitations
- `README.md` — top-level MPK explanation + API examples (uses Qwen3 as reference)
- `INSTALL.md` — source-install + Docker setup
- `NCU_Usage_Manual.md` — how to profile MPK kernels with Nsight Compute
