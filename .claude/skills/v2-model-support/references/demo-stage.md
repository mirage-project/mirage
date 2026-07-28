# Phase (a) — DEMO stage: builder anatomy + the footgun list

Deliverable: `demo/<model>/demo.py` + `python/mirage/mpk/models/<model>/builder.py`
(+ `tasks.py` for model-specific task wrappers) that build the v2 graph and pass the
0-GPU gates. Reference instances: `demo/deepseek_v3/demo.py` (~1575 lines) +
`python/mirage/mpk/models/deepseek_v3/builder.py` (~3969) + its `tasks.py` (~1571)
(multi-GPU MoE/EP, builder-side v2 gating); and `demo/qwen3/demo.py` (dense,
single-GPU-capable, inline graph + demo-level v2 plan wiring — see SKILL.md
§"Worked example #2").

## 1. Where things live (current reality — supersedes the older add-mpk-model note)

- `python/mirage/mpk/models/<model>/builder.py` — `GraphBuilder` subclass; the graph.
- `python/mirage/mpk/models/<model>/tasks.py` — model-specific `*_layer` wrappers that
  call `pk.kn_graph.register_task(tb_graph, "<task_name>", params)` directly (fused
  megas, model-only ops). Generic wrappers live in `persistent_kernel.py`.
- `demo/<model>/demo.py` — CLI, weight load/convert/shard/cache, MPKMetadata,
  `mpk.build()` → `mpk.compile()` → request loop. Register in `model_registry.py`.

## 2. Builder anatomy (crib `DeepSeekV3Builder`)

- `__init__` derives ALL parallelism once: `routed_tp_size = world_size // ep_size`,
  `ep_rank = rank // routed_tp_size`, `local_expert_start/end`, per-rank head counts,
  per-rank intermediate sizes. Everything downstream reads these fields.
- **Capability predicates** (`_use_attn_megakernel`-style `@property`): a fused path is
  selected ONLY when the build matches the exact geometry the kernel hard-asserts
  (mbt==1, num_workers==136, world_size==8, …). Any other config falls through to the
  compat/chain path — this is what keeps TP1/TP2/TP4 smoke builds from tripping
  megakernel asserts. Restate the predicate as asserts inside the build method too.
- Layer loop (`build_layers`): attach norm weights → attention block → residual →
  post-attn norm → MLP (dense/MoE branch) → AR(+residual) at TP>1 → tail
  (final norm → lm_head → argmax). Keep the residual stream in `self.x`.
- **CHAIN-FIRST**: build every layer from existing generic v2 tasks first. The chain
  is the ground truth every fused kernel is later diffed against.

## 3. Weight mapping / sharding — TWO places, always both

1. `demo.py`'s conversion pass: `_TP_SHARD_RULES` = ordered `[(regex, dim|None)]` over
   POST-conversion keys (DSv3: demo.py ~:1140). dim=0 col-parallel (shard output),
   dim=1 row-parallel (shard input, AR after), None replicate; 3D expert tensors get
   EP slicing + an inner-dim shard (w13 dim=1, w2 dim=2). First match wins — order
   specific before generic.
2. The builder's attach sites consume the ALREADY-SHARDED state_dict but re-encode the
   same decisions (which buffer is partial vs full, where the AR goes, zero-residual
   bindings). A rule changed in one place and not the other = silent garbage at TP>1.

Conversion/absorption runs in demo.py at load: emit EVERY weight form any path
consumes (decode absorbed + prefill unabsorbed + BMM repacks + requantized scales).

## 4. Weight-cache key contract (demo.py ~:585-628)

`cache_payload` = {format string, model realpath, layers, world_size, rank, ep_size,
vocab_parallel_lm_head (+ derived lm_head fields), hidden/vocab/num_layers} →
sha256 → `<cache_dir>/<key>/rank{r}.safetensors`.
- The key CANNOT see conversion-CODE changes (absorb/quantize/fuse math, added/removed
  cached keys, `_TP_SHARD_RULES`). **Bump the `format` version string on ANY of those**
  (DSv3 is at `deepseek_v3_mpk_fp8_runtime_cache_v2`). Silent stale weights is the
  worst failure; missing-key errors are loud and fine.
- `vocab_parallel_lm_head` is in the key ⇒ flipping `--disable-vocab-parallel-lm-head`
  forces a full re-convert. Budget for it on box sessions.
- Cold-convert host-RAM controls:
  - `MPK_CONVERT_SEMAPHORE=K` — flock semaphore, ≤K ranks convert concurrently
    (TP8 cold convert is ~330 GB/rank un-sharded ⇒ 8 ranks OOM a 1.76 TB box).
    K = floor(MemoryMax_GB / 350) clamped [2,4]. Safe: loader has no collectives.
  - `MPK_BUILD_CACHE_ONLY=1` + `MPK_FORCE_BUILD_RANK=r` — serial pre-builder: one
    isolated no-MPI subprocess per rank builds `rank{r}.safetensors` then `sys.exit(0)`
    (idempotent: exits early if present). Fallback when even K-wave doesn't fit.

## 5. Tensor lifetimes / aliasing (the silent-corruption class)

- `mpk.new_tensor(dims, dtype, name, io_category)` — megakernel-pool buffer.
  `io_category="cuda_tensor"` normal; `"nvshmem_tensor"` for ANYTHING a collective
  reads/writes (AR in/out, cross-rank argmax scratch). **new_tensor does NOT zero** —
  anything read-before-write (barrier counters!) needs a `tensor_init_layer`.
- `mpk.attach_input(torch_tensor=..., name=...)` — binds the RAW GPU POINTER. The
  Python tensor MUST be kept alive (`self._something = t`) or you dangle. Attach each
  name ONCE (`_attach_cache` dedup); for a second attach of the same underlying data
  use a DISTINCT name (`_safe_attach`) — a duplicate name = duplicate
  `model_tensors.at(...)` declaration = nvcc "already declared".
- A root cuda_tensor's INPUT and OUTPUT TensorDescs resolve to the SAME physical
  address — in-place read+write through an input slot persists (the attn-mega KV cache
  works this way). Don't "fix" it, and don't theorize non-persisting in-place writes.
- `MAX_INPUTS_PER_TASK = 14` is hard. Concat small same-dtype weights into one buffer
  (DSv3 ln_weights `[input_ln|q_a_ln|kv_a_ln]` (9216,), cos_sin `[cos|sin]` per row)
  — and static_assert the offsets in the kernel.

## 6. Task-registration footguns (generic, v1+v2)

- **grid_dim/dim_maps must match kernel indexing.** If the kernel does
  `request_id = blockIdx.y` / `kv_idx = blockIdx.x`, the Python `grid_dim` and each
  input's dim_map (3rd arg of `tb_graph.new_input(t, (dx,dy,dz), fdim, store)`) must
  line up; wrong maps silently offset per-CTA pointers. `(-1,-1,-1)` = whole tensor.
- **Outputs via `new_input(..., store_in_dmem=True)`** is the MPK convention for most
  DSv3-class tasks: the register tuple in `graph.cc` is then `(N+1, 0, TASK_X, v)` not
  `(N, 1, ...)`, and codegen reads `input_ptrs[N]` as the output. Mismatch = "Invalid
  __global__ read" at runtime.
- Params order in `register_task(tb_graph, name, params)` is positional and must match
  the `task_register.cc` reads exactly.

## 7. v2 wiring specifics (what `--use-v2` actually requires)

Plumbing (small): `--use-v2` argparse; `use_v2_runtime=args.use_v2` into the
`mi.PersistentKernel(...)` ctor; profiler sizing via
`get_profiler_buffer_entries(use_v2=...)` if profiling. `compile()` ITSELF then runs
the v2 pipeline (persistent_kernel.py ~:5619-5649): the §1.1 deadlock guard
(`v2_unsafe_task_types` from runtime.cc → loud RuntimeError), then
`build_v2_worker_task_queues` (round-robin, continuous cursor, task 1 prepended to
worker 0 — bit-identical C++ twin `build_v2_plan`) + `add_v2_region_smem_plan`
(14 pages × 16 KB, capacity 224256 B). No demo-side scheduling code.

Builder-side v2 work:
- Most generic wrappers self-switch (`"..._v2" if self.use_v2_runtime`); model
  `tasks.py` wrappers need an explicit `if pk.use_v2_runtime:` branch that calls the
  v2 layer method (crib `attn_block_megakernel_layer` → `pk.dsv3_attn_mega_layer`,
  `num_tasks=pk.num_workers`).
- **`skip_after_step0=True` on tensor_init of MONOTONIC barrier scratch is a
  CORRECTNESS REQUIREMENT** (Form-2 megas: `bar_need = num_tasks*(iter+1)`, never
  self-reset). Re-zeroing on step≥1 ⇒ iter-0 fine / iter-1 hang. Step 0 still zeroes
  (cudaMalloc garbage); the task/event stays in the graph every step.
- v2-only allocations: e.g. attn-mega scratch is `+16 B` under v2 (3×u64=24 B barrier
  vs v1's 8 B); FFN-mega wants the packed `scales` (MEGA_SC_ order) + `xfer` f32 +
  `bar` i64[2] + `artifacts` u8 tensors — sizes mirror the `_spec.h`, cross-checked
  against `tests/runtime_python/blackwell_v2/dsv3_ffn_harness.py`.
- **Skip emitting v1-only tasks whose outputs are dead under the v2 fused path**
  (DSv3: the fused_rmsnorm_qkv_a quantize is skipped when
  `use_v2_runtime and _use_attn_megakernel`, builder ~:3618) — a v1-only task in a v2
  graph is a silent no-op that also wedges the ring (the guard catches it at build).
- Re-route fragile tails: DSv3's v2 lm_head is `dsv3_lmhead_gemv_layer` (plain GEMV;
  the TMA+tcgen05 v3 linear faulted as lm_head); dense layers route to
  `_build_dense_mlp_fused_v2` because the unfused dense chain's FP8 tasks are
  v1-only. Expect one or two such re-routes per model.
- RowParallel outputs at TP>1: kernel writes a residual-FREE partial into an
  `nvshmem_tensor` (bind a persistent ZERO buffer as its residual input), then
  `allreduce_layer`/`_allreduce_residual` sums partials + adds the real residual once.

## 8. Gates before the first GPU run (in order)

1. **Graph-build**, smallest slice (`--layers <first-real-layer>` equivalent), CPU-only:
   the builder runs, all asserts/shape checks pass, `generate_task_graph` succeeds.
2. **§1.1 guard green**: `compile()` under `use_v2_runtime` does not raise
   `v2_unsafe_task_types`. If it raises: add the v2 role variant OR skip the emission
   under v2 (never ship a task the guard flags — it wedges the box, D-state zombies).
3. **Reachability diff** for sliced builds: dump the task list (`--dump-task-graph`)
   for the slice vs the full build; confirm no head/tail seed task (embedding, KV
   init, rope buffers, first residual) is dropped by the slicing.
4. **test-mode** (see the `test-mode` skill): single-pass, per-layer, vs a
   `pytorch_reference.py` in the same folder; bf16 tol ~1e-2. The blackwell_v2 suite
   (`tests/runtime_python/blackwell_v2/run_suite.py --what correctness`) covers the
   generic v2 ops; add model-specific ops as harness branches.

## 9. Demo CLI surface worth copying (from `demo/deepseek_v3/demo.py`)

`--model-path --use-mirage --use-v2 --layers 0-60 --ep-size --max-num-batched-tokens
--max-num-batched-requests --max-seq-length --max-num-pages --page-size --prompt-length
--ignore-eos --max-new-tokens --save-tokens --profiling --trace-name --dump-task-graph
--weight-cache-dir --disable-vocab-parallel-lm-head`. Notes: `max_seq_length ≈
prompt + new_tokens` (over-allocating costs O(max_seq²)); mbt>8 flips dual-dispatch
prefill on (decode bring-up wants mbt=1); `--save-tokens` is the token-identity gate's
raw material — wire it from day 1.
