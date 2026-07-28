---
name: dpskv3-logistic-review
description: Audit the DeepSeek V3 MPK demo + builder chain end-to-end and confirm logical equivalence with vLLM's reference implementation. Use after structural changes to `python/mirage/mpk/models/deepseek_v3/builder.py`, `demo/deepseek_v3/demo.py`, or any MLA / MoE / MTP task in `python/mirage/mpk/persistent_kernel.py`, `src/kernel/task_register.cc`, and `include/mirage/persistent_kernel/tasks/blackwell/mla_*.cuh` / `moe_*.cuh`. Produces a structured drift report so the change reviewer can confirm the math/topology is still equivalent.
---

# DeepSeek V3 Logistic Review

This skill produces a **drift report** that compares the current MPK
DeepSeek V3 demo chain against vLLM's reference implementation,
flagging any places where the two diverge in math, KV-cache topology,
weight layout, or scheduling. Use it whenever someone (Claude, Codex,
human) edits the demo / builder / MLA / MoE / MTP path and you need
to confirm the change preserves correctness.

The skill is **read-only** — it audits and reports, it does not edit.
Apply fixes separately, then re-run the skill.

---

## When to invoke

- After any commit that touches `python/mirage/mpk/models/deepseek_v3/builder.py`,
  `demo/deepseek_v3/demo.py`, or `python/mirage/mpk/persistent_kernel.py`'s
  MLA / MoE / MTP / KV-gather wrappers.
- After kernel changes under `include/mirage/persistent_kernel/tasks/blackwell/mla_*.cuh`,
  `moe_*.cuh`, or `linear_fp8_*.cuh`.
- After updating task registrations in `src/kernel/graph.cc` or
  `src/kernel/task_register.cc` for MLA / MoE / MTP.
- Before merging a feature branch back to `mpk`.
- When a regression-test PASS leaves you uncertain whether the
  *behavior* is right (PASS only proves "ran without crash and
  produced a latency line"; it does not prove correctness).

This skill is **not** a substitute for `scripts/regression_test.sh` —
they are complementary. Regression catches "does it run and produce
output?"; this skill catches "is what it produces actually equivalent
to the reference implementation?".

---

## The four-pillar review structure

Every audit pass should cover four pillars in order. Skip a pillar
only when the change being reviewed obviously cannot affect it.

1. **Weight pipeline** — `demo.py` weight load, FP8 absorption,
   per-rank shard, MTP-layer-included absorption, weight-cache load.
2. **Builder topology** — `builder.py` graph construction, tensor
   lifetime, KV cache layout, dual-dispatch gates.
3. **Task / kernel wiring** — Python wrapper → graph.cc dispatch →
   task_register.cc codegen → device kernel header.
4. **Runtime + scheduler** — task metadata setup in `runtime.cc`,
   per-iteration prepare_next_batch, AllReduce/sync-counter ordering.

Each pillar has a checklist below. Cite `file:line` for every claim
in the report.

---

## Pillar 1 — Weight pipeline

**Reference: vLLM's DeepSeek V2 model loader**
- `vllm/model_executor/models/deepseek_v2.py` — `DeepseekV2ForCausalLM`
  load + weight absorption.
- `vllm/v1/spec_decode/eagle/speculator.py` — MTP-specific weight
  loading (eh_proj, enorm, hnorm, shared lm_head).

**MPK chain to audit**:
- `demo/deepseek_v3/demo.py` lines ~600-760 (selective load + absorption):
  - `absorb_layers` includes `num_layers` (the MTP layer at idx 61) when
    `args.mtp > 0`. Double-check on every audit.
  - For each `attn` key in `absorb_layers`:
    - Original `q_b_proj.weight` becomes the absorbed fused 576-d Q
      (per `absorb_kv_into_q`); written back to `state_dict[q_key]`.
    - Original Q is *also* split into per-head `q_b_nope.weight`
      (`[H*128, q_lora]`, unabsorbed nope) and `q_b_pe.weight`
      (`[H*64, q_lora]`).
    - `kv_b_proj.weight` is split into `kv_b_k.weight` (`[H*128, kv_lora]`)
      and `kv_b_v.weight` (`[H*128, kv_lora]`).
    - The original `o_proj` is preserved as `o_proj_original` for the
      unabsorbed prefill output; the absorbed `o_proj` (post-W_UV
      fold-in) is what stays as `o_proj.weight`.
  - `pad_token_id = eos_token_id` fallback in qwen3 demo (orthogonal).

**Audit checklist**:
- [ ] Are all three forms (fused absorbed + split absorbed + split
      unabsorbed) produced for every layer including idx 61?
- [ ] Do `q_b_nope`, `q_b_pe`, `kv_b_k`, `kv_b_v` shapes match what
      the chunked-prefill kernel expects (`[H * d_per_head, q_lora]`)?
      See `mla_prefill_tp8_chunked_sm100.cuh` for the consumer.
- [ ] Does the absorbed `q_b_proj` shape match what the absorbed
      decode + absorbed prefill kernels expect (`[H*576, q_lora]`)?
- [ ] When `args.mtp > 0`, does layer 61 also get `q_b_nope` /
      `q_b_pe` / `kv_b_k` / `kv_b_v`? Per `demo.py:644`
      `absorb_layers.append(num_layers)`.
- [ ] TP sharding rules in `demo.py`'s `SHARD_RULES` regex list match
      the corresponding rules in `builder.py` for any new parallel
      linear?
- [ ] Pre-converted `model{rank}-mp{world_size}.safetensors` (the
      DeepSeek-V3-Demo-mp8 path) is consistent with the absorb path —
      meaning the same weight set is produced regardless of which
      load path is taken?

**Common drift**:
- Adding a new sharded linear without updating BOTH `demo.py` shard
  rules and `builder.py`'s `SHARD_RULES`.
- Forgetting to extend `absorb_layers` when adding a new layer that
  needs the absorbed split (e.g., a second MTP layer).
- Changing the absorbed Q layout in the kernel without updating the
  load-time absorption math.

---

## Pillar 2 — Builder topology

**Reference**: vLLM's `DeepseekV2DecoderLayer` (per-layer pattern)
and `EagleSpeculator` (MTP forward path), specifically:
- `vllm/model_executor/models/deepseek_v2.py:1100-1200` —
  decoder layer forward.
- `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py:374-420` —
  MTP first-pass + draft loop.

**MPK chain to audit** (`builder.py`):
- `__init__` and `_new_intermediate_tensors` (line 811): tensor
  allocation. Critical buffers:
  - `q_a_out`, `q_nope_pe` (fused 576-d), `q_nope`, `q_pe` (split),
    `c_latent_out`, `k_pe_out`, `ckv_sep`, `kpe_sep`,
    `prefill_k_nope`, `prefill_v`, `attn_out`, `attn_unabsorbed`,
    `mla_partial_o`, `mla_partial_lse`, `contiguous_kv`.
  - `_use_prefill = mbt > 8` — chunked-prefill gate.
  - `_direct_paged_decode_kv` — eligibility for skipping the dense
    KV gather copy (TP=1 single-request, TP=2, TP=4; TP=8 disabled).
- `_build_mla_attention_layer` (line 1190) — main model MLA.
- `_build_mla_attention_layer_with_prefix` (line 2114) — MTP MLA.
  **Note**: `use_mtp_prefill_attention` flag at line 2144 should be
  `True` (post-fix `f526c7ab`). If you find `False`, that's a
  regression.
- `_build_moe_mlp` (line 1660) and `_build_moe_mlp_with_prefix`
  (line 2463) — routed-experts group GEMM + topk + scatter +
  mul_sum_add. Watch w13 (dim=1) + w2 (dim=2) shard directions.
- `_build_dense_mlp` (line 1604) — gate/up + down with residual
  fusion.
- `_build_mtp_layer` (line 2654) — MTP draft loop. Step-0 input is
  `mtp_step0_input_tokens` (shifted prompt + main argmax tail);
  steps 1+ are autoregressive draft tokens.
- `build_from_dict` (line 3154) — top-level orchestration.

**Audit checklist**:
- [ ] `_use_prefill = mbt > 8`. Demo readme says `>=32`; trust the
      code.
- [ ] `use_mtp_prefill_attention = True` at `builder.py:2144`. If
      regressed to `False`, MTP's prefill attention is silently
      skipped → garbage hidden states.
- [ ] `_direct_paged_decode_kv` eligibility includes the TP variant
      under test. TP=8 stays on dense gather until the hang is fixed.
- [ ] Each `mla_kv_gather*` writes to the cache the subsequent
      attention task reads. Mismatched buffers = empty KV history.
- [ ] Main model uses `mla_prefill_tp8_chunked` with unabsorbed
      Q/K/V; MTP uses `mla_prefill_absorbed` with fused 576-d Q.
- [ ] Decode wrappers (`mla_mtp_decode_tp{1,2,4,8}_layer`) are
      registered alongside prefill so the dual-dispatch runtime gate
      can route by Q_LEN. Decode kernel returns early on `Q_LEN > 8`.
- [ ] Reduce stage (`mla_mtp_decode_tp*_reduce_layer` or
      `mla_reduce_layer`) is registered iff `num_splits > 1`. Single
      split uses `attn_out` directly.
- [ ] AllReduce + residual fusion (`linear_fp8_with_residual` or
      explicit `allreduce_layer(residual=...)`) — does TP > 1
      external AllReduce only when fuse_residual is False?
- [ ] LM head: vocab-parallel split at TP > 1 with MTP off; full
      vocab when MTP on (`build_from_dict:3206-3231`).
- [ ] MTP draft loop: step-0 hidden_input is `main_hidden_states`
      (target's final hidden state), step-1+ is `self.mtp_x` from
      previous draft.
- [ ] `mtp_build_embed_input_layer` runs once per MPK iteration
      *before* the draft loop, populating shifted prompt tokens.

**Common drift**:
- Forgetting the runtime Q_LEN gate when adding a new prefill or
  decode kernel — both will run at all Q_LEN.
- Allocating a buffer for one path (e.g., split q_nope/q_pe) but the
  consumer kernel reads a different one (e.g., q_nope_pe fused).
- Re-using a shared partial buffer across MTP draft steps without
  resetting it.
- Changing `_use_prefill` threshold in builder without updating
  the runtime Q_LEN gate in the kernel wrappers.

---

## Pillar 3 — Task / kernel wiring

**MPK paths**:
- Python wrapper: `python/mirage/mpk/persistent_kernel.py`. Each
  layer-method registers a `tb_graph` and calls
  `kn_graph.register_task(tb_graph, "<task_name>", params)`. The
  wrapper signature decides what gets bound to `input_ptrs[]`.
- Graph dispatch: `src/kernel/graph.cc::Graph::register_task` matches
  the task-name string to a `TASK_*` enum + `register_*_task`
  function.
- Codegen: `src/kernel/task_register.cc::register_<task>_task` emits
  the C++ snippet that runs in the per-worker dispatch switch. The
  snippet:
  - Reads `task_desc->task_metadata.{request_id, kv_idx, merge_task_offset}`
    set by the runtime scheduler (per `runtime.cc::register_mugraph`).
  - Reads `runtime_config.{qo_indptr_buffer, paged_kv_indptr_buffer,
    paged_kv_last_page_len_buffer, request_ids, step, prompt_length}`
    to compute current Q_LEN, S, KV pointer offsets.
  - Calls into the device function in
    `include/mirage/persistent_kernel/tasks/blackwell/<task>.cuh`.
- Device kernel: `*.cuh` headers under `tasks/blackwell/`. SM100a
  only — Hopper variants live under `tasks/hopper/` and are
  separate.

**Audit checklist**:
- [ ] Python wrapper's `tb_graph.new_input(...)` count matches the
      tuple in `graph.cc` (`(num_inputs, num_outputs, TASK_*, variant_id)`).
      Mismatch → "Invalid __global__ read" at runtime.
- [ ] If the task uses `store_in_dmem=True` for an output (the "MPK
      convention"), the tuple says `(N+1, 0, ...)` not `(N, 1, ...)`.
- [ ] `task_register.cc` codegen reads input pointers from the right
      indices and computes per-request offsets correctly:
      `qo_fp_ * (H * D)` for Q, `bi_ * MPK_MAX_SEQ_LENGTH * D` for
      KV (per-request stride), etc.
- [ ] Runtime Q_LEN gate matches the kernel's intended dispatch:
      prefill returns when `Q_LEN <= 8`; decode returns when
      `Q_LEN > 8`. Off-by-one is a silent miscompare.
- [ ] `runtime.cc::register_mugraph`'s metadata setup matches the
      kernel's expected mapping. e.g., `TASK_MLA_KV_GATHER_SM100`
      uses `request_id = bid.x`; `TASK_MLA_PREFILL_TP8_CHUNKED_SM100`
      uses `request_id = bid.x` (head), `kv_idx = bid.y` (q_block),
      `merge_task_offset = bid.z` (batch).
- [ ] Newly-added tasks have entries in `runtime_header.h` (enum) +
      `task_register.h` (declaration) + `task_register.cc` (impl) +
      `graph.cc` (dispatch) + `runtime.cc` (`task_type_to_name` +
      metadata handler if non-default) + `task_header.cuh` (include) +
      Python wrapper. Check ALL of these.
- [ ] `grid_dim` in the Python wrapper matches what the kernel and
      the metadata setup expect. Wrong `grid_dim` silently produces
      wrong pointer offsets per task.
- [ ] Dynamic SMEM per worker stays under ~205 KB on B200. New tile
      shape that exceeds this hits `Invalid __shared__ write` at
      runtime, not compile time.

**Common drift**:
- Adding a kernel that reads `q_nope_pe` (fused) when the wrapper
  passes `q_nope` (split), or vice versa.
- Forgetting to register the task in `runtime.cc::register_mugraph`'s
  metadata switch — `request_id` defaults to 0 silently.
- Editing a `.cuh` file's TMA descriptor signature without updating
  the corresponding `task_register.cc` snippet.
- Adding a new TP variant without mirroring the warp-0
  `tcgen05.alloc` + `bar.sync 1, 128` pattern from the existing
  TP{2,4,8} kernels.

---

## Pillar 4 — Runtime + scheduler

**MPK paths**:
- `include/mirage/persistent_kernel/persistent_kernel.cuh` — worker
  loop, scheduler, NVSHMEM team setup, prepare_next_batch.
- `src/kernel/runtime.cc::print_task_graph` — emits the megakernel
  test.cu, schedules tasks via dependency graph.
- `python/mirage/mpk/persistent_kernel.py::compile` — invokes nvcc,
  loads the built .so via ctypes / Python ext.

**Audit checklist**:
- [ ] `MPK_MAX_NUM_BATCHED_TOKENS`, `MPK_MAX_NUM_BATCHED_REQUESTS`,
      `MPK_MAX_SEQ_LENGTH`, `MPK_PAGE_SIZE`, `MPK_MAX_NUM_PAGES`
      compile-time constants reflect the demo CLI args.
- [ ] `-rdc=true` is the default for NVSHMEM builds. `MPK_RDC_FALSE=1`
      is an escape hatch only.
- [ ] CUDA toolchain: `/usr/local/cuda-13.2/bin/nvcc`. CUDA 12.8
      segfaults on the post-PR674 megakernel
      (`project_cuda128_nvcc_segfault.md`).
- [ ] NVSHMEM 3.6.5 + libnvshmem_host.so.3.6.5 LD_PRELOAD. AllReduce
      teams built from `nvshmem_team_split` per layer-shape.
- [ ] `prepare_next_batch` advances `step[]`, `qo_indptr_buffer[]`,
      `paged_kv_indptr_buffer[]`, `paged_kv_last_page_len_buffer[]`
      consistently across MPK iterations.
- [ ] AllReduce sync_counter advances in lockstep across ranks (no
      rank skips a layer's allreduce due to fuse_residual mismatch).

**Common drift**:
- Forgetting to add a new task type to `prepare_next_batch`'s
  dependency calculation, breaking the scheduler's task ordering.
- Changing `MAX_WORKER_PER_SCHEDULER` or `MAX_NUM_WORKERS` without
  re-evaluating per-worker SMEM budget.
- AllReduce team count exceeding NVSHMEM resource budget (~56 teams
  per peer is the practical cap).

---

## Audit procedure

1. **Establish baseline**. Run `scripts/regression_test.sh` against
   the unmodified code. Capture the perfetto traces and the summary
   PASS/FAIL + latencies.

2. **Apply the change** under review. Do not skip this — the audit
   must be against the actual diff, not against an imagined diff.

3. **Diff the perfetto trace**. Compare `task_type_name` counts and
   per-task durations between baseline and modified runs. New tasks
   appearing or expected tasks missing both warrant investigation.

4. **Walk the four pillars** in order. For each item in each
   checklist, either confirm `file:line` shows the expected pattern
   or open a finding with severity:
   - `definitely-broken` — math/topology is wrong, regression test
     PASS does not save you.
   - `suspicious` — divergence from vLLM that may or may not be
     intentional; needs author confirmation.
   - `probably-fine` — divergence with a documented justification.
   - `unclear` — needs more investigation.

5. **Cross-check against vLLM**. Where divergent, name the vLLM file
   and line and explain whether the difference is performance-driven
   (acceptable for now), correctness-driven (must align), or just a
   different API surface (orthogonal).

6. **Report**. Produce a structured drift report in this format:

   ```
   # DeepSeek V3 Logistic Review — <commit-or-branch>

   ## Change under review
   <one-line summary; commit hash if any>

   ## Summary
   <X PASS, Y SUSPICIOUS, Z BROKEN>

   ## Findings

   ### Pillar 1 — Weight pipeline
   - [PASS] <one-line description, file:line>
   - [SUSPICIOUS] <description, file:line, vLLM ref>
   ...

   ### Pillar 2 — Builder topology
   ...

   ### Pillar 3 — Task / kernel wiring
   ...

   ### Pillar 4 — Runtime + scheduler
   ...

   ## Cross-cuts
   <findings that span multiple pillars, e.g., a layout change that
   affects both weight-load and kernel>

   ## Recommended next actions
   <ordered list, smallest-fix first>
   ```

7. **Do NOT edit** while in audit mode. The skill is read-only by
   design; mixing audit and patching makes it impossible to know
   what fixed what. Hand the report to the user (or to a separate
   patch session) and let them apply fixes.

---

## Known drift baseline (as of 2026-05-07)

These are the deltas from vLLM that exist in the current codebase
and have been verified. Note them in every audit report under
"Probably-fine, deferred":

- **Separate MTP KV cache** (`mtp_ckv_kpe_cache_tensor`) vs vLLM's
  shared KV cache. Functionally equivalent if the cache is correctly
  populated; sharing would simplify the code but is a refactor.
- **Absorbed prefill kernel for MTP, chunked-unabsorbed for main**.
  vLLM uses one (chunked) for both; MPK uses two because the absorbed
  form was wired earlier and never refactored to chunked. Performance
  delta only.
- **`mla_unified_layer` framework exists but unused**. The unified
  layer would fold prefill + decode into one task with internal
  Q_LEN-based dispatch (vLLM-style). Not yet wired into the DeepSeek
  builder; current dual-dispatch with runtime Q_LEN gates is the live
  path.
- **TP=8 stays on dense KV gather**. Direct-paged decode hangs on
  TP=8; eligibility check forces dense for `world_size == 8`.
- **TP=4 V-split is pseudo-parallel**. The kernel re-runs QK+softmax
  for every V-split work unit, only splitting PV. `MPK_MLA_TP4_V_SPLITS`
  default switches between 2 and 8 based on max_seq_length.
- **Vocab-parallel LM head off when MTP on**. Compatibility with
  shared lm_head between target and MTP.

If your audit report says "diverges from vLLM" for any of these, mark
them `probably-fine, deferred`, not `suspicious`.

---

## Skill output: where to put findings

Write the audit report to a fresh file (don't overwrite an old one
unless the user asks). Suggested path:

```
outputs/dpskv3_review_<YYYYMMDD>_<HHMMSS>.md
```

`outputs/` is gitignored, so the report is local-only. Quote
file:line references and the relevant snippet so the user can
verify without re-deriving the audit. Length budget: 1000-2000 words
for a focused-change audit; longer reports rarely get read.
