# Streaming weight loading + EP-aware DeepSeek V3 loading — Design

**Date:** 2026-06-10
**Status:** Approved (design); pending implementation plan
**Branch:** `feat/new-api`

## 1. Problem

`MPKModule.load_weights` (`python/mirage/mpk/layers/_base.py:92`) does not stream:

```python
weights_list = list(weights)        # _base.py:128 — drains the whole generator
...
groups = {}                         # _base.py:143 — references to every tensor, live at once
for name, tensor in weights_list: ...
for mod, prefix, group in groups.values(): ...   # processed only after all are held
```

`safetensors_weights_iterator` (`python/mirage/mpk/weight_loader.py:25`) yields lazy
mmap-backed views (verified: `list()`-ing ~1 GB of tensors added +3 MB RSS; touching
them faulted in the full ~940 MB). But because `load_weights` holds a reference to
**every** tensor (the `list`, then `groups`, then the recursive `stripped` lists) until
it returns, the faulted-in pages are never released between weights. Host RSS therefore
climbs toward the full *unsharded* checkpoint per process; under tensor-parallelism every
rank does this concurrently → host OOM on large models (observed: full DeepSeek V3 at
TP=8 via the demo exhausts host RAM and crashes the node).

Two secondary issues found in the same path:

- **Silent missing weights.** The consumed-keys set returned by `load_weights` is
  discarded by the caller (`persistent_kernel.py:689`), and params are `torch.empty`
  (not `zeros`), so a missing/misnamed weight loads as garbage with no error.
- **DeepSeek V3 catalog loading is the worst case.** `demo/deepseek_v3/demo_new.py::
  _load_hf_weights_with_absorption` builds the *entire* model state_dict on CPU
  (fp8→bf16 dequant, MLA absorption, stacking all 256 experts per layer), then
  `{k: v.to("cuda")}` moves all of it to GPU at once and `load_state_dict`s. It also
  loads **all** experts — no expert-parallel (EP) support. The catalog `modeling.py`
  loads via `_load_from_state_dict` hooks (single-GPU only, `ep_size=1`, no NVShmem).

## 2. Reference: how vLLM does it (`deps/vllm`)

- Lazy `yield`-based `safetensors_weights_iterator` (`weight_utils.py:893`) consumed
  one `(name, tensor)` at a time by `for name, loaded_weight in weights:`
  (`deepseek_v2.py:1378`). No `list()`, no group-by-module. Peak host RAM ≈ one tensor.
- TP sharding still narrows *after* materializing (`loaded_weight.narrow(...)`); vLLM does
  **not** use `get_slice` for per-rank disk reads. So `get_slice` is **not** adopted here.
- The disk-I/O win for big (MoE) models is the **EP weight filter**: each EP rank skips
  non-local expert tensors entirely (`FusedMoE.weight_loader` maps global→local expert id
  and returns `False` for non-local; `ep_weight_filter.should_skip_weight` checked before
  `get_tensor`). Experts are ~85-90% of a V3 checkpoint.
- Model built on the target device, then `param.copy_(loaded_weight)` is the H2D
  (`base_loader.py:54`). MPK already does this (`persistent_kernel.py:685`).

## 3. Goals / non-goals

**Goals**
- `MPKModule.load_weights` streams: process-and-release one tensor at a time. Peak host
  footprint bounded by ~one source tensor, not the checkpoint. Identical per rank.
- Loud failure on missing/extra keys.
- Catalog DeepSeek V3 (`models/deepseek_v3/modeling.py`) migrated off `_load_from_state_dict`
  onto the streaming `load_weights(iterator)` + `process_weights()` path, with **EP-aware
  expert loading**: each rank loads only its local experts (`local_expert_start ..
  local_expert_end`); non-local expert tensors are never read or resident.
- All of DSV3's current preprocessing (fp8 dequant, MLA absorption, expert stacking) moves
  **into the model** (`load_weights` / `process_weights`); the demo-driver preprocessing
  is deleted.

**Non-goals (explicitly out of scope)**
- The distributed MoE **runtime** (all-to-all / NVShmem token dispatch+combine). End-to-end
  multi-GPU EP correctness is *not* delivered here — only EP-*compatible loading*. Catalog
  DSV3 correctness stays gated to `ep_size=1` (where local range = all experts, so behavior
  is identical to today).
- Routing DSV3 through `PersistentKernel.build_from_config` (its KV-cache allocation is
  Qwen3/standard-attention shaped, not MLA). DSV3 keeps its `demo_new.py` construction +
  compile path; only its *loading* changes. `@register_model` + `build_from_config`
  unification is a noted follow-up.
- `get_slice`-based per-rank disk reads for dense/attention weights (vLLM doesn't do this
  either).

## 4. Design

### 4.1 Mapping-driven streaming `MPKModule.load_weights` (Approach A)

Replace the `list()` + group-then-recurse body with a single streaming loop modeled on
vLLM's `load_weights`:

1. Build, **once** at the top-level call:
   - a flat param table `{full_param_path → (param, weight_loader_or_None)}` from
     `named_parameters()` / `named_modules()`;
   - the model's **mapping lists** (collected from the module tree): name-remaps, fused/
     stacked mappings `(target_param, source_substr, shard_id)`, and expert mappings
     `(target_param "experts.w13"/"w2", source_substr, expert_id, slot)`.
2. `for name, tensor in weights:`  ← lazy generator, **one at a time**
   - match `name` against mappings (most-specific first): resolve `(target_param,
     loader_kwargs)`; fall back to default 1:1 routing by deepest `named_modules` prefix.
   - call `param.weight_loader(param, tensor, **kwargs)` if the leaf defines one, else
     `param.data.copy_(tensor)`.
   - record consumed; **drop the tensor reference** (loop rebind) so its pages are
     reclaimable before the next.
3. After the loop: assert every yielded key was consumed and every required param was
   filled; raise listing the offenders otherwise.

Leaf weight_loader callbacks are **unchanged**: `ColumnParallelLinear._weight_loader`
(`narrow(0,…)`), `RowParallel*` (`narrow(1,…)`), `Embed`/`Linear` full copy. Models override
only the **mappings**, never the method body.

Composite overrides shrink: `Qwen3Attention.load_weights` (q_norm/k_norm remap) becomes two
name-remap entries; q/k/v stay separate `ColumnParallelLinear` leaves (fused at compile via
`shuffle_tensors`, so no load-time stacking).

### 4.2 EP-aware experts (DSV3 `MoEW13` / `MoEW2`)

- Wire `ParallelConfig` EP fields (`ep_size`, `ep_rank`; `parallel.py:42-58`) into the
  catalog MoE module: `num_local_experts = n_routed_experts // ep_size`,
  `local_expert_start = ep_rank * num_local_experts`.
- Allocate the 3D expert params at **local** size: `MoEW13.weight` →
  `(num_local_experts, 2*moe_inter, hidden)`, `MoEW2.weight` →
  `(num_local_experts, hidden, moe_inter)` (today they are full `num_experts`;
  `layers/moe/w13.py:125`).
- Per-expert mapping yields `(expert_id, slot∈{gate,up,down})`. The expert weight_loader
  maps global→local id; **returns early (not consumed) for non-local experts** — the mmap
  view is never touched ⇒ no disk read, no GPU residency. Mirrors `FusedMoE.weight_loader`.
- `ep_size=1` ⇒ local range = all experts ⇒ byte-identical to today.

### 4.3 DeepSeek V3 catalog migration (`models/deepseek_v3/modeling.py`)

- **Delete** the three `_load_from_state_dict` hooks (MLA `:197`, MLP `:511`, MoE `:691`)
  and the driver fn `demo_new.py::_load_hf_weights_with_absorption` (`:168`).
- Implement `load_weights(iterator)` (via §4.1 mappings) that streams raw HF keys:
  - name-remaps: `q_a_proj`, `kv_a_proj_with_mqa`, `gate.weight`/`gate.e_score_correction_bias`,
    `shared_experts.{gate,up,down}_proj`;
  - EP-filtered expert stacking into local `w13`/`w2` slots;
  - **inline fp8 dequant**: buffer each fp8 weight with its `weight_scale_inv` partner in a
    small bounded dict; dequant (`convert.dequantize_fp8`) when both halves are present;
    assert the buffer is empty at end. Non-local experts are skipped before buffering.
- Implement `process_weights()` for the **cross-parameter MLA absorption** post-load
  (`kv_b_proj`→`q_b_proj`, `W_UV`→`o_proj`; via `convert.absorb_kv_into_q`).
- `demo_new.py`: replace `_load_hf_weights_with_absorption(...)` + `load_state_dict(...)`
  (`:510-513`) with `model.load_weights(safetensors_weights_iterator(files))` +
  `model.process_weights()`.

## 5. Error handling

- **Missing/extra keys:** top-level assertion in `load_weights` (§4.1 step 3). Listed,
  fatal. Resolves the silent-`torch.empty`-garbage footgun.
- **fp8 pairing:** bounded buffer keyed by base name; asserted empty after the stream so a
  weight without its scale (or vice-versa) fails loudly.
- **EP divisibility:** assert `n_routed_experts % ep_size == 0` (already asserted in the
  legacy builder; mirror it in the catalog MoE module).

## 6. Testing

- **Streaming property (unit):** feed `load_weights` a generator of N tensors tracked by
  `weakref`; assert ≤ small constant alive simultaneously (proves process-and-release).
- **Missing-key (unit):** omit a required key; assert `load_weights` raises listing it.
- **Qwen3 regression (CLAUDE.md rule 2):** Qwen3-8B end-to-end via `demo/qwen3/demo.py
  --use-mirage` → correct output + ~4.3 ms/token on 1×B200.
- **DSV3 EP filter (unit):** tiny synthetic MoE config + `ep_size=2`; assert each rank's
  `w13`/`w2` holds only its local experts and non-local keys are skipped (not consumed,
  view untouched).
- **DSV3 correctness:** `demo/deepseek_v3/demo_new.py` single-GPU (`ep_size=1`, reduced
  `num_hidden_layers_override` to avoid host OOM) → output unchanged vs. current driver.

## 7. Affected files

| File | Change |
|------|--------|
| `python/mirage/mpk/layers/_base.py` | Rewrite `load_weights` to streaming mapping-driven loop; add missing-key assert; mapping-collection hooks |
| `python/mirage/mpk/weight_loader.py` | Unchanged — EP filtering happens in the expert `weight_loader` (the non-local mmap view is never touched), so the iterator needs no `skip_fn` |
| `python/mirage/mpk/models/qwen3/modeling.py` | `Qwen3Attention`: q_norm/k_norm override → name-remap mapping |
| `python/mirage/mpk/layers/moe/w13.py`, `w2.py` | Local-sized expert params + per-expert EP-aware `weight_loader` |
| `python/mirage/mpk/models/deepseek_v3/modeling.py` | Delete `_load_from_state_dict` hooks; add `load_weights` (stream + EP + fp8) + `process_weights` (MLA absorption); wire EP from `ParallelConfig` |
| `demo/deepseek_v3/demo_new.py` | Delete `_load_hf_weights_with_absorption`; use `load_weights` + `process_weights` |
| `tests/...` | New unit tests (streaming, missing-key, EP filter) |

## 8. Follow-ups (not in this work)

- Distributed MoE runtime for catalog DSV3 (all-to-all/NVShmem) → end-to-end multi-GPU EP.
- `@register_model("DeepseekV3ForCausalLM")` + MLA-aware `build_from_config` unification.
- Optional async/pinned-memory H2D overlap during load.
