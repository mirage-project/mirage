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

Model implementations live in **two places**, not just `demo/<model_name>/`:

- **The registry path** — `python/mirage/mpk/models/<model_name>/builder.py`, a `GraphBuilder`
  subclass (base class `GraphBuilder` at `python/mirage/mpk/models/graph_builder.py:44`, config
  class `MirageModelConfig` at `:8`) decorated with `@register_model_builder(...)` (decorator
  defined at `python/mirage/mpk/model_registry.py:5`, resolved by `get_builder()` at `:25`).
  `MPK.build()` calls `get_builder(self.model_name)` then
  `build_from_model()`/`build_from_config()` (`python/mirage/mpk/mpk.py:458-467`). Two models
  actually do this today: `models/qwen3/builder.py:12`
  (`@register_model_builder("Qwen3", "Qwen/Qwen3-8B", ...)`) and
  `models/deepseek_v3/builder.py:63` (`@register_model_builder("deepseek-v3", "DeepSeek-V3",
  ...)`). `models/` also holds `eagle3/` and `dflash/` builder classes, but neither's `builder.py`
  contains a `@register_model_builder(...)` call (verified by grep) — check each one directly
  before assuming `MPK.build()` can reach it by name; not every class under `models/` is
  registered yet.
- **The inline demo path** — `demo/<model_name>/demo.py` builds a `PersistentKernel` directly and
  never calls `MPK`/`get_builder`. This is the path CI actually exercises (e.g.
  `demo/qwen3/demo.py:307`, driven by `tests/ci-tests/run_batch_perf.py`); the registry builder
  for the same model isn't covered by CI.

```
demo/<model_name>/
  demo.py                    # End-to-end inference demo
  models/                    # HuggingFace model files
    modeling_<model>.py      # HF model definition (for reference)
    configuration_<model>.py # HF config class
  <model>_shard_loader.py    # Weight name mapping + sharding (if multi-GPU)
```

**Reference implementations:**
- `demo/qwen3/` — Canonical dense transformer model (inline path)
- `demo/deepseek_v3/` — MoE model (DeepSeek V3 with MLA + MoE) (inline path)
- `python/mirage/mpk/models/qwen3/builder.py`, `models/deepseek_v3/builder.py` — the same two
  models wired through the registry path

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

### The Router Row Loop — Check It Covers Your Batch

The router kernels map one query row to one `(warp, sub-group)` and derive `thread_row` from
`threadIdx` alone, so a block natively covers only
`ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP` rows. Before M3-I5b there was no loop around that,
just `if (thread_row < num_rows)` — so every row past the bound was **silently dropped**, and a
second task instance recomputed the same rows rather than the next slice (the cap was per-graph,
not per-task). At Qwen3.5's 256-expert shape that bound is **16 rows**.

The failure mode is the reason this matters to a model port: `topk_w` and `mpk_routing_indices`
stay **zero** for the surplus rows, so those tokens lose their routed experts at all layers while
the shared expert and the residual keep flowing. There is no crash and no NaN — just a quiet
quality loss. M2-I9 hit it at `mbt=128`: rows 16+ degraded, and every AC-3 prompt diverged at
generated position 0. In a pre-fix probe dump, rows 0–15 hold plausible next-token predictions
while rows 16–29 collapse to one degenerate token
(`demo/qwen3_5/accept/opt/m3i5b/prep.md:14-34`, `:276-278`).

Both routers now loop (`topk_softmax_sm100.cuh:207-208`, `topk_sigmoid_sm100.cuh:176-177`), so the
cap is gone. Two things still follow for a new model:

- **Assert coverage, don't assume it.** Whenever you raise `max_num_batched_tokens` past a
  router's `ROWS_PER_CTA`, run a per-bs row-coverage check that every row got a nonzero routing
  weight. A row-count regression cannot be caught by a mean-error tolerance.
- **The row bound is a compile-time constant.** It follows from `WARP_SIZE * VPT / NUM_EXPERTS`,
  so a different expert count changes it. Read it from the kernel for *your* expert count rather
  than reusing another model's number.

## Verification

1. **Test individual layers** using test mode before wiring the full model. See `tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py` for the canonical pattern — it tests gate+up linear, silu_mul, and down+residual individually and as a pipeline.
2. **Compile test**: `mpk.compile(output_dir="./debug_output")` to inspect generated CUDA code and task graph JSON.
3. **Correctness test**: Compare MPK output against a HuggingFace reference model on the same prompt. Outputs should match within bfloat16 tolerance (~1e-2 max abs error per token).

## Runtime Geometry: Know What You Actually Ran

MPK's runtime parameters do not mean what the same-named parameters mean in vLLM or SGLang. Porting
a number across is how a benchmark ends up measuring a geometry nobody chose.

### `--max-seq-length` gates DECODE LENGTH, not KV capacity

In vLLM, `max_model_len` is a capacity ceiling and `--output-len` independently decides how many
tokens generate. In MPK, `max_seq_length` **is** the decode bound: MODE_OFFLINE retires a request at
`step + step_advance + 1 >= config.max_seq_length`
(`include/mirage/persistent_kernel/persistent_kernel.cuh:282`), so

```
decode_steps = max_seq_length - prompt_len - 1
```

and `--max-new-tokens` only slices the *reported* window afterwards
(`demo/qwen3_5/accept/mpk_engine_run.py:422`). Carrying vLLM's `256 (prompt) + 1024 (output) = 1280`
across therefore runs 1023 decode steps where 96 were intended — roughly an order of magnitude too
much work, and it silently changed the context length every per-stage number was measured at. Set it
forward instead:

```
--max-seq-length = prompt_len + desired_decode_steps + 1     # e.g. 256 + 96 + 1 = 353
```

Root cause and confirmation (`meta.waves[0].max_decode_steps == 96`):
`demo/qwen3_5/accept/opt/m3i10/remeasure/logs/ROOT_CAUSE_msl.txt:9-34`.

One caveat learned later: a small `max_seq_length` cannot express a wide batch. At `msl=353` no
prefill-free regime wider than five live requests exists at bs8 or bs16 — the first request retires
at iteration 112 while the last is still prefilling — so a capture fell through the steady-window
guard into a last-eight-iterations fallback and recorded a **single-request** step labelled bs8 and
bs16 (`demo/qwen3_5/accept/opt/m3i7/README.md:233-239`). Pin the geometry, then assert the wave
metadata matches it — regime, live count and `tokens_per_step`, not just the flag you passed.

### Verify a CLI flag is actually CONSUMED before trusting a geometry label

Passing a flag is not evidence the run used it. `mpk_engine_run.py --prompts-file` is not a prompt
source — it is read only under `--verify-chat-template`
(`demo/qwen3_5/accept/mpk_engine_run.py:687`; its own argparse help says "Only needed for
`--verify-chat-template`"). A campaign passed it alongside `--reference`, so every run consumed the
AC-3 reference prompts while the analysis divided by a hardcoded 1024. Both the prompt length and
the token count were wrong, in opposite directions, and a full set of "matched 256/1024" gaps
(3.84/3.63/3.26/3.36/4.17×) had to be retired — the true figures were 2.79/2.64/2.38/2.38/2.25×
(`demo/qwen3_5/accept/opt/m3i7/README.md:107-116`, `:269-276`).

Two habits: **grep the reader**, not the argparse block, before believing a geometry label; and have
the harness emit the geometry it actually ran (prompt ids, `max_decode_steps`, wave count) into the
results artifact, so a mismatch is visible without re-deriving it.

### The admission cap is a compile-time define with one authority

Uncapped, `prepare_next_batch` walks slots in order and gives each prefilling request
`min(remaining, mbt - used)`, so the *j*-th slot to finish prefill only ever gets `mbt - j` tokens
per iteration — a harmonic blow-up that turned 36 iterations of prefill work into 108
(`demo/qwen3_5/accept/opt/m3i9/README.md:26-29`). `MPK_MAX_TOKENS_PER_REQUEST` caps how many tokens
one request may contribute to one megakernel iteration; the shipped policy is
`cap = max(1, mbt // batch_size)` at every batch size ≥ 4 (a global 1 is a 19% regression at bs1).

- **One authority file**, and every caller derives from it:
  `demo/qwen3_5/accept/admission_policy.py` (`CAP_MODE`, `CAP_MIN_BATCH_SIZE`, `policy_cap()`).
  Don't re-implement the rule in a harness.
- **It is a compile-time `-D`** emitted at `python/mirage/mpk/persistent_kernel.py:323-325`, only
  when asked for, so an unset knob leaves the compile command byte-identical. Consequence: **two
  arms that share a `--kernel-dir` under `--reuse-kernel` run one binary** and differ only in the
  CPU-side admission replay, which will happily report a difference the binary does not have. Give
  every knob value its own kernel directory — see `/add-mpk-task` → "A Kernel Directory Must Carry
  Every Compile-Time `-D` Knob in Its Name".
