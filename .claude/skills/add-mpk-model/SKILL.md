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
