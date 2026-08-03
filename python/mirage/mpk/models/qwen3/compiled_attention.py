"""The COMPILER-GENERATED attention path for Qwen3 decode.

Replaces the handwritten paged-attention monolith with three layers:
a handwritten PREP task (qk-norm + RoPE + KV-cache append + staging), the
generated softmax(Q@K^T + mask)@V core, and a handwritten FINALIZE that packs
the valid rows into the attention output. Both graph builders (the model
builder and the batch-perf script) call build_hybrid_attention so the wiring
exists exactly once.

Constraints of this path (attention_prep traps at runtime if violated):
- decode-oriented: multi-token (chunked prefill) steps append every new
  token's K/V but stage and attend only the LAST token -- the only row
  decode-only generation consumes;
- page_size >= max_seq_length, with the allocator handing request r page r
  (one page per request, admitted in order).
"""

import os

import torch

# The generated core's Q operand is padded to 8 rows per kv head: swapAB puts
# the token count into the MMA's N dimension, which tcgen05 requires to be a
# multiple of 8. Prep zeroes the pad rows so they are benign in the matmul.
Q_PAD_ROWS = 8

MASK_NEG = -30000.0


def parse_layer_spec(spec, layer_idx):
    """Which layers an env flag selects: "0"/""/"none" = no layers,
    "1"/"all" = every layer, "first:N" = layers below N, or a comma list of
    indices and inclusive ranges like "0,2,5-8"."""
    if spec in ("0", "", "none"):
        return False
    if spec in ("1", "all"):
        return True
    if spec.startswith("first:"):
        return layer_idx < int(spec.split(":")[1])
    chosen = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            chosen.update(range(int(lo), int(hi) + 1))
        else:
            chosen.add(int(part))
    return layer_idx in chosen


def compiled_attention_spec():
    return os.environ.get("MPK_COMPILED_ATTENTION", "0")


def build_hybrid_attention(mpk, *, layer_idx, attn_in, attn_out, k_cache,
                           v_cache, q_norm, k_norm, cos, sin, num_kv_heads,
                           head_dim):
    """Add prep -> generated core -> finalize for one decoder layer.

    All staging buffers are FOLD-DIM kvh-major (dim0 = kv_heads * max_reqs):
    the core is then ONE generated layer with a 1-D grid over the fold dim,
    and its operands share DTensor guids with prep's outputs, which is what
    gives the scheduler real dependency edges between the three stages.
    """
    reqs = mpk.max_num_batched_requests
    S = mpk.max_seq_length
    assert mpk.page_size >= S, "compiled attention needs page_size >= max_seq"
    fold = num_kv_heads * reqs
    dev = "cuda"
    bf16 = torch.bfloat16

    q_staged_t = torch.zeros(fold, Q_PAD_ROWS, head_dim, dtype=bf16, device=dev)
    mask_t = torch.full((fold, 1, S), MASK_NEG, dtype=bf16, device=dev)
    kt_staged_t = torch.zeros(fold, head_dim, S, dtype=bf16, device=dev)
    v_staged_t = torch.zeros(fold, S, head_dim, dtype=bf16, device=dev)
    pad_t = torch.zeros(fold, Q_PAD_ROWS, head_dim, dtype=bf16, device=dev)

    q_staged = mpk.attach_input(q_staged_t, name=f"layer_{layer_idx}_q_staged")
    mask_staged = mpk.attach_input(mask_t, name=f"layer_{layer_idx}_attn_mask")
    kt_staged = mpk.attach_input(
        kt_staged_t, name=f"layer_{layer_idx}_kt_staged")
    v_staged = mpk.attach_input(v_staged_t, name=f"layer_{layer_idx}_v_staged")
    attn_pad = mpk.attach_input(pad_t, name=f"layer_{layer_idx}_attn_pad")

    mpk.attention_prep_layer(
        input=attn_in, k_cache=k_cache, v_cache=v_cache,
        q_norm=q_norm, k_norm=k_norm,
        cos_pos_embed=cos, sin_pos_embed=sin,
        q_staged=q_staged, mask_staged=mask_staged,
        kt_staged=kt_staged, v_staged=v_staged,
        grid_dim=(reqs, num_kv_heads, 1), block_dim=(128, 1, 1),
    )
    mpk.generated_attention_layer(
        q_staged=q_staged, kt_staged=kt_staged, v_staged=v_staged,
        mask_staged=mask_staged, attn_pad=attn_pad,
        grid_dim=(fold, 1, 1), block_dim=(256, 1, 1),
    )
    mpk.attention_finalize_layer(
        attn_pad=attn_pad, output=attn_out,
        grid_dim=(reqs, 1, 1), block_dim=(128, 1, 1),
    )
