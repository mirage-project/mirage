"""Handlers for the nodes the IR cannot express.

embedding is a gather, argmax a reduction to indices, and attention appends to
the KV cache -- none is a value produced from inputs, so each stays a
hand-written MPK task. Nothing here is model-specific.
"""


from .node import OPAQUE_OPS


def standard_handlers():
    """The handler table: group, bound input DTensors, one tensor per declared
    output. Parameterless -- a handler reads what it needs from group.attrs."""

    def embedding(pk, group, ins, outs):
        (out,) = outs
        tokens, weight = ins
        pk.embed_layer(input=tokens, weight=weight, output=out,
                       grid_dim=(group.attrs["tokens"], 1, 1), block_dim=(128, 1, 1))

    def _attn_args(pk, group, ins):
        """Seven inputs; one block per (request, kv head)."""
        qkv, q_norm, k_norm, cos, sin, k_cache, v_cache = ins
        return dict(
            input=qkv, k_cache=k_cache, v_cache=v_cache, q_norm=q_norm,
            k_norm=k_norm, cos_pos_embed=cos, sin_pos_embed=sin,
            grid_dim=(pk.max_num_batched_requests,
                      group.attrs["num_kv_heads"], 1),
            block_dim=(128, 1, 1))

    def attention(pk, group, ins, outs):
        """All of attention as one task. The split path cannot lower today --
        see docs/superoptimizer_ir.md."""
        (out,) = outs
        pk.paged_attention_layer(output=out, **_attn_args(pk, group, ins))

    def attn_prep(pk, group, ins, outs):
        q_st, mask_st, kt_st, v_st = outs
        pk.attention_prep_layer(
            q_staged=q_st, mask_staged=mask_st, kt_staged=kt_st,
            v_staged=v_st, **_attn_args(pk, group, ins))

    def attn_finalize(pk, group, ins, outs):
        (pad,) = ins
        (out,) = outs
        pk.attention_finalize_layer(
            attn_pad=pad, output=out,
            grid_dim=(pk.max_num_batched_requests, 1, 1), block_dim=(128, 1, 1))

    def argmax(pk, group, ins, outs):
        # Scratch buffers are DECLARED outputs so MPK sees the dependency.
        (logits,) = ins
        out, part_value, part_index = outs
        pk.argmax_partial_layer(
            input=logits, output=(part_value, part_index),
            grid_dim=(pk.num_workers, 1, 1), block_dim=(128, 1, 1))
        pk.argmax_reduce_layer(
            input=(part_value, part_index), output=out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    table = {"embedding": embedding, "attention": attention,
             "attn_prep": attn_prep, "attn_finalize": attn_finalize,
             "argmax": argmax}
    missing = sorted(set(OPAQUE_OPS) - set(table))
    if missing:
        raise AssertionError(
            f"node.OPAQUE_OPS declares {missing} with no handler here")
    return table
