"""What computes a node the graph cannot model.

Four things in a decoder LLM have no muGraph op at any level: embedding is a
gather, argmax is a reduction to indices, and attention both scatters into the
KV cache -- a side effect, not a value -- and, on the split path, stages
tensors for the generated core. There is no gather, scatter or index op in the
IR, so these stay hand-written MPK tasks and the graph carries them as opaque
nodes: still real nodes, so the graph stays connected and a partition can see
the boundary, but lower() hands them here instead of to search.

None of this is model-specific: every decoder has these same four holes and
the MPK layers that fill them are generic. A model supplies only its token
count, its kv-head count, and the buffers argmax reduces through.
"""


from .node import OPAQUE_OPS


def standard_handlers():
    """The handler table, complete and parameterless.

    Everything a handler needs comes from the node: its inputs, its declared
    outputs, and group.attrs for the few numbers the graph cannot otherwise
    carry (a token count, a kv-head count). That is what lets lower() default
    this rather than making every caller pass it.

    Each handler receives the group, its already-bound input DTensors in the
    order the graph names them, and one output tensor per declared output. Both
    attention paths are registered; which one fires is decided by the graph,
    and only one set of nodes is ever present.
    """

    def embedding(pk, group, ins, outs):
        (out,) = outs
        tokens, weight = ins
        pk.embed_layer(input=tokens, weight=weight, output=out,
                       grid_dim=(group.attrs["tokens"], 1, 1), block_dim=(128, 1, 1))

    def _attn_args(pk, group, ins):
        """The seven inputs both attention paths read, and their shared grid:
        one block per (request, kv head)."""
        qkv, q_norm, k_norm, cos, sin, k_cache, v_cache = ins
        return dict(
            input=qkv, k_cache=k_cache, v_cache=v_cache, q_norm=q_norm,
            k_norm=k_norm, cos_pos_embed=cos, sin_pos_embed=sin,
            grid_dim=(pk.max_num_batched_requests,
                      group.attrs["num_kv_heads"], 1),
            block_dim=(128, 1, 1))

    def attention(pk, group, ins, outs):
        pk.paged_attention_layer(output=outs[0], **_attn_args(pk, group, ins))

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
        # The two-stage reduction's scratch buffers are DECLARED outputs, not
        # tensors the caller threads in: that is what keeps their dependency
        # on this task visible to MPK.
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
