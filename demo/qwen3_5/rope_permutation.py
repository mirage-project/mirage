"""Qwen3.5 partial-RoPE permutation — re-export shim.

The implementation MOVED into the installed package at
`python/mirage/mpk/models/qwen3_5/rope.py` (M2-I8): the registry builder cannot
import from `demo/`, and the permutation must have exactly ONE definition shared
by probe P4, the M2-I6 kernel tests and the M2-I8 weight loader.

Existing callers (`accept/probes/attn/p4_rope_perm.py`, the sm100_attention
tests) keep importing `rope_permutation` unchanged.
"""

from mirage.mpk.models.qwen3_5.rope import (  # noqa: F401
    HEAD_DIM,
    ROPE_THETA,
    ROTARY_DIM,
    build_cos_sin_table,
    permute_head_dim,
    permute_q_gate_rows,
    rope_permutation_inv,
    rope_permutation_src,
)

__all__ = [
    "HEAD_DIM",
    "ROTARY_DIM",
    "ROPE_THETA",
    "rope_permutation_src",
    "rope_permutation_inv",
    "permute_head_dim",
    "permute_q_gate_rows",
    "build_cos_sin_table",
]
