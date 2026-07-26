"""Qwen3.5 partial-RoPE via load-time head_dim column permutation.

`docs/qwen35/v1-architecture.md` §2.0 (weight-loading transforms) and §4.4.

The problem
-----------
Qwen3.5's full-attention layers rotate only the FIRST `rotary_dim = 64` of each
256-wide head, NeoX-style, pairing column `j` with `j + 32`
(`docs/qwen35/vllm-graph.md` §2.2.3). MPK's SM100 attention kernel hard-codes the
full-width NeoX pairing `(i, i + HEAD_DIM/2) = (i, i + 128)`
(`attention_sm100.cuh` -> `rms_norm_sm100`'s fused rotary block, and
`rotary_embedding_sm100.cuh`). Padding cos/sin with identity values alone is
therefore NOT sufficient: the kernel would rotate `(j, j+128)` pairs that Qwen
never pairs.

The transform
-------------
Permute the head_dim columns at LOAD TIME so Qwen's rotated pairs land on MPK's
pairs, and pad the cos/sin table with `cos=1, sin=0` at every column MPK rotates
but Qwen does not:

    MPK column i        <- Qwen column SRC[i]
    ----------------------------------------------------
    i in [0, 32)        <- i                (rotated, "first half"  of Qwen's pair)
    i in [128, 160)     <- (i - 128) + 32   (rotated, "second half" of Qwen's pair)
    everything else     <- the remaining Qwen columns [64, 256), in order

so MPK's pair `(i, i+128)` for `i < 32` is exactly Qwen's pair `(i, i+32)`, and
every other MPK pair is an identity rotation because `cos=1, sin=0` there.

Why it is exact
---------------
* `q·k` is invariant under a permutation applied identically to q and k.
* RMSNorm over the full head_dim is permutation-EQUIVARIANT provided the norm
  weight is permuted with the same map (the variance is a permutation-invariant
  sum over all 256 columns -- Qwen normalizes over the full 256, not over
  `rotary_dim`, `vllm-graph.md` §2.2.3).
* `cos=1, sin=0` makes MPK's rotation the identity on the un-rotated columns:
  `v1*1 - v2*0 == v1` and `v1*1 + v2*0 == v1`.
* V and the attention output are NOT permuted -- only q/k columns are, and they
  only ever meet inside `q·k`.

Probe P4 (`accept/probes/attn/p4_rope_perm.py`) proves this numerically against
the HF oracle before the loader relies on it.

What the loader must permute (all with `PERM_SRC`, i.e. `w_new[i] = w_old[SRC[i]]`
along the head_dim axis):
  * `q_proj` rows -- the q half of each per-head `[q(256)|gate(256)]` slice ONLY
    (the gate half is never rotated or normed: `vllm-graph.md` §2.2.2/§2.2.4);
  * `k_proj` rows (all 256 per kv head);
  * `q_norm` / `k_norm` weights (after the Gemma `1+w` fold);
  * the cos/sin tables -- built by `build_cos_sin_table()` below.
`v_proj` and `o_proj` are untouched.
"""

HEAD_DIM = 256
ROTARY_DIM = 64  # int(256 * partial_rotary_factor=0.25), vllm-graph.md §2.2.3
ROPE_THETA = 1e7  # rope_parameters.rope_theta, vllm-graph.md §2.2.3


def rope_permutation_src(head_dim: int = HEAD_DIM, rotary_dim: int = ROTARY_DIM):
    """`src[i]` = the source (Qwen) column that becomes MPK column `i`.

    Returns a plain list so this module stays import-light (no torch/numpy
    dependency) -- the loader and the probe both convert it as needed.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even (NeoX pairing)")
    if rotary_dim % 2 != 0 or rotary_dim > head_dim:
        raise ValueError("rotary_dim must be even and <= head_dim")

    half = head_dim // 2  # MPK pairs (i, i + half)
    r_half = rotary_dim // 2  # Qwen pairs (j, j + r_half) for j < r_half

    src = [None] * head_dim
    for j in range(r_half):
        src[j] = j  # MPK "first half" slot
        src[half + j] = r_half + j  # MPK "second half" slot

    # The un-rotated Qwen columns [rotary_dim, head_dim) fill the remaining MPK
    # slots in increasing order. Any bijection works (cos=1/sin=0 makes those
    # slots identity), but a deterministic, order-preserving one keeps the
    # permuted weights readable and the inverse trivial to reason about.
    free_slots = [i for i in range(head_dim) if src[i] is None]
    passthrough = list(range(rotary_dim, head_dim))
    assert len(free_slots) == len(passthrough), (len(free_slots), len(passthrough))
    for slot, col in zip(free_slots, passthrough):
        src[slot] = col

    assert sorted(src) == list(range(head_dim)), "not a permutation"
    return src


def rope_permutation_inv(head_dim: int = HEAD_DIM, rotary_dim: int = ROTARY_DIM):
    """`inv[j]` = the MPK column that Qwen column `j` moves to."""
    src = rope_permutation_src(head_dim, rotary_dim)
    inv = [None] * head_dim
    for i, j in enumerate(src):
        inv[j] = i
    return inv


def permute_head_dim(tensor, dim: int = -1, head_dim: int = HEAD_DIM,
                     rotary_dim: int = ROTARY_DIM):
    """Apply the load-time permutation along `dim` (must have size `head_dim`).

    `torch` is imported lazily so importing this module costs nothing in a
    non-torch context.
    """
    import torch

    src = torch.as_tensor(rope_permutation_src(head_dim, rotary_dim),
                          dtype=torch.long, device=tensor.device)
    if tensor.shape[dim] != head_dim:
        raise ValueError(f"axis {dim} has size {tensor.shape[dim]}, expected {head_dim}")
    return torch.index_select(tensor, dim, src)


def permute_q_gate_rows(q_proj_weight, num_q_heads: int, head_dim: int = HEAD_DIM,
                        rotary_dim: int = ROTARY_DIM):
    """Permute ONLY the q half of a packed `[q|gate]`-per-head projection.

    `q_proj_weight` is `[num_q_heads * 2 * head_dim, in_features]` laid out as
    `[h0_q | h0_gate | h1_q | h1_gate | ...]` (`vllm-graph.md` §2.2.2). The gate
    half is returned untouched.
    """
    import torch

    out_features = q_proj_weight.shape[0]
    expected = num_q_heads * 2 * head_dim
    if out_features != expected:
        raise ValueError(f"q_proj rows {out_features} != num_q_heads*2*head_dim {expected}")
    w = q_proj_weight.view(num_q_heads, 2, head_dim, *q_proj_weight.shape[1:]).clone()
    src = torch.as_tensor(rope_permutation_src(head_dim, rotary_dim),
                          dtype=torch.long, device=q_proj_weight.device)
    w[:, 0] = torch.index_select(w[:, 0], 1, src)
    return w.view(out_features, *q_proj_weight.shape[1:])


def build_cos_sin_table(positions, head_dim: int = HEAD_DIM,
                        rotary_dim: int = ROTARY_DIM, theta: float = ROPE_THETA,
                        dtype=None, device=None):
    """Build MPK's `head_dim`-wide cos/sin tables in the PERMUTED layout.

    Mirrors HF's construction (`inv_freq = 1/theta**(arange(0, rotary_dim, 2)/rotary_dim)`,
    `emb = cat(freqs, freqs)`, `cos = emb.cos()`) and then scatters it into the
    permuted 256-wide table, filling the un-rotated slots with `cos=1, sin=0`.

    Returns `(cos, sin)` of shape `[len(positions), head_dim]`.
    """
    import torch

    pos = torch.as_tensor(positions, dtype=torch.float32, device=device).reshape(-1)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                             device=pos.device) / rotary_dim))
    freqs = pos[:, None] * inv_freq[None, :]           # [T, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)            # [T, rotary_dim]
    hf_cos, hf_sin = emb.cos(), emb.sin()

    t = pos.shape[0]
    cos = torch.ones(t, head_dim, dtype=torch.float32, device=pos.device)
    sin = torch.zeros(t, head_dim, dtype=torch.float32, device=pos.device)
    inv = rope_permutation_inv(head_dim, rotary_dim)
    idx = torch.as_tensor([inv[j] for j in range(rotary_dim)], dtype=torch.long,
                          device=pos.device)
    cos[:, idx] = hf_cos
    sin[:, idx] = hf_sin
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin
