"""PyTorch reference for the DeepSeek-V3 MLA chunked-prefill attention.

The canonical torch reference for the chunked-prefill kernel, used by the
MPK test-mode test.

Per-head causal MLA chunked prefill (true unabsorbed, per-head K/V). Q covers
the chunk ``[q_start, q_start + q_len)`` of a longer sequence; KV covers
``[0, kv_len)``. The mask is causal w.r.t. the *absolute* sequence position:
key j attends to query i iff ``j <= q_start + i``.

Layout (post kv_b_proj decompression, per TP rank, B leading batch dim):
  Q_nope: [B, q_len,  H, 128]   per-head
  Q_rope: [B, q_len,  H,  64]   per-head
  K_nope: [B, kv_len, H, 128]   per-head
  K_rope: [B, kv_len, 1,  64]   shared across heads
  V:      [B, kv_len, H, 128]   per-head
  O:      [B, q_len,  H, 128]   per-head

Softmax scale: the chunked-prefill MPK task
(``register_mla_prefill_tp8_chunked_sm100_task`` in src/kernel/task_register.cc)
passes the BARE scale ``1/sqrt(192)`` to the kernel — it does NOT apply the YARN
``mscale**2`` factor that the sibling MLA tasks (decode / absorbed / mtp / the
non-chunked tp8 prefill) use. The reference therefore takes ``sm_scale`` as an
argument and the test passes the same bare scale the kernel receives, so the
comparison is valid. (See the test docstring / decision log for the YARN-scale
discrepancy this surfaces.)
"""

import math

import torch

D_QK_NOPE = 128
D_QK_ROPE = 64
D_QK = 192
D_V = 128


def bare_sm_scale():
    """The unadjusted 1/sqrt(d) scale (documentation/comparison only).

    Since the 2026-06-12 graph-audit fix the chunked-prefill register applies
    the YARN scale below, same as every sibling MLA task — use
    ``yarn_sm_scale`` for kernel comparisons.
    """
    return 1.0 / math.sqrt(D_QK)


def yarn_sm_scale():
    """The scale the chunked-prefill MPK task passes to the kernel: YARN
    mscale**2 for the DSv3 checkpoint (rope_scaling yarn, factor=40,
    mscale_all_dim=1.0), matching the decode/absorbed/mtp siblings and
    vLLM/SGLang serving behavior."""
    mscale = 0.1 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(D_QK)) * mscale * mscale


def mla_chunked_prefill_ref(qn, qp, k_nope, k_rope, v, q_start, sm_scale):
    """Per-head causal MLA chunked prefill reference (fp32 accumulation).

    Args:
      qn:     [B, q_len,  H, 128] bf16  per-head Q_nope
      qp:     [B, q_len,  H,  64] bf16  per-head Q_rope
      k_nope: [B, kv_len, H, 128] bf16  per-head K_nope
      k_rope: [B, kv_len, 1,  64] bf16  shared K_rope (broadcast over heads)
      v:      [B, kv_len, H, 128] bf16  per-head V
      q_start: int absolute position of the first query row in the sequence
      sm_scale: float softmax scale applied to the raw QK^T scores
    Returns:
      o: [B, q_len, H, 128] in qn.dtype
    """
    B, q_len, H, _ = qn.shape
    kv_len = k_nope.shape[1]
    q = torch.cat([qn, qp], dim=-1).float()                  # [B, q_len, H, 192]
    kr = k_rope.float().expand(B, kv_len, H, D_QK_ROPE)       # broadcast
    k = torch.cat([k_nope.float(), kr], dim=-1)              # [B, kv_len, H, 192]
    vf = v.float()
    scores = torch.einsum("bihd,bjhd->bhij", q, k) * sm_scale
    j = torch.arange(kv_len, device=q.device)
    i = torch.arange(q_len, device=q.device)
    mask = j[None, :] > (q_start + i[:, None])               # [q_len, kv_len]
    scores.masked_fill_(mask[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", probs, vf)
    return out.to(qn.dtype)
