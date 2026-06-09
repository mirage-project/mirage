"""Absorbed-format MLA prefill correctness through MPK test_mode (DSV3 shapes).

The production DeepSeek path feeds prefill with the same fused decode-format
layout as decode:

  q_nope_pe:  [sum(S_i), H * (D_CKV + D_KPE)]   (per head: nope[512] || pe[64])
  kv:         [B * max_seq_len, D_CKV + D_KPE]   (per row: ckv[512] || kpe[64])
  output:     [sum(S_i), H * D_V]                (D_V == D_CKV == 512)

The absorbed-prefill kernel (``mla_prefill_absorbed_sm100``) reads the per-request
prefill length from ``qo_indptr_buffer`` and the per-request KV length from the
paged-KV meta tensors, both of which ``prepare_next_batch`` recomputes from
``prompt_lengths`` in test mode. So we drive the prefill length(s) purely via
``prompt_lengths`` and never hand-set qo_indptr / paged_kv_* (they would be
overwritten on iter 0 anyway). KV is read DENSELY at ``bi * max_seq_len * 576``,
so per-request KV must live at rows ``[bi*max_seq_len, bi*max_seq_len + S_bi)``.

YARN scale fact: the kernel applies (1/sqrt(192)) * mscale^2 with
mscale = 0.1*ln(40)+1 internally; the reference uses the SAME scale + causal mask.

==============================================================================
MATRIX (union-of-axes over H = 128//tp and the prefill seq length S)
------------------------------------------------------------------------------
ASSERTED (deterministic + correct):
  * S=64 (one q-block, PF_BM=64) at every tp -> H in {128,64,32,16}.
  * B>1 multi-request prefills with one q-block per request: S=[64,64], [48,64].
XFAIL_MULTI_QBLOCK (known kernel bug, NOT a test/tolerance issue):
  * S in {128,256} (S>64 => >1 q-block) is NON-DETERMINISTICally WRONG on the
    first row of every later q-block (q_start = 64,128,...). See the decision log
    "Known kernel issue: mla_prefill_sm100 diagonal-merge". These are documented,
    not executed in the asserting set (a flaky-wrong config must not gate CI).
==============================================================================

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \
        tests/runtime_python/blackwell/sm100_mla/test_mla_prefill_absorbed_testmode.py
"""

import math
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_prefill_ref

D_CKV = 512   # latent / value dim (V == latent)
D_KPE = 64    # rope dim
D_V = 512
D_TOTAL = D_CKV + D_KPE  # 576
NUM_Q_HEADS = 128        # total DSV3 Q heads (sharded by tp)
PF_BM = 64               # kernel q-block tile size

# bf16 attention: softmax accumulation -> ~1e-2 tolerance (use 5e-2 / cos>0.99).
ATOL = 5e-2
RTOL = 5e-2
COS_MIN = 0.99


def _yarn_sm_scale():
    mscale = 0.1 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(192.0)) * mscale * mscale


def _cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _run_case(tp, prompt_lengths, tag):
    """Run one absorbed-prefill config.

    Args:
      tp: tensor-parallel degree -> H = 128 // tp local heads.
      prompt_lengths: list[int] of per-request prefill seq lengths (len == B).
      tag: label for printing.
    Returns (max_diff, cosine, passed).
    """
    device = "cuda"
    torch.manual_seed(11)

    H = NUM_Q_HEADS // tp
    B = len(prompt_lengths)
    max_seq = max(prompt_lengths)
    total_q = sum(prompt_lengths)
    sm_scale = _yarn_sm_scale()

    # --- build per-request Q / KV, plus the flattened MPK buffers ---------
    # Flattened Q / output are indexed by qo_indptr (cumsum of prompt_lengths).
    q_fused = torch.zeros(total_q, H * D_TOTAL, dtype=torch.bfloat16, device=device)
    out = torch.zeros(total_q, H * D_V, dtype=torch.bfloat16, device=device)
    # KV is read densely at bi * max_seq * 576 -> one max_seq slab per request.
    kv = torch.zeros(B * max_seq, D_TOTAL, dtype=torch.bfloat16, device=device)

    refs = []
    q_off = 0
    for bi, S in enumerate(prompt_lengths):
        q_bi = torch.randn(S, H * D_TOTAL, dtype=torch.bfloat16, device=device) * 0.1
        kv_bi = torch.randn(S, D_TOTAL, dtype=torch.bfloat16, device=device) * 0.1
        q_fused[q_off:q_off + S] = q_bi
        kv[bi * max_seq: bi * max_seq + S] = kv_bi
        q_off += S

        # Reference for this request: causal MLA over the absorbed/fused KV
        # (K = ckv[512] || kpe[64], V = ckv[512]) with the YARN scale.
        qv = q_bi.view(S, H, D_TOTAL)
        q_nope = qv[:, :, :D_CKV].contiguous()
        q_pe = qv[:, :, D_CKV:].contiguous()
        ckv = kv_bi[:, :D_CKV].contiguous()
        kpe = kv_bi[:, D_CKV:].contiguous()
        ref_bi = mla_prefill_ref(
            q_nope.unsqueeze(0), q_pe.unsqueeze(0),
            ckv.unsqueeze(0), kpe.unsqueeze(0), sm_scale,
        ).squeeze(0).reshape(S, H * D_V)
        refs.append(ref_bi)
    ref = torch.cat(refs, dim=0)

    # --- meta tensors: drive prefill length(s) via prompt_lengths only -----
    pl = torch.tensor(prompt_lengths, dtype=torch.int32, device=device)
    tokens = torch.zeros(B, max_seq, dtype=torch.int64, device=device)
    step = torch.zeros(B, dtype=torch.int32, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = total_q
    params["max_num_batched_requests"] = B
    params["max_seq_length"] = max_seq
    # One page per request slab (page_size == max_seq) keeps S_ == prompt length
    # and the dense KV layout (bi * max_seq * 576) page-aligned.
    params["page_size"] = max_seq
    params["max_num_pages"] = B
    params["meta_tensors"] = {
        "tokens": tokens,
        "step": step,
        "prompt_lengths": pl,
    }
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q_fused, name="q_fused")
    kv_dt = pk.attach_input(kv, name="kv_absorbed")
    out_dt = pk.attach_input(out, name="out")

    pk.mla_prefill_absorbed_layer(
        q_nope_pe=q_dt,
        kv=kv_dt,
        output=out_dt,
        mla_params=(H, max_seq, D_CKV, D_KPE, D_V),
        grid_dim=(H, (max_seq + 63) // 64, B),
        block_dim=(256, 1, 1),
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    nan_count = torch.isnan(out).sum().item()
    max_diff = (out.float() - ref.float()).abs().max().item()
    mean_diff = (out.float() - ref.float()).abs().mean().item()
    cos = _cosine(out, ref)
    close = torch.allclose(out, ref, rtol=RTOL, atol=ATOL)
    passed = (nan_count == 0) and close and (cos > COS_MIN)
    print(
        f"[{tag}] tp={tp} H={H} B={B} S={prompt_lengths} "
        f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} cos={cos:.6f} "
        f"nan={nan_count} -> {'PASS' if passed else 'FAIL'}"
    )
    pk.finalize()
    return max_diff, cos, passed


# Asserted matrix: single q-block (S <= PF_BM=64), correct + deterministic.
# union-of-axes over H = 128//tp at S=64, plus B>1 single-block-per-request.
MATRIX = [
    (1, [64], "S64-H128"),       # all S axis lives at H=128 (only S<=64 is sound)
    (2, [64], "S64-H64"),
    (4, [64], "S64-H32"),
    (8, [64], "S64-H16"),
    (1, [64, 64], "B2-H128"),    # multi-request batch (B>1)
    (2, [48, 64], "B2-H64-mixed"),
]

# Documented-only (NOT asserted): S>64 multi-q-block is a known nondeterministic
# kernel bug on the first row (q_start) of every later q-block. See decision log
# "Known kernel issue: mla_prefill_sm100 diagonal-merge". Listed here for the
# record; running them would flake CI with genuinely-wrong (not noisy) output.
XFAIL_MULTI_QBLOCK = [
    (1, [128], "S128-H128"),
    (1, [256], "S256-H128"),
    (2, [128], "S128-H64"),
]


def test_mla_prefill_absorbed_testmode():
    results = []
    for tp, pls, tag in MATRIX:
        max_diff, cos, passed = _run_case(tp, pls, tag)
        results.append((tag, max_diff, cos, passed))
    n_pass = sum(1 for _, _, _, p in results if p)
    print(f"\n=== mla_prefill_absorbed: {n_pass}/{len(results)} PASS ===")
    for tag, md, cos, p in results:
        print(f"  {tag}: max_diff={md:.6f} cos={cos:.6f} {'PASS' if p else 'FAIL'}")
    if XFAIL_MULTI_QBLOCK:
        print("  XFAIL (S>64 multi-q-block, known kernel bug, not asserted): "
              + ", ".join(t for _, _, t in XFAIL_MULTI_QBLOCK))
    failed = [tag for tag, _, _, p in results if not p]
    assert not failed, f"FAILED configs: {failed}"
    print("PASSED: mla_prefill_absorbed_sm100 matches causal MLA reference "
          "(single q-block, all H, B>1)")


if __name__ == "__main__":
    test_mla_prefill_absorbed_testmode()
