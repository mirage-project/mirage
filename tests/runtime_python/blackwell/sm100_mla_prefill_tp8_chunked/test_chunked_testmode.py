"""DeepSeek-V3 MLA chunked-prefill attention through MPK test_mode (DSV3 shapes).

Exercises ``mla_prefill_tp8_chunked_layer`` -> ``mla_prefill_tp8_chunked_sm100``
end-to-end through the real MPK compile/codegen/runtime pipeline, comparing the
kernel output against the lifted PyTorch reference
(``pytorch_reference.mla_chunked_prefill_ref``).

This is the TP-shaped, true-unabsorbed, per-head K/V prefill kernel. Q covers a
chunk ``[q_start, q_start+q_len)`` of a longer sequence; KV covers ``[0, kv_len)``.
The causal mask is w.r.t. the absolute position: key j attends to query i iff
``j <= q_start + i``. Setting ``q_start = kv_len - q_len`` places the chunk at
the tail of the sequence (the DSV3 chunked-prefill usage), so when ``kv_len >
q_len`` the chunk loop runs over the full KV history (the "chunked" path).

============================== KEY DECISIONS ===============================
* KV BUFFERS ARE PADDED TO A BN=128-ALIGNED ROW COUNT (zero-filled tail).
  The kernel loads each KV block (BN=128 rows) via TMA; the LAST block is
  partial when kv_len % 128 != 0. The K/V tensors must therefore extend to
  ceil(kv_len/128)*128 rows so the partial block reads VALID (zero) rows.
  If the buffer is sized exactly kv_len, TMA OOB-fills the tail with NaN
  (CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA), and although the
  causal/length mask sets those scores to -inf -> prob 0, the PV product
  computes 0 * NaN(V) = NaN and the whole output is NaN. Production never
  hits this: the real DSV3 KV cache is sized to max_seq_length (page-aligned,
  a multiple of 128), so the partial-block tail is valid cached data, not OOB.
  The reference attends only to the first kv_len rows (the kernel masks
  kvp >= kv_len), so the zero padding changes nothing numerically. See the
  decision log "Known kernel issue: chunked partial-last-KV-block 0*NaN".

* B == 1. The K_nope/K_rope/V TMA descriptors (tma.cuh case
  TASK_MLA_PREFILL_TP8_CHUNKED_SM100) carry NO batch coordinate — the kernel's
  ``tma3d(KN, ..., 0, kvb, head*2+half)`` reads from a single per-tensor
  descriptor whose base_ptr is the whole KV tensor, with no per-batch offset.
  The kernel applies the batch offset only to Q/O arithmetically (``bat *
  q_len * row_stride``). So a single descriptor can only address ONE batch's
  KV. B=1 is the kernel's correct domain (the CUDA-ext test is also B=1; the
  production grid.z is per-request but each request's KV TMA desc is built from
  that request's narrowed slice). H is the real swept axis (= 128 // tp).

* TENSOR LAYOUT = 3D, B dropped. K_nope / V are attached as [kv_len, H, 128]
  (3D) so the TMA branch resolves H_local = dim[1], d_last = dim[2]. K_rope as
  [kv_len, 64] (2D) -> the param_id==3 branch (total_rows=kv_len, d_last=64).
  Q_nope [q_len, H, 128], Q_pe [q_len, H, 64], O [q_len, H, 128] (3D). A 4D
  [B, kv_len, H, 128] tensor would mis-resolve the TMA dims, so we keep 3D.

* SCALE = BARE 1/sqrt(192), NOT YARN. The chunked-prefill task's codegen
  (register_mla_prefill_tp8_chunked_sm100_task) hardcodes
  ``sm_scale = 1/sqrtf(192)`` and does NOT multiply by the YARN ``mscale**2``
  (= (0.1*ln(40)+1)**2 ~= 1.874) that every SIBLING MLA task (decode, absorbed,
  mtp, the non-chunked tp8 prefill) applies. The reference uses the SAME bare
  scale the kernel receives, so the comparison is valid. This non-YARN scale is
  flagged as a finding (see decision log) — NOT a tolerance loosening.

* In test mode the chunked codegen takes the MPK_TEST_MODE branch, which passes
  q_len / kv_len / q_start as literal params (from ``mla_params``) — it does NOT
  read the qo_indptr / paged_kv meta tensors. So seq lengths are driven directly
  via ``mla_params``; default meta tensors only keep prepare_next_batch happy.

* qfused_mode 0 (legacy split Qn/Qp) is the primary sweep; one fused-mode (=1)
  spot check covers the row-swap [all_nope || all_pe] fused-Q layout.

MATRIX (union-of-axes over H = 128//tp, plus q/kv lengths incl. chunked kv>q):
  {tp=1} x {(q,kv) = (64,64),(64,128),(64,256),(128,128),(128,256)}
  union {(q,kv)=(64,256) chunked} x {tp=2,4,8}  (H=64,32,16)
  union {tp=8,(64,64)} corner
  plus fused-mode spot check {tp=1,(q,kv)=(64,256)} qfused=1.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \
        tests/runtime_python/blackwell/sm100_mla_prefill_tp8_chunked/test_chunked_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import (  # noqa: E402
    D_QK_NOPE,
    D_QK_ROPE,
    D_V,
    bare_sm_scale,
    mla_chunked_prefill_ref,
)

NUM_Q_HEADS = 128  # total DSV3 Q heads (sharded by tp -> H = 128 // tp)
BM = 64            # kernel q-block tile size
BN = 128           # kernel KV-block tile size (KV buffers padded to a BN boundary)

# bf16 attention (softmax accumulation): atol/rtol ~ 1e-2, cos > 0.99.
# Use a slightly looser 3e-2 (the CUDA-ext test's own atol) AND require cos>0.99
# AND no NaN. NOT a loosening to mask a mismatch — observed margins are reported.
ATOL = 3e-2
RTOL = 3e-2
COS_MIN = 0.99


def _cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _make_inputs(q_len, kv_len, H, device, seed):
    """Returns (qn, qp, kn, kr, v, kn_real, kr_real, v_real).

    K/V buffers are padded to a BN=128-aligned row count (zero tail) so the
    kernel's partial last KV block reads valid zeros, not TMA-OOB NaN. The
    *_real views hold the first kv_len rows the reference attends to.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    dt = torch.bfloat16
    kv_pad = (kv_len + BN - 1) // BN * BN
    qn = torch.randn(q_len, H, D_QK_NOPE, generator=g, dtype=dt, device=device) * 0.2
    qp = torch.randn(q_len, H, D_QK_ROPE, generator=g, dtype=dt, device=device) * 0.2
    kn_real = torch.randn(kv_len, H, D_QK_NOPE, generator=g, dtype=dt, device=device) * 0.2
    kr_real = torch.randn(kv_len, D_QK_ROPE, generator=g, dtype=dt, device=device) * 0.2
    v_real = torch.randn(kv_len, H, D_V, generator=g, dtype=dt, device=device) * 0.2
    kn = torch.zeros(kv_pad, H, D_QK_NOPE, dtype=dt, device=device); kn[:kv_len] = kn_real
    kr = torch.zeros(kv_pad, D_QK_ROPE, dtype=dt, device=device);    kr[:kv_len] = kr_real
    v = torch.zeros(kv_pad, H, D_V, dtype=dt, device=device);        v[:kv_len] = v_real
    return qn, qp, kn, kr, v, kn_real, kr_real, v_real


def _run_case(tp, q_len, kv_len, tag, qfused_mode=0):
    """Run one chunked-prefill config. Returns (max_diff, cos, nan, passed)."""
    device = "cuda"
    H = NUM_Q_HEADS // tp
    q_start = kv_len - q_len  # chunk at the tail (DSV3 chunked-prefill usage)
    assert q_start >= 0, "q_len must be <= kv_len"
    sm_scale = bare_sm_scale()

    qn, qp, kn, kr, v, kn_real, kr_real, v_real = _make_inputs(
        q_len, kv_len, H, device, seed=0)
    out = torch.zeros(q_len, H, D_V, dtype=torch.bfloat16, device=device)

    # Reference (B=1) attends only to the real (unpadded) kv_len rows.
    ref = mla_chunked_prefill_ref(
        qn.unsqueeze(0), qp.unsqueeze(0),
        kn_real.unsqueeze(0), kr_real.unsqueeze(0).unsqueeze(2),  # [1, kv, 1, 64]
        v_real.unsqueeze(0), q_start, sm_scale,
    ).squeeze(0)  # [q_len, H, 128]

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    # The chunked task reads q_len/kv_len/q_start from mla_params in test mode,
    # not from meta tensors; defaults keep prepare_next_batch happy.
    params["max_num_batched_tokens"] = q_len
    params["max_num_batched_requests"] = 1
    params["max_seq_length"] = max(q_len, (kv_len + BN - 1) // BN * BN)
    pk = PersistentKernel(**params)

    if qfused_mode == 1:
        # Row-swap fused-Q buffer: [q_len, H*192] = [all-heads-nope || all-heads-pe].
        q_fused = torch.zeros(q_len, H * (D_QK_NOPE + D_QK_ROPE),
                              dtype=torch.bfloat16, device=device)
        # nope region: per head h -> cols [h*128, h*128+128)
        q_fused[:, : H * D_QK_NOPE] = qn.reshape(q_len, H * D_QK_NOPE)
        # pe region: per head h -> cols [H*128 + h*64, +64)
        q_fused[:, H * D_QK_NOPE:] = qp.reshape(q_len, H * D_QK_ROPE)
        qn_dt = pk.attach_input(q_fused, name="q_fused")
        qp_dt = qn_dt  # fused: Qp slices from the same tensor (codegen adds offset)
    else:
        qn_dt = pk.attach_input(qn.contiguous(), name="q_nope")
        qp_dt = pk.attach_input(qp.contiguous(), name="q_pe")
    kn_dt = pk.attach_input(kn.contiguous(), name="k_nope")
    kr_dt = pk.attach_input(kr.contiguous(), name="k_rope")  # [kv_len, 64] (2D)
    v_dt = pk.attach_input(v.contiguous(), name="v")
    out_dt = pk.attach_input(out, name="out")

    pk.mla_prefill_tp8_chunked_layer(
        q_nope=qn_dt,
        q_pe=qp_dt,
        k_nope=kn_dt,
        k_rope=kr_dt,
        v=v_dt,
        output=out_dt,
        mla_params=(H, q_len, kv_len, q_start),
        grid_dim=(H, (q_len + BM - 1) // BM, 1),
        block_dim=(128, 1, 1),
        qfused_mode=qfused_mode,
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
    fused_s = " fused" if qfused_mode == 1 else ""
    print(
        f"[{tag}]{fused_s} tp={tp} H={H} q={q_len} kv={kv_len} qs={q_start} "
        f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} cos={cos:.6f} "
        f"nan={nan_count} -> {'PASS' if passed else 'FAIL'}"
    )
    pk.finalize()
    return max_diff, cos, nan_count, passed


# Union-of-axes: every tp (H) and every (q,kv) length at least once, incl.
# chunked (kv > q), a multi-q-block (q=128 > BM=64) case, and partial-last-KV-
# block cases (kv % 128 != 0 -> exercises the BN-padding path).
MATRIX = [
    # tp=1 (H=128): full (q,kv) sweep.
    (1, 64, 64, "H128-q64-kv64"),       # single partial block (kv<BN), kv==q
    (1, 64, 128, "H128-q64-kv128"),     # chunked: kv > q, full block
    (1, 64, 192, "H128-q64-kv192"),     # chunked + partial last block (192%128)
    (1, 64, 256, "H128-q64-kv256"),     # chunked: longer history, 2 full blocks
    (1, 128, 128, "H128-q128-kv128"),   # multi-q-block (q=128 -> 2 q-blocks)
    (1, 128, 256, "H128-q128-kv256"),   # multi-q-block + chunked history
    # union: chunked case across the other H (= 64/32/16).
    (2, 64, 256, "H64-q64-kv256"),
    (4, 64, 256, "H32-q64-kv256"),
    (8, 64, 256, "H16-q64-kv256"),
    # corner: smallest H with smallest case.
    (8, 64, 64, "H16-q64-kv64"),
]

# Fused-Q (qfused_mode=1, row-swap [all_nope || all_pe]) spot check.
FUSED = [
    (1, 64, 256, "H128-q64-kv256-FUSED"),
]


def test_chunked_testmode():
    results = []
    for tp, q_len, kv_len, tag in MATRIX:
        md, cos, nan, passed = _run_case(tp, q_len, kv_len, tag, qfused_mode=0)
        results.append((tag, md, cos, nan, passed))
    for tp, q_len, kv_len, tag in FUSED:
        md, cos, nan, passed = _run_case(tp, q_len, kv_len, tag, qfused_mode=1)
        results.append((tag, md, cos, nan, passed))

    n_pass = sum(1 for *_, p in results if p)
    print(f"\n=== mla_prefill_tp8_chunked: {n_pass}/{len(results)} PASS ===")
    for tag, md, cos, nan, p in results:
        print(f"  {tag}: max_diff={md:.6f} cos={cos:.6f} nan={nan} "
              f"{'PASS' if p else 'FAIL'}")
    failed = [tag for tag, _, _, _, p in results if not p]
    assert not failed, f"FAILED configs: {failed}"
    print("PASSED: mla_prefill_tp8_chunked_sm100 matches causal chunked-MLA "
          "reference (all H, chunked kv>q, multi-q-block, fused-Q)")


if __name__ == "__main__":
    test_chunked_testmode()
