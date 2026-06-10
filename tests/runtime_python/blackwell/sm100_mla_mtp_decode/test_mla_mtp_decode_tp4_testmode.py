"""
Test: fused mla_mtp_decode_tp4_sm100 -> mla_mtp_decode_tp4_reduce_sm100 via test_mode.

Builds both layers in one PersistentKernel graph (decode writes partial-O /
partial-LSE, reduce consumes them and produces the final reduced output) and
compares the final output against the TP-agnostic full causal-MLA reference
``mla_full_ref`` with NUM_HEADS=128, taking the first H_per_rank=32 heads.

This simulates rank 0 of a TP=4 run. The kernel processes 32 heads (quarter
of the 128 total).

OFFLINE TEST-MODE CONSTRAINT (same as TP1):
  In MODE_OFFLINE test mode step=0 is forced, so the runtime derives
  kv_len_ == q_len_rt_ == prompt_length. Effective kv is locked to q_len.
  We still register with static_kv_len=256 (num_splits=2) so the kernel uses
  the partial-write path; the inactive split is reduced away via pre-filled
  LSE=-inf.

  For TP=4, the decode kernel also packs a V-split dimension into block_x.
  For static_kv_len=256: v_splits = _mla_tp4_v_splits(256) = 8 (< 3072 threshold).
  This is handled internally by mla_mtp_decode_tp4_layer.

Buffer shapes (TP=4, H_per_rank=32):
  q:              [batch * q_len * 32,   D_K=576]  bf16
  kv:             [batch * eff_kv,       D_K=576]  bf16  (eff_kv = q_len)
  output_partial: [batch * num_groups * num_splits, D_V*128]  bf16
  output_lse:     [batch * num_groups * num_splits, 128]      float32
  final_out:      [batch * q_len,        32 * D_V]  bf16

  qpg        = min(4, q_len)
  num_groups = ceil(q_len / qpg)
  num_splits = (static_kv_len + 128 - 1) // 128  = 2

Sweep (bs x q_len):
    bs    in {1, 2, 4, 8, 16}
    q_len in {1, 2, 4}
"""

import math
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import (
    PersistentKernel,
    _mla_tp4_v_splits,
    _mla_tp4_head_groups,
    _mla_tp4_rd_dv,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_full_ref, NUM_HEADS, D_K, D_V

H_PER_RANK = 32      # TP=4
STATIC_KV_LEN = 256
MATRIX = [
    (bs, q_len)
    for bs in (1, 2, 4, 8, 16)
    for q_len in (1, 2, 4)
]


def _run_case(batch_size, q_len, static_kv_len=STATIC_KV_LEN):
    device = "cuda"
    torch.manual_seed(42)

    eff_kv = q_len
    qpg = min(4, q_len)
    num_groups = math.ceil(q_len / qpg)
    num_splits = (static_kv_len + 128 - 1) // 128  # = 2
    v_splits = _mla_tp4_v_splits(static_kv_len)
    head_groups = _mla_tp4_head_groups()
    rd_dv = _mla_tp4_rd_dv()
    partial_blocks = batch_size * num_groups * num_splits

    print(f"\n{'='*64}")
    print("Test: mla_mtp_decode_tp4 + reduce (TP=4) via test_mode")
    print(f"  B={batch_size}, Q_LEN={q_len}, eff_kv(runtime)={eff_kv}, "
          f"static_kv={static_kv_len}")
    print(f"  H_per_rank={H_PER_RANK}, D_K={D_K}, D_V={D_V}")
    print(f"  qpg={qpg}, num_groups={num_groups}, num_splits={num_splits}, "
          f"v_splits={v_splits}, head_groups={head_groups}, rd_dv={rd_dv}")
    print(f"{'='*64}")

    # Build full-128-head Q for reference, then slice to rank-0 (first 32 heads).
    q_full = torch.randn(
        batch_size * q_len * NUM_HEADS, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(
        batch_size * eff_kv, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    q_rank = (q_full.view(batch_size * q_len, NUM_HEADS, D_K)
              [:, :H_PER_RANK, :]
              .reshape(-1, D_K).contiguous())

    # Intermediate partial buffers. Pre-fill LSE=-inf so the inactive split
    # reduces to weight 0.
    partial_o = torch.zeros(
        partial_blocks, D_V * 128, device=device, dtype=torch.bfloat16)
    partial_lse = torch.full(
        (partial_blocks, 128), float('-inf'),
        device=device, dtype=torch.float32)

    # Final output: [batch * q_len, H_per_rank * D_V]
    final_out = torch.zeros(
        batch_size * q_len, H_PER_RANK * D_V,
        device=device, dtype=torch.bfloat16)

    prompt_lengths = torch.full(
        (batch_size,), q_len, dtype=torch.int32, device=device)
    tokens = torch.zeros(
        (batch_size, max(static_kv_len, 256)), dtype=torch.int64, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = q_len * batch_size
    params["max_num_batched_requests"] = batch_size
    params["max_seq_length"] = max(static_kv_len, 256)
    params["max_num_pages"] = eff_kv * batch_size + 8
    params["meta_tensors"] = {
        "tokens": tokens,
        "prompt_lengths": prompt_lengths,
    }
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q_rank, name="q_input")
    kv_dt = pk.attach_input(kv, name="kv_input")
    po_dt = pk.attach_input(partial_o, name="partial_o")
    pl_dt = pk.attach_input(partial_lse, name="partial_lse")
    out_dt = pk.attach_input(final_out, name="final_out")

    pk.mla_mtp_decode_tp4_layer(q_dt, kv_dt, po_dt, pl_dt, q_len, static_kv_len)
    pk.mla_mtp_decode_tp4_reduce_layer(po_dt, pl_dt, out_dt, q_len, static_kv_len)

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    # Full causal-MLA reference with 128 heads, slice first H_per_rank=32.
    ref_full = mla_full_ref(
        q_full, kv, batch_size=batch_size, q_len=q_len, kv_len=eff_kv,
        num_heads=NUM_HEADS)  # [B, q_len, 128, D_V]
    ref_out = ref_full[:, :, :H_PER_RANK, :].reshape(
        batch_size * q_len, H_PER_RANK * D_V)

    out_view = final_out.reshape(batch_size * q_len, H_PER_RANK * D_V)
    diff = (out_view.float() - ref_out.float()).abs().max().item()
    a = out_view.float().reshape(-1)
    b = ref_out.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"  max |final - ref| = {diff:.6f}  cos={cos:.6f}")

    ok_cos = cos > 0.99
    ok_close = True
    try:
        torch.testing.assert_close(out_view, ref_out, rtol=2e-2, atol=2e-2)
    except AssertionError:
        ok_close = False
    ok = ok_cos or ok_close
    if not ok:
        print(f"  FAILED (bs={batch_size}, q_len={q_len}): "
              f"cos={cos:.6f} max_diff={diff:.6f}")

    pk.finalize()
    return ok, diff, cos


def test_mla_mtp_decode_tp4_testmode():
    results = []
    for bs, q_len in MATRIX:
        ok, d, cos = _run_case(bs, q_len)
        results.append((bs, q_len, ok, d, cos))

    print(f"\n{'='*64}")
    print("SUMMARY  mla_mtp_decode_tp4 + reduce (TP=4) final-output")
    n_pass = 0
    for bs, q_len, ok, d, cos in results:
        tag = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        print(f"  bs={bs:2d} q_len={q_len} eff_kv={q_len}  "
              f"max|o|={d:.5f} cos={cos:.6f}  {tag}")
    print(f"  {n_pass}/{len(results)} PASS")
    print(f"{'='*64}")
    if n_pass == len(results):
        print("ALL PASSED")
        return True
    else:
        print(f"FAILED: {len(results)-n_pass} config(s) failed")
        return False


if __name__ == "__main__":
    ok = test_mla_mtp_decode_tp4_testmode()
    sys.exit(0 if ok else 1)
