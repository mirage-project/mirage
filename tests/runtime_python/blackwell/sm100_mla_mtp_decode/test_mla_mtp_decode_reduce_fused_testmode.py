"""
Test: fused mla_mtp_decode_sm100 -> mla_mtp_reduce_sm100 (TP1) via test_mode.

Builds BOTH layers in ONE PersistentKernel graph (decode writes partial-O /
partial-LSE, reduce consumes them and produces the final reduced output) and
compares the FINAL reduced output [B, q_len, H=128, D_V=512] against the
TP-agnostic full causal-MLA reference ``mla_full_ref`` (a single un-split
softmax). This mirrors the production decode wiring in
``deepseek_v3/builder.py`` (decode -> mla_partial_o/lse -> reduce -> attn_out)
at TP1 (NUM_HEADS=128).

Despite "mtp" in the names these are the MAIN TP1 decode attention kernels.

OFFLINE TEST-MODE CONSTRAINT (verified empirically — see decision log entry
"sm100_mla_mtp_decode"):
  In MODE_OFFLINE test mode `step` is forced to 0 on the only iteration the
  layers run, so the runtime derives kv_len_ == q_len_rt_ == prompt_length
  (<= 8). The decode kernel reads the runtime kv_len_ as both the per-batch KV
  stride AND the attention length, so the *effective* KV sequence is locked to
  q_len and is always a single TILE_S(128) tile -> runtime sk = 1. It is
  therefore IMPOSSIBLE to make >1 split SIMULTANEOUSLY active in offline test
  mode; the task's "force num_splits>1 via kv_len>128" is unreachable here (a
  fundamental harness limitation, NOT a kernel defect). We still register the
  decode with a static kv_len>128 (num_splits=2) so the kernel takes the
  partial-write path (WRITE_FINAL=false) and the reduce performs a real
  LSE-merge over the static splits; the runtime-inactive split (si=1) takes the
  kernel's t0>=t1 no-op and is reduced away because we pre-fill its LSE=-inf.
  This still exercises the genuine decode-partial -> reduce -> final pipeline
  (one active split) end-to-end against the full-MLA reference.

  KV-cache batch stride == runtime kv_len_ (= q_len), NOT the static kv_len.

Sweep (bs x q_len x static_kv):
    bs    in {1, 2, 4, 8, 16}
    q_len in {1, 2, 4}
    static_kv in {256 (sk=2), 384 (sk=3)}   (grid/block_linear variety)

Run:
    python tests/runtime_python/blackwell/sm100_mla_mtp_decode/test_mla_mtp_decode_reduce_fused_testmode.py
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import (
    mla_full_ref,
    NUM_HEADS,
    D_K,
    D_V,
)

# Union-of-axes: every bs at static_kv=256 (sk=2), plus the q_len axis and the
# sk=3 grid swept at the corners. Keeps the GPU compile count bounded while
# covering every bs, every q_len, and both static sk={2,3}.
_BASE_QLEN = 2
MATRIX = []
for bs in (1, 2, 4, 8, 16):
    MATRIX.append((bs, _BASE_QLEN, 256))  # sk=2, all bs
for q_len in (1, 4):
    MATRIX.append((1, q_len, 256))
    MATRIX.append((16, q_len, 256))
# sk=3 grid variety at the corners.
MATRIX.append((1, 2, 384))
MATRIX.append((16, 2, 384))


def _run_case(batch_size, q_len, static_kv_len):
    device = "cuda"
    torch.manual_seed(1234)

    # Effective (runtime) kv length is locked to q_len in offline test mode.
    eff_kv = q_len

    hpb = 128 // q_len
    while 128 % hpb != 0:
        hpb -= 1
    num_head_groups = 128 // hpb
    num_splits = (static_kv_len + 128 - 1) // 128
    assert num_splits > 1, "decode must take the partial-write path"

    print(f"\n{'='*64}")
    print("Test: fused mla_mtp_decode -> mla_mtp_reduce (TP1) test_mode")
    print(f"  B={batch_size}, Q_LEN={q_len}, eff_kv(runtime)={eff_kv}, "
          f"static_kv={static_kv_len}")
    print(f"  H={NUM_HEADS}, D_K={D_K}, D_V={D_V}")
    print(f"  num_head_groups={num_head_groups}, hpb={hpb}, sk(static)={num_splits}")
    print(f"{'='*64}")

    # Inputs. KV batch stride == runtime kv_len_ (= eff_kv = q_len).
    q = torch.randn(
        batch_size * q_len * NUM_HEADS, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(
        batch_size * eff_kv, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    q = q.contiguous()
    kv = kv.contiguous()

    # Intermediate partial buffers. Pre-fill LSE=-inf so the runtime-inactive
    # split reduces to weight 0.
    partial_blocks = batch_size * num_head_groups * num_splits
    partial_o = torch.zeros(
        partial_blocks, D_V * 128, device=device, dtype=torch.bfloat16)
    partial_lse = torch.full(
        (partial_blocks, 128), float('-inf'),
        device=device, dtype=torch.float32)

    # Final reduced output: [B*q_len, H*D_V] (attn_out flat layout in builder).
    final_out = torch.zeros(
        batch_size * q_len, NUM_HEADS * D_V,
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

    q_dt = pk.attach_input(q, name="q_input")
    kv_dt = pk.attach_input(kv, name="kv_input")
    po_dt = pk.attach_input(partial_o, name="partial_o")
    pl_dt = pk.attach_input(partial_lse, name="partial_lse")
    out_dt = pk.attach_input(final_out, name="final_out")

    pk.mla_mtp_decode_layer(q_dt, kv_dt, po_dt, pl_dt, q_len, static_kv_len)
    pk.mla_mtp_reduce_layer(po_dt, pl_dt, out_dt, q_len, static_kv_len)

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    # Full causal-MLA reference at the effective kv length, TP1 -> H=128.
    ref = mla_full_ref(
        q, kv, batch_size=batch_size, q_len=q_len, kv_len=eff_kv,
        num_heads=NUM_HEADS)
    out_view = final_out.reshape(batch_size, q_len, NUM_HEADS, D_V)

    diff = (out_view.float() - ref.float()).abs().max().item()
    a = out_view.float().reshape(-1)
    b = ref.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"  max |final - ref| = {diff:.6f}  cos={cos:.6f}")

    # bf16 softmax tolerance (decision-log spec): cos > 0.99 OR atol/rtol ~ 2e-2.
    ok_cos = cos > 0.99
    ok_close = True
    try:
        torch.testing.assert_close(out_view, ref, rtol=2e-2, atol=2e-2)
    except AssertionError:
        ok_close = False
    ok = ok_cos or ok_close
    if not ok:
        print(f"  FAILED (bs={batch_size}, q_len={q_len}, static_kv={static_kv_len}): "
              f"cos={cos:.6f} max_diff={diff:.6f}")

    pk.finalize()
    return ok, diff, cos


def test_mla_mtp_decode_reduce_fused_testmode():
    results = []
    for bs, q_len, static_kv in MATRIX:
        ok, d, cos = _run_case(bs, q_len, static_kv)
        results.append((bs, q_len, static_kv, ok, d, cos))

    print(f"\n{'='*64}")
    print("SUMMARY  fused mla_mtp_decode -> reduce (TP1) final-output")
    n_pass = 0
    for bs, q_len, static_kv, ok, d, cos in results:
        tag = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        print(f"  bs={bs:2d} q_len={q_len} static_kv={static_kv:3d} eff_kv={q_len}  "
              f"max|o|={d:.5f} cos={cos:.6f}  {tag}")
    print(f"  {n_pass}/{len(results)} PASS")
    print(f"{'='*64}")
    assert n_pass == len(results), f"{len(results)-n_pass} config(s) FAILED"
    print("ALL PASSED")


if __name__ == "__main__":
    test_mla_mtp_decode_reduce_fused_testmode()
