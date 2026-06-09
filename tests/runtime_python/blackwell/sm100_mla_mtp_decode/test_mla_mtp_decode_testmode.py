"""
Test: mla_mtp_decode_sm100 (TP1) via PersistentKernel test_mode.

Builds a single-layer PK with mla_mtp_decode_layer, compiles it, runs once,
and compares the per-split partial attention outputs (bf16) and LSE values
(fp32) against the PyTorch partial-level reference in pytorch_reference.py.

Despite "mtp" in the name this is the MAIN TP1 decode attention kernel
(core forward path, not speculative/MTP).

  Constants hard-coded in the kernel (DeepSeek V3 MLA): NUM_HEADS=128,
  D_K=576, D_V=512, TILE_S=128.

OFFLINE TEST-MODE CONSTRAINT (verified empirically — see decision log):
  In MODE_OFFLINE test mode the runtime forces step=0 on the only iteration
  the layer-under-test runs, so prepare_next_batch derives
      kv_len_ (runtime) == q_len_rt_ (runtime) == prompt_length   (<= 8).
  There is NO way to inject KV history, so the *runtime* kv length is locked
  to q_len and is always a single TILE_S(128) tile -> runtime sk = 1. The
  multi-split (sk>1) partial path with >1 simultaneously-active split is
  therefore UNREACHABLE in offline test mode. We still register the layer with
  a static kv_len=256 (num_splits=2) so the kernel takes the *partial-write*
  path (WRITE_FINAL=false) rather than the write-final shortcut, exercising the
  real partial-O / LSE store; the inactive second split takes the kernel's
  t0>=t1 no-op. The KV-cache batch stride equals the *runtime* kv_len_ (=q_len),
  NOT the static param.

Sweep (bs x q_len):
    bs    in {1, 2, 4, 8, 16}
    q_len in {1, 2, 4}        (decode regime, <= 8)

Run:
    python tests/runtime_python/blackwell/sm100_mla_mtp_decode/test_mla_mtp_decode_testmode.py
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# Make sibling import work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import (
    mla_mtp_decode_ref,
    NUM_HEADS,
    D_K,
    D_V,
)

# Static kv_len for the layer params -> num_splits=2 so the kernel uses the
# partial-write path (not WRITE_FINAL). Runtime kv length is still q_len.
STATIC_KV_LEN = 256
MATRIX = [
    (bs, q_len)
    for bs in (1, 2, 4, 8, 16)
    for q_len in (1, 2, 4)
]


def _run_case(batch_size, q_len, static_kv_len=STATIC_KV_LEN):
    device = "cuda"
    torch.manual_seed(42)

    # Effective (runtime) kv length is locked to q_len in offline test mode.
    eff_kv = q_len

    # Mirror the layer's internal derivation.
    hpb = 128 // q_len
    while 128 % hpb != 0:
        hpb -= 1
    num_head_groups = 128 // hpb
    num_splits = (static_kv_len + 128 - 1) // 128  # = 2 (grid / block_linear)

    print(f"\n{'='*64}")
    print("Test: mla_mtp_decode_sm100 (TP1) via PersistentKernel test_mode")
    print(f"  B={batch_size}, Q_LEN={q_len}, eff_kv(runtime)={eff_kv}, "
          f"static_kv={static_kv_len}")
    print(f"  H={NUM_HEADS}, D_K={D_K}, D_V={D_V}")
    print(f"  num_head_groups={num_head_groups}, hpb={hpb}, sk(static)={num_splits}")
    print(f"{'='*64}")

    # Inputs (bf16, contiguous on CUDA).
    # KV batch stride == runtime kv_len_ (= eff_kv = q_len). The kernel reads
    # KV at row (bi*kv_len_ + kvs), so the buffer must use that exact stride.
    q = torch.randn(
        batch_size * q_len * NUM_HEADS, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(
        batch_size * eff_kv, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    q = q.contiguous()
    kv = kv.contiguous()

    # Outputs. Pre-fill LSE with -inf so the (runtime-)inactive split matches
    # the reference's inactive-split convention.
    partial_blocks = batch_size * num_head_groups * num_splits
    output_partial = torch.zeros(
        partial_blocks, D_V * 128, device=device, dtype=torch.bfloat16)
    output_lse = torch.full(
        (partial_blocks, 128), float('-inf'),
        device=device, dtype=torch.float32)

    # Drive the decode scenario via prompt_lengths (step forced to 0 at init,
    # so num_new_tokens = prompt_length = q_len -> kv_len_ = q_len_rt_ = q_len).
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
    op_dt = pk.attach_input(output_partial, name="output_partial")
    ol_dt = pk.attach_input(output_lse, name="output_lse")

    pk.mla_mtp_decode_layer(q_dt, kv_dt, op_dt, ol_dt, q_len, static_kv_len)

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    # Reference at the EFFECTIVE (runtime) kv length, same num_splits as the
    # static grid so block_linear indexing matches. si=0 is the active split,
    # si=1 is inactive (t0>=t1) -> zeros / -inf LSE.
    ref_part, ref_lse = mla_mtp_decode_ref(
        q, kv,
        batch_size=batch_size, q_len=q_len, kv_len=eff_kv,
        num_head_groups=num_head_groups, num_splits=num_splits)

    # Compare the ACTIVE split (si=0) blocks and active tids only. Layout:
    #   block_linear = bi*num_head_groups*num_splits + gi*num_splits + si
    used_tid = hpb * q_len
    out_part_r = output_partial.reshape(partial_blocks, D_V, 128)
    ref_part_r = ref_part.reshape(partial_blocks, D_V, 128)
    active_blocks = [
        bi * num_head_groups * num_splits + gi * num_splits + 0
        for bi in range(batch_size) for gi in range(num_head_groups)
    ]
    ab = torch.tensor(active_blocks, device=device)
    out_part_used = out_part_r[ab][..., :used_tid].contiguous()
    ref_part_used = ref_part_r[ab][..., :used_tid].contiguous()
    out_lse_used = output_lse[ab][..., :used_tid].contiguous()
    ref_lse_used = ref_lse[ab][..., :used_tid].contiguous()

    part_diff = (out_part_used.float() - ref_part_used.float()).abs().max().item()
    lse_diff = (out_lse_used - ref_lse_used).abs().max().item()
    a = out_part_used.float().reshape(-1)
    b = ref_part_used.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"  max |partial - ref| = {part_diff:.6f}  cos={cos:.6f}")
    print(f"  max |lse - ref|     = {lse_diff:.6f}")

    ok = True
    try:
        torch.testing.assert_close(
            out_part_used, ref_part_used, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            out_lse_used, ref_lse_used, rtol=2e-2, atol=2e-2)
    except AssertionError as e:
        print(f"  FAILED (bs={batch_size}, q_len={q_len}): {e}")
        ok = False

    pk.finalize()
    return ok, part_diff, lse_diff, cos


def test_mla_mtp_decode_testmode():
    results = []
    for bs, q_len in MATRIX:
        ok, pd, ld, cos = _run_case(bs, q_len)
        results.append((bs, q_len, ok, pd, ld, cos))

    print(f"\n{'='*64}")
    print("SUMMARY  mla_mtp_decode_sm100 (TP1) partial-level (active split)")
    n_pass = 0
    for bs, q_len, ok, pd, ld, cos in results:
        tag = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        print(f"  bs={bs:2d} q_len={q_len} eff_kv={q_len}  "
              f"max|p|={pd:.5f} max|lse|={ld:.5f} cos={cos:.6f}  {tag}")
    print(f"  {n_pass}/{len(results)} PASS")
    print(f"{'='*64}")
    assert n_pass == len(results), f"{len(results)-n_pass} config(s) FAILED"
    print("ALL PASSED")


if __name__ == "__main__":
    test_mla_mtp_decode_testmode()
