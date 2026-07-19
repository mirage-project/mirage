"""
Test: mla_mtp_reduce_sm100 (TP1) via PersistentKernel test_mode.

Builds a single-layer PK with mla_mtp_reduce_layer (no decode), feeds it
random partial-O / LSE buffers shaped exactly as decode would emit them, and
compares the reduced output against the partial-level PyTorch reference.

Despite "mtp" in the name this is the MAIN TP1 decode reduce kernel.

The reduce kernel reads its partials directly from device memory (it does NOT
read the KV cache), so — unlike the decode kernel — it is NOT subject to the
offline-test-mode kv_len==q_len lock and CAN be exercised with a genuine
multi-split (sk=2) LSE merge. The only runtime value it reads is
q_len_rt_ = qo_indptr_buffer diff, which prepare_next_batch derives from
prompt_length (step forced to 0 -> q_len_rt_ = prompt_length = q_len).

Sweep (bs x q_len), static kv_len=256 -> sk=2 (real 2-way LSE merge):
    bs    in {1, 2, 4, 8, 16}
    q_len in {1, 2, 4}

Run:
    python tests/runtime_python/blackwell/sm100_mla_mtp_decode/test_mla_mtp_reduce_testmode.py
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import (
    mla_mtp_reduce_ref,
    NUM_HEADS,
    D_V,
)

STATIC_KV_LEN = 256  # -> sk = 2 (genuine 2-way LSE merge)
MATRIX = [
    (bs, q_len)
    for bs in (1, 2, 4, 8, 16)
    for q_len in (1, 2, 4)
]


def _run_case(batch_size, q_len, static_kv_len=STATIC_KV_LEN):
    device = "cuda"
    torch.manual_seed(43)

    hpb = 128 // q_len
    while 128 % hpb != 0:
        hpb -= 1
    num_head_groups = 128 // hpb
    num_splits = (static_kv_len + 128 - 1) // 128  # = 2

    print(f"\n{'='*64}")
    print("Test: mla_mtp_reduce_sm100 (TP1) via PersistentKernel test_mode")
    print(f"  B={batch_size}, Q_LEN={q_len}, static_kv={static_kv_len}")
    print(f"  num_head_groups={num_head_groups}, hpb={hpb}, sk={num_splits}")
    print(f"{'='*64}")

    # Synthetic decode outputs (both splits active, real 2-way merge).
    partial_blocks = batch_size * num_head_groups * num_splits
    input_partial = torch.randn(
        partial_blocks, D_V * 128, device=device, dtype=torch.bfloat16) * 0.1
    input_lse = torch.randn(
        partial_blocks, 128, device=device, dtype=torch.float32) * 0.5
    input_partial = input_partial.contiguous()
    input_lse = input_lse.contiguous()

    # Output: [B*Q_LEN, NUM_HEADS*D_V] (attn_out flat layout in builder).
    output = torch.zeros(
        batch_size * q_len, NUM_HEADS * D_V,
        device=device, dtype=torch.bfloat16)

    # Drive q_len_rt_ via prompt_lengths (step=0 -> q_len_rt_ = q_len).
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
    params["meta_tensors"] = {
        "tokens": tokens,
        "prompt_lengths": prompt_lengths,
    }
    pk = PersistentKernel(**params)

    ip_dt = pk.attach_input(input_partial, name="input_partial")
    il_dt = pk.attach_input(input_lse, name="input_lse")
    out_dt = pk.attach_input(output, name="output")

    pk.mla_mtp_reduce_layer(ip_dt, il_dt, out_dt, q_len, static_kv_len)

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    ref = mla_mtp_reduce_ref(
        input_partial, input_lse,
        batch_size=batch_size, q_len=q_len,
        num_head_groups=num_head_groups, num_splits=num_splits)
    # Output layout from kernel:
    #   O[(bi*Q_LEN + q) * H * D_V + h_global * D_V + d]
    out_view = output.reshape(batch_size, q_len, NUM_HEADS, D_V)

    diff = (out_view.float() - ref.float()).abs().max().item()
    a = out_view.float().reshape(-1)
    b = ref.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"  max |output - ref| = {diff:.6f}  cos={cos:.6f}")

    ok = True
    try:
        torch.testing.assert_close(out_view, ref, rtol=2e-2, atol=2e-2)
    except AssertionError as e:
        print(f"  FAILED (bs={batch_size}, q_len={q_len}): {e}")
        ok = False

    pk.finalize()
    return ok, diff, cos


def test_mla_mtp_reduce_testmode():
    results = []
    for bs, q_len in MATRIX:
        ok, d, cos = _run_case(bs, q_len)
        results.append((bs, q_len, ok, d, cos))

    print(f"\n{'='*64}")
    print("SUMMARY  mla_mtp_reduce_sm100 (TP1) partial-level (sk=2 merge)")
    n_pass = 0
    for bs, q_len, ok, d, cos in results:
        tag = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        print(f"  bs={bs:2d} q_len={q_len}  max|o|={d:.5f} cos={cos:.6f}  {tag}")
    print(f"  {n_pass}/{len(results)} PASS")
    print(f"{'='*64}")
    assert n_pass == len(results), f"{len(results)-n_pass} config(s) FAILED"
    print("ALL PASSED")


if __name__ == "__main__":
    test_mla_mtp_reduce_testmode()
