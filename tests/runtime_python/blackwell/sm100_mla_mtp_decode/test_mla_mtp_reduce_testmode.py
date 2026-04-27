"""
Test: mla_mtp_reduce_sm100 via PersistentKernel test_mode.

Builds a single-layer PK with mla_mtp_reduce_layer (no decode), feeds it
random partial-O / LSE buffers shaped exactly as decode would emit them,
and compares the reduced output against the PyTorch reference.

Test shape: B=1, Q_LEN=4, KV_LEN=256 → sk=2, num_head_groups=4 (hpb=32).

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


def test_mla_mtp_reduce_testmode():
    device = "cuda"
    torch.manual_seed(43)

    batch_size = 1
    q_len = 4
    kv_len = 256

    hpb = 128 // q_len
    while 128 % hpb != 0:
        hpb -= 1
    num_head_groups = 128 // hpb            # = 4
    num_splits = (kv_len + 128 - 1) // 128  # = 2

    print(f"\n{'='*60}")
    print("Test: mla_mtp_reduce_sm100 via PersistentKernel test_mode")
    print(f"  B={batch_size}, Q_LEN={q_len}, KV_LEN={kv_len}")
    print(f"  num_head_groups={num_head_groups}, hpb={hpb}, sk={num_splits}")
    print(f"{'='*60}")

    # Synthetic decode outputs.
    partial_blocks = batch_size * num_head_groups * num_splits
    input_partial = torch.randn(
        partial_blocks, D_V * 128, device=device, dtype=torch.bfloat16) * 0.1
    input_lse = torch.randn(
        partial_blocks, 128, device=device, dtype=torch.float32) * 0.5
    input_partial = input_partial.contiguous()
    input_lse = input_lse.contiguous()

    # Output: [B, Q_LEN, NUM_HEADS, D_V] — kernel writes per-(q,h_global,d).
    # Allocated as [B, Q_LEN*NUM_HEADS*D_V] in deepseek_v3 builder
    # (attn_out [mbt, num_local_q_heads*v_head_dim]); use that flat layout.
    output = torch.zeros(
        batch_size * q_len, NUM_HEADS * D_V,
        device=device, dtype=torch.bfloat16)

    # Meta-tensor stub: only qo_indptr_buffer is read by the reduce task.
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for bi in range(batch_size):
        qo_indptr[bi + 1] = qo_indptr[bi] + q_len

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(q_len * batch_size, 1)
    params["max_num_batched_requests"] = batch_size
    params["max_seq_length"] = max(kv_len, 256)
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo_indptr,
    }
    pk = PersistentKernel(**params)

    # Attach.
    ip_dt = pk.attach_input(input_partial, name="input_partial")
    il_dt = pk.attach_input(input_lse, name="input_lse")
    out_dt = pk.attach_input(output, name="output")

    # Layer.
    pk.mla_mtp_reduce_layer(ip_dt, il_dt, out_dt, q_len, kv_len)

    print("Compiling test kernel...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref = mla_mtp_reduce_ref(
        input_partial, input_lse,
        batch_size=batch_size, q_len=q_len,
        num_head_groups=num_head_groups, num_splits=num_splits)
    # Output layout from kernel:
    #   O[(bi*Q_LEN + q) * H * D_V + h_global * D_V + d]
    # so reshape to [B, Q_LEN, H, D_V].
    out_view = output.reshape(batch_size, q_len, NUM_HEADS, D_V)

    diff = (out_view.float() - ref.float()).abs().max().item()
    print(f"max |output - ref| = {diff}")

    try:
        torch.testing.assert_close(
            out_view, ref, rtol=1e-2, atol=1e-2)
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    print("PASSED")
    pk.finalize()


if __name__ == "__main__":
    test_mla_mtp_reduce_testmode()
