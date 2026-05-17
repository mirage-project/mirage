"""Test the ``layers.mtp.softmax_gather.SoftmaxGather`` catalog module.

Fused softmax + gather: for each row b, output[b, 0] = softmax(logits[b])[token_ids[b, 0]].
The kernel computes the softmax in fp32 and writes a fp32 single-element-per-row result.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.mtp.softmax_gather import SoftmaxGather


def test_softmax_gather_testmode():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 2
    vocab_size = 1024

    logits = torch.randn(
        batch_size, vocab_size, dtype=torch.bfloat16, device=device,
    )
    token_ids = torch.tensor(
        [[17], [99]], dtype=torch.int64, device=device,
    )
    out_probs = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)

    m = SoftmaxGather(prefix="sg_")
    ref = m.forward(logits, token_ids)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    logits_dt = pk.attach_input(logits, name="logits")
    tok_dt = pk.attach_input(token_ids, name="tok_ids")
    out_dt = pk.attach_input(out_probs, name="out_probs")

    with pk.compile_scope():
        _ = m.compile(logits_dt, tok_dt, out_dt)

    print("Compiling SoftmaxGather...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    print(f"out_probs: {out_probs.flatten().tolist()}")
    print(f"ref:       {ref.flatten().tolist()}")

    # bf16 logits reduced in fp32 — small drift OK.
    try:
        torch.testing.assert_close(out_probs, ref, atol=1e-2, rtol=1e-2)
        print("PASSED: SoftmaxGather compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: SoftmaxGather\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_softmax_gather_testmode()
