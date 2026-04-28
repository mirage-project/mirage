"""Test mode unit test for softmax_gather_layer (sm100)."""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import softmax_gather_ref


def test_softmax_gather_testmode():
    device = "cuda"
    torch.manual_seed(42)

    batch_size = 4
    vocab = 4096

    # Inputs: bf16 logits, int64 token_ids (kernel casts long long -> int)
    logits = torch.randn(batch_size, vocab, dtype=torch.bfloat16, device=device)
    token_ids = torch.randint(0, vocab, (batch_size, 1), dtype=torch.int64, device=device)
    out = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)

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

    l_dt = pk.attach_input(logits, name="logits")
    t_dt = pk.attach_input(token_ids, name="tok")
    o_dt = pk.attach_input(out, name="out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.softmax_gather_layer(
        logits=l_dt,
        token_ids=t_dt,
        output_probs=o_dt,
        grid_dim=(batch_size, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling test kernel...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ref = softmax_gather_ref(logits, token_ids)
    print(f"Output:\n{out}")
    print(f"Reference:\n{ref}")

    max_diff = (out - ref).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)
    print("PASSED: softmax_gather test_mode produces correct output")

    pk.finalize()


if __name__ == "__main__":
    test_softmax_gather_testmode()
