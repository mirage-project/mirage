"""Smoke test for layers.mtp.build_embed.MTPBuildEmbedInput.

Forward() raises NotImplementedError. Smoke test: instantiate → compile
→ run → no crash, no NaN/Inf in mtp_input_tokens.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.mtp.build_embed import MTPBuildEmbedInput


def test_build_embed_smoke():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 1
    max_seq_len = 16
    mbt = batch_size  # max_num_batched_tokens (used by kernel for "mbt")

    output_tokens = torch.zeros(mbt, 1, dtype=torch.int64, device=device)
    mtp_input_tokens = torch.zeros(mbt, 1, dtype=torch.int64, device=device)

    m = MTPBuildEmbedInput(
        batch_size=batch_size, max_seq_len=max_seq_len, prefix="be_",
    )

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_seq_length"] = max_seq_len
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    ot_dt = pk.attach_input(output_tokens, name="output_tokens_in")
    mt_dt = pk.attach_input(mtp_input_tokens, name="mtp_input_tokens")

    with pk.compile_scope():
        _ = m.compile(ot_dt, mt_dt)

    print("Compiling MTPBuildEmbedInput (smoke)...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    if mtp_input_tokens.isnan().any() or mtp_input_tokens.isinf().any():
        print("FAILED: mtp_input_tokens contains NaN/Inf")
        pk.finalize()
        sys.exit(1)
    print(f"mtp_input_tokens: {mtp_input_tokens.tolist()}")
    print("PASSED: MTPBuildEmbedInput smoke (no crash, no NaN/Inf)")
    pk.finalize()


if __name__ == "__main__":
    test_build_embed_smoke()
