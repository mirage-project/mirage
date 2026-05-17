"""Smoke test for layers.mtp.prepare_verify.MTPPrepareVerify.

Forward() raises NotImplementedError (kernel reads runtime meta-tensors).
We exercise instantiate → compile → run → no-crash / no-NaN check on
the output buffer.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.mtp.prepare_verify import MTPPrepareVerify


def test_prepare_verify_smoke():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 1
    num_draft_tokens = 2
    max_seq_len = 16
    mbt = batch_size  # max_num_batched_tokens

    main_token = torch.zeros(mbt, 1, dtype=torch.int64, device=device)
    draft_tokens = torch.zeros(
        batch_size, num_draft_tokens, dtype=torch.int64, device=device,
    )
    tokens_buffer = torch.zeros(
        batch_size, max_seq_len, dtype=torch.int64, device=device,
    )
    step = torch.zeros(batch_size, dtype=torch.int32, device=device)
    num_new_tokens = torch.zeros(1, dtype=torch.int32, device=device)

    m = MTPPrepareVerify(
        num_draft_tokens=num_draft_tokens, max_seq_len=max_seq_len,
        prefix="prep_",
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

    mt_dt = pk.attach_input(main_token, name="main_token")
    dt_dt = pk.attach_input(draft_tokens, name="draft_tokens")
    tb_dt = pk.attach_input(tokens_buffer, name="tokens_buffer")
    step_dt = pk.attach_input(step, name="step_in")
    nnt_dt = pk.attach_input(num_new_tokens, name="num_new_tokens_in")

    with pk.compile_scope():
        _ = m.compile(mt_dt, dt_dt, tb_dt, step_dt, nnt_dt)

    print("Compiling MTPPrepareVerify (smoke)...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    if tokens_buffer.isnan().any() or tokens_buffer.isinf().any():
        print("FAILED: tokens_buffer contains NaN/Inf")
        pk.finalize()
        sys.exit(1)
    print(f"tokens_buffer: {tokens_buffer.tolist()}")
    print("PASSED: MTPPrepareVerify smoke (no crash, no NaN/Inf)")
    pk.finalize()


if __name__ == "__main__":
    test_prepare_verify_smoke()
