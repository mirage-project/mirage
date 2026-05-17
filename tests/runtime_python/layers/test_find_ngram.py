"""Smoke tests for layers.mtp.find_ngram.FindNgram (2 scopes).

``forward()`` raises ``NotImplementedError`` (the kernel depends on
runtime meta tensors). Smoke test: instantiate → compile → run → no
crash, finite output buffers.

The kernel headers (``tasks/speculative_decoding/prompt_lookup.cuh``)
are now included in ``task_header.cuh`` so the megakernel compiles.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.mtp.find_ngram import FindNgram


def _make_pk(batch_size, max_seq):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    params["max_seq_length"] = max_seq
    return PersistentKernel(**params)


def test_find_ngram_partial_smoke():
    device = "cuda"
    batch_size = 1
    seq_len = 32
    num_tasks = 4
    ngram_size = 3

    tokens = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
    output = torch.zeros(batch_size, num_tasks, dtype=torch.int64, device=device)
    module = FindNgram(ngram_size=ngram_size, scope="partial")

    pk = _make_pk(batch_size, seq_len)
    tokens_dt = pk.attach_input(tokens, name="tokens")
    output_dt = pk.attach_input(output, name="output")

    print("Building FindNgram(partial) ...")
    with pk.compile_scope():
        module.compile(input=tokens_dt, output=output_dt,
                       grid_dim=(num_tasks, 1, 1), block_dim=(128, 1, 1))

    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    assert torch.isfinite(output.float()).all(), "FindNgram(partial) emitted non-finite"
    print(f"FindNgram(partial) output[0,:]: {output[0].tolist()}")
    print("PASSED: FindNgram(partial) smoke")
    pk.finalize()


def test_find_ngram_global_smoke():
    device = "cuda"
    batch_size = 1
    num_tasks = 4
    spec_length = 3
    vocab = 128
    ngram_size = 3

    partial_results = torch.zeros(batch_size, num_tasks, dtype=torch.int64,
                                  device=device)
    tokens = torch.zeros(batch_size, vocab, dtype=torch.int64, device=device)
    output = torch.zeros(batch_size, spec_length + 1, dtype=torch.int64,
                        device=device)
    module = FindNgram(ngram_size=ngram_size, spec_length=spec_length,
                       scope="global")

    pk = _make_pk(batch_size, vocab)
    partial_dt = pk.attach_input(partial_results, name="partial")
    tokens_dt = pk.attach_input(tokens, name="tokens_in")
    output_dt = pk.attach_input(output, name="output")

    print("Building FindNgram(global) ...")
    with pk.compile_scope():
        module.compile(partial_results=partial_dt, tokens=tokens_dt,
                       output=output_dt, grid_dim=(1, 1, 1),
                       block_dim=(128, 1, 1))

    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    assert torch.isfinite(output.float()).all(), "FindNgram(global) emitted non-finite"
    print(f"FindNgram(global) output[0,:]: {output[0].tolist()}")
    print("PASSED: FindNgram(global) smoke")
    pk.finalize()


if __name__ == "__main__":
    test_find_ngram_partial_smoke()
    test_find_ngram_global_smoke()
    print("All FindNgram tests completed.")
