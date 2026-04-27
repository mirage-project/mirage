"""Test mode for argmax_partial_layer (sm100).

Splits vocab into ``num_tasks`` chunks; each task writes (max_value,
relative_index_within_chunk) for every batch element.
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import argmax_partial_ref


def test_argmax_partial_testmode():
    device = "cuda"
    torch.manual_seed(42)

    batch_size = 4
    vocab = 4096
    num_tasks = 4  # vocab must be divisible by num_tasks
    assert vocab % num_tasks == 0

    x = torch.randn(batch_size, vocab, dtype=torch.bfloat16, device=device)
    # Output buffers: per-task (value, index) for each batch row.
    part_value = torch.zeros(batch_size, num_tasks, dtype=torch.bfloat16, device=device)
    # Indices are int64 in the kernel (long long *).
    part_index = torch.full(
        (batch_size, num_tasks), -1, dtype=torch.int64, device=device
    )

    # The argmax kernel reads num_active_tokens from
    # qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]; provide the standard stub
    # that marks all batch rows valid.
    qo_indptr_buffer = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr_buffer[batch_size] = batch_size

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    params["meta_tensors"] = {"qo_indptr_buffer": qo_indptr_buffer}
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    v_dt = pk.attach_input(part_value, name="part_value")
    i_dt = pk.attach_input(part_index, name="part_index")

    # Use 128-thread blocks (matches deepseek_v3 builder usage of argmax).
    block_dim = (128, 1, 1)

    pk.argmax_partial_layer(
        input=x_dt,
        output=(v_dt, i_dt),
        grid_dim=(num_tasks, 1, 1),
        block_dim=block_dim,
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    print("Compiling argmax_partial test kernel...")
    pk.compile(output_dir=folder_path)

    print("Running argmax_partial test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref_v, ref_i = argmax_partial_ref(x, num_tasks)

    # Compare values exactly: pure max over disjoint chunks of the same input
    # in bfloat16 should produce identical results in PyTorch and the kernel.
    if not torch.equal(part_value, ref_v.to(torch.bfloat16)):
        max_diff = (part_value.float() - ref_v.float()).abs().max().item()
        print(f"VALUE MISMATCH: max abs diff = {max_diff}")
        print(f"got    [:2,:]: {part_value[:2]}")
        print(f"ref    [:2,:]: {ref_v[:2]}")
        sys.exit(1)

    if not torch.equal(part_index.to(torch.int64), ref_i.to(torch.int64)):
        diff = (part_index.to(torch.int64) != ref_i.to(torch.int64))
        n_bad = int(diff.sum().item())
        print(f"INDEX MISMATCH: {n_bad} entries differ (out of {part_index.numel()})")
        print(f"got    [:2,:]: {part_index[:2]}")
        print(f"ref    [:2,:]: {ref_i[:2]}")
        sys.exit(1)

    print("PASSED: argmax_partial test_mode produces correct (value, index) pairs")
    pk.finalize()


if __name__ == "__main__":
    test_argmax_partial_testmode()
