"""Test mode for argmax_reduce_layer (sm100).

Reduces per-partition (value, relative_index) pairs into a global vocab
index. We synthesize the partial inputs in Python so the reduce layer is
exercised in isolation (no dependency on argmax_partial running first).
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import argmax_reduce_ref_with_chunk_size


def test_argmax_reduce_testmode():
    device = "cuda"
    torch.manual_seed(7)

    batch_size = 4
    num_partitions = 4
    chunk_size = 1024  # CHUNK_SIZE template parameter (vocab // num_partitions)
    vocab = num_partitions * chunk_size  # implied global vocab

    # Synthesize partial outputs as if argmax_partial had already run.
    part_value = torch.randn(
        batch_size, num_partitions, dtype=torch.bfloat16, device=device
    )
    # Relative indices within each chunk: in [0, chunk_size).
    part_index = torch.randint(
        0, chunk_size, (batch_size, num_partitions), dtype=torch.int64, device=device
    )

    # Final output: global argmax index, [B, 1] int64.
    final_out = torch.full((batch_size, 1), -1, dtype=torch.int64, device=device)

    # The reduce kernel reads num_active_tokens from
    # qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]; provide the standard stub.
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

    # The reduce layer reads `pk.argmax_partial_output_size` (the CHUNK_SIZE
    # template arg). Normally this is set by argmax_partial_layer; we set it
    # explicitly because we're testing reduce in isolation.
    pk.argmax_partial_output_size = chunk_size

    v_dt = pk.attach_input(part_value, name="part_value")
    i_dt = pk.attach_input(part_index, name="part_index")
    out_dt = pk.attach_input(final_out, name="final_out")

    block_dim = (128, 1, 1)

    pk.argmax_reduce_layer(
        input=(v_dt, i_dt),
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    print("Compiling argmax_reduce test kernel...")
    pk.compile(output_dir=folder_path)

    print("Running argmax_reduce test kernel...")
    pk()
    torch.cuda.synchronize()

    ref = argmax_reduce_ref_with_chunk_size(part_value, part_index, chunk_size)

    if not torch.equal(final_out, ref):
        diff = (final_out != ref)
        n_bad = int(diff.sum().item())
        print(f"FAILED: {n_bad}/{final_out.numel()} entries differ")
        print(f"got: {final_out.flatten()}")
        print(f"ref: {ref.flatten()}")
        # Debug: show partial inputs for the failing batch row(s).
        bad_rows = diff.flatten().nonzero(as_tuple=False).flatten().tolist()
        for r in bad_rows[:4]:
            print(
                f"  row {r}: values={part_value[r]}, "
                f"rel_indices={part_index[r]}"
            )
        sys.exit(1)

    print("PASSED: argmax_reduce test_mode produces correct global argmax indices")
    pk.finalize()


if __name__ == "__main__":
    test_argmax_reduce_testmode()
