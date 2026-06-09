"""Test: embed_layer via PersistentKernel test_mode (DSV3 shapes).

Op:  out[b, :] = weight[input_ids[b], :]
     weight: (VOCAB=129280, HIDDEN=7168) bf16
     input_ids: (bs, 1) int64  (kernel reads as flat int64 array)
     out: (bs, HIDDEN) bf16

Builder call (main embed at line 5219):
    grid_dim=(HIDDEN//128, 1, 1) = (56, 1, 1), block_dim=(128,1,1), input_source=1.
    input_map on output is (1, 0, -1): grid.x partitions dim 1 (HIDDEN into 128-col chunks).

Embedding is exact memory copy → assert byte-exact (torch.equal).

DSV3 sweep: bs ∈ {1,2,4,8,16}.

Run:
    python tests/runtime_python/blackwell/sm100_embed/test_embed_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import embed_ref

VOCAB = 129280
HIDDEN = 7168
BS_SWEEP = [1, 2, 4, 8, 16]


def _run_case(bs: int):
    device = "cuda"
    torch.manual_seed(42 + bs)

    # input_ids: flat int64 array of length bs (shape (bs, 1) per existing test convention).
    input_ids = torch.randint(0, VOCAB, (bs, 1), dtype=torch.int64, device=device)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device=device)
    out = torch.zeros(bs, HIDDEN, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs

    pk = PersistentKernel(**params)
    in_dt = pk.attach_input(input_ids, name="input_ids")
    w_dt = pk.attach_input(weight, name="weight")
    out_dt = pk.attach_input(out, name="out")

    # Mirror DSV3 builder: grid=(HIDDEN//128, 1, 1), block=(128,1,1), input_source=1.
    grid_x = HIDDEN // 128  # = 56
    pk.embed_layer(
        input=in_dt,
        weight=w_dt,
        output=out_dt,
        grid_dim=(grid_x, 1, 1),
        block_dim=(128, 1, 1),
        input_source=1,
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    ref = embed_ref(input_ids, weight)
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  bs={bs:2d}  max_diff={max_diff:.6f}", end="")

    # Embedding is a pure gather (no FP math) → byte-exact.
    if not torch.equal(out, ref):
        print(f"  FAIL  (max_diff={max_diff})")
        sys.exit(1)
    print("  PASS (byte-exact)")
    pk.finalize()


def test_embed_testmode():
    print(f"\n{'='*60}")
    print(f"embed_layer  VOCAB={VOCAB}  HIDDEN={HIDDEN}  bs sweep={BS_SWEEP}")
    for bs in BS_SWEEP:
        _run_case(bs)
    print("ALL PASS")


if __name__ == "__main__":
    test_embed_testmode()
