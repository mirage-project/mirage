"""
Test: BF16 MoE weighted-sum + residual via PersistentKernel test_mode (DSV3 shapes).

Exercises moe_mul_sum_add_layer (the OLD-MoE final reduction = topk-weighted
combine of the per-expert W2 outputs + shared-expert residual) end-to-end
through the full MPK compilation pipeline:

  output[b, :] = sum_k (moe_down_out[b, k, :] * topk_weights[b, k]) + shared_residual[b, :]

DSV3 shapes (ep_size=1): moe_down_out (bs, TOPK, HIDDEN), topk_weights (bs, TOPK),
shared_residual (bs, HIDDEN) → output (bs, HIDDEN). HIDDEN=7168 is NOT TP-sharded,
so this is a bs-only sweep per the matrix policy.

Run:
    python tests/runtime_python/blackwell/sm100_moe/test_moe_mul_sum_add_testmode.py
"""

import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import moe_mul_sum_add_ref

# DSV3 constants.
HIDDEN = 7168
NUM_EXPERTS_PER_TOK = 8
# builder _moe_hidden_split(7168, 56) → 56 (7168/56=128, 128%128==0).
HIDDEN_SPLIT = 56

MATRIX = [1, 2, 4, 8, 16]  # bs-only sweep (HIDDEN not sharded)


def _run_case(bs: int) -> tuple[bool, float]:
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    print(f"\n{'='*60}")
    print(f"moe_mul_sum_add: bs={bs} TOPK={NUM_EXPERTS_PER_TOK} HIDDEN={HIDDEN}")
    print(f"{'='*60}")

    # moe_down_out: (bs, TOPK, HIDDEN) — per-expert down-proj outputs.
    x = torch.randn(bs, NUM_EXPERTS_PER_TOK, HIDDEN, dtype=dtype, device=device)
    # shared_residual: (bs, HIDDEN) — shared-expert output (added once).
    residual = torch.randn(bs, HIDDEN, dtype=dtype, device=device)
    # topk_weights: (bs, TOPK) float32 — DSV3 sigmoid routing weights are
    # normalized & scaled (scaling_factor=2.5); positive, sum ~ scaling.
    raw = torch.rand(bs, NUM_EXPERTS_PER_TOK, dtype=torch.float, device=device)
    topk_weights = (raw / raw.sum(dim=1, keepdim=True)) * 2.5
    output = torch.zeros(bs, HIDDEN, dtype=dtype, device=device)

    ref = moe_mul_sum_add_ref(x, topk_weights, residual)

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

    input_dt = pk.attach_input(x, name="input")
    weight_dt = pk.attach_input(topk_weights, name="weight")
    residual_dt = pk.attach_input(residual, name="residual")
    output_dt = pk.attach_input(output, name="output")

    # grid: (bs, _moe_hidden_split(HIDDEN), 1) — mirrors builder L3888/L4564.
    block_dim = (128, 1, 1)
    pk.moe_mul_sum_add_layer(
        input=input_dt,
        weight=weight_dt,
        residual=residual_dt,
        output=output_dt,
        grid_dim=(bs, HIDDEN_SPLIT, 1),
        block_dim=block_dim,
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    max_diff = (output.float() - ref.float()).abs().max().item()
    try:
        torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
        ok = True
    except AssertionError as e:
        ok = False
        print(e)
    print(f"  max_diff={max_diff:.6g}  {'PASS' if ok else 'FAIL'}")

    pk.finalize()
    return ok, max_diff


def main():
    results = []
    for bs in MATRIX:
        ok, md = _run_case(bs)
        results.append((bs, ok, md))

    n_pass = sum(1 for _, ok, _ in results)
    print(f"\n{'='*60}")
    print("moe_mul_sum_add matrix summary")
    print(f"{'='*60}")
    for bs, ok, md in results:
        print(f"  bs={bs:<3} max_diff={md:.6g}  {'PASS' if ok else 'FAIL'}")
    print(f"\n{n_pass}/{len(results)} PASS")
    if n_pass == len(results):
        print("ALL PASS")
    else:
        raise AssertionError(
            f"moe_mul_sum_add: {len(results)-n_pass} config(s) FAILED")


def test_moe_mul_sum_add_testmode():
    main()


if __name__ == "__main__":
    main()
