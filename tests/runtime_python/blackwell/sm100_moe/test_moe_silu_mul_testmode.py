"""
Test: BF16 MoE SiLU-mul via PersistentKernel test_mode (DSV3 shapes).

Exercises the per-expert fused SiLU + multiply (moe_silu_mul_layer) end-to-end
through the full MPK compilation pipeline at real DeepSeek-V3 routed-expert
shapes:

  output[b, k, :I] = silu(input[b, k, :I]) * input[b, k, I:]

DSV3 OLD-MoE 3D path: input (bs, TOPK, 2*I) gate||up → output (bs, TOPK, I),
where I = MOE_INTERMEDIATE / routed_tp_size (ep_size=1 → routed_tp = world_size).
The gate||up chunk is split at `I` inside the kernel (first I = gate, last I = up).

Per-element op → I is TP-sharded only as a SHAPE selector, so we sweep both the
TP-shard (routed_tp ∈ {1,2,4,8} → I ∈ {2048,1024,512,256}) and bs ∈ {1,2,4,8,16}.

Run:
    python tests/runtime_python/blackwell/sm100_moe/test_moe_silu_mul_testmode.py
"""

import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import moe_silu_mul_ref

# DSV3 constants.
MOE_INTERMEDIATE = 2048
NUM_EXPERTS_PER_TOK = 8

# Union-of-axes matrix: every routed_tp value × every bs value at least once.
#   {routed_tp=1} × {bs=1,2,4,8,16} ∪ {bs=16} × {routed_tp=2,4,8} ∪ {routed_tp=8, bs=1}
MATRIX = (
    [(1, bs) for bs in (1, 2, 4, 8, 16)]
    + [(rtp, 16) for rtp in (2, 4, 8)]
    + [(8, 1)]
)


def _run_case(routed_tp: int, bs: int) -> tuple[bool, float]:
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    intermediate = MOE_INTERMEDIATE // routed_tp  # I (per-rank)

    print(f"\n{'='*60}")
    print(f"moe_silu_mul: routed_tp={routed_tp} bs={bs} "
          f"TOPK={NUM_EXPERTS_PER_TOK} I={intermediate}")
    print(f"{'='*60}")

    # input: (bs, TOPK, 2*I) gate||up, output: (bs, TOPK, I)
    input_act = torch.randn(
        bs, NUM_EXPERTS_PER_TOK, intermediate * 2, dtype=dtype, device=device)
    output = torch.zeros(
        bs, NUM_EXPERTS_PER_TOK, intermediate, dtype=dtype, device=device)

    ref = moe_silu_mul_ref(input_act, intermediate)

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

    input_dt = pk.attach_input(input_act, name="input")
    output_dt = pk.attach_input(output, name="output")

    # grid: (bs, TOPK, 1) — one TB per (token, slot). Mirrors builder L3768/L4473.
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.moe_silu_mul_layer(
        input=input_dt,
        output=output_dt,
        grid_dim=(bs, NUM_EXPERTS_PER_TOK, 1),
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
    for routed_tp, bs in MATRIX:
        ok, md = _run_case(routed_tp, bs)
        results.append((routed_tp, bs, ok, md))

    n_pass = sum(1 for *_, ok, _ in results)
    print(f"\n{'='*60}")
    print("moe_silu_mul matrix summary")
    print(f"{'='*60}")
    for routed_tp, bs, ok, md in results:
        print(f"  routed_tp={routed_tp:<2} bs={bs:<3} "
              f"max_diff={md:.6g}  {'PASS' if ok else 'FAIL'}")
    print(f"\n{n_pass}/{len(results)} PASS")
    if n_pass == len(results):
        print("ALL PASS")
    else:
        raise AssertionError(f"moe_silu_mul: {len(results)-n_pass} config(s) FAILED")


def test_moe_silu_mul_testmode():
    main()


if __name__ == "__main__":
    main()
