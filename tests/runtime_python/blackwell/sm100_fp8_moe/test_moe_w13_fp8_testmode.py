"""DSV3 routed-expert FP8 MoE W13 (gate||up) group GEMM via test_mode.

Validates `pk.moe_w13_fp8_layer` end-to-end (Python -> codegen -> nvcc ->
runtime) against the pure-PyTorch reference `moe_w13_fp8_ref` in
pytorch_reference.py, across the DSV3 union-of-axes (tp, bs) matrix plus a
secondary ep>1 (num_local_experts=256/ep) check.

Shapes (DSV3, ep_size=1 default -> routed_tp = world_size):
  * input_fp8:    (bs, HIDDEN=7168)              FP8 E4M3 + per-128 f32 scale
  * weight_fp8:   (EL, 2*MOE_INTERMEDIATE/tp, 7168) FP8 E4M3 + f32 scale
  * output bf16:  (bs, NUM_TOPK=8, 2*MOE_INTERMEDIATE/tp)
  N = 2*2048/tp = 4096/2048/1024/512 for tp = 1/2/4/8.

TP is a SHAPE selector only (N shards by routed_tp): world_size=1, per-rank N
passed directly (no NVSHMEM). The OLD-MoE kernel `fp8_moe_group_gemm_sm100`
reads mMask(num_local_experts) directly and strides the activated-expert list
by grid_dim.x -- it has NO 128-expert scan cap (unlike the decode largem
kernel), so num_local_experts is a free knob.

EL reduction (LOGGED): real ep=1 DSV3 has EL=256, giving a
(256, 4096, 7168) FP8 weight (~7.5 GB) of which only the ~8-64 round-robin-
activated experts are ever read. The kernel correctness is per-expert
independent, so EL is reduced to EL=64 for the primary sweep (matches the
prior test + the sm100_fp8_group_gemm_decode unit) with real per-expert N/K
and a realistic number of activated experts (8 at bs=1 .. 64 at bs>=8). The
secondary ep=2 check uses num_local_experts = 256/2 = 128 (production-faithful
for ep>=2).

Grid mirrors the builder:
  grid = (_moe_expert_grid_x(bs, EL, preferred_groups=8) = min(8, bs*8),
          _moe_fp8_m_split(N, preferred=16), 1), block = (256,1,1)
(the kernel hard-requires 8 warps = 256 threads).

Run:
    python tests/runtime_python/blackwell/sm100_fp8_moe/test_moe_w13_fp8_testmode.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from pytorch_reference import (  # noqa: E402
    moe_w13_fp8_ref,
    quantize_fp8_2d,
    quantize_fp8_3d,
    make_routing,
    cosine_sim,
    rel_mean,
)

HIDDEN_SIZE = 7168          # K
MOE_INTERMEDIATE = 2048     # per-expert routed intermediate (TP=1)
NUM_TOPK = 8
NUM_EXPERTS = 256           # global expert count (for ep sizing)
EL_REDUCED = 64             # reduced local-expert count for the primary sweep
_MMA_M = 128


def _w13_n(tp: int) -> int:
    # gate||up routed output dim, sharded by routed_tp.
    n = 2 * (MOE_INTERMEDIATE // tp)
    assert n % 128 == 0, (tp, n)
    return n


def _moe_fp8_m_split(output_size: int, preferred: int = 16) -> int:
    """Mirror builder._moe_fp8_m_split: per-CTA N-slice multiple of MMA_M=128."""
    max_y = min(preferred, max(1, output_size // _MMA_M))
    for y in range(max_y, 0, -1):
        if output_size % y == 0 and (output_size // y) % _MMA_M == 0:
            return y
    return 1


def _moe_expert_grid_x(bs: int, num_local_experts: int,
                       preferred_groups: int = 8) -> int:
    active_slots = max(1, bs * NUM_TOPK)
    return min(min(num_local_experts, preferred_groups), active_slots)


def _run_case(tp, bs, EL, seed=42):
    """One W13 config. Returns (passed, cos, rel, tag)."""
    N = _w13_n(tp)
    m_split = _moe_fp8_m_split(N, 16)
    grid_x = _moe_expert_grid_x(bs, EL, preferred_groups=8)
    tag = (f"[W13] tp={tp} bs={bs} EL={EL} K={HIDDEN_SIZE} N={N} "
           f"grid=({grid_x},{m_split},1)")
    print(f"\n{'='*80}\n{tag}\n{'='*80}", flush=True)

    device = "cuda"
    torch.manual_seed(seed)

    input_val = torch.randn(bs, HIDDEN_SIZE, device=device) * 0.1
    weight_val = torch.randn(EL, N, HIDDEN_SIZE, device=device) \
        / (HIDDEN_SIZE ** 0.5)

    input_fp8, input_scale = quantize_fp8_2d(input_val)
    weight_fp8, weight_scale = quantize_fp8_3d(weight_val)
    routing, mask, token_to_experts = make_routing(bs, EL, NUM_TOPK, device)
    n_active = int(mask[EL].item())

    output = torch.zeros(bs, NUM_TOPK, N, dtype=torch.bfloat16, device=device)

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

    i_fp8 = pk.attach_input(input_fp8, name="input_fp8")
    i_sc = pk.attach_input(input_scale, name="input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="weight_scale")
    rt = pk.attach_input(routing, name="routing_indices")
    mk = pk.attach_input(mask, name="mask")
    out = pk.attach_input(output, name="output")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.moe_w13_fp8_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        moe_routing_indices=rt, moe_mask=mk, output=out,
        grid_dim=(grid_x, m_split, 1), block_dim=block_dim,
    )

    compile_dir = os.path.join(THIS_DIR, f".pk_w13_tp{tp}_bs{bs}_el{EL}")
    os.makedirs(compile_dir, exist_ok=True)
    pk.compile(output_dir=compile_dir)
    pk()
    torch.cuda.synchronize()

    ref = moe_w13_fp8_ref(input_fp8, input_scale, weight_fp8, weight_scale,
                          bs, token_to_experts, use_ue8m0=True)

    cos = cosine_sim(output, ref)
    rel = rel_mean(output, ref)
    max_abs = (output.float() - ref.float()).abs().max().item()
    # fp8 MoE tolerance (decision log): cosine > 0.99 OR rel <= 5%.
    passed = (cos > 0.99 or rel <= 0.05)
    print(f"  active_experts={n_active} cos={cos:.6f} rel={rel*100:.4f}% "
          f"max_abs_diff={max_abs:.4f} -> {'PASS' if passed else 'FAIL'}",
          flush=True)

    pk.finalize()
    return passed, cos, rel, tag


def main():
    results = []

    # ── Union-of-axes (tp, bs) matrix ──
    # {tp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{tp=2,4,8} ∪ {tp=8,bs=1}
    for bs in (1, 2, 4, 8, 16):
        results.append(_run_case(tp=1, bs=bs, EL=EL_REDUCED))
    for tp in (2, 4, 8):
        results.append(_run_case(tp=tp, bs=16, EL=EL_REDUCED))
    results.append(_run_case(tp=8, bs=1, EL=EL_REDUCED))

    # ── Secondary ep>1 check: num_local_experts = 256/ep ──
    # ep=2 -> EL=128 (production-faithful for ep>=2). tp=1 (routed_tp=world/ep).
    results.append(_run_case(tp=1, bs=16, EL=NUM_EXPERTS // 2))

    return _summary(results)


def _summary(results):
    print(f"\n{'='*80}\nSummary (moe_w13_fp8):\n{'='*80}", flush=True)
    all_passed = True
    for passed, cos, rel, tag in results:
        print(f"  {'PASS' if passed else 'FAIL'}  cos={cos:.5f} "
              f"rel={rel*100:.4f}%  {tag}", flush=True)
        all_passed = all_passed and passed
    n_pass = sum(1 for r in results if r[0])
    print(f"\n{'ALL PASS' if all_passed else 'SOME FAILED'} "
          f"({n_pass}/{len(results)})", flush=True)
    return 0 if all_passed else 1


def test_moe_w13_fp8_testmode():
    rc = main()
    assert rc == 0, "some moe_w13_fp8 configs failed"


if __name__ == "__main__":
    sys.exit(main())
