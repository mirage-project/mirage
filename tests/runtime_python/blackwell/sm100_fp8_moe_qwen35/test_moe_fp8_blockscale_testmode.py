"""Test mode: fp32-scale quantize -> blockscale w13 -> silu_mul -> quantize ->
blockscale w2, all inside one megakernel.

Exercises everything the standalone kernel tests cannot: the Python layer API
(quantize_fp8_layer(scale_ue8m0=False) + moe_fp8_blockscale_layer), task
registration for BOTH new task types (241 w13 / 242 w2), C++ code generation,
the in-window tma.cuh case, `expert_offset` metadata delivery, nvcc
compilation, and runtime dispatch -- with a 2D grid so expert distribution
(grid.x) and N-splitting (grid.y) are both live.

Shapes are deliberately small (E=8, topk=2, N=256, K=256): task registration is
shape-generic, and a 256-expert megakernel compile would dominate the runtime
of a codegen test.

Run:
    python tests/runtime_python/blackwell/sm100_fp8_moe_qwen35/\
test_moe_fp8_blockscale_testmode.py
"""

import json
import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

torch.backends.cuda.matmul.allow_tf32 = False

_MEGAKERNEL_SCALE_FORMS = []

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10

BATCH = 4
NUM_EXPERTS = 8
NUM_TOPK = 2
HIDDEN = 256          # K for w13, N for w2
INTERMEDIATE = 128    # K for w2
W13_N = 2 * INTERMEDIATE
EXPERT_STRIDE = 2
N_SPLITS = 2


def _true_div(t, denom):
    return torch.div(t, torch.tensor(denom, dtype=torch.float32, device=t.device))


def quantize_activation(x_bf16):
    shape = x_bf16.shape
    k = shape[-1]
    xf = x_bf16.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = _true_div(absmax, FP8_MAX)
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return (
        q.reshape(shape).to(torch.float8_e4m3fn).contiguous(),
        scale.reshape(*shape[:-1], k // BLOCK).float().contiguous(),
    )


def quantize_weight_blocks(w_bf16):
    n, k = w_bf16.shape
    wf = w_bf16.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3)).clamp(min=EPS)
    scale = _true_div(absmax, FP8_MAX)
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(n, k).to(torch.float8_e4m3fn), scale.contiguous()


def dequant_groups(q, scale):
    shape = q.shape
    k = shape[-1]
    return (
        q.float().reshape(-1, k // BLOCK, BLOCK) * scale.reshape(-1, k // BLOCK, 1)
    ).reshape(shape)


def dequant_blocks(q, block_scale):
    n, k = q.shape
    wf = q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    return (wf * block_scale[:, None, :, None]).reshape(n, k)


def classify_scale(label, got_scale, x_bf16):
    """The megakernel compiles with `-use_fast_math` (see the nvcc line MPK
    prints), which rewrites the quantizer's `group_max / 448.0f` into a
    reciprocal multiply. The standalone extension, built without fast-math,
    divides -- and the two differ by exactly 1 fp32 ULP for divisors like 448
    whose reciprocal is inexact (measured in
    test_quantize_fp8_f32scale_moe.py).

    Assert what is actually contractual: the scale must be ONE of the two
    forms of absmax/448, and never more than 1 fp32 ULP from true division.
    1 fp32 ULP is ~2e-7 relative, four orders of magnitude below one e4m3 LSB
    and below p10b's measured 1-e4m3-ULP disagreement between two independent
    implementations of this same primitive.
    """
    k = x_bf16.shape[-1]
    xf = x_bf16.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    ref_div = _true_div(absmax, FP8_MAX).reshape(got_scale.shape)
    ref_recip = (absmax * (1.0 / FP8_MAX)).reshape(got_scale.shape)
    if torch.equal(got_scale, ref_div):
        form = "true_division"
    elif torch.equal(got_scale, ref_recip):
        form = "reciprocal_multiply (-use_fast_math)"
    else:
        form = "NEITHER"
    ulp = (
        (got_scale.view(torch.int32) - ref_div.contiguous().view(torch.int32))
        .abs()
        .max()
        .item()
    )
    print(f"  {label} scale form: {form}; max fp32 ULP from true division: {ulp}")
    assert form != "NEITHER", (
        f"{label} scale matches neither absmax/448 form -- max |delta| "
        f"{(got_scale - ref_div).abs().max().item():.3e}"
    )
    assert ulp <= 1, f"{label} scale is {ulp} fp32 ULP from absmax/448"
    _MEGAKERNEL_SCALE_FORMS.append(
        {"tensor": label, "form": form, "max_fp32_ulp_from_true_division": ulp}
    )


def main():
    device = "cuda"
    torch.manual_seed(20260726)

    # --- routing: token t takes experts (t, t+1) mod NUM_EXPERTS ---
    routing = torch.zeros(NUM_EXPERTS, BATCH, dtype=torch.int32, device=device)
    token_experts = {}
    for t in range(BATCH):
        picks = [(t * 2) % NUM_EXPERTS, (t * 2 + 1) % NUM_EXPERTS]
        token_experts[t] = picks
        for slot, e in enumerate(picks):
            routing[e, t] = slot + 1
    activated = sorted({e for v in token_experts.values() for e in v})
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    for i, e in enumerate(activated):
        mask[i] = e
    mask[NUM_EXPERTS] = len(activated)

    x = torch.randn(BATCH, HIDDEN, dtype=torch.bfloat16, device=device)
    w13 = torch.zeros(NUM_EXPERTS, W13_N, HIDDEN, dtype=torch.float8_e4m3fn,
                      device=device)
    w13_s = torch.zeros(NUM_EXPERTS, W13_N // BLOCK, HIDDEN // BLOCK,
                        dtype=torch.float32, device=device)
    w2 = torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE,
                     dtype=torch.float8_e4m3fn, device=device)
    w2_s = torch.zeros(NUM_EXPERTS, HIDDEN // BLOCK, INTERMEDIATE // BLOCK,
                       dtype=torch.float32, device=device)
    for e in activated:
        q, s = quantize_weight_blocks(
            torch.randn(W13_N, HIDDEN, dtype=torch.bfloat16, device=device)
        )
        w13[e], w13_s[e] = q, s
        q, s = quantize_weight_blocks(
            torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16,
                        device=device)
        )
        w2[e], w2_s[e] = q, s

    # Runtime buffers the megakernel fills.
    x_q = torch.zeros(BATCH, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    x_s = torch.zeros(BATCH, HIDDEN // BLOCK, dtype=torch.float32, device=device)
    mid = torch.zeros(BATCH, NUM_TOPK, W13_N, dtype=torch.bfloat16, device=device)
    act = torch.zeros(BATCH, NUM_TOPK, INTERMEDIATE, dtype=torch.bfloat16,
                      device=device)
    act_q = torch.zeros(BATCH, NUM_TOPK, INTERMEDIATE,
                        dtype=torch.float8_e4m3fn, device=device)
    act_s = torch.zeros(BATCH, NUM_TOPK, INTERMEDIATE // BLOCK,
                        dtype=torch.float32, device=device)
    out = torch.zeros(BATCH, NUM_TOPK, HIDDEN, dtype=torch.bfloat16,
                      device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=BATCH,
        max_num_batched_requests=BATCH,
    )
    pk = PersistentKernel(**params)
    assert pk.target_cc >= 100, "moe_fp8_blockscale_sm100 requires Blackwell"

    x_dt = pk.attach_input(x, name="x")
    x_q_dt = pk.attach_input(x_q, name="x_q")
    x_s_dt = pk.attach_input(x_s, name="x_s")
    w13_dt = pk.attach_input(w13, name="w13")
    w13_s_dt = pk.attach_input(w13_s, name="w13_scale")
    w2_dt = pk.attach_input(w2, name="w2")
    w2_s_dt = pk.attach_input(w2_s, name="w2_scale")
    routing_dt = pk.attach_input(routing, name="routing")
    mask_dt = pk.attach_input(mask, name="mask")
    mid_dt = pk.attach_input(mid, name="mid")
    act_dt = pk.attach_input(act, name="act")
    act_q_dt = pk.attach_input(act_q, name="act_q")
    act_s_dt = pk.attach_input(act_s, name="act_s")
    out_dt = pk.attach_input(out, name="out")

    pk.quantize_fp8_layer(
        input=x_dt, output_fp8=x_q_dt, output_scale=x_s_dt,
        grid_dim=(BATCH, 1, 1), block_dim=(128, 1, 1), scale_ue8m0=False,
    )
    pk.moe_fp8_blockscale_layer(
        input_fp8=x_q_dt, input_scale=x_s_dt,
        weight_fp8=w13_dt, weight_scale=w13_s_dt,
        moe_routing_indices=routing_dt, moe_mask=mask_dt, output=mid_dt,
        grid_dim=(EXPERT_STRIDE, N_SPLITS, 1), block_dim=(256, 1, 1),
        w13_linear=True,
    )
    pk.moe_silu_mul_layer(
        input=mid_dt, output=act_dt,
        grid_dim=(BATCH, NUM_TOPK, 1), block_dim=(128, 1, 1),
    )
    pk.quantize_fp8_layer(
        input=act_dt, output_fp8=act_q_dt, output_scale=act_s_dt,
        grid_dim=(BATCH, 1, 1), block_dim=(128, 1, 1), scale_ue8m0=False,
    )
    pk.moe_fp8_blockscale_layer(
        input_fp8=act_q_dt, input_scale=act_s_dt,
        weight_fp8=w2_dt, weight_scale=w2_s_dt,
        moe_routing_indices=routing_dt, moe_mask=mask_dt, output=out_dt,
        grid_dim=(EXPERT_STRIDE, N_SPLITS, 1), block_dim=(256, 1, 1),
        w13_linear=False,
    )

    pk.compile(output_dir="./test_output_moe_fp8_blockscale")
    pk()
    torch.cuda.synchronize()

    # --- PyTorch reference on the SAME quantized bytes the kernel produced ---
    ref_x_q, ref_x_s = quantize_activation(x)
    assert torch.equal(x_q.view(torch.uint8), ref_x_q.view(torch.uint8)), (
        "the fp32-scale quantize task disagrees with the reference primitive"
    )
    classify_scale("activation", x_s, x)

    x_deq = dequant_groups(x_q, x_s)
    ref_mid = torch.zeros_like(mid, dtype=torch.float32)
    for t, experts in token_experts.items():
        for slot, e in enumerate(experts):
            ref_mid[t, slot] = (
                x_deq[t: t + 1] @ dequant_blocks(w13[e], w13_s[e]).t()
            ).squeeze(0)

    ref_act = (
        torch.nn.functional.silu(ref_mid[..., :INTERMEDIATE])
        * ref_mid[..., INTERMEDIATE:]
    )
    classify_scale("w2 activation", act_s, act)
    act_deq = dequant_groups(act_q, act_s)
    ref_out = torch.zeros_like(out, dtype=torch.float32)
    for t, experts in token_experts.items():
        for slot, e in enumerate(experts):
            ref_out[t, slot] = (
                act_deq[t, slot: slot + 1] @ dequant_blocks(w2[e], w2_s[e]).t()
            ).squeeze(0)

    for name, got, ref in (("w13", mid, ref_mid), ("w2", out, ref_out)):
        g = got.float()
        err = (g - ref).norm().item() / ref.norm().item()
        floor = (
            (ref.to(torch.bfloat16).float() - ref).norm().item()
            / ref.norm().item()
        )
        print(
            f"  {name}: frob_rel={err:.3e} bf16_output_floor={floor:.3e} "
            f"ratio={err / floor:.2f} "
            f"max_abs_diff={(g - ref).abs().max().item():.3e}"
        )
        assert err <= 1.6 * floor, (
            f"{name}: frob_rel {err:.3e} exceeds 1.6x the bf16 output-rounding "
            f"floor {floor:.3e}"
        )

    # The chained activation must also have gone through the megakernel's own
    # silu_mul, not just matched at the endpoints.
    torch.testing.assert_close(
        act.float(), ref_act.to(torch.bfloat16).float(), rtol=2e-2, atol=2e-2
    )

    out_path = os.environ.get("P2_ACTIVATION_JSON")
    if out_path:
        prior = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                prior = json.load(f)
        prior["megakernel_build"] = {
            "note": "MPK compiles the megakernel with -use_fast_math, which "
            "rewrites the quantizer's group_max/448.0f into a reciprocal "
            "multiply. That is 1 fp32 ULP (~2e-7 relative) from true division "
            "-- four orders of magnitude below one e4m3 LSB and below p10b's "
            "measured 1-e4m3-ULP disagreement between two independent "
            "implementations of this primitive.",
            "scale_forms": _MEGAKERNEL_SCALE_FORMS,
            "both_blockscale_tasks_ran_in_megakernel": True,
        }
        with open(out_path, "w") as f:
            json.dump(prior, f, indent=1)
        print(f"WROTE {out_path}")

    pk.finalize()
    print("MOE_FP8_BLOCKSCALE TEST-MODE PIPELINE PASSED")


if __name__ == "__main__":
    main()
