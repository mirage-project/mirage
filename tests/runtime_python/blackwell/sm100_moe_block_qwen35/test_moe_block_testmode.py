"""Test mode: the WHOLE Qwen3.5 MoE block, router through combine, in one
megakernel, at production shapes (256 experts / hidden 2048 / moe_intermediate
512) with 2 requests.

    router GEMM -> topk_softmax -> quantize -> w13(blockscale) -> silu_mul ->
    quantize -> w2(blockscale) -> mul_sum_add
                        |
    shared gate_up -> silu_mul -> down -> sigmoid_gate_mul_add --(residual)--^

What only this test can prove (the standalone kernel tests cannot): the Python
layer API for the new `sigmoid_gate_mul_add_layer`, its task registration,
codegen, the in-window tma.cuh case, the `round_weights_to_input_dtype` router
parameter, and that a 40-way-wide MoE block with grid.y-split grouped GEMMs and
a concurrent shared-expert branch actually schedules and terminates at
`moe_intermediate = 512` -- the regime mpk-gaps.md Gap 7 flagged as never run.

Numerics are checked at EVERY boundary the block exposes, each against a torch
reference computed on the bytes the megakernel itself produced, so a wiring
error cannot hide behind a loose end-to-end tolerance.

Run:  python tests/runtime_python/blackwell/sm100_moe_block_qwen35/test_moe_block_testmode.py
"""

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.utils import grid_for_rmsnorm_linear_layer

torch.backends.cuda.matmul.allow_tf32 = False

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10

BATCH = 2
NUM_EXPERTS = 256
NUM_TOPK = 8
HIDDEN = 2048
INTER = 512          # moe_intermediate_size -- the untested-small regime
W13_N = 2 * INTER    # 1024
EXPERT_STRIDE = 16   # CTAs sharing the expert loop; >= BATCH * NUM_TOPK covers all
N_SPLITS = 2


def _true_div(t, d):
    return torch.div(t, torch.tensor(d, dtype=torch.float32, device=t.device))


def quantize_activation(x):
    shape = x.shape
    k = shape[-1]
    xf = x.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = _true_div(absmax, FP8_MAX)
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return (
        q.reshape(shape).to(torch.float8_e4m3fn).contiguous(),
        scale.reshape(*shape[:-1], k // BLOCK).float().contiguous(),
    )


def quantize_weight_blocks(w):
    n, k = w.shape
    wf = w.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
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


def dequant_blocks(q, s):
    n, k = q.shape
    return (
        q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        * s[:, None, :, None]
    ).reshape(n, k)


def report(name, got, ref, limit):
    g, r = got.float(), ref.float()
    err = (g - r).norm().item() / r.norm().item()
    floor = (r.to(torch.bfloat16).float() - r).norm().item() / r.norm().item()
    # a reference that is already bf16 has no rounding floor of its own; printing
    # a ratio against zero would be noise, so say so instead
    ratio = f"{err / floor:.2f}" if floor > 0 else "n/a (bf16 ref)"
    print(
        f"  {name:22s} frob_rel={err:.3e}  bf16_floor={floor:.3e}  "
        f"ratio={ratio}  max_abs={(g - r).abs().max().item():.3e}"
    )
    assert err <= limit, f"{name}: frob_rel {err:.3e} > {limit:.3e}"
    return err, floor


def main():
    dev = "cuda"
    torch.manual_seed(20260726)

    # ---------------- weights ----------------
    x = torch.randn(BATCH, HIDDEN, dtype=torch.bfloat16, device=dev)
    residual = torch.randn(BATCH, HIDDEN, dtype=torch.bfloat16, device=dev)
    w_router = (torch.randn(NUM_EXPERTS, HIDDEN, device=dev) * 0.02).to(torch.bfloat16)
    w_sg = (torch.randn(1, HIDDEN, device=dev) * 0.02).to(torch.bfloat16)
    # Shared expert on the M2 ACCEPTANCE dense path: fp8 with the checkpoint's
    # preserved fp32 block scales (M2-I12, amended docs/qwen35 6.2) -- NOT the
    # bf16-dequant scaffold the pre-amendment 6.1 table lists.
    w_gate_q, w_gate_s = quantize_weight_blocks(
        (torch.randn(INTER, HIDDEN, device=dev) * 0.02).to(torch.bfloat16))
    w_up_q, w_up_s = quantize_weight_blocks(
        (torch.randn(INTER, HIDDEN, device=dev) * 0.02).to(torch.bfloat16))
    w_down_q, w_down_s = quantize_weight_blocks(
        (torch.randn(HIDDEN, INTER, device=dev) * 0.02).to(torch.bfloat16))
    # silu_mul reads [gate_chunk | up_chunk] pairs, so gate/up are interleaved at
    # a granularity that divides BOTH the weight rows and the scale rows: the
    # scale carries one row per 128 weight rows, so the split is bounded by
    # INTER // 128 (DeepSeek-V3's builder does the same clamp).
    SHARED_SPLIT = INTER // BLOCK  # 4

    # routed experts: only the ones the router can pick need real values, but
    # allocating the full [256,...] tensors is the point -- production shapes.
    w13 = torch.zeros(NUM_EXPERTS, W13_N, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
    w13_s = torch.zeros(NUM_EXPERTS, W13_N // BLOCK, HIDDEN // BLOCK,
                        dtype=torch.float32, device=dev)
    w2 = torch.zeros(NUM_EXPERTS, HIDDEN, INTER, dtype=torch.float8_e4m3fn, device=dev)
    w2_s = torch.zeros(NUM_EXPERTS, HIDDEN // BLOCK, INTER // BLOCK,
                       dtype=torch.float32, device=dev)

    # which experts the router will pick, computed exactly as the kernel does
    logits_ref = (x.float() @ w_router.float().t()).to(torch.bfloat16)
    probs_ref = torch.softmax(logits_ref.float(), dim=-1)
    order = torch.argsort(probs_ref, dim=-1, descending=True, stable=True)
    ids_ref = order[:, :NUM_TOPK]
    for e in sorted(set(ids_ref.flatten().tolist())):
        q, s = quantize_weight_blocks(
            (torch.randn(W13_N, HIDDEN, device=dev) * 0.05).to(torch.bfloat16))
        w13[e], w13_s[e] = q, s
        q, s = quantize_weight_blocks(
            (torch.randn(HIDDEN, INTER, device=dev) * 0.05).to(torch.bfloat16))
        w2[e], w2_s[e] = q, s

    # ---------------- runtime buffers ----------------
    z = lambda *a, **k: torch.zeros(*a, **k, device=dev)  # noqa: E731
    logits = z(BATCH, NUM_EXPERTS, dtype=torch.bfloat16)
    topk_w = z(BATCH, NUM_TOPK, dtype=torch.float32)
    routing = z(NUM_EXPERTS, BATCH, dtype=torch.int32)
    mask = z(NUM_EXPERTS + 1, dtype=torch.int32)
    x_q = z(BATCH, HIDDEN, dtype=torch.float8_e4m3fn)
    x_s = z(BATCH, HIDDEN // BLOCK, dtype=torch.float32)
    # The shared branch needs its OWN copy: MPK's annotated-graph builder rejects
    # a task that is both a fork-producer and a join-producer ("a task cannot
    # have two dependent_events"), so one quantize task cannot feed both the
    # routed w13 and the shared gate_up. One extra task, same bytes.
    xs_q = z(BATCH, HIDDEN, dtype=torch.float8_e4m3fn)
    xs_s = z(BATCH, HIDDEN // BLOCK, dtype=torch.float32)
    mid = z(BATCH, NUM_TOPK, W13_N, dtype=torch.bfloat16)
    act = z(BATCH, NUM_TOPK, INTER, dtype=torch.bfloat16)
    act_q = z(BATCH, NUM_TOPK, INTER, dtype=torch.float8_e4m3fn)
    act_s = z(BATCH, NUM_TOPK, INTER // BLOCK, dtype=torch.float32)
    down = z(BATCH, NUM_TOPK, HIDDEN, dtype=torch.bfloat16)
    shared_mid = z(BATCH, W13_N, dtype=torch.bfloat16)
    shared_act = z(BATCH, INTER, dtype=torch.bfloat16)
    sact_q = z(BATCH, INTER, dtype=torch.float8_e4m3fn)
    sact_s = z(BATCH, INTER // BLOCK, dtype=torch.float32)
    shared_out = z(BATCH, HIDDEN, dtype=torch.bfloat16)
    r_prime = z(BATCH, HIDDEN, dtype=torch.bfloat16)
    out = z(BATCH, HIDDEN, dtype=torch.bfloat16)

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
        meta_tensors={
            "tokens": torch.zeros((BATCH, 1), dtype=torch.int64, device=dev),
            "prompt_lengths": torch.ones(BATCH, dtype=torch.int32, device=dev),
        },
    )
    pk = PersistentKernel(**params)
    assert pk.target_cc >= 100, "the Qwen3.5 MoE block is Blackwell-only"
    bd = (256, 1, 1)

    at = pk.attach_input
    x_dt, resid_dt = at(x, name="x"), at(residual, name="residual")
    wr_dt, wsg_dt = at(w_router, name="w_router"), at(w_sg, name="w_sg")
    wg_dt, wu_dt = at(w_gate_q, name="w_gate"), at(w_up_q, name="w_up")
    wgs_dt, wus_dt = at(w_gate_s, name="w_gate_s"), at(w_up_s, name="w_up_s")
    wd_dt, wds_dt = at(w_down_q, name="w_down"), at(w_down_s, name="w_down_s")
    w13_dt, w13s_dt = at(w13, name="w13"), at(w13_s, name="w13_scale")
    w2_dt, w2s_dt = at(w2, name="w2"), at(w2_s, name="w2_scale")
    logits_dt, topkw_dt = at(logits, name="logits"), at(topk_w, name="topk_w")
    routing_dt, mask_dt = at(routing, name="routing"), at(mask, name="mask")
    xq_dt, xs_dt = at(x_q, name="x_q"), at(x_s, name="x_s")
    xsq_dt, xss_dt = at(xs_q, name="xs_q"), at(xs_s, name="xs_s")
    mid_dt, act_dt = at(mid, name="mid"), at(act, name="act")
    actq_dt, acts_dt = at(act_q, name="act_q"), at(act_s, name="act_s")
    down_dt = at(down, name="down")
    smid_dt, sact_dt = at(shared_mid, name="shared_mid"), at(shared_act, name="shared_act")
    sactq_dt, sacts_dt = at(sact_q, name="sact_q"), at(sact_s, name="sact_s")
    sout_dt, rp_dt = at(shared_out, name="shared_out"), at(r_prime, name="r_prime")
    out_dt = at(out, name="out")

    # 1. router GEMM (bf16, never quantized -- vllm-graph.md 2.3.1)
    router_grid = min(grid_for_rmsnorm_linear_layer(NUM_EXPERTS), NUM_EXPERTS // 8)
    pk.linear_layer(input=x_dt, weight=wr_dt, output=logits_dt,
                    grid_dim=(router_grid, 1, 1), block_dim=bd)
    # 2. routing. round_weights reproduces HF's bf16 topk_renorm_weights (P5 E).
    pk.moe_topk_softmax_routing_layer(
        input=logits_dt, output=(topkw_dt, routing_dt, mask_dt),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
        round_weights_to_input_dtype=True)
    # 3-4. activation quant -> w13
    pk.quantize_fp8_layer(input=x_dt, output_fp8=xq_dt, output_scale=xs_dt,
                          grid_dim=(BATCH, 1, 1), block_dim=(128, 1, 1),
                          scale_ue8m0=False)
    pk.moe_fp8_blockscale_layer(
        input_fp8=xq_dt, input_scale=xs_dt, weight_fp8=w13_dt,
        weight_scale=w13s_dt, moe_routing_indices=routing_dt, moe_mask=mask_dt,
        output=mid_dt, grid_dim=(EXPERT_STRIDE, N_SPLITS, 1), block_dim=bd,
        w13_linear=True)
    # 5-7. SwiGLU -> quant -> w2
    pk.moe_silu_mul_layer(input=mid_dt, output=act_dt,
                          grid_dim=(BATCH, NUM_TOPK, 1), block_dim=(128, 1, 1))
    pk.quantize_fp8_layer(input=act_dt, output_fp8=actq_dt, output_scale=acts_dt,
                          grid_dim=(BATCH, 1, 1), block_dim=(128, 1, 1),
                          scale_ue8m0=False)
    pk.moe_fp8_blockscale_layer(
        input_fp8=actq_dt, input_scale=acts_dt, weight_fp8=w2_dt,
        weight_scale=w2s_dt, moe_routing_indices=routing_dt, moe_mask=mask_dt,
        output=down_dt, grid_dim=(EXPERT_STRIDE, N_SPLITS, 1), block_dim=bd,
        w13_linear=False)
    # 8-10. shared expert, fp8 with preserved fp32 block scales (M2-I12).
    # grid.x splits the weight's output rows AND the scale's dim0 (one row per
    # 128 weight rows), so grid.x must divide N // 128.
    wgu_dt = pk.shuffle_tensors(inputs=[wg_dt, wu_dt], shuffled_dim=0,
                                num_groups=SHARED_SPLIT, name="w_gate_up")
    wgus_dt = pk.shuffle_tensors(inputs=[wgs_dt, wus_dt], shuffled_dim=0,
                                 num_groups=SHARED_SPLIT, name="w_gate_up_s")
    pk.quantize_fp8_layer(input=x_dt, output_fp8=xsq_dt, output_scale=xss_dt,
                          grid_dim=(BATCH, 1, 1), block_dim=(128, 1, 1),
                          scale_ue8m0=False)
    pk.linear_fp8_blockscale_layer(
        input_fp8=xsq_dt, input_scale=xss_dt, weight_fp8=wgu_dt,
        weight_scale=wgus_dt, output=smid_dt,
        grid_dim=(W13_N // BLOCK, 1, 1), block_dim=bd)
    pk.silu_mul_layer(input=smid_dt, output=sact_dt,
                      grid_dim=(SHARED_SPLIT, 1, 1), block_dim=(128, 1, 1))
    pk.quantize_fp8_layer(input=sact_dt, output_fp8=sactq_dt,
                          output_scale=sacts_dt, grid_dim=(BATCH, 1, 1),
                          block_dim=(128, 1, 1), scale_ue8m0=False)
    pk.linear_fp8_blockscale_layer(
        input_fp8=sactq_dt, input_scale=sacts_dt, weight_fp8=wd_dt,
        weight_scale=wds_dt, output=sout_dt,
        grid_dim=(HIDDEN // BLOCK, 1, 1), block_dim=bd)
    # 11. the new task: r' = residual + sigmoid(x . w_sg) * shared
    pk.sigmoid_gate_mul_add_layer(input=x_dt, gate_weight=wsg_dt, shared=sout_dt,
                                  residual=resid_dt, output=rp_dt,
                                  grid_dim=(BATCH, 1, 1), block_dim=bd)
    # 12. combine
    pk.moe_mul_sum_add_layer(input=down_dt, weight=topkw_dt, residual=rp_dt,
                             output=out_dt, grid_dim=(BATCH, 1, 1),
                             block_dim=(128, 1, 1))

    pk.compile(output_dir="./test_output_moe_block")
    pk()
    torch.cuda.synchronize()

    # ================= per-boundary checks =================
    print(f"active experts: {int(mask[NUM_EXPERTS].item())} "
          f"(expected {len(set(ids_ref.flatten().tolist()))})")
    assert int(mask[NUM_EXPERTS].item()) == len(set(ids_ref.flatten().tolist()))

    # router: ids from the routing table, weights against HF's bf16 semantics
    ids = torch.full((BATCH, NUM_TOPK), -1, dtype=torch.int64, device=dev)
    nz = routing.nonzero()
    ids[nz[:, 1], routing[nz[:, 0], nz[:, 1]].long() - 1] = nz[:, 0]
    for b in range(BATCH):
        assert set(ids[b].tolist()) == set(ids_ref[b].tolist()), (
            f"row {b}: {sorted(ids[b].tolist())} != {sorted(ids_ref[b].tolist())}")
    w_ref = torch.gather(probs_ref, 1, ids)
    w_ref = (w_ref / w_ref.sum(dim=-1, keepdim=True)).to(torch.bfloat16).float()
    report("router weights", topk_w, w_ref, 1e-2)
    assert torch.equal(topk_w, topk_w.to(torch.bfloat16).float()), (
        "round_weights_to_input_dtype=True must leave bf16-exact weights")

    # w13 on the megakernel's own quantized activations
    x_deq = dequant_groups(x_q, x_s)
    ref_mid = torch.zeros(BATCH, NUM_TOPK, W13_N, dtype=torch.float32, device=dev)
    for b in range(BATCH):
        for slot in range(NUM_TOPK):
            e = int(ids[b, slot])
            ref_mid[b, slot] = x_deq[b: b + 1] @ dequant_blocks(w13[e], w13_s[e]).t()
    report("w13", mid, ref_mid, 4e-3)

    ref_act = torch.nn.functional.silu(ref_mid[..., :INTER]) * ref_mid[..., INTER:]
    report("silu_mul (routed)", act, ref_act, 6e-3)

    act_deq = dequant_groups(act_q, act_s)
    ref_down = torch.zeros(BATCH, NUM_TOPK, HIDDEN, dtype=torch.float32, device=dev)
    for b in range(BATCH):
        for slot in range(NUM_TOPK):
            e = int(ids[b, slot])
            ref_down[b, slot] = (
                act_deq[b, slot: slot + 1] @ dequant_blocks(w2[e], w2_s[e]).t())
    report("w2", down, ref_down, 4e-3)

    # shared expert branch, on the megakernel's own quantized activations
    assert torch.equal(xs_q.view(torch.uint8), x_q.view(torch.uint8))
    assert torch.equal(xs_s, x_s)
    xs_deq = dequant_groups(xs_q, xs_s)
    ref_smid_g = xs_deq @ dequant_blocks(w_gate_q, w_gate_s).t()
    ref_smid_u = xs_deq @ dequant_blocks(w_up_q, w_up_s).t()
    ref_sact = (torch.nn.functional.silu(ref_smid_g).to(torch.bfloat16).float()
                * ref_smid_u.to(torch.bfloat16).float())
    report("shared silu_mul", shared_act, ref_sact, 1e-2)
    sact_deq = dequant_groups(sact_q, sact_s)
    ref_sout = sact_deq @ dequant_blocks(w_down_q, w_down_s).t()
    report("shared down_proj", shared_out, ref_sout, 4e-3)

    logit = (x.float() @ w_sg.float().t()).to(torch.bfloat16)
    gate = torch.sigmoid(logit.float()).to(torch.bfloat16)
    ref_rp = (residual.float() + gate.float() * shared_out.float()).to(torch.bfloat16)
    assert torch.equal(r_prime.view(torch.int16), ref_rp.view(torch.int16)), (
        "sigmoid_gate_mul_add in the megakernel must be bit-identical to its "
        "declared cast positions")
    print("  sigmoid_gate_mul_add    BIT-EXACT vs the declared cast positions")

    ref_out = (r_prime.float() + (down.float() * topk_w.unsqueeze(-1)).sum(dim=1)
               ).to(torch.bfloat16)
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16)), (
        "combine must be bit-identical to fp32-accumulate + one rounding")
    print("  combine                 BIT-EXACT vs fp32 accumulate + one rounding")

    # end-to-end, against a reference built from the ORIGINAL inputs
    ref_block = (ref_rp.float()
                 + (ref_down * w_ref.unsqueeze(-1)).sum(dim=1)).to(torch.bfloat16)
    report("BLOCK end-to-end", out, ref_block, 6e-3)

    pk.finalize()
    print("MOE BLOCK TEST-MODE PIPELINE PASSED")


if __name__ == "__main__":
    main()
