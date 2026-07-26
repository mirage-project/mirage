"""The Qwen3.5 MoE block on REAL layer weights, checked at EVERY boundary the HF
oracle exposes -- decode (1 token) and a prefill chunk (8 tokens), MoE layers 0
and 3.

This is the numerics counterpart of `test_moe_block_testmode.py`: that one proves
the graph compiles, schedules and terminates; this one proves the numbers, on
checkpoint bytes, without paying a megakernel compile. Each stage runs through the
SAME kernels the megakernel dispatches, driven here through the two standalone
harnesses:

    router            sm100_moe_block_qwen35 : topk_softmax_sm100
    activation quant  sm100_fp8_moe_qwen35   : quantize_fp8_f32scale_sm100
    w13 / w2          sm100_fp8_moe_qwen35   : moe_w13/w2_blockscale_sm100  (241/242)
    shared gate       sm100_moe_block_qwen35 : sigmoid_gate_mul_add_sm100   (238)
    combine           sm100_moe_block_qwen35 : mul_sum_add_sm100

Boundaries and their oracle tensors:

    router logits/probs      moe*.router_logits, moe*.router_probs
    top-8 ids                moe*.topk_ids
    renormalized weights     moe*.topk_weights_raw, moe*.topk_renorm_weights
    routed-expert sum        moe*.routed_expert_output
    shared MLP output        moe*.shared_down_proj_out
    shared gate + multiply   moe*.shared_gate_logit/_sigmoid/_output_gated
    BLOCK output             moe*.combined_output

The shared expert is measured on BOTH dense paths: fp8 with the checkpoint's
preserved fp32 block scales (the M2 acceptance path after the 6.2 amendment,
M2-I12) and bf16-dequant (the pre-amendment 6.1 entry, now a scaffold whose
number is REPORTED, not gated).

Run:  QWEN35_SNAPSHOT=... python tests/runtime_python/blackwell/sm100_moe_block_qwen35/test_moe_block_oracle.py
"""

import json
import os
import sys

import torch
from safetensors import safe_open

import runtime_kernel_blackwell_moe_block_qwen35 as blk

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sm100_fp8_moe_qwen35"),
)
import runtime_kernel_blackwell_fp8_moe_qwen35 as moe  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False

BLOCK = 128
NUM_EXPERTS = 256
NUM_TOPK = 8
HARNESS_BATCH = 16  # the fp8 harness is instantiated at Q35_BATCH_SIZE = 16
HIDDEN = 2048
INTER = 512
W13_N, W13_K = 1024, 2048
W2_N, W2_K = 2048, 512
DEVICE = "cuda"

CKPT_REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
SNAPSHOT = os.environ.get(
    "QWEN35_SNAPSHOT",
    os.path.expanduser(
        "~/mpk-qwen35/hf/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/"
        + CKPT_REVISION
    ),
)
ORACLE = os.environ.get(
    "QWEN35_ORACLE_DUMPS", os.path.expanduser("~/mpk-qwen35/oracle-work/dumps")
)

RESULTS = []


FP8_MAX = 448.0
EPS = 1e-10


def quantize_activation(x):
    """The pinned activation-quant primitive (docs/qwen35 6.1): group 128,
    absmax clamped at 1e-10, scale = absmax/448 by true division, clamp then
    RN-even cast. M2-I13 showed `quantize_fp8_f32scale_sm100` reproduces it
    bit-exactly across 9 shapes, so using it as the reference for the shared
    branch keeps this test independent of that kernel's shape dispatch table."""
    shape = x.shape
    k = shape[-1]
    xf = x.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = torch.div(absmax, torch.tensor(FP8_MAX, dtype=torch.float32, device=x.device))
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return (q.reshape(shape).to(torch.float8_e4m3fn).contiguous(),
            scale.reshape(*shape[:-1], k // BLOCK).float().contiguous())


def dequant_groups(q, scale):
    shape = q.shape
    k = shape[-1]
    return (
        q.float().reshape(-1, k // BLOCK, BLOCK) * scale.reshape(-1, k // BLOCK, 1)
    ).reshape(shape)


def dequant_blocks(q, s):
    n, k = q.shape
    return (
        q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * s[:, None, :, None]
    ).reshape(n, k)


def load_oracle(mode, layer, key):
    man = json.load(open(os.path.join(ORACLE, mode, "manifest.json")))
    return torch.load(
        os.path.join(ORACLE, man["tensors"][f"{layer}.{key}"]["file"]),
        map_location=DEVICE,
    )


class Shards:
    def __init__(self):
        with open(os.path.join(SNAPSHOT, "model.safetensors.index.json")) as f:
            self.index = json.load(f)
        self.open = {}

    def get(self, key):
        path = os.path.join(SNAPSHOT, self.index["weight_map"][key])
        if path not in self.open:
            self.open[path] = safe_open(path, framework="pt")
        return self.open[path].get_tensor(key)


def load_experts(shards, layer_idx, expert_ids):
    p = f"model.language_model.layers.{layer_idx}.mlp.experts."
    w13 = torch.zeros((NUM_EXPERTS, W13_N, W13_K), dtype=torch.float8_e4m3fn)
    w13_s = torch.zeros((NUM_EXPERTS, W13_N // BLOCK, W13_K // BLOCK))
    w2 = torch.zeros((NUM_EXPERTS, W2_N, W2_K), dtype=torch.float8_e4m3fn)
    w2_s = torch.zeros((NUM_EXPERTS, W2_N // BLOCK, W2_K // BLOCK))
    for e in expert_ids:
        gate, up = shards.get(f"{p}{e}.gate_proj.weight"), shards.get(f"{p}{e}.up_proj.weight")
        # [gate; up] packed, NOT interleaved (vllm-graph.md 2.3.4)
        w13[e] = torch.cat([gate, up], dim=0)
        w13_s[e] = torch.cat(
            [shards.get(f"{p}{e}.gate_proj.weight_scale_inv"),
             shards.get(f"{p}{e}.up_proj.weight_scale_inv")], dim=0).float()
        w2[e] = shards.get(f"{p}{e}.down_proj.weight")
        w2_s[e] = shards.get(f"{p}{e}.down_proj.weight_scale_inv").float()
    return (w13.to(DEVICE), w13_s.to(DEVICE), w2.to(DEVICE), w2_s.to(DEVICE))


def build_routing(topk_ids):
    routing = torch.zeros((NUM_EXPERTS, HARNESS_BATCH), dtype=torch.int32)
    for t in range(topk_ids.shape[0]):
        for slot in range(NUM_TOPK):
            routing[int(topk_ids[t, slot]), t] = slot + 1
    activated = sorted({int(e) for e in topk_ids.flatten().tolist()})
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32)
    for i, e in enumerate(activated):
        mask[i] = e
    mask[NUM_EXPERTS] = len(activated)
    return routing.to(DEVICE), mask.to(DEVICE), activated


def report(tag, name, got, ref, limit):
    g, r = got.float(), ref.float()
    err = (g - r).norm().item() / r.norm().item()
    floor = (r.to(torch.bfloat16).float() - r).norm().item() / max(r.norm().item(), 1e-30)
    print(f"  {tag:16s} {name:24s} frob_rel={err:.3e}  bf16_floor={floor:.3e}  "
          f"max_abs={(g - r).abs().max().item():.3e}")
    RESULTS.append({"case": tag, "boundary": name, "frob_rel": err,
                    "bf16_floor": floor, "limit": limit})
    assert err <= limit, f"{tag}/{name}: frob_rel {err:.3e} > {limit:.3e}"
    return err


def run_case(shards, mode, layer, layer_idx):
    tag = f"{mode}/{layer}"
    x = load_oracle(mode, layer, "layer_input")
    tokens = x.shape[0]
    assert tokens <= HARNESS_BATCH

    # ---------- 1. router ----------
    hf_logits = load_oracle(mode, layer, "router_logits")
    hf_probs = load_oracle(mode, layer, "router_probs")
    hf_ids = load_oracle(mode, layer, "topk_ids")
    hf_bf16_w = load_oracle(mode, layer, "topk_renorm_weights")
    w_router = load_oracle(mode, layer, "__weight.router_gate_weight")

    logits = (x.float() @ w_router.float().t()).to(torch.bfloat16)
    report(tag, "router_logits", logits, hf_logits, 3e-3)
    probs = torch.softmax(logits.float(), dim=-1)
    report(tag, "router_probs", probs, hf_probs, 3e-3)

    g = hf_logits.clone().contiguous()
    topk_w = torch.zeros(tokens, NUM_TOPK, dtype=torch.float32, device=DEVICE)
    routing_r = torch.zeros(NUM_EXPERTS, tokens, dtype=torch.int32, device=DEVICE)
    mask_r = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    blk.topk_softmax_sm100(g, topk_w, routing_r, mask_r, 0, True)
    torch.cuda.synchronize()
    ids = torch.full((tokens, NUM_TOPK), -1, dtype=torch.int64, device=DEVICE)
    nz = routing_r.nonzero()
    ids[nz[:, 1], routing_r[nz[:, 0], nz[:, 1]].long() - 1] = nz[:, 0]
    for b in range(tokens):
        assert set(ids[b].tolist()) == set(hf_ids[b].tolist()), f"{tag} row {b}"
    pos = (ids.unsqueeze(2) == hf_ids.unsqueeze(1)).float().argmax(dim=1)
    w_hf_order = torch.gather(topk_w, 1, pos)
    assert torch.equal(
        w_hf_order.to(torch.bfloat16).view(torch.int16), hf_bf16_w.view(torch.int16)
    ), f"{tag}: renormalized weights are not HF's bf16 values bit-for-bit"
    print(f"  {tag:16s} {'topk ids + bf16 weights':24s} EXACT")

    # ---------- 2. routed experts ----------
    activated_ids = sorted({int(e) for e in hf_ids.flatten().tolist()})
    w13, w13_s, w2, w2_s = load_experts(shards, layer_idx, activated_ids)
    routing, mask, _ = build_routing(hf_ids)

    xp = torch.zeros(HARNESS_BATCH, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    xp[:tokens] = x
    x_q = torch.zeros(HARNESS_BATCH, HIDDEN, dtype=torch.float8_e4m3fn, device=DEVICE)
    x_s = torch.zeros(HARNESS_BATCH, HIDDEN // BLOCK, dtype=torch.float32, device=DEVICE)
    moe.quantize_fp8_f32scale_sm100(xp, x_q, x_s)

    mid = torch.zeros(HARNESS_BATCH, NUM_TOPK, W13_N, dtype=torch.bfloat16, device=DEVICE)
    moe.moe_w13_blockscale_sm100(x_q, x_s, w13, w13_s, routing, mask, mid)
    x_deq = dequant_groups(x_q, x_s)
    ref_mid = torch.zeros(tokens, NUM_TOPK, W13_N, dtype=torch.float32, device=DEVICE)
    for t in range(tokens):
        for slot in range(NUM_TOPK):
            e = int(hf_ids[t, slot])
            ref_mid[t, slot] = x_deq[t: t + 1] @ dequant_blocks(w13[e], w13_s[e]).t()
    report(tag, "w13 (vs fp32 of same bytes)", mid[:tokens], ref_mid, 4e-3)

    act = (torch.nn.functional.silu(mid[..., :INTER].float()).to(torch.bfloat16).float()
           * mid[..., INTER:].float()).to(torch.bfloat16).contiguous()
    act_q = torch.zeros(HARNESS_BATCH, NUM_TOPK, INTER, dtype=torch.float8_e4m3fn, device=DEVICE)
    act_s = torch.zeros(HARNESS_BATCH, NUM_TOPK, INTER // BLOCK, dtype=torch.float32, device=DEVICE)
    moe.quantize_fp8_f32scale_sm100(act, act_q, act_s)
    down = torch.zeros(HARNESS_BATCH, NUM_TOPK, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    moe.moe_w2_blockscale_sm100(act_q, act_s, w2, w2_s, routing, mask, down)
    act_deq = dequant_groups(act_q, act_s)
    ref_down = torch.zeros(tokens, NUM_TOPK, HIDDEN, dtype=torch.float32, device=DEVICE)
    for t in range(tokens):
        for slot in range(NUM_TOPK):
            e = int(hf_ids[t, slot])
            ref_down[t, slot] = (
                act_deq[t, slot: slot + 1] @ dequant_blocks(w2[e], w2_s[e]).t())
    report(tag, "w2 (vs fp32 of same bytes)", down[:tokens], ref_down, 4e-3)

    # the routed sum, against HF's own opaque FP8Experts module boundary
    hf_routed = load_oracle(mode, layer, "routed_expert_output")
    routed = (down[:tokens].float() * w_hf_order.unsqueeze(-1)).sum(dim=1)
    report(tag, "routed_expert_output", routed, hf_routed, 2e-2)

    # ---------- 3. shared expert ----------
    # TWO dense paths are measured, because the architecture doc was amended:
    #   fp8 + PRESERVED fp32 block scales -- the M2 acceptance path (M2-I12,
    #     amended docs/qwen35 6.2); modelled here as the pinned activation quant
    #     followed by an fp32 contraction of fp32-dequantized weights, which is
    #     exactly what linear_fp8_blockscale_sm100 computes;
    #   bf16-dequant -- the pre-amendment 6.1 entry, kept as a REPORTED scaffold
    #     number (MAIN.md: "bf16-dense = scaffold only per AC-1").
    sp = f"model.language_model.layers.{layer_idx}.mlp.shared_expert."
    wq, ws, wbf = {}, {}, {}
    for nm in ("gate_proj", "up_proj", "down_proj"):
        wq[nm] = shards.get(f"{sp}{nm}.weight").to(DEVICE)
        ws[nm] = shards.get(f"{sp}{nm}.weight_scale_inv").float().to(DEVICE)
        wbf[nm] = dequant_blocks(wq[nm], ws[nm]).to(torch.bfloat16)

    def fp8_dense(inp, nm):
        q, sc = quantize_activation(inp)
        return (dequant_groups(q, sc) @ dequant_blocks(wq[nm], ws[nm]).t()).to(
            torch.bfloat16)

    gate_out = fp8_dense(x, "gate_proj")
    up_out = fp8_dense(x, "up_proj")
    silu = (torch.nn.functional.silu(gate_out.float()).to(torch.bfloat16).float()
            * up_out.float()).to(torch.bfloat16)
    shared = fp8_dense(silu, "down_proj")
    report(tag, "shared gate_proj fp8bs", gate_out,
           load_oracle(mode, layer, "shared_gate_proj_out"), 1e-2)
    report(tag, "shared silu_mul fp8bs", silu,
           load_oracle(mode, layer, "shared_silu_mul_out"), 3e-2)
    report(tag, "shared down_proj fp8bs", shared,
           load_oracle(mode, layer, "shared_down_proj_out"), 2e-2)

    gate_bf = (x.float() @ wbf["gate_proj"].float().t()).to(torch.bfloat16)
    up_bf = (x.float() @ wbf["up_proj"].float().t()).to(torch.bfloat16)
    silu_bf = (torch.nn.functional.silu(gate_bf.float()).to(torch.bfloat16).float()
               * up_bf.float()).to(torch.bfloat16)
    shared_bf = (silu_bf.float() @ wbf["down_proj"].float().t()).to(torch.bfloat16)
    report(tag, "shared down_proj bf16 (scaffold)", shared_bf,
           load_oracle(mode, layer, "shared_down_proj_out"), 1.0)

    # ---------- 4. gate task (238) on HF's own shared output ----------
    w_sg = load_oracle(mode, layer, "__weight.shared_expert_gate_weight")
    hf_shared = load_oracle(mode, layer, "shared_down_proj_out").contiguous()
    hf_gated = load_oracle(mode, layer, "shared_output_gated")
    zero = torch.zeros_like(hf_shared)
    r_prime = torch.zeros_like(hf_shared)
    blk.sigmoid_gate_mul_add_sm100(x.contiguous(), w_sg.contiguous(), hf_shared,
                                   zero, r_prime)
    torch.cuda.synchronize()
    assert torch.equal(r_prime.view(torch.int16), hf_gated.view(torch.int16)), (
        f"{tag}: sigmoid_gate_mul_add must reproduce shared_output_gated exactly")
    print(f"  {tag:16s} {'shared_output_gated':24s} BIT-EXACT")

    # ---------- 5. block output ----------
    hf_combined = load_oracle(mode, layer, "combined_output")
    y = down[:tokens].contiguous()
    wt = w_hf_order.contiguous()
    out = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    blk.mul_sum_add_sm100(y, wt, r_prime, out)
    torch.cuda.synchronize()
    report(tag, "BLOCK combined_output", out, hf_combined, 2e-2)

    # the same block with OUR shared expert instead of HF's -- i.e. what M2-I8
    # will actually build, end to end, from checkpoint bytes only
    for label, sh, lim in (("BLOCK (fp8bs shared)", shared, 1e-2),
                           ("BLOCK (bf16 shared)", shared_bf, 1.0)):
        rp = torch.zeros_like(sh)
        blk.sigmoid_gate_mul_add_sm100(x.contiguous(), w_sg.contiguous(),
                                       sh.contiguous(), zero, rp)
        torch.cuda.synchronize()
        o = torch.zeros_like(out)
        blk.mul_sum_add_sm100(y, wt, rp, o)
        torch.cuda.synchronize()
        report(tag, label, o, hf_combined, lim)


def main():
    shards = Shards()
    for mode in ("decode", "prefill"):
        for layer, idx in (("moe0", 0), ("moe3", 3)):
            print(f"=== {mode} / {layer} ===")
            run_case(shards, mode, layer, idx)
    out = os.environ.get("MOE_BLOCK_ORACLE_JSON")
    if out:
        with open(out, "w") as f:
            json.dump({"boundaries": RESULTS}, f, indent=1)
        print(f"WROTE {out}")
    print("MOE BLOCK ORACLE TEST PASSED")


if __name__ == "__main__":
    main()
