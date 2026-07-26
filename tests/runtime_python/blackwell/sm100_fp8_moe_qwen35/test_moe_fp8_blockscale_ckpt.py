"""fp32-block-scale grouped FP8 MoE GEMM on REAL Qwen3.5-35B-A3B-FP8 tensors.

The regression form of probe P2's decisive half: layer-0 routed-expert weights
([256,1024,2048] w13 / [256,2048,512] w2) with their checkpoint
`weight_scale_inv` fed through untouched, driven by the HF oracle's own
activations and routing (demo/qwen3_5/oracle: moe0.layer_input, moe0.topk_ids).

Gates, following the P10/M2-I12 methodology
(demo/qwen3_5/accept/probes/fp8/p10_fp8_dense_bar.py, M2-I12's ckpt test):
  * frob_rel vs an fp32 contraction of the same bytes must sit at the kernel's
    own bf16 output-rounding floor (ratio <= 1.6) -- magnitude-weighted L2,
    never an elementwise max.
  * The per-row projection slope <actual,ref>/<ref,ref> must be ~1. This is the
    statistic the UE8M0 mechanism moves (P2 measured 0.49-0.51 for the shipped
    grouped kernel); a raw residual mean/std would NOT catch a multiplicative
    gain error, so it is reported but not gated on.
  * frob_rel vs the bf16-dequant reference must stay in P10's preserved-scale
    class (2-4.4e-3, with headroom to 6e-3 as in M2-I12).

Run:  QWEN35_SNAPSHOT=... python test_moe_fp8_blockscale_ckpt.py
"""

import json
import os
import sys

import torch
from safetensors import safe_open

import runtime_kernel_blackwell_fp8_moe_qwen35 as moe

torch.backends.cuda.matmul.allow_tf32 = False

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10
NUM_EXPERTS = 256
NUM_TOPK = 8
BATCH = 16
W13_N, W13_K = 1024, 2048
W2_N, W2_K = 2048, 512

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

FLOOR_RATIO_THRESHOLD = 1.6
FROB_REL_VS_BF16_THRESHOLD = 6e-3
SLOPE_TOLERANCE = 5e-3


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


def load_oracle():
    d = os.path.join(ORACLE, "prefill", "tensors")
    return (
        torch.load(os.path.join(d, "moe0.layer_input.pt"), map_location="cpu"),
        torch.load(os.path.join(d, "moe0.topk_ids.pt"), map_location="cpu"),
    )


def load_experts(expert_ids, device):
    with open(os.path.join(SNAPSHOT, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shards = {}

    def get(key):
        path = os.path.join(SNAPSHOT, index["weight_map"][key])
        if path not in shards:
            shards[path] = safe_open(path, framework="pt")
        return shards[path].get_tensor(key)

    p = "model.language_model.layers.0.mlp.experts."
    w13 = torch.zeros((NUM_EXPERTS, W13_N, W13_K), dtype=torch.float8_e4m3fn)
    w13_s = torch.zeros((NUM_EXPERTS, W13_N // BLOCK, W13_K // BLOCK))
    w2 = torch.zeros((NUM_EXPERTS, W2_N, W2_K), dtype=torch.float8_e4m3fn)
    w2_s = torch.zeros((NUM_EXPERTS, W2_N // BLOCK, W2_K // BLOCK))
    for e in expert_ids:
        gate, up = get(f"{p}{e}.gate_proj.weight"), get(f"{p}{e}.up_proj.weight")
        gate_s = get(f"{p}{e}.gate_proj.weight_scale_inv")
        up_s = get(f"{p}{e}.up_proj.weight_scale_inv")
        down = get(f"{p}{e}.down_proj.weight")
        down_s = get(f"{p}{e}.down_proj.weight_scale_inv")
        assert gate.dtype == torch.float8_e4m3fn
        # The checkpoint ships weight_scale_inv in BF16; MPK's loader widens
        # with .float() and nothing else (mpk-gaps.md 2.2.1).
        assert gate_s.dtype == torch.bfloat16
        w13[e] = torch.cat([gate, up], dim=0)
        w13_s[e] = torch.cat([gate_s, up_s], dim=0).float()
        w2[e] = down
        w2_s[e] = down_s.float()
    return w13.to(device), w13_s.to(device), w2.to(device), w2_s.to(device)


def build_routing(topk_ids, device):
    routing = torch.zeros((NUM_EXPERTS, BATCH), dtype=torch.int32)
    for t in range(topk_ids.shape[0]):
        for slot in range(NUM_TOPK):
            routing[int(topk_ids[t, slot]), t] = slot + 1
    activated = sorted({int(e) for e in topk_ids.flatten().tolist()})
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32)
    for i, e in enumerate(activated):
        mask[i] = e
    mask[NUM_EXPERTS] = len(activated)
    return routing.to(device), mask.to(device), activated


def reference(x_q, x_s, w_q, w_s, topk_ids, tokens, out_n, per_slot, to_bf16):
    out = torch.zeros((tokens, NUM_TOPK, out_n), dtype=torch.float32,
                      device=x_q.device)
    for t in range(tokens):
        for slot in range(NUM_TOPK):
            e = int(topk_ids[t, slot])
            xq = x_q[t, slot] if per_slot else x_q[t]
            xs = x_s[t, slot] if per_slot else x_s[t]
            xd = dequant_groups(xq.unsqueeze(0), xs.unsqueeze(0))
            wd = dequant_blocks(w_q[e], w_s[e])
            if to_bf16:
                out[t, slot] = (
                    xd.to(torch.bfloat16) @ wd.to(torch.bfloat16).t()
                ).float().squeeze(0)
            else:
                out[t, slot] = (xd @ wd.t()).squeeze(0)
    return out


def row_slopes(a, r):
    a2 = a.reshape(a.shape[0], -1).double()
    r2 = r.reshape(r.shape[0], -1).double()
    den = (r2 * r2).sum(dim=1)
    keep = den > 0
    return ((a2 * r2).sum(dim=1)[keep] / den[keep]).float()


def check(label, got, ref_fp32, ref_bf16):
    g = got.float()
    err = (g - ref_fp32).norm().item() / ref_fp32.norm().item()
    floor = (
        (ref_fp32.to(torch.bfloat16).float() - ref_fp32).norm().item()
        / ref_fp32.norm().item()
    )
    err_bf16 = (g - ref_bf16).norm().item() / ref_bf16.norm().item()
    sl = row_slopes(g, ref_fp32)
    diff = (g - ref_fp32).flatten().double()
    raw_effect = diff.mean().item() / max(diff.std(unbiased=True).item(), 1e-30)
    print(
        f"  {label:<12} frob_rel(fp32)={err:.3e} floor={floor:.3e} "
        f"ratio={err / floor:.2f} frob_rel(bf16_dequant)={err_bf16:.3e} "
        f"slope=[{sl.min().item():.6f}, {sl.max().item():.6f}] "
        f"raw_bias_effect={raw_effect:+.4f}"
    )
    assert err <= FLOOR_RATIO_THRESHOLD * floor, (
        f"{label}: frob_rel {err:.3e} is {err / floor:.2f}x the bf16 "
        f"output-rounding floor {floor:.3e}"
    )
    assert err_bf16 <= FROB_REL_VS_BF16_THRESHOLD, (
        f"{label}: frob_rel vs bf16 dequant {err_bf16:.3e} left P10's "
        f"preserved-scale class"
    )
    assert (sl - 1.0).abs().max().item() <= SLOPE_TOLERANCE, (
        f"{label}: per-row gain drifted from 1 (slope range "
        f"[{sl.min().item():.6f}, {sl.max().item():.6f}]) -- a scale is being "
        f"rounded or dropped"
    )


def main():
    if not os.path.isdir(SNAPSHOT):
        raise SystemExit(f"checkpoint snapshot not found: {SNAPSHOT}")
    if not os.path.isdir(ORACLE):
        raise SystemExit(f"oracle dumps not found: {ORACLE}")
    dev = "cuda"
    layer_input, topk_ids = load_oracle()
    tokens = layer_input.shape[0]
    routing, mask, activated = build_routing(topk_ids, dev)
    w13, w13_s, w2, w2_s = load_experts(activated, dev)

    print("=== fp32-block-scale grouped MoE GEMM on REAL layer-0 experts ===")
    print(f"snapshot: {SNAPSHOT}")
    print(f"tokens={tokens} activated_experts={len(activated)}")

    x = torch.zeros((BATCH, W13_K), dtype=torch.bfloat16, device=dev)
    x[:tokens] = layer_input.to(dev).to(torch.bfloat16)
    x_q, x_s = quantize_activation(x)

    mid = torch.zeros((BATCH, NUM_TOPK, W13_N), dtype=torch.bfloat16, device=dev)
    moe.moe_w13_blockscale_sm100(
        x_q, x_s, w13, w13_s.contiguous(), routing, mask, mid
    )
    ref13 = reference(x_q, x_s, w13, w13_s, topk_ids, tokens, W13_N, False, False)
    ref13_bf16 = reference(
        x_q, x_s, w13, w13_s, topk_ids, tokens, W13_N, False, True
    )
    check("w13", mid[:tokens], ref13, ref13_bf16)

    half = W13_N // 2
    act = torch.zeros((BATCH, NUM_TOPK, W2_K), dtype=torch.bfloat16, device=dev)
    act[:tokens] = (
        torch.nn.functional.silu(ref13[..., :half].float()) * ref13[..., half:]
    ).to(torch.bfloat16)
    a_q, a_s = quantize_activation(act)

    out = torch.zeros((BATCH, NUM_TOPK, W2_N), dtype=torch.bfloat16, device=dev)
    moe.moe_w2_blockscale_sm100(
        a_q, a_s, w2, w2_s.contiguous(), routing, mask, out
    )
    ref2 = reference(a_q, a_s, w2, w2_s, topk_ids, tokens, W2_N, True, False)
    ref2_bf16 = reference(a_q, a_s, w2, w2_s, topk_ids, tokens, W2_N, True, True)
    check("w2", out[:tokens], ref2, ref2_bf16)

    print("ALL REAL-CHECKPOINT MOE BLOCKSCALE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
