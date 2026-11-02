"""Every task of the K3 MoE layer on ONE AMD GPU (or 2 or 4: --gpus), with random weights, checked step by step against torch.

The graph is the demo's (demo/kimi_k3_moe/demo.py build_graph), built for --gpus GPUs. On one GPU, GPU 0 computes every row of
the row-split nodes and the all-reduce adds over one GPU, so the cross-GPU exchange is the only part not tested (its stores and
polls run, to this GPU only); with 2 or 4 it is tested too. The plan: the first GPUs' lists of a 256-CU plan, with the tasks
each GPU owns at that GPU count (amd/tools/plan_one_gpu.py). The weights every GPU has are the same on all; each GPU has its
own random shard of the shared expert and the expert banks; the reference adds the GPUs' partial sums in GPU order, as the
kernel.
Random weights in the real shapes (no weight files): bf16 weights scaled so every activation is O(1); the expert banks random
bytes straight in the gfx950 tile layout (MXFP4 values, scales 2^-8 .. 2^-5), dequantized back from that layout for the
reference (tiles_to_matrix; --self-test checks it against amd/tools/shuffle_weights.py on the CPU).
The kernel's intermediate outputs that are not re-armed by their readers are bound to torch tensors by name and read back:
routing_pairs, shared_gate_up, shared_act, routed_normed, then R, S, y. Each step's reference starts from the kernel's own
inputs to that step (where readable), so a wrong step shows itself and not its consumers:
  1. routing    the kernel's pairs vs top-16 of sigmoid(x Wg^T) + bias (expert sets per token; weights)
  2. experts    R (per token and routed expert: weight x W2 . MXFP8(SiTU(W13 . MXFP8(x Wd^T)))) from the kernel's pairs
  3. shared     shared_gate_up vs x Wsgu^T; shared_act vs SiTU of the kernel's shared_gate_up; S vs shared_act Wsd^T
  4. norm       routed_normed vs RMSNorm(sum over GPUs of bf16(sum_k R)) x gamma from the kernel's R
  5. output     y vs bf16(bf16(bf16(routed_normed Wup^T) + bf16(sum over GPUs of bf16(S))) + prefix) from the kernel's
                routed_normed and S
  pairs, routed_normed and y bit-identical on every GPU. Then a second launch: every output bit-identical to the first (the
  re-arm / reset path).

python amd/tests/test_layer_one_gpu.py [--gpus 1] [--plan demo/kimi_k3_moe/solver_plan_256cu.json] [--out DIR] [--flag -DX=1 ...]
                                       [--profile]   (the timing build: a launch that does not finish prints, per node, the
                                                      tasks started and not finished and the SMs that have not started them)
                                       [--record COSTS --reps N]   (timing build: after the checks, N more launches; their
                                                      cost records appended to COSTS, mirage.mpk.cost_records)
                                       [--search COSTS ... --time S]   (the plan: the compiler's search (search.search_plan)
                                                      from those records, starting from --plan; <out>/plan.json)
python amd/tests/test_layer_one_gpu.py --self-test          (CPU: the tile-layout dequantization vs shuffle_weights.py)
"""
import argparse
import importlib.util
from collections import Counter
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "amd", "tools"))

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def load_demo():
    spec = importlib.util.spec_from_file_location("k3_moe_demo", os.path.join(ROOT, "demo", "kimi_k3_moe", "demo.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


# ---- the gfx950 tile layout back to matrices (amd/tools/shuffle_weights.py, inverted) ----
def tiles_to_values(tiles: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[N/16][K/128][64 lanes][16 B] (lane l: row 16 rt + l % 16, K 128 ks + 32 (l / 16) .. + 31, the lower K in the low nibble)
    -> e2m1 codes [N][K]."""
    t = tiles.reshape(n // 16, k // 128, 64, 16)
    v = torch.stack((t & 0xF, t >> 4), dim=-1).reshape(n // 16, k // 128, 4, 16, 32)   # [rt][ks][g][r][32]
    return v.permute(0, 3, 1, 2, 4).reshape(n, k)


def scales_to_matrix(sc: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Scale bytes in the gfx950 layout -> [N][K/32]. K/128 steps a multiple of 4 (W13): [N/16][K/512][64][4], byte j of lane l =
    row 16 rt + l % 16, K block 4 (4 kg + j) + l / 16. Otherwise (W2): [N/64][64][4 x KS], byte KS (rt % 4) + st = row
    16 rt + l % 16, K block 4 st + l / 16."""
    ks = k // 128
    if ks % 4 == 0:
        s = sc.reshape(n // 16, ks // 4, 4, 16, 4)       # [rt][kg][g][r][j]
        return s.permute(0, 3, 1, 4, 2).reshape(n, k // 32)
    s = sc.reshape(n // 64, 4, 16, 4, ks)                 # [rt4][g][r][r4][st]
    return s.permute(0, 3, 2, 4, 1).reshape(n, k // 32)


def dequant(tiles: torch.Tensor, sc: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """One expert's MXFP4 matrix in the gfx950 layout -> float64 [N][K]."""
    table = torch.tensor(E2M1, dtype=torch.float64, device=tiles.device)
    vals = table[tiles_to_values(tiles, n, k).long()]
    scale = torch.exp2(scales_to_matrix(sc, n, k).double() - 127)
    return vals * scale.repeat_interleave(32, dim=1)


def w13_rows(ir: int, device) -> tuple:
    """The item-order rows of W13 (item i = gate features 32 i .. + 31, then up 32 i .. + 31) of gate and up feature f."""
    f = torch.arange(ir, device=device)
    gate = 64 * (f // 32) + f % 32
    return gate, gate + 32


# ---- the kernel's arithmetic, in torch ----
def situ(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)


def ceil_log2_bits(a: torch.Tensor) -> torch.Tensor:
    """ceil(log2(a)) of positive normal fp32 a from its bits (runtime.cuh ceil_log2_bits)."""
    b = a.float().contiguous().view(torch.int32)
    return ((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0).int()


def mxfp8(v: torch.Tensor) -> torch.Tensor:
    """MXFP8 of v (fp32, blocks of 32 along the last axis) as the kernel makes it, dequantized: scale 2^ex, ex = ceil(log2(amax /
    448)) clamped to [-127, 127], values e4m3 of v 2^-ex (round to nearest even, |.| <= 448)."""
    shape = v.shape
    b = v.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True)
    ex = ceil_log2_bits(torch.clamp(amax, min=1e-30) * torch.tensor(1 / 448, dtype=torch.float32)).clamp(-127, 127)
    sc = torch.exp2(ex.float())
    q = (b / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q.double() * sc.double()).reshape(shape)


def bf16(v: torch.Tensor) -> torch.Tensor:
    return v.float().to(torch.bfloat16)


# ---- random weights in the real shapes ----
def moe_ours_to_aiter(w13b, w13s, w2b, w2s):
    """Our gfx950 tile layout of the routed experts -> AITER's (SGLang's process_weights_after_loading on K3's a16w4 path; the
    kernel's STATIC_MOE_AITER): the same bytes in another order (box agent_tmp/aiterlay/layout_check.py). W13 row tiles 1 and 2
    of each 4-tile item swapped; W13 scales [E][IR / 16][LAT / 256][64][4], byte (ks % 2) 2 + gate / up; W2 the same; W2 scales
    [E][LAT / 32][2][64][4], byte (ks % 2) 2 + row tile % 2, k-steps padded to 4 with 0x7F (shape (E, LAT, 16))."""
    E, R1, KB = w13b.shape[0], w13b.shape[1] // 16, w13b.shape[2] * 16   # (row tiles, bytes per tile)
    L, IR = w2b.shape[1], w2b.shape[2] * 2
    KS1, KG1, KS2, RT2 = L // 128, L // 512, IR // 128, L // 16
    perm = [0, 2, 1, 3]
    idx = torch.tensor([4 * (r // 4) + perm[r % 4] for r in range(R1)], device=w13b.device)
    a13 = w13b.reshape(E, R1, KB)[:, idx].reshape(w13b.shape).contiguous()
    od = w13s.reshape(E, R1, KG1, 64, 4)
    ad = torch.empty(E, IR // 16, KS1 // 2, 64, 4, dtype=torch.uint8, device=w13s.device)
    for rt in range(R1):
        it, ta = rt // 4, perm[rt % 4]
        for ks in range(KS1):
            ad[:, 2 * it + ta // 2, ks // 2, :, (ks % 2) * 2 + ta % 2] = od[:, rt, ks // 4, :, ks % 4]
    os_ = w2s.reshape(E, RT2 // 4, 64, 4, KS2)
    as_ = torch.full((E, L // 32, 2, 64, 4), 0x7F, dtype=torch.uint8, device=w2s.device)
    for rt in range(RT2):
        for ks in range(KS2):
            as_[:, rt // 2, ks // 2, :, (ks % 2) * 2 + rt % 2] = os_[:, rt // 4, :, rt % 4, ks]
    return a13, ad.reshape(w13s.shape), w2b.clone(), as_.reshape(E, L, 16)


def random_weights(d, dev, seed: int = 1, common: dict = None, token_share: float = 0.6) -> dict:
    """GPU dev's tensors: its own random shard (seed), the weights every GPU has from `common` (GPU 0's dict) when given.
    token_share: the 8 tokens = token_share x one common vector + the rest their own, so they share experts as real tokens do
    (0.6: ~88 experts for the 8 tokens, as K3's real routing ~90 on B300; 0: independent tokens, ~114)."""
    T, H, L, E, IR, SHR = d.T, d.H, d.L, d.E, d.IR_LOCAL, d.SHR_LOCAL
    g = torch.Generator(device=dev).manual_seed(seed)
    bf = lambda shape, std: (torch.randn(shape, generator=g, device=dev) * std).to(torch.bfloat16)
    u8 = lambda shape, lo, hi: torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.uint8)
    w = {
        "router_weight": bf((E, H), H ** -0.5),
        "score_correction_bias": torch.randn(E, generator=g, device=dev) * 0.05,
        "latent_down_weight": bf((L, H), H ** -0.5),
        "latent_up_weight": bf((H, L), L ** -0.5),
        "routed_expert_norm_weight": (1 + 0.1 * torch.randn(L, generator=g, device=dev)).to(torch.bfloat16),
        "moe_in": (token_share * torch.randn(1, H, generator=g, device=dev)
                   + (1 - token_share ** 2) ** 0.5 * torch.randn(T, H, generator=g, device=dev)).to(torch.bfloat16),
        "moe_prefix": bf((T, H), 1.0),
        "shared_gate_up_weight": bf((2 * SHR, H), H ** -0.5),
        "shared_down_weight": bf((H, SHR), SHR ** -0.5),
        # e2m1 values (all 16 codes), scales: W13 rows . z (L = 3584 terms of rms ~3) ~ 1 at 2^-7, W2 (IR = 384) ~ 1 at 2^-6
        "w13_blocks": u8((E, 2 * IR, L // 2), 0, 256),
        "w13_scales": u8((E, 2 * IR, L // 32), 119, 122),
        "w2_blocks": u8((E, L, IR // 2), 0, 256),
        "w2_scales": u8((E, L, IR // 32), 120, 123),
        # the outputs
        "moe_out": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "routed_sum_partial": torch.zeros(T * d.TOPK, L, dtype=torch.float32, device=dev),
        "shared_down_partial": torch.zeros(T, H, dtype=torch.float32, device=dev),
    }
    # intermediate outputs bound by name (host.cuh build_bufs: a node's output buffer at a given tensor of its name)
    w["routing_pairs"] = torch.zeros(T, d.TOPK, dtype=torch.int64, device=dev)
    w["shared_gate_up"] = torch.zeros(T, 2 * SHR, dtype=torch.int64, device=dev)
    w["shared_act"] = torch.zeros(T, SHR, dtype=torch.bfloat16, device=dev)
    w["routed_normed"] = torch.zeros(T, L, dtype=torch.bfloat16, device=dev)
    # MPK_MOE_INPUT_WAIT: input_wait's copies of x and the prefix (node outputs the front GEMMs read through maps)
    w["moe_in_w"] = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    w["moe_prefix_w"] = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    # --agg (MPK_MOE_AGG, tasks/moe_attn_residual.cuh): the next layer's aggregation 1 (the same on every GPU)
    w["agg_bank"] = bf((T, AGG_NB_ALLOC, H), 1.0)
    w["agg_q"] = bf((H,), H ** -0.5)
    w["agg_gain"] = (1 + 0.1 * torch.randn(H, generator=g, device=dev)).to(torch.bfloat16)
    w["agg_ow"] = (1 + 0.1 * torch.randn(H, generator=g, device=dev)).to(torch.bfloat16)
    w["agg_info"] = torch.tensor([AGG[0], AGG_NB_ALLOC * H, H, AGG[1], 0, 0, 0, 0], dtype=torch.int32, device=dev)
    w["agg_out"] = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    if common is not None:
        for name, _, _, _, sharded in d.WEIGHT_FILES:
            if not sharded:
                w[name] = common[name].to(dev)
        for name in ("agg_bank", "agg_q", "agg_gain", "agg_ow"):
            w[name] = common[name].to(dev)
    if os.environ.get("MPK_MOE_AITER") == "1":   # (the kernel's STATIC_MOE_AITER: the GPU copies in AITER's layout; the
        # reference keeps ours)
        keys = ("w13_blocks", "w13_scales", "w2_blocks", "w2_scales")
        for k in keys:   # (tensors under their own names: code that walks w's values sees only tensors)
            w["_ours_" + k] = w[k]
        for k, v in zip(keys, moe_ours_to_aiter(*(w[k] for k in keys))):
            w[k] = v
    return w


AGG_NB_ALLOC, AGG = 8, [3, 0]   # --agg: bank rows allocated per token; [bank rows used, write] (main sets them)


def check_agg(w0, ws: list, rep, bank0, eps: float = 1e-5) -> None:
    """--agg: the next layer's aggregation 1 (SGLang ROCm attn_res_hip after a MoE layer, prefix = y) from the kernel's own y,
    float64; the same on every GPU; a write layer: bank row nb = y."""
    nb = AGG[0]
    y = w0["moe_out"].double()
    rows = bank0[:, :nb].double().to(y.device)
    src = torch.cat([rows, y.unsqueeze(1)], 1)
    c = (w0["agg_q"].float() * w0["agg_gain"].float()).double()
    sc = (src * c).sum(-1) / torch.sqrt((src * src).mean(-1) + eps)
    hh = (torch.softmax(sc, -1).unsqueeze(-1) * src).sum(1)
    ref = bf16(hh / torch.sqrt((hh * hh).mean(-1, keepdim=True) + eps) * w0["agg_ow"].double())
    rep.check("aggregation 1 (next layer's input)", relrms(w0["agg_out"], ref) < 5e-3,
              f"relRMS {relrms(w0['agg_out'], ref):.2e}, bf16 bits equal {100 * float((w0['agg_out'].view(torch.int16) == ref.view(torch.int16)).float().mean()):.1f}%")
    if "agg_mix" in w0:   # (MPK_AGG_MIX: the pre-norm mixture, SGLang's aggregate_stream for its dspark capture)
        mref = bf16(hh)
        rep.check("aggregation 1's pre-norm mixture (agg_mix)", relrms(w0["agg_mix"], mref) < 5e-3,
                  f"relRMS {relrms(w0['agg_mix'], mref):.2e}, bf16 bits equal "
                  f"{100 * float((w0['agg_mix'].view(torch.int16) == mref.view(torch.int16)).float().mean()):.1f}%")
    if len(ws) > 1:
        rep.check("aggregation 1 the same on every GPU", same_bits([w["agg_out"] for w in ws]), "bits")
    if AGG[1]:
        rep.check("aggregation 1: bank row written (= y)", all(same_bits([w["agg_bank"][:, nb].cpu(), w["moe_out"].cpu()])
                                                             for w in ws), "bits")


# ---- the checks ----
class Report:
    def __init__(self):
        self.failed = []

    def check(self, name: str, ok: bool, text: str):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {text}", flush=True)
        if not ok:
            self.failed.append(name)


def relrms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in bf16 units of the last place (same-sign values: the difference of their bit patterns)."""
    ai, bi = a.to(torch.bfloat16).view(torch.int16).int(), b.to(torch.bfloat16).view(torch.int16).int()
    return (ai - bi).abs()


def same_bits(ts) -> bool:
    return all(torch.equal(ts[0].cpu().view(torch.uint8), t.cpu().view(torch.uint8)) for t in ts[1:])


# --gemm-shuffled: the kernel gets the bf16 GEMM weights in gemm_tile's shuffled layout; the row-major originals (per GPU, by
# the weight tensor's data pointer) stay here for the references
FLAGS = []   # main's --flag list (the references follow the build's numerics defines)
GEMM_WEIGHTS = ("router_weight", "latent_down_weight", "latent_up_weight", "shared_gate_up_weight", "shared_down_weight")
ROWMAJOR = {}


def gw(w: dict, name: str) -> torch.Tensor:
    """GEMM weight `name` of GPU dict w, row-major (the original when the kernel got a shuffled copy)."""
    return ROWMAJOR.get(w[name].data_ptr(), w[name])


def check_all(d, ws: list, rep: Report, eps: float = 1e-5, solo: bool = False, fold: bool = False):
    """ws[g]: GPU g's tensors after the launch (weights and outputs). solo: GPU 0 of a TP8 layer run alone (--solo): the other
    GPUs' parts are zeros, so only the steps that need no other GPU are checked (routing, the K parts, the shared expert)."""
    T, H, L, E, K, IR, SHR = d.T, d.H, d.L, d.E, d.TOPK, d.IR_LOCAL, d.SHR_LOCAL
    ngpu, w0 = len(ws), ws[0]
    x = w0["moe_in"].double()
    # 1. routing (every GPU routes for itself: the same pairs)
    logits = x @ gw(w0, "router_weight").double().T
    sig = torch.sigmoid(logits)
    top = torch.topk(sig + w0["score_correction_bias"].double(), K, dim=1)
    pk = w0["routing_pairs"].cpu()
    e_k = (pk >> 32).int()
    wt_k = (pk & 0xFFFFFFFF).int().view(torch.float32)
    ref_sets = top.indices.cpu()
    same = sum(set(e_k[t].tolist()) == set(ref_sets[t].tolist()) for t in range(T))
    # the 16th and 17th scores of each token: a tie closer than fp32 rounding may pick either
    s_sorted = torch.sort(sig + w0["score_correction_bias"].double(), dim=1, descending=True).values
    gap = (s_sorted[:, K - 1] - s_sorted[:, K]).min().item()
    rep.check("routing: expert sets", same == T, f"{same}/{T} tokens pick the reference's 16 experts (smallest gap 16th-17th "
              f"score {gap:.2e}); {len(set(e_k.flatten().tolist()))} experts used")
    order = torch.argsort(torch.argsort(-(sig + w0["score_correction_bias"].double()), dim=1), dim=1).cpu()   # reference rank
    for t in range(T):
        if set(e_k[t].tolist()) != set(ref_sets[t].tolist()):
            print(f"    token {t}: the kernel's experts' reference ranks {sorted(order[t, e_k[t].long()].tolist())}", flush=True)
    wref = sig.cpu().gather(1, e_k.long()) / sig.cpu().gather(1, e_k.long()).sum(1, keepdim=True)
    rep.check("routing: weights", relrms(wt_k, wref) < 1e-5, f"relRMS {relrms(wt_k, wref):.2e} (weights of the kernel's experts)")
    if ngpu > 1:
        rep.check("routing: the same on every GPU", same_bits([w["routing_pairs"] for w in ws]), "bits")
    for name, ref in (("router_logits", logits), ("latent_z", x @ gw(w0, "latent_down_weight").double().T)):
        if name in w0:
            parts = w0[name].float().reshape(-1, T, ref.shape[1])
            if name == "latent_z":   # rows split over the GPUs: GPU 0 computes its 128-column blocks only
                lo, hi = 0, (L // 128) // (8 if solo else ngpu)
                parts, ref = parts[:, :, lo * 128:hi * 128], ref[:, lo * 128:hi * 128]
            ff = (parts.view(torch.int32) == -1).sum().item()
            tot = torch.zeros_like(parts[0])
            for q in range(parts.shape[0]):
                tot = tot + parts[q]
            rep.check(f"K parts: {name}", ff == 0 and relrms(tot, ref) < 1e-5,
                      f"{parts.shape[0]} parts, {ff} words still 0xFF; their sum vs the reference relRMS {relrms(tot, ref):.2e}")
            if not (ff == 0 and relrms(tot, ref) < 1e-5):   # (which tokens / parts)
                print(f"    per token relRMS {[round(relrms(tot[t], ref[t]), 5) for t in range(T)]}", flush=True)
                if name == "router_logits":   # per K part (KP columns each): the tokens off
                    KP = x.shape[1] // parts.shape[0]
                    wr = gw(w0, "router_weight").double()
                    off = []
                    for q in range(parts.shape[0]):
                        rq = x[:, q * KP:(q + 1) * KP] @ wr[:, q * KP:(q + 1) * KP].T
                        bt = [t for t in range(T) if not relrms(parts[q][t].double(), rq[t]) < 1e-4]
                        if bt:
                            off.append((q, bt))
                    print(f"    K parts off (part, tokens): {off}", flush=True)
    z = (x @ gw(w0, "latent_down_weight").double().T).float()
    z = bf16(z).float()   # (sum_quant_send.cuh: z rounded to bf16 first, as SGLang ROCm)
    zq = mxfp8(z)                                              # [T][L] dequantized (the same z_q on every GPU)
    gate_r, up_r = w13_rows(IR, x.device)
    rsum = torch.zeros(T, L, dtype=torch.float32, device=x.device)    # sum over GPUs of bf16(sum_k R) (fp32, GPU order)
    ssum = torch.zeros(T, H, dtype=torch.float32, device=x.device)    # sum over GPUs of bf16(S)
    for gi, w in enumerate(ws):
        tag = f"GPU {gi}: " if ngpu > 1 else ""
        dev = w["moe_in"].device
        # 2. experts, from the kernel's pairs, with this GPU's banks (solo: z_q has other GPUs' columns as zeros: not checked)
        R_ref = torch.zeros(T * K, L, dtype=torch.float64, device=dev)
        if solo:
            e_k_all = []
        else:
            e_k_all = sorted(set(e_k.flatten().tolist()))
        zq_g = zq.to(dev)
        for e in e_k_all:
            wo = {k[6:]: v for k, v in w.items() if k.startswith("_ours_")} or w   # (MPK_MOE_AITER: our layout's copy)
            w13 = dequant(wo["w13_blocks"][e], wo["w13_scales"][e], 2 * IR, L)       # [2 IR][L], item-order rows
            w2 = dequant(wo["w2_blocks"][e], wo["w2_scales"][e], L, IR)              # [L][IR]
            for t, k in (e_k == e).nonzero().tolist():
                hg, hu = w13[gate_r.to(dev)] @ zq_g[t], w13[up_r.to(dev)] @ zq_g[t]
                h = situ(hg, hu)
                R_ref[t * K + k] = wt_k[t, k].item() * (w2 @ mxfp8(h.float()))
        R = w["routed_sum_partial"].double()
        e_R = relrms(R, R_ref)
        worst = ((R - R_ref).abs().amax(1) / R_ref.abs().amax(1).clamp_min(1e-30)).max().item()
        if not solo:
            rep.check(tag + "experts: R", e_R < 1e-2, f"relRMS {e_R:.2e}, worst row max|d|/max|ref| {worst:.2e} (MXFP8 rounding "
                      f"of h and z can differ where the fp32 sums differ in the last bits)")
        # 3. shared expert (this GPU's shard)
        xg = w["moe_in"].double()
        sgu = w["shared_gate_up"].double() * 2.0 ** -40
        sgu_ref = xg @ gw(w, "shared_gate_up_weight").double().T
        rep.check(tag + "shared: gate_up", relrms(sgu, sgu_ref) < 1e-5, f"relRMS {relrms(sgu, sgu_ref):.2e} (fixed point 2^-40)")
        sgu_f = sgu.float().reshape(T, 2 * SHR // 128, 2, 64)                     # blocks of 128: 64 gate, their 64 up
        hs_ref = bf16(situ(sgu_f[:, :, 0, :].double(), sgu_f[:, :, 1, :].double()).reshape(T, SHR))
        ul = bf16_ulps(w["shared_act"], hs_ref)
        rep.check(tag + "shared: act", ul.max().item() <= 1, f"bf16 values differing: {(ul > 0).sum().item()} by 1 ulp, "
                  f"{(ul > 1).sum().item()} by more (of {ul.numel()})")
        S_ref = w["shared_act"].double() @ gw(w, "shared_down_weight").double().T
        S = w["shared_down_partial"].double()
        rep.check(tag + "shared: down S", relrms(S, S_ref) < 1e-5, f"relRMS {relrms(S, S_ref):.2e}")
        if not relrms(S, S_ref) < 1e-5:   # (diagnosis: the tokens and 128-row blocks of S that differ)
            bad = ((S - S_ref).abs() > 1e-3 * S_ref.abs().max()).nonzero()
            print(f"    S differs: tokens {sorted(set(bad[:, 0].tolist()))[:24]}; 128-row blocks "
                  f"{sorted(set((bad[:, 1] // 128).tolist()))[:24]}; {len(bad)} values", flush=True)
            # per wrong (token, block), up to 4: the reference row (any token) S matches best there, and S / S_ref
            for t, b in sorted({(int(i), int(j) // 128) for i, j in bad.tolist()})[:4]:
                cs = slice(128 * b, 128 * b + 128)
                errs = [relrms(S[t, cs], S_ref[u, cs]) for u in range(T)]
                u = min(range(T), key=lambda v: errs[v])
                ratio = (S[t, cs] / S_ref[t, cs]).median().item()
                print(f"    S[{t}] block {b}: relRMS vs its own row {errs[t]:.3f}, best row {u} ({errs[u]:.3f}); "
                      f"median S / S_ref {ratio:.3f}", flush=True)
        # this GPU's parts of the all-reduces, from the kernel's R and S
        r = w["routed_sum_partial"].float().reshape(T, K, L)
        rs = torch.zeros(T, L, dtype=torch.float32, device=dev)
        for k in range(K):                                                          # in order, fp32 (allreduce_send)
            rs = rs + r[:, k]
        rsum = rsum + bf16(rs).float().to(x.device)
        ssum = ssum + bf16(w["shared_down_partial"]).float().to(x.device)
    # 4. RMSNorm of the routed sum over the GPUs (solo: the other GPUs' parts are zeros, so GPU 0's alone: still checkable)
    rstd = torch.rsqrt((rsum.double() ** 2).sum(1, keepdim=True) / L + eps)
    if fold:   # no routed_normed: latent_up made bf16(R x gamma) itself and scales its sums by rstd (gemm_tile norm)
        if solo:
            return
        a_ref = bf16(rsum * w0["routed_expert_norm_weight"].float())
        up = ((a_ref.double() @ gw(w0, "latent_up_weight").double().T) * rstd).float()
        check_y(w0, ws, ngpu, rep, up, ssum)
        return
    rn_ref = bf16((rsum.double() * rstd * w0["routed_expert_norm_weight"].double()).float())
    ul = bf16_ulps(w0["routed_normed"], rn_ref)
    rep.check("norm: routed_normed", ul.max().item() <= 1, f"bf16 values differing: {(ul > 0).sum().item()} by 1 ulp, "
              f"{(ul > 1).sum().item()} by more (of {ul.numel()})")
    if ngpu > 1:
        rep.check("norm: the same on every GPU", same_bits([w["routed_normed"] for w in ws]), "bits")
    if solo:
        bad = (ul > 1).nonzero()
        if len(bad):
            print(f"    routed_normed: values off by more than 1 ulp at (token, column): {bad[:8].tolist()} ...; tokens "
                  f"{sorted(set(bad[:, 0].tolist()))}", flush=True)
            t0, c0 = bad[0].tolist()
            got, ref = w0["routed_normed"][t0].float().cpu(), rn_ref[t0].float().cpu()
            cols = sorted(set(bad[bad[:, 0] == t0][:, 1].tolist()))
            print(f"    token {t0}: bad columns {cols[:40]} ... ({len(cols)}); kernel / reference at the first 6: "
                  f"{[(c, round(got[c].item(), 4), round(ref[c].item(), 4)) for c in cols[:6]]}; ratio "
                  f"{[round(got[c].item() / ref[c].item(), 3) if ref[c].item() else None for c in cols[:6]]}; the landed sum there "
                  f"(bf16 of R summed): {[round(rsum[t0, c].item(), 4) for c in cols[:6]]}; kernel at the good columns "
                  f"{c0 - 8}, {c0 + 2}: {round(got[c0 - 8].item(), 4)}, {round(got[c0 + 2].item(), 4)}, sums "
                  f"{round(rsum[t0, c0 - 8].item(), 4)}, {round(rsum[t0, c0 + 2].item(), 4)}", flush=True)
        return
    # 5. latent_up (each GPU its rows), sum_send, residual_add, from the kernel's routed_normed and the S sum
    up = (w0["routed_normed"].double() @ gw(w0, "latent_up_weight").double().T).float()
    check_y(w0, ws, ngpu, rep, up, ssum)


def check_y(w0, ws: list, ngpu: int, rep: Report, up, ssum) -> None:
    """y (GPU 0) vs y from the latent_up output up (fp32) and the S sum, and y the same on every GPU."""
    terms = (bf16(up).float(), bf16(ssum).float(), w0["moe_prefix"].float())
    y_ref = bf16(bf16(terms[0] + terms[1]).float() + terms[2])
    ul = bf16_ulps(w0["moe_out"], y_ref)
    # y is three bf16 roundings of a sum: where the terms cancel, one ulp of a term is many ulps of y. The bound: 2 bf16 ulps
    # of the largest term (the latent_up output's fp32 K-part sums round differently from the reference's)
    mag = torch.stack([t.abs() for t in terms]).amax(0).clamp_min(1e-30)
    ulp_term = torch.exp2(torch.floor(torch.log2(mag)) - 7)
    over = ((w0["moe_out"].float() - y_ref.float()).abs() > 2 * ulp_term.to(y_ref.device)).sum().item()
    rep.check("output: y", over == 0 and relrms(w0["moe_out"], y_ref) < 1e-3,
              f"relRMS {relrms(w0['moe_out'], y_ref):.2e}; bf16 values differing: {(ul == 1).sum().item()} by 1 ulp, "
              f"{(ul == 2).sum().item()} by 2, {(ul > 2).sum().item()} by more (of {ul.numel()}); {over} beyond 2 ulps of "
              f"the largest term")
    if ngpu > 1:
        rep.check("output: y the same on every GPU", same_bits([w["moe_out"] for w in ws]), "bits")


def where_stuck(h, paths: list, nodes) -> None:
    """Timing build, a launch that does not finish: per GPU and node, the tasks started but not finished (SM: list entry, pos)
    and the number of SMs whose next task (the first not started) is of that node."""
    names = {n.graph_idx: n.name for n in nodes}
    times = h.task_times()
    for g, path in enumerate(paths[: h.sk.num_gpus]):   # solo: GPU 0's only
        with open(path) as f:
            sched = json.load(f)
        tasks, queues = sched["all_tasks"], sched["worker_task_queues"]
        running, waiting_next, done = {}, {}, 0
        for sm, queue in enumerate(queues):
            for i, tid in enumerate(queue):
                start, end, st0, st1 = times[g][sm][i]
                t = tasks[tid]
                key = f"{names[t['node']]}#{t['node']}"
                if start and not end:
                    running.setdefault(key, []).append((sm, i, tuple(t["pos"]), "stage stamps: " + ("0" if st0 else "-") +
                                                        ("1" if st1 else "-")))
                elif not start:
                    waiting_next[key] = waiting_next.get(key, 0) + 1
                    break
                else:
                    done += 1
        print(f"GPU {g}: {done} tasks finished; started, not finished (SM, list entry, pos, which stage stamps were "
              f"written):", flush=True)
        for key, lst in sorted(running.items()):
            stamps = {}
            for item in lst:
                stamps[item[3]] = stamps.get(item[3], 0) + 1
            print(f"  {key}: {len(lst)} tasks {stamps}, e.g. {lst[:4]}", flush=True)
        print(f"  next task not started, per node (SMs): {waiting_next}", flush=True)


def node_times(h, paths: list, nodes) -> None:
    """Timing build, after a launch: per GPU and node, its tasks, the median and longest task (start to the last warp's end),
    and when its first task started and its last ended (us from the last SM past the start barrier)."""
    names = {n.graph_idx: n.name for n in nodes}
    times = h.task_times()
    starts = [b for b, _ in h.timing()]
    dump = os.environ.get("MPK_DUMP_TIMES")
    if dump:   # every task of every GPU: node, position, start, end, stage stamps 0 / 1 (ns after the GPU's start barrier)
        rows = []
        for g, path in enumerate(paths[: h.sk.num_gpus]):
            with open(path) as f:
                sched = json.load(f)
            for sm, queue in enumerate(sched["worker_task_queues"]):
                for i, tid in enumerate(queue):
                    start, end, st0, st1 = times[g][sm][i]
                    task = sched["all_tasks"][tid]
                    rel = lambda v: int(v) - int(starts[g]) if v else None
                    rows.append([g, sm, task["node"], names[task["node"]], task.get("pos"), rel(start), rel(end), rel(st0), rel(st1)])
        cus = []   # per GPU and CU its stamps (config.cuh STAMP_*; ns after the GPU's start barrier, 0: not written)
        for g in range(h.sk.num_gpus):
            lay = h.layout
            st = h._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"])
            cus.append([[int(v) - int(starts[g]) if v else 0 for v in row] for row in st.tolist()])
        with open(dump, "w") as f:
            json.dump({"cols": ["gpu", "cu", "node", "name", "pos", "start", "end", "st0", "st1"], "rows": rows,
                       "cu_stamps": cus}, f)
        print(f"task times of every GPU -> {dump}", flush=True)
    for g, path in enumerate(paths[: h.sk.num_gpus]):   # solo: GPU 0's only
        with open(path) as f:
            sched = json.load(f)
        tasks, queues = sched["all_tasks"], sched["worker_task_queues"]
        per, moe, topk, landed = {}, [], [], {}
        lay = h.layout
        stamps = h._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"])
        for sm, queue in enumerate(queues):
            for i, tid in enumerate(queue):
                start, end, st0, st1 = times[g][sm][i]
                if start and end:
                    node = tasks[tid]["node"]
                    per.setdefault(node, []).append((start, end))
                    if names[node] not in ("topk_route", "moe_experts") and start <= st0 <= end:
                        # stage stamp 0 a time inside the task: gemm_tile its activation in LDS, allreduce_send its
                        # producer's counter done, sum_rmsnorm its rows landed and summed; stamp 1 when it is a time
                        # inside the task too (gemm_tile with the RMSNorm folded: its R poll's start)
                        landed.setdefault(node, []).append((st0, end, st1 if start <= st1 <= end else 0))
                    if names[node] == "topk_route":   # stage stamps: 0 from the K parts landing (10 ns units, packed): keys
                        # made, top K picked, pass end; 1 wave 3's pick: shader cycles | ns << 32
                        cyc, ns = st1 & 0xFFFFFFFF, st1 >> 32
                        topk.append((round((st0 & 0xFFFF) / 100, 1), round(((st0 >> 16) & 0xFFFF) / 100, 1),
                                     round(((st0 >> 32) & 0xFFFF) / 100, 1), round(ns / 1e3, 2), round(cyc / max(ns, 1), 2)))
                    if names[node] == "moe_experts":   # stage stamps: W13 / W2 item counts and W13 us (packed), the W13 end
                        w13end = (start & ~((1 << 44) - 1)) | (st1 & ((1 << 44) - 1))   # its low 44 bits: ns since boot
                        if w13end < start:                                                  # mod 2^44 (~4.9 h)
                            w13end += 1 << 44
                        moe.append(((stamps[sm, lay["stamp_task0"]] - start) / 1e3, st0 & 0xFFFF, (w13end - start) / 1e3,
                                    (end - start) / 1e3, (start - starts[g]) / 1e3, (st0 >> 16) & 0xFFFF, st0 >> 32,
                                    (st1 >> 44) & 0x3FF, (st1 >> 54) & 0x3FF, (end - w13end) / 1e3))
        print(f"GPU {g}: per node: tasks, median / longest task us, first start .. last end us after the start barrier", flush=True)
        for node in sorted(per):
            se = per[node]
            d = sorted((e - b) / 1e3 for b, e in se)
            first, last = (min(b for b, _ in se) - starts[g]) / 1e3, (max(e for _, e in se) - starts[g]) / 1e3
            print(f"  {names[node]}#{node}: {len(se)} tasks, {d[len(d) // 2]:.1f} / {d[-1]:.1f} us, {first:.1f} .. {last:.1f}",
                  flush=True)
            if node in landed:   # input landed (stage stamp 0) after the start barrier, and from it to the task end
                la = sorted((b - starts[g]) / 1e3 for b, _, _ in landed[node])
                le = sorted((e - b) / 1e3 for b, e, _ in landed[node])
                print(f"    input landed {la[len(la) // 2]:.1f} (median) .. {la[-1]:.1f} (last); landed -> task end "
                      f"{le[len(le) // 2]:.1f} / {le[-1]:.1f} us ({len(la)} tasks)", flush=True)
                ps = sorted((p - starts[g]) / 1e3 for _, _, p in landed[node] if p)
                if ps:
                    print(f"    stage stamp 1 (gemm_tile fold: its R poll's start; the recurrence: token loop done; "
                          f"attn_residual: scores done) {ps[len(ps) // 2]:.1f} (median) .. {ps[-1]:.1f} (last)", flush=True)
        if topk:
            print(f"  topk_route per task (us after its K parts landed: keys made, top K picked, pass end; the pick: us, "
                  f"shader GHz): "
                  f"{topk}", flush=True)
        if moe:
            cols = list(zip(*moe))
            med = lambda c: sorted(c)[len(c) // 2]
            print(f"  moe_experts per CU (median): task start {med(cols[4]):.1f} us after the start barrier; after the task start: "
                  f"z_q in LDS {med(cols[0]):.1f} us, W13 items left {med(cols[2]):.1f} us, task end {med(cols[3]):.1f} us. "
                  f"Latest W13 end {max(c[4] + c[2] for c in moe):.1f}, latest task end {max(c[4] + c[3] for c in moe):.1f} us "
                  f"after the start barrier", flush=True)
            print(f"  moe_experts W2 per CU (median, us summed over its 4 waves): h_q wait {med(cols[7])}, item takes {med(cols[8])}; "
                  f"last W13 wave end -> task end {med(cols[9]):.1f} us", flush=True)
            n13s, n2s = Counter(cols[1]), Counter(cols[5])
            tot13 = sum(cols[1])
            print(f"  moe_experts items per CU: W13 {dict(sorted(n13s.items()))} (CUs per count; {tot13} items, mean item "
                  f"{sum(cols[6]) / max(tot13, 1):.1f} us), W2 {dict(sorted(n2s.items()))}", flush=True)


def read_curve(h, paths: list, nodes, w13_items: int, gpu: int = 0, bin_us: float = 2.0, dump_dir: str = None) -> None:
    """Timing build: the HBM bytes the layer's weight reads put in flight per microsecond, from the moe_experts item trace (each
    W13 / W2 item's start and end per wave) and the GEMM tasks' times (each task's weight bytes spread evenly over the task).
    Printed per bin_us: TB/s of moe weights, of GEMM weights, the waves in a W13 item, in a W2 item."""
    names = {n.graph_idx: n.name for n in nodes}
    lay = h.layout
    start = [b for b, _ in h.timing()][gpu]
    try:   # the trace in moe_experts' misc scratch, after the queues and the W13 part counters (moe_experts.cuh MISC_TRACE)
        misc = h._array(gpu, "moe_experts.misc", np.uint8)
        off = 16 * 128 + 8192
        tr = misc[off:].view(np.uint64).reshape(-1, 32, 2)
    except Exception as e:   # no trace buffer (CUDA, or an older build)
        print(f"  read curve: no moe trace ({e})", flush=True)
        return
    with open(paths[gpu]) as f:
        sched = json.load(f)
    T1 = 4 * 28 * 1024 + 4 * 7 * 256          # W13 item: 4 row tiles x 28 K steps of 1 KB, scales
    T2 = 8 * 3 * 1024 + 8 * 3 * 64            # W2 item: 8 row tiles x 3 K steps of 1 KB, scales
    spans = []                                 # (t0 us, t1 us, bytes, kind)
    for w in range(tr.shape[0]):
        for i in range(32):
            t0, d = int(tr[w, i, 0]), int(tr[w, i, 1])
            if not t0:
                break
            kind = (d >> 32) & 0xFF
            dur = d & 0xFFFFFFFF
            if kind == 4:   # half a W13 item (the removed tail split; old traces): counted as W13
                spans.append(((t0 - start) / 1e3, (t0 + dur - start) / 1e3, T1 // 2, 1))
                continue
            if kind not in (1, 2):   # the clock (8, 9) and start-phase (10, 11) entries move no weights
                continue
            spans.append(((t0 - start) / 1e3, (t0 + dur - start) / 1e3, T1 if kind == 1 else T2, kind))
    times = h.task_times()[gpu]
    for sm, queue in enumerate(sched["worker_task_queues"]):
        for i, tid in enumerate(queue):
            t0, t1 = times[sm][i][0], times[sm][i][1]
            task = sched["all_tasks"][tid]
            node = task["node"]
            if not (t0 and t1) or names[node] != "gemm_tile":
                continue
            nd = sched["nodes"][str(node)]
            K, N = nd["params"][1], nd["params"][2]
            gx, gy = nd["grid"][0], nd["grid"][1]
            spans.append(((t0 - start) / 1e3, (t1 - start) / 1e3, (N // gx) * (K // gy) * 2, 3))
    if not spans:
        return
    if dump_dir:   # the raw trace, the task times and the schedule, for analysis off the box
        os.makedirs(dump_dir, exist_ok=True)
        np.save(os.path.join(dump_dir, f"moe_trace_gpu{gpu}.npy"), tr)
        np.save(os.path.join(dump_dir, f"task_times_gpu{gpu}.npy"), np.array(times, dtype=np.uint64))
        with open(os.path.join(dump_dir, f"schedule_gpu{gpu}.json"), "w") as f:
            json.dump(sched, f)
        with open(os.path.join(dump_dir, f"start_gpu{gpu}.txt"), "w") as f:
            f.write(str(start))
    end = max(t1 for _, t1, _, _ in spans)
    nb = int(end / bin_us) + 1
    by = np.zeros((4, nb))
    act = np.zeros((4, nb))
    for t0, t1, b, k in spans:
        t0 = max(t0, 0.0)
        if t1 <= t0:
            continue
        rate = b / (t1 - t0)
        for j in range(int(t0 / bin_us), min(int(t1 / bin_us) + 1, nb)):
            lo, hi = max(t0, j * bin_us), min(t1, (j + 1) * bin_us)
            if hi > lo:
                by[k, j] += rate * (hi - lo)
                act[k, j] += (hi - lo) / bin_us
    print(f"  read curve GPU {gpu} (per {bin_us:g} us after the start barrier: TB/s moe W13 + W2 | GEMM weights; waves in W13 / W2 "
          f"items; GEMM tasks running):", flush=True)
    for j in range(nb):
        moe, gem = (by[1, j] + by[2, j]) / (bin_us * 1e6), by[3, j] / (bin_us * 1e6)
        print(f"    {j * bin_us:5.0f} us: {moe + gem:4.2f} TB/s = moe {moe:4.2f} + gemm {gem:4.2f}; W13 {act[1, j]:5.0f} W2 {act[2, j]:5.0f} "
              f"gemm {act[3, j]:4.0f}", flush=True)


def self_test():
    """The tile-layout dequantization (tiles_to_values, scales_to_matrix) vs amd/tools/shuffle_weights.py on random matrices."""
    from shuffle_weights import dequant as np_dequant, shuffle
    rng = np.random.default_rng(3)
    for n, k in ((64, 512), (64, 384), (128, 3584)):
        wq = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        s = rng.integers(118, 136, (n, k // 32), dtype=np.uint8)
        tiles, sc = shuffle(wq, s, "lo")
        got = dequant(torch.from_numpy(tiles.copy()), torch.from_numpy(sc.copy()), n, k).numpy()
        ref = np_dequant(wq, s)
        assert np.array_equal(got, ref), (n, k)
        print(f"N {n} K {k}: the gfx950 tiles dequantized == the logical MXFP4 matrix")
    v = torch.tensor([0.1, -3.0, 448.0, 1000.0] * 8)
    print("mxfp8 sample:", mxfp8(v)[:4].tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=int, default=1, choices=(1, 2, 4, 8))
    p.add_argument("--solo", action="store_true", help="the layer for 8 GPUs (TP8, the plan as given), GPU 0's share run alone on "
                   "one GPU, its waits for other GPUs skipped (STATIC_SOLO): TP8 times without the communication; checks only "
                   "the steps that need no other GPU")
    p.add_argument("--lup-push", default=None, help="GX,GY,GZ: latent_up with one K part pushing its bf16 rows to every "
                   "GPU (gemm_tile COMBINE_PUSH, MPK_LATENT_UP_PUSH), at this grid; sum_send dropped from the plan")
    p.add_argument("--zq-finish", action="store_true", help="latent_down's last K part per 128 rows quantizes and sends z_q "
                   "(gemm_tile COMBINE_FINISH_Q, MPK_ZQ_FINISH): no sum_quant_send")
    p.add_argument("--agg", default=None, metavar="NB,WRITE[,KIND]", help="the graph ends with the next layer's aggregation 1 "
                   "over NB bank rows, WRITE 1: a block-write layer; KIND (MPK_MOE_AGG) 1: tasks/moe_attn_residual.cuh "
                   "(7 chunk tasks per token, the default), 2: tasks/moe_agg1.cuh (one task per token)")
    p.add_argument("--fold", action="store_true", help="RMSNorm folded into latent_up (gemm_tile norm, MPK_RMSNORM_FOLD): "
                   "no sum_rmsnorm, no sum_send; latent_up's finisher scales by rstd")
    p.add_argument("--lup-finish", action="store_true", help="latent_up's last K part per row block adds the parts and pushes "
                   "them (gemm_tile COMBINE_FINISH, MPK_LATENT_UP_FINISH); sum_send dropped from the plan")
    p.add_argument("--shared-early", action="store_true", help="the shared expert (gate_up, situ_and_mul, down) first on the "
                   "CUs whose list starts with moe_experts (plan_one_gpu.shared_early_plan)")
    p.add_argument("--shared-late", action="store_true", help="the shared expert's situ_and_mul, down and its send after "
                   "moe_experts' R send, on CUs without the R path's later tasks (plan_one_gpu.shared_late_plan)")
    p.add_argument("--front", default=None, help="GX,GY,GZ: plan_one_gpu.front_plan with the router at this grid")
    p.add_argument("--apart", action="append", default=[],
                   help="N,A1,A2..: node N's tasks one per CU, on CUs without tasks of the nodes A* (plan_one_gpu.apart_plan); repeatable")
    p.add_argument("--pack", action="append", default=[],
                   help="NODE:PER_CU: the node's tasks PER_CU in a row on fewer CUs (plan_one_gpu.pack_plan); repeatable")
    p.add_argument("--follow", action="append", default=[],
                   help="NODE:LEAD[:AFTER,...[:AVOID,...]]: the node's tasks dealt over LEAD's CUs (not those holding an AVOID "
                        "node's task), each right after LEAD (and AFTER) there (plan_one_gpu.follow_plan; after the regrids); "
                        "repeatable")
    p.add_argument("--regrid", action="append", default=[],
                   help="NODE:GX,GY,GZ:AFTER,...: the plan with one node regridded, its tasks right after those nodes "
                        "(plan_one_gpu.regrid_plan); repeatable")
    p.add_argument("--plan", default=os.path.join(ROOT, "demo", "kimi_k3_moe", "solver_plan_256cu.json"))
    p.add_argument("--out", default="layer_one_gpu")
    p.add_argument("--flag", action="append", default=[], help="extra hipcc flag (repeatable)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--moe-in", default=None, help="x / prefix per GPU from test_kda_moe_layer.py --save-moe-in")
    p.add_argument("--time-wrong", action="store_true", help="a timing probe whose outputs are wrong by design (e.g. "
                   "--zero-z-from): the e2e launches run although the checks failed")
    p.add_argument("--zero-z-from", type=float, default=None, help="a timing probe: latent_down_weight rows from this "
                   "fraction of L on set to 0, so z there is 0 on every GPU (--solo's GPU 0 sees the other GPUs' z as 0)")
    p.add_argument("--dump-stamps", action="store_true", help="the timed launches' per-CU stamps to <out>/stamps_timed.npy")
    p.add_argument("--token-share", type=float, default=0.6, help="the tokens' common share (random_weights): 0.6 ~ 88 experts")
    p.add_argument("--zero-rows", type=int, default=0, help="the last N token rows of x and the prefix zeros (as the server's "
                   "padded rows of a prefill chunk)")
    p.add_argument("--wait", type=float, default=60)
    p.add_argument("--profile", action="store_true", help="the timing build (where_stuck on a launch that does not finish; "
                   "per-node times of launch 1)")
    p.add_argument("--flush-mb", type=int, default=0, help="MB read before each launch (1024: the 256 MB Infinity Cache and the "
                   "L2s hold none of the weights, as in a real model)")
    p.add_argument("--record", default=None, metavar="COSTS", help="timing build: cost records of --reps launches appended here")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--e2e-reps", type=int, default=20, help="after the checks: this many launches, e2e and span medians (0: none)")
    p.add_argument("--search", nargs="+", default=None, metavar="COSTS", help="the plan: the search's, from these records")
    p.add_argument("--time", type=float, default=120, help="--search: the solver's time limit (s)")
    p.add_argument("--grids", default=None, help='JSON {graph_idx: grid} of nodes to regrid; the plan is then a simple one '
                   '(plan_one_gpu.simple_plan: each node\'s tasks dealt round-robin over the CUs) instead of --plan, e.g. to record '
                   'costs of other grids for --search')
    p.add_argument("--spread", default=None, help="comma-separated graph_idx: plan_one_gpu.spread_plan of --plan with --grids")
    p.add_argument("--phases", default=None, help="JSON [[graph_idx, ...], ...]: a hand plan (plan_one_gpu.phase_plan) with "
                   "--grids over the --plan's grids")
    p.add_argument("--gemm-shuffled", action="store_true", help="the bf16 GEMM weights in gemm_tile's shuffled layout (HIP)")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--save-outs", default=None, help="launch 1's moe_out of every GPU to this .pt (test_rank_lib.py compares)")
    p.add_argument("--codegen", action="store_true", help="no GPU: the graph and plan as given, layer.cu (<out>/layer.cu) and a "
                   "hipcc compile only (for amd/tools/gen_rank_lib.py: the server library of exactly this plan)")
    p.add_argument("--dump-y", default=None, help="after launch 0: GPU 0's y and routing pairs (torch.save) for an A/B of the "
                   "outputs (amd/bench/sglang_moe_layer.py --compare-y)")
    a = p.parse_args()
    FLAGS[:] = a.flag
    if a.record:
        a.profile = True
    if a.self_test:
        self_test()
        return
    if a.codegen and not torch.cuda.is_available():   # as test_kda_layer.py --codegen: CPU tensors, an MI355X's props
        for fn in ("empty", "zeros", "ones", "full", "arange", "tensor", "empty_like", "zeros_like", "randn", "randint", "cat"):
            def on_cpu(*args, _orig=getattr(torch, fn), **kw):
                if str(kw.get("device", "")).startswith("cuda"):
                    kw["device"] = "cpu"
                return _orig(*args, **kw)
            setattr(torch, fn, on_cpu)

        class Props:
            major, minor, multi_processor_count, gcnArchName = 9, 5, 256, "gfx950"
            name, total_memory = "no GPU (codegen check)", 288 << 30
        torch.cuda.get_device_properties = lambda *_: Props
        torch.cuda.get_device_name = lambda *_: Props.name
        torch.cuda.current_device = lambda: 0
        torch.cuda.device_count = lambda: 8
        torch.cuda.set_device = lambda *_: None
        _gen = torch.Generator
        torch.Generator = lambda device=None: _gen()
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import fewer_gpus_plan
    d = load_demo()
    if a.solo:
        a.gpus = 8
    run_gpus = 1 if a.solo else a.gpus
    assert torch.cuda.device_count() >= run_gpus, f"{torch.cuda.device_count()} GPUs visible, {run_gpus} needed"
    torch.cuda.set_device(0)
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, "CUs x",
          a.gpus, flush=True)
    if a.agg:
        f = [int(v) for v in a.agg.split(",")]
        AGG[:] = f[:2]
        os.environ["MPK_MOE_AGG"] = str(f[2] if len(f) > 2 else 1)
    ws = [random_weights(d, torch.device("cpu") if a.codegen else torch.device("cuda", 0), a.seed, token_share=a.token_share)]
    if not a.codegen:   # (codegen: one GPU's tensors, shapes only)
        ws += [random_weights(d, torch.device("cuda", g), a.seed + g, common=ws[0]) for g in range(1, run_gpus)]
    if a.zero_rows:
        for w in ws:
            w["moe_in"][-a.zero_rows:].zero_()
            w["moe_prefix"][-a.zero_rows:].zero_()
    canaries = []
    if os.environ.get("MPK_CANARY") == "1" and not a.codegen:   # (guard rows after R and S: a write past either shows)
        for w in ws:
            for k in ("routed_sum_partial", "shared_down_partial"):
                t = w[k]
                big = torch.full((t.shape[0] + 256,) + tuple(t.shape[1:]), -12345.0, dtype=t.dtype, device=t.device)
                big[:t.shape[0]].copy_(t)
                w[k] = big[:t.shape[0]]
                canaries.append((k, big, t.shape[0]))
    if a.moe_in:   # x / prefix saved by test_kda_moe_layer.py --save-moe-in (the same routing as that fused run)
        for w, src in zip(ws, torch.load(a.moe_in)):
            w["moe_in"].copy_(src["moe_in"].to(w["moe_in"].device))
            w["moe_prefix"].copy_(src["moe_prefix"].to(w["moe_prefix"].device))
    if a.zero_z_from is not None:   # z's columns from this fraction of L on are 0 on every GPU (as GPU 0's z in --solo)
        for w in ws:
            w["latent_down_weight"][int(a.zero_z_from * d.L):] = 0
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=a.gpus)
    if a.gemm_shuffled:
        from mirage.mpk.static_megakernel import shuffle_gemm_weight
        mpk.gemm_weight_layout = "shuffled"
        for w in ws:
            for name in GEMM_WEIGHTS:
                sh = shuffle_gemm_weight(w[name])
                ROWMAJOR[sh.data_ptr()] = w[name]
                w[name] = sh
    if a.lup_push:   # latent_up pushes its rows (bf16) to every GPU itself: no sum_send (demo.build_graph)
        os.environ["MPK_LATENT_UP_PUSH"] = "1"
    if a.lup_finish:
        os.environ["MPK_LATENT_UP_FINISH"] = "1"
    if a.fold:
        os.environ["MPK_RMSNORM_FOLD"] = "1"
    if a.zq_finish:
        os.environ["MPK_ZQ_FINISH"] = "1"
    d.build_graph(mpk, ws[0])
    if a.grids or a.phases or a.spread:
        from plan_one_gpu import phase_plan, simple_plan, tokens_grids
        base = compiler.plan_from_file(a.plan)["grids"]
        if d.T != 8:   # (the plan file's grids are for 8 tokens per step: the token axes to this graph's count)
            base = tokens_grids(mpk, base, 8)
        base.update({int(k): tuple(v) for k, v in json.loads(a.grids or "{}").items()})
        if a.spread:
            from plan_one_gpu import spread_plan
            plan = spread_plan(mpk, compiler.plan_from_file(a.plan), json.loads(a.grids or "{}"),
                               [int(i) for i in a.spread.split(",")])
            print(f"spread plan: nodes {a.spread} first, grids {plan['grids']}", flush=True)
        elif a.phases:
            plan = phase_plan(mpk, base, json.loads(a.phases))
            if not a.solo:
                plan = fewer_gpus_plan(mpk, plan) if a.gpus < 8 else plan
            print(f"phase plan {a.phases} with grids {plan['grids']}", flush=True)
        else:
            plan = simple_plan(mpk, base)
            print(f"simple plan with grids {plan['grids']}", flush=True)
    else:
        plan = compiler.plan_from_file(a.plan)
        # the plan file is for the graph with sum_rmsnorm and sum_send: drop them first (plan_one_gpu.fold_plan,
        # lup_finish_plan), then fit the plan to this graph's tasks per GPU
        drop = set()
        if a.fold:
            drop |= {38, 41}
            print("rmsnorm fold: sum_rmsnorm and sum_send dropped", flush=True)
        if a.zq_finish:
            drop |= {29}
            print("z_q finish: sum_quant_send dropped", flush=True)
        if drop:
            from plan_one_gpu import drop_plan
            plan = drop_plan(plan, drop)
        if os.environ.get("MPK_MOE_INPUT_WAIT") in ("1", "2"):   # two input_wait nodes before the router: 26 (x) on CUs 0-7, 27
            from plan_one_gpu import insert_plan                    # (prefix) on CUs 8-15, first; the others renumbered
            gn = compiler.graph_nodes(mpk)
            iw = [n.graph_idx for n in gn if n.name == "input_wait"]
            k = min(n.graph_idx for n in gn if n.name == "gemm_tile") - 26
            plan = insert_plan(plan, 26, k, {iw[0]: ((8, 1, 1), [(None, t, (t, 0, 0)) for t in range(8)]),
                                             iw[1]: ((8, 1, 1), [(None, 8 + t, (t, 0, 0)) for t in range(8)])})
            print(f"input_wait: nodes {iw} first on CUs 0-15; the later nodes + {k}", flush=True)
        if a.lup_finish:
            from plan_one_gpu import lup_finish_plan
            plan = lup_finish_plan(plan)
            print("latent_up finish: sum_send dropped", flush=True)
        if d.T != 8:   # (the plan files are for 8 tokens per step: plan_one_gpu.tokens_plan re-deals the per-token nodes)
            from plan_one_gpu import tokens_plan
            plan = tokens_plan(mpk, plan, 8)
        plan = plan if a.solo else fewer_gpus_plan(mpk, plan)
    if a.lup_push:   # the plan of the graph with sum_send: plan_one_gpu.lup_push_plan
        from plan_one_gpu import lup_push_plan
        plan = lup_push_plan(mpk, plan, tuple(int(v) for v in a.lup_push.split(",")))
        print(f"latent_up push: grid {plan['grids'][40]}, sum_send dropped", flush=True)
    if a.front:   # ROUTER_GRID (GX,GY,GZ): plan_one_gpu.front_plan (router 26 first, latent_down 28, gate_up 30)
        from plan_one_gpu import front_plan
        plan = front_plan(mpk, plan, 26, tuple(int(v) for v in a.front.split(",")), 28, 30)
        print(f"front plan: router at {plan['grids'][26]}, first on its CUs; gate_up after it", flush=True)
    if a.shared_early:
        from plan_one_gpu import shared_early_plan
        plan = shared_early_plan(mpk, plan)
        print("shared early: gate_up, situ_and_mul, down first on the CUs that start with moe_experts", flush=True)
    if a.shared_late:
        from plan_one_gpu import shared_late_plan
        plan = shared_late_plan(mpk, plan)
        print("shared late: situ_and_mul, down and its send after moe_experts' R send", flush=True)
    for pc in a.pack:   # NODE:PER_CU (plan_one_gpu.pack_plan)
        from plan_one_gpu import pack_plan
        f = pc.split(":")
        plan = pack_plan(plan, int(f[0]), int(f[1]))
        print(f"pack: node {f[0]}, {f[1]} tasks per CU", flush=True)
    for rg in a.regrid:   # NODE:GX,GY,GZ:AFTER1,AFTER2,... (plan_one_gpu.regrid_plan)
        from plan_one_gpu import regrid_plan
        f = rg.split(":")
        plan = regrid_plan(mpk, plan, int(f[0]), tuple(int(v) for v in f[1].split(",")),
                           {int(v) for v in f[2].split(",") if v} if len(f) > 2 else set())
        print(f"regrid: node {f[0]} at grid {plan['grids'][int(f[0])]}, after nodes {f[2] if len(f) > 2 else '-'}", flush=True)
    for ap in a.apart:   # N,A1,A2..: node N's tasks one per CU, on CUs without tasks of the nodes A* (plan_one_gpu.apart_plan)
        from plan_one_gpu import apart_plan
        f = [int(v) for v in ap.split(",")]
        plan = apart_plan(plan, f[0], set(f[1:]))
        print(f"apart: node {f[0]} off the CUs of nodes {f[1:]}", flush=True)
    for fo in a.follow:   # NODE:LEAD[:AFTER,...[:AVOID,...]] (plan_one_gpu.follow_plan; after --apart)
        from plan_one_gpu import follow_plan
        f = fo.split(":")
        plan = follow_plan(plan, int(f[0]), int(f[1]), {int(v) for v in f[2].split(",")} if len(f) > 2 and f[2] else set(),
                           {int(v) for v in f[3].split(",")} if len(f) > 3 and f[3] else set())
        print(f"follow: node {f[0]} over node {f[1]}'s CUs", flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.search:
        from mirage.mpk import plan_viewer, search, static_schedule
        costs = search.Costs(a.search, code=static_schedule.code_tag(), gpu=torch.cuda.get_device_name(0))
        plan, summary = search.search_plan(mpk, costs, time_limit_s=a.time, start_from=plan)
        compiler.write_plan_file(os.path.join(a.out, "plan.json"), plan)
        plan_viewer.save(os.path.join(a.out, "plan_view.json"), plan, summary, f"searched from {a.plan}")
        print(f"search: {summary.get('status')}, predicted {summary.get('predicted_us')} us", flush=True)
    compiler.write_plan_file(os.path.join(a.out, f"plan_{a.gpus}gpu.json"), plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    if a.codegen:
        import re
        import subprocess
        from mirage.mpk import static_schedule
        code = static_schedule.generate_code(mpk, [json.load(open(q)) for q in paths], a.profile)
        cu = os.path.join(a.out, "layer.cu")
        with open(cu, "w") as f:
            f.write(code)
        cmd = static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["--save-temps"], arch="gfx950")
        r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
        print(f"layer.cu {len(code.splitlines())} lines; hipcc exit {r.returncode}", flush=True)
        if r.returncode:
            print("\n".join(l for l in r.stderr.splitlines() if "error" in l)[:6000])
        else:
            sasm = open(os.path.join(a.out, "layer-hip-amdgcn-amd-amdhsa-gfx950.s")).read()
            print({k: re.search(rf"; {k}: (\d+)", sasm).group(1) for k in ("NumVgprs", "NumAgprs", "ScratchSize", "Occupancy")})
        return
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag,
                            profile=a.profile, solo=a.solo)
    h = Harness(sk)
    outs = ("routing_pairs", "shared_gate_up", "shared_act") + (() if a.fold else ("routed_normed",)) + (
        "routed_sum_partial", "shared_down_partial", "moe_out") + (("agg_out", "agg_bank") if a.agg else ())
    bank0 = ws[0]["agg_bank"].clone() if a.agg else None
    cmp_outs = outs
    runs = []
    # (a launch must not write the tensors that are not its outputs: GPU 0's, compared after launch 0)
    pre = {k: v.clone() for k, v in ws[0].items() if k not in outs and isinstance(v, torch.Tensor)}
    for launch in range(2):
        try:
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
        except (TimeoutError, RuntimeError) as e:
            print(f"KERNEL DID NOT FINISH (launch {launch}): {e}", flush=True)
            if a.profile:
                where_stuck(h, paths, compiler.graph_nodes(mpk))
            # the checks anyway, on the CPU: the tensors copied on a side stream (the layer's stream is still busy, and
            # the default stream would wait for it); the steps after the one that hangs check whatever is there
            side = [torch.cuda.Stream(device=g) for g in range(run_gpus)]
            ws_cpu = []
            for g, w in enumerate(ws):
                with torch.cuda.stream(side[g]):
                    ws_cpu.append({k: v.to("cpu") for k, v in w.items()})
            rep = Report()
            try:
                check_all(d, ws_cpu, rep, fold=a.fold)
            except Exception as e:   # a step on the hung layer's leftovers
                print(f"checks stopped: {type(e).__name__}: {e}", flush=True)
            os._exit(2)
        for g in range(run_gpus):
            torch.cuda.synchronize(g)
        runs.append([{n: w[n].clone() for n in outs} for w in ws])
        print(f"launch {launch}: e2e {h.e2e_us():.1f} us, span {h.span_us():.1f} us", flush=True)
        if launch == 1:   # (a named read returns the arena's copy 0: the copy launch 1 used)
            try:   # (the router's K parts as launch 1 left them, for check_all's "K parts: router_logits")
                import numpy as np
                raw = np.frombuffer(h.sk.read(0, "router_logits"), dtype=np.float32).copy()
                ws[0]["router_logits"] = torch.from_numpy(raw).to(ws[0]["moe_in"].device)
            except Exception as e:  # noqa
                print(f"  (router_logits not readable: {e})", flush=True)
        if launch == 0:
            for k, big, rows in canaries:
                tail = big[rows:]
                bad = (tail != -12345.0).reshape(tail.shape[0], -1).any(1).nonzero().flatten().tolist()
                print(f"  canary after {k}: {'untouched' if not bad else f'WRITTEN rows +{bad[:12]}'} ({big.device})", flush=True)
            for k, v0 in pre.items():
                v1 = ws[0][k]
                if not torch.equal(v0.view(torch.uint8) if v0.dtype != torch.uint8 else v0,
                                   v1.view(torch.uint8) if v1.dtype != torch.uint8 else v1):
                    rows = (v0 != v1).reshape(v0.shape[0], -1).any(1).nonzero().flatten().tolist() if v0.dim() >= 1 else []
                    print(f"  WRITTEN by the launch (not an output): {k} {tuple(v0.shape)} {v0.dtype}, rows {rows[:20]}"
                          f"{' ..' if len(rows) > 20 else ''}", flush=True)
        if a.profile and launch == 1:
            node_times(h, paths, compiler.graph_nodes(mpk))
            read_curve(h, paths, compiler.graph_nodes(mpk), d.T * d.TOPK * (d.IR_LOCAL // 32),
                       dump_dir=os.path.join(a.out, "trace"))
            dd = os.path.join(a.out, "trace")   # every GPU's stamps, start reference and task times, for analysis off the box
            os.makedirs(dd, exist_ok=True)
            lay = h.layout
            for g, (st, _) in enumerate(h.timing()):
                np.save(os.path.join(dd, f"stamps_gpu{g}.npy"),
                        h._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"]))
                np.save(os.path.join(dd, f"task_times_all_gpu{g}.npy"), np.array(h.task_times()[g], dtype=np.uint64))
                with open(os.path.join(dd, f"start_all_gpu{g}.txt"), "w") as f:
                    f.write(str(st))

    if a.dump_y:
        torch.save({"y": runs[0][0]["moe_out"].cpu(), "pairs": runs[0][0]["routing_pairs"].cpu(), "seed": a.seed,
                    "token_share": a.token_share}, a.dump_y)
        print(f"dumped y and the routing pairs of launch 0 to {a.dump_y}", flush=True)
    rep = Report()
    print("checks of launch 0:")
    check_all(d, [{**w, **r} for w, r in zip(ws, runs[0])], rep, solo=a.solo, fold=a.fold)
    if a.agg and not a.solo:
        check_agg({**ws[0], **runs[0][0]}, [{**w, **r} for w, r in zip(ws, runs[0])], rep, bank0)
    if a.solo:
        print("checks of launch 1:")
        check_all(d, [{**w, **r} for w, r in zip(ws, runs[1])], rep, solo=a.solo, fold=a.fold)
    for n in cmp_outs:
        rep.check(f"launch 1 == launch 0: {n}", all(same_bits([r0[n], r1[n]]) for r0, r1 in zip(runs[0], runs[1])),
                  f"bits, all {run_gpus} GPUs")
    print("ALL CHECKS PASSED" if not rep.failed else f"FAILED: {rep.failed}", flush=True)
    if a.save_outs:
        torch.save([{"moe_out": r["moe_out"].cpu(), **({"agg_out": r["agg_out"].cpu()} if a.agg else {})} for r in runs[1]],
                   a.save_outs)
        print("saved launch 1's outputs to", a.save_outs, flush=True)
    if a.e2e_reps and (not rep.failed or a.time_wrong):   # end to end (events right before / after the kernel, max over GPUs) and span, cold L2
        e2e, spans, per_gpu, parts, pre, dstamps = [], [], [], [], [], []
        lay = h.layout
        for i in range(a.e2e_reps):
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            per_gpu.append(sk.elapsed_us())
            e2e.append(max(per_gpu[-1]))
            spans.append(h.span_us())
            if a.dump_stamps:   # this launch's per-CU stamps, every GPU ([GPUs][CUs][START, TASK0, TASK1, END..] ns)
                dstamps.append(np.stack([h._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"])
                                         for g in range(run_gpus)]))
            for g in range(run_gpus):   # this GPU's own clock: launch -> first CU in, CUs entering, wait at the start barrier, run
                st = h._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"])
                entry, end = st[:, lay["stamp_start"]], st[:, lay["stamp_end"]].max()
                barrier = int(h._array(g, "start_barrier", np.int64)[0])
                first, last = entry[entry > 0].min(), entry[entry > 0].max()
                parts.append((per_gpu[-1][g] - (end - first) / 1e3, (last - first) / 1e3, (barrier - last) / 1e3,
                              (end - barrier) / 1e3))
                if lay["num_stamps"] >= 7:   # STATIC_PRE_STAMPS / timing build: [4] entry, [5] the reset barrier passed, [6] stores landed
                    ent, bar, sto = st[:, 4], st[:, 5], st[:, 6]
                    early = (ent > 0) & (ent < first)   # in before the go (not the one waiting for the line-up's CU)
                    if early.any():
                        e0 = ent[early].min()
                        q = lambda x: (np.median(x[early]) - e0) / 1e3
                        iss = st[:, 7] if lay["num_stamps"] >= 8 else sto
                        pre.append((g, (first - e0) / 1e3, q(ent), (ent[early].max() - e0) / 1e3, q(iss), q(sto),
                                    (sto[early].max() - e0) / 1e3, q(bar), (bar[early].max() - e0) / 1e3, (end - first) / 1e3))
        for g in range(run_gpus):
            torch.cuda.synchronize(g)
        if a.dump_stamps:
            os.makedirs(a.out, exist_ok=True)
            np.save(os.path.join(a.out, "stamps_timed.npy"), np.stack(dstamps))
            print(f"saved the timed launches' per-CU stamps to {os.path.join(a.out, 'stamps_timed.npy')}", flush=True)
        # the last launch's outputs, bit for bit those of launch 0 (the resets / re-arms across many launches)
        last = [{n: w[n].clone() for n in outs} for w in ws]
        for n in cmp_outs:
            rep.check(f"launch {a.e2e_reps + 1} == launch 0: {n}",
                      all(same_bits([r0[n], r1[n]]) for r0, r1 in zip(runs[0], last)), f"bits, all {run_gpus} GPUs")
        print("CHECK OF THE LAST TIMED LAUNCH " + ("PASSED" if not rep.failed else f"FAILED: {rep.failed}"), flush=True)
        e2e.sort()
        spans.sort()
        print(f"LATENCY ({a.e2e_reps} launches, {'solo TP8 on 1 GPU' if a.solo else f'{run_gpus} GPUs'}, "
              f"{'timing build' if a.profile else 'normal build'}, {a.flush_mb} MB flushed before each): e2e median "
              f"{e2e[len(e2e) // 2]:.1f} us (min {e2e[0]:.1f}, max {e2e[-1]:.1f}); span median {spans[len(spans) // 2]:.1f} us",
              flush=True)
        med = [sorted(c)[len(c) // 2] for c in zip(*parts)]
        print(f"  e2e split (medians over launches and GPUs, us): launch -> first CU in {med[0]:.1f}, first -> last CU in "
              f"{med[1]:.1f}, last CU in -> past the start barrier {med[2]:.1f}, barrier -> last CU end {med[3]:.1f}", flush=True)
        if run_gpus > 1:   # per GPU (in launch order: the host launches GPU 0 first): its e2e median
            med = [sorted(p[g] for p in per_gpu)[len(per_gpu) // 2] for g in range(run_gpus)]
            print(f"  e2e median per GPU (launch order 0..{run_gpus - 1}): {[round(m, 1) for m in med]}", flush=True)
        if pre:   # per GPU, medians over launches (us from its first CU's entry): the go, the CUs past the reset barrier
            for g in range(run_gpus):
                r = [p[1:] for p in pre if p[0] == g]
                if r:
                    m = [sorted(c)[len(c) // 2] for c in zip(*r)]
                    print(f"  GPU {g} before the go (us from its first CU in): first CU past the go {m[0]:.1f}; CUs in median "
                          f"{m[1]:.1f}, last {m[2]:.1f}; stores issued median {m[3]:.1f}; landed median {m[4]:.1f}, last {m[5]:.1f}; "
                          f"past the reset barrier median {m[6]:.1f}, last {m[7]:.1f}; go -> last CU end {m[8]:.1f}", flush=True)
    if a.record and not rep.failed:
        import socket
        from mirage.mpk import cost_records
        launches = []
        for i in range(a.reps):
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            launches.append((h.task_times(), h.timing()))
        spans = sorted(h2 - h1 for h1, h2 in (t[0] for _, t in launches))
        machine = {"gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
                   "hip": str(torch.version.hip), "solo": bool(a.solo)}
        records = cost_records.records_of_run(mpk, paths[:run_gpus], launches, os.path.basename(os.path.normpath(a.out)),
                                              machine)
        cost_records.append_records(a.record, records)
        print(f"cost records: {len(records)} added to {a.record} ({a.reps} launches, span median "
              f"{spans[len(spans) // 2] / 1e3:.1f} us; code {records[0]['code'] if records else '-'})", flush=True)
    sk.finalize()
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
