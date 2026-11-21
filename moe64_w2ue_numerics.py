"""CPU checks of STATIC_MOE64_W2UE's numerics (tasks/scaled_moe_experts.cuh, fp8_fp32scale_weights.requant_ue8m0).

  1. requant_ue8m0 against SGLang's own code (fp8_utils.py's block_quant_dequant, per_block_cast_to_fp8 and
     ceil_to_ue8m0, read from --sglang: requant_weight_ue8m0's 2D path without the DeepGEMM scale layout), bit for bit:
     random e4m3 weights with random fp32 block scales (some blocks tiny: amax under the 1e-4 clamp), and real GLM-5.3
     down shards (--model, layer --layer: the TP8 rank 0 shard and the full weight; the shard of the full weight's
     requantization equals the shard's requantization). Negative control (must differ): the amax clamp at 1e-10.
  2. the UE8M0 bytes the issuer writes, eh + ew - 127 (h_q's and the weight's scale exponents): their range over real
     requantized down scales (--experts experts and the shared one) and the h_q scales of their h for x = N(0, 1) bf16
     (T = 64 rows), against the clamp [1, 254]; and the bound from the clamps alone (h: amax >= 1e-10, w: amax >= 1e-4).
  3. the issuer's chunk words and the block-scaled MMA, simulated: per W2 tile and K tile kt, lane l writes the 16 B at
     16 l of the chunk (rows l + 32 c, c = 0..3: the UE8M0 byte of the slot's place, 127 past its tokens); the MMA
     scales column n's K block j by byte j of the word at 16 (n % 32) + 4 (n / 32) (cp_sf 32x128b.warpx4's layout), the
     weight side by the constant chunk; both K tiles in one accumulator. Against the dequantized product (fp64, power-of-
     two scales: equal up to the order of the fp64 sums), for random slots of 1..64 tokens. Negative controls (must differ): row % 32 and row / 32
     swapped; the byte eh + ew - 126; the h_q scales of the token id instead of the place.
  4. the numerics change on real weights (the experts of 2.), routed rows of x = N(0, 1) bf16 (T = 64 rows) against the
     exact reference (bf16 x, fp64 h, the checkpoint's weights; relRMS): the fp32-scale kernel (old), W2UE (the requantized
     down, power-of-two h_q scales), and SGLang's DeepGEMM path (W13 and W2 requantized, power-of-two x and h scales).
Run: python moe64_w2ue_numerics.py [--model /path/GLM-5.3] [--layer 3] [--experts 8] [--sglang .../fp8_utils.py]
     (torch with float8_e4m3fn; exit 0: every check passes and every control differs)
"""
import argparse
import ast
import json
import os
import sys

import torch

import importlib.util  # noqa: E402

# python/mirage/mpk/fp8_fp32scale_weights.py by its path (the mirage package needs its native libraries)
_spec = importlib.util.spec_from_file_location(
    "fp8_fp32scale_weights",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "mirage", "mpk", "fp8_fp32scale_weights.py"))
_w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w)
dequant, requant_ue8m0 = _w.dequant, _w.requant_ue8m0

SGL = "/tmp/golden-x11k/base/python/sglang/srt/layers/quantization/fp8_utils.py"


def sglang_funcs(path):
    """block_quant_dequant, per_block_cast_to_fp8, ceil_to_ue8m0 from SGLang's source (exec of their definitions)."""
    src = open(path).read()
    keep = {"block_quant_dequant", "per_block_cast_to_fp8", "ceil_to_ue8m0"}
    mod = ast.Module(body=[n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in keep],
                     type_ignores=[])
    ns = {"torch": torch, "List": list, "Tuple": tuple, "ceil_align": lambda x, a: (x + a - 1) // a * a}
    exec(compile(mod, path, "exec"), ns)
    assert keep <= set(ns), f"{path}: missing {keep - set(ns)}"
    return ns


def sglang_requant(ns, w, s):
    """requant_weight_ue8m0's 2D path: dequantize to bf16, per_block_cast_to_fp8 (no DeepGEMM scale layout)."""
    v = ns["block_quant_dequant"](w.view(torch.float8_e4m3fn), s, [128, 128], torch.bfloat16)
    q, sf = ns["per_block_cast_to_fp8"](v)
    return q.view(torch.uint8), sf


def requant_1e10(w, s):
    """negative control: requant_ue8m0 with the amax clamped at 1e-10 instead of 1e-4"""
    N, K = w.shape
    v = dequant(w, s).to(torch.bfloat16).view(N // 128, 128, K // 128, 128)
    amax = v.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-10)
    bits = (amax / 448.0).view(torch.int32)
    e = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254)
    sf = (e << 23).view(torch.float32)
    return (v * (1.0 / sf)).to(torch.float8_e4m3fn).view(N, K).view(torch.uint8), sf.view(N // 128, K // 128)


def rand_weight(g, N, K, tiny=False):
    """random finite e4m3 bytes [N, K] and fp32 block scales; tiny: a quarter of the blocks with scales of 1e-9"""
    q = torch.randn(N, K, generator=g).mul(64).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    s = torch.rand(N // 128, K // 128, generator=g).mul(3e-3).add(1e-5)
    if tiny:
        s[torch.rand(s.shape, generator=g) < 0.25] = 1e-9
    return q, s


def ue_exp(v):
    """fp32 powers of two -> their UE8M0 bytes"""
    return ((v.float().view(torch.int32) >> 23) & 0xFF).long()


def e4m3_groups(v, ue8m0):
    """fp32 [n, K] -> (e4m3 [n, K], fp32 scales [n, K / 128]): max(amax, 1e-10) x fp32(1 / 448), or ue8m0: the power of two
    at or above max(amax, 1e-10) / 448 (group_scale<true>)"""
    n, K = v.shape
    g = v.float().view(n, K // 128, 128)
    am = g.abs().amax(-1).clamp_min(1e-10)
    if ue8m0:
        b = (am / 448.0).view(torch.int32)
        s = ((((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254) << 23).view(torch.float32)
    else:
        s = am * (torch.tensor(1.0) / torch.tensor(448.0))
    return (g / s.unsqueeze(-1)).to(torch.float8_e4m3fn).view(n, K), s


def fp8_gemm(q, s, w, ws):
    """e4m3 [n, K] x scales [n, K / 128] times e4m3 bytes [N, K] x block scales [N / 128, K / 128] -> fp64 [n, N]"""
    n, K = q.shape
    N, KB = w.shape[0], K // 128
    part = torch.einsum("tbk,nbk->tnb", q.double().view(n, KB, 128), w.view(torch.float8_e4m3fn).double().view(N, KB, 128))
    return (part * s.double().unsqueeze(1) * ws.double().repeat_interleave(128, 0)[:N].unsqueeze(0)).sum(-1)


# ---- 3. the issuer's chunks and the block-scaled MMA ----
def chunk_bytes(eh_place, ew, nt, ctl=None):
    """the issuer's chunk of one K tile (512 B): lane l stores words {row l, l + 32, l + 64, l + 96} at 16 l; row r < nt
    (r < 64: the code's two words ue_w[kt][c]) holds clamp(eh(r) + ew - 127) x 0x01010101, every other row 127s.
    ctl swap: row r at 16 (r / 32) + 4 (r % 32) instead (row % 32 and row / 32 swapped); bias126: eh + ew - 126"""
    out = bytearray(512)
    for lane in range(32):
        for c in range(4):
            r = lane + 32 * c
            b = 127
            if c < 2 and r < nt:
                b = min(max(eh_place[r] + ew - (126 if ctl == "bias126" else 127), 1), 254)
            off = 16 * (r // 32) + 4 * (r % 32) if ctl == "swap" else 16 * lane + 4 * c
            out[off:off + 4] = bytes([b] * 4)
    return out


def mma_scale(chunk, n, j):
    """the block-scaled MMA's UE8M0 scale of row / column n, K block j (scale id j): byte j of the word of row n"""
    return chunk[16 * (n % 32) + 4 * (n // 32) + j]


def check_mx(g, trials=300, ctl=None):
    """one W2 tile (128 output rows, K = 256) of a slot of nt tokens: the simulated MMA vs the dequantized product.
    Returns the max difference relative to the largest value (~1e-16 expected: the fp64 sums' order)"""
    worst = 0.0
    for _ in range(trials):
        nt = int(torch.randint(1, 65, (1,), generator=g))
        n = 16 * ((nt + 15) // 16)
        tok = torch.randperm(64, generator=g)[:nt].sort().values          # the slot's tokens in token order (place c)
        w = torch.randn(128, 256, generator=g).mul(40).clamp(-448, 448).to(torch.float8_e4m3fn).double()
        hq = torch.randn(64, 256, generator=g).mul(40).clamp(-448, 448).to(torch.float8_e4m3fn).double()  # rows by place
        eh_tok = torch.randint(100, 125, (64, 2), generator=g)            # [token][kt]
        ew = torch.randint(108, 120, (2,), generator=g)
        want = torch.zeros(128, n, dtype=torch.float64)
        got = torch.zeros(128, n, dtype=torch.float64)
        for kt in range(2):
            ks = slice(128 * kt, 128 * kt + 128)
            eh_place = [int(eh_tok[(c if ctl == "by_token" else int(tok[c])), kt]) for c in range(nt)]
            ch = chunk_bytes(eh_place, int(ew[kt]), nt, ctl)
            for j in range(4):   # the 4 MMAs of the K tile, 32 values each, scale id j
                kj = slice(128 * kt + 32 * j, 128 * kt + 32 * j + 32)
                p = w[:, kj] @ hq[:n, kj].t()
                sb = torch.tensor([2.0 ** (mma_scale(ch, c, j) - 127) for c in range(n)], dtype=torch.float64)
                got += p * sb.unsqueeze(0)   # the weight side: the constant chunk (2^0)
            sh = torch.tensor([2.0 ** (int(eh_tok[int(tok[c]), kt]) - 127) for c in range(nt)], dtype=torch.float64)
            want[:, :nt] += (w[:, ks] @ hq[:nt, ks].t()) * sh.unsqueeze(0) * 2.0 ** (int(ew[kt]) - 127)
        worst = max(worst, (got[:, :nt] - want[:, :nt]).abs().max().item() / want[:, :nt].abs().max().item())
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/mnt/shared/models/GLM-5.3")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--sglang", default=SGL)
    args = ap.parse_args()
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(5)
    ok = True

    # 1. requant_ue8m0 vs SGLang
    ns = sglang_funcs(args.sglang)
    for name, (w, s) in (("random", rand_weight(g, 512, 768)), ("random, tiny blocks", rand_weight(g, 512, 768, True))):
        q, sf = requant_ue8m0(w, s)
        q2, sf2 = sglang_requant(ns, w, s)
        eq = torch.equal(q, q2) and torch.equal(sf.view(torch.int32), sf2.view(torch.int32))
        q3, sf3 = requant_1e10(w, s)
        neg = torch.equal(q3, q2) and torch.equal(sf3.view(torch.int32), sf2.view(torch.int32))
        print(f"1. {name}: requant_ue8m0 bit-equal to SGLang's: {eq}; control amax clamp 1e-10 equal: {neg}")
        ok &= eq and (name == "random" or not neg)
    model = args.model if os.path.isfile(os.path.join(args.model, "model.safetensors.index.json")) else None
    if model is None:
        print(f"   {args.model}: no checkpoint, the real-weight checks skipped")
    else:
        from safetensors import safe_open
        with open(os.path.join(model, "model.safetensors.index.json")) as f:
            where = json.load(f)["weight_map"]
        p = f"model.layers.{args.layer}.mlp."

        def get(name, cols=None):
            with safe_open(os.path.join(model, where[p + name]), framework="pt") as f:
                return f.get_tensor(p + name) if cols is None else f.get_slice(p + name)[:, cols]

        names = [f"experts.{e}." for e in range(args.experts)] + ["shared_experts."]
        IR, rows, blk = 256, slice(0, 256), slice(0, 2)   # TP8 rank 0
        ew_all, eh_all, errs = [], [], {k: [] for k in ("old", "w2ue", "sglang", "w2ue_vs_old", "w2ue_vs_sglang")}
        x = torch.randn(64, 6144, generator=g).to(torch.bfloat16)
        for nm in names:
            d_full, ds_full = get(nm + "down_proj.weight").view(torch.uint8), get(nm + "down_proj.weight_scale_inv").float()
            d, ds = d_full[:, rows].contiguous(), ds_full[:, blk].contiguous()
            q, sf = requant_ue8m0(d, ds)
            q2, sf2 = sglang_requant(ns, d, ds)
            qf, sff = requant_ue8m0(d_full, ds_full)
            eq = torch.equal(q, q2) and torch.equal(sf, sf2)
            sh = torch.equal(qf[:, rows], q) and torch.equal(sff[:, blk], sf)
            ok &= eq and sh
            if nm == names[0] or not (eq and sh):
                print(f"1. {nm}down shard [6144, 256]: bit-equal to SGLang's {eq}; = the shard of the full weight's "
                      f"requantization {sh}")
            ew_all.append(ue_exp(sf).flatten())
            # 2. / 4. the expert's rows for x
            gt, gs = get(nm + "gate_proj.weight")[rows].view(torch.uint8), get(nm + "gate_proj.weight_scale_inv")[blk].float()
            ut, us = get(nm + "up_proj.weight")[rows].view(torch.uint8), get(nm + "up_proj.weight_scale_inv")[blk].float()
            xq, xs = e4m3_groups(x.float(), False)
            h = torch.nn.functional.silu(fp8_gemm(xq, xs, gt, gs).float()) * fp8_gemm(xq, xs, ut, us).float()
            hq_u, hs_u = e4m3_groups(h, True)
            eh_all.append(ue_exp(hs_u).flatten())
            hq_o, hs_o = e4m3_groups(h, False)
            old = fp8_gemm(hq_o, hs_o, d, ds)
            new = fp8_gemm(hq_u, hs_u, q, sf)
            # exact: bf16 x, fp64 h, the checkpoint's weights
            xe = x.double()
            he = torch.nn.functional.silu(xe @ dequant(gt, gs).double().t()) * (xe @ dequant(ut, us).double().t())
            ex = he @ dequant(d, ds).double().t()
            # SGLang's DeepGEMM path: W13 and W2 requantized, power-of-two x and h scales
            gq, gqs = requant_ue8m0(gt, gs)
            uq, uqs = requant_ue8m0(ut, us)
            xq2, xs2 = e4m3_groups(x.float(), True)
            h2 = torch.nn.functional.silu(fp8_gemm(xq2, xs2, gq, gqs).float()) * fp8_gemm(xq2, xs2, uq, uqs).float()
            sg = fp8_gemm(*e4m3_groups(h2, True), q, sf)
            rr = lambda a, b: ((a - b).norm() / b.norm()).item()
            errs["old"].append(rr(old, ex))
            errs["w2ue"].append(rr(new, ex))
            errs["sglang"].append(rr(sg, ex))
            errs["w2ue_vs_old"].append(rr(new, old))
            errs["w2ue_vs_sglang"].append(rr(new, sg))
        ew, eh = torch.cat(ew_all), torch.cat(eh_all)
        # 2. the UE8M0 bytes: every (token, K tile) h scale with every weight block scale of the same K tile
        lo, hi = int(eh.min() + ew.min() - 127), int(eh.max() + ew.max() - 127)
        print(f"2. real exponents ({len(names)} experts, layer {args.layer}, x = N(0, 1)): ew {int(ew.min())}..{int(ew.max())}, "
              f"eh {int(eh.min())}..{int(eh.max())}: eh + ew - 127 in {lo}..{hi} (clamp [1, 254]: "
              f"{'never' if 1 <= lo and hi <= 254 else 'REACHED'})")
        ok &= 1 <= lo and hi <= 254
        print("4. routed rows' relRMS (median over the experts) against the exact reference: fp32-scale kernel "
              f"{sorted(errs['old'])[len(names) // 2]:.2e}, W2UE {sorted(errs['w2ue'])[len(names) // 2]:.2e}, SGLang DeepGEMM "
              f"{sorted(errs['sglang'])[len(names) // 2]:.2e}; W2UE vs fp32-scale {sorted(errs['w2ue_vs_old'])[len(names) // 2]:.2e}, "
              f"W2UE vs SGLang {sorted(errs['w2ue_vs_sglang'])[len(names) // 2]:.2e}")
        print("   per expert (old / w2ue / sglang vs exact): " + ", ".join(
            f"{a:.2e}/{b:.2e}/{c:.2e}" for a, b, c in zip(errs["old"], errs["w2ue"], errs["sglang"])))
    # 2. the bound from the clamps alone
    eh_min = int(ue_exp(e4m3_groups(torch.full((1, 128), 1e-12), True)[1]))
    w0 = torch.zeros(128, 128, dtype=torch.uint8)
    ew_min = int(ue_exp(requant_ue8m0(w0, torch.ones(1, 1))[1]))
    print(f"2. bound: eh >= {eh_min} (amax 1e-10), ew >= {ew_min} (amax 1e-4): eh + ew - 127 >= {eh_min + ew_min - 127}; "
          f"above 254 only with eh + ew >= 382 (scales of 2^127 and more)")
    ok &= eh_min + ew_min - 127 >= 1
    # 3. the chunk and the MMA
    d0 = check_mx(g)
    print(f"3. simulated block-scaled MMA vs the dequantized product: max rel diff {d0:.1e} (must be < 1e-12: the fp64 "
          f"sums' order)")
    ok &= d0 < 1e-12
    for ctl in ("swap", "bias126", "by_token"):
        d = check_mx(g, 60, ctl)
        print(f"   control {ctl}: max rel diff {d:.1e} (must be > 0)")
        ok &= d > 1e-6
    print("NUMERICS OK" if ok else "NUMERICS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
