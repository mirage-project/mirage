"""CPU checks of STATIC_MOE64_W13UE's numerics (tasks/scaled_moe_experts.cuh issue_w13_ue, fp8_fp32scale_weights.pack_w13).

  1. pack_w13(ue8m0=True) of a TP8 rank-0 gate / up shard (real GLM-5.3 experts of --layer) against SGLang's
     requantization of the expert's full fused w13 [2 I, H] (gate rows, then up rows; fp8_utils.py's block_quant_dequant +
     per_block_cast_to_fp8, read from --sglang), cut to the shard's gate and up rows and packed: pieces and scales bit for
     bit. Control (must differ): pack_w13 without ue8m0.
  2. the issuer's chunk words and the block-scaled MMA of one W13 item, simulated: piece rows 0..63 gate, 64..127 up of
     features 64 m ..; stage s's weight scales from lane s (the float4 of GateUpScales [KT][2] at 16 s B: gate / up of K
     tiles 2 s, 2 s + 1), per K tile the weight-side chunk (lane l: words {gate, gate, up, up}) and the z_q-side chunk (lane
     l: words {token l, token l + 32, 1, 1}); the MMA scales row r / column n of K block j by byte j of the word at
     16 (r % 32) + 4 (r / 32) (cp_sf 32x128b.warpx4's layout); stage s into issuer s % 2's accumulator, the epilogue adds
     the two. Against the dequantized product (fp64: equal up to the order of the sums), random e4m3 values and power-of-
     two scales. Controls (must differ): gate and up words swapped; row % 32 and row / 32 swapped in the z_q chunk; the
     weight scales of the stage's other K tile; one issuer's accumulator only.
  3. the numerics change on real weights (--experts experts and the shared one; x = N(0, 1) bf16, T = 64 rows): routed
     rows' relRMS against the exact reference (bf16 x, fp64 h, the checkpoint's weights): the fp32-scale kernel (old),
     W2UE (the default before this change: requantized down, power-of-two h_q scales), W2UE + W13UE (requantized gate /
     up too, power-of-two x scales: the quantization points of SGLang's DeepGEMM path), and W13UE against W2UE.
Run: python moe64_w13ue_numerics.py [--model /path/GLM-5.3] [--layer 3] [--experts 8] [--sglang .../fp8_utils.py]
     (torch with float8_e4m3fn; exit 0: every check passes and every control differs)
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# by path (the mirage package needs its native libraries)
_w = _load("fp8_fp32scale_weights", os.path.join(_here, "python", "mirage", "mpk", "fp8_fp32scale_weights.py"))
_n2 = _load("moe64_w2ue_numerics", os.path.join(_here, "moe64_w2ue_numerics.py"))
dequant, requant_ue8m0, pack_w13 = _w.dequant, _w.requant_ue8m0, _w.pack_w13
sglang_funcs, sglang_requant, e4m3_groups, fp8_gemm = (_n2.sglang_funcs, _n2.sglang_requant, _n2.e4m3_groups,
                                                       _n2.fp8_gemm)


# ---- 2. the issuer's chunks and the block-scaled MMA of a W13 item ----
def word(b):
    return bytes([b] * 4)


def ue(v):
    """a power-of-two fp32 -> its UE8M0 byte"""
    return (int(torch.tensor([float(v)], dtype=torch.float32).view(torch.int32)) >> 23) & 0xFF


def chunks(ws4, xs, s, at, ctl=None):
    """the two 512-B chunks of piece at of stage s (K tile kt = 2 s + at), as issue_w13_ue writes them: ws4 [NST][4] the
    lanes' float4s (lane s: gate, up of 2 s, gate, up of 2 s + 1), xs [KT][64] z_q's scales"""
    kt = 2 * s + at
    g_, u_ = (ws4[s][0], ws4[s][1]) if at == 0 else (ws4[s][2], ws4[s][3])
    if ctl == "other_kt":
        g_, u_ = (ws4[s][2], ws4[s][3]) if at == 0 else (ws4[s][0], ws4[s][1])
    if ctl == "gate_up":
        g_, u_ = u_, g_
    a, b = bytearray(512), bytearray(512)
    for lane in range(32):
        a[16 * lane:16 * lane + 16] = word(ue(g_)) * 2 + word(ue(u_)) * 2
        x0, x1 = ue(xs[kt][lane]), ue(xs[kt][32 + lane])
        if ctl == "swap":   # z_q chunk with row % 32 and row / 32 swapped: row r at 16 (r / 32) + 4 (r % 32)
            for r, x in ((lane, x0), (32 + lane, x1)):
                o = 16 * (r // 32) + 4 * (r % 32)
                b[o:o + 4] = word(x)
            continue
        b[16 * lane:16 * lane + 16] = word(x0) + word(x1) + word(127) * 2
    return a, b


def mma_scale(chunk, n, j):
    """the MMA's UE8M0 scale of row / column n, K block j (scale id j): byte j of the word of row n"""
    return chunk[16 * (n % 32) + 4 * (n // 32) + j]


def check_item(g, KT=8, trials=40, ctl=None):
    """one W13 item (128 piece rows x 64 tokens, KT K tiles): simulated MMA vs the dequantized product; max diff relative
    to the largest value"""
    worst = 0.0
    for _ in range(trials):
        w = torch.randn(128, 128 * KT, generator=g).mul(40).clamp(-448, 448).to(torch.float8_e4m3fn).double()
        x = torch.randn(64, 128 * KT, generator=g).mul(40).clamp(-448, 448).to(torch.float8_e4m3fn).double()
        sf = 2.0 ** torch.randint(-20, -5, (KT, 2), generator=g).double()   # GateUpScales [KT][2] of the item's block row
        xs = 2.0 ** torch.randint(-12, 0, (KT, 64), generator=g).double()   # z_q's scales [KT][T]
        ws4 = sf.reshape(KT // 2, 4).tolist()                                # lane s: the float4 at 16 s B
        acc = [torch.zeros(128, 64, dtype=torch.float64) for _ in range(2)]
        for s in range(KT // 2):
            iss = s % 2
            for at in range(2):
                kt = 2 * s + at
                a, b = chunks(ws4, xs.tolist(), s, at, ctl)
                for j in range(4):
                    kj = slice(128 * kt + 32 * j, 128 * kt + 32 * j + 32)
                    sa = torch.tensor([2.0 ** (mma_scale(a, r, j) - 127) for r in range(128)], dtype=torch.float64)
                    sb = torch.tensor([2.0 ** (mma_scale(b, n, j) - 127) for n in range(64)], dtype=torch.float64)
                    acc[iss] += (w[:, kj] @ x[:, kj].t()) * sa.unsqueeze(1) * sb.unsqueeze(0)
        got = acc[0] + (acc[1] if ctl != "one_issuer" else 0)
        want = torch.zeros(128, 64, dtype=torch.float64)
        for kt in range(KT):
            ks = slice(128 * kt, 128 * kt + 128)
            rs = torch.cat([sf[kt, 0].repeat(64), sf[kt, 1].repeat(64)])   # rows 0..63 gate, 64..127 up
            want += (w[:, ks] * rs.unsqueeze(1)) @ (x[:, ks] * xs[kt].unsqueeze(1)).t()
        worst = max(worst, (got - want).abs().max().item() / want.abs().max().item())
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/mnt/shared/models/GLM-5.3")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--sglang", default=_n2.SGL)
    args = ap.parse_args()
    g = torch.Generator().manual_seed(7)
    ok = True
    # 2. (no weights needed)
    d0 = check_item(g)
    print(f"2. simulated W13UE item vs the dequantized product: max rel diff {d0:.1e} (must be < 1e-12: the fp64 sums' order)")
    ok &= d0 < 1e-12
    for ctl in ("gate_up", "swap", "other_kt", "one_issuer"):
        d = check_item(g, trials=8, ctl=ctl)
        print(f"   control {ctl}: max rel diff {d:.1e} (must be > 1e-6)")
        ok &= d > 1e-6
    model = args.model if os.path.isfile(os.path.join(args.model, "model.safetensors.index.json")) else None
    if model is None:
        print(f"   {args.model}: no checkpoint, the real-weight checks (1, 3) skipped")
    else:
        from safetensors import safe_open
        ns = sglang_funcs(args.sglang)
        with open(os.path.join(model, "model.safetensors.index.json")) as f:
            where = json.load(f)["weight_map"]
        p = f"model.layers.{args.layer}.mlp."

        def get(name):
            with safe_open(os.path.join(model, where[p + name]), framework="pt") as f:
                return f.get_tensor(p + name)

        names = [f"experts.{e}." for e in range(args.experts)] + ["shared_experts."]
        rows, blk = slice(0, 256), slice(0, 2)   # TP8 rank 0
        x = torch.randn(64, 6144, generator=g).to(torch.bfloat16)
        errs = {k: [] for k in ("old", "w2ue", "w13ue", "w13ue_vs_w2ue")}
        rr = lambda a, b: ((a - b).norm() / b.norm()).item()
        for i, nm in enumerate(names):
            gf, gsf = get(nm + "gate_proj.weight").view(torch.uint8), get(nm + "gate_proj.weight_scale_inv").float()
            uf, usf = get(nm + "up_proj.weight").view(torch.uint8), get(nm + "up_proj.weight_scale_inv").float()
            gt, gs, ut, us = gf[rows].contiguous(), gsf[blk].contiguous(), uf[rows].contiguous(), usf[blk].contiguous()
            # 1. against SGLang's requantization of the fused full w13, then the shard's rows
            I = gf.shape[0]
            q, sf = sglang_requant(ns, torch.cat([gf, uf]), torch.cat([gsf, usf]))
            want = pack_w13(q[rows], sf[blk], q[I + rows.start:I + rows.stop], sf[I // 128 + blk.start:I // 128 + blk.stop])
            got = pack_w13(gt, gs, ut, us, ue8m0=True)
            plain = pack_w13(gt, gs, ut, us)
            eq = torch.equal(got[0], want[0]) and torch.equal(got[1].view(torch.int32), want[1].view(torch.int32))
            neg = torch.equal(plain[0], want[0]) and torch.equal(plain[1], want[1])
            if i < 2 or not eq or neg:
                print(f"1. {nm}w13 shard: pack_w13(ue8m0=True) bit-equal to SGLang's fused requantization, sharded: {eq}; "
                      f"control (no ue8m0) equal: {neg}")
            ok &= eq and not neg
            # 3. routed rows for x
            d, ds = get(nm + "down_proj.weight").view(torch.uint8)[:, rows].contiguous(), \
                get(nm + "down_proj.weight_scale_inv").float()[:, blk].contiguous()
            dq, dsf = requant_ue8m0(d, ds)
            xq, xs = e4m3_groups(x.float(), False)
            h = torch.nn.functional.silu(fp8_gemm(xq, xs, gt, gs).float()) * fp8_gemm(xq, xs, ut, us).float()
            old = fp8_gemm(*e4m3_groups(h, False), d, ds)
            w2ue = fp8_gemm(*e4m3_groups(h, True), dq, dsf)
            gq, gqs = requant_ue8m0(gt, gs)
            uq, uqs = requant_ue8m0(ut, us)
            xq2, xs2 = e4m3_groups(x.float(), True)
            h2 = torch.nn.functional.silu(fp8_gemm(xq2, xs2, gq, gqs).float()) * fp8_gemm(xq2, xs2, uq, uqs).float()
            w13ue = fp8_gemm(*e4m3_groups(h2, True), dq, dsf)
            xe = x.double()
            he = torch.nn.functional.silu(xe @ dequant(gt, gs).double().t()) * (xe @ dequant(ut, us).double().t())
            ex = he @ dequant(d, ds).double().t()
            errs["old"].append(rr(old, ex))
            errs["w2ue"].append(rr(w2ue, ex))
            errs["w13ue"].append(rr(w13ue, ex))
            errs["w13ue_vs_w2ue"].append(rr(w13ue, w2ue))
            # the UE8M0 bytes the issuer writes: the weights' and x's exponents directly (no sum: never clamped)
            e = torch.cat([gqs.flatten(), uqs.flatten(), xs2.flatten()]).view(torch.int32) >> 23
            ok &= bool(((e >= 1) & (e <= 254)).all())
        med = lambda k: sorted(errs[k])[len(names) // 2]
        print(f"3. routed rows' relRMS (median over {len(names)} experts, layer {args.layer}, x = N(0, 1), T = 64) against the "
              f"exact reference: fp32-scale kernel {med('old'):.2e}, W2UE {med('w2ue'):.2e}, W2UE + W13UE {med('w13ue'):.2e}; "
              f"W2UE + W13UE vs W2UE {med('w13ue_vs_w2ue'):.2e}")
        print("   per expert (old / w2ue / w2ue+w13ue vs exact): " + ", ".join(
            f"{a:.2e}/{b:.2e}/{c:.2e}" for a, b, c in zip(errs["old"], errs["w2ue"], errs["w13ue"])))
    print("NUMERICS OK" if ok else "NUMERICS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
