#!/usr/bin/env python3
"""M3-I9b: diff two probe arms.

`--curves A.json B.json`  per-position arm-to-arm comparison of the engine's
                          own logits (top1/top2 margin, delta at the reference
                          candidate ids, row hash equality).
`--states  A.pt   B.pt`   per-LAYER first divergence over every persistent
                          prefill state: GDN conv state, GDN fp32 recurrent
                          state, paged K cache, paged V cache -- plus the
                          lm-head logit row.  This is the layer bisect: the
                          lowest layer index whose state differs is where the
                          chunked path first departs, everything below it is
                          bit-identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def ulp(x: float) -> float:
    import math
    if x == 0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)   # bf16: 8-bit mantissa


def tstat(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.float(), b.float()
    d = (a32 - b32).abs()
    n_diff = int((a != b).sum().item()) if a.dtype == b.dtype else int((d > 0).sum().item())
    denom = a32.abs().max().item() or 1.0
    return {"identical": n_diff == 0, "n_elem": a.numel(), "n_diff": n_diff,
            "frac_diff": round(n_diff / max(a.numel(), 1), 6),
            "max_abs_delta": float(d.max().item()),
            "max_rel_delta": float(d.max().item() / denom),
            "rms_a": float(a32.pow(2).mean().sqrt().item())}


def cmp_states(pa: Path, pb: Path) -> dict:
    A, B = torch.load(pa, weights_only=False), torch.load(pb, weights_only=False)
    out = {"meta_a": A["meta"], "meta_b": B["meta"], "layers": {}, "summary": {}}
    for key in ("conv_state", "recurrent_state", "k_cache", "v_cache"):
        a, b = A[key], B[key]
        per = []
        first = None
        for l in range(a.shape[0]):
            s = tstat(a[l], b[l])
            per.append(s)
            if first is None and not s["identical"]:
                first = l
        out["layers"][key] = per
        out["summary"][key] = {"n_layers": a.shape[0],
                               "first_divergent_layer": first,
                               "all_identical": first is None}
    a, b = A["argmax_in"], B["argmax_in"]
    out["summary"]["argmax_in"] = tstat(a, b)
    return out


def cmp_curves(pa: Path, pb: Path) -> dict:
    A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
    pa_pts = {p["position"]: p for p in A["points"]}
    pb_pts = {p["position"]: p for p in B["points"]}
    rows = []
    first_logit_div = first_token_div = None
    for n in sorted(set(pa_pts) & set(pb_pts)):
        x, y = pa_pts[n], pb_pts[n]
        r = {"position": n,
             "A_emitted": x["emitted"], "B_emitted": y["emitted"],
             "ref_token": x["ref_token"],
             "A_margin": x.get("engine_margin_top1_top2"),
             "B_margin": y.get("engine_margin_top1_top2"),
             "ref_margin": x.get("ref_margin_top1_top2"),
             "row_sha_equal": x.get("row_sha") == y.get("row_sha"),
             "A_row": x.get("row_used"), "B_row": y.get("row_used")}
        ax, by = x.get("engine_at_ref_ids"), y.get("engine_at_ref_ids")
        if ax and by:
            d = [b - a for a, b in zip(ax, by)]
            r["engine_at_ref_ids_A"] = ax
            r["engine_at_ref_ids_B"] = by
            r["delta_B_minus_A"] = d
            r["max_abs_delta_at_ref_ids"] = max(abs(v) for v in d)
            u = ulp(ax[0]) if ax[0] else 1.0
            r["max_delta_ulps"] = r["max_abs_delta_at_ref_ids"] / u
            r["bf16_ulp"] = u
            # the arm-to-arm change in the (top1 - top2) gap at the reference ids
            if len(ax) >= 2:
                r["gap_A"] = ax[0] - ax[1]
                r["gap_B"] = by[0] - by[1]
                r["gap_shift"] = (by[0] - by[1]) - (ax[0] - ax[1])
        sa, sb = x.get("state_sha"), y.get("state_sha")
        if sa and sb:
            div = {}
            for kind in sa:
                bad = [l for l, (u, v) in enumerate(zip(sa[kind], sb[kind]))
                       if u != v]
                div[kind] = {"n_div": len(bad), "first": bad[0] if bad else None,
                             "layers": bad[:8]}
            r["state_divergence"] = div
        if not r["row_sha_equal"] and first_logit_div is None:
            first_logit_div = n
        if x["emitted"] != y["emitted"] and first_token_div is None:
            first_token_div = n
        rows.append(r)
    return {"arm_A": A["arm"], "arm_B": B["arm"], "target": A["target"],
            "first_logit_row_divergence": first_logit_div,
            "first_token_divergence": first_token_div,
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", nargs=2)
    ap.add_argument("--states", nargs=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    if a.curves:
        res["curves"] = cmp_curves(Path(a.curves[0]), Path(a.curves[1]))
        c = res["curves"]
        print(f"first logit-row divergence: {c['first_logit_row_divergence']}   "
              f"first token divergence: {c['first_token_divergence']}")
        print(f"{'pos':>4} {'A_emit':>8} {'B_emit':>8} {'ref':>8} "
              f"{'A_marg':>8} {'B_marg':>8} {'gapshift':>9} {'maxULP':>8} same")
        for r in c["rows"]:
            print(f"{r['position']:>4} {str(r['A_emitted']):>8} "
                  f"{str(r['B_emitted']):>8} {str(r['ref_token']):>8} "
                  f"{r['A_margin'] if r['A_margin'] is None else round(r['A_margin'],4):>8} "
                  f"{r['B_margin'] if r['B_margin'] is None else round(r['B_margin'],4):>8} "
                  f"{round(r.get('gap_shift', 0.0), 4):>9} "
                  f"{round(r.get('max_delta_ulps', 0.0), 2):>8} "
                  f"{'Y' if r['row_sha_equal'] else 'N'}"
                  + ("  state:" + " ".join(
                      f"{k}={v['n_div']}@{v['first']}"
                      for k, v in r["state_divergence"].items())
                     if "state_divergence" in r else ""))
    if a.states:
        res["states"] = cmp_states(Path(a.states[0]), Path(a.states[1]))
        s = res["states"]["summary"]
        for k, v in s.items():
            print(k, json.dumps(v))
        for key in ("conv_state", "recurrent_state", "k_cache", "v_cache"):
            per = res["states"]["layers"][key]
            bad = [(i, p) for i, p in enumerate(per) if not p["identical"]]
            print(f"--- {key}: {len(bad)}/{len(per)} layers differ")
            for i, p in bad[:6]:
                print(f"    layer {i}: n_diff={p['n_diff']}/{p['n_elem']} "
                      f"max_abs={p['max_abs_delta']:.3e} "
                      f"max_rel={p['max_rel_delta']:.3e} rms={p['rms_a']:.3e}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
