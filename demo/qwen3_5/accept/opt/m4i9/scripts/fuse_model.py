#!/usr/bin/env python3
"""M4-I9 -- per-SITE chain enumeration + FUSION counterfactuals on the exact floors.

This is a thin driver over M4-I8's `sched_gap.py`: it imports that module's
reconstruction (load_graph / predicted_order / fit_assignment / decompose /
cp_priority) unchanged, so the schedule it reasons about is the SAME verified
reconstruction, and adds exactly two things M4-I8 did not need.

1. SITE resolution.  M4-I8 aggregated the realized critical chain by task TYPE.
   A fusion decision is per CALL SITE: `quantize_fp8` appears at 5 sites per
   layer with completely different producers (rms-norm, gdn_recurrent /
   attention, moe_silu_mul, ...), and only some of them are fusable at all.
   The site is read off the task's own tensor names in `task_graph_rank0.json`
   (`outputs[0]` with the `layer_<i>_` prefix stripped) -- the same handle
   M4-I5's cp_decompose.py uses for per-layer structure.

2. FUSION counterfactuals on cp_exact and the work bound.  M4-I8's
   `floor_counterfactuals` zeroes one task TYPE at a time; that prices "this
   whole stage becomes free" and is an UPPER bound on any fusion's benefit.
   Fusing victim V into host H replaces (dur[H], dur[V]) by one task of
   duration dur[H] + inc, where `inc` is the incremental cost of V's arithmetic
   inside H.  On the longest-weighted-path computation that is EXACTLY
   `dur[V] <- inc` (the edge V->H stays; a zero/inc-weight node on it adds inc),
   and on the work bound it is exactly `total -= (dur[V] - inc)`.  So both
   floors are exact functions of `inc`, and `inc = 0` is the optimistic bound.

Neither floor contains the per-edge dispatch cost, so the measured-step benefit
of removing a chain record is reported separately: each removed record also
removes its own data-gap or resource-gap term (M4-I8 measured ~1.15 us event
visibility / ~1.55 us queue pop) and its barrier pair.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))                        # opt/  (trace_lib)
sys.path.insert(0, str(HERE.parents[2] / "m4i8" / "scripts"))    # M4-I8's tools
import trace_lib as tl          # noqa: E402
import sched_gap as sg          # noqa: E402

LAYER_RE = re.compile(r"layer_(\d+)_")

# vLLM's whole decode step, the AC-4 denominator (M4-I8 README table).
VLLM_US = {1: 3503.0, 8: 4727.0, 16: 5301.0}


def tname(x) -> str:
    """Tensor name: the graph dumps either a bare name or a TensorDesc dict."""
    if isinstance(x, dict):
        for k in ("base_ptr", "name", "tensor_name"):
            if isinstance(x.get(k), str):
                return x[k]
        return ""
    return x or ""


def strip_layer(name) -> str:
    return LAYER_RE.sub("", tname(name))


def site_labels(graph_path, names):
    """position -> (site, layer) from the task's own tensor names."""
    g = json.load(open(graph_path))
    tasks = g["all_tasks"]
    site = [""] * len(tasks)
    layer = [None] * len(tasks)
    meta = {}
    for p, t in enumerate(tasks):
        tt = t["task_type"]
        outs = t.get("outputs") or []
        ins = t.get("inputs") or []
        tag = strip_layer(outs[0]) if outs else ("<" + (strip_layer(ins[0]) if ins else "?") + ">")
        site[p] = f"{names.get(str(tt), str(tt))}|{tag}"
        if site[p] not in meta:
            meta[site[p]] = dict(
                task_type=tt,
                out_tile=[o.get("dims") for o in outs if isinstance(o, dict)],
                out_names=[strip_layer(o) for o in outs],
                in_names=[strip_layer(i) for i in ins])
        for nm in list(outs) + list(ins):
            m = LAYER_RE.search(tname(nm))
            if m:
                layer[p] = int(m.group(1))
                break
    return site, layer, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("meta")
    ap.add_argument("names")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--window", required=True, help="lo,hi iteration window")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--out")
    ap.add_argument("--inc-ns", type=int, default=0,
                    help="incremental ns charged to the host per fused victim")
    ap.add_argument("--sets", help="JSON {name: [site-substr, ...]} evaluated "
                                   "JOINTLY (sub-additivity is the whole point)")
    a = ap.parse_args()

    lo_it, hi_it = (int(x) for x in a.window.split(","))
    names = tl.load_names(Path(a.names))
    gr = sg.load_graph(a.graph)
    site, layer, smeta = site_labels(a.graph, names)
    meta = json.load(open(a.meta))
    bs = int(meta.get("batch_size", meta.get("bs", 0)))

    z = np.load(a.raw)                      # sparse dump: idx / val / header
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    buf[:1] = z["header"].view(np.uint64)
    del idx, val, z
    ev = tl.decode_events(buf)
    del buf
    pairs = tl.pair_events(ev)
    bounds = tl.iteration_bounds(pairs)
    n_it = len(bounds) - 1
    it = lo_it
    assert 0 <= it < n_it, f"iteration {it} outside [0,{n_it})"

    per, lo, hi = sg.iteration_records(pairs, bounds, it, a.workers)
    pred = sg.predicted_order(gr, a.workers)
    assign, qc = sg.fit_assignment(per, pred, gr, a.workers)
    assert qc["verdict"] == "PASS", f"assign_qc FAIL: {qc}"
    d, aux = sg.decompose(per, assign, gr, lo, hi, a.workers, names)

    out = dict(raw=a.raw, graph=a.graph, batch_size=bs, iteration=it,
               window=[lo_it, hi_it], assign_qc=qc, step_us=d["step_us"],
               identity_error_ns=d["identity_error_ns"],
               data_gap_us=d["data_gap_us"], resource_gap_us=d["resource_gap_us"],
               n_data_edges=d["n_data_edges"],
               n_resource_edges=d["n_resource_edges"],
               data_gap_median_ns=d["data_gap_median_ns"],
               res_gap_median_ns=d["res_gap_median_ns"],
               chain_len=d["chain_len"], chain_n_task=d["chain_n_task"],
               n_tasks=gr["n_tasks"], n_events=gr["n_events"])

    # ---------------- durations of THIS iteration ---------------------------
    dur = (aux["end"] - aux["start"]).astype(np.int64)
    dur[aux["start"] < 0] = 0
    allpos = np.concatenate([assign[w] for w in range(a.workers)])
    tot = int(sum(int(dur[p]) for p in allpos))
    prio0 = sg.cp_priority(gr, dur, allpos)
    cp0 = max(prio0.values())
    out["floors"] = dict(cp_exact_us=cp0 / 1e3, work_bound_us=tot / a.workers / 1e3,
                         total_task_us=tot / 1e3)

    # ---------------- (1) per-SITE chain enumeration -------------------------
    ch = aux["chain"]
    by_site = defaultdict(lambda: dict(n=0, dur_ns=0, data_gap_ns=0,
                                       res_gap_ns=0, n_data=0, n_res=0,
                                       layers=set(), task_type=None))
    for k, c in enumerate(ch):
        p = c["pos"]
        s = site[p] if p >= 0 else f"{names.get(str(c['tt']), c['tt'])}|<sev>"
        r = by_site[s]
        r["n"] += 1
        r["dur_ns"] += c["dur_ns"]
        r["task_type"] = c["tt"]
        if p >= 0 and layer[p] is not None:
            r["layers"].add(layer[p])
        if c["binding"] == "data":
            r["data_gap_ns"] += c["gap_ns"]; r["n_data"] += 1
        else:
            r["res_gap_ns"] += c["gap_ns"]; r["n_res"] += 1
    rows = []
    for s, r in by_site.items():
        rows.append(dict(site=s, task_type=r["task_type"], n=r["n"],
                         dur_us=r["dur_ns"] / 1e3,
                         us_per_record=r["dur_ns"] / 1e3 / max(r["n"], 1),
                         data_gap_us=r["data_gap_ns"] / 1e3, n_data=r["n_data"],
                         res_gap_us=r["res_gap_ns"] / 1e3, n_res=r["n_res"],
                         gap_us=(r["data_gap_ns"] + r["res_gap_ns"]) / 1e3,
                         n_layers=len(r["layers"])))
    for r in rows:
        r.update(smeta.get(r["site"], {}))
    rows.sort(key=lambda x: -(x["dur_us"] + x["gap_us"]))
    out["chain_by_site"] = rows

    # ---------------- site -> positions (whole step) ------------------------
    pos_of_site = defaultdict(list)
    for p in allpos:
        pos_of_site[site[int(p)]].append(int(p))
    out["site_census"] = sorted(
        ({"site": s, "n_tasks": len(ps),
          "total_us": sum(int(dur[p]) for p in ps) / 1e3}
         for s, ps in pos_of_site.items()),
        key=lambda x: -x["total_us"])

    # ---------------- (2) fusion counterfactuals ----------------------------
    def floors_with(zero_sites, inc_ns):
        d2 = dur.copy()
        removed = 0
        for s in zero_sites:
            for p in pos_of_site.get(s, ()):
                if d2[p] > inc_ns:
                    removed += int(d2[p]) - inc_ns
                    d2[p] = inc_ns
        p2 = sg.cp_priority(gr, d2, allpos)
        cp = max(p2.values())
        wb = (tot - removed) / a.workers
        return dict(cp_exact_us=cp / 1e3, work_bound_us=wb / 1e3,
                    binding_floor_us=max(cp, wb) / 1e3,
                    removed_task_us=removed / 1e3)

    chain_sites = [r["site"] for r in rows if r["task_type"] != tl.TASK_SCHD_EVENTS]
    single = []
    for s in chain_sites:
        f = floors_with([s], a.inc_ns)
        f.update(site=s, cp_delta_us=(cp0 / 1e3) - f["cp_exact_us"],
                 chain_records=next(r["n"] for r in rows if r["site"] == s),
                 chain_gap_us=next(r["gap_us"] for r in rows if r["site"] == s))
        single.append(f)
    single.sort(key=lambda r: -r["cp_delta_us"])
    out["fusion_single"] = single

    # ---------------- (3) JOINT sets ---------------------------------------
    sets_out = []
    if a.sets:
        spec = json.loads(Path(a.sets).read_text()
                          if Path(a.sets).exists() else a.sets)
        for label, pats in spec.items():
            hit = sorted({s for s in pos_of_site
                          for pt in pats if pt in s})
            f = floors_with(hit, a.inc_ns)
            gap = sum(r["gap_us"] for r in rows if r["site"] in hit)
            nrec = sum(r["n"] for r in rows if r["site"] in hit)
            f.update(label=label, sites=hit, n_sites=len(hit),
                     cp_delta_us=(cp0 / 1e3) - f["cp_exact_us"],
                     chain_records_removed=nrec, chain_gap_removed_us=gap)
            sets_out.append(f)
    out["fusion_sets"] = sets_out
    out["inc_ns"] = a.inc_ns
    out["vllm_us"] = VLLM_US.get(bs)
    if out["vllm_us"]:
        v = out["vllm_us"]
        out["floors"]["binding_floor_us"] = max(out["floors"]["cp_exact_us"],
                                                out["floors"]["work_bound_us"])
        out["floors"]["ratio_vllm"] = out["floors"]["binding_floor_us"] / v
        for r in single + sets_out:
            r["ratio_vllm"] = r["binding_floor_us"] / v

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))

    print(f"=== bs{bs} it{it} step={d['step_us']:.1f}us ident_err="
          f"{d['identity_error_ns']}ns assign={qc['verdict']}")
    print(f"    cp_exact={out['floors']['cp_exact_us']:.1f} "
          f"work_bound={out['floors']['work_bound_us']:.1f} "
          f"vllm={out['vllm_us']} ratio={out['floors'].get('ratio_vllm', 0):.3f}")
    print(f"{'chain site':52s}{'n':>4s}{'dur_us':>9s}{'us/rec':>8s}"
          f"{'gap_us':>8s}{'nD':>5s}{'nR':>5s}")
    for r in rows[:34]:
        print(f"{r['site'][:52]:52s}{r['n']:4d}{r['dur_us']:9.1f}"
              f"{r['us_per_record']:8.2f}{r['gap_us']:8.1f}"
              f"{r['n_data']:5d}{r['n_res']:5d}")
    print(f"\n--- fuse ONE site (inc={a.inc_ns}ns): floors ---")
    print(f"{'site':52s}{'cpΔ':>9s}{'cp':>9s}{'work':>9s}{'floor':>9s}{'x vLLM':>8s}")
    for r in single[:24]:
        print(f"{r['site'][:52]:52s}{r['cp_delta_us']:9.1f}"
              f"{r['cp_exact_us']:9.1f}{r['work_bound_us']:9.1f}"
              f"{r['binding_floor_us']:9.1f}{r.get('ratio_vllm', 0):8.3f}")
    if sets_out:
        print(f"\n--- fuse a SET jointly (inc={a.inc_ns}ns) ---")
        print(f"{'set':34s}{'nsite':>6s}{'nrec':>6s}{'gapΔ':>8s}{'cpΔ':>9s}"
              f"{'cp':>9s}{'work':>9s}{'floor':>9s}{'x vLLM':>8s}")
        for r in sets_out:
            print(f"{r['label'][:34]:34s}{r['n_sites']:6d}"
                  f"{r['chain_records_removed']:6d}"
                  f"{r['chain_gap_removed_us']:8.1f}{r['cp_delta_us']:9.1f}"
                  f"{r['cp_exact_us']:9.1f}{r['work_bound_us']:9.1f}"
                  f"{r['binding_floor_us']:9.1f}{r.get('ratio_vllm', 0):8.3f}")


if __name__ == "__main__":
    main()
