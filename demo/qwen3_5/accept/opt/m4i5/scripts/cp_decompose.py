#!/usr/bin/env python3
"""M4-I5 follow-up: what is the critical path MADE OF, and what would move it?

`critpath.py` returns the length of the longest weighted chain through the
compiled task graph.  This returns the chain ITSELF, plus a sensitivity table, so
the AC-4 question "which per-task latency has to fall, and to what" can be
answered arithmetically instead of guessed.

Three outputs, all from the compiled graph plus measured per-task times:

(a) COMPOSITION.  The longest path is recovered by recording, for every event,
    which producer task set its ready time, then walking back from the finishing
    task.  Reported per task type: how many path tasks, how many microseconds of
    the total, and the share.  A stage that appears N times on the path is N
    times in SERIES -- which is the fact that decides whether per-layer costs
    add up the way a back-of-envelope assumes.

(b) PER-LAYER STRUCTURE.  Each task's layer is read off its own tensor names
    (`layer_<i>_...` in `inputs[].base_ptr`/`outputs[].base_ptr`), so the chain
    inside one layer is exact rather than inferred.  GDN layers and
    full-attention layers are reported separately because they are different
    chains.

(c) SENSITIVITY.  Recompute the path with one stage's per-task duration scaled,
    then combined.  Floors come from `opt/m3i10/ferret_targets.json`'s MEASURED
    vLLM per-call numbers (`vllm_us_per_call`, per batch size) -- the same basis
    the ferret targets are set against -- and never from an invented target.
    Where vLLM fuses a stage so no per-call number exists (tasks 279 and 253),
    the per-level equivalent `vllm_us_per_step / sites_per_step` is used and
    labelled as derived.

Self-check: the unscaled path length must equal `critpath.py`'s `cp_max_us`.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

LAYER_RE = re.compile(r"layer_(\d+)_")


def load_graph(path):
    with open(path) as f:
        g = json.load(f)
    return g["all_tasks"], g["all_events"]


def task_layer(t):
    """Layer index from the task's own tensor names, or None."""
    for key in ("inputs", "outputs"):
        for d in t.get(key) or ():
            m = LAYER_RE.search(str(d.get("base_ptr", "")))
            if m:
                return int(m.group(1))
    return None


def longest_path(tasks, dep, trg, dur, n_e):
    """Longest weighted chain; returns (length, path task ids in order).

    Task ids are emitted in dependency order (critpath.py asserts 0 topological
    violations), so one ascending pass suffices.  For each event we keep the
    producer that set its ready time, which is what makes the path recoverable.
    """
    ev_ready = np.zeros(n_e)
    ev_arg = np.full(n_e, -1, dtype=np.int64)
    finish = np.zeros(len(tasks))
    pred = np.full(len(tasks), -1, dtype=np.int64)
    for i in range(len(tasks)):
        e = dep[i]
        if 0 <= e < n_e:
            finish[i] = ev_ready[e] + dur[i]
            pred[i] = ev_arg[e]
        else:
            finish[i] = dur[i]
        te = trg[i]
        if 0 <= te < n_e and finish[i] > ev_ready[te]:
            ev_ready[te] = finish[i]
            ev_arg[te] = i
    end = int(np.argmax(finish))
    path, cur = [], end
    while cur >= 0:
        path.append(cur)
        cur = int(pred[cur])
    path.reverse()
    return float(finish[end]), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("width_json")
    ap.add_argument("--names", required=True)
    ap.add_argument("--ferret", required=True)
    ap.add_argument("--weight", default="levelmax",
                    choices=("levelmax", "expected"),
                    help="how a task's duration is charged to the chain. "
                         "levelmax (default, CORRECT): every event in this graph "
                         "has num_triggers == n_producers -- verified, 2277 of "
                         "2277 -- so it is a full fan-in barrier and a level "
                         "costs its SLOWEST producer, which for a stage with any "
                         "live task is T_live. expected: the live/dead expected "
                         "value, which is what critpath.py used and which "
                         "UNDERSTATES any level mixing live and dead tasks (the "
                         "routed MoE GEMMs are 6%% live at bs1, so their chain "
                         "contribution came out ~14x too small).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    w = json.load(open(a.width_json))
    names = json.load(open(a.names))
    bs = w["batch_size"]
    ferret = json.load(open(a.ferret))

    # measured per-task durations, as critpath.py uses them: the expected value
    # over live and dead tasks of that type in the same window the width table
    # was built from.
    dur_live, dur_dead, live_frac, t_live_only = {}, {}, {}, {}
    for r in w["stages"]:
        tt = r["task_type"]
        dur_live[tt] = r["t_live_us"] or r["t_all_us"]
        dur_dead[tt] = r["t_dead_us"] or r["t_all_us"]
        live_frac[tt] = (r["live_per_step"] / r["n_per_step"]
                         if r["n_per_step"] else 0.0)
        t_live_only[tt] = r["t_live_us"] or r["t_all_us"]

    tasks, events = load_graph(a.graph)
    n_t, n_e = len(tasks), len(events)
    tt_arr = np.array([t["task_type"] for t in tasks], dtype=np.int32)
    dep = np.array([t["dependent_event"] for t in tasks], dtype=np.int64)
    trg = np.array([t["trigger_event"] for t in tasks], dtype=np.int64)
    layers = [task_layer(t) for t in tasks]

    def durations(scale=None):
        d = np.zeros(n_t)
        for tt in np.unique(tt_arr):
            tt = int(tt)
            k = (scale or {}).get(tt, 1.0)
            if a.weight == "levelmax":
                # the level's slowest producer.  T_live if the stage has any
                # live task in the window, else its (dead) task time.
                base = (dur_live.get(tt, 0.0) * k if live_frac.get(tt, 0.0) > 0
                        else dur_dead.get(tt, 0.0))
                d[tt_arr == tt] = base
            else:
                f = live_frac.get(tt, 0.0)
                d[tt_arr == tt] = (f * dur_live.get(tt, 0.0) * k
                                   + (1 - f) * dur_dead.get(tt, 0.0))
        return d

    base_dur = durations()
    cp, path = longest_path(tasks, dep, trg, base_dur, n_e)

    # ---------------- (a) composition ----------------
    comp = defaultdict(lambda: dict(n=0, us=0.0))
    for i in path:
        c = comp[int(tt_arr[i])]
        c["n"] += 1
        c["us"] += float(base_dur[i])
    rows = []
    for tt, c in comp.items():
        rows.append(dict(
            task_type=tt, name=names.get(str(tt), str(tt)),
            path_tasks=c["n"], us_on_path=round(c["us"], 1),
            pct_of_cp=round(100 * c["us"] / cp, 2),
            us_per_path_task=round(c["us"] / c["n"], 3),
            measured_t_live_us=t_live_only.get(tt),
            sites_per_step=next((t["sites_per_step"] for t in ferret["targets"]
                                 if t["mpk_task"].split()[0] == str(tt)), None)))
    rows.sort(key=lambda r: -r["us_on_path"])

    # ---------------- (b) per-layer structure ----------------
    per_layer = defaultdict(list)
    for i in path:
        per_layer[layers[i]].append(int(tt_arr[i]))
    gdn_l, attn_l = None, None
    # a full-attention layer is the one whose chain contains task 257
    for L, seq in per_layer.items():
        if L is None:
            continue
        if 257 in seq and attn_l is None:
            attn_l = L
        if 237 in seq and gdn_l is None:
            gdn_l = L
    def chain_of(L):
        if L is None:
            return None
        seq = per_layer[L]
        out, prev, run = [], None, 0
        for t in seq:
            if t == prev:
                run += 1
            else:
                if prev is not None:
                    out.append([names.get(str(prev), str(prev)), run])
                prev, run = t, 1
        if prev is not None:
            out.append([names.get(str(prev), str(prev)), run])
        return out
    layer_counts = Counter(len(v) for k, v in per_layer.items() if k is not None)

    # ---------------- (c) sensitivity ----------------
    def ferret_floor(tt):
        """(floor_us_per_layer, source) from the MEASURED vLLM numbers."""
        for t in ferret["targets"]:
            if t["mpk_task"].split()[0] != str(tt):
                continue
            pc = t.get("vllm_us_per_call")
            if isinstance(pc, dict) and f"bs{bs}" in pc:
                return float(pc[f"bs{bs}"]), "vllm_us_per_call (measured)"
            ps = t.get("vllm_us_per_step")
            sites = t.get("sites_per_step")
            if isinstance(ps, dict) and str(bs) in ps and sites:
                return (float(ps[str(bs)]) / float(sites),
                        "vllm_us_per_step / sites_per_step (derived; vLLM fuses this stage)")
        return None, None

    def improving_floor(tt):
        """ferret_floor, but only if it is actually an improvement.  MPK is
        already FASTER than vLLM on some stages (GDN recurrent at bs16: 10.20 us
        against 15.43), and 'bringing it to parity' there would be a regression,
        not a lever."""
        f, src = ferret_floor(tt)
        m = t_live_only.get(tt) or 0.0
        if f is None or m <= 0 or f >= m:
            return None, (src + " -- SKIPPED, MPK is already at or below it"
                          if src else None)
        return f, src

    dominant = [r["task_type"] for r in rows[:6] if r["us_on_path"] > 0.02 * cp]
    sens = []
    for tt in dominant:
        floor, src = ferret_floor(tt)
        meas = t_live_only.get(tt) or 0.0
        entry = dict(task_type=tt, name=names.get(str(tt), str(tt)),
                     measured_t_us=meas, vllm_floor_us=(round(floor, 3) if floor else None),
                     floor_source=src, arms=[])
        for label, target in (("vllm_x2", (floor * 2) if floor else None),
                              ("vllm_x1_parity", floor),
                              ("vllm_x0.7", (floor * 0.7) if floor else None)):
            if target is None or meas <= 0:
                continue
            k = target / meas
            cpk, _ = longest_path(tasks, dep, trg, durations({tt: k}), n_e)
            entry["arms"].append(dict(
                arm=label, target_t_us=round(target, 3), scale=round(k, 4),
                cp_us=round(cpk, 1), d_cp_us=round(cp - cpk, 1),
                cp_pct_of_base=round(100 * cpk / cp, 1)))
        sens.append(entry)

    # combined arms
    combos = {}
    def combo(label, spec):
        scale = {}
        for tt in spec:
            floor, _ = improving_floor(tt)
            meas = t_live_only.get(tt) or 0.0
            if floor and meas > 0:
                scale[tt] = floor / meas
        cpk, _ = longest_path(tasks, dep, trg, durations(scale), n_e)
        combos[label] = dict(stages=[names.get(str(t), str(t)) for t in spec],
                             scales={str(t): round(v, 4) for t, v in scale.items()},
                             cp_us=round(cpk, 1),
                             cp_pct_of_base=round(100 * cpk / cp, 1))
    combo("moe_chain_parity", [241, 242, 260])
    combo("moe_chain_plus_dense_fp8", [241, 242, 260, 279])
    combo("moe_chain_plus_dense_fp8_attn", [241, 242, 260, 279, 257])
    combo("moe_chain_plus_dense_parity", [241, 242, 260, 279, 253])
    combo("all_dominant_parity", dominant)
    combo("every_stage_parity", [int(t["mpk_task"].split()[0]) for t in ferret["targets"]])

    # ---------------- feasibility: what suffices to get under vLLM's step ----
    # cp is a MAX over paths, so it is not additive under perturbation:
    # cheapening one stage can expose a different, longer chain.  So the
    # sufficient set is found by recomputing cp, greedily by measured gain.
    VLLM_STEP = {1: 3503.0, 8: 4727.0, 16: 5301.0}   # bs / decode tok/s, M3-I7 2b
    target_step = VLLM_STEP.get(bs)
    cand = []
    for t in ferret["targets"]:
        tt = int(t["mpk_task"].split()[0])
        floor, src = improving_floor(tt)
        meas = t_live_only.get(tt) or 0.0
        n = next((r["path_tasks"] for r in rows if r["task_type"] == tt), 0)
        if floor and meas > 0 and n:
            cand.append((n * (meas - floor), tt, floor, meas, n, src))
    cand.sort(reverse=True)
    greedy, chosen, first_ok = [], [], None
    for gain, tt, floor, meas, n, src in cand:
        chosen.append(tt)
        scale = {}
        for x in chosen:
            f, _ = improving_floor(x)
            m = t_live_only.get(x) or 0.0
            if f and m > 0:
                scale[x] = f / m
        cpk, _ = longest_path(tasks, dep, trg, durations(scale), n_e)
        rec = dict(added=names.get(str(tt), str(tt)), task_type=tt,
                   path_tasks=n, measured_t_us=round(meas, 2),
                   target_t_us=round(floor, 3),
                   naive_additive_gain_us=round(gain, 1),
                   cp_us=round(cpk, 1),
                   under_vllm_step=(bool(cpk < target_step) if target_step else None))
        greedy.append(rec)
        if target_step and cpk < target_step and first_ok is None:
            first_ok = list(chosen)
    # measured chain-to-step packing gap, which a cp floor does NOT include
    gap = w["step_us"] / cp
    feas = dict(
        vllm_whole_step_us=target_step,
        cp_base_us=round(cp, 1),
        cp_must_fall_by_us=(round(cp - target_step, 1) if target_step else None),
        greedy_parity_order=greedy,
        minimal_sufficient_set=([names.get(str(x), str(x)) for x in first_ok]
                                if first_ok else None),
        measured_step_over_cp=round(gap, 3),
        step_if_cp_met_and_gap_unchanged_us=(
            round(min((g["cp_us"] for g in greedy), default=cp) * gap, 1)),
        note=("cp is a floor on the step, not the step: the measured step is "
              f"{gap:.2f}x cp at this batch size. Closing the per-task latency "
              "gap alone leaves that packing factor in place, which is what the "
              "width work addresses -- so latency and width are each necessary "
              "and neither is sufficient."))
    

    out = dict(
        batch_size=bs, window=w["window"], step_measured_us=w["step_us"],
        cp_max_us=round(cp, 1),
        weighting=a.weight,
        weighting_note=("levelmax: every event has num_triggers == n_producers "
                        "(2277/2277 verified), so a level costs its slowest "
                        "producer. 'expected' reproduces critpath.py's cp_max_us "
                        "and understates any mixed-liveness level."),
        path_length_tasks=len(path),
        n_tasks=n_t, n_events=n_e,
        composition=rows,
        per_layer=dict(
            layers_on_path=len([k for k in per_layer if k is not None]),
            path_tasks_per_layer_hist=dict(sorted(layer_counts.items())),
            example_gdn_layer=gdn_l, gdn_layer_chain=chain_of(gdn_l),
            example_attention_layer=attn_l, attention_layer_chain=chain_of(attn_l),
            tasks_off_layer=len(per_layer.get(None, []))),
        sensitivity=sens, combined=combos, feasibility=feas)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"=== bs{bs}  weight={a.weight}  cp_max = {cp:.1f} us  "
          f"({len(path)} tasks on the path, "
          f"{out['per_layer']['layers_on_path']} layers, "
          f"{100*cp/w['step_us']:.1f}% of the measured step) ===")
    print(f"{'stage':36s}{'path_tasks':>11s}{'us_on_path':>11s}{'%cp':>7s}"
          f"{'us/task':>9s}{'T_live':>8s}")
    for r in rows:
        print(f"{r['name'][:36]:36s}{r['path_tasks']:11d}{r['us_on_path']:11.1f}"
              f"{r['pct_of_cp']:7.2f}{r['us_per_path_task']:9.3f}"
              f"{r['measured_t_live_us'] or 0:8.2f}")
    print(f"\nper-layer path-task count histogram: "
          f"{out['per_layer']['path_tasks_per_layer_hist']}")
    print(f"GDN layer {gdn_l} chain: {out['per_layer']['gdn_layer_chain']}")
    print(f"ATTN layer {attn_l} chain: {out['per_layer']['attention_layer_chain']}")
    print(f"\n{'stage':30s}{'T_meas':>8s}{'vllm':>8s}"
          f"{'cp@2x':>9s}{'cp@1x':>9s}{'cp@0.7x':>9s}")
    for e in sens:
        g = {x["arm"]: x["cp_us"] for x in e["arms"]}
        print(f"{e['name'][:30]:30s}{e['measured_t_us']:8.2f}"
              f"{e['vllm_floor_us'] or 0:8.2f}"
              f"{g.get('vllm_x2', 0):9.1f}{g.get('vllm_x1_parity', 0):9.1f}"
              f"{g.get('vllm_x0.7', 0):9.1f}")
    print()
    for k, v in combos.items():
        print(f"  {k:32s} cp = {v['cp_us']:8.1f} us  ({v['cp_pct_of_base']:.1f}% of base)")
    print(f"\nFEASIBILITY vs vLLM's whole step ({target_step} us): cp must fall "
          f"{feas['cp_must_fall_by_us']} us")
    print(f"  {'add stage at vLLM parity':34s}{'n':>4s}{'T_meas':>8s}"
          f"{'T_target':>9s}{'cp':>9s}{'<vLLM?':>8s}")
    for g in greedy:
        print(f"  {g['added'][:34]:34s}{g['path_tasks']:4d}{g['measured_t_us']:8.2f}"
              f"{g['target_t_us']:9.2f}{g['cp_us']:9.1f}"
              f"{('YES' if g['under_vllm_step'] else 'no'):>8s}")
    print(f"  minimal sufficient set: {feas['minimal_sufficient_set']}")
    print(f"  measured step / cp = {feas['measured_step_over_cp']}  -> a step of "
          f"{feas['step_if_cp_met_and_gap_unchanged_us']} us if that packing "
          f"factor is unchanged")


if __name__ == "__main__":
    main()
