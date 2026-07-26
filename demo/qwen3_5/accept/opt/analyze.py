#!/usr/bin/env python3
"""Assemble the M3-I1 deliverables from the parsed per-batch-size artifacts.

Inputs (all committed under this directory):
  tables/bs<N>_attrib.json       per-step attribution + per-task-type table
  tables/bs<N>_concurrency.json  concurrency profile + worker gap structure
  tables/bs<N>_iters.csv         per-iteration timeline with schedule labels
  meta/, meta_noprof/            run metadata (profiled / unprofiled control)

Outputs:
  attribution.csv       the per-batch-size step audit
  pertask_by_bs.csv     per-task-type us table at every batch size
  layer_type_by_bs.csv  the same rolled up to layer types
  gap_vs_vllm.csv       decode-throughput gap re-derivation
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BSL = [1, 2, 4, 8, 16]
NW = 128

# M1-I6 pinned vLLM baseline, baselines/vllm-0.25.1-20260725/summary.json
VLLM = {1: 285.5, 2: 529.8, 4: 934.4, 8: 1692.5, 16: 3018.1}
# M2 first-light wave-level numbers (.memory/MAIN.md, M2-I9 dumps_final)
M2_WAVE = {1: 65.0, 2: 126.2, 4: 203.8, 8: 240.5, 16: 233.9}


def load():
    A, C = {}, {}
    for bs in BSL:
        A[bs] = json.load(open(HERE / f"tables/bs{bs}_attrib.json"))
        p = HERE / f"tables/bs{bs}_concurrency.json"
        C[bs] = json.load(open(p)) if p.exists() else None
    return A, C


def walls(sub):
    out = {}
    for bs in BSL:
        v = [json.load(open(f))["waves"][0]["wall_ms"]
             for f in sorted(glob.glob(str(HERE / sub / f"meta_bs{bs}_rep*.json")))]
        out[bs] = v
    return out


def main():
    A, C = load()
    wp, wn = walls("meta"), walls("meta_noprof")

    # ---------------- attribution ----------------
    rows = []
    for bs in BSL:
        s = A[bs]["summary"]
        c = C[bs]
        rows.append(dict(
            batch_size=bs,
            regime=" ".join(map(str, s["steady_regime_live_prefill_decode_tokens"])),
            steady_iters=f"{s['steady_window'][0]}-{s['steady_window'][1]}",
            step_us=round(s["step_us"], 1),
            step_us_spread_pct=round(100 * (s["step_us_max"] - s["step_us_min"])
                                     / s["step_us"], 2),
            tokens_per_step=s["tokens_per_step"],
            decode_tok_s=round(s["decode_tokens_per_s"], 1),
            task_sum_us_all_workers=round(s["task_sum_us"], 0),
            task_sum_per_worker_us=round(s["perfect_pack_us"], 1),
            sched_gap_all_idle_us=round(s["dead_all_idle_us"], 1),
            of_which_prepare_batch_us=round(s["prepare_batch_us"], 2),
            worker_idle_us=round(s["worker_idle_us"], 1),
            sum_us=round(s["perfect_pack_us"] + s["dead_all_idle_us"]
                         + s["worker_idle_us"], 1),
            closure_vs_step_pct=round(
                100 * (s["perfect_pack_us"] + s["dead_all_idle_us"]
                       + s["worker_idle_us"] - s["step_us"]) / s["step_us"], 4),
            closure_trace_vs_cudaevent_pct=round(s["closure_error_pct"], 2),
            occupancy=round(s["occupancy"], 3),
            mean_concurrency=round(c["mean_concurrency"], 1) if c else None,
            us_at_conc_le16=round(
                c["us_at_concurrency"]["zero"] + c["us_at_concurrency"]["c1_4"]
                + c["us_at_concurrency"]["c5_16"], 0) if c else None,
            gaps_per_worker=round(c["gaps"]["gaps_per_worker"], 1) if c else None,
            mean_gap_us=round(c["gaps"]["mean_gap_us"], 1) if c else None,
            tasks_per_step=int(s["tasks_per_step"]),
            wave_wall_ms_profiled=round(st.median(wp[bs]), 1),
            wave_wall_ms_unprofiled=round(st.median(wn[bs]), 1),
            profiling_overhead_pct=round(
                100 * (st.median(wp[bs]) / st.median(wn[bs]) - 1), 2),
            mixed_phase_iters=s["n_prefill_or_mixed"],
            mixed_phase_ms=round(s["prefill_total_ms"] or 0, 1),
            drain_phase_iters=s["n_decode_draining"],
            drain_phase_ms=round(s["drain_total_ms"], 1),
        ))
    write_csv(HERE / "attribution.csv", rows)

    # ---------------- per task type ----------------
    keys = ["n_per_iter", "total_us_per_iter", "per_worker_us_per_iter",
            "mean_us", "p50_us", "p95_us", "n_short_per_iter",
            "n_long_per_iter", "long_mean_us"]
    names = {}
    for bs in BSL:
        for r in A[bs]["per_task"]:
            names.setdefault(r["task_type"], (r["name"], r["bucket"]))
    pt = []
    for tt, (nm, bucket) in sorted(names.items()):
        row = dict(task_type=tt, name=nm, layer_type=bucket)
        for bs in BSL:
            r = next((x for x in A[bs]["per_task"] if x["task_type"] == tt), None)
            c = C[bs]["per_task_concurrency"].get(nm) if C[bs] else None
            row[f"n_bs{bs}"] = round(r["n_per_iter"], 1) if r else 0
            row[f"us_bs{bs}"] = round(r["total_us_per_iter"], 1) if r else 0
            row[f"perwkr_us_bs{bs}"] = round(r["per_worker_us_per_iter"], 2) if r else 0
            row[f"mean_us_bs{bs}"] = round(r["mean_us"], 2) if r else 0
            row[f"nlong_bs{bs}"] = round(r["n_long_per_iter"], 1) if r else 0
            row[f"wallspan_us_bs{bs}"] = round(c["wall_span_us"], 0) if c else None
            row[f"conc_bs{bs}"] = round(c["mean_concurrency_during"], 1) if c else None
        pt.append(row)
    pt.sort(key=lambda r: -r["us_bs1"])
    write_csv(HERE / "pertask_by_bs.csv", pt)

    # ---------------- layer-type rollup ----------------
    lt = {}
    for bs in BSL:
        for b in A[bs]["buckets"]:
            e = lt.setdefault(b["bucket"], dict(layer_type=b["bucket"]))
            e[f"n_bs{bs}"] = round(b["n_per_iter"], 1)
            e[f"us_bs{bs}"] = round(b["total_us_per_iter"], 1)
            e[f"perwkr_us_bs{bs}"] = round(b["per_worker_us_per_iter"], 2)
    lt_rows = sorted(lt.values(), key=lambda r: -r.get("us_bs1", 0))
    for r in lt_rows:
        for bs in BSL:
            tot = sum(x.get(f"us_bs{bs}", 0) for x in lt_rows)
            r[f"pct_bs{bs}"] = round(100 * r.get(f"us_bs{bs}", 0) / tot, 1)
    write_csv(HERE / "layer_type_by_bs.csv", lt_rows)

    # ---------------- vLLM gap ----------------
    gap = []
    for bs in BSL:
        s = A[bs]["summary"]
        dec = s["decode_tokens_per_s"]
        wave = M2_WAVE[bs]
        gap.append(dict(
            batch_size=bs, vllm_decode_tok_s=VLLM[bs],
            mpk_m2_wave_tok_s=wave, m2_headline_gap=round(VLLM[bs] / wave, 2),
            mpk_decode_step_us=round(s["step_us"], 1),
            mpk_decode_tok_s=round(dec, 1),
            true_decode_gap=round(VLLM[bs] / dec, 2),
            gap_from_prefill_and_drain=round(VLLM[bs] / wave - VLLM[bs] / dec, 2),
            perfect_pack_tok_s=round(
                s["tokens_per_step"] / (s["perfect_pack_us"] * 1e-6), 1),
            gap_if_perfectly_packed=round(
                VLLM[bs] / (s["tokens_per_step"] / (s["perfect_pack_us"] * 1e-6)), 2),
        ))
    write_csv(HERE / "gap_vs_vllm.csv", gap)
    print("wrote attribution.csv pertask_by_bs.csv layer_type_by_bs.csv "
          "gap_vs_vllm.csv")
    for r in gap:
        print(r)


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
