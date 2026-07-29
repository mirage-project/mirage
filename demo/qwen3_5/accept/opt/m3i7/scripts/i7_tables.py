#!/usr/bin/env python3
"""M3-I7 -- the milestone performance tables, from the raw per-rep artifacts.

Three geometries, three different jobs:

  A1  AC-3 geometry, M3-I1's EXACT shape (profile_wave.py --prompt-ids over the
      ascending-length reference subsets, msl=132, ONE wave per process).  This
      is the only shape whose wave wall is directly comparable to the committed
      M3-I1 baseline (opt/attribution.csv `wave_wall_ms_unprofiled`), so it is
      what the "how far has M3 moved the AC-3 geometry" row is built from.
      The profiled rep additionally gives the true steady decode step, which is
      the same quantity as M3-I1's `step_us` / `decode_tok_s` columns.

  A   AC-3 geometry, the FULL gate shape (mpk_engine_run.py over all ten pinned
      prompts, msl=132, ceil(10/bs) waves per process).  This is what the AC-3
      run itself costs end to end.

  M   The PINNED benchmark geometry: 256-token prompts / 1024 new tokens, the
      geometry the binding vLLM table was captured at.  Steady-state decode
      throughput is the SLOPE between a full run (msl=1280) and a prefill-only
      run (msl=259) on the SAME prompts:

          decode tok/s = bs * (D_full - D_pre) / (wall_full - wall_pre)

      which is vLLM's own definition (tokens / decode-window seconds,
      bench-protocol.md 5).  Reporting bs*1024/wall instead -- as M3-I9's
      stage-7 analysis did -- bills the 256-token prefill to decode and, at
      bs16, bills 256 prefill iterations to 1024 decode steps.

Dispersion is full range over reps (min..max), median is the reported value,
per the pinned statistical rule.  Any rep whose pinned device already held more
than the recorded foreign floor + 400 MiB at run start is listed as DIRTY and
excluded from the median (M3-I6a's fake-2.1x-regression lesson).

Usage:
    python3 i7_tables.py --root ~/mpk-qwen35/i7 --gpu 5 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from pathlib import Path

BSL = [1, 2, 4, 8, 16]

# The binding vLLM baseline (baselines/vllm-0.25.1-20260725/README.md), pinned,
# never re-run here: median decode tok/s and e2e seconds at 256 in / 1024 out.
VLLM_DECODE = {1: 285.5, 2: 529.8, 4: 934.4, 8: 1692.5, 16: 3018.1}
VLLM_E2E_S = {1: 3.60, 2: 3.89, 4: 4.45, 8: 4.953, 16: 5.568}

# M3-I1's committed AC-3-geometry baseline (opt/attribution.csv), the
# "before M3" column: profiled steady decode step and the wave wall it came with.
I1_STEP_US = {1: 15264.0, 2: 15647.6, 4: 15645.1, 8: 18618.2, 16: 22005.2}
I1_DECODE_TOK_S = {1: 65.5, 2: 127.8, 4: 255.7, 8: 429.7, 16: 681.7}
I1_WAVE_MS_UNPROF = {1: 1616.9, 2: 1670.2, 4: 1686.5, 8: 2117.0, 16: 4566.5}


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def med_range(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None, 0
    m = st.median(vals)
    return m, min(vals), max(vals), len(vals)


def pct_range(m, lo, hi):
    return None if not m else 100.0 * (hi - lo) / m


# --------------------------------------------------------------- dirty reps --
def tag_gpu(root):
    """tag -> the device that phase actually pinned.

    CORRECTION (M3-I7, second pass): the audit file records ALL EIGHT devices per
    run, so selecting by a hand-supplied device -- or, as this function first did,
    letting the last matching line win across a list of candidate devices --
    reports whichever co-tenant happened to be last, not the device the run was
    pinned to. That is exactly the difference between "this rep started dirty"
    and "some other card did", and it produced three phantom dirty reps in this
    issue's first pass. Each phase log names its own device once (`GATE gpu=N`)
    and lists every tag that phase ran, so the mapping is recoverable exactly.
    """
    out = {}
    for log in sorted((Path(root) / "logs").glob("run_*.log")):
        txt = log.read_text(errors="replace")
        m = re.search(r"GATE gpu=(\d+)", txt)
        if not m:
            continue
        g = int(m.group(1))
        for t in set(re.findall(r"^\s*\[(\S+)\] rc=", txt, re.M)):
            out[t] = g
    return out


def load_audit(root, gpu, floor):
    """tag -> MiB resident on the PINNED device just before that run started."""
    tg = tag_gpu(root)
    out = {}
    p = Path(root) / "audit" / "gpu_before.txt"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = re.match(r"^(\S+)\s+(\d+),\s*(\d+)\s*MiB,\s*(\d+)\s*%", line.strip())
        if not m:
            continue
        tag, dev, mib = m.group(1), int(m.group(2)), int(m.group(3))
        if tag in tg and dev == tg[tag]:
            out[tag] = mib
    return out


# ------------------------------------------------------------- geometry A1 --
def geomA1(root, audit, limit):
    """profile_wave.py, msl=132, --prompt-ids, one wave per process."""
    rows = {}
    for bs in BSL:
        vals, dirty = [], []
        for rep in range(3):
            tag = f"pA1_bs{bs}_rep{rep}"
            d = jload(f"{root}/perf/A1/meta_bs{bs}_rep{rep}.json")
            if not d or not d.get("waves"):
                continue
            if audit.get(tag, 0) > limit:
                dirty.append((rep, audit[tag]))
                continue
            vals.append(d["waves"][0]["wall_ms"])
        m, lo, hi, n = med_range(vals)
        # profiled steady step, from the inline parse of rep0
        att = jload(f"{root}/perf/A1/tables/bs{bs}_attrib.json")
        step_us = att["summary"]["step_us"] if att else None
        regime = att["summary"]["steady_regime_live_prefill_decode_tokens"] if att else None
        window = att["summary"]["steady_window"] if att else None
        rows[bs] = dict(wall_ms=m, lo=lo, hi=hi, n=n, dirty=dirty,
                        step_us=step_us, regime=regime, window=window)
    return rows


# -------------------------------------------------------------- geometry A --
def geomA(root, audit, limit, sub="A", pfx="pA"):
    """mpk_engine_run.py over all ten pinned prompts, msl=132."""
    rows = {}
    for bs in BSL:
        tot, nw, dirty = [], None, []
        for rep in range(3):
            tag = f"{pfx}_bs{bs}_rep{rep}" if sub == "A" else f"{pfx}_rep{rep}"
            d = jload(f"{root}/perf/{sub}/timings_bs{bs}_rep{rep}.json")
            if not d or not d.get("waves"):
                continue
            if audit.get(tag, 0) > limit:
                dirty.append((rep, audit[tag]))
                continue
            tot.append(sum(w["wall_ms"] for w in d["waves"]))
            nw = len(d["waves"])
        m, lo, hi, n = med_range(tot)
        rows[bs] = dict(wall_ms=m, lo=lo, hi=hi, n=n, waves=nw, dirty=dirty,
                        tok_s=(640.0 / (m / 1000.0)) if m else None)
    return rows


# -------------------------------------------------------------- geometry M --
def geomM(root, audit, limit, full="full", pre="pre", tagf="pM_full", tagp="pM_pre"):
    rows = {}
    for bs in BSL:
        acc = {}
        for kind, sub, tg in (("full", full, tagf), ("pre", pre, tagp)):
            walls, steps, dirty = [], [], []
            for rep in range(3):
                tag = f"{tg}_bs{bs}_rep{rep}"
                d = jload(f"{root}/perf/M/{sub}/timings_bs{bs}_rep{rep}.json")
                if not d or not d.get("waves"):
                    continue
                if audit.get(tag, 0) > limit:
                    dirty.append((rep, audit[tag]))
                    continue
                w = d["waves"][0]
                walls.append(w["wall_ms"])
                steps.append(w["max_decode_steps"])
            m, lo, hi, n = med_range(walls)
            acc[kind] = dict(wall_ms=m, lo=lo, hi=hi, n=n, dirty=dirty,
                             D=(st.median(steps) if steps else None))
        f, p = acc.get("full"), acc.get("pre")
        dec = None
        if f and p and f["wall_ms"] and p["wall_ms"] and f["D"] and p["D"]:
            dtok = bs * (f["D"] - p["D"])
            dsec = (f["wall_ms"] - p["wall_ms"]) / 1000.0
            dec = dtok / dsec if dsec > 0 else None
        rows[bs] = dict(full=f, pre=p, decode_tok_s=dec,
                        wave_tok_s=((bs * f["D"]) / (f["wall_ms"] / 1000.0)
                                    if f and f["wall_ms"] and f["D"] else None),
                        e2e_s=(f["wall_ms"] / 1000.0 if f and f["wall_ms"] else None))
    return rows


def fmt(x, w=9, p=1):
    return " " * w if x is None else f"{x:{w}.{p}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/mpk-qwen35/i7"))
    ap.add_argument("--gpu", required=True,
                    help="pinned device index; a comma list is allowed when phases ran on "
                         "different devices (each run's audit line names its own device, "
                         "so the union is unambiguous)")
    ap.add_argument("--floor", type=int, default=0,
                    help="foreign resident MiB the guard recorded at claim time")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    limit = a.floor + 400
    audit = {}
    for g in [int(x) for x in str(a.gpu).split(",")]:
        audit.update(load_audit(a.root, g, limit))
    out = Path(a.out_dir) if a.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    a1 = geomA1(a.root, audit, limit)
    ga = geomA(a.root, audit, limit)
    gm = geomM(a.root, audit, limit)
    # NOTE: this issue's first pass also captured `A_cap16` / `full_nocap16` /
    # `pre_nocap16` arms. Those are DISCARDED, not reported: they shared a
    # --kernel-dir with their opposite arm, and --per-request-token-cap is a
    # compile-time define, so both arms executed one binary. The valid A/B, with
    # a kernel dir per arm, is cap_policy.py -> tables/cap_policy.json.

    print("=" * 108)
    print("GEOMETRY A1 -- AC-3 geometry, M3-I1's shape (one wave/process, msl=132). "
          "vs the committed M3-I1 baseline.")
    print("=" * 108)
    print(f"{'bs':>3} | {'wave ms':>9} {'range%':>7} {'n':>2} | {'I1 wave ms':>10} "
          f"{'speedup':>8} | {'step us':>9} {'I1 step us':>10} {'speedup':>8} | "
          f"{'dec tok/s':>9} {'I1 tok/s':>9} {'x':>6} | regime")
    a1rows = []
    for bs in BSL:
        r = a1.get(bs)
        if not r or r["wall_ms"] is None:
            print(f"{bs:3d} | incomplete"); continue
        spd = I1_WAVE_MS_UNPROF[bs] / r["wall_ms"]
        dec = (bs * 1e6 / r["step_us"]) if r["step_us"] else None
        sspd = (I1_STEP_US[bs] / r["step_us"]) if r["step_us"] else None
        print(f"{bs:3d} |{fmt(r['wall_ms'])} {fmt(pct_range(r['wall_ms'], r['lo'], r['hi']), 7, 2)} "
              f"{r['n']:2d} |{fmt(I1_WAVE_MS_UNPROF[bs], 10)} {fmt(spd, 8, 3)}x |"
              f"{fmt(r['step_us'])} {fmt(I1_STEP_US[bs], 10)} {fmt(sspd, 8, 3)}x |"
              f"{fmt(dec)} {fmt(I1_DECODE_TOK_S[bs])} {fmt((dec/I1_DECODE_TOK_S[bs]) if dec else None, 6, 2)} | "
              f"{r['regime']} w={r['window']}"
              + ("" if (r['regime'] and r['regime'][0] == bs and r['regime'][1] == 0)
                 else "  <== NOT a full-width prefill-free step; see basis notes"))
        a1rows.append(dict(bs=bs, wave_ms=r["wall_ms"], range_pct=pct_range(r["wall_ms"], r["lo"], r["hi"]),
                           n=r["n"], i1_wave_ms=I1_WAVE_MS_UNPROF[bs], wave_speedup=spd,
                           step_us=r["step_us"], i1_step_us=I1_STEP_US[bs], step_speedup=sspd,
                           decode_tok_s=dec, i1_decode_tok_s=I1_DECODE_TOK_S[bs],
                           regime=r["regime"], dirty=r["dirty"]))

    print()
    print("=" * 108)
    print("GEOMETRY A -- AC-3 geometry, the full gate shape (all ten prompts, "
          "ceil(10/bs) waves, msl=132). e2e = sum of wave walls, 640 tokens.")
    print("=" * 108)
    print(f"{'bs':>3} {'waves':>6} | {'e2e ms':>9} {'range%':>7} {'n':>2} | "
          f"{'tok/s':>8} | dirty reps")
    garows = []
    for bs in BSL:
        r = ga.get(bs)
        if not r or r["wall_ms"] is None:
            print(f"{bs:3d} | incomplete"); continue
        print(f"{bs:3d} {str(r['waves']):>6} |{fmt(r['wall_ms'])} "
              f"{fmt(pct_range(r['wall_ms'], r['lo'], r['hi']), 7, 2)} {r['n']:2d} |"
              f"{fmt(r['tok_s'], 8)} | {r['dirty'] or '-'}")
        garows.append(dict(bs=bs, waves=r["waves"], e2e_ms=r["wall_ms"], n=r["n"],
                           range_pct=pct_range(r["wall_ms"], r["lo"], r["hi"]),
                           tok_s=r["tok_s"], dirty=r["dirty"]))
    print("  NOTE: these rows are the UNCAPPED build at every bs. The bs16 rows here are "
          "therefore NOT the pinned benchmark policy;\n        the admission-cap A/B needs "
          "per-arm compiled kernels and lives in cap_policy.py / tables/cap_policy.json.")

    print()
    print("=" * 108)
    print("GEOMETRY M -- the PINNED 256/1024 benchmark geometry, vs the binding vLLM table.")
    print("  decode tok/s = bs*(D_full - D_pre)/(wall_full - wall_pre)  [prefill subtracted]")
    print("=" * 108)
    print(f"{'bs':>3} | {'full ms':>9} {'r%':>5} {'D':>5} | {'pre ms':>8} {'D':>4} | "
          f"{'decode tok/s':>12} {'vLLM':>8} {'gap x':>7} | {'e2e s':>7} {'vLLM':>6} "
          f"{'gap x':>7} | {'wave tok/s':>10}")
    gmrows = []
    for bs in BSL:
        r = gm.get(bs)
        if not r or not r["full"] or r["full"]["wall_ms"] is None:
            print(f"{bs:3d} | incomplete"); continue
        f, p = r["full"], r["pre"]
        gapd = VLLM_DECODE[bs] / r["decode_tok_s"] if r["decode_tok_s"] else None
        gape = r["e2e_s"] / VLLM_E2E_S[bs] if r["e2e_s"] else None
        print(f"{bs:3d} |{fmt(f['wall_ms'])} {fmt(pct_range(f['wall_ms'], f['lo'], f['hi']), 5, 2)} "
              f"{str(f['D']):>5} |{fmt(p['wall_ms'] if p else None, 8)} {str(p['D'] if p else '-'):>4} |"
              f"{fmt(r['decode_tok_s'], 12)} {fmt(VLLM_DECODE[bs], 8)} {fmt(gapd, 7, 2)} |"
              f"{fmt(r['e2e_s'], 7, 2)} {fmt(VLLM_E2E_S[bs], 6, 2)} {fmt(gape, 7, 2)} |"
              f"{fmt(r['wave_tok_s'], 10)}")
        gmrows.append(dict(bs=bs, full_ms=f["wall_ms"], full_D=f["D"], n=f["n"],
                           range_pct=pct_range(f["wall_ms"], f["lo"], f["hi"]),
                           pre_ms=p["wall_ms"] if p else None, pre_D=p["D"] if p else None,
                           decode_tok_s=r["decode_tok_s"], vllm_decode_tok_s=VLLM_DECODE[bs],
                           gap_decode_x=gapd, e2e_s=r["e2e_s"], vllm_e2e_s=VLLM_E2E_S[bs],
                           gap_e2e_x=gape, wave_tok_s=r["wave_tok_s"],
                           dirty_full=f["dirty"], dirty_pre=p["dirty"] if p else None))
    print("  bs16 runs the pinned --per-request-token-cap auto policy; bs1-8 are uncapped.\n"
          "  The capped-vs-uncapped A/B (per-arm kernels, all of bs4/8/16) is in "
          "tables/cap_policy.json.")
    if out:
        def dump(name, rows):
            if not rows:
                return
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r_ in rows:
                    w.writerow(r_)
            print(f"wrote {out / name}")
        dump("geomA1_ac3_shape.csv", a1rows)
        dump("geomA_ac3_full.csv", garows)
        # window-1 only; the CANONICAL 256/1024 table is perrep.py's
        # geomM_matched_256_1024.csv, which selects per bs between this window
        # and the perfM2 re-capture and labels every row with its window.
        dump("geomM_window1_perfM.csv", gmrows)
        json.dump(dict(geomA1=a1, geomA=ga, geomM=gm, gpu=str(a.gpu), floor=a.floor,
                       dirty_limit_mib=limit),
                  open(out / "perf_raw.json", "w"), indent=1, default=str)
        print(f"wrote {out / 'perf_raw.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
