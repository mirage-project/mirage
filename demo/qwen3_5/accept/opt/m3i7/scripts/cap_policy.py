#!/usr/bin/env python3
"""M3-I7 -- the admission-cap A/B, with per-arm compiled kernels.

`--per-request-token-cap` is a COMPILE-TIME knob: the adapter turns it into
`-DMPK_MAX_TOKENS_PER_REQUEST=<n>` on the JIT command line (persistent_kernel.py
:323). Two arms sharing one `--kernel-dir` under `--reuse-kernel` therefore run
the SAME binary and differ only in a CPU-side admission replay -- which is
exactly what this gate's first pass did, reporting the arms as identical to
0.05% while the replay in the timings artifact still claimed 203-vs-131
iterations. Every arm below has its own kernel directory.

Decode throughput is the prefill-subtracted slope, as everywhere else in this
issue: bs*(D_full - D_pre) / (wall_full - wall_pre).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

VLLM_DECODE = {1: 285.5, 2: 529.8, 4: 934.4, 8: 1692.5, 16: 3018.1}
VLLM_E2E_S = {1: 3.60, 2: 3.89, 4: 4.45, 8: 4.953, 16: 5.568}


def waves(path):
    try:
        return json.load(open(path))["waves"]
    except Exception:
        return None


def med_range(v):
    v = [x for x in v if x is not None]
    return (st.median(v), min(v), max(v), len(v)) if v else (None, None, None, 0)


def one_wave(d, bs, reps=3):
    walls, steps = [], []
    for r in range(reps):
        w = waves(f"{d}/timings_bs{bs}_rep{r}.json")
        if w:
            walls.append(w[0]["wall_ms"])
            steps.append(w[0]["max_decode_steps"])
    m, lo, hi, n = med_range(walls)
    return dict(ms=m, lo=lo, hi=hi, n=n, D=(st.median(steps) if steps else None))


def sum_waves(d, bs, reps=3):
    tot = []
    for r in range(reps):
        w = waves(f"{d}/timings_bs{bs}_rep{r}.json")
        if w:
            tot.append(sum(x["wall_ms"] for x in w))
    if not tot:                      # single un-repped run written as bs<N>.json
        w = waves(f"{d}/timings_bs{bs}.json")
        if w:
            tot.append(sum(x["wall_ms"] for x in w))
    m, lo, hi, n = med_range(tot)
    return dict(ms=m, lo=lo, hi=hi, n=n)


def decode(full, pre, bs):
    if not (full["ms"] and pre["ms"] and full["D"] and pre["D"]):
        return None
    return bs * (full["D"] - pre["D"]) / ((full["ms"] - pre["ms"]) / 1000.0)


def pct(a, b):
    return None if not (a and b) else 100.0 * (a / b - 1.0)


def fmt(x, w=9, p=1):
    return " " * w if x is None else f"{x:{w}.{p}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/mpk-qwen35/i7"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    R = a.root
    rows = []

    # bs16 -- both arms captured in the dedicated per-arm-kernel phase
    arms16 = {
        "nocap": dict(full=one_wave(f"{R}/cap16/Mfull_nocap", 16),
                      pre=one_wave(f"{R}/cap16/Mpre_nocap", 16),
                      ac3=sum_waves(f"{R}/cap16/A_nocap", 16)),
        "cap": dict(full=one_wave(f"{R}/cap16/Mfull_cap", 16),
                    pre=one_wave(f"{R}/cap16/Mpre_cap", 16),
                    ac3=sum_waves(f"{R}/cap16/A_cap", 16)),
    }
    # bs4 / bs8 -- capped arm from the probe, uncapped from the main perfM sweep
    for bs in (4, 8):
        arms = {
            "nocap": dict(full=one_wave(f"{R}/perf/M/full", bs),
                          pre=one_wave(f"{R}/perf/M/pre", bs), ac3=None),
            "cap": dict(full=one_wave(f"{R}/capsweep/Mfull_bs{bs}", bs),
                        pre=one_wave(f"{R}/capsweep/Mpre_bs{bs}", bs), ac3=None),
        }
        rows.append((bs, arms))
    rows.append((16, arms16))
    rows.sort()

    print("=" * 118)
    print("ADMISSION CAP (--per-request-token-cap auto) A/B at the PINNED 256/1024 geometry, "
          "per-arm compiled kernels")
    print("=" * 118)
    print(f"{'bs':>3} {'arm':>6} | {'full ms':>9} {'r%':>5} {'n':>2} | {'prefill ms':>10} "
          f"{'n':>2} | {'decode tok/s':>12} | {'e2e s':>7} | {'vs uncapped: prefill':>21} "
          f"{'decode':>8} {'e2e':>8}")
    out = {}
    for bs, arms in rows:
        base = arms["nocap"]
        bd = decode(base["full"], base["pre"], bs)
        for arm in ("nocap", "cap"):
            x = arms[arm]
            d = decode(x["full"], x["pre"], bs)
            f_, p_ = x["full"], x["pre"]
            rel = ""
            if arm == "cap" and bd and base["pre"]["ms"] and base["full"]["ms"]:
                rel = (f"{base['pre']['ms'] / p_['ms']:9.2f}x"
                       f"{pct(bd and d / bd, 1) or 0:+8.1f}%"
                       f"{pct(base['full']['ms'] / f_['ms'], 1) or 0:+8.1f}%")
            rng = None
            if f_["ms"]:
                rng = 100.0 * (f_["hi"] - f_["lo"]) / f_["ms"]
            print(f"{bs:3d} {arm:>6} |{fmt(f_['ms'])} {fmt(rng, 5, 2)} {f_['n']:2d} |"
                  f"{fmt(p_['ms'], 10)} {p_['n']:2d} |{fmt(d, 12)} |"
                  f"{fmt(f_['ms'] / 1000 if f_['ms'] else None, 7, 2)} | {rel}")
            out[f"bs{bs}_{arm}"] = dict(full_ms=f_["ms"], full_range_pct=rng, n=f_["n"],
                                        pre_ms=p_["ms"], decode_tok_s=d,
                                        e2e_s=(f_["ms"] / 1000 if f_["ms"] else None),
                                        vllm_decode=VLLM_DECODE[bs],
                                        gap_x=(VLLM_DECODE[bs] / d if d else None),
                                        ac3_geometry_e2e_ms=(x["ac3"]["ms"] if x["ac3"] else None))
    a16 = arms16
    if a16["cap"]["ac3"]["ms"] and a16["nocap"]["ac3"]["ms"]:
        print(f"\nAC-3 geometry (bs16, 10 prompts, msl=132): uncapped "
              f"{a16['nocap']['ac3']['ms']:.1f} ms (n={a16['nocap']['ac3']['n']}) vs capped "
              f"{a16['cap']['ac3']['ms']:.1f} ms (n={a16['cap']['ac3']['n']}) = "
              f"{a16['nocap']['ac3']['ms'] / a16['cap']['ac3']['ms']:.3f}x")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
