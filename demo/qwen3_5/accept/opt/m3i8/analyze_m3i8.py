#!/usr/bin/env python3
"""M3-I8 A/B analysis -- graded against the predictions, not against hope.

Reads the parsed tables produced by `run_m3i8.sh` for two or more arms and
prints, per batch size and per MoE stage:

  activated groups   nlong / (40 layers * moe_n_splits)  -- the MECHANISM
  live tasks         nlong / 40
  waves              ceil(live tasks / 128)              -- the COST DRIVER
  per-task us        long_mean_us                        -- must stay flat
  wall span / layer  the stage's union span / 40
  step us, tok/s     from the concurrency table

plus the delta vs the base arm and, for the mechanism row, the pre-registered
prediction from `predictions.md` so a reader sees immediately whether the
number that was promised is the number that arrived.

Usage: python3 analyze_m3i8.py <root> <armA> <armB> [...]
"""
import json
import math
import sys
from pathlib import Path

BS = [1, 2, 4, 8, 16]
LAYERS = 40
WORKERS = 128
STAGES = {241: ("moe_w13", "TASK_MOE_W13_FP8_BLOCKSCALE_SM100"),
          242: ("moe_w2", "TASK_MOE_W2_FP8_BLOCKSCALE_SM100")}
# arm -> moe_n_splits (needed to turn live tasks back into activated groups)
SPLITS = {"base": 2, "v1": 2, "v2a": 4, "v2b": 8}
# M3-I1 baseline, from opt/backlog.json
I1_STEP = {1: 15264.0, 2: 15647.6, 4: 15645.1, 8: 18618.2, 16: 22005.2}
# pre-registered in predictions.md (P2): activated groups per layer after v1
PRED_ACTIVATED = {1: 8.0, 2: 14.7, 4: 24.6, 8: 47.9, 16: 86.7}
PRED_ACTIVATED_CAP = {1: 8, 2: 16, 4: 32, 8: 64, 16: 128}


def load(root: Path, arm: str, bs: int):
    a = root / f"tables_{arm}" / f"bs{bs}_attrib.json"
    c = root / f"tables_{arm}" / f"bs{bs}_concurrency.json"
    if not a.exists() or not c.exists():
        return None, None
    return json.load(open(a)), json.load(open(c))


def rows_for(root: Path, arm: str):
    out = {}
    for bs in BS:
        att, con = load(root, arm, bs)
        if att is None:
            continue
        per = {r["task_type"]: r for r in att["per_task"]}
        pc = con["per_task_concurrency"]
        r = dict(step_us=con["step_us"])
        for tt, (short, long_name) in STAGES.items():
            if tt not in per:
                continue
            p = per[tt]
            live = p["n_long_per_iter"] / LAYERS
            r[short] = dict(
                n=p["n_per_iter"], live_tasks=live,
                activated=live / SPLITS.get(arm, 2),
                T=p["long_mean_us"],
                waves=max(1, math.ceil(live / WORKERS)),
                wall_layer=pc[long_name]["wall_span_us"] / LAYERS,
                total_us=p["total_us_per_iter"])
        out[bs] = r
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    arms = sys.argv[2:]
    data = {a: rows_for(root, a) for a in arms}
    base = arms[0]

    print("# M3-I8 A/B\n")
    print("## Mechanism: activated expert groups per layer "
          "(the number M3-I1 measured at 56.4 / 59.4 / 60.2 / 70.1 / 86.7)\n")
    hdr = f"| bs | cap 8*bs | predicted | " + " | ".join(arms) + " |"
    print(hdr)
    print("|" + "---|" * (3 + len(arms)))
    for bs in BS:
        cells = []
        for a in arms:
            d = data[a].get(bs, {}).get("moe_w13")
            cells.append(f"{d['activated']:.1f}" if d else "-")
        print(f"| {bs} | {PRED_ACTIVATED_CAP[bs]} | "
              f"{PRED_ACTIVATED[bs]:.1f} | " + " | ".join(cells) + " |")

    for short, _ in STAGES.values():
        print(f"\n## {short}: per-task us / waves / wall span per layer\n")
        print("| bs | " + " | ".join(
            f"{a} T | {a} waves | {a} wall" for a in arms) + " |")
        print("|" + "---|" * (1 + 3 * len(arms)))
        for bs in BS:
            cells = []
            for a in arms:
                d = data[a].get(bs, {}).get(short)
                cells += ([f"{d['T']:.1f}", str(d["waves"]),
                           f"{d['wall_layer']:.1f}"] if d else ["-", "-", "-"])
            print(f"| {bs} | " + " | ".join(cells) + " |")

    print("\n## Step time and decode throughput\n")
    print("| bs | I1 step | " + " | ".join(
        f"{a} step | {a} vs base" for a in arms) + " |")
    print("|" + "---|" * (2 + 2 * len(arms)))
    for bs in BS:
        b = data[base].get(bs, {}).get("step_us")
        cells = []
        for a in arms:
            s = data[a].get(bs, {}).get("step_us")
            if s is None:
                cells += ["-", "-"]
            else:
                d = f"{100.0 * (b / s - 1.0):+.1f}%" if b else "-"
                cells += [f"{s:.0f}", d]
        print(f"| {bs} | {I1_STEP[bs]:.0f} | " + " | ".join(cells) + " |")

    print("\n## Falsifier check (predictions.md F1/F2)\n")
    for bs in BS:
        for a in arms:
            if a == "base":
                continue
            d = data[a].get(bs, {}).get("moe_w13")
            if not d:
                continue
            cap = PRED_ACTIVATED_CAP[bs]
            ok = d["activated"] <= cap + 0.5
            verdict = "OK" if ok else \
                "FALSIFIED (F1): the router gate did not take effect"
            print(f"  bs{bs:<3} {a:<5} activated={d['activated']:.1f} "
                  f"cap={cap:<4} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
