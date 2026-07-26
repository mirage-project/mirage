#!/usr/bin/env python3
"""M3-I2b A/B analysis: before/after step time, per-task attribution, occupancy.

Reads the parsed tables + meta produced by run_m3i2b.sh for two (or more) arms
and emits:
  ab_step.csv          per-bs step_us / decode tok/s / occupancy / concurrency,
                       one row per arm plus the delta
  ab_pertask.csv       per task type: total_us, wall_span_us, mean_us, per arm
  ab_tokens.json       token-id sha per (arm, bs, rep) -- AC-3 cross-check
  ab.md                the human-readable table

Usage: python3 analyze_m3i2b.py <root> <armA> <armB> [...]
"""
import csv
import json
import statistics
import sys
from pathlib import Path

BS = [1, 2, 4, 8, 16]
I1_STEP = {1: 15264.0, 2: 15647.6, 4: 15645.1, 8: 18618.2, 16: 22005.2}


def load(root: Path, arm: str, bs: int):
    p = root / f"tables_{arm}" / f"bs{bs}_attrib.json"
    return json.load(open(p)) if p.exists() else None


def conc(root: Path, arm: str, bs: int):
    p = root / f"tables_{arm}" / f"bs{bs}_concurrency.json"
    return json.load(open(p)) if p.exists() else None


def wave_ms(root: Path, arm: str, bs: int, profiled: bool):
    """Median unprofiled/profiled wave wall ms over the reps that exist."""
    d = root / (f"prof_{arm}" if profiled else f"noprof_{arm}")
    vals, shas = [], []
    for rep in range(8):
        p = d / f"meta_bs{bs}_rep{rep}.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        vals.append(m["run_seconds"] * 1000.0)
        shas.append(m["tokens_sha256"])
    if not vals:
        return None, None, None, []
    med = statistics.median(vals)
    spread = (max(vals) - min(vals)) / med * 100 if med else 0.0
    return med, spread, len(vals), shas


def steps(root: Path, arm: str, bs: int):
    """step_us from every profiled rep that was parsed (rep0) + the summary."""
    a = load(root, arm, bs)
    if a is None:
        return None
    s = a["summary"] if "summary" in a else a
    return s


def main():
    root = Path(sys.argv[1])
    arms = sys.argv[2:]
    assert len(arms) >= 1

    rows, pertask_rows, tokens = [], [], {}
    md = ["# M3-I2b A/B", "", "## Step time (profiled steady decode step)", ""]
    md.append("| bs | " + " | ".join(f"{a} step us" for a in arms)
              + " | delta% | " + " | ".join(f"{a} tok/s" for a in arms)
              + " | I1 step us |")
    md.append("|---:|" + "---:|" * (2 * len(arms) + 2))

    for bs in BS:
        cells, tps = [], []
        base_step = None
        for a in arms:
            s = steps(root, a, bs)
            if s is None:
                cells.append("-")
                tps.append("-")
                continue
            st = s["step_us"]
            cells.append(f"{st:.0f}")
            tps.append(f"{s['decode_tokens_per_s']:.1f}")
            if base_step is None:
                base_step = st
            c = conc(root, a, bs)
            rows.append(dict(
                arm=a, bs=bs, step_us=round(st, 1),
                step_us_p50=round(s.get("step_us_p50", 0), 1),
                step_us_min=round(s.get("step_us_min", 0), 1),
                step_us_max=round(s.get("step_us_max", 0), 1),
                step_spread_pct=round(
                    100 * (s.get("step_us_max", 0) - s.get("step_us_min", 0))
                    / max(st, 1e-9), 2),
                decode_tok_s=round(s["decode_tokens_per_s"], 1),
                occupancy=round(s.get("occupancy", 0), 4),
                mean_concurrency=round((c or {}).get("mean_concurrency", 0), 2),
                task_sum_us=round(s.get("task_sum_us", 0), 1),
                perfect_pack_us=round(s.get("perfect_pack_us", 0), 1),
                worker_idle_us=round(s.get("worker_idle_us", 0), 1),
                dead_all_idle_us=round(s.get("dead_all_idle_us", 0), 1),
                closure_error_pct=round(s.get("closure_error_pct", 0), 3),
                us_at_conc_le16=round(
                    (c["us_at_concurrency"]["c1_4"]
                     + c["us_at_concurrency"]["c5_16"]) if c else 0, 1),
            ))
        last = steps(root, arms[-1], bs)
        d = (f"{100 * (base_step - last['step_us']) / last['step_us']:+.1f}"
             if last and base_step else "-")
        md.append(f"| {bs} | " + " | ".join(cells) + f" | {d} | "
                  + " | ".join(tps) + f" | {I1_STEP[bs]:.0f} |")

    md += ["", "## Wave wall time (unprofiled, median of reps)", "",
           "| bs | " + " | ".join(f"{a} ms (spread%, n)" for a in arms) + " |",
           "|---:|" + "---:|" * len(arms)]
    for bs in BS:
        cells = []
        for a in arms:
            med, spread, n, shas = wave_ms(root, a, bs, False)
            cells.append("-" if med is None
                         else f"{med:.0f} ({spread:.2f}%, {n})")
            if shas:
                tokens[f"{a}/bs{bs}/noprof"] = sorted(set(shas))
            pm, ps, pn, pshas = wave_ms(root, a, bs, True)
            if pshas:
                tokens[f"{a}/bs{bs}/prof"] = sorted(set(pshas))
        md.append(f"| {bs} | " + " | ".join(cells) + " |")

    md += ["", "## Per task type (bs1 / bs16), total worker us and wall span", ""]
    for bs in (1, 16):
        md += [f"### bs{bs}", "",
               "| task | " + " | ".join(
                   f"{a} tot us | {a} wall us | {a} mean us" for a in arms)
               + " |",
               "|---|" + "---:|" * (3 * len(arms))]
        names = []
        for a in arms:
            c = conc(root, a, bs)
            if c:
                names += list(c["per_task_concurrency"])
        seen = []
        for n in names:
            if n not in seen:
                seen.append(n)
        for name in seen:
            cells = []
            for a in arms:
                c = conc(root, a, bs)
                e = (c or {}).get("per_task_concurrency", {}).get(name)
                if not e:
                    cells += ["-", "-", "-"]
                    continue
                cells += [f"{e['total_us']:.0f}", f"{e['wall_span_us']:.0f}",
                          f"{e['total_us'] / max(e['n'], 1):.2f}"]
                pertask_rows.append(dict(
                    arm=a, bs=bs, task=name, n=e["n"],
                    total_us=round(e["total_us"], 1),
                    wall_span_us=round(e["wall_span_us"], 1),
                    mean_us=round(e["total_us"] / max(e["n"], 1), 3),
                    mean_concurrency_during=round(
                        e["mean_concurrency_during"], 2)))
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")

    (root / "ab.md").write_text("\n".join(md) + "\n")
    if rows:
        with open(root / "ab_step.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if pertask_rows:
        with open(root / "ab_pertask.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pertask_rows[0]))
            w.writeheader()
            w.writerows(pertask_rows)
    json.dump(tokens, open(root / "ab_tokens.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
