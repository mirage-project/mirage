#!/usr/bin/env python3
"""M4-I5: turn the retained per-rep records of the MPK_MOE_N_SPLITS sweep into
the A/B table, and re-derive every discard decision from the reps themselves.

Runs against a scratch root or against the committed `raw/` copy -- the layout is
the same, which is what makes the published table reproducible without a GPU.

Per rep it reads that rep's OWN `meta.cuda_visible_devices` + `meta.gpu_before`
to decide whether the pinned device was clean at start, never the candidate list
the guard was handed (M3-I7's audit-join bug: `gpu_before` records all eight
devices, and selecting by a hand-supplied list with last-match-wins scores
whichever co-tenant card came last).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

DIRTY_MIB = 500


def pinned_before(meta):
    """MiB resident on the device this rep actually ran on, at its start."""
    dev = meta.get("cuda_visible_devices")
    gb = meta.get("gpu_before")
    if dev is None or gb is None:
        return None
    want = str(dev).split(",")[0].strip()
    if isinstance(gb, str):
        for line in gb.splitlines():
            f = [x.strip() for x in line.split(",")]
            if f and f[0] == want:
                m = re.search(r"(\d+)", f[1] if len(f) > 1 else "")
                return int(m.group(1)) if m else None
        return None
    if isinstance(gb, list):
        # profile_wave.gpu_state() rows: "<index>, <used> MiB, <util> %, ..."
        for row in gb:
            if isinstance(row, str):
                f = [x.strip() for x in row.split(",")]
                if f and f[0] == want:
                    m = re.search(r"(\d+)", f[1] if len(f) > 1 else "")
                    return int(m.group(1)) if m else None
            elif isinstance(row, dict) and str(row.get("index")) == want:
                v = row.get("memory_used_mib", row.get("memory.used"))
                m = re.search(r"(\d+)", str(v))
                return int(m.group(1)) if m else None
    if isinstance(gb, dict):
        v = gb.get(want)
        if v is not None:
            m = re.search(r"(\d+)", str(v))
            return int(m.group(1)) if m else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    root = Path(a.root)

    reps = {}
    for md in sorted(root.glob("noprof*_k*")):
        m = re.match(r"noprof([A-Z])_k(\d+)$", md.name)
        if not m:
            continue
        geom, k = m.group(1), int(m.group(2))
        for f in sorted(md.glob("meta_bs*_rep*_k*.json")):
            mm = re.match(r"meta_bs(\d+)_rep(\d+)_k(\d+)\.json$", f.name)
            if not mm:
                continue
            bs, rep = int(mm.group(1)), int(mm.group(2))
            meta = json.load(open(f))
            wall = meta["waves"][0]["wall_ms"]
            mspd = meta["waves"][0].get("ms_per_decode_step")
            before = pinned_before(meta)
            reps.setdefault((geom, bs, k), []).append(dict(
                rep=rep, wall_ms=wall, ms_per_decode_step=mspd,
                device=meta.get("cuda_visible_devices"),
                mib_before=before,
                dirty=(before is None or before >= DIRTY_MIB),
                tokens_sha256=meta.get("tokens_sha256"),
                max_seq_length=meta.get("max_seq_length"),
                cap_compiled=meta.get("per_request_token_cap_compiled"),
                utc=meta.get("generated_utc"), path=str(f)))

    out = dict(root=str(root), dirty_threshold_mib=DIRTY_MIB, arms={}, table={})
    for key in sorted(reps):
        geom, bs, k = key
        rs = sorted(reps[key], key=lambda r: r["rep"])
        clean = [r for r in rs if not r["dirty"]]
        walls = [r["wall_ms"] for r in clean]
        out["arms"][f"{geom}/bs{bs}/k{k}"] = dict(
            n_reps=len(rs), n_clean=len(clean), n_discarded=len(rs) - len(clean),
            per_rep_wall_ms=[round(r["wall_ms"], 1) for r in rs],
            per_rep_mib_before=[r["mib_before"] for r in rs],
            per_rep_device=[r["device"] for r in rs],
            per_rep_tokens_sha256=[r["tokens_sha256"] for r in rs],
            median_wall_ms=round(st.median(walls), 1) if walls else None,
            spread_pct=(round(100 * (max(walls) - min(walls)) / st.median(walls), 2)
                        if len(walls) > 1 else 0.0),
            ms_per_decode_step=[r["ms_per_decode_step"] for r in rs],
            cap_compiled=sorted({r["cap_compiled"] for r in rs}),
            reps=rs)

    # ratios vs the k=2 base, per (geometry, bs)
    rows = []
    geoms = sorted({g for g, _, _ in reps})
    bss = sorted({b for _, b, _ in reps})
    ks = sorted({k for _, _, k in reps})
    for geom in geoms:
        for bs in bss:
            base = out["arms"].get(f"{geom}/bs{bs}/k2")
            if not base or base["median_wall_ms"] is None:
                continue
            row = dict(geom=geom, bs=bs, base_k=2,
                       base_median_ms=base["median_wall_ms"],
                       base_spread_pct=base["spread_pct"], n_clean=base["n_clean"])
            for k in ks:
                arm = out["arms"].get(f"{geom}/bs{bs}/k{k}")
                if not arm or arm["median_wall_ms"] is None:
                    continue
                row[f"k{k}_median_ms"] = arm["median_wall_ms"]
                row[f"k{k}_spread_pct"] = arm["spread_pct"]
                row[f"k{k}_n_clean"] = arm["n_clean"]
                row[f"k{k}_vs_k2"] = round(arm["median_wall_ms"]
                                           / base["median_wall_ms"], 4)
                row[f"k{k}_speedup"] = round(base["median_wall_ms"]
                                             / arm["median_wall_ms"], 4)
                # token identity across arms (P5), compared PER REP.  The
                # synthetic seed is SEED_BASE + bs*1000 + rep, so reps have
                # DIFFERENT prompts and therefore legitimately different
                # outputs; the claim is that at a fixed rep every arm produces
                # the same bytes.  Comparing arm-wide sets instead would call a
                # prompt change a split-induced difference.
                bmap = {r["rep"]: r["tokens_sha256"] for r in base["reps"]}
                kmap = {r["rep"]: r["tokens_sha256"] for r in arm["reps"]}
                shared = sorted(set(bmap) & set(kmap))
                row[f"k{k}_reps_compared"] = len(shared)
                row[f"k{k}_tokens_identical_to_k2"] = bool(
                    shared and all(bmap[r] == kmap[r] for r in shared))
                row[f"k{k}_distinct_shas_over_reps"] = len(set(bmap.values()))
            rows.append(row)
    out["table"] = rows

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    hdr = f"{'geom':5s}{'bs':>4s}" + "".join(f"{'k'+str(k)+' ms':>11s}{'sp%':>6s}"
                                             for k in ks)
    hdr += "".join(f"{'k'+str(k)+'/k2':>8s}" for k in ks if k != 2)
    hdr += f"{'tok=':>6s}{'n':>3s}"
    print(hdr)
    for r in rows:
        line = f"{r['geom']:5s}{r['bs']:4d}"
        for k in ks:
            line += (f"{r.get(f'k{k}_median_ms', 0) or 0:11.1f}"
                     f"{r.get(f'k{k}_spread_pct', 0) or 0:6.2f}")
        for k in ks:
            if k == 2:
                continue
            line += f"{r.get(f'k{k}_vs_k2', 0) or 0:8.4f}"
        ident = all(r.get(f"k{k}_tokens_identical_to_k2", True)
                    for k in ks if k != 2)
        line += f"{'Y' if ident else 'N':>6s}{r['n_clean']:3d}"
        print(line)
    disc = sum(v["n_discarded"] for v in out["arms"].values())
    tot = sum(v["n_reps"] for v in out["arms"].values())
    print(f"\n{tot} reps, {disc} discarded (pinned device >= {DIRTY_MIB} MiB at start)")
    if a.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["geom", "bs"]
                               + [k for k in keys if k not in ("geom", "bs")])
            wr.writeheader()
            for r in rows:
                wr.writerow(r)


if __name__ == "__main__":
    main()
