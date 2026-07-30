#!/usr/bin/env python3
"""M4-I6 tables: the MoE-router integration A/B, per-rep and median.

Two arms, TWO CLONES (this change has no runtime switch -- the router body is a
header, so the arm is selected by which tree built the run):
  A  mirage-m4i6-base  -- HEAD~1, the M4-I2-terminal router (M3-I5b + I5c + I8)
  B  mirage-m4i6       -- + the ferret workspace5 v013 router

Every rep is listed, not just the median, so the >=3-rep rule can be checked arm
by arm without opening a raw artifact.

THE gpu_before AUDIT IS DERIVED PER RUN. Each meta json records its own
`cuda_visible_devices` and its own `gpu_before` snapshot of all eight devices;
the pinned device's occupancy is read out of THAT pair. It is never taken from
the guard's candidate list -- M3-I7's phantom-dirty-rep bug was exactly that
substitution, and it silently discarded three clean reps.

The tokens_sha256 equality column is a second, independent correctness signal:
the kernel-level gate proved the two paths bit-exact on synthetic tensors, and
this shows the whole 40-layer megakernel emits identical token ids from the two
arms on real weights.
"""
import argparse
import glob
import json
import os
import statistics
import sys

ARMS = ["A", "B"]
ARM_LABEL = {"A": "base(M3-I5b/I5c/I8 router)", "B": "new(ferret v013 router)"}
# A run whose pinned device already held more than this at start is dirty.
DIRTY_MIB = 500


def pinned_used_mib(meta):
    """MiB in use on THIS run's own pinned device, from its own gpu_before."""
    dev = meta.get("cuda_visible_devices")
    if dev is None:
        return None, "no cuda_visible_devices recorded"
    dev = str(dev).split(",")[0].strip()
    for row in meta.get("gpu_before", []):
        parts = [p.strip() for p in row.split(",")]
        if parts and parts[0] == dev:
            mib = int(parts[1].split()[0])
            return mib, None
    return None, f"device {dev} absent from this run's gpu_before"


def load(root):
    out = {}
    for arm in ARMS:
        for f in sorted(glob.glob(os.path.join(root, f"noprof_{arm}",
                                               f"meta_bs*_rep*_{arm}.json"))):
            m = json.load(open(f))
            bs, rep = m["batch_size"], m["rep"]
            waves = m.get("waves") or []
            if len(waves) != 1:
                print(f"WARN {f}: {len(waves)} waves, expected 1", file=sys.stderr)
            w = waves[0] if waves else {}
            used, err = pinned_used_mib(m)
            out[(arm, bs, rep)] = {
                "wall_ms": w.get("wall_ms"),
                "ms_per_step": w.get("ms_per_decode_step"),
                "steps": w.get("max_decode_steps"),
                "tokens_sha256": m.get("tokens_sha256"),
                "dev": m.get("cuda_visible_devices"),
                "pinned_used_mib": used,
                "audit_err": err,
                "run_seconds": m.get("run_seconds"),
                "path": f,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/var/tmp/m4i6_sweep")
    ap.add_argument("--out", required=True, help="artifact dir for the tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.root)
    bss = sorted({k[1] for k in d})
    reps = sorted({k[2] for k in d})

    lines = []
    P = lines.append
    P("M4-I6 MoE-router integration A/B -- geometry B (synthetic 256-token")
    P("prompts, msl=353, 96 decode steps), arms interleaved per (bs, rep) in one")
    P("GPU claim.")
    P("")
    P(f"arm A = {ARM_LABEL['A']}")
    P(f"arm B = {ARM_LABEL['B']}")
    P("")

    # ---- dirty-rep audit first: a perf table above an unaudited run is noise --
    dirty, missing = [], []
    for k, v in sorted(d.items()):
        if v["audit_err"]:
            missing.append((k, v["audit_err"]))
        elif v["pinned_used_mib"] is not None and v["pinned_used_mib"] > DIRTY_MIB:
            dirty.append((k, v["pinned_used_mib"]))
    P("== gpu_before audit (each run's OWN pinned device, from its OWN record) ==")
    P(f"   runs: {len(d)}   dirty(>{DIRTY_MIB}MiB at start): {len(dirty)}   "
      f"unauditable: {len(missing)}")
    devs = sorted({str(v['dev']) for v in d.values()})
    P(f"   pinned device(s) across all runs: {','.join(devs)}")
    floors = sorted({v['pinned_used_mib'] for v in d.values()
                     if v['pinned_used_mib'] is not None})
    P(f"   observed pinned-device floors (MiB): {floors}")
    for k, m in dirty:
        P(f"   DIRTY {k}: {m} MiB")
    for k, e in missing:
        P(f"   UNAUDITABLE {k}: {e}")
    P("")

    # ---- per-rep table --------------------------------------------------------
    P("== per-rep wall_ms (every rep, both arms) ==")
    hdr = f"{'bs':>3} {'arm':>3} " + " ".join(f"{'rep'+str(r):>10}" for r in reps) \
        + f" {'median':>10} {'range':>8}"
    P(hdr)
    med = {}
    for bs in bss:
        for arm in ARMS:
            vals = [d[(arm, bs, r)]["wall_ms"] for r in reps if (arm, bs, r) in d]
            cells = " ".join(
                f"{d[(arm,bs,r)]['wall_ms']:10.1f}" if (arm, bs, r) in d
                else f"{'--':>10}" for r in reps)
            if vals:
                med[(arm, bs)] = statistics.median(vals)
                rng = max(vals) - min(vals)
                P(f"{bs:>3} {arm:>3} {cells} {med[(arm,bs)]:10.1f} {rng:8.1f}")
            else:
                P(f"{bs:>3} {arm:>3} {cells} {'--':>10} {'--':>8}")
    P("")

    # ---- the A/B result -------------------------------------------------------
    P("== median wall_ms, and the integration's e2e effect ==")
    P(f"{'bs':>3} {'A base':>10} {'B new':>10} {'delta_ms':>10} "
      f"{'speedup':>8} {'B ms/step':>10} {'A ms/step':>10}")
    rows = []
    for bs in bss:
        if (("A", bs) not in med) or (("B", bs) not in med):
            continue
        A, Bv = med[("A", bs)], med[("B", bs)]
        sp = A / Bv
        bstep = statistics.median([d[("B", bs, r)]["ms_per_step"]
                                   for r in reps if ("B", bs, r) in d])
        astep = statistics.median([d[("A", bs, r)]["ms_per_step"]
                                   for r in reps if ("A", bs, r) in d])
        P(f"{bs:>3} {A:10.1f} {Bv:10.1f} {Bv-A:10.1f} {sp:7.4f}x "
          f"{bstep:10.4f} {astep:10.4f}")
        rows.append({"bs": bs, "A_median_ms": A, "B_median_ms": Bv,
                     "delta_ms": Bv - A, "speedup": sp,
                     "A_ms_per_step": astep, "B_ms_per_step": bstep,
                     "A_reps": [d[("A", bs, r)]["wall_ms"] for r in reps
                                if ("A", bs, r) in d],
                     "B_reps": [d[("B", bs, r)]["wall_ms"] for r in reps
                                if ("B", bs, r) in d]})
    P("")

    # ---- token identity ------------------------------------------------------
    P("== tokens_sha256: does the megakernel emit identical tokens in both arms? ==")
    P("   (gate 1 proved live-row bit-exactness on synthetic tensors and HF")
    P("    dumps; this is the same property through 40 real layers, and it is")
    P("    also the check that the padding-row zero-weight divergence stays")
    P("    unobservable in the emitted tokens)")
    tok_ok, tok_bad = 0, []
    for bs in bss:
        for r in reps:
            ka, kb = ("A", bs, r), ("B", bs, r)
            if ka in d and kb in d:
                sa, sb = d[ka]["tokens_sha256"], d[kb]["tokens_sha256"]
                if sa == sb:
                    tok_ok += 1
                else:
                    tok_bad.append((bs, r, sa, sb))
    P(f"   identical: {tok_ok}   differing: {len(tok_bad)}")
    for bs, r, sa, sb in tok_bad:
        P(f"   DIFFER bs{bs} rep{r}: A={sa[:16]} B={sb[:16]}")
    P("")

    txt = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "m4i6_tables.txt"), "w").write(txt)
    json.dump({"arms": ARM_LABEL, "rows": rows,
               "per_run": {f"{k[0]}_bs{k[1]}_rep{k[2]}": v for k, v in d.items()},
               "audit": {"n_runs": len(d), "dirty": len(dirty),
                         "unauditable": len(missing),
                         "devices": devs, "floors_mib": floors},
               "tokens_identical": tok_ok, "tokens_differing": len(tok_bad)},
              open(os.path.join(a.out, "m4i6_tables.json"), "w"), indent=1)
    with open(os.path.join(a.out, "ab_per_rep.csv"), "w") as fh:
        fh.write("bs,arm,rep,wall_ms,ms_per_step,steps,pinned_dev,"
                 "pinned_used_mib,tokens_sha256\n")
        for (arm, bs, r), v in sorted(d.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
            fh.write(f"{bs},{arm},{r},{v['wall_ms']},{v['ms_per_step']},"
                     f"{v['steps']},{v['dev']},{v['pinned_used_mib']},"
                     f"{v['tokens_sha256']}\n")
    print(txt)
    return 1 if (tok_bad or dirty or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
