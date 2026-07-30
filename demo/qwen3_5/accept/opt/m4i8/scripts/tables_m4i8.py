#!/usr/bin/env python3
"""M4-I8 tables: the scheduler A/B, per-rep and median.

Three arms, all default-off compile-time knobs on ONE tree:
  A  no -D                       -- HEAD
  S  MPK_EVENT_WAIT_GPU_SCOPE=1  -- ld.acquire.gpu instead of ld.acquire.sys on
                                    the local-event-counter spin
  O  MPK_WORKER_OOO_POP=1        -- out-of-order pop inside the loaded task-desc
                                    buffer (run as the simulator's falsifier)

Every non-A arm is reported against arm A at the SAME batch size, from reps that
were interleaved with A inside one GPU claim.

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

ARMS = ["A", "S", "O"]
ARM_LABEL = {"A": "base(HEAD, both knobs off)",
             "S": "MPK_EVENT_WAIT_GPU_SCOPE=1 (ld.acquire.gpu)",
             "O": "MPK_WORKER_OOO_POP=1 (out-of-order pop)"}
NEW_ARMS = ["S", "O"]
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
    ap.add_argument("--root", default="/var/tmp/m4i8_sweep")
    ap.add_argument("--out", required=True, help="artifact dir for the tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.root)
    bss = sorted({k[1] for k in d})
    reps = sorted({k[2] for k in d})

    lines = []
    P = lines.append
    P("M4-I8 scheduler A/B -- geometry B (synthetic 256-token prompts, msl=353,")
    P("96 decode steps, mbt=16), arms interleaved per (bs, rep) in one GPU claim.")
    P("")
    for k in ARMS:
        P(f"arm {k} = {ARM_LABEL[k]}")
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
    P("== per-rep wall_ms (every rep, every arm) ==")
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
    P("== median wall_ms, and each arm's e2e effect against arm A ==")
    P(f"{'bs':>3} {'arm':>3} {'A base':>10} {'arm':>10} {'delta_ms':>10} "
      f"{'speedup':>9} {'arm ms/step':>12} {'A ms/step':>11}")
    rows = []
    for bs in bss:
        if ("A", bs) not in med:
            continue
        A = med[("A", bs)]
        astep = statistics.median([d[("A", bs, r)]["ms_per_step"]
                                   for r in reps if ("A", bs, r) in d])
        for arm in NEW_ARMS:
            if (arm, bs) not in med:
                continue
            X = med[(arm, bs)]
            xstep = statistics.median([d[(arm, bs, r)]["ms_per_step"]
                                       for r in reps if (arm, bs, r) in d])
            P(f"{bs:>3} {arm:>3} {A:10.1f} {X:10.1f} {X-A:10.1f} {A/X:8.4f}x "
              f"{xstep:12.4f} {astep:11.4f}")
            rows.append({"bs": bs, "arm": arm, "A_median_ms": A,
                         "arm_median_ms": X, "delta_ms": X - A,
                         "speedup": A / X, "A_ms_per_step": astep,
                         "arm_ms_per_step": xstep,
                         "A_reps": [d[("A", bs, r)]["wall_ms"] for r in reps
                                    if ("A", bs, r) in d],
                         "arm_reps": [d[(arm, bs, r)]["wall_ms"] for r in reps
                                      if (arm, bs, r) in d]})
    P("")

    # ---- token identity ------------------------------------------------------
    P("== tokens_sha256: does the megakernel emit identical tokens in every arm? ==")
    P("   Arm S changes only a load's memory scope, so identity is expected by")
    P("   construction; arm O reorders execution among tasks that are already")
    P("   independent, so identity is the empirical claim. Either way this is the")
    P("   same property through 40 real layers on the real checkpoint.")
    tok_ok, tok_bad = 0, []
    for bs in bss:
        for r in reps:
            ka = ("A", bs, r)
            if ka not in d:
                continue
            for arm in NEW_ARMS:
                kx = (arm, bs, r)
                if kx not in d:
                    continue
                sa, sx = d[ka]["tokens_sha256"], d[kx]["tokens_sha256"]
                if sa == sx:
                    tok_ok += 1
                else:
                    tok_bad.append((arm, bs, r, sa, sx))
    P(f"   identical: {tok_ok}   differing: {len(tok_bad)}")
    for arm, bs, r, sa, sx in tok_bad:
        P(f"   DIFFER arm{arm} bs{bs} rep{r}: A={sa[:16]} {arm}={sx[:16]}")
    P("")

    txt = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "m4i8_tables.txt"), "w").write(txt)
    json.dump({"arms": ARM_LABEL, "rows": rows,
               "per_run": {f"{k[0]}_bs{k[1]}_rep{k[2]}": v for k, v in d.items()},
               "audit": {"n_runs": len(d), "dirty": len(dirty),
                         "unauditable": len(missing),
                         "devices": devs, "floors_mib": floors},
               "tokens_identical": tok_ok, "tokens_differing": len(tok_bad)},
              open(os.path.join(a.out, "m4i8_tables.json"), "w"), indent=1)
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
