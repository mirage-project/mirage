#!/usr/bin/env python3
"""AC-3 cold-run stability gate -- the fingerprint-scored engine behind
``gate_ac3_stable.sh``.

WHY THIS EXISTS
---------------
The AC-3 token gate is only as trustworthy as the run that produced it.  M3-I11
campaign 2 measured **10-16% divergence per COLD rep** on a device-state-prone
B200 (114 runs; ``opt/m3i11/CAMPAIGN2.md``) and showed that scoring by token md5
alone reads such a run as clean: the KV/GDN cache fingerprint caught 6/59
diverging reps in the arm whose token md5s were 0/59 divergent.  A perturbation
only reaches the token ids when it crosses an argmax margin (~2% of the time),
so "the tokens matched" is a *weak* statement about whether the engine actually
did the same arithmetic twice.

Determinism protocol v2 (``docs/qwen35/bench-protocol.md``) therefore binds M4's
gate to score by FINGERPRINT and to compare >=2 reps by fingerprint rather than
by token md5.  This module implements that.

WHAT IT DOES *NOT* DO
---------------------
It does not relax AC-3 by one bit.  Token equality against the committed
baseline ``results/dumps_final`` is still required, per case, for **every** rep
-- including reps that are quarantined for fingerprint divergence.  The gate
adds a second, strictly harder condition (state-level reproducibility); it never
subtracts from the first.

TWO SUBCOMMANDS
---------------
``rep``    run ONE independent cold rep at one batch size and emit
           ``bs<N>.json`` (drop-in for ``run_ac3.py --engine-dump-dir``),
           ``timings_bs<N>.json``, ``fp_<tag>.npz`` (KV/GDN wave-boundary
           fingerprints + token arrays) and ``meta_<tag>.json``.

``score``  aggregate a tree of reps into the machine-readable gate report and
           the STABLE / UNSTABLE / FAIL verdict.

Exit codes (``score``, and mirrored by the shell driver):
  0  STABLE    -- every bs reached N mutually fingerprint-identical reps and
                  every rep's tokens are byte-identical to the baseline.
  1  FAIL      -- some rep's tokens differ from the baseline.  This is an AC-3
                  correctness result, not a stability result, and outranks
                  UNSTABLE.
  2  UNSTABLE  -- some bs could not reach N mutually fingerprint-identical reps
                  inside the quarantine budget.  The observed divergence rate is
                  in the report.
  3  usage / integrity error.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ACC = HARNESS.parent
sys.path.insert(0, str(ACC))
sys.path.insert(0, str(HARNESS))

SCHEMA = "gate_ac3_stable/v1"

# The AC-3 geometry, pinned by .pm/goal.md AC-3 + docs/qwen35/bench-protocol.md.
# These are the values mpk_engine_run.py was invoked with to produce the
# committed results/dumps_final baseline (opt/m3i11/scripts/i11b_ac3.sh).
AC3_MSL = 132
AC3_NEW_TOKENS = 64
AC3_MBT = 16
AC3_PAGE_SIZE = 256
AC3_BATCH_SIZES = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------
def bitfp(t, keep_dims: int):
    """Bitwise 64-bit fingerprint of ``t``, reduced over all dims after the
    first ``keep_dims``.

    Verbatim (behaviour-identical) copy of ``opt/m3i11/scripts/e2_fingerprint.py``
    ``bitfp``, the detector whose sensitivity campaign 2 measured.  It lives here
    too so the gate is self-contained under ``harness/`` and does not reach into
    a closed issue's evidence directory -- do not "improve" it without
    re-measuring, the numbers in CAMPAIGN2.md are relative to this exact mix.

    Exact on the bit pattern: bf16/fp32 are reinterpreted as unsigned integers,
    so a 1-ULP change moves the fingerprint.  Two mixes (plain sum and
    index-weighted sum) are folded together so that a pure permutation or a
    compensating pair cannot collide.
    """
    import torch

    flat = t.reshape(*t.shape[:keep_dims], -1)
    if t.dtype == torch.bfloat16:
        u = flat.view(torch.int16).to(torch.int64) & 0xFFFF
    elif t.dtype == torch.float32:
        u = flat.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    elif t.dtype in (torch.int32, torch.int64):
        u = flat.to(torch.int64)
    else:
        raise TypeError(str(t.dtype))
    n = u.shape[-1]
    idx = torch.arange(1, n + 1, device=u.device, dtype=torch.int64)
    M = (1 << 61) - 1
    a = u.sum(-1) % M
    b = ((u * (idx * 2654435761 + 1)) % M).sum(-1) % M
    return ((a * 1000003 + b) % M).cpu().numpy()


def _pinned_device() -> dict:
    """Identify the device this process is ACTUALLY running on, from the process
    itself -- never from a candidate list.

    M3-I7 lesson: a gate that reports "GPU 6" because 6 was first in ``CANDS``
    while the guard actually claimed 3 has mislabelled every rep in its record.
    So: ask torch for the device UUID, then map that UUID back to a physical
    index through nvidia-smi.  ``CUDA_VISIBLE_DEVICES`` is recorded as a
    third-party claim to be cross-checked, not as the answer.
    """
    import torch

    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "uuid": None, "phys_index": None, "name": None}
    try:
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["uuid"] = str(getattr(props, "uuid", "") or "") or None
    except Exception as e:  # pragma: no cover - diagnostics only
        info["error"] = f"{type(e).__name__}: {e}"
    if info["uuid"]:
        try:
            rows = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=60).stdout
            want = info["uuid"].replace("GPU-", "").strip()
            for line in rows.splitlines():
                idx, _, uu = line.partition(",")
                if want and want in uu.strip():
                    info["phys_index"] = int(idx.strip())
                    break
        except Exception as e:  # pragma: no cover
            info["nvidia_smi_error"] = f"{type(e).__name__}: {e}"
    return info


def _gpu_sample(phys_index) -> dict:
    if phys_index is None:
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", str(phys_index)],
            capture_output=True, text=True, timeout=60).stdout.strip()
        used, _, util = out.partition(",")
        return {"memory_used_mib": int(used.strip()), "utilization_pct": int(util.strip())}
    except Exception as e:  # pragma: no cover
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# subcommand: rep
# ---------------------------------------------------------------------------
def cmd_rep(args) -> int:
    import numpy as np

    import admission_policy                             # noqa: E402  THE authority
    from mpk_engine_run import MPKOfflineAdapter, load_reference_requests  # noqa: E402

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    device = _pinned_device()
    gpu_before = _gpu_sample(device.get("phys_index"))
    print(f"[gate] rep={tag} bs={args.bs} device={device} gpu_before={gpu_before}",
          flush=True)

    requests = load_reference_requests(Path(args.reference))
    if len(requests) != args.expect_num_prompts:
        print(f"[gate] INTEGRITY: expected {args.expect_num_prompts} prompts, "
              f"got {len(requests)}")
        return 3

    # The cap request is passed THROUGH; admission_policy.py resolves it. This
    # gate must not restate the policy, or it would certify a configuration the
    # runtime no longer ships (admission_policy.py's module docstring).
    adapter = MPKOfflineAdapter(
        model_name=args.model, model_path=args.model_path,
        mbt=AC3_MBT, page_size=AC3_PAGE_SIZE,
        max_new_tokens=AC3_NEW_TOKENS,
        kernel_dir=Path(args.kernel_dir),
        reuse_kernel=False,               # COLD: compile this rep's own kernel
        pinned_max_seq_length=AC3_MSL,
        audit_compaction=True,            # mpk_engine_run.py's AC-3 default
        per_request_token_cap=args.per_request_token_cap,
    )

    fps: dict = {}
    state = {"n": 0}

    def snapshot(label: str) -> None:
        b = adapter._builder
        if b is None:
            return
        fps[f"{label}_k"] = bitfp(b.k_cache.reshape(b.k_cache.shape[0], -1,
                                  b.k_cache.shape[-2] * b.k_cache.shape[-1]), 2)
        fps[f"{label}_v"] = bitfp(b.v_cache.reshape(b.v_cache.shape[0], -1,
                                  b.v_cache.shape[-2] * b.v_cache.shape[-1]), 2)
        fps[f"{label}_conv"] = bitfp(b.conv_state, 2)
        fps[f"{label}_rec"] = bitfp(b.recurrent_state, 2)

    orig_reset = adapter._reset_runtime

    def reset_hook():
        # called at the START of every wave: snapshots the state the PREVIOUS
        # wave left behind (nothing on the first call).  Same hook as
        # opt/m3i11/scripts/e4_full.py, which is the protocol campaign 2 scored.
        if state["n"]:
            snapshot(f"w{state['n'] - 1}")
        state["n"] += 1
        return orig_reset()

    adapter._reset_runtime = reset_hook

    t0 = time.time()
    result = adapter.run(requests, args.bs)
    snapshot(f"w{state['n'] - 1}")
    secs = time.time() - t0

    # bs<N>.json, byte-for-byte the shape mpk_engine_run.py writes, so the same
    # directory is a drop-in for run_ac3.py --engine-dump-dir.
    dump = {pid: {"token_ids": seq.token_ids} for pid, seq in result.items()}
    dump_path = out_dir / f"bs{args.bs}.json"
    with open(dump_path, "w") as f:
        json.dump(dump, f, indent=2)
    dump_md5 = hashlib.md5(dump_path.read_bytes()).hexdigest()

    with open(out_dir / f"timings_bs{args.bs}.json", "w") as f:
        json.dump({"batch_size": args.bs,
                   "note": "informational only -- no perf claim (cold compile)",
                   "waves": adapter.timings,
                   "slot_isolation_checks": adapter.dup_checks}, f, indent=2)

    for pid, seq in result.items():
        fps[f"tok_{pid}"] = np.asarray(seq.token_ids, dtype=np.int64)
    np.savez_compressed(out_dir / f"fp_{tag}.npz", **fps)

    gpu_after = _gpu_sample(device.get("phys_index"))
    meta = {"tag": tag, "status": "ok", "bs": args.bs, "rep": args.rep,
            "msl": AC3_MSL, "new_tokens": AC3_NEW_TOKENS, "mbt": AC3_MBT,
            "page_size": AC3_PAGE_SIZE, "cold_compile": True,
            # What was REQUESTED and what the policy actually compiled -- the
            # cap is a compile-time define, so the resolved value is part of this
            # rep's identity. (348a601a moved resolution into
            # admission_policy.py and left a bare `cap` here, which raised
            # NameError on every rep; caught by M4-I1's first real run.)
            "per_request_token_cap": args.per_request_token_cap,
            "per_request_token_cap_compiled": admission_policy.resolve_int(
                args.per_request_token_cap, AC3_MBT, args.bs),
            "admission_policy": admission_policy.summary(),
            "kernel_dir": str(args.kernel_dir),
            "secs": round(secs, 1), "n_waves": state["n"],
            "dump_md5": dump_md5,
            "per_prompt_md5": {pid: hashlib.md5(
                json.dumps(s.token_ids).encode()).hexdigest()
                for pid, s in result.items()},
            "device": device, "gpu_before": gpu_before, "gpu_after": gpu_after,
            "pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())}
    with open(out_dir / f"meta_{tag}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[gate] {tag} md5={dump_md5} waves={state['n']} fp_keys={len(fps)} "
          f"{secs:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# subcommand: score
# ---------------------------------------------------------------------------
def _load_reps(root: Path, bs: int, launched: set | None = None):
    """Every rep for one batch size, in rep order.

    ``launched`` is the driver's append-only ledger. A tag in the ledger with no
    directory on disk is a LOST rep: the filesystem was full enough that even the
    rep directory could not be created, so it left no artifacts. It is reported
    as an error, never dropped -- dropping it would quietly shrink the
    denominator of the divergence rate.
    """
    import numpy as np

    reps = []
    found = set()
    for d in sorted(root.glob(f"bs{bs}_r*"), key=lambda p: _rep_index(p.name)):
        found.add(d.name)
        metas = sorted(d.glob("meta_*.json"))
        rec = {"tag": d.name, "dir": str(d), "rep_index": _rep_index(d.name)}
        # Every artifact read is fault-tolerant. A rep killed by ENOSPC leaves a
        # zero-byte meta or a truncated npz, and an unattended gate that raises
        # on one of those exits with a traceback that reads like a FAIL. A
        # damaged rep is a RUN ERROR: recorded, excluded from the consensus, and
        # never able to turn a divergence into a pass.
        if metas:
            try:
                rec["meta"] = json.loads(metas[0].read_text())
            except Exception as e:  # noqa: BLE001 - any unreadable artifact = damaged rep
                rec["meta"] = {"status": "error",
                               "note": f"unreadable {metas[0].name}: "
                                       f"{type(e).__name__}: {e}"}
        else:
            rec["meta"] = {"status": "error", "note": "no meta_*.json emitted"}
        npzs = sorted(d.glob("fp_*.npz"))
        rec["fp"] = None
        if npzs:
            try:
                with np.load(npzs[0]) as z:
                    rec["fp"] = {k: z[k] for k in z.files}
            except Exception as e:  # noqa: BLE001 - any unreadable artifact = damaged rep
                rec["meta"] = {"status": "error",
                               "note": f"unreadable {npzs[0].name}: "
                                       f"{type(e).__name__}: {e}"}
        dump = d / f"bs{bs}.json"
        rec["dump"] = None
        if dump.exists():
            try:
                rec["dump"] = json.loads(dump.read_text())
            except Exception as e:  # noqa: BLE001 - any unreadable artifact = damaged rep
                rec["meta"] = {"status": "error",
                               "note": f"unreadable {dump.name}: "
                                       f"{type(e).__name__}: {e}"}
        reps.append(rec)
    for tag in sorted(launched or (), key=_rep_index):
        if _rep_bs(tag) == bs and tag not in found:
            reps.append({"tag": tag, "dir": None, "rep_index": _rep_index(tag),
                         "meta": {"status": "error", "note": "LOST: rep was "
                                  "launched (ledger) but left no directory -- "
                                  "filesystem full at launch"},
                         "fp": None, "dump": None})
    reps.sort(key=lambda r: r["rep_index"])
    return reps


def _rep_bs(tag: str) -> int:
    try:
        return int(tag.split("_r", 1)[0][2:])
    except (IndexError, ValueError):
        return -1


def _rep_index(name: str) -> int:
    try:
        return int(name.rsplit("_r", 1)[1])
    except (IndexError, ValueError):
        return 10**6


def _sig(fp: dict, prefix_test) -> str:
    h = hashlib.sha256()
    for k in sorted(fp):
        if not prefix_test(k):
            continue
        h.update(k.encode())
        h.update(fp[k].tobytes())
    return h.hexdigest()[:16]


def _token_verdict(dump, baseline) -> dict:
    """Per-case token equality against the committed baseline. No tolerance."""
    if dump is None:
        return {"available": False, "reason": "no dump emitted"}
    if baseline is None:
        return {"available": False, "reason": "no baseline for this bs"}
    per_case, mismatched = {}, []
    for pid in sorted(set(baseline) | set(dump)):
        ref = baseline.get(pid, {}).get("token_ids")
        got = dump.get(pid, {}).get("token_ids")
        ok = ref is not None and got is not None and ref == got
        per_case[pid] = ok
        if not ok:
            first = None
            if ref is not None and got is not None:
                for i, (a, b) in enumerate(zip(ref, got)):
                    if a != b:
                        first = i
                        break
                if first is None and len(ref) != len(got):
                    first = min(len(ref), len(got))
            mismatched.append({"prompt_id": pid, "first_divergent_position": first,
                               "ref_len": None if ref is None else len(ref),
                               "got_len": None if got is None else len(got)})
    return {"available": True, "per_case": per_case,
            "n_cases": len(per_case),
            "n_identical": sum(1 for v in per_case.values() if v),
            "all_identical": all(per_case.values()) and bool(per_case),
            "mismatched": mismatched}


def _fp_deltas(fp, ref_fp) -> dict:
    import numpy as np

    keys, waves = [], set()
    for k in sorted(set(fp) | set(ref_fp)):
        a, b = fp.get(k), ref_fp.get(k)
        if a is None or b is None or a.shape != b.shape:
            keys.append({"key": k, "note": "missing-or-shape-mismatch"})
            if k.startswith("w"):
                waves.add(k.rsplit("_", 1)[0])
            continue
        bad = int(np.count_nonzero(a != b))
        if bad:
            entry = {"key": k, "entries_differ": bad, "entries_total": int(a.size),
                     "frac": round(bad / max(a.size, 1), 6)}
            if k.startswith("tok_"):
                entry["first_pos"] = int(np.argwhere(a != b)[0][0])
            keys.append(entry)
            if k.startswith("w"):
                waves.add(k.rsplit("_", 1)[0])
    return {"n_keys": len(keys), "keys": keys, "waves_touched": sorted(waves)}


def cmd_score(args) -> int:
    root = Path(args.reps_root)
    baseline_dir = Path(args.baseline)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    need = args.reps

    report = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"reps_required": need, "batch_sizes": batch_sizes,
                   "baseline": str(baseline_dir), "reps_root": str(root),
                   "max_extra_reps": args.max_extra,
                   "geometry": {"max_seq_length": AC3_MSL,
                                "max_new_tokens": AC3_NEW_TOKENS,
                                "mbt": AC3_MBT, "page_size": AC3_PAGE_SIZE,
                                "cold_compile": True},
                   "detector": "KV/GDN wave-boundary bitfp (e2_fingerprint.bitfp)"},
        "per_bs": {}, "totals": {}, "verdict": None,
    }
    if args.run_meta:
        report["run"] = json.loads(Path(args.run_meta).read_text())

    ledger = root.parent / "launched.txt"
    launched = ({t.strip() for t in ledger.read_text().split() if t.strip()}
                if ledger.exists() else set())
    report["config"]["launch_ledger"] = (str(ledger) if launched else None)

    tot = collections.Counter()
    devices = set()
    any_token_fail = False
    any_unstable = False

    for bs in batch_sizes:
        bpath = baseline_dir / f"bs{bs}.json"
        baseline = json.loads(bpath.read_text()) if bpath.exists() else None
        if baseline is None:
            print(f"[gate] INTEGRITY: baseline missing: {bpath}")
            return 3
        reps = _load_reps(root, bs, launched)
        if not reps:
            print(f"[gate] INTEGRITY: no reps found for bs={bs} under {root}")
            return 3

        # group the OK reps by state-fingerprint signature; the largest group is
        # the consensus trajectory, everything else is quarantined.
        ok = [r for r in reps if r["fp"] is not None
              and r["meta"].get("status") == "ok"]
        groups = collections.defaultdict(list)
        for r in ok:
            r["state_sig"] = _sig(r["fp"], lambda k: k.startswith("w"))
            r["token_sig"] = _sig(r["fp"], lambda k: k.startswith("tok_"))
            groups[r["state_sig"]].append(r)
        consensus_sig = (max(groups.items(), key=lambda kv: (len(kv[1]),
                                                             -_rep_index(kv[1][0]["tag"])))[0]
                         if groups else None)
        accepted = groups.get(consensus_sig, [])
        ref_fp = accepted[0]["fp"] if accepted else None

        rep_records = []
        for r in reps:
            meta = r["meta"]
            status = meta.get("status", "error")
            tv = _token_verdict(r["dump"], baseline)
            rec = {"tag": r["tag"], "rep_index": r["rep_index"],
                   "status": status,
                   "secs": meta.get("secs"), "n_waves": meta.get("n_waves"),
                   "dump_md5": meta.get("dump_md5"),
                   "device": meta.get("device"),
                   "gpu_before": meta.get("gpu_before"),
                   "gpu_after": meta.get("gpu_after"),
                   "state_sig": r.get("state_sig"),
                   "token_sig": r.get("token_sig"),
                   "tokens": tv}
            if status != "ok":
                rec["classification"] = "run_error"
                rec["error"] = meta.get("note") or meta.get("error")
                tot["errors"] += 1
            elif r["state_sig"] == consensus_sig:
                rec["classification"] = "accepted"
                tot["accepted"] += 1
            else:
                rec["classification"] = "quarantined"
                rec["fingerprint_delta_vs_consensus"] = _fp_deltas(r["fp"], ref_fp)
                tot["quarantined"] += 1
            if tv.get("available") and not tv["all_identical"]:
                rec["token_mismatch"] = True
                any_token_fail = True
                tot["token_mismatch_reps"] += 1
            # Measurement only, no bearing on the verdict: of the reps the
            # fingerprint detector flagged, how many ALSO moved the token ids?
            # This is the sub-argmax question -- a state perturbation only
            # reaches the tokens when it crosses an argmax margin somewhere in
            # the 64 decoded positions.
            if rec["classification"] == "quarantined":
                if rec.get("token_mismatch"):
                    rec["divergence_reached_tokens"] = True
                    tot["quarantined_token_reaching"] += 1
                else:
                    rec["divergence_reached_tokens"] = False
                    tot["quarantined_state_only"] += 1
            if meta.get("device", {}).get("phys_index") is not None:
                devices.add(meta["device"]["phys_index"])
            # Co-tenancy audit (measurement only, never a verdict input).
            # Compared ACROSS reps, not within one: gpu_after is sampled while
            # this rep's own ~37 GB is still resident, so an after-minus-before
            # delta measures OUR OWN allocation and says nothing about
            # co-tenants. The pre-run samples are the clean instrument -- the
            # quietest gpu_before in the window is the device's foreign floor,
            # and a rep that started above it did not have the device to itself.
            gb = meta.get("gpu_before") or {}
            if "memory_used_mib" in gb:
                rec["gpu_before_mib"] = gb["memory_used_mib"]
            rep_records.append(rec)
            tot["reps_run"] += 1

        # foreign floor for this batch size = the quietest pre-run sample seen
        befores = [r["gpu_before_mib"] for r in rep_records if "gpu_before_mib" in r]
        foreign_floor = min(befores) if befores else None
        for r in rep_records:
            if foreign_floor is not None and "gpu_before_mib" in r:
                r["mib_above_foreign_floor"] = r["gpu_before_mib"] - foreign_floor
                r["device_not_clean_at_start"] = r["mib_above_foreign_floor"] > 1024
                if r["device_not_clean_at_start"]:
                    tot["device_not_clean"] += 1

        scored = len(ok)
        n_quar = scored - len(accepted)
        bs_verdict = "STABLE"
        if len(accepted) < need:
            bs_verdict = "UNSTABLE"
            any_unstable = True
        if any(r.get("token_mismatch") for r in rep_records):
            bs_verdict = "FAIL"

        # "reps needed": how many attempts (including quarantined + errored)
        # had to be launched before `need` mutually-consistent reps existed.
        reps_needed = None
        if len(accepted) >= need:
            nth = sorted(r["rep_index"] for r in accepted)[need - 1]
            reps_needed = sum(1 for r in reps if r["rep_index"] <= nth)

        report["per_bs"][str(bs)] = {
            "verdict": bs_verdict,
            "reps_required": need,
            "reps_launched": len(reps),
            "reps_scored": scored,
            "accepted": len(accepted),
            "quarantined": n_quar,
            "errors": len(reps) - scored,
            "reps_needed_to_reach_verdict": reps_needed,
            "divergence_rate": (round(n_quar / scored, 4) if scored else None),
            "consensus_state_signature": consensus_sig,
            "distinct_state_signatures": len(groups),
            "all_reps_tokens_identical_to_baseline": all(
                r["tokens"].get("all_identical", False) for r in rep_records
                if r["tokens"].get("available")),
            "accepted_reps_tokens_identical_to_baseline": all(
                r["tokens"].get("all_identical", False) for r in rep_records
                if r["classification"] == "accepted" and r["tokens"].get("available")),
            "quarantined_token_reaching": sum(
                1 for r in rep_records if r.get("divergence_reached_tokens") is True),
            "quarantined_state_only": sum(
                1 for r in rep_records if r.get("divergence_reached_tokens") is False),
            "foreign_floor_mib": foreign_floor,
            "reps_starting_on_a_non_clean_device": sum(
                1 for r in rep_records if r.get("device_not_clean_at_start")),
            "reps": rep_records,
        }

    scored_total = tot["accepted"] + tot["quarantined"]
    report["totals"] = {
        "reps_launched": tot["reps_run"],
        "reps_scored": scored_total,
        "accepted": tot["accepted"],
        "quarantined": tot["quarantined"],
        "run_errors": tot["errors"],
        "token_mismatch_reps": tot["token_mismatch_reps"],
        "fingerprint_divergence_rate": (round(tot["quarantined"] / scored_total, 4)
                                        if scored_total else None),
        "token_divergence_rate": (round(tot["token_mismatch_reps"] / scored_total, 4)
                                  if scored_total else None),
        # of the fingerprint-divergent reps, how many surfaced in the token ids
        # (supra-argmax) vs stayed below every argmax margin (sub-argmax)
        "quarantined_token_reaching": tot["quarantined_token_reaching"],
        "quarantined_state_only": tot["quarantined_state_only"],
        "fraction_of_divergences_reaching_tokens": (
            round(tot["quarantined_token_reaching"] / tot["quarantined"], 4)
            if tot["quarantined"] else None),
        "physical_gpus_used": sorted(devices),
        "reps_starting_on_a_non_clean_device": tot["device_not_clean"],
    }

    if any_token_fail:
        report["verdict"] = "FAIL"
        rc = 1
    elif any_unstable:
        report["verdict"] = "UNSTABLE"
        rc = 2
    else:
        report["verdict"] = "STABLE"
        rc = 0

    dr = report["totals"]["fingerprint_divergence_rate"]
    report["assertion"] = _assertion(report, need, dr, rc)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    _print_summary(report)
    print(f"\n[gate] report -> {out}")
    return rc


def _assertion(report, need, dr, rc) -> str:
    bss = ",".join(sorted(report["per_bs"], key=int))
    # `dr` is None when NO rep scored (e.g. every rep errored). Formatting it
    # with %-precision raised TypeError and destroyed the report that was trying
    # to explain the failure -- M4-I1 hit exactly that.
    drs = "n/a (no rep scored)" if dr is None else f"{dr:.1%}"
    if rc == 0:
        return (f"AC-3 STABLE: at bs {{{bss}}}, {need} independent COLD reps per bs are "
                f"mutually bit-identical in KV/GDN state AND byte-identical to "
                f"results/dumps_final per case; fingerprint divergence rate "
                f"{drs} of scored reps, all divergent reps quarantined, re-run "
                f"and retained in the record.")
    if rc == 1:
        return ("AC-3 FAIL: at least one rep's token ids differ from "
                "results/dumps_final. This is a correctness result, not a "
                "stability result -- see per_bs[*].reps[*].tokens.mismatched.")
    return (f"AC-3 UNSTABLE: at least one batch size could not produce {need} "
            f"mutually fingerprint-identical COLD reps inside the quarantine "
            f"budget (observed fingerprint divergence rate {drs}). The gate "
            f"refuses to assert stability; token verdicts per rep are in the report.")


def _print_summary(report) -> None:
    t = report["totals"]
    print(f"\n=== gate_ac3_stable — verdict {report['verdict']} ===")
    print(f"  reps launched {t['reps_launched']} | scored {t['reps_scored']} | "
          f"accepted {t['accepted']} | quarantined {t['quarantined']} | "
          f"run-errors {t['run_errors']}")
    fdr = t["fingerprint_divergence_rate"]
    tdr = t["token_divergence_rate"]
    print(f"  fingerprint divergence rate: "
          f"{'n/a' if fdr is None else f'{fdr:.1%}'}   "
          f"token divergence rate: {'n/a' if tdr is None else f'{tdr:.1%}'}")
    frt = t["fraction_of_divergences_reaching_tokens"]
    print(f"  of the {t['quarantined']} fingerprint-divergent rep(s): "
          f"{t['quarantined_token_reaching']} reached the token ids "
          f"(supra-argmax), {t['quarantined_state_only']} stayed sub-argmax"
          + ("" if frt is None else f"  -> {frt:.1%} reached the tokens"))
    print(f"  physical GPUs used (from each run's own device UUID): "
          f"{t['physical_gpus_used']}")
    for bs in sorted(report["per_bs"], key=int):
        b = report["per_bs"][bs]
        print(f"  bs={bs:<3} {b['verdict']:<8} accepted {b['accepted']}/"
              f"{b['reps_required']}  quarantined {b['quarantined']}  "
              f"errors {b['errors']}  reps-needed "
              f"{b['reps_needed_to_reach_verdict']}  "
              f"tokens-vs-baseline "
              f"{'ALL IDENTICAL' if b['all_reps_tokens_identical_to_baseline'] else 'MISMATCH'}")
        for r in b["reps"]:
            extra = ""
            if r["classification"] == "quarantined":
                d = r["fingerprint_delta_vs_consensus"]
                extra = (f"  [{d['n_keys']} keys differ, waves "
                         f"{d['waves_touched'] or '(none)'}]")
            if r.get("token_mismatch"):
                extra += "  [TOKEN MISMATCH]"
            print(f"      {r['tag']:<12} {r['classification']:<12} "
                  f"state_sig={r['state_sig']} md5={r['dump_md5']}{extra}")
    print(f"\n  {report['assertion']}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rep", help="run ONE independent cold rep")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--kernel-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    p.add_argument("--model-path", default=None)
    p.add_argument("--reference",
                   default=str(ACC / "reference" / "reference_outputs.json"))
    p.add_argument("--expect-num-prompts", type=int, default=10)
    # DERIVED, not restated: "policy" hands the decision to admission_policy.py
    # so the gate certifies what the runtime ships. "none" reproduces the
    # pre-policy uncapped runs that produced results/dumps_final.
    p.add_argument("--per-request-token-cap", default="policy")
    p.set_defaults(fn=cmd_rep)

    s = sub.add_parser("score", help="aggregate reps into the gate report")
    s.add_argument("--reps-root", required=True)
    s.add_argument("--baseline",
                   default=str(ACC / "results" / "dumps_final"))
    s.add_argument("--batch-sizes", default=",".join(str(b) for b in AC3_BATCH_SIZES))
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--max-extra", type=int, default=3)
    s.add_argument("--output-json", required=True)
    s.add_argument("--run-meta", default=None)
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
