#!/usr/bin/env python3
"""Minimal in-process multi-wave reproducer for HAZARD-WAVE-RESET (M3-I2a).

Runs the SAME wave of prompts N times inside ONE process, calling the
launcher's ``init_request_func`` between waves exactly as
``mpk_engine_run.py`` does. Two things are asserted, in order:

1. **No wedge.** Every wave completes; a wave that stops making progress is
   reported with the per-request `step`, `qo_indptr` and `kv_indptr` the
   runtime actually reached, rather than hanging silently.
2. **Reset is exact.** Wave *k* must emit byte-identical token ids to wave 0.
   Same prompts, same slots, same kernel: any difference is reset state that
   leaked across the launch boundary, which is the same defect class as the
   wedge and would otherwise pass a "did not hang" check.

Repeating one wave (rather than walking distinct waves) isolates the reset:
prompt content, slot geometry and the compiled kernel are all held fixed, so
the only variable is how many launches have already happened in this process.

    python two_wave_repro.py --batch-size 4 --waves 3 \
        --kernel-dir ~/mpk-qwen35/m3i2a/kernel_bs4 --reuse-kernel \
        --max-seq-length 132 --out /tmp/repro_bs4.json

The GPU must be EXCLUSIVE. MPK's grid claims every SM and its blocks spin-wait
on each other, so a co-tenant stops the grid from becoming co-resident and the
megakernel deadlocks -- which is exactly what M2-I9 mistook for a broken
in-process reset. The launcher's residency check refuses such a launch; if you
see it fire, free the GPU rather than setting MPK_SKIP_RESIDENCY_CHECK=1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ACCEPT = HERE.parent.parent
sys.path.insert(0, str(ACCEPT))
sys.path.insert(0, str(ACCEPT / "harness"))

from ac3_types import PromptRequest  # noqa: E402
from mpk_engine_run import MPKOfflineAdapter, load_reference_requests  # noqa: E402


def log(msg: str) -> None:
    print(f"[two_wave] {msg}", flush=True)


def run_wave(adapter: MPKOfflineAdapter, slots: List[PromptRequest],
             w_idx: int, timeout_s: float,
             poll_s: float = 2.0) -> Dict[int, List[int]]:
    import torch

    meta = adapter._meta
    meta["tokens"].zero_()
    meta["step"].zero_()
    meta["num_new_tokens"].fill_(1)
    for r_i, req in enumerate(slots):
        ids = req.input_ids
        meta["tokens"][r_i, :len(ids)] = torch.tensor(
            ids, dtype=torch.long, device="cuda")
        meta["prompt_lengths"][r_i] = len(ids)
    adapter._reset_runtime()

    torch.cuda.synchronize()
    t0 = time.time()
    adapter._mpk()

    side = torch.cuda.Stream()
    done = torch.cuda.Event()
    done.record()
    last, stalled = None, 0.0
    while not done.query():
        if stalled > timeout_s:
            with torch.cuda.stream(side):
                steps = meta["step"].to("cpu").tolist()
                qo = meta["qo_indptr_buffer"].to("cpu").tolist()
                kvp = meta["paged_kv_indptr_buffer"].to("cpu").tolist()
            side.synchronize()
            raise RuntimeError(
                f"WEDGED: wave={w_idx} no progress for {stalled:.0f}s "
                f"step={steps} plen={[len(r.input_ids) for r in slots]} "
                f"qo_indptr={qo} kv_indptr={kvp}")
        time.sleep(poll_s)
        with torch.cuda.stream(side):
            cur = meta["step"].to("cpu").tolist()
        side.synchronize()
        stalled = stalled + poll_s if cur == last else 0.0
        last = cur
    torch.cuda.synchronize()

    toks = meta["tokens"].cpu().tolist()
    per_slot = {r_i: toks[r_i][len(req.input_ids):
                              len(req.input_ids) + adapter.max_new_tokens]
                for r_i, req in enumerate(slots)}
    log(f"wave={w_idx} ok in {time.time() - t0:.1f}s "
        f"steps={meta['step'].tolist()}")
    return per_slot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--reference",
                    default=str(ACCEPT / "reference" / "reference_outputs.json"))
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--mode", choices=("repeat", "distinct", "cycle"),
                    default="repeat",
                    help="repeat: run the SAME wave every time (isolates the "
                         "reset; wave k must be byte-identical to wave 0). "
                         "distinct: walk successive waves of the prompt set, "
                         "which is what mpk_engine_run.py does. cycle: walk "
                         "them round-robin for --waves launches, so a long "
                         "run keeps changing prompt geometry across the "
                         "launch boundary (the stressor).")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=132)
    ap.add_argument("--mbt", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--kernel-dir", default=None)
    ap.add_argument("--reuse-kernel", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="Seconds of zero `step` progress before declaring a "
                         "wedge.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    requests = load_reference_requests(Path(args.reference))
    # Same slot-filling rule as the AC-3 adapter: ascending prompt length, so
    # retirement never compacts a reported slot (HAZARD-COMPACTION).
    ordered = sorted(requests, key=lambda r: len(r.input_ids))
    bs = args.batch_size
    if args.mode == "repeat":
        base = ordered[:bs]
        while len(base) < bs:
            base.append(ordered[len(base) % len(ordered)])
        wave_slots = [list(base) for _ in range(args.waves)]
    else:
        distinct = []
        for i in range(0, len(ordered), bs):
            wave = ordered[i:i + bs]
            slots = list(wave)
            while len(slots) < bs:
                slots.append(wave[len(slots) % len(wave)])
            distinct.append(slots)
        if args.mode == "distinct":
            if args.waves > len(distinct):
                raise SystemExit(
                    f"prompt set only yields {len(distinct)} distinct waves "
                    f"at bs={bs}; use --mode cycle for more")
            wave_slots = distinct[:args.waves]
        else:
            wave_slots = [distinct[w % len(distinct)]
                          for w in range(args.waves)]
    log(f"bs={bs} waves={args.waves} mode={args.mode} plens per wave="
        f"{[[len(r.input_ids) for r in s] for s in wave_slots]}")

    adapter = MPKOfflineAdapter(
        model_name=args.model, model_path=args.model_path, mbt=args.mbt,
        page_size=args.page_size, max_new_tokens=args.max_new_tokens,
        kernel_dir=Path(args.kernel_dir) if args.kernel_dir else None,
        reuse_kernel=args.reuse_kernel,
        pinned_max_seq_length=args.max_seq_length)
    adapter._build(args.batch_size, args.max_seq_length, args.batch_size)

    report = {"batch_size": bs, "waves": args.waves, "mode": args.mode,
              "max_seq_length": args.max_seq_length, "mbt": args.mbt,
              "prompt_ids": [[r.prompt_id for r in s] for s in wave_slots],
              "prompt_lens": [[len(r.input_ids) for r in s]
                              for s in wave_slots],
              "wave_results": [], "identical_to_wave0": [], "status": "unknown"}
    baseline = None
    try:
        for w, slots in enumerate(wave_slots):
            per_slot = run_wave(adapter, slots, w, args.timeout)
            report["wave_results"].append(
                {str(k): v for k, v in per_slot.items()})
            if args.mode != "repeat":
                continue
            if baseline is None:
                baseline = per_slot
                report["identical_to_wave0"].append(True)
            else:
                same = all(per_slot[k] == baseline[k] for k in baseline)
                report["identical_to_wave0"].append(same)
                if not same:
                    diffs = [k for k in baseline if per_slot[k] != baseline[k]]
                    log(f"WAVE {w} TOKEN MISMATCH vs wave 0, slots {diffs}")
    except Exception as exc:  # noqa: BLE001 — recorded, then re-raised as exit
        report["status"] = "wedged"
        report["error"] = str(exc)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        log(f"FAIL: {exc}")
        log(f"wrote {out}")
        return 3

    report["status"] = ("pass" if all(report["identical_to_wave0"])
                        else "token_mismatch")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log(f"{report['status']}: {args.waves} waves, "
        f"identical_to_wave0={report['identical_to_wave0']}")
    log(f"wrote {out}")
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    sys.exit(main())
