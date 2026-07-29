#!/usr/bin/env python3
"""M3-I11 E4 -- reproduce the m3i9 stage-7 configuration exactly, with the
KV fingerprint attached.

Same command shape as plan_m3i9.sh stage 7 (all ten reference prompts, one
pinned max_seq_length, 1024 new tokens, warm cached kernel), so the emitted
dump's md5 is directly comparable to opt/m3i9b/results/census_window2.json.
On top of that it snapshots a KV/GDN fingerprint at every wave boundary, which
sees a perturbation whether or not it ever crosses an argmax margin.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ACC = Path(os.environ.get("ACC", str(Path.home() / "mpk-qwen35/mirage/demo/qwen3_5/accept")))
sys.path.insert(0, str(ACC))
sys.path.insert(0, str(ACC / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpk_engine_run import MPKOfflineAdapter, load_reference_requests  # noqa: E402
from e2_fingerprint import bitfp                                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--msl", type=int, default=1280)
    ap.add_argument("--new-tokens", type=int, default=1024)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--fresh-compile", action="store_true")
    ap.add_argument("--reference", default=str(ACC / "reference/reference_outputs.json"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    requests = load_reference_requests(Path(args.reference))

    adapter = MPKOfflineAdapter(
        model_name="Qwen/Qwen3.5-35B-A3B-FP8", mbt=16, page_size=256,
        max_new_tokens=args.new_tokens,
        kernel_dir=Path(args.kernel_dir), reuse_kernel=not args.fresh_compile,
        pinned_max_seq_length=args.msl, audit_compaction=False,
        # Pinned to the PRE-POLICY uncapped runtime: this probe's committed
        # census is the comparison basis, and the shipped default is now the
        # admission policy (accept/admission_policy.py, M4-I4).
        per_request_token_cap="none")

    fps: dict[str, np.ndarray] = {}
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
        # wave left behind (nothing on the first call).
        if state["n"]:
            snapshot(f"w{state['n']-1}")
        state["n"] += 1
        return orig_reset()

    adapter._reset_runtime = reset_hook

    t0 = time.time()
    result = adapter.run(requests, args.bs)
    snapshot(f"w{state['n']-1}")
    secs = time.time() - t0

    dump = {pid: {"token_ids": seq.token_ids} for pid, seq in result.items()}
    dump_path = out_dir / f"dump_{args.tag}.json"
    with open(dump_path, "w") as f:
        json.dump(dump, f, indent=2)
    md5 = hashlib.md5(open(dump_path, "rb").read()).hexdigest()
    for pid, seq in result.items():
        fps[f"tok_{pid}"] = np.asarray(seq.token_ids, dtype=np.int64)
    np.savez_compressed(out_dir / f"fp_{args.tag}.npz", **fps)
    json.dump({"tag": args.tag, "bs": args.bs, "msl": args.msl,
               "fresh_compile": args.fresh_compile, "secs": round(secs, 1),
               "dump_md5": md5, "n_waves": state["n"],
               "per_prompt_md5": {pid: hashlib.md5(
                   json.dumps(s.token_ids).encode()).hexdigest()
                   for pid, s in result.items()},
               "timings": adapter.timings},
              open(out_dir / f"meta_{args.tag}.json", "w"), indent=1)
    print(f"[e4] {args.tag} md5={md5} waves={state['n']} {secs:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
