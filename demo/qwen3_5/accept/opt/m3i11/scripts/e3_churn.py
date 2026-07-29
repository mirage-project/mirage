#!/usr/bin/env python3
"""M3-I11 E3 -- does uninitialised device memory reach a LIVE row?

src/kernel/runtime.cc:1274 emits `cudaMalloc` for every megakernel intermediate
and never memsets it.  A fresh CUDA context hands back zeroed pages, so a
load-path process usually starts from all-zero intermediates -- which is why E2
saw 16/16 identical runs.  This probe deliberately DIRTIES the pages the
launcher is about to allocate: right before `load_mpk_kernel` (i.e. after the
weights are resident and immediately before the launcher's cudaMallocs) it
allocates N MiB in launcher-sized blocks, fills them with a chosen byte, frees
them and empties the torch caching allocator so the driver can hand the same
physical pages straight back.

Two runs that differ ONLY in the fill byte are the same program on the same
inputs.  If their KV fingerprints differ, uninitialised intermediate memory
demonstrably reaches live results.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ACC = Path(os.environ.get("ACC", str(Path.home() / "mpk-qwen35/mirage/demo/qwen3_5/accept")))
sys.path.insert(0, str(ACC))
sys.path.insert(0, str(ACC / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpk_engine_run import MPKOfflineAdapter  # noqa: E402
from ac3_types import PromptRequest           # noqa: E402
from e2_fingerprint import bitfp              # noqa: E402


def churn(mb: int, fill: int, block_mb: int = 4) -> None:
    """Dirty `mb` MiB of device memory with byte `fill`, then release it."""
    blocks = []
    n = max(1, mb // block_mb)
    for _ in range(n):
        b = torch.empty(block_mb << 20, dtype=torch.uint8, device="cuda")
        b.fill_(fill)
        blocks.append(b)
    del blocks
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[e3] churned {n * block_mb} MiB with byte 0x{fill:02x}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt", default="p03-python")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--msl", type=int, default=1280)
    ap.add_argument("--new-tokens", type=int, default=1024)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--fresh-compile", action="store_true",
                    help="compile in-process instead of loading (cold arm)")
    ap.add_argument("--churn-mb", type=int, default=0)
    ap.add_argument("--churn-fill", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--reference", default=str(ACC / "reference/reference_outputs.json"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(json.load(open(args.reference))["results"][args.prompt]["input_ids"])

    if args.churn_mb:
        import mirage as mi
        target = "compile" if args.fresh_compile else "load_mpk_kernel"
        orig = getattr(mi.PersistentKernel, target)

        def wrapper(self, *a, **kw):
            churn(args.churn_mb, args.churn_fill)
            return orig(self, *a, **kw)
        setattr(mi.PersistentKernel, target, wrapper)

    adapter = MPKOfflineAdapter(
        model_name="Qwen/Qwen3.5-35B-A3B-FP8", mbt=16, page_size=256,
        max_new_tokens=args.new_tokens,
        kernel_dir=Path(args.kernel_dir), reuse_kernel=not args.fresh_compile,
        pinned_max_seq_length=args.msl, audit_compaction=False,
        # Pinned to the PRE-POLICY uncapped runtime: this probe's committed
        # census is the comparison basis, and the shipped default is now the
        # admission policy (accept/admission_policy.py, M4-I4).
        per_request_token_cap="none")

    rec = {"tag": args.tag, "prompt": args.prompt, "bs": args.bs,
           "fresh_compile": args.fresh_compile, "churn_mb": args.churn_mb,
           "churn_fill": args.churn_fill, "kernel_dir": args.kernel_dir,
           "waves": []}
    fps = {}
    for w in range(args.waves):
        req = [PromptRequest(prompt_id=f"{args.prompt}#w{w}", input_ids=ids)]
        t0 = time.time()
        res = adapter.run(req, args.bs)
        b = adapter._builder
        toks = res[f"{args.prompt}#w{w}"].token_ids
        fps[f"w{w}_k"] = bitfp(b.k_cache.reshape(b.k_cache.shape[0], -1,
                               b.k_cache.shape[-2] * b.k_cache.shape[-1]), 2)
        fps[f"w{w}_v"] = bitfp(b.v_cache.reshape(b.v_cache.shape[0], -1,
                               b.v_cache.shape[-2] * b.v_cache.shape[-1]), 2)
        fps[f"w{w}_conv"] = bitfp(b.conv_state, 2)
        fps[f"w{w}_rec"] = bitfp(b.recurrent_state, 2)
        fps[f"w{w}_tok"] = np.asarray(toks, dtype=np.int64)
        rec["waves"].append({"wave": w, "secs": round(time.time() - t0, 1),
                             "n_tokens": len(toks)})
        print(f"[e3] {args.tag} wave {w} in {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed(out_dir / f"fp_{args.tag}.npz", **fps)
    json.dump(rec, open(out_dir / f"meta_{args.tag}.json", "w"), indent=1)
    print(f"[e3] wrote {out_dir}/fp_{args.tag}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
