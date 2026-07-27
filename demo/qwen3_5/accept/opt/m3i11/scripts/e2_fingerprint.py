#!/usr/bin/env python3
"""M3-I11 E2 -- the sensitive nondeterminism detector.

Comparing emitted token ids is a terrible instrument for numerical
nondeterminism: a perturbation only becomes visible when it happens to cross an
argmax margin, which the M3-I9b census measured at ~2% of trajectories.  The
paged KV cache is a much better one.  It is a torch tensor the builder owns, it
holds K and V for EVERY position of EVERY attention layer, and MPK writes it
once per position and never rewrites it.  So the first (layer, position) at
which two runs' KV caches differ is the first moment the two runs' arithmetic
differed -- ULP or not, argmax flip or not.

Per run we record, for every attention layer and every cache slot, a 64-bit
bitwise fingerprint of that slot's K and V vectors, plus fingerprints of the
GDN conv/recurrent state and the emitted token ids.

Usage:
  e2_fingerprint.py --waves 4 --out <dir> [--prompt p03-python] [--tag rep1]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ACC = Path(os.environ.get("ACC", str(Path.home() / "mpk-qwen35/mirage/demo/qwen3_5/accept")))
sys.path.insert(0, str(ACC))
sys.path.insert(0, str(ACC / "harness"))

from mpk_engine_run import MPKOfflineAdapter  # noqa: E402
from ac3_types import PromptRequest           # noqa: E402


def bitfp(t: torch.Tensor, keep_dims: int) -> np.ndarray:
    """Bitwise 64-bit fingerprint of `t`, reduced over all dims after the first
    `keep_dims`.  Exact on the bit pattern: bf16/fp32 are reinterpreted as
    unsigned integers, so a 1-ULP change moves the fingerprint.  Two mixes
    (plain sum and index-weighted sum) are folded together so that a pure
    permutation or a compensating pair cannot collide."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="rep")
    ap.add_argument("--prompt", default="p03-python")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--msl", type=int, default=1280)
    ap.add_argument("--new-tokens", type=int, default=1024)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--reference", default=str(ACC / "reference/reference_outputs.json"))
    ap.add_argument("--per-request-token-cap", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ref = json.load(open(args.reference))["results"][args.prompt]
    ids = list(ref["input_ids"])

    adapter = MPKOfflineAdapter(
        model_name="Qwen/Qwen3.5-35B-A3B-FP8", mbt=16, page_size=256,
        max_new_tokens=args.new_tokens,
        kernel_dir=Path(args.kernel_dir), reuse_kernel=True,
        pinned_max_seq_length=args.msl,
        audit_compaction=False,
        per_request_token_cap=args.per_request_token_cap)

    rec = {"tag": args.tag, "prompt": args.prompt, "bs": args.bs,
           "msl": args.msl, "new_tokens": args.new_tokens,
           "kernel_dir": args.kernel_dir, "waves": []}
    fps = {}
    for w in range(args.waves):
        req = [PromptRequest(prompt_id=f"{args.prompt}#w{w}", input_ids=ids)]
        t0 = time.time()
        res = adapter.run(req, args.bs)
        b = adapter._builder
        toks = res[f"{args.prompt}#w{w}"].token_ids
        k_fp = bitfp(b.k_cache.reshape(b.k_cache.shape[0], -1,
                                       b.k_cache.shape[-2] * b.k_cache.shape[-1]), 2)
        v_fp = bitfp(b.v_cache.reshape(b.v_cache.shape[0], -1,
                                       b.v_cache.shape[-2] * b.v_cache.shape[-1]), 2)
        c_fp = bitfp(b.conv_state, 2)
        r_fp = bitfp(b.recurrent_state, 2)
        fps[f"w{w}_k"] = k_fp
        fps[f"w{w}_v"] = v_fp
        fps[f"w{w}_conv"] = c_fp
        fps[f"w{w}_rec"] = r_fp
        fps[f"w{w}_tok"] = np.asarray(toks, dtype=np.int64)
        rec["waves"].append({"wave": w, "secs": round(time.time() - t0, 1),
                             "n_tokens": len(toks),
                             "k_shape": list(b.k_cache.shape),
                             "prompt_len": len(ids)})
        print(f"[e2] wave {w} done in {time.time()-t0:.1f}s, "
              f"k_cache{list(b.k_cache.shape)}", flush=True)
    np.savez_compressed(out_dir / f"fp_{args.tag}.npz", **fps)
    json.dump(rec, open(out_dir / f"meta_{args.tag}.json", "w"), indent=1)
    print(f"[e2] wrote {out_dir}/fp_{args.tag}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
