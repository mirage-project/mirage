#!/usr/bin/env python3
"""M3-I11 E7 -- is there a LIVE run-to-run-varying write inside the decode path?

M3-I5c found that both SM100 routers compacted `mpk_active_expert_ids` with
`atomicAdd`, so the compacted list's ORDER is decided by CTA arrival order, and
the read-then-scatter had no barrier (phantom experts possible). That is a
concrete, source-level run-to-run nondeterminism source in a task every layer
runs every step -- and unlike token ids it can be observed directly.

`layer_<i>_moe_mask` IS `mpk_active_expert_ids`: entries [0, count) are the
compacted active expert ids and entry [NUM_EXPERTS] is the count. Building the
graph with `expose_intermediates` makes it a torch tensor, so after a wave we
can read all 40 layers' masks and compare them across processes:

  order differs, set+count identical -> the atomicAdd permutation is live but
                                        benign (matches I5c's bit-exactness
                                        argument and the clean token evidence)
  set or count differs               -> the phantom-expert race is live and is
                                        a real correctness hazard
  everything identical               -> no live nondeterminism at this sample
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--msl", type=int, default=1280)
    ap.add_argument("--new-tokens", type=int, default=1024)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--prompt", default="p03-python")
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--reference", default=str(ACC / "reference/reference_outputs.json"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(json.load(open(args.reference))["results"][args.prompt]["input_ids"])

    # expose every intermediate as a torch tensor so the router mask is readable
    from mirage.mpk.models.qwen3_5.builder import Qwen35Builder
    orig_build = Qwen35Builder.build_from_model

    def wrapper(self, *a, **kw):
        self.expose_intermediates = True
        return orig_build(self, *a, **kw)
    Qwen35Builder.build_from_model = wrapper

    adapter = MPKOfflineAdapter(
        model_name="Qwen/Qwen3.5-35B-A3B-FP8", mbt=16, page_size=256,
        max_new_tokens=args.new_tokens,
        kernel_dir=Path(args.kernel_dir),
        reuse_kernel=Path(args.kernel_dir).joinpath("task_graph_rank0.json").exists(),
        pinned_max_seq_length=args.msl, audit_compaction=False)

    rec = {"tag": args.tag, "bs": args.bs, "waves": []}
    saved = {}
    for w in range(args.waves):
        req = [PromptRequest(prompt_id=f"{args.prompt}#w{w}", input_ids=ids)]
        t0 = time.time()
        res = adapter.run(req, args.bs)
        b = adapter._builder
        masks = {}
        for name, buf in b.buffers.items():
            if name.endswith("_moe_mask"):
                masks[name] = buf.detach().cpu().numpy().copy()
        for name, arr in masks.items():
            saved[f"w{w}::{name}"] = arr
        saved[f"w{w}::tokens"] = np.asarray(
            res[f"{args.prompt}#w{w}"].token_ids, dtype=np.int64)
        rec["waves"].append({"wave": w, "secs": round(time.time() - t0, 1),
                             "n_masks": len(masks)})
        print(f"[e7] wave {w}: {len(masks)} router masks, {time.time()-t0:.1f}s",
              flush=True)
    np.savez_compressed(out_dir / f"mask_{args.tag}.npz", **saved)
    json.dump(rec, open(out_dir / f"meta_{args.tag}.json", "w"), indent=1)
    print(f"[e7] wrote {out_dir}/mask_{args.tag}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
