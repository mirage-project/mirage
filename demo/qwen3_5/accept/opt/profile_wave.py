#!/usr/bin/env python3
"""Run ONE AC-3 wave with the megakernel profiler on and dump the raw buffer.

Reuses ``accept/mpk_engine_run.py``'s ``MPKOfflineAdapter`` verbatim -- same
wave protocol, same geometry, same token-ids-out -- and only injects a
profiler tensor.  No MPK source is modified: the injection is a scoped
monkeypatch of ``mirage.PersistentKernel`` around the adapter's ``_build``,
and immediately after the kernel is compiled/loaded the adapter's ``mpk``
object has ``profiler_tensor`` set back to ``None`` so
``PersistentKernel.__call__`` skips its own (unusably slow at this event
count) Perfetto/CSV exporters.  The device-side pointer was already handed to
``init_func`` at compile/load time, so the kernel still fills the buffer.

Prompt source is one of two mutually exclusive modes:

* ``--prompt-ids`` -- the AC-3 reference set (24-68 token prompts). Default
  mode; what every prior M3 issue captured with.
* ``--synthetic-prompt-len`` / ``--synthetic-seed`` -- M3-I10's matched-geometry
  arm A: ``--batch-size`` DISTINCT synthetic prompts of exactly
  ``--synthetic-prompt-len`` real-vocabulary token ids each, built byte-for-byte
  like ``bench_vllm.py``'s ``build_synthetic_prompts`` (same rng threading, same
  ``tokenizer.vocab_size`` source), so for the same seed both engines consume
  literally the same token ids.

Output per run (``--out-dir``):

    raw_bs<N>_rep<R>.npz   {idx: uint32 slot, val: uint64 entry}  non-zero only
    meta_bs<N>_rep<R>.json geometry, wall time, token-id sha, GPU state
    tokens_bs<N>_rep<R>.json  per-prompt generated ids (AC-3 non-regression)
    task_names.json        live event_name_list, so the parser never drifts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# MPK_ACCEPT_DIR lets the box copy of this script live OUTSIDE the mirage
# clone, so profiling never leaves the pinned tree dirty.
ACCEPT = Path(os.environ.get("MPK_ACCEPT_DIR", str(HERE.parent))).resolve()
sys.path.insert(0, str(ACCEPT))
sys.path.insert(0, str(ACCEPT / "harness"))

from mpk_engine_run import (MPKOfflineAdapter, PromptRequest,  # noqa: E402
                            load_reference_requests, log)


class ProfiledAdapter(MPKOfflineAdapter):
    def __init__(self, *a, prof_tensor=None, **kw):
        super().__init__(*a, **kw)
        self.prof_tensor = prof_tensor

    def _build(self, batch_size, max_seq_length, total_requests):
        import mirage as mi
        orig = mi.PersistentKernel
        prof = self.prof_tensor

        def wrapper(*args, **kwargs):
            kwargs["profiler_tensor"] = prof
            kwargs["trace_name"] = ""
            return orig(*args, **kwargs)

        mi.PersistentKernel = wrapper
        try:
            super()._build(batch_size, max_seq_length, total_requests)
        finally:
            mi.PersistentKernel = orig
        # Detach from the stock exporters -- we export with trace_lib instead.
        # The device pointer is already registered inside the launcher.
        self._mpk.profiler_tensor = None
        if prof is None:
            log("no profiler buffer (unprofiled control run)")
        else:
            log(f"profiler buffer attached: {prof.numel()} slots "
                f"({prof.numel() * 8 / 2**20:.0f} MiB), stock exporters bypassed")


def gpu_state():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,"
             "clocks.sm,clocks.mem", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return q.stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        return [f"nvidia-smi failed: {e!r}"]


def build_synthetic_requests(model_name: str, batch_size: int, input_len: int,
                             seed: int):
    """Byte-for-byte mirror of ``bench_vllm.py``'s ``build_synthetic_prompts``:
    ONE ``random.Random(seed)`` instance walked across all ``batch_size``
    prompts IN ORDER (not re-seeded per prompt), ``tokenizer.vocab_size`` as
    the draw range, ``rng.randrange(0, vocab_n)`` per token -- so for the same
    seed both engines consume literally the same token ids. Emits
    ``PromptRequest`` (prompt_id/input_ids) instead of vLLM's ``TokensPrompt``,
    since MPK is driven token-ids-in like every other adapter path here.
    """
    import random as _random
    from transformers import AutoTokenizer
    from mirage.mpk.models.qwen3_5.weight_loader import resolve_snapshot

    # Same snapshot resolution the adapter itself uses when model_path is
    # unset (see mpk_engine_run.py's CLI), so the tokenizer's vocab_size
    # matches the checkpoint the wave actually loads.
    tok = AutoTokenizer.from_pretrained(resolve_snapshot(model_name, None))
    rng = _random.Random(seed)
    vocab_n = tok.vocab_size
    requests = []
    for i in range(batch_size):
        ids = [rng.randrange(0, vocab_n) for _ in range(input_len)]
        requests.append(PromptRequest(prompt_id=f"synth{i}", input_ids=ids))
    return requests


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--prompt-ids", default=None,
                    help="AC-3 reference prompt ids (comma-separated). "
                         "Mutually exclusive with --synthetic-prompt-len.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--reuse-kernel", action="store_true")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--slots", type=int, default=48_000_000,
                    help="uint64 profiler slots; baked into the compiled "
                         "kernel as MPK_PROFILER_BUFFER_ENTRIES, so it must "
                         "match across --reuse-kernel runs.")
    ap.add_argument("--no-profiler", action="store_true",
                    help="Unprofiled control run (same code path, no buffer).")
    ap.add_argument("--max-seq-length", type=int, default=132)
    ap.add_argument("--mbt", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--reference",
                    default=str(ACCEPT / "reference" / "reference_outputs.json"))
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--synthetic-prompt-len", type=int, default=None,
                    help="M3-I10 matched-geometry arm A: synthesize "
                         "--batch-size DISTINCT prompts of exactly this many "
                         "real-vocabulary token ids each (bench_vllm.py's "
                         "build_synthetic_prompts, byte-for-byte). Mutually "
                         "exclusive with --prompt-ids.")
    ap.add_argument("--synthetic-seed", type=int, default=None,
                    help="Seed for --synthetic-prompt-len. Pass the SAME seed "
                         "the vLLM side used for the same (bs, rep) so both "
                         "engines consume literally the same token ids "
                         "(remeasure_spec.md sec 4: 20260725 + bs*1000 + rep).")
    args = ap.parse_args(argv)

    if (args.prompt_ids is None) == (args.synthetic_prompt_len is None):
        ap.error("exactly one of --prompt-ids / --synthetic-prompt-len is required")
    if args.synthetic_prompt_len is not None and args.synthetic_seed is None:
        ap.error("--synthetic-prompt-len requires --synthetic-seed")

    import torch
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"bs{args.batch_size}_rep{args.rep}"

    if args.synthetic_prompt_len is not None:
        requests = build_synthetic_requests(
            args.model, args.batch_size, args.synthetic_prompt_len,
            args.synthetic_seed)
        wanted = [r.prompt_id for r in requests]
    else:
        all_requests = load_reference_requests(Path(args.reference))
        wanted = [s.strip() for s in args.prompt_ids.split(",") if s.strip()]
        by_id = {r.prompt_id: r for r in all_requests}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise SystemExit(f"unknown prompt ids: {missing}")
        requests = [by_id[w] for w in wanted]
    if len(requests) > args.batch_size:
        raise SystemExit("one wave per process: len(prompt-ids) must be <= bs")

    prof = None
    if not args.no_profiler:
        prof = torch.zeros(args.slots, dtype=torch.uint64,
                           device="cuda").contiguous()

    gpu_before = gpu_state()
    adapter = ProfiledAdapter(
        model_name=args.model, mbt=args.mbt, page_size=args.page_size,
        max_new_tokens=args.max_new_tokens, kernel_dir=Path(args.kernel_dir),
        reuse_kernel=args.reuse_kernel,
        pinned_max_seq_length=args.max_seq_length, prof_tensor=prof)

    t0 = time.time()
    result = adapter.run(requests, args.batch_size)
    run_s = time.time() - t0

    from mirage.mpk.profiler_persistent import event_name_list
    with open(out / "task_names.json", "w") as f:
        json.dump({str(k): v for k, v in event_name_list.items()}, f, indent=1)

    tokens = {pid: seq.token_ids for pid, seq in result.items()}
    with open(out / f"tokens_{tag}.json", "w") as f:
        json.dump(tokens, f)
    sha = hashlib.sha256(
        json.dumps(tokens, sort_keys=True).encode()).hexdigest()

    n_events = 0
    if prof is not None:
        import numpy as np
        buf = prof.cpu().numpy()
        idx = np.flatnonzero(buf)
        idx = idx[idx > 0]
        n_events = int(len(idx))
        hdr = buf[:1].view(np.uint32)
        if args.save_raw:
            np.savez(out / f"raw_{tag}.npz", idx=idx.astype(np.uint32),
                     val=buf[idx], header=hdr.copy())
        log(f"profiler: {n_events} events, header nblocks={int(hdr[0])} "
            f"ngroups={int(hdr[1])}, fill={n_events/args.slots:.1%}")
        del buf

    meta = dict(
        tag=tag, batch_size=args.batch_size, rep=args.rep,
        prompt_ids=wanted,
        prompt_lens=[len(r.input_ids) for r in requests],
        synthetic_prompt_len=args.synthetic_prompt_len,
        synthetic_seed=args.synthetic_seed,
        max_seq_length=args.max_seq_length, mbt=args.mbt,
        page_size=args.page_size, max_new_tokens=args.max_new_tokens,
        profiler_slots=(0 if prof is None else args.slots),
        profiled=prof is not None, n_events=n_events,
        waves=adapter.timings, slot_isolation=adapter.dup_checks,
        run_seconds=run_s, tokens_sha256=sha,
        gpu_before=gpu_before, gpu_after=gpu_state(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with open(out / f"meta_{tag}.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"{tag}: wall={adapter.timings[0]['wall_ms']:.1f}ms "
        f"steps={adapter.timings[0]['max_decode_steps']} events={n_events}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
