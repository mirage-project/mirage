"""Batched reference runner: load model ONCE, run multiple workloads.

Saves the dominant per-workload cost (FP8 dequant + dist init + model
construction; ~5-6 min on TP=4 layers 0-19) so a 28-workload sweep
runs in ~20 min instead of ~3 hours.

Workloads are read from a JSON file:

    [
        {"tag": "A1", "prompt_length": 100, "decode": 32, "mtp": 0,
         "max_num_batched_tokens": 128},
        {"tag": "A2", "prompt_length": 200, "decode": 32, "mtp": 2, ...},
        ...
    ]

Each workload's output goes to <out_dir>/<tag>_mtp<mtp>/.

Usage (from `/home/muhengl/mirage`):

    torchrun --nproc_per_node=4 \
      tests/dpskv3_reference/runner_batched.py \
      --model-path /raid/catalyst/models/DeepSeek-V3 \
      --layers 0-19 \
      --tp-size 4 --ep-size 2 \
      --workloads tests/dpskv3_reference/plan_a_v2.json \
      --out-dir outputs/dpskv3_ref_plan_a_v2_<ts>
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from tests.dpskv3_reference.config import Config  # noqa: E402
from tests.dpskv3_reference.modeling import DeepseekV3Model  # noqa: E402
from tests.dpskv3_reference.loader import load_into  # noqa: E402
from tests.dpskv3_reference.parallel import (  # noqa: E402
    ParallelConfig, init_distributed_if_needed,
)
from tests.dpskv3_reference.runner import (  # noqa: E402
    _build_mtp_prev_input_ids, _save_iter, _is_rank0, RunResult,
)


def _parse_layers(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def run_workload(
    model: DeepseekV3Model,
    pcfg: ParallelConfig,
    cfg: Config,
    *,
    tag: str,
    prompt_length: int,
    decode: int,
    mtp: int,                      # 0 = MTP off, >0 = enable
    max_num_batched_tokens: int,
    out_dir: Path,
    device: str,
) -> dict:
    """Run one workload through the already-loaded model. Returns
    a JSON-serialisable result summary."""
    enable_mtp = (mtp > 0)
    # MTP-enabled model can run mtp=0 workloads (just skip the MTP
    # forward path); a non-MTP model cannot run mtp>0 workloads.
    if enable_mtp and not model.enable_mtp:
        return {
            "tag": tag, "skipped": True,
            "reason": f"model.enable_mtp=False but mtp={mtp} (>0)",
        }

    if _is_rank0(pcfg):
        out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic prompt (matches MPK demo's --prompt-length mode).
    full_prompt_ids = (
        (torch.arange(prompt_length, dtype=torch.long, device=device) % 4096)
        + 1024
    )
    total_tokens = prompt_length + decode
    tokens = torch.full((total_tokens,), 0, dtype=torch.long, device=device)
    tokens[:prompt_length] = full_prompt_ids

    if _is_rank0(pcfg):
        with open(out_dir / "config.json", "w") as f:
            json.dump(
                {
                    "tag": tag, "prompt_length": prompt_length,
                    "decode": decode, "mtp": mtp,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "tp_size": pcfg.tp_size, "ep_size": pcfg.ep_size,
                    "layers": list(model.layer_indices),
                },
                f, indent=2,
            )

    iter_idx = 0
    step = 0
    elapsed_start = time.time()
    prefill_end_time: Optional[float] = None

    while step < total_tokens:
        if step < prompt_length:
            chunk_len = min(max_num_batched_tokens, prompt_length - step)
        else:
            chunk_len = 1
        if chunk_len == 0:
            break
        cur_input_ids = tokens[step : step + chunk_len].clone()
        cur_positions = torch.arange(step, step + chunk_len, device=device)

        prev_mtp_input_ids = None
        if enable_mtp:
            with torch.no_grad():
                tmp_out = model(
                    input_ids=cur_input_ids, positions=cur_positions,
                )
                main_argmax = tmp_out["argmax"]
            prev_mtp_input_ids = _build_mtp_prev_input_ids(
                cur_input_ids, main_argmax, chunk_start=step,
                prompt_length=prompt_length,
                full_prompt_ids=full_prompt_ids,
            )

        with torch.no_grad():
            out = model(
                input_ids=cur_input_ids, positions=cur_positions,
                prev_mtp_input_ids=prev_mtp_input_ids,
            )

        if pcfg.tp_size > 1:
            torch.cuda.synchronize()
        if step + chunk_len >= prompt_length and prefill_end_time is None:
            prefill_end_time = time.time()

        if _is_rank0(pcfg):
            iter_dir = out_dir / f"iter_{iter_idx:04d}"
            save_dict = {"input_ids": cur_input_ids, "positions": cur_positions}
            save_dict.update(out)
            _save_iter(iter_dir, save_dict)

        if step + chunk_len >= prompt_length:
            next_tok = out["argmax"][-1].item()
            if step + chunk_len < total_tokens:
                tokens[step + chunk_len] = next_tok
        step += chunk_len
        iter_idx += 1

    elapsed = time.time() - elapsed_start
    prefill_ms = None
    decode_tpot_ms = None
    if prefill_end_time is not None:
        prefill_ms = (prefill_end_time - elapsed_start) * 1000.0
        if decode > 0:
            decode_total = (time.time() - prefill_end_time) * 1000.0
            decode_tpot_ms = decode_total / decode

    if _is_rank0(pcfg):
        with open(out_dir / "tokens.json", "w") as f:
            json.dump(
                {
                    "prompt_token_ids": full_prompt_ids.tolist(),
                    "all_token_ids": tokens.tolist(),
                    "decoded_suffix_ids": tokens[prompt_length:].tolist(),
                    "prefill_ms": prefill_ms,
                    "decode_tpot_ms": decode_tpot_ms,
                    "elapsed_s": elapsed,
                },
                f, indent=2,
            )

    if pcfg.tp_size > 1:
        dist.barrier()

    return {
        "tag": tag,
        "prompt_length": prompt_length,
        "decode": decode,
        "mtp": mtp,
        "elapsed_s": elapsed,
        "prefill_ms": prefill_ms,
        "decode_tpot_ms": decode_tpot_ms,
        "decoded_suffix_ids": tokens[prompt_length:].tolist() if _is_rank0(pcfg) else None,
        "out_dir": str(out_dir),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--layers", default="0-19")
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--ep-size", type=int, default=2)
    p.add_argument("--workloads", required=True,
                   help="JSON file with workload list.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--enable-mtp", action="store_true",
                   help="Build model with MTP head. All workloads with mtp>0 "
                        "must use this; mtp=0 workloads can run with or "
                        "without (the head is just unused).")
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    pcfg = ParallelConfig(tp_size=args.tp_size, ep_size=args.ep_size, rank=rank)
    init_distributed_if_needed(pcfg, "cuda")
    if pcfg.tp_size > 1:
        torch.cuda.set_device(rank)
    device = f"cuda:{rank}" if pcfg.tp_size > 1 else "cuda"

    cfg = Config.from_hf(args.model_path)
    layers = _parse_layers(args.layers)

    if _is_rank0(pcfg):
        print(f"[batch] Building model: layers={layers} tp={args.tp_size} "
              f"ep={args.ep_size} enable_mtp={args.enable_mtp}",
              flush=True)
    t0 = time.time()
    model = DeepseekV3Model(
        cfg, layer_indices=layers, enable_mtp=args.enable_mtp,
        parallel_config=pcfg,
    ).to(device=device, dtype=torch.bfloat16).eval()
    if _is_rank0(pcfg):
        print(f"[batch] Loading weights ...", flush=True)
    load_into(model, args.model_path, target_dtype=torch.bfloat16, device=device)
    load_elapsed = time.time() - t0
    if _is_rank0(pcfg):
        print(f"[batch] Loaded in {load_elapsed:.1f}s. Free GPU mem: ", flush=True)
        for r in range(pcfg.tp_size):
            free, total = torch.cuda.mem_get_info(r)
            print(f"  rank {r}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    with open(args.workloads) as f:
        workloads = json.load(f)
    out_root = Path(args.out_dir)
    if _is_rank0(pcfg):
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / "load_info.json", "w") as f:
            json.dump({
                "load_elapsed_s": load_elapsed,
                "tp_size": args.tp_size,
                "ep_size": args.ep_size,
                "layers": layers,
                "enable_mtp": args.enable_mtp,
            }, f, indent=2)

    summary = []
    for w in workloads:
        tag = w["tag"]
        sub = out_root / f"{tag}_mtp{w.get('mtp', 0)}"
        if _is_rank0(pcfg):
            print(f"[batch] === {tag} mtp={w.get('mtp', 0)} "
                  f"prompt={w['prompt_length']} decode={w['decode']} ===",
                  flush=True)
        try:
            res = run_workload(
                model, pcfg, cfg,
                tag=tag,
                prompt_length=w["prompt_length"],
                decode=w["decode"],
                mtp=w.get("mtp", 0),
                max_num_batched_tokens=w.get("max_num_batched_tokens", 128),
                out_dir=sub,
                device=device,
            )
            summary.append(res)
            if _is_rank0(pcfg) and not res.get("skipped"):
                print(f"[batch] {tag} done: prefill={res['prefill_ms']:.1f}ms "
                      f"tpot={res['decode_tpot_ms']:.2f}ms elapsed={res['elapsed_s']:.1f}s",
                      flush=True)
        except Exception as e:
            if _is_rank0(pcfg):
                print(f"[batch] {tag} FAILED: {e}", flush=True)
            summary.append({"tag": tag, "error": str(e)})

    if _is_rank0(pcfg):
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[batch] All done. Summary: {out_root / 'summary.json'}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
