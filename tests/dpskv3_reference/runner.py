"""Run a subset of DeepSeek V3 layers using the official inference
implementation, dump per-layer hidden states for MPK comparison.

Single-rank, BF16 reference. Configurable layer subset. Synthetic
prompt option (matches MPK demo's `--prompt-length N` mode) for
deterministic comparisons.

Usage:

    from tests.dpskv3_reference.runner import run_reference

    result = run_reference(
        model_path="/raid/catalyst/models/DeepSeek-V3",
        prompt_length=128,
        layer_indices=[0, 1, 2, 3],
        max_new_tokens=1,
        dump_dir="outputs/dpskv3_ref_official_<ts>",
    )
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .model_wrapper import DeepseekV3SubsetModel, deepseek_v3_args
from .loader import load_official_subset


@dataclass
class RunResult:
    token_ids: list[int]
    dump_dir: Path
    elapsed_s: float
    prefill_ms: Optional[float] = None
    decode_tpot_ms: Optional[float] = None


def _save_iter(out_dir: Path, named_tensors: dict[str, torch.Tensor]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in named_tensors.items():
        torch.save(t.cpu(), out_dir / f"{name}.pt")


def run_reference(
    model_path: str,
    prompt: str = "Give me a short introduction to large language model.",
    prompt_length: int = 0,
    layer_indices: Optional[list[int]] = None,
    max_new_tokens: int = 1,
    max_num_batched_tokens: int = 128,
    dump_dir: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    skip_weight_load: bool = False,
    verbose: bool = False,
) -> RunResult:
    """Run the official-inference-based reference.

    Args:
        model_path: HF DeepSeek V3 checkpoint dir.
        prompt: Text prompt (used if `prompt_length == 0`).
        prompt_length: If > 0, use synthetic prompt
            `arange(prompt_length) % 4096 + 1024` matching MPK
            demo's `--prompt-length N` mode.
        layer_indices: Layer indices to build. None = all 61 layers
            (huge, requires the full 671B-param checkpoint to fit
            in memory; not recommended for casual testing).
        max_new_tokens: Number of decode tokens after prefill.
        max_num_batched_tokens: Prefill chunk size (matches MPK's
            mbt).
        dump_dir: Where to write per-iter dumps.
        device: torch device.
        dtype: weight + activation dtype (bf16 recommended).
        skip_weight_load: If True, build the model with random
            weights (smoke test only).
        verbose: print loading stats.

    Returns:
        RunResult with token_ids, dump_dir, timings.
    """
    if layer_indices is None:
        # Default to all 61 layers. This requires 671B params loaded —
        # only do this if you really mean it.
        layer_indices = list(range(61))
    layer_indices = sorted(layer_indices)

    if dump_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        dump_dir = f"outputs/dpskv3_ref_official_{ts}"
    dump_dir_path = Path(dump_dir)
    dump_dir_path.mkdir(parents=True, exist_ok=True)

    # Build model.
    args = deepseek_v3_args(
        n_layers=max(layer_indices) + 1,  # for freqs_cis sizing — doesn't allocate Blocks
        n_dense_layers=3,
        max_seq_len=max(max_num_batched_tokens + max_new_tokens, 4096),
        dtype="bf16",
        max_batch_size=1,
    )
    if verbose:
        print(f"[ref] building subset model: layers={layer_indices}")
    model = DeepseekV3SubsetModel(
        args=args,
        layer_indices=layer_indices,
        device=device,
        dtype=dtype,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Load weights.
    if not skip_weight_load:
        if verbose:
            print(f"[ref] loading weights from {model_path}")
        t0 = time.time()
        stats = load_official_subset(
            model, model_path=model_path,
            layer_indices=layer_indices,
            device=device, dtype=dtype,
            verbose=verbose,
        )
        load_time = time.time() - t0
        if verbose:
            print(f"[ref] weights loaded in {load_time:.1f}s: {stats}")

    # Prompt construction.
    if prompt_length > 0:
        # Synthetic prompt (matches MPK demo.py:280-294).
        full_prompt_ids = (
            (torch.arange(prompt_length, dtype=torch.long, device=device) % 4096)
            + 1024
        )
        tokenizer = None
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        full_prompt_ids = tokenizer.encode(
            prompt, return_tensors="pt"
        ).squeeze(0).to(device)

    prompt_len = int(full_prompt_ids.shape[0])
    total_tokens = prompt_len + max_new_tokens
    tokens = torch.zeros(total_tokens, dtype=torch.long, device=device)
    tokens[:prompt_len] = full_prompt_ids

    # Save config snapshot.
    with open(dump_dir_path / "config.json", "w") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "prompt": prompt if prompt_length == 0 else f"synthetic_pl{prompt_length}",
                "prompt_length": prompt_len,
                "layer_indices": layer_indices,
                "max_new_tokens": max_new_tokens,
                "max_num_batched_tokens": max_num_batched_tokens,
                "dtype": str(dtype),
                "reference": "DeepSeek-V3/inference (official, BF16)",
            },
            f, indent=2,
        )

    elapsed_start = time.time()
    prefill_end: Optional[float] = None
    step = 0
    iter_idx = 0

    while step < total_tokens:
        if step < prompt_len:
            chunk_len = min(max_num_batched_tokens, prompt_len - step)
        else:
            chunk_len = 1
        chunk_len = min(chunk_len, total_tokens - step)
        if chunk_len == 0:
            break
        cur_input_ids = tokens[step : step + chunk_len].clone()
        cur_positions = torch.arange(step, step + chunk_len,
                                     device=device, dtype=torch.long)

        with torch.no_grad():
            out = model(
                input_ids=cur_input_ids,
                positions=cur_positions,
                record_hidden=True,
            )

        torch.cuda.synchronize()
        if step + chunk_len >= prompt_len and prefill_end is None:
            prefill_end = time.time()

        # Dump this iter.
        iter_dir = dump_dir_path / f"iter_{iter_idx:04d}"
        save = {
            "input_ids": cur_input_ids,
            "positions": cur_positions,
        }
        save.update(out)
        _save_iter(iter_dir, save)

        # Commit next token (greedy).
        next_tok = int(out["argmax"][-1].item())
        if step + chunk_len < total_tokens:
            tokens[step + chunk_len] = next_tok
        step += chunk_len
        iter_idx += 1

    elapsed = time.time() - elapsed_start
    prefill_ms = None
    decode_tpot_ms = None
    if prefill_end is not None:
        prefill_ms = (prefill_end - elapsed_start) * 1000.0
        if max_new_tokens > 0:
            decode_total = (time.time() - prefill_end) * 1000.0
            decode_tpot_ms = decode_total / max_new_tokens

    with open(dump_dir_path / "tokens.json", "w") as f:
        json.dump(
            {
                "prompt_token_ids": full_prompt_ids.tolist(),
                "all_token_ids": tokens.tolist(),
                "decoded_suffix_ids": tokens[prompt_len:].tolist(),
                "prefill_ms": prefill_ms,
                "decode_tpot_ms": decode_tpot_ms,
                "elapsed_s": elapsed,
            },
            f, indent=2,
        )

    return RunResult(
        token_ids=tokens.tolist(),
        dump_dir=dump_dir_path,
        elapsed_s=elapsed,
        prefill_ms=prefill_ms,
        decode_tpot_ms=decode_tpot_ms,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/raid/catalyst/models/DeepSeek-V3")
    p.add_argument("--prompt", default="Hello, world.")
    p.add_argument("--prompt-length", type=int, default=0)
    p.add_argument("--layers", default="0-3",
                   help="comma-separated indices or a-b range")
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument("--dump-dir", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if "-" in args.layers:
        a, b = args.layers.split("-")
        layer_indices = list(range(int(a), int(b) + 1))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    result = run_reference(
        model_path=args.model_path,
        prompt=args.prompt,
        prompt_length=args.prompt_length,
        layer_indices=layer_indices,
        max_new_tokens=args.max_new_tokens,
        dump_dir=args.dump_dir,
        verbose=args.verbose,
    )
    print(f"[ref] DONE.")
    print(f"  dump_dir = {result.dump_dir}")
    print(f"  elapsed = {result.elapsed_s:.1f}s")
    print(f"  prefill_ms = {result.prefill_ms}")
    print(f"  decode_tpot_ms = {result.decode_tpot_ms}")
    print(f"  decoded tokens = {result.token_ids[-args.max_new_tokens:]}")
