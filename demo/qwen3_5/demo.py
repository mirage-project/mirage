#!/usr/bin/env python3
"""Qwen3.5-35B-A3B-FP8 demo on the Mirage persistent kernel.

Mirrors the ``demo/qwen3`` pattern (``--use-mirage`` selects the megakernel)
against the registry builder in ``mirage.mpk.models.qwen3_5``. The whole 40-layer
graph -- 30 gated-DeltaNet layers, 10 full-attention layers, MoE on every layer --
is assembled by ``Qwen35Builder`` and run in ``MODE_OFFLINE``: one launch decodes
the prompt to completion inside the kernel.

    # megakernel (the point of this demo)
    python demo.py --use-mirage --max-new-tokens 64

    # plain-torch reference path, same prompt and chat template (optional,
    # needs a second copy of the checkpoint's worth of activations)
    python demo.py --max-new-tokens 64

Exits 0 on a successful decode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
DEFAULT_PROMPT = "Give me a short introduction to large language models."


def build_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--use-mirage", action="store_true",
                   help="Run the MPK megakernel (otherwise plain transformers).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-path", default=None,
                   help="Local snapshot dir; otherwise resolved from the hub cache.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=16,
                   help=">= the prompt length gives a single-chunk prefill; "
                        ">= --max-num-batched-requests is required. Values "
                        "above 16 were silently miscomputed by the MoE router "
                        "before M3-I5b (M2-I9); the default stays 16 so "
                        "measurements remain comparable.")
    p.add_argument("--max-num-batched-requests", type=int, default=1)
    p.add_argument("--page-size", type=int, default=256)
    p.add_argument("--output-dir", default=None,
                   help="Compiled-kernel output dir (reusable across runs).")
    p.add_argument("--reuse-kernel", action="store_true",
                   help="Load a previously compiled kernel from --output-dir.")
    p.add_argument("--ignore-eos", action="store_true",
                   help="Decode the full --max-new-tokens even past EOS.")
    p.add_argument("--save-tokens", default=None,
                   help="Dump {token_ids, text, latency} JSON here.")
    return p.parse_args()


def chat_prompt(tokenizer, prompt: str):
    """Same call the AC-3 reference generator uses
    (``accept/reference/generate_reference.py``): the model-shipped template with
    a generation prompt appended."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(enc, "input_ids"):
        return enc.input_ids[0].tolist()
    return enc[0].tolist()


# ---------------------------------------------------------------------------
def run_mirage(args, tokenizer, input_ids):
    import mirage as mi
    from mirage.mpk.models.qwen3_5.builder import Qwen35Builder

    mbt = args.max_num_batched_tokens
    mbr = args.max_num_batched_requests
    assert mbt >= mbr, "max_num_batched_tokens must be >= max_num_batched_requests"
    prompt_len = len(input_ids)
    max_seq_length = prompt_len + args.max_new_tokens
    pages_per_req = -(-max_seq_length // args.page_size)
    max_num_pages = max(mbr * pages_per_req + 4, 8)

    dev = "cuda"
    step = torch.zeros(mbr, dtype=torch.int32, device=dev)
    tokens = torch.zeros((mbr, max_seq_length), dtype=torch.long, device=dev)
    prompt_lengths = torch.full((mbr,), prompt_len, dtype=torch.int32, device=dev)
    for r in range(mbr):
        tokens[r, :prompt_len] = torch.tensor(input_ids, dtype=torch.long,
                                              device=dev)
    meta = {
        "step": step,
        "tokens": tokens,
        "input_tokens": torch.zeros((mbt, 1), dtype=torch.long, device=dev),
        "output_tokens": torch.zeros((mbt, 1), dtype=torch.long, device=dev),
        "num_new_tokens": torch.ones(mbr, dtype=torch.int32, device=dev),
        "prompt_lengths": prompt_lengths,
        "qo_indptr_buffer": torch.zeros(mbr + 1, dtype=torch.int32, device=dev),
        "paged_kv_indptr_buffer": torch.zeros(mbr + 1, dtype=torch.int32, device=dev),
        "paged_kv_indices_buffer": torch.zeros(max_num_pages, dtype=torch.int32, device=dev),
        "paged_kv_last_page_len_buffer": torch.zeros(mbr, dtype=torch.int32, device=dev),
        "paged_kv_indices_snapshot": torch.zeros(max_num_pages, dtype=torch.int32, device=dev),
    }

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    torch.set_default_dtype(torch.bfloat16)
    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0,
        num_workers=num_workers, num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=max_seq_length,
        max_num_batched_requests=mbr,
        max_num_batched_tokens=mbt,
        max_num_pages=max_num_pages,
        page_size=args.page_size,
        eos_token_id=-1,
        meta_tensors=meta,
        profiler_tensor=None,
        trace_name="",
        spec_decode_config=None,
        use_cutlass_kernel=True,
    )
    builder = Qwen35Builder(mpk)
    t0 = time.time()
    builder.build_from_model(model_name=args.model, model_path=args.model_path)
    print(f"[demo] graph assembled + weights loaded in {time.time() - t0:.1f}s")
    if not args.ignore_eos:
        mpk.eos_token_id = builder.eos_token_id

    t0 = time.time()
    if args.reuse_kernel and args.output_dir:
        mpk.load_mpk_kernel(output_dir=args.output_dir)
        print(f"[demo] reused compiled kernel ({time.time() - t0:.1f}s)")
    else:
        mpk.compile(output_dir=args.output_dir)
        print(f"[demo] megakernel compiled in {time.time() - t0:.1f}s")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    wall_ms = starter.elapsed_time(ender)

    end = int(step[0].item()) + 1
    out_ids = tokens[0, prompt_len:end].tolist()
    return out_ids, wall_ms, prompt_len


def run_torch(args, tokenizer, input_ids):
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path or args.model, dtype="auto", device_map="cuda:0")
    model.eval()
    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(input_ids=ids,
                             attention_mask=torch.ones_like(ids),
                             max_new_tokens=args.max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             top_k=None, num_beams=1, use_cache=True)
    wall_ms = (time.time() - t0) * 1000.0
    return gen[0].tolist()[len(input_ids):], wall_ms, len(input_ids)


def main() -> int:
    args = build_args()
    from transformers import AutoTokenizer
    from mirage.mpk.models.qwen3_5.weight_loader import resolve_snapshot

    snapshot = resolve_snapshot(args.model, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    input_ids = chat_prompt(tokenizer, args.prompt)
    print(f"[demo] prompt_len={len(input_ids)} "
          f"mode={'mirage' if args.use_mirage else 'torch'}")

    if args.use_mirage:
        out_ids, wall_ms, prompt_len = run_mirage(args, tokenizer, input_ids)
    else:
        out_ids, wall_ms, prompt_len = run_torch(args, tokenizer, input_ids)

    if not out_ids:
        print("[demo] FAILED: no tokens generated")
        return 1

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    per_tok = wall_ms / max(len(out_ids), 1)
    print("-" * 70)
    print(text)
    print("-" * 70)
    print(f"[demo] prompt {prompt_len} tok, generated {len(out_ids)} tok, "
          f"{wall_ms:.1f} ms total, {per_tok:.3f} ms/token")

    if args.save_tokens:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_tokens)) or ".",
                    exist_ok=True)
        with open(args.save_tokens, "w") as f:
            json.dump({"token_ids": out_ids, "text": text,
                       "latency_ms_per_token": per_tok,
                       "prompt_length": prompt_len,
                       "generate_length": len(out_ids),
                       "mode": "mpk" if args.use_mirage else "torch"}, f, indent=2)
        print(f"[demo] saved tokens to {args.save_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
