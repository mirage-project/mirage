"""Qwen3 driver — the lightweight, config-driven flow.

The whole driver collapses to:

  1. Build configs.
  2. ``mpk = PersistentKernel.build_from_config(cfg)``.
  3. ``text = mpk.run(prompt)``.

Single-GPU::

  python demo/qwen3/demo_new.py --model /raid/catalyst/models/Qwen3-8B/

TP=2::

  CUDA_VISIBLE_DEVICES=2,3 mpirun --np 2 python demo/qwen3/demo_new.py \\
      --tp-size 2 --model /raid/catalyst/models/Qwen3-8B/

Reduced-layer debug::

  python demo/qwen3/demo_new.py --model /raid/catalyst/models/Qwen3-8B/ \\
      --num-hidden-layers-override 2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import torch

from mirage.mpk import PersistentKernel
from mirage.mpk.configs import (
    HFConfig,
    KVCacheConfig,
    MPKConfig,
    ParallelConfig,
    RuntimeConfig,
)

# Importing the modeling module registers Qwen3ForCausalLM in the model registry.
# (Without this, build_from_config wouldn't know about the class.)
import mirage.mpk.models.qwen3.modeling  # noqa: F401


DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trace-name", default="")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor-parallel size; must equal mpirun world size when > 1.")
    parser.add_argument("--num-hidden-layers-override", type=int, default=None,
                        help="Debug knob: compile only this many decoder layers.")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help="Dump first N generated tokens to JSON.")
    return parser.parse_args()


def _bootstrap_distributed(tp_size: int) -> Tuple[int, int]:
    """Read rank/world_size from MPI; bind CUDA device + init NCCL for tp>1.

    Critically, ``torch.cuda.set_device(rank)`` must happen BEFORE
    ``dist.init_process_group("nccl")`` — otherwise NCCL initializes on
    each rank's default device (rank-0-bound on all ranks), and the
    later ``set_device`` collides with that binding.
    """
    if tp_size <= 1:
        torch.cuda.set_device(0)
        return 0, 1
    try:
        from mpi4py import MPI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "tp_size > 1 requires mpi4py to read the per-process rank. "
            "Install mpi4py and launch with mpirun."
        ) from e
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    if world_size != tp_size:
        raise RuntimeError(
            f"--tp-size ({tp_size}) must match mpirun world size ({world_size})."
        )
    # Bind CUDA device first, then NCCL init.
    torch.cuda.set_device(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size


def main() -> None:
    args = _parse_args()
    rank, world_size = _bootstrap_distributed(args.tp_size)
    torch.set_default_dtype(torch.bfloat16)

    global print  # silence non-root ranks
    if rank != 0:
        print = lambda *_, **__: None

    cfg = MPKConfig(
        hf=HFConfig.from_pretrained(
            args.model,
            num_hidden_layers_override=args.num_hidden_layers_override,
        ),
        parallel=ParallelConfig(
            world_size=world_size, rank=rank, tp_size=world_size,
        ),
        kv_cache=KVCacheConfig(
            max_num_pages=args.max_num_pages, page_size=args.page_size,
        ),
        runtime=RuntimeConfig(
            max_seq_length=args.max_seq_length,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_batched_requests=args.max_num_batched_requests,
            output_dir=args.output_dir,
            trace_name=args.trace_name,
            eos_token_id=-1 if args.ignore_eos else None,  # None => pull from HF
        ),
    )

    mpk = PersistentKernel.build_from_config(cfg)
    text = mpk.run(args.prompt)
    print(text)

    # Optional: save tokens for cross-run diff.
    if args.save_tokens and rank == 0:
        if args.save_tokens == "auto":
            fname = f"mpk_output_new_tp{world_size}.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, fname)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokens = mpk.meta_tensors["tokens"][0]
        step = int(mpk.meta_tensors["step"][0].item())
        prompt_lens = int(mpk.meta_tensors["prompt_lengths"][0].item())
        end = step + 1
        slice_end = min(end, prompt_lens + 100)
        out = {
            "token_ids": tokens[prompt_lens:slice_end].tolist(),
            "text": text,
            "prompt_length": prompt_lens,
            "generate_length": max(0, end - prompt_lens),
            "tp_size": world_size,
            "mode": "mpk_new_config",
        }
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved tokens to {save_path}")


if __name__ == "__main__":
    main()
