"""One-shot reference layer-dump.

Loads the dpskv3 reference model with `--layers` (default 0-19), runs ONE
prefill of `--prompt-length` synthetic tokens with `record_hidden=True`,
and saves each layer's `residual` tensor to
`<out-dir>/layer_NN_residual.pt` so it lines up with the MPK
`--dump-hidden-dir` output for direct row-by-row diff.

Usage:
    torchrun --nproc_per_node=4 scripts/dpskv3_ref_dump_one.py \\
        --model-path /raid/catalyst/models/DeepSeek-V3 \\
        --layers 0-19 --tp-size 4 --ep-size 2 \\
        --prompt-length 84 \\
        --out-dir outputs/dpskv3_ref_dump_<ts>
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist

from tests.dpskv3_reference.config import Config
from tests.dpskv3_reference.modeling import DeepseekV3Model
from tests.dpskv3_reference.loader import load_into
from tests.dpskv3_reference.parallel import ParallelConfig, init_distributed_if_needed
from tests.dpskv3_reference.runner import _is_rank0


def _parse_layers(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--layers", default="0-19")
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--ep-size", type=int, default=2)
    p.add_argument("--prompt-length", type=int, default=84)
    p.add_argument("--out-dir", required=True)
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
        print(f"[ref_dump] Building model: layers={layers} tp={args.tp_size} "
              f"ep={args.ep_size}", flush=True)
    t0 = time.time()
    model = DeepseekV3Model(
        cfg, layer_indices=layers, enable_mtp=False, parallel_config=pcfg,
    ).to(device=device, dtype=torch.bfloat16).eval()
    if _is_rank0(pcfg):
        print(f"[ref_dump] Loading weights ...", flush=True)
    load_into(model, args.model_path, target_dtype=torch.bfloat16, device=device)
    if _is_rank0(pcfg):
        print(f"[ref_dump] Loaded in {time.time() - t0:.1f}s", flush=True)

    out_dir = Path(args.out_dir)
    if _is_rank0(pcfg):
        out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic prompt — must match runner_batched + MPK demo.
    pl = args.prompt_length
    full_prompt_ids = (
        (torch.arange(pl, dtype=torch.long, device=device) % 4096) + 1024
    )
    positions = torch.arange(pl, device=device)

    with torch.no_grad():
        out = model(
            input_ids=full_prompt_ids, positions=positions,
            record_hidden=True,
        )

    if pcfg.tp_size > 1:
        torch.cuda.synchronize()

    if _is_rank0(pcfg):
        # Save per-layer hidden states. Reference returns (output, residual)
        # at each layer where output = MLP_li delta and residual is everything
        # before MLP_li (embed + attn_0 + mlp_0 + ... + attn_li). MPK's
        # self.x at layer-li-end is the FULL hidden = output + residual.
        # We dump both for use cases that need either.
        for li in layers:
            res_key = f"layer_{li}_residual"
            out_key = f"layer_{li}_output"
            if res_key in out:
                # MPK-comparable hidden = output + residual
                if out_key in out:
                    full = (out[out_key] + out[res_key]).detach().cpu()
                    torch.save(full, out_dir / f"layer_{li:02d}_residual.pt")
                else:
                    torch.save(out[res_key].detach().cpu(),
                               out_dir / f"layer_{li:02d}_residual.pt")
            # Also save the bare residual + output components for inspection.
            if res_key in out:
                torch.save(out[res_key].detach().cpu(),
                           out_dir / f"layer_{li:02d}_carry_only.pt")
            if out_key in out:
                torch.save(out[out_key].detach().cpu(),
                           out_dir / f"layer_{li:02d}_mlp_delta.pt")
        # Also save the embed and final_norm and the argmax tokens for
        # downstream consistency checks.
        if "embed" in out:
            torch.save(out["embed"].detach().cpu(), out_dir / "embed.pt")
        if "final_norm" in out:
            torch.save(out["final_norm"].detach().cpu(),
                       out_dir / "final_norm.pt")
        if "argmax" in out:
            torch.save(out["argmax"].detach().cpu(), out_dir / "argmax.pt")
        # Save the actually-loaded embed weight for diagnostic — used to check
        # whether the loader successfully populated embed_tokens.weight from
        # the checkpoint rather than leaving it at random init.
        torch.save(model.embed_tokens.weight.detach().cpu(),
                   out_dir / "embed_weight.pt")
        print(f"[ref_dump] Saved {len(layers)} layer residual dumps to "
              f"{out_dir}", flush=True)
        last_argmax = out["argmax"][-1].item()
        print(f"[ref_dump] Last position argmax: {last_argmax}", flush=True)
        # Sanity check: print embed_tokens.weight[1024] L2 vs the raw checkpoint.
        ew = model.embed_tokens.weight.detach().cpu()
        print(f"[ref_dump] embed_tokens.weight shape: {tuple(ew.shape)} "
              f"dtype: {ew.dtype}", flush=True)
        print(f"[ref_dump] embed_tokens.weight[1024] L2: "
              f"{ew[1024].float().norm().item():.4f}", flush=True)
        print(f"[ref_dump] out['embed'][0] L2: "
              f"{out['embed'][0].float().norm().item():.4f}", flush=True)

    if pcfg.tp_size > 1:
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
