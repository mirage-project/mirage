"""Runner: orchestrate prefill / decode (with and without MTP), dump.

API entry point: `run_reference(...)`. Returns a `RunResult` dataclass
with the produced token IDs and the on-disk dump directory containing
per-iteration tensors.

Three modes:
    enable_mtp=False  → main model only (cases #1 and #3 from the brief)
    enable_mtp=True   → main + MTP head (cases #1, #2, #3 unified)

Per-iteration dump layout:

    <dump_dir>/
        config.json                 — model + run config snapshot
        iter_<i>/
            input_ids.pt            — int64 [T]
            positions.pt            — int64 [T]
            embed.pt                — bf16 [T, H]
            layer_<L>_output.pt     — bf16 [T, H]   (per layer in layer_indices)
            layer_<L>_residual.pt   — bf16 [T, H]
            final_norm.pt           — bf16 [T, H]
            logits.pt               — bf16 [T, V]
            argmax.pt               — int64 [T]
            mtp_output.pt           — bf16 [T, H]   (if enable_mtp)
            mtp_logits.pt           — bf16 [T, V]   (if enable_mtp)
            mtp_argmax.pt           — int64 [T]    (if enable_mtp)
        tokens.json                 — final accepted token sequence

The runner does NOT implement spec-decode verify/reject. It just
exposes the draft tokens MTP would produce; whether they get accepted
is MPK's job (and is verified separately at the token-id level).

Mirroring of MPK's iteration scheme:

    prefill_chunk_size = max_num_batched_tokens (mbt)
    while step < prompt_length:
        chunk_len = min(prefill_chunk_size, prompt_length - step)
        run_iteration(input_ids[step:step+chunk_len])
        step += chunk_len
    while step < prompt_length + max_new_tokens:
        run_iteration(input_ids[step:step+1])  # decode
        step += 1

For MTP-on, every iteration also runs the MTP head once, with
`prev_mtp_input_ids` set to the shifted ground-truth (vLLM's
`mtp_build_embed_input` semantics):

    For prefill positions [s, s+chunk_len):
        prev_mtp_input_ids[i] = input_ids[s+i+1]    if s+i+1 < prompt_length
                              = main_argmax[i]      otherwise
    For decode (chunk_len=1):
        prev_mtp_input_ids[0] = main_argmax[0]      (always)

This matches MPK's `mtp_build_embed_input_layer` at
`python/mirage/mpk/models/deepseek_v3/builder.py:2778-2785`.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import torch

from .config import Config
from .modeling import DeepseekV3Model
from .loader import load_into


@dataclass
class RunResult:
    token_ids: list[int]
    dump_dir: Path
    elapsed_s: float


def _save_iter(out_dir: Path, names_to_tensors: dict[str, torch.Tensor]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in names_to_tensors.items():
        torch.save(t.cpu(), out_dir / f"{name}.pt")


def _build_mtp_prev_input_ids(
    input_ids: torch.Tensor,            # [T]
    main_argmax: torch.Tensor,           # [T]
    chunk_start: int,
    prompt_length: int,
    full_prompt_ids: torch.Tensor,       # [prompt_length]
) -> torch.Tensor:
    """Per `mtp_build_embed_input_layer` in
    `python/mirage/mpk/models/deepseek_v3/builder.py:2778-2785`:

        For each position i in this iteration:
          if (chunk_start + i + 1) < prompt_length:
              # still inside prompt — use shifted ground-truth.
              out[i] = full_prompt_ids[chunk_start + i + 1]
          else:
              # past prompt — use main model's argmax at this position.
              out[i] = main_argmax[i]
    """
    T = input_ids.shape[0]
    out = torch.zeros_like(input_ids)
    for i in range(T):
        gt_idx = chunk_start + i + 1
        if gt_idx < prompt_length:
            out[i] = full_prompt_ids[gt_idx]
        else:
            out[i] = main_argmax[i]
    return out


def run_reference(
    model_path: str,
    prompt: str = "Give me a short introduction to large language model.",
    layers: Optional[list[int]] = None,
    enable_mtp: bool = False,
    spec_length: int = 1,         # only relevant when enable_mtp=True
    max_new_tokens: int = 4,
    max_num_batched_tokens: int = 128,
    dump_dir: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    skip_weight_load: bool = False,   # for unit tests with random init
) -> RunResult:
    """Run the PyTorch reference, dump everything, return final tokens.

    `skip_weight_load=True` is for unit tests — model parameters stay
    at PyTorch's default init. Useful for shape / no-NaN checks
    without needing a 671B checkpoint on disk.
    """
    if dump_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        dump_dir = f"outputs/dpskv3_reference_dump_{ts}"
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    if not skip_weight_load:
        cfg = Config.from_hf(model_path)
    else:
        cfg = Config()
    if layers is None:
        layers = list(range(cfg.num_hidden_layers))

    # Build model.
    model = DeepseekV3Model(cfg, layer_indices=layers, enable_mtp=enable_mtp)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    if not skip_weight_load:
        load_into(model, model_path, target_dtype=dtype, device=device)
    # If skip_weight_load is True, we keep the random init — caller is
    # responsible for not interpreting outputs as semantically meaningful.

    # Tokenise. We need a tokenizer — use HF's via the checkpoint.
    if not skip_weight_load:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        full_prompt_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    else:
        # Synthetic small prompt.
        full_prompt_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)

    prompt_length = full_prompt_ids.shape[0]
    total_tokens = prompt_length + max_new_tokens
    # Token buffer holds prompt + decoded suffix.
    tokens = torch.full(
        (total_tokens,), 0, dtype=torch.long, device=device
    )
    tokens[:prompt_length] = full_prompt_ids

    # Save run config snapshot.
    with open(dump_dir / "config.json", "w") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "prompt": prompt,
                "layers": list(layers),
                "enable_mtp": enable_mtp,
                "spec_length": spec_length,
                "max_new_tokens": max_new_tokens,
                "max_num_batched_tokens": max_num_batched_tokens,
                "dtype": str(dtype),
                "vllm_aligned_to": "vllm @ /home/muhengl/vllm",
                "config": {
                    "num_hidden_layers": cfg.num_hidden_layers,
                    "hidden_size": cfg.hidden_size,
                    "n_routed_experts": cfg.n_routed_experts,
                },
            },
            f, indent=2,
        )

    # Iteration loop.
    iter_idx = 0
    step = 0
    elapsed_start = time.time()

    while step < total_tokens:
        if step < prompt_length:
            chunk_len = min(max_num_batched_tokens, prompt_length - step)
        else:
            chunk_len = 1
        if chunk_len == 0:
            break
        cur_input_ids = tokens[step : step + chunk_len].clone()
        cur_positions = torch.arange(step, step + chunk_len, device=device)

        # Build MTP's shifted-input token IDs (if MTP on).
        prev_mtp_input_ids = None
        if enable_mtp:
            # We need main_argmax to fill the tail beyond the prompt;
            # do a forward without MTP first to get it, then redo with MTP.
            # NOTE: vLLM does the two together via shared hidden state;
            # in this reference we run twice for simplicity. The math
            # is identical; the second call's main path is purely
            # redundant compute (could be optimised by caching).
            with torch.no_grad():
                tmp_out = model(
                    input_ids=cur_input_ids,
                    positions=cur_positions,
                )
                main_argmax = tmp_out["argmax"]
            prev_mtp_input_ids = _build_mtp_prev_input_ids(
                cur_input_ids, main_argmax, chunk_start=step,
                prompt_length=prompt_length,
                full_prompt_ids=full_prompt_ids,
            )

        with torch.no_grad():
            out = model(
                input_ids=cur_input_ids,
                positions=cur_positions,
                prev_mtp_input_ids=prev_mtp_input_ids,
            )

        # Dump.
        iter_dir = dump_dir / f"iter_{iter_idx:04d}"
        save_dict = {
            "input_ids": cur_input_ids,
            "positions": cur_positions,
        }
        save_dict.update(out)
        _save_iter(iter_dir, save_dict)

        # Advance: for prefill, jump by chunk_len. For decode (step >=
        # prompt_length), the next token is the argmax of the LAST
        # position in this iter.
        if step + chunk_len >= prompt_length:
            # Last position's argmax is the predicted next token.
            next_tok = out["argmax"][-1].item()
            if step + chunk_len < total_tokens:
                tokens[step + chunk_len] = next_tok
        step += chunk_len
        iter_idx += 1

    elapsed = time.time() - elapsed_start

    # Save final token sequence.
    with open(dump_dir / "tokens.json", "w") as f:
        json.dump(
            {
                "prompt_token_ids": full_prompt_ids.tolist(),
                "all_token_ids": tokens.tolist(),
                "decoded_suffix_ids": tokens[prompt_length:].tolist(),
                "elapsed_s": elapsed,
            },
            f, indent=2,
        )

    return RunResult(
        token_ids=tokens.tolist(),
        dump_dir=dump_dir,
        elapsed_s=elapsed,
    )
