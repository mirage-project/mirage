"""Runner: orchestrate prefill / decode (with and without MTP), dump.

Two entry points:

  - `run_reference(...)` — function called from a process where
    `torch.distributed` is either uninitialized (tp_size=1) or already
    initialized (caller used `torchrun` or `mp.spawn`).

  - `tests/dpskv3_reference/runner_distributed.py` — torchrun-launchable
    script that initialises distributed, calls `run_reference`, and
    exits.

Per-iteration dump layout (rank 0 only — other ranks compute but skip
disk I/O):

    <dump_dir>/
        config.json                 — model + run config snapshot
        iter_<i>/
            input_ids.pt            — int64 [T]
            positions.pt            — int64 [T]
            embed.pt                — bf16 [T, H]
            layer_<L>_output.pt     — bf16 [T, H]
            layer_<L>_residual.pt   — bf16 [T, H]
            final_norm.pt           — bf16 [T, H]
            logits.pt               — bf16 [T, V]
            argmax.pt               — int64 [T]
            mtp_output.pt           — bf16 [T, H]   (if enable_mtp)
            mtp_logits.pt           — bf16 [T, V]   (if enable_mtp)
            mtp_argmax.pt           — int64 [T]    (if enable_mtp)
        tokens.json                 — final accepted token sequence

For `tp_size > 1`, all ranks compute (the math requires it: AllReduce
across the world); only rank 0 writes to disk to avoid file races.
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from .config import Config
from .modeling import DeepseekV3Model
from .loader import load_into
from .parallel import ParallelConfig, init_distributed_if_needed


@dataclass
class RunResult:
    token_ids: list[int]
    dump_dir: Path
    elapsed_s: float
    prefill_ms: Optional[float] = None
    decode_tpot_ms: Optional[float] = None


def _save_iter(out_dir: Path, names_to_tensors: dict[str, torch.Tensor]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in names_to_tensors.items():
        torch.save(t.cpu(), out_dir / f"{name}.pt")


def _build_mtp_prev_input_ids(
    input_ids: torch.Tensor,
    main_argmax: torch.Tensor,
    chunk_start: int,
    prompt_length: int,
    full_prompt_ids: torch.Tensor,
) -> torch.Tensor:
    """`mtp_build_embed_input_layer` semantics (MPK builder.py:2778-2785)."""
    T = input_ids.shape[0]
    out = torch.zeros_like(input_ids)
    for i in range(T):
        gt_idx = chunk_start + i + 1
        if gt_idx < prompt_length:
            out[i] = full_prompt_ids[gt_idx]
        else:
            out[i] = main_argmax[i]
    return out


def _is_rank0(pcfg: ParallelConfig) -> bool:
    return pcfg.rank == 0


def run_reference(
    model_path: str,
    prompt: str = "Give me a short introduction to large language model.",
    prompt_length: int = 0,
    layers: Optional[list[int]] = None,
    enable_mtp: bool = False,
    spec_length: int = 1,
    max_new_tokens: int = 4,
    max_num_batched_tokens: int = 128,
    dump_dir: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    skip_weight_load: bool = False,
    tp_size: int = 1,
    ep_size: int = 1,
    rank: Optional[int] = None,
    fp8_faithful: bool = False,
    record_hidden: bool = False,
    force_accept_n: Optional[int] = None,
) -> RunResult:
    """Run the PyTorch reference, dump everything (rank 0 only),
    return final tokens.

    Prompt selection (mirrors MPK demo's behavior):
        - If `prompt_length > 0`: use a deterministic synthetic prompt
          `arange(prompt_length) % 4096 + 1024`. Same as MPK demo's
          `--prompt-length N` mode (demo.py:280-294). This gives bit-
          identical prompt token IDs on both sides.
        - Else: tokenize `prompt` via AutoTokenizer (no chat template).

    `rank` defaults to env `LOCAL_RANK` (torchrun) or 0.
    """
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
    pcfg = ParallelConfig(tp_size=tp_size, ep_size=ep_size, rank=rank)
    init_distributed_if_needed(pcfg, device)

    if pcfg.tp_size > 1:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

    if dump_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        dump_dir = f"outputs/dpskv3_reference_dump_{ts}"
    dump_dir = Path(dump_dir)
    if _is_rank0(pcfg):
        dump_dir.mkdir(parents=True, exist_ok=True)

    if not skip_weight_load:
        cfg = Config.from_hf(model_path)
    else:
        cfg = Config()
    if layers is None:
        layers = list(range(cfg.num_hidden_layers))

    # Build model.
    model = DeepseekV3Model(
        cfg, layer_indices=layers, enable_mtp=enable_mtp,
        parallel_config=pcfg,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    if not skip_weight_load:
        fp8_state = load_into(
            model, model_path, target_dtype=dtype, device=device,
            fp8_faithful=fp8_faithful,
        )
        if fp8_faithful and fp8_state:
            from .fp8_runtime import attach_fp8_faithful
            report = attach_fp8_faithful(
                model, fp8_state, device=device, pcfg_rank=pcfg.rank,
            )
            if _is_rank0(pcfg):
                print(
                    f"[fp8-faithful] linears patched={report['linears_patched']} "
                    f"skipped={report['linears_skipped']}"
                )

    # Tokenise. Two modes (mirrors MPK demo.py:280-294):
    #   prompt_length > 0  →  synthetic deterministic prompt
    #   prompt_length == 0 →  tokenize the text via AutoTokenizer
    if prompt_length > 0:
        # Synthetic, deterministic. No tokenizer needed; identical on
        # every rank without broadcast.
        full_prompt_ids = (
            (torch.arange(prompt_length, dtype=torch.long, device=device) % 4096)
            + 1024
        )
        tokenizer = None
    elif not skip_weight_load:
        if _is_rank0(pcfg):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            full_prompt_ids = tokenizer.encode(
                prompt, return_tensors="pt"
            ).squeeze(0).to(device)
            ids_len = torch.tensor([full_prompt_ids.numel()], device=device,
                                   dtype=torch.long)
        else:
            ids_len = torch.zeros(1, device=device, dtype=torch.long)
            tokenizer = None
        if pcfg.tp_size > 1:
            dist.broadcast(ids_len, src=0)
            if not _is_rank0(pcfg):
                full_prompt_ids = torch.zeros(
                    int(ids_len.item()), device=device, dtype=torch.long
                )
            dist.broadcast(full_prompt_ids, src=0)
    else:
        full_prompt_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
        tokenizer = None

    prompt_length = full_prompt_ids.shape[0]
    total_tokens = prompt_length + max_new_tokens
    tokens = torch.full(
        (total_tokens,), 0, dtype=torch.long, device=device
    )
    tokens[:prompt_length] = full_prompt_ids

    if _is_rank0(pcfg):
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
                    "tp_size": pcfg.tp_size,
                    "ep_size": pcfg.ep_size,
                    "vllm_aligned_to": "vllm @ /home/muhengl/vllm",
                    "config": {
                        "num_hidden_layers": cfg.num_hidden_layers,
                        "hidden_size": cfg.hidden_size,
                        "n_routed_experts": cfg.n_routed_experts,
                    },
                },
                f, indent=2,
            )

    iter_idx = 0
    step = 0
    elapsed_start = time.time()
    prefill_end_time: Optional[float] = None

    # Decode "width" tracking for force-accept spec decode. Mirrors MPK's
    # `meta_tensors["num_new_tokens"]`: initially 1 (single-token decode);
    # after a verify+accept_commit iter it becomes accepted_count.
    # Only meaningful when `force_accept_n is not None` and `enable_mtp`.
    # `K` is the spec_length (number of draft slots per iter).
    K = spec_length if enable_mtp else 0
    decode_num_new = 1  # width for the first decode iter

    while step < total_tokens:
        if step < prompt_length:
            chunk_len = min(max_num_batched_tokens, prompt_length - step)
        else:
            chunk_len = decode_num_new
        if chunk_len == 0:
            break
        # Cap so we don't run past max_new_tokens budget.
        chunk_len = min(chunk_len, total_tokens - step)
        cur_input_ids = tokens[step : step + chunk_len].clone()
        cur_positions = torch.arange(step, step + chunk_len, device=device)

        prev_mtp_input_ids = None
        if enable_mtp:
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
                record_hidden=record_hidden,
            )

        if pcfg.tp_size > 1:
            torch.cuda.synchronize()

        if step + chunk_len >= prompt_length and prefill_end_time is None:
            prefill_end_time = time.time()

        # Dump (rank 0 only).
        if _is_rank0(pcfg):
            iter_dir = dump_dir / f"iter_{iter_idx:04d}"
            save_dict = {
                "input_ids": cur_input_ids,
                "positions": cur_positions,
            }
            save_dict.update(out)
            _save_iter(iter_dir, save_dict)

        # Decide what gets committed to the sequence this iter.
        # Prefill: iters with step < prompt_length and chunk_len == prompt_length-step.
        # Decode: iters AFTER the prefill iter (step >= prompt_length).
        # The "prefill-completing iter" (step < prompt_length but
        # step+chunk_len == prompt_length) is treated as PREFILL — it just
        # commits the first generated token via greedy and advances step
        # by chunk_len (matching MPK's prepare_next_batch prefill path
        # which uses `num_new_tokens = prompt_length - step` not
        # `new_token_nums`).
        in_prefill_iter = step < prompt_length
        if in_prefill_iter or force_accept_n is None or not enable_mtp:
            # Greedy commit: one new token at position step+chunk_len
            # (= the predicted token after the last input position).
            # For prefill-completing iter, this places the first
            # generated token at position prompt_length.
            if step + chunk_len >= prompt_length:
                next_tok = out["argmax"][-1].item()
                if step + chunk_len < total_tokens:
                    tokens[step + chunk_len] = next_tok
            step += chunk_len
            iter_idx += 1
            continue

        # ---- Force-accept spec-decode commit path ----
        # Mirrors MPK's mtp_prepare_verify + accept_commit semantics:
        #   tokens[step+1]                    = main_argmax at input pos 0
        #   tokens[step+2..step+K+1]          = MTP drafts D_1..D_K
        #   accepted_count = N+1 (= force_accept_n + bonus)
        #   step += accepted_count
        #   next iter decode_num_new = accepted_count
        # Even rejected slots are *written* by prepare_verify in MPK; the
        # next iter then overwrites them at its own positions. We do the
        # same here so the sequence-buffer state matches MPK byte-for-byte.
        main_argmax = out["argmax"]  # [chunk_len]
        # MTP drafts: K predicted tokens for positions step+2..step+K+1.
        # When `prev_mtp_input_ids` is set (which we did above), `out`
        # includes "mtp_argmax" — the MTP head's predicted next token at
        # each input position. For spec_length=K we treat the LAST K of
        # those as drafts (since input length matches K+1 in steady state).
        mtp_argmax = out.get("mtp_argmax", None)
        if mtp_argmax is not None and mtp_argmax.numel() >= K:
            drafts = mtp_argmax[-K:] if K > 0 else mtp_argmax[:0]
        else:
            drafts = torch.zeros(K, dtype=torch.long, device=device)
        # accepted = min(force_accept_n, K)
        accepted = max(0, min(int(force_accept_n), K))
        # Commit tokens at positions step+1..step+K+1 (K+1 writes).
        # Position step+1: the natural greedy next token = argmax at
        #   last input position (which predicts the NEXT sequence pos).
        if step + 1 < total_tokens:
            tokens[step + 1] = main_argmax[-1]
        # Positions step+2..step+K+1: draft tokens.
        for i in range(K):
            wp = step + 2 + i
            if wp >= total_tokens:
                break
            tokens[wp] = drafts[i]
        # Advance by accepted+1 (matches MPK's accept_commit.num_new).
        step_advance = accepted + 1
        step += step_advance
        # Next iter's width.
        decode_num_new = step_advance
        iter_idx += 1
        # End-of-budget bail.
        if step >= total_tokens:
            break

    elapsed = time.time() - elapsed_start

    prefill_ms = None
    decode_tpot_ms = None
    if prefill_end_time is not None:
        prefill_ms = (prefill_end_time - elapsed_start) * 1000.0
        if max_new_tokens > 0:
            decode_total = (time.time() - prefill_end_time) * 1000.0
            decode_tpot_ms = decode_total / max_new_tokens

    if _is_rank0(pcfg):
        with open(dump_dir / "tokens.json", "w") as f:
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

    return RunResult(
        token_ids=tokens.tolist(),
        dump_dir=dump_dir,
        elapsed_s=elapsed,
        prefill_ms=prefill_ms,
        decode_tpot_ms=decode_tpot_ms,
    )
