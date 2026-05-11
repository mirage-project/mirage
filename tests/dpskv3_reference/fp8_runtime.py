"""Post-init hook that switches selected `nn.Linear`-shaped modules in a
DeepSeek-V3 reference model to FP8-simulated forward.

Why this lives outside `modeling.py`: keeping the model structure
identical to vLLM makes the BF16 reference easier to audit. The FP8
reference is a *runtime mode* — same modules, same parameter shapes,
just a different `forward()`.

Usage:

    from tests.dpskv3_reference.fp8_runtime import attach_fp8_faithful

    model = DeepseekV3Model(cfg, pcfg)
    load_weights_into(model, state_dict)  # state_dict has BF16 weights
    attach_fp8_faithful(model, fp8_state_dict)
    # ... now model.forward(...) runs the FP8 path for matching linears.

`fp8_state_dict` carries the *un-dequantized* FP8 weights from the
checkpoint plus their `_scale_inv` companions. The keys are matched
against the model's parameter names so the same loader plumbing can fill
both views.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fp8_sim import (
    fp8_simulated_linear,
    GROUP_SIZE,
    FP8_MAX,
)


def _matching_fp8_pair(
    name: str, module: nn.Module, fp8_state: dict,
    pcfg_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (weight_fp8, weight_scale_inv) from fp8_state, sharded to
    match the module's BF16 `.weight` shape. Handles three layouts:

      - Replicated: full FP8 weight, no sharding.
      - ColumnParallel: BF16 weight is [N_local, K] = full [N, K] sliced
                        along dim 0. Slice FP8 + scale identically.
      - RowParallel:    BF16 weight is [N, K_local] = full [N, K] sliced
                        along dim 1. Slice FP8 + scale identically.

    `pcfg_rank` (when provided) is the TP rank of the current process; it
    selects the matching slice deterministically. Falling back to the
    abs-mean heuristic in `_detect_rank` is unreliable because TP slices
    of an i.i.d. weight all have similar abs-means — it commonly resolves
    every rank to rank 0, which makes RowParallel AllReduce undercount
    the result by ~4× and is the source of the layer-0 attn-magnitude
    mismatch seen in 2026-05-11 distributed runs.

    Returns None if the keys don't exist or the shapes are ambiguous.
    """
    # The HF checkpoint keys carry a leading `model.` namespace
    # (`model.layers.0.self_attn.q_a_proj.weight`), but
    # `DeepseekV3Model.named_modules()` strips that prefix
    # (`layers.0.self_attn.q_a_proj`). Try both orderings so we match
    # regardless of which side the loader chose.
    candidates = (f"{name}.weight", f"model.{name}.weight")
    weight_key = next((k for k in candidates if k in fp8_state), None)
    if weight_key is None:
        return None
    scale_key = weight_key[: -len(".weight")] + ".weight_scale_inv"
    if scale_key not in fp8_state:
        return None
    w_fp8_full = fp8_state[weight_key]
    w_scale_full = fp8_state[scale_key]
    if w_fp8_full.dtype != torch.float8_e4m3fn:
        return None
    if not hasattr(module, "weight"):
        return None
    bf16_shape = tuple(module.weight.shape)
    full_shape = tuple(w_fp8_full.shape)

    # The per-block scale has shape (N // 128, K // 128) for an FP8 weight
    # of shape (N, K). Shard the scale on the same dim as the weight so
    # `_dequant_block_weight` reconstructs the right slice.
    from .fp8_sim import GROUP_SIZE

    if bf16_shape == full_shape:
        return w_fp8_full, w_scale_full
    if (
        bf16_shape[0] != full_shape[0]
        and bf16_shape[1] == full_shape[1]
        and full_shape[0] % bf16_shape[0] == 0
    ):
        # Column-parallel: dim-0 sharded.
        tp = full_shape[0] // bf16_shape[0]
        if pcfg_rank is not None and 0 <= pcfg_rank < tp:
            rank = pcfg_rank
        else:
            rank = _detect_rank(module, w_fp8_full, w_scale_full, axis=0, tp=tp)
            if rank is None:
                return None
        n_local = bf16_shape[0]
        w_fp8 = w_fp8_full[rank * n_local : (rank + 1) * n_local].contiguous()
        nb_local = n_local // GROUP_SIZE
        w_scale = w_scale_full[
            rank * nb_local : (rank + 1) * nb_local
        ].contiguous()
        return w_fp8, w_scale
    if (
        bf16_shape[1] != full_shape[1]
        and bf16_shape[0] == full_shape[0]
        and full_shape[1] % bf16_shape[1] == 0
    ):
        # Row-parallel: dim-1 sharded.
        tp = full_shape[1] // bf16_shape[1]
        if pcfg_rank is not None and 0 <= pcfg_rank < tp:
            rank = pcfg_rank
        else:
            rank = _detect_rank(module, w_fp8_full, w_scale_full, axis=1, tp=tp)
            if rank is None:
                return None
        k_local = bf16_shape[1]
        w_fp8 = w_fp8_full[:, rank * k_local : (rank + 1) * k_local].contiguous()
        kb_local = k_local // GROUP_SIZE
        w_scale = w_scale_full[
            :, rank * kb_local : (rank + 1) * kb_local
        ].contiguous()
        return w_fp8, w_scale
    return None


def _detect_rank(
    module: nn.Module,
    w_fp8_full: torch.Tensor,
    w_scale_full: torch.Tensor,
    axis: int,
    tp: int,
) -> int | None:
    """Find which TP rank's slice this module already holds, by matching
    its BF16 weight against the dequant of each candidate FP8 slice. Cheap
    1-shot probe: compare the abs-mean of a few rows / cols.

    Returns the rank in [0, tp) on match, or None if none match clearly.
    """
    from .fp8_sim import _dequant_block_weight

    bf16 = module.weight.float().abs().mean().item()
    best_rank = None
    best_diff = float("inf")
    for r in range(tp):
        if axis == 0:
            n_local = module.weight.shape[0]
            slice_fp8 = w_fp8_full[r * n_local : (r + 1) * n_local]
            from .fp8_sim import GROUP_SIZE
            nb_local = n_local // GROUP_SIZE
            slice_scale = w_scale_full[r * nb_local : (r + 1) * nb_local]
        else:
            k_local = module.weight.shape[1]
            slice_fp8 = w_fp8_full[:, r * k_local : (r + 1) * k_local]
            from .fp8_sim import GROUP_SIZE
            kb_local = k_local // GROUP_SIZE
            slice_scale = w_scale_full[:, r * kb_local : (r + 1) * kb_local]
        try:
            dq = _dequant_block_weight(slice_fp8, slice_scale)
        except Exception:
            continue
        candidate = dq.float().abs().mean().item()
        diff = abs(candidate - bf16) / max(bf16, 1e-9)
        if diff < best_diff:
            best_diff = diff
            best_rank = r
    if best_diff > 0.02:  # >2% mismatch on the abs-mean
        return None
    return best_rank


def _replace_forward(module: nn.Module, fp8_pair: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Attach (weight_fp8, weight_scale_inv) as buffers and override the
    module's forward to dispatch through `fp8_simulated_linear`. Keeps the
    original BF16 weight + bias so other code (e.g. all-gather views,
    saving) still works unchanged.

    Preserves the collective semantics of `RowParallelLinear` (post-matmul
    AllReduce) and any future input-gather step on `ColumnParallelLinear`
    by inspecting the module class and re-emitting the collective after
    the FP8 GEMM. Dropping the AllReduce caused TP=4 layer-0 attn-only to
    underflow by tp_size× (observed 2026-05-11: BF16-ref attn=15.1 vs
    FP8-sim attn=4.24 = 3.53×, with cosine similarity preserved)."""
    from .parallel import RowParallelLinear, all_reduce_tp

    w_fp8, w_scale = fp8_pair
    # Register as buffers so state_dict() picks them up if someone saves.
    module.register_buffer("weight_fp8", w_fp8.contiguous(), persistent=False)
    module.register_buffer(
        "weight_scale_inv", w_scale.float().contiguous(), persistent=False
    )
    bias = getattr(module, "bias", None)

    if isinstance(module, RowParallelLinear):
        pcfg = module.pcfg
        all_reduce_after = module.all_reduce_after

        def fp8_forward(x: torch.Tensor) -> torch.Tensor:
            # Mirror RowParallelLinear.forward: matmul → AllReduce → bias.
            out = fp8_simulated_linear(
                x, module.weight_fp8, module.weight_scale_inv, None
            )
            if all_reduce_after:
                out = all_reduce_tp(out, pcfg)
            if bias is not None:
                out = out + bias
            return out
    else:
        # ColumnParallelLinear, replicated Linear, RoutedExpert variants —
        # forward is just `F.linear(x, weight, bias)` with no collective,
        # so the simple replacement is faithful.
        def fp8_forward(x: torch.Tensor) -> torch.Tensor:
            return fp8_simulated_linear(
                x, module.weight_fp8, module.weight_scale_inv, bias
            )

    # Preserve original forward for any code that wants to compare.
    module._bf16_forward = module.forward  # type: ignore[attr-defined]
    module.forward = fp8_forward  # type: ignore[assignment]
    module._fp8_faithful = True  # type: ignore[attr-defined]


def attach_fp8_faithful(
    model: nn.Module,
    fp8_state: dict[str, torch.Tensor],
    device: torch.device | str | None = None,
    pcfg_rank: int | None = None,
) -> dict[str, int]:
    """Walk `model`, replacing forward of every Linear-like module whose
    `.weight` shape matches an `fp8_state[name + '.weight']` entry.

    `pcfg_rank` is this process's TP rank; pass it so RowParallel /
    ColumnParallel modules get their own FP8 slice rather than falling
    back to the abs-mean heuristic (which collapses every rank to slice
    0 on i.i.d. weights and divides RowParallel AllReduce results by
    ~tp).

    Returns a small report (`{linears_patched, linears_skipped}`) for
    logging.
    """
    if device is None:
        device = next(model.parameters()).device

    patched = 0
    skipped = 0
    for name, module in model.named_modules():
        # We patch any module that owns a 2-D `.weight` tensor. The
        # ColumnParallel / RowParallel / RoutedExpert variants all match.
        if not hasattr(module, "weight"):
            continue
        if not isinstance(getattr(module, "weight", None), torch.Tensor):
            continue
        if module.weight.ndim != 2:
            continue
        pair = _matching_fp8_pair(name, module, fp8_state, pcfg_rank=pcfg_rank)
        if pair is None:
            skipped += 1
            continue
        # Move buffers to the same device as the module's BF16 weight.
        w_fp8, w_scale = pair
        _replace_forward(
            module, (w_fp8.to(device), w_scale.to(device))
        )
        patched += 1
    return {"linears_patched": patched, "linears_skipped": skipped}


def detach_fp8_faithful(model: nn.Module) -> int:
    """Reverse `attach_fp8_faithful`: restore the BF16 forward. Returns
    the number of modules reverted."""
    reverted = 0
    for module in model.modules():
        if getattr(module, "_fp8_faithful", False):
            module.forward = module._bf16_forward  # type: ignore[attr-defined]
            del module._bf16_forward
            del module._fp8_faithful
            if hasattr(module, "weight_fp8"):
                del module.weight_fp8
            if hasattr(module, "weight_scale_inv"):
                del module.weight_scale_inv
            reverted += 1
    return reverted
