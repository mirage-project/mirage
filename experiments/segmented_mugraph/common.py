"""Shared helpers for the segmented-muGraph benchmarks: deterministic inputs,
FP32-accumulated references, correctness metrics, CUDA-event timing, env capture.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

DEFAULT_TOKENS = 8
DEFAULT_HIDDEN = 4096
DEFAULT_INTERMEDIATE = 2048
DEFAULT_DTYPE = torch.bfloat16

#: Tolerance inherited from tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py
MAX_ABS_TOL_FULL_MLP = 1.0
MAX_ABS_TOL_REGION = 0.5


# --------------------------------------------------------------------------
# deterministic inputs
# --------------------------------------------------------------------------


def make_mlp_tensors(
    tokens: int = DEFAULT_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    intermediate: int = DEFAULT_INTERMEDIATE,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Identical inputs/weights in every process, given the same seed.

    Generation order is fixed so each implementation -- run in its own
    subprocess -- reconstructs bit-identical tensors without any IPC.
    Weights follow the PyTorch ``[out_features, in_features]`` convention.
    """
    torch.manual_seed(seed)
    g = dict(dtype=dtype, device=device)
    return {
        "x": torch.randn(tokens, hidden, **g),
        "w_gate": torch.randn(intermediate, hidden, **g) * 0.01,
        "w_up": torch.randn(intermediate, hidden, **g) * 0.01,
        "w_down": torch.randn(hidden, intermediate, **g) * 0.01,
        "residual": torch.randn(tokens, hidden, **g),
        # standalone Region B driver input (used when Region B is timed alone)
        "mid": torch.randn(tokens, intermediate, **g),
    }


# --------------------------------------------------------------------------
# FP32-accumulated PyTorch reference
# --------------------------------------------------------------------------


def torch_region_a(x, w_gate, w_up, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    gate = x.float() @ w_gate.float().t()
    up = x.float() @ w_up.float().t()
    return (torch.nn.functional.silu(gate) * up).to(dtype)


def torch_region_b(mid, w_down, residual=None, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    out = mid.float() @ w_down.float().t()
    if residual is not None:
        out = out + residual.float()
    return out.to(dtype)


#: The ``torch_*`` helpers above upcast to FP32 and are the *correctness
#: oracle*.  They are deliberately slower than a real bf16 model, so the timed
#: PyTorch baseline below uses ``F.linear`` in native bf16 -- what an actual
#: Qwen3 MLP executes -- keeping the speed comparison honest.


def torch_region_a_native(x, w_gate, w_up) -> torch.Tensor:
    return torch.nn.functional.silu(
        torch.nn.functional.linear(x, w_gate)
    ) * torch.nn.functional.linear(x, w_up)


def torch_region_b_native(mid, w_down, residual=None) -> torch.Tensor:
    out = torch.nn.functional.linear(mid, w_down)
    return out if residual is None else out + residual


def torch_full_mlp_native(x, w_gate, w_up, w_down, residual=None) -> torch.Tensor:
    return torch_region_b_native(
        torch_region_a_native(x, w_gate, w_up), w_down, residual
    )


def torch_full_mlp(x, w_gate, w_up, w_down, residual=None, dtype=DEFAULT_DTYPE):
    """FP32-accumulated reference for the whole MLP (no bf16 round-trip in between)."""
    gate = x.float() @ w_gate.float().t()
    up = x.float() @ w_up.float().t()
    mid = torch.nn.functional.silu(gate) * up
    out = mid @ w_down.float().t()
    if residual is not None:
        out = out + residual.float()
    return out.to(dtype)


# --------------------------------------------------------------------------
# correctness metrics
# --------------------------------------------------------------------------


def correctness_metrics(got: torch.Tensor, ref: torch.Tensor) -> Dict[str, Any]:
    g, r = got.detach().float().flatten(), ref.detach().float().flatten()
    diff = (g - r).abs()
    rel_l2 = (torch.linalg.vector_norm(g - r) / torch.linalg.vector_norm(r).clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(g, r, dim=0).item()
    return {
        "shape": list(got.shape),
        "dtype": str(got.dtype),
        "all_finite": bool(torch.isfinite(g).all().item()),
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "rel_l2_err": rel_l2,
        "cosine_sim": cos,
    }


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


@dataclass
class Timing:
    warmups: int
    iters: int
    mean_ms: float
    median_ms: float
    p5_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    tokens_per_s: float


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def time_fn(
    fn: Callable[[], Any],
    tokens: int,
    warmups: int = 20,
    iters: int = 100,
) -> Timing:
    """Time *fn* with CUDA events, synchronizing only at benchmark boundaries.

    Each measured iteration is bracketed by its own event pair; we synchronize
    once after the whole measured loop, never between iterations and never
    between Region A and Region B inside *fn*.
    """
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    samples = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    mean_ms = statistics.fmean(samples)
    return Timing(
        warmups=warmups,
        iters=iters,
        mean_ms=mean_ms,
        median_ms=_percentile(samples, 0.50),
        p5_ms=_percentile(samples, 0.05),
        p95_ms=_percentile(samples, 0.95),
        min_ms=samples[0],
        max_ms=samples[-1],
        tokens_per_s=(tokens / (mean_ms / 1e3)) if mean_ms > 0 else float("nan"),
    )


# --------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def env_info(device: int = 0) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mirage_git_rev": _git_rev(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        major, minor = torch.cuda.get_device_capability(device)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_compute_capability": f"{major}.{minor}",
                "gpu_target_cc": major * 10 + minor,
                "gpu_total_mem_bytes": props.total_memory,
                "gpu_count": torch.cuda.device_count(),
            }
        )
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL)
        info["nvcc_version"] = nvcc.decode().strip().splitlines()[-2].strip()
    except Exception:  # noqa: BLE001
        info["nvcc_version"] = "unavailable"
    return info


def peak_memory() -> Dict[str, int]:
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def fmt_table(rows: List[List[Any]], headers: List[str]) -> str:
    """Render a plain-text comparison table."""
    cells = [[str(h) for h in headers]] + [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(cells[0])), line]
    out += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in cells[1:]]
    return "\n".join(out)


def num(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and (x != x):
        return "nan"
    return f"{x:.{digits}f}"
