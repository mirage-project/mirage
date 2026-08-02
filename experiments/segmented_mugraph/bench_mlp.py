"""Stage 1: three-way Qwen3-like dense MLP microbenchmark.

Compares, on identical inputs and weights:

  1. ``torch``    -- PyTorch reference (FP32-accumulated matmuls).
  2. ``mpk``      -- the existing, unmodified MPK task-graph implementation
                     (``linear_layer`` / ``silu_mul_layer`` /
                     ``linear_with_residual_layer``).
  3. ``mugraph``  -- the new segmented path: ordinary ``KNGraph`` execution via
                     :class:`SegmentedMuGraphRunner`, no ``PersistentKernel``.

Each (implementation, scope) pair runs in its **own subprocess** so CUDA
allocator, cuBLAS handle and compiler state cannot contaminate another
measurement.

Scopes: ``region_a`` (gate+up+SiLU+mul), ``region_b`` (down+residual),
``full`` (the whole MLP; Region A and Region B are *not* synchronized between).

Example
-------
    PYTHONPATH=. python -m experiments.segmented_mugraph.bench_mlp \
        --tokens 8 --hidden 4096 --intermediate 2048 \
        --warmups 20 --iters 100 \
        --out experiments/outputs/stage1_mlp.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from . import common
from .common import (
    correctness_metrics,
    env_info,
    fmt_table,
    make_mlp_tensors,
    num,
    peak_memory,
    time_fn,
    torch_full_mlp,
    torch_region_a,
    torch_region_b,
    write_json,
)

IMPLS = ("torch", "mpk", "mugraph")
SCOPES = ("region_a", "region_b", "full")


# ==========================================================================
# workers
# ==========================================================================


def _run_torch(args, t) -> Dict[str, Any]:
    """PyTorch baseline: native bf16 ``F.linear``, i.e. what a real Qwen3 runs.

    The FP32-accumulated helpers are the correctness oracle and are applied to
    every implementation alike in :func:`run_worker`; timing them here would
    understate PyTorch by measuring an upcast the real model never performs.
    """
    if args.scope == "region_a":
        fn = lambda: common.torch_region_a_native(t["x"], t["w_gate"], t["w_up"])
    elif args.scope == "region_b":
        fn = lambda: common.torch_region_b_native(t["mid"], t["w_down"], t["residual"])
    else:
        fn = lambda: common.torch_full_mlp_native(
            t["x"], t["w_gate"], t["w_up"], t["w_down"], t["residual"]
        )
    out = fn()
    torch.cuda.synchronize()
    timing = time_fn(fn, args.tokens, args.warmups, args.iters)
    return {
        "output": out,
        "timing": asdict(timing),
        "cold": {"search_time_s": 0.0, "cuda_compile_time_s": 0.0, "total_s": 0.0},
        "compiler": {"mode": "eager-pytorch-bf16", "regions": []},
    }


def _run_mugraph(args, t) -> Dict[str, Any]:
    """Segmented muGraph path -- ordinary KNGraph compilation, no MPK."""
    from .runner import SegmentedMuGraphRunner, no_task_graph_guard

    dtype = t["x"].dtype
    with no_task_graph_guard():
        runner = SegmentedMuGraphRunner(
            device=t["x"].device,
            try_superoptimize=not args.no_superoptimize,
            verbose=True,
        )
        if args.scope == "region_a":
            fn = lambda: runner.region_a(t["x"], t["w_gate"], t["w_up"])
        elif args.scope == "region_b":
            fn = lambda: runner.region_b(t["mid"], t["w_down"], t["residual"])
        else:
            fn = lambda: runner.mlp(
                t["x"], t["w_gate"], t["w_up"], t["w_down"], t["residual"]
            )

        cold0 = time.perf_counter()
        out = fn()  # triggers lazy compilation
        torch.cuda.synchronize()
        cold_total = time.perf_counter() - cold0
        out = out.clone()  # detach from the reused region buffer

        report = runner.report()
        timing = time_fn(fn, args.tokens, args.warmups, args.iters)

    return {
        "output": out,
        "timing": asdict(timing),
        "cold": {
            "search_time_s": sum(r["search_time_s"] for r in report),
            "cuda_compile_time_s": sum(r["compile_time_s"] for r in report),
            "total_s": cold_total,
        },
        "compiler": {
            "mode": "+".join(sorted({r["mode"] for r in report})) or "none",
            "regions": report,
        },
    }


def _grid_for_linear(output_dim: int) -> int:
    """Grid dim for MPK linear layers -- matches demo/qwen3/demo.py."""
    if output_dim % 96 == 0:
        return output_dim // 96
    if output_dim % 64 == 0:
        return output_dim // 64
    raise AssertionError(f"Unsupported linear output_dim={output_dim}")


def _run_mpk(args, t) -> Dict[str, Any]:
    """Existing MPK task-graph implementation -- used unmodified as baseline."""
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel

    dtype, device = t["x"].dtype, t["x"].device
    B, H, I = args.tokens, args.hidden, args.intermediate
    fused = 2 * I

    mlp_mid = torch.zeros(B, fused, dtype=dtype, device=device)
    silu_out = torch.zeros(B, I, dtype=dtype, device=device)
    mlp_out = torch.zeros(B, H, dtype=dtype, device=device)

    cold0 = time.perf_counter()
    num_workers, num_scheds = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_scheds,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=B,
        max_num_batched_requests=B,
    )
    pk = PersistentKernel(**params)
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    n_gatedup = _grid_for_linear(fused)

    if args.scope in ("region_a", "full"):
        in_dt = pk.attach_input(t["x"], name="input")
        wg_dt = pk.attach_input(t["w_gate"], name="w_gate")
        wu_dt = pk.attach_input(t["w_up"], name="w_up")
        mid_dt = pk.attach_input(mlp_mid, name="mlp_mid")
        silu_dt = pk.attach_input(silu_out, name="silu_mul_out")
        w_gatedup = pk.shuffle_tensors(
            inputs=[wg_dt, wu_dt], shuffled_dim=0,
            num_groups=n_gatedup // 2, name="w_gatedup",
        )
        pk.linear_layer(
            input=in_dt, weight=w_gatedup, output=mid_dt,
            grid_dim=(n_gatedup, 1, 1), block_dim=block_dim,
        )
        pk.silu_mul_layer(
            input=mid_dt, output=silu_dt,
            grid_dim=(n_gatedup // 2, 1, 1), block_dim=block_dim,
        )
        b_input_dt, b_input = silu_dt, silu_out
    else:
        b_input = t["mid"]
        b_input_dt = pk.attach_input(b_input, name="silu_mul_out")

    if args.scope in ("region_b", "full"):
        wd_dt = pk.attach_input(t["w_down"], name="w_down")
        res_dt = pk.attach_input(t["residual"], name="residual")
        out_dt = pk.attach_input(mlp_out, name="mlp_out")
        pk.linear_with_residual_layer(
            input=b_input_dt, weight=wd_dt, residual=res_dt, output=out_dt,
            grid_dim=(H // 64, 1, 1), block_dim=block_dim,
        )

    t_compile = time.perf_counter()
    pk.compile()  # temp dir; no repo artifacts
    compile_s = time.perf_counter() - t_compile

    pk()
    torch.cuda.synchronize()
    cold_total = time.perf_counter() - cold0

    result = {"region_a": silu_out, "region_b": mlp_out, "full": mlp_out}[args.scope]
    out = result.clone()
    timing = time_fn(lambda: pk(), args.tokens, args.warmups, args.iters)
    payload = {
        "output": out,
        "timing": asdict(timing),
        "cold": {
            "search_time_s": 0.0,
            "cuda_compile_time_s": compile_s,
            "total_s": cold_total,
        },
        "compiler": {"mode": "mpk-task-graph", "regions": []},
    }
    pk.finalize()
    return payload


# ==========================================================================
# worker entry point
# ==========================================================================


def run_worker(args) -> Dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    t = make_mlp_tensors(
        args.tokens, args.hidden, args.intermediate,
        device=f"cuda:{args.device}", seed=args.seed,
    )
    dtype = t["x"].dtype

    runner = {"torch": _run_torch, "mpk": _run_mpk, "mugraph": _run_mugraph}[args.impl]
    res = runner(args, t)
    got = res.pop("output")

    if args.scope == "region_a":
        ref = torch_region_a(t["x"], t["w_gate"], t["w_up"], dtype)
        tol = common.MAX_ABS_TOL_REGION
    elif args.scope == "region_b":
        ref = torch_region_b(t["mid"], t["w_down"], t["residual"], dtype)
        tol = common.MAX_ABS_TOL_REGION
    else:
        ref = torch_full_mlp(
            t["x"], t["w_gate"], t["w_up"], t["w_down"], t["residual"], dtype
        )
        tol = common.MAX_ABS_TOL_FULL_MLP

    metrics = correctness_metrics(got, ref)
    metrics["max_abs_tol"] = tol
    metrics["passed"] = bool(metrics["all_finite"] and metrics["max_abs_err"] < tol)

    return {
        "impl": args.impl,
        "scope": args.scope,
        "dims": {
            "tokens": args.tokens,
            "hidden_size": args.hidden,
            "intermediate_size": args.intermediate,
            "dtype": str(dtype),
        },
        "correctness": metrics,
        "memory": peak_memory(),
        **res,
    }


# ==========================================================================
# parent orchestration
# ==========================================================================


def _spawn(args, impl: str, scope: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "res.json")
        cmd = [
            sys.executable, "-m", "experiments.segmented_mugraph.bench_mlp",
            "--worker", "--impl", impl, "--scope", scope,
            "--tokens", str(args.tokens), "--hidden", str(args.hidden),
            "--intermediate", str(args.intermediate),
            "--warmups", str(args.warmups), "--iters", str(args.iters),
            "--seed", str(args.seed), "--device", str(args.device),
            "--worker-out", out_path,
        ]
        if args.no_superoptimize:
            cmd.append("--no-superoptimize")
        env = dict(os.environ)
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"\n=== {impl} / {scope} ===", flush=True)
        proc = subprocess.run(cmd, env=env, cwd=repo,
                              stdout=None if args.verbose else subprocess.DEVNULL,
                              stderr=None if args.verbose else subprocess.DEVNULL)
        if proc.returncode != 0 or not os.path.exists(out_path):
            return {
                "impl": impl, "scope": scope, "status": "failed",
                "returncode": proc.returncode,
            }
        with open(out_path) as f:
            res = json.load(f)
        res["status"] = "ok"
        return res


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=common.DEFAULT_TOKENS)
    p.add_argument("--hidden", type=int, default=common.DEFAULT_HIDDEN)
    p.add_argument("--intermediate", type=int, default=common.DEFAULT_INTERMEDIATE)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--impls", default=",".join(IMPLS))
    p.add_argument("--scopes", default=",".join(SCOPES))
    p.add_argument("--no-superoptimize", action="store_true",
                   help="Skip the superoptimizer and compile the high-level KNGraph directly")
    p.add_argument("--out", default="experiments/outputs/stage1_mlp.json")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    # worker-only
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--impl", choices=IMPLS, help=argparse.SUPPRESS)
    p.add_argument("--scope", choices=SCOPES, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; the MLP benchmark needs a GPU.")
        return 0

    if args.worker:
        res = run_worker(args)
        write_json(args.worker_out, res)
        c = res["correctness"]
        print(f"[{args.impl}/{args.scope}] max_abs={c['max_abs_err']:.5f} "
              f"cos={c['cosine_sim']:.6f} mean_ms={res['timing']['mean_ms']:.4f}",
              flush=True)
        return 0

    impls = [i for i in args.impls.split(",") if i]
    scopes = [s for s in args.scopes.split(",") if s]
    results = [_spawn(args, impl, scope) for scope in scopes for impl in impls]

    payload = {
        "benchmark": "stage1_qwen3_mlp",
        "environment": env_info(args.device),
        "config": {
            "tokens": args.tokens, "hidden_size": args.hidden,
            "intermediate_size": args.intermediate, "dtype": "torch.bfloat16",
            "warmups": args.warmups, "iters": args.iters, "seed": args.seed,
            "superoptimize": not args.no_superoptimize,
        },
        "results": results,
        "note": (
            "Apples-to-apples kernel comparison: all three implementations run the "
            "same MLP on the same inputs/weights in isolated processes."
        ),
    }
    write_json(args.out, payload)
    print("\n" + _render(payload))
    print(f"\nJSON written to {args.out}")
    return 0


def _render(payload: Dict[str, Any]) -> str:
    rows = []
    for r in payload["results"]:
        if r.get("status") != "ok":
            rows.append([r["impl"], r["scope"], "FAILED", "-", "-", "-", "-", "-", "-", "-"])
            continue
        t, c, cold = r["timing"], r["correctness"], r["cold"]
        rows.append([
            r["impl"], r["scope"], r["compiler"]["mode"],
            num(t["mean_ms"], 4), num(t["median_ms"], 4),
            num(t["p5_ms"], 4), num(t["p95_ms"], 4),
            f"{t['tokens_per_s']:.0f}",
            num(c["max_abs_err"], 5),
            num(cold["total_s"], 1),
        ])
    headers = ["impl", "scope", "compiler", "mean ms", "med ms", "p5 ms",
               "p95 ms", "tok/s", "max|err|", "cold s"]
    env = payload["environment"]
    head = (f"Stage 1 -- Qwen3 dense MLP  "
            f"[{payload['config']['tokens']}x{payload['config']['hidden_size']}"
            f"->{payload['config']['intermediate_size']}, bf16]  "
            f"on {env.get('gpu_name','?')} (cc {env.get('gpu_compute_capability','?')})")
    return head + "\n" + fmt_table(rows, headers)


if __name__ == "__main__":
    raise SystemExit(main())
