#!/usr/bin/env python3
"""
Evaluation script for non-symbolic vs symbolic search across multiple kernels and shapes.

Symbolic search runs once (shape-independent) and produces a checkpoint.
Auto-tuning then specializes the symbolic graphs to each concrete shape.
Non-symbolic search must re-run from scratch for each shape.

Usage:
  python3 eval_symbolic.py                           # run all kernels
  python3 eval_symbolic.py --kernels swiglu,lora     # subset of kernels
  python3 eval_symbolic.py --time-limit 60           # short search (seconds)
  python3 eval_symbolic.py --dry-run                 # print commands only
  python3 eval_symbolic.py --force                   # re-run everything
  python3 eval_symbolic.py --timeout 3600            # per-subprocess timeout
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

KERNELS = {
    "rmsnorm_mlp": {
        "binary": "symbolic_rmsnorm_mlp",
        "shapes": [
            (8, 1024), (8, 2048), (8, 4096), (16, 1024), (16, 4096),
        ],
        "config_fmt": lambda s: f"{s[0]},{s[1]}",
        "shape_label": lambda s: f"n={s[0]},d={s[1]}",
        "shape_keys": lambda s: {"n": s[0], "d": s[1]},
        "ckpt_dir": "checkpoints/rmsnorm_mlp",
        "results_file": "results_rmsnorm_mlp.json",
    },
    "swiglu": {
        "binary": "symbolic_swiglu",
        "shapes": [
            (8, 1024), (8, 2048), (8, 4096), (16, 1024), (16, 4096),
        ],
        "config_fmt": lambda s: f"{s[0]},{s[1]}",
        "shape_label": lambda s: f"n={s[0]},d={s[1]}",
        "shape_keys": lambda s: {"n": s[0], "d": s[1]},
        "ckpt_dir": "checkpoints/swiglu",
        "results_file": "results_swiglu.json",
    },
    "lora": {
        "binary": "symbolic_lora",
        "shapes": [
            (8, 4096, 16), (8, 4096, 64), (8, 4096, 128),
            (8, 2048, 64), (16, 4096, 64),
        ],
        "config_fmt": lambda s: f"{s[0]},{s[1]},{s[2]}",
        "shape_label": lambda s: f"n={s[0]},d={s[1]},r={s[2]}",
        "shape_keys": lambda s: {"n": s[0], "d": s[1], "r": s[2]},
        "ckpt_dir": "checkpoints/lora",
        "results_file": "results_lora.json",
    },
    "two_layer_mlp": {
        "binary": "symbolic_two_layer_mlp",
        "shapes": [
            (8, 1024), (8, 2048), (8, 4096), (16, 1024), (16, 4096),
        ],
        "config_fmt": lambda s: f"{s[0]},{s[1]}",
        "shape_label": lambda s: f"n={s[0]},d={s[1]}",
        "shape_keys": lambda s: {"n": s[0], "d": s[1]},
        "ckpt_dir": "checkpoints/two_layer_mlp",
        "results_file": "results_two_layer_mlp.json",
    },
}

# ---------------------------------------------------------------------------
# Stdout parsers
# ---------------------------------------------------------------------------

def parse_search_time(stdout: str) -> float | None:
    """Parse '[Search] ... Time elapsed: Xsec' or 'Symbolic search ... Time elapsed: Xsec'."""
    m = re.search(r"Time elapsed:\s*([\d.]+)\s*sec", stdout)
    return float(m.group(1)) if m else None


def parse_kernel_time(stdout: str, kind: str) -> float | None:
    """Parse 'Best time (non-symbolic): X ms' or 'Best time (symbolic): X ms'."""
    pattern = rf"Best time \({kind}\):\s*([\d.eE+\-]+)\s*ms"
    m = re.search(pattern, stdout)
    if m:
        val = float(m.group(1))
        if val >= 1e30:
            return None
        return val
    return None


def parse_num_graphs(stdout: str) -> int | None:
    """Parse number of graphs found from checkpoint or search output."""
    m = re.search(r"Profiling\s+(\d+)\s+graphs", stdout)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], cwd: str, timeout: float,
                dry_run: bool = False) -> dict:
    """Run a command and return {stdout, stderr, returncode, wall_time, error}."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY-RUN] {cmd_str}")
        return {
            "stdout": "", "stderr": "", "returncode": 0,
            "wall_time": 0.0, "error": None, "cmd": cmd_str,
        }

    print(f"  [RUN] {cmd_str}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        wall = time.time() - t0
        if proc.returncode != 0:
            snippet = (proc.stderr or proc.stdout or "")[:500]
            return {
                "stdout": proc.stdout, "stderr": proc.stderr,
                "returncode": proc.returncode, "wall_time": wall,
                "error": f"exit code {proc.returncode}: {snippet}",
                "cmd": cmd_str,
            }
        return {
            "stdout": proc.stdout, "stderr": proc.stderr,
            "returncode": 0, "wall_time": wall, "error": None,
            "cmd": cmd_str,
        }
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        return {
            "stdout": "", "stderr": "", "returncode": -1,
            "wall_time": wall, "error": f"TIMEOUT after {timeout}s",
            "cmd": cmd_str,
        }
    except Exception as e:
        wall = time.time() - t0
        return {
            "stdout": "", "stderr": str(e), "returncode": -1,
            "wall_time": wall, "error": str(e), "cmd": cmd_str,
        }


# ---------------------------------------------------------------------------
# Results file helpers
# ---------------------------------------------------------------------------

def load_eval_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_eval_results(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate_kernel(kernel_name: str, kdef: dict, args, build_dir: str,
                    bin_dir: str, eval_results: dict):
    """Run full evaluation for one kernel."""
    print(f"\n{'='*60}")
    print(f"  KERNEL: {kernel_name}")
    print(f"{'='*60}")

    binary = os.path.join(bin_dir, kdef["binary"])
    if not args.dry_run and not os.path.isfile(binary):
        print(f"  ERROR: binary not found: {binary}")
        return

    kr = eval_results.setdefault(kernel_name, {})

    # --- Phase 1: Symbolic search (once) -----------------------------------
    sym_key = "symbolic_search"
    if not args.force and sym_key in kr and kr[sym_key].get("error") is None:
        print(f"\n  Symbolic search: cached (search_time={kr[sym_key].get('search_time')}s)")
    else:
        print(f"\n  --- Symbolic search (once) ---")
        # Use first shape as the debug config for search
        first_shape = kdef["shapes"][0]
        cmd = [
            binary,
            "--config", kdef["config_fmt"](first_shape),
            "--skip-nonsym",
            "--force-sym",
        ]
        if args.time_limit is not None:
            cmd += ["--time-limit", str(args.time_limit)]

        result = run_command(cmd, build_dir, args.timeout, args.dry_run)
        sr = {
            "wall_time": result["wall_time"],
            "search_time": parse_search_time(result["stdout"]),
            "error": result["error"],
            "cmd": result["cmd"],
        }
        kr[sym_key] = sr
        save_eval_results(args.output, eval_results)

        if result["error"]:
            print(f"  FAILED: {result['error']}")

    # --- Phase 2: Per-shape evaluation -------------------------------------
    shapes_results = kr.setdefault("shapes", {})

    for shape in kdef["shapes"]:
        label = kdef["shape_label"](shape)
        config_arg = kdef["config_fmt"](shape)

        print(f"\n  --- Shape: {label} ---")
        shape_r = shapes_results.setdefault(label, {})

        # Non-symbolic search
        ns_key = "nonsym"
        if not args.force and ns_key in shape_r and shape_r[ns_key].get("error") is None:
            print(f"    Non-symbolic: cached (kernel={shape_r[ns_key].get('kernel_ms')} ms)")
        else:
            cmd = [
                binary,
                "--config", config_arg,
                "--skip-sym",
                "--force-nonsym",
            ]
            if args.time_limit is not None:
                cmd += ["--time-limit", str(args.time_limit)]

            result = run_command(cmd, build_dir, args.timeout, args.dry_run)
            nr = {
                "wall_time": result["wall_time"],
                "search_time": parse_search_time(result["stdout"]),
                "kernel_ms": parse_kernel_time(result["stdout"], "non-symbolic"),
                "error": result["error"],
                "cmd": result["cmd"],
            }
            shape_r[ns_key] = nr
            save_eval_results(args.output, eval_results)

            if result["error"]:
                print(f"    Non-symbolic FAILED: {result['error']}")
            else:
                st = nr['search_time']
                kt = nr['kernel_ms']
                print(f"    Non-symbolic: search={st if st is not None else '?'}s, "
                      f"kernel={kt if kt is not None else '?'} ms")

        # Symbolic tune (checkpoint already exists from phase 1)
        sym_key_shape = "sym"
        if not args.force and sym_key_shape in shape_r and shape_r[sym_key_shape].get("error") is None:
            print(f"    Symbolic tune: cached (kernel={shape_r[sym_key_shape].get('kernel_ms')} ms)")
        else:
            cmd = [
                binary,
                "--config", config_arg,
                "--skip-nonsym",
                "--force-sym",
            ]
            # No --time-limit here: checkpoint exists, so search is skipped
            # and only auto-tuning runs. Wall clock = tune time.

            result = run_command(cmd, build_dir, args.timeout, args.dry_run)
            st = {
                "wall_time": result["wall_time"],
                "tune_time": result["wall_time"],  # entire run = tuning
                "kernel_ms": parse_kernel_time(result["stdout"], "symbolic"),
                "error": result["error"],
                "cmd": result["cmd"],
            }
            shape_r[sym_key_shape] = st
            save_eval_results(args.output, eval_results)

            if result["error"]:
                print(f"    Symbolic tune FAILED: {result['error']}")
            else:
                print(f"    Symbolic tune: tune={st['tune_time']:.1f}s, "
                      f"kernel={st['kernel_ms']} ms")


def print_summary_table(eval_results: dict):
    """Print markdown summary tables."""
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")

    for kname, kr in eval_results.items():
        sym_search = kr.get("symbolic_search", {})
        sym_search_time = sym_search.get("search_time")
        sym_search_str = f"{sym_search_time:.1f}" if sym_search_time is not None else "N/A"

        shapes = kr.get("shapes", {})
        if not shapes:
            continue

        print(f"\n### {kname}")
        print()

        header = (
            f"| {'Shape':<22} "
            f"| {'NS Search (s)':>14} | {'NS Kernel (ms)':>15} "
            f"| {'Sym Search (s)':>14} | {'Sym Tune (s)':>12} | {'Sym Kernel (ms)':>16} |"
        )
        sep = (
            f"|{'-'*24}"
            f"|{'-'*16}|{'-'*17}"
            f"|{'-'*16}|{'-'*14}|{'-'*18}|"
        )
        print(header)
        print(sep)

        for label, sr in shapes.items():
            ns = sr.get("nonsym", {})
            sy = sr.get("sym", {})

            def fmt(val, suffix=""):
                if val is None:
                    return "N/A"
                if isinstance(val, str):
                    return val
                return f"{val:.3f}{suffix}" if val < 100 else f"{val:.1f}{suffix}"

            ns_search = fmt(ns.get("search_time"))
            ns_kernel = fmt(ns.get("kernel_ms"))
            sy_search = sym_search_str  # same for all shapes (run once)
            sy_tune = fmt(sy.get("tune_time"))
            sy_kernel = fmt(sy.get("kernel_ms"))

            if ns.get("error"):
                ns_search = "FAILED"
                ns_kernel = "FAILED"
            if sy.get("error"):
                sy_tune = "FAILED"
                sy_kernel = "FAILED"

            print(
                f"| {label:<22} "
                f"| {ns_search:>14} | {ns_kernel:>15} "
                f"| {sy_search:>14} | {sy_tune:>12} | {sy_kernel:>16} |"
            )

        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate non-symbolic vs symbolic search across kernels and shapes",
    )
    parser.add_argument(
        "--kernels", type=str, default=None,
        help="Comma-separated list of kernels to evaluate (default: all)",
    )
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Search time limit in seconds (passed to C++ binaries)",
    )
    parser.add_argument(
        "--timeout", type=float, default=7200,
        help="Per-subprocess wall-clock timeout in seconds (default: 7200)",
    )
    parser.add_argument(
        "--build-dir", type=str, default=None,
        help="Build directory containing binaries (default: auto-detect)",
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="Output JSON file (default: eval_results.json)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run everything, ignoring cached results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Find build directory (root of the cmake build tree)
    if args.build_dir:
        build_dir = args.build_dir
    else:
        # Try common locations
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent
        candidates = [
            repo_root / "build",
            Path("/data/ubuntu/mirage-build"),
            Path("/data/ubuntu/mirage/build"),
        ]
        build_dir = None
        for c in candidates:
            if c.is_dir():
                build_dir = str(c)
                break
        if build_dir is None:
            print("ERROR: Could not find build directory. Use --build-dir.")
            sys.exit(1)

    # Binaries are output to cpp_examples/symbolic/ under the build root
    bin_dir = os.path.join(build_dir, "cpp_examples", "symbolic")
    if not os.path.isdir(bin_dir):
        # Fallback: build_dir itself might already point to the binary dir
        bin_dir = build_dir

    print(f"Build directory: {build_dir}")
    print(f"Binary directory: {bin_dir}")
    print(f"Output file:     {args.output}")

    # Select kernels
    if args.kernels:
        selected = [k.strip() for k in args.kernels.split(",")]
        for k in selected:
            if k not in KERNELS:
                print(f"ERROR: Unknown kernel '{k}'. Available: {', '.join(KERNELS)}")
                sys.exit(1)
    else:
        selected = list(KERNELS.keys())

    print(f"Kernels:         {', '.join(selected)}")
    if args.time_limit is not None:
        print(f"Time limit:      {args.time_limit}s")
    print(f"Timeout:         {args.timeout}s")

    # Load existing results (for resume support)
    eval_results = {} if args.force else load_eval_results(args.output)

    # Run evaluations
    for kname in selected:
        evaluate_kernel(kname, KERNELS[kname], args, build_dir, bin_dir,
                        eval_results)

    # Print summary
    if not args.dry_run:
        print_summary_table(eval_results)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
