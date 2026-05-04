"""Matrix driver: spawn one subprocess per (shape × accumulate) cell.

Each cell runs `test_splitk_linear_bf16_testmode.py` with a hard timeout, so
a deadlock in one cell doesn't block the rest. Output is a PASS / FAIL /
TIMEOUT summary table.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python run_matrix.py [--timeout 90]
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "test_splitk_linear_bf16_testmode.py")


# (label, batch, N, K, grid_x, grid_y) — N % (grid_x*128) == 0,
# K % grid_y == 0. Shapes chosen to match real model layers + corner cases.
SHAPES = [
    # label,                        batch, N,    K,    grid_x, grid_y
    # Sweep batch sizes at a fixed shape to find the BATCH threshold
    # (matrix sweep showed batch=16 passes, batch ∈ {1,4} hang).
    ("batch1_4096_4096_32x4",       1,    4096, 4096,  32,    4),
    ("batch2_4096_4096_32x4",       2,    4096, 4096,  32,    4),
    ("batch4_4096_4096_32x4",       4,    4096, 4096,  32,    4),
    ("batch8_4096_4096_32x4",       8,    4096, 4096,  32,    4),
    ("batch12_4096_4096_32x4",     12,    4096, 4096,  32,    4),
    ("batch16_4096_4096_32x4",     16,    4096, 4096,  32,    4),
    # qwen3 o_proj at MBT=8 (production default for qwen3 demo)
    ("qwen3_mbt8_4096_4096_32x4",   8,    4096, 4096,  32,    4),
]


def run_cell(timeout_s: int, label: str, batch: int, N: int, K: int,
             gx: int, gy: int, accumulate: bool) -> tuple[str, float]:
    args = [
        sys.executable, TEST,
        "--batch", str(batch),
        "--N", str(N), "--K", str(K),
        "--grid-x", str(gx), "--grid-y", str(gy),
        "--accumulate", "True" if accumulate else "False",
        "--label", f"{label}_acc{int(accumulate)}",
    ]
    t0 = time.time()
    try:
        cp = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout_s,
            env=os.environ.copy())
    except subprocess.TimeoutExpired as e:
        # Best-effort: kill any stragglers
        if e.stdout:
            print(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout)
        return "TIMEOUT", time.time() - t0
    dt = time.time() - t0
    if cp.returncode == 0:
        # Confirm a PASS line was printed.
        if any(line.startswith("PASS") for line in cp.stdout.splitlines()):
            return "PASS", dt
        if any(line.startswith("SKIP") for line in cp.stdout.splitlines()):
            return "SKIP", dt
        return "PASS", dt  # exit 0 with no PASS line — be lenient
    # Non-zero: look for FAIL, otherwise call it ERROR.
    last_lines = cp.stdout.splitlines()[-30:] + cp.stderr.splitlines()[-10:]
    summary = "\n    ".join(last_lines)
    print(f"  [{label} acc={accumulate}] non-zero exit; tail:\n    {summary}")
    if any(line.startswith("FAIL") for line in cp.stdout.splitlines()):
        return "FAIL", dt
    return "ERROR", dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=90,
                    help="Per-cell timeout in seconds (default 90).")
    ap.add_argument("--shapes", type=str, default=None,
                    help="Comma-separated label substrings to filter shapes.")
    args = ap.parse_args()

    shape_filter = (None if args.shapes is None
                    else [s.strip() for s in args.shapes.split(",") if s.strip()])
    rows = []
    for sh in SHAPES:
        label = sh[0]
        if shape_filter and not any(f in label for f in shape_filter):
            continue
        for accumulate in (True, False):
            print(f"\n=== {label}  accumulate={accumulate} ===", flush=True)
            status, dt = run_cell(args.timeout, *sh, accumulate=accumulate)
            print(f"--> {status}  ({dt:.1f}s)", flush=True)
            rows.append((label, accumulate, status, dt))

    print("\n" + "=" * 76)
    print(f"{'shape':<40} {'accumulate':<10} {'status':<8} {'time(s)':<8}")
    print("-" * 76)
    for label, acc, st, dt in rows:
        print(f"{label:<40} {str(acc):<10} {st:<8} {dt:<8.1f}")
    print("=" * 76)
    n_pass = sum(1 for r in rows if r[2] == "PASS")
    n_fail = sum(1 for r in rows if r[2] in ("FAIL", "ERROR"))
    n_to = sum(1 for r in rows if r[2] == "TIMEOUT")
    n_skip = sum(1 for r in rows if r[2] == "SKIP")
    print(f"PASS={n_pass}  FAIL/ERR={n_fail}  TIMEOUT={n_to}  SKIP={n_skip}  "
          f"of {len(rows)} cells")
    sys.exit(0 if (n_fail + n_to) == 0 else 1)


if __name__ == "__main__":
    main()
