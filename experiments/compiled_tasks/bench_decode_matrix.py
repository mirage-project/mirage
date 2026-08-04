"""Run the Qwen3 full-model decode benchmark across compiled-task modes.

This wraps tests/ci-tests/run_batch_perf.py, keeping its one-token prompt and
varying logical request batch and total sequence length. The MPK tensor shape
stays at eight tokens, which is the supported generated-task decode bucket.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tests" / "ci-tests" / "run_batch_perf.py"
PYTHON_DIR = ROOT / "python"
MODES = (
    "handwritten",
    "generated_mlp_fused",
    "generated_mlp_separate",
    "generated_mlp_three_task",
    "generated_mlp_up_silu_fused",
    "generated_mlp_silu_down_fused",
    "generated_attention",
    "generated_attention_mlp_three_task",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--sequence-lengths", default="128,512")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="comma-separated subset of: " + ",".join(MODES),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    batches = [int(v) for v in args.batches.split(",")]
    lengths = [int(v) for v in args.sequence_lengths.split(",")]
    modes = [v.strip() for v in args.modes.split(",") if v.strip()]
    unknown_modes = set(modes) - set(MODES)
    if unknown_modes:
        raise ValueError(f"unknown modes: {sorted(unknown_modes)}")
    gpu_ids = [v.strip() for v in args.gpu_ids.split(",") if v.strip()]
    gpu_locks = {gpu: threading.Lock() for gpu in gpu_ids}
    jobs = [(mode, batch, length) for length in lengths for batch in batches
            for mode in modes]

    with tempfile.TemporaryDirectory(prefix="mpk-decode-matrix-") as temp:
        temp_path = Path(temp)

        def launch(index_job):
            index, (mode, batch, length) = index_job
            # Compare modes on the same physical GPU for each (batch, length)
            # case. The per-GPU lock then runs those modes sequentially.
            case_index = lengths.index(length) * len(batches) + batches.index(batch)
            gpu = gpu_ids[case_index % len(gpu_ids)]
            job_dir = temp_path / f"{mode}-b{batch}-s{length}"
            job_dir.mkdir()
            command = [
                sys.executable,
                str(RUNNER),
                "--model", str(Path(args.model).resolve()),
                "--max-num-batched-tokens", "8",
                "--max-num-batched-requests", str(batch),
                "--max-seq-length", str(length),
                "--ignore-eos",
                "--output-dir", str(job_dir / "compiled"),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(PYTHON_DIR)
            if mode in (
                "generated_mlp_fused",
                "generated_mlp_separate",
                "generated_mlp_three_task",
                "generated_mlp_up_silu_fused",
                "generated_mlp_silu_down_fused",
                "generated_attention_mlp_three_task",
            ):
                env["MPK_COMPILED_MLP"] = "all"
            if mode == "generated_mlp_separate":
                env["MPK_COMPILED_MLP_IMPL"] = "separate"
            elif mode in (
                "generated_mlp_three_task",
                "generated_attention_mlp_three_task",
            ):
                env["MPK_COMPILED_MLP_IMPL"] = "three_task"
            elif mode == "generated_mlp_up_silu_fused":
                env["MPK_COMPILED_MLP_IMPL"] = "two_task_up_silu"
            elif mode == "generated_mlp_silu_down_fused":
                env["MPK_COMPILED_MLP_IMPL"] = "two_task_silu_down"
            if mode in (
                "generated_attention",
                "generated_attention_mlp_three_task",
            ):
                env["MPK_COMPILED_ATTENTION"] = "all"
            # Generated-attention compilation is slower than the other modes.
            # Without a per-device lock, modulo assignment can overlap two
            # later jobs on one GPU after the faster jobs advance the queue.
            with gpu_locks[gpu]:
                start = time.perf_counter()
                completed = subprocess.run(
                    command, cwd=job_dir, env=env,
                    capture_output=True, text=True
                )
                wall_s = time.perf_counter() - start
            log = completed.stdout + "\n" + completed.stderr
            base = {
                "mode": mode,
                "batch": batch,
                "sequence_length": length,
                "gpu_id": gpu,
                "wall_s_including_compile": wall_s,
            }
            total = re.search(r"total time:\s+([0-9.]+) ms", log)
            throughput = re.search(r"throughput:\s+([0-9.]+) tokens/s", log)
            generated = re.search(r"generated \(total\):\s+([0-9]+)", log)
            reached = re.search(r"sequence length:\s+([0-9]+) /", log)
            sample = re.search(r"Sample output .*?: (.+)", log)
            if total and throughput:
                base.update(
                    total_time_ms=float(total.group(1)),
                    throughput_tokens_per_s=float(throughput.group(1)),
                    generated_tokens=int(generated.group(1)) if generated else None,
                    reached_sequence_length=int(reached.group(1)) if reached else None,
                    sample=sample.group(1) if sample else None,
                )
            if completed.returncode != 0:
                base["error"] = log[-12000:]
                return base
            if not total or not throughput or not generated or not reached:
                base["error"] = "could not parse benchmark output\n" + log[-6000:]
                return base
            expected = batch * (length - 1)  # runner uses a one-token prompt
            if int(generated.group(1)) != expected or int(reached.group(1)) != length:
                base["error"] = (
                    f"incomplete decode: generated {generated.group(1)} of "
                    f"{expected} expected tokens; reached sequence "
                    f"{reached.group(1)} of {length}"
                )
            return base

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpu_ids)
        ) as executor:
            rows = list(executor.map(launch, enumerate(jobs)))

    rows.sort(key=lambda r: (
        r["sequence_length"], r["batch"], modes.index(r["mode"])))
    payload = {
        "description": "Qwen3-0.6B full-model decode sweep, one-token prompt",
        "max_num_batched_tokens": 8,
        "results": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print("seq  batch  mode                    total_ms      tok/s")
    for row in rows:
        if "error" in row:
            print(f"{row['sequence_length']:>3} {row['batch']:>6}  "
                  f"{row['mode']:<22} ERROR")
        else:
            print(f"{row['sequence_length']:>3} {row['batch']:>6}  "
                  f"{row['mode']:<22} {row['total_time_ms']:>10.1f} "
                  f"{row['throughput_tokens_per_s']:>10.1f}")
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
