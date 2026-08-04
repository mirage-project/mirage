"""Benchmark complete Qwen3 MPK prefill through the first generated token.

All modes use the normal handwritten paged attention. The default matrix
compares handwritten MLP, three generated tasks, and the two possible
two-generated-task fusion boundaries. Token counts are prompt lengths per
request; batch is the number of prompts in the request. The flattened token
count passed to the runner is therefore ``batch * prompt_length``.
Generated attention is intentionally excluded: its prep/core path only produces
the last query row and is not a correct full-model prefill across multiple
transformer layers.
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
)
ABLATION_MODES = (
    "handwritten",
    "generated_mlp_three_task",
    "generated_mlp_up_silu_fused",
    "generated_mlp_silu_down_fused",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument(
        "--token-counts",
        default="32,64,128,256,512,1024",
        help="prompt tokens per request (one prompt per batch element)",
    )
    parser.add_argument(
        "--prompt-lengths",
        help="deprecated alias for --token-counts",
    )
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument(
        "--job-timeout-seconds",
        type=int,
        default=900,
        help="per-mode timeout including compilation (default: 900)",
    )
    parser.add_argument(
        "--modes",
        default=",".join(ABLATION_MODES),
        help="comma-separated subset of: " + ",".join(MODES),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    batches = [int(v) for v in args.batches.split(",")]
    modes = [v.strip() for v in args.modes.split(",") if v.strip()]
    unknown_modes = set(modes) - set(MODES)
    if unknown_modes:
        raise ValueError(f"unknown modes: {sorted(unknown_modes)}")
    gpu_ids = [v.strip() for v in args.gpu_ids.split(",") if v.strip()]
    prompt_lengths = [
        int(v) for v in (args.prompt_lengths or args.token_counts).split(",")
    ]
    cases = [
        (b, prompt_length, b * prompt_length)
        for prompt_length in prompt_lengths
        for b in batches
    ]
    jobs = [(mode, b, p, t) for b, p, t in cases for mode in modes]
    locks = {gpu: threading.Lock() for gpu in gpu_ids}

    with tempfile.TemporaryDirectory(prefix="mpk-prefill-matrix-") as temp:
        temp_path = Path(temp)

        def launch(index_job):
            _, (mode, batch, prompt, token_count) = index_job
            case_index = cases.index((batch, prompt, token_count))
            gpu = gpu_ids[case_index % len(gpu_ids)]
            job_dir = temp_path / f"{mode}-t{token_count}-b{batch}-p{prompt}"
            job_dir.mkdir()
            command = [
                sys.executable,
                str(RUNNER),
                "--model", str(Path(args.model).resolve()),
                "--max-num-batched-tokens", str(token_count),
                "--max-num-batched-requests", str(batch),
                "--prompt-length", str(prompt),
                # Exactly one generated token after the prompt: the measured
                # launch is complete prefill through first-token production.
                "--max-seq-length", str(prompt + 1),
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
            ):
                env["MPK_COMPILED_MLP"] = "all"
            if mode == "generated_mlp_separate":
                env["MPK_COMPILED_MLP_IMPL"] = "separate"
            elif mode == "generated_mlp_three_task":
                env["MPK_COMPILED_MLP_IMPL"] = "three_task"
            elif mode == "generated_mlp_up_silu_fused":
                env["MPK_COMPILED_MLP_IMPL"] = "two_task_up_silu"
            elif mode == "generated_mlp_silu_down_fused":
                env["MPK_COMPILED_MLP_IMPL"] = "two_task_silu_down"
            with locks[gpu]:
                start = time.perf_counter()
                try:
                    completed = subprocess.run(
                        command,
                        cwd=job_dir,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=args.job_timeout_seconds,
                    )
                except subprocess.TimeoutExpired as exc:
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode(errors="replace")
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                    return {
                        "mode": mode,
                        "batch": batch,
                        "prompt_length": prompt,
                        "flattened_tokens": token_count,
                        "gpu_id": gpu,
                        "wall_s_including_compile": time.perf_counter() - start,
                        "error": (
                            f"timed out after {args.job_timeout_seconds}s\n"
                            + stdout[-6000:]
                            + "\n"
                            + stderr[-6000:]
                        ),
                    }
                wall_s = time.perf_counter() - start
            log = completed.stdout + "\n" + completed.stderr
            row = {
                "mode": mode,
                "batch": batch,
                "prompt_length": prompt,
                "flattened_tokens": token_count,
                "gpu_id": gpu,
                "wall_s_including_compile": wall_s,
            }
            total = re.search(r"total time:\s+([0-9.]+) ms", log)
            generated = re.search(r"generated \(total\):\s+([0-9]+)", log)
            reached = re.search(r"sequence length:\s+([0-9]+) /", log)
            sample = re.search(r"Sample output .*?: (.+)", log)
            # Prefer the runner's full-precision JSON over its rounded console
            # table. The regexes remain the failure-path fallback.
            result_files = list((job_dir / "outputs" / "qwen3").glob("*.json"))
            perf = json.loads(result_files[0].read_text()) if result_files else None
            if perf:
                row["prefill_to_first_token_ms"] = perf["total_time_ms"]
                row["generated_tokens"] = perf["generate_length_total"]
                row["reached_sequence_length"] = perf["sequence_length"]
            else:
                if total:
                    row["prefill_to_first_token_ms"] = float(total.group(1))
                if generated:
                    row["generated_tokens"] = int(generated.group(1))
                if reached:
                    row["reached_sequence_length"] = int(reached.group(1))
            if sample:
                row["sample"] = sample.group(1)
            if completed.returncode != 0:
                row["error"] = log[-12000:]
            elif perf is None and (not total or not generated or not reached):
                row["error"] = "could not parse benchmark output\n" + log[-6000:]
            elif (row["generated_tokens"] != batch or
                  row["reached_sequence_length"] != prompt + 1):
                row["error"] = (
                    f"incomplete prefill/first-token run: generated "
                    f"{row['generated_tokens']} of {batch}; reached "
                    f"{row['reached_sequence_length']} of {prompt + 1}"
                )
            return row

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpu_ids)
        ) as executor:
            rows = list(executor.map(launch, enumerate(jobs)))

    rows.sort(key=lambda r: (
        r["prompt_length"], r["batch"], modes.index(r["mode"])))
    payload = {
        "description": (
            "Qwen3-0.6B complete MPK prefill through first generated token; "
            "handwritten attention in all modes"
        ),
        "results": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print("prompt batch flattened mode                    TTFT_ms")
    for row in rows:
        if "error" in row:
            print(f"{row['prompt_length']:>6} {row['batch']:>5} "
                  f"{row['flattened_tokens']:>9}  {row['mode']:<30} ERROR")
        else:
            print(f"{row['prompt_length']:>6} {row['batch']:>5} "
                  f"{row['flattened_tokens']:>9}  {row['mode']:<30} "
                  f"{row['prefill_to_first_token_ms']:>9.1f}")
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
