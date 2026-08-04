"""Benchmark complete Qwen3 MPK prefill through the first generated token.

All modes use the normal handwritten paged attention. The two generated MLP
modes compare one fused SwiGLU task against three separate generated tasks.
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
MODES = ("handwritten", "generated_mlp_fused", "generated_mlp_separate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--prompt-lengths", default="16,32,64,128")
    parser.add_argument("--max-flattened-tokens", type=int, default=64,
                        help="Current Qwen3 SM100 attention smem limit is 64")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    batches = [int(v) for v in args.batches.split(",")]
    prompts = [int(v) for v in args.prompt_lengths.split(",")]
    gpu_ids = [v.strip() for v in args.gpu_ids.split(",") if v.strip()]
    cases = [(b, p) for p in prompts for b in batches
             if b * p <= args.max_flattened_tokens]
    jobs = [(mode, b, p) for b, p in cases for mode in MODES]
    locks = {gpu: threading.Lock() for gpu in gpu_ids}

    with tempfile.TemporaryDirectory(prefix="mpk-prefill-matrix-") as temp:
        temp_path = Path(temp)

        def launch(index_job):
            _, (mode, batch, prompt) = index_job
            case_index = cases.index((batch, prompt))
            gpu = gpu_ids[case_index % len(gpu_ids)]
            job_dir = temp_path / f"{mode}-b{batch}-p{prompt}"
            job_dir.mkdir()
            command = [
                sys.executable,
                str(RUNNER),
                "--model", str(Path(args.model).resolve()),
                "--max-num-batched-tokens", str(batch * prompt),
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
            if mode in ("generated_mlp_fused", "generated_mlp_separate"):
                env["MPK_COMPILED_MLP"] = "all"
            if mode == "generated_mlp_separate":
                env["MPK_COMPILED_MLP_IMPL"] = "separate"
            with locks[gpu]:
                start = time.perf_counter()
                completed = subprocess.run(
                    command, cwd=job_dir, env=env,
                    capture_output=True, text=True
                )
                wall_s = time.perf_counter() - start
            log = completed.stdout + "\n" + completed.stderr
            row = {
                "mode": mode,
                "batch": batch,
                "prompt_length": prompt,
                "flattened_tokens": batch * prompt,
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
        r["prompt_length"], r["batch"], MODES.index(r["mode"])))
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

    print("prompt batch flat  mode                    TTFT_ms")
    for row in rows:
        if "error" in row:
            print(f"{row['prompt_length']:>6} {row['batch']:>5} "
                  f"{row['flattened_tokens']:>4}  {row['mode']:<22} ERROR")
        else:
            print(f"{row['prompt_length']:>6} {row['batch']:>5} "
                  f"{row['flattened_tokens']:>4}  {row['mode']:<22} "
                  f"{row['prefill_to_first_token_ms']:>9.1f}")
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
