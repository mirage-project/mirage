"""Benchmark handwritten and compiler-generated Qwen3 SwiGLU MPK tasks.

Dense MLP kernels see a flattened token dimension M = batch * sequence.  The
matrix driver therefore measures unique M values and labels the same result as
decode (sequence=1) or prefill configurations where appropriate.

Examples:
    PYTHONPATH=python python experiments/compiled_tasks/bench_mlp_shapes.py \
        --matrix --tokens 1,2,4,8,16,32,64,128 --gpu-ids 0,2,3,4,7

    PYTHONPATH=python python experiments/compiled_tasks/bench_mlp_shapes.py \
        --worker --mode generated_fused --tokens 128 --output result.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

MODES = ("handwritten", "generated_fused", "generated_split")


def parse_args():
    parser = argparse.ArgumentParser()
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--matrix", action="store_true")
    kind.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _grid_for_linear(output_dim):
    if output_dim % 96 == 0:
        return output_dim // 96
    if output_dim % 64 == 0:
        return output_dim // 64
    raise ValueError(f"unsupported handwritten output dimension {output_dim}")


def run_worker(args):
    import torch
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel

    assert args.mode in MODES
    token_values = [int(v) for v in args.tokens.split(",")]
    assert len(token_values) == 1, "worker accepts exactly one token count"
    M = token_values[0]
    H, I = args.hidden, args.intermediate
    assert H % 64 == 0 and I % 64 == 0

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    x = torch.randn(M, H, device=device, dtype=dtype)
    # Keep both layouts resident. Transposition is graph-construction work and
    # is deliberately excluded from steady-state task timings.
    wg = torch.randn(I, H, device=device, dtype=dtype)
    wu = torch.randn(I, H, device=device, dtype=dtype)
    wg_t = wg.t().contiguous()
    wu_t = wu.t().contiguous()
    output = torch.zeros(M, I, device=device, dtype=dtype)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=max(M, 8),
        max_num_batched_requests=max(M, 8),
    )
    pk = PersistentKernel(**params)
    x_dt = pk.attach_input(x, name="shape_bench_x")
    out_dt = pk.attach_input(output, name="shape_bench_out")

    if args.mode == "handwritten":
        fused = torch.zeros(M, 2 * I, device=device, dtype=dtype)
        wg_dt = pk.attach_input(wg, name="shape_bench_wg")
        wu_dt = pk.attach_input(wu, name="shape_bench_wu")
        fused_dt = pk.attach_input(fused, name="shape_bench_fused")
        num_linear_tasks = _grid_for_linear(2 * I)
        shuffled = pk.shuffle_tensors(
            inputs=[wg_dt, wu_dt],
            shuffled_dim=0,
            num_groups=num_linear_tasks // 2,
            name="shape_bench_gatedup",
        )
        pk.linear_layer(
            input=x_dt,
            weight=shuffled,
            output=fused_dt,
            grid_dim=(num_linear_tasks, 1, 1),
            block_dim=(256, 1, 1),
        )
        pk.silu_mul_layer(
            input=fused_dt,
            output=out_dt,
            grid_dim=(num_linear_tasks // 2, 1, 1),
            block_dim=(256, 1, 1),
        )
    elif args.mode == "generated_fused":
        wg_dt = pk.attach_input(wg_t, name="shape_bench_wg_t")
        wu_dt = pk.attach_input(wu_t, name="shape_bench_wu_t")
        pk.generated_swiglu_layer(
            input=x_dt,
            gate_weight_t=wg_dt,
            up_weight_t=wu_dt,
            output=out_dt,
            grid_dim=(I // 64, 1, 1),
            block_dim=(256, 1, 1),
            forloop_range=H // 64,
        )
    else:
        gate = torch.zeros(M, I, device=device, dtype=dtype)
        up = torch.zeros(M, I, device=device, dtype=dtype)
        wg_dt = pk.attach_input(wg_t, name="shape_bench_wg_t")
        wu_dt = pk.attach_input(wu_t, name="shape_bench_wu_t")
        gate_dt = pk.attach_input(gate, name="shape_bench_gate")
        up_dt = pk.attach_input(up, name="shape_bench_up")
        for weight, temporary in ((wg_dt, gate_dt), (wu_dt, up_dt)):
            pk.generated_linear_layer(
                input=x_dt,
                weight_t=weight,
                output=temporary,
                grid_dim=(I // 64, 1, 1),
                block_dim=(256, 1, 1),
                forloop_range=H // 64,
            )
        pk.generated_silu_mul_layer(
            gate=gate_dt,
            up=up_dt,
            output=out_dt,
            grid_dim=(I // 64, 1, 1),
            block_dim=(256, 1, 1),
        )

    compile_start = time.perf_counter()
    pk.compile(output_dir=None)
    compile_s = time.perf_counter() - compile_start

    pk()
    torch.cuda.synchronize()
    gate_ref = (x.float() @ wg_t.float()).to(dtype).float()
    up_ref = (x.float() @ wu_t.float()).to(dtype).float()
    ref = (torch.nn.functional.silu(gate_ref) * up_ref).to(dtype)
    max_abs = (output.float() - ref.float()).abs().max().item()
    rel = max_abs / max(ref.float().abs().max().item(), 1e-12)

    for _ in range(args.warmups):
        pk()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for start, end in zip(starts, ends):
        start.record()
        pk()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    pk.finalize()

    result = {
        "mode": args.mode,
        "tokens": M,
        "hidden": H,
        "intermediate": I,
        "gpu": torch.cuda.get_device_name(0),
        "compile_s": compile_s,
        "mean_ms": statistics.fmean(samples),
        "p5_ms": _percentile(samples, 0.05),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "tokens_per_s": M * 1000.0 / statistics.fmean(samples),
        "max_abs_error": max_abs,
        "relative_max_error": rel,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def run_matrix(args):
    tokens = [int(v) for v in args.tokens.split(",")]
    gpu_ids = [v.strip() for v in args.gpu_ids.split(",") if v.strip()]
    jobs = [(mode, m) for m in tokens for mode in MODES]
    python = sys.executable
    script = Path(__file__).resolve()
    gpu_locks = {gpu: threading.Lock() for gpu in gpu_ids}

    with tempfile.TemporaryDirectory(prefix="mpk-shape-bench-") as temp:
        temp_path = Path(temp)

        def launch(index_job):
            index, (mode, m) = index_job
            # Keep all implementations of one shape on the same physical GPU;
            # B200 clock/power variation is material for ~0.15 ms tasks.
            shape_index = tokens.index(m)
            gpu = gpu_ids[shape_index % len(gpu_ids)]
            result_path = temp_path / f"{mode}-{m}.json"
            command = [
                python,
                str(script),
                "--worker",
                "--mode", mode,
                "--tokens", str(m),
                "--hidden", str(args.hidden),
                "--intermediate", str(args.intermediate),
                "--warmups", str(args.warmups),
                "--iters", str(args.iters),
                "--output", str(result_path),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(PYTHON_DIR)
            # Job durations differ (especially generated vs handwritten
            # compilation), so modulo assignment alone does not prevent a
            # later worker from reaching the same GPU while its predecessor is
            # still running. Serialize explicitly per device.
            with gpu_locks[gpu]:
                completed = subprocess.run(
                    command, env=env, cwd=ROOT, capture_output=True, text=True
                )
            if completed.returncode != 0:
                return {
                    "mode": mode,
                    "tokens": m,
                    "gpu_id": gpu,
                    "error": (completed.stdout + "\n" + completed.stderr)[-8000:],
                }
            result = json.loads(result_path.read_text())
            result["gpu_id"] = gpu
            return result

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpu_ids)
        ) as executor:
            results = list(executor.map(launch, enumerate(jobs)))

    results.sort(key=lambda r: (r["tokens"], MODES.index(r["mode"])))
    payload = {
        "description": "Qwen3-0.6B gated-up SwiGLU MPK task shape sweep",
        "shape_semantics": "MLP M equals batch_size * sequence_length",
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print("tokens  mode                 mean_ms    p95_ms      tok/s      rel_err")
    for row in results:
        if "error" in row:
            print(f"{row['tokens']:>6}  {row['mode']:<20} ERROR")
        else:
            print(
                f"{row['tokens']:>6}  {row['mode']:<20} "
                f"{row['mean_ms']:>9.4f} {row['p95_ms']:>9.4f} "
                f"{row['tokens_per_s']:>10.0f} {row['relative_max_error']:>12.3e}"
            )
    return 1 if any("error" in row for row in results) else 0


def main():
    args = parse_args()
    if args.worker:
        if args.mode is None:
            raise SystemExit("--worker requires --mode")
        run_worker(args)
        return 0
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
