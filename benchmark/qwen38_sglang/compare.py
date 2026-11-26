"""Run one exploratory Qwen3.8-27B backend pair per selected batch.

Each backend/batch gets a fresh process and exactly one timed generation.
No repeated trials, median, or one-token subtraction are performed.
The opt-in B8 path compares stock SGLang with SGLang using static GDN layers.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKER = HERE / "worker.py"
STATIC_GDN_BOOT = REPO / "demo" / "qwen38_static" / "boot"
STATIC_GDN_CAPTURE = re.compile(
    r"QWEN38_STATIC_GDN_CAPTURED layer=(\d+) T=(\d+) mode=([^\s]+)")
EXPECTED_GDN_LAYERS = 48
MAX_IDLE_MIB = 10 * 1024
MAX_IDLE_UTILIZATION = 5
MONITOR_INTERVAL_SECONDS = 0.2
MAX_MONITOR_GAP_SECONDS = 0.5
CUDA_PROBE = """import os
import torch
if torch.cuda.device_count() != 1:
    raise RuntimeError('Expected exactly one visible CUDA device')
torch.cuda.set_device(0)
properties = torch.cuda.get_device_properties(0)
if str(properties.uuid).lower().removeprefix('gpu-') != os.environ['QWEN38_BENCH_GPU_UUID'].lower().removeprefix('gpu-'):
    raise RuntimeError('CUDA selected a different GPU')
if torch.cuda.get_device_capability(0) != (10, 0) or 'B200' not in properties.name:
    raise RuntimeError('Selected CUDA device is not a B200')
torch.empty(1, device='cuda').fill_(1)
torch.cuda.synchronize()
print(str(properties.uuid), properties.name)
"""


def gpu_inventory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line}")
        index, uuid, name, memory, utilization = fields
        gpus.append(dict(index=int(index), uuid=uuid, name=name,
                         memory_used_mib=int(memory),
                         utilization_percent=int(utilization)))
    return gpus


def idle_candidates(inventory, requested="auto"):
    candidates = [gpu for gpu in inventory if "B200" in gpu["name"] and
                  (requested == "auto" or gpu["uuid"].lower() == requested.lower() or
                   str(gpu.get("index")) == requested) and
                  gpu["memory_used_mib"] <= MAX_IDLE_MIB and
                  gpu["utilization_percent"] <= MAX_IDLE_UTILIZATION]
    if not candidates:
        raise RuntimeError("No selected B200 is currently idle (<=10 GiB, <=5% compute)")
    return sorted(candidates, key=lambda gpu: (gpu["utilization_percent"],
                                                gpu["memory_used_mib"], gpu["uuid"]))


def choose_gpu(inventory, requested="auto"):
    return idle_candidates(inventory, requested)[0]


def probe_cuda(python, gpu, output_dir, stage, backend="mpk", sglang_cuda_home=None):
    """Sequential short-lived probe; no model load or timed generation."""
    log_path = output_dir / f"{stage}-probe-gpu{gpu['index']}.log"
    command = [str(python), "-c", CUDA_PROBE]
    try:
        completed = subprocess.run(command,
                                   env=child_env(backend, gpu["uuid"], sglang_cuda_home),
                                   text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=30, check=False)
        output = completed.stdout or ""
        ok = completed.returncode == 0
        error = None if ok else (output.strip().splitlines()[-1]
                                 if output.strip() else f"Exit code {completed.returncode}")
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        error = "CUDA initialization timed out after 30 seconds"
        ok = False
    log_path.write_text(output)
    return {"stage": stage, "gpu_index": gpu["index"], "gpu_uuid": gpu["uuid"],
            "ok": ok, "error": error, "log": log_path.name}


def select_usable_gpu(inventory, requested, probe, attempts):
    """Try idle B200s in order; a failed CUDA context never blocks the next."""
    for gpu in idle_candidates(inventory, requested):
        attempt = probe(gpu)
        attempts.append(attempt)
        if attempt["ok"]:
            return gpu
    if requested == "auto":
        raise RuntimeError("No idle B200 passed the CUDA initialization probe; "
                           "see gpu_probes and probe logs")
    raise RuntimeError(f"Selected GPU {requested} is idle in nvidia-smi but CUDA "
                       "initialization failed; see gpu_probes and probe log")


def first_mismatch(left, right, left_name="mpk", right_name="sglang"):
    if len(left) != len(right):
        return {"reason": "different row counts", f"{left_name}_rows": len(left),
                f"{right_name}_rows": len(right)}
    for row, (a, b) in enumerate(zip(left, right)):
        if len(a) != len(b):
            return {"row": row, "reason": "different output lengths",
                    f"{left_name}_tokens": len(a), f"{right_name}_tokens": len(b)}
        for step, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return {"row": row, "generated_index": step,
                        f"{left_name}_token": x, f"{right_name}_token": y}
    return None


def summarize_pair(batch, mpk, sglang, *, left_name="mpk", right_name="sglang",
                   ratio_key="raw_latency_ratio_sglang_over_mpk",
                   speedup_key="mpk_vs_sglang_speedup"):
    repeats = mpk.get("timed_generations", 1)
    warmups = mpk.get("warmup_generations", 0)
    result = {"batch_size": batch, left_name: mpk, right_name: sglang,
              "exploratory": warmups == 0 or repeats < 3,
              "timed_generations_per_backend": repeats,
              "warmup_generations_per_backend": warmups}
    if mpk.get("status") != "ok" or sglang.get("status") != "ok":
        result["status"] = "backend_error"
        result[speedup_key] = None
        return result
    if mpk["inputs_sha256"] != sglang["inputs_sha256"] or \
            mpk["config_sha256"] != sglang["config_sha256"] or \
            mpk["model_path"] != sglang["model_path"] or \
            mpk["checkpoint_manifest_sha256"] != sglang["checkpoint_manifest_sha256"] or \
            mpk["gpu_before"]["uuid"] != sglang["gpu_before"]["uuid"] or \
            mpk["batch_size"] != sglang["batch_size"] or mpk["batch_size"] != batch or \
            mpk["prompt_tokens"] != sglang["prompt_tokens"] or \
            mpk["new_tokens"] != sglang["new_tokens"] or \
            repeats != sglang.get("timed_generations", 1) or \
            warmups != sglang.get("warmup_generations", 0):
        result["status"] = "incomparable_configuration"
        result[speedup_key] = None
        return result
    result[f"{left_name}_elapsed_seconds"] = mpk["elapsed_seconds"]
    result[f"{right_name}_elapsed_seconds"] = sglang["elapsed_seconds"]
    result[f"{left_name}_output_tokens_per_second"] = mpk["output_tokens_per_second"]
    result[f"{right_name}_output_tokens_per_second"] = sglang["output_tokens_per_second"]
    result[ratio_key] = (
        sglang["elapsed_seconds"] / mpk["elapsed_seconds"])
    mismatch = first_mismatch(mpk["output_ids"], sglang["output_ids"],
                              left_name, right_name)
    result["output_ids_exact"] = mismatch is None
    result["first_mismatch"] = mismatch
    monitors = (mpk.get("gpu_monitor", {}), sglang.get("gpu_monitor", {}))
    monitor_covered = all(item.get("timed_window_covered") is True for item in monitors)
    external_pids = sorted(set().union(*(
        set(item.get("external_pids_during_timing", [])) for item in monitors)))
    result["gpu_monitor_covered"] = monitor_covered
    result["external_gpu_pids_during_timing"] = external_pids
    result["timing_uncontended"] = monitor_covered and not external_pids
    result["status"] = ("token_mismatch" if mismatch else
                        "external_gpu_processes" if external_pids else
                        "monitor_incomplete" if not monitor_covered else "ok")
    # Raw timing is retained, but exact IDs and monitored isolation gate speedup.
    result[speedup_key] = (
        result[ratio_key]
        if mismatch is None and result["timing_uncontended"] else None)
    return result


def capture_markers(log_path):
    markers = [(int(layer), int(tokens), mode) for layer, tokens, mode in
               STATIC_GDN_CAPTURE.findall(log_path.read_text(errors="replace"))]
    layer_ids = sorted({layer for layer, tokens, mode in markers
                        if tokens == 8 and mode == "hybrid"})
    return {"marker_count": len(markers), "unique_b8_layer_ids": layer_ids,
            "unique_b8_layer_count": len(layer_ids),
            "unexpected_token_counts": sorted({tokens for _, tokens, _ in markers
                                                if tokens != 8}),
            "unexpected_modes": sorted({mode for _, _, mode in markers
                                        if mode != "hybrid"})}


def summarize_static_gdn_pair(batch, stock, hooked):
    """A speedup above one means the hooked SGLang request was faster."""
    result = summarize_pair(
        batch, hooked, stock, left_name="sglang_hooked", right_name="sglang",
        ratio_key="raw_latency_ratio_stock_over_hooked",
        speedup_key="sglang_hooked_vs_stock_speedup")
    stock_markers = stock.get("static_gdn_capture_markers", {})
    hooked_markers = hooked.get("static_gdn_capture_markers", {})
    hook_active = (stock_markers.get("marker_count") == 0 and
                   hooked_markers.get("unique_b8_layer_count") == EXPECTED_GDN_LAYERS and
                   not hooked_markers.get("unexpected_token_counts") and batch == 8)
    hook_active = hook_active and not hooked_markers.get("unexpected_modes")
    result["hook_capture_proof"] = {"passed": hook_active,
                                    "stock": stock_markers, "hooked": hooked_markers,
                                    "expected_unique_b8_layers": EXPECTED_GDN_LAYERS}
    if not hook_active:
        if result["status"] == "ok":
            result["status"] = "hook_not_proven"
        result["sglang_hooked_vs_stock_speedup"] = None
    return result


def run_command(command, env, log_path):
    with log_path.open("w") as log:
        finished = subprocess.run(command, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, check=False)
    return finished.returncode


def belongs_to_worker(pid, worker_pid):
    if pid == worker_pid:
        return True
    try:
        if os.getpgid(pid) == os.getpgid(worker_pid):
            return True
    except (OSError, ProcessLookupError):
        pass
    current = pid
    for _ in range(64):
        try:
            status = Path(f"/proc/{current}/status").read_text()
        except OSError:
            return False
        match = re.search(r"^PPid:\s*(\d+)$", status, re.MULTILINE)
        if not match:
            return False
        current = int(match.group(1))
        if current == worker_pid:
            return True
        if current <= 1:
            break
    return False


class NvmlProcessSampler:
    """Poll GPU processes without launching nvidia-smi during generation."""

    def __init__(self, gpu_uuid):
        import pynvml

        self.nvml = pynvml
        self.nvml.nvmlInit()
        try:
            self.handle = self.nvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode())
            actual = self.nvml.nvmlDeviceGetUUID(self.handle)
            if isinstance(actual, bytes):
                actual = actual.decode()
            if actual.lower() != gpu_uuid.lower():
                raise RuntimeError(f"NVML selected {actual}, expected {gpu_uuid}")
        except BaseException:
            self.nvml.nvmlShutdown()
            raise

    def sample(self, worker_pid):
        started = time.time()
        compute = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        graphics = self.nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)
        pids = {int(process.pid) for process in (*compute, *graphics)}
        external = sorted(pid for pid in pids if not belongs_to_worker(pid, worker_pid))
        ended = time.time()
        return {"timestamp_unix": ended, "sample_started_unix": started,
                "sample_ended_unix": ended, "monitor_backend": "nvml",
                "gpu_process_pids": sorted(pids), "external_pids": external}

    def close(self):
        self.nvml.nvmlShutdown()


def run_monitored_command(command, env, log_path, monitor_path, gpu_uuid):
    samples = []
    try:
        sampler = NvmlProcessSampler(gpu_uuid)
        sampler_error = None
    except Exception as exc:
        sampler = None
        sampler_error = f"NVML initialization failed: {type(exc).__name__}: {exc}"
    with log_path.open("w") as log, monitor_path.open("w") as monitor:
        try:
            process = subprocess.Popen(command, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            if sampler is None:
                now = time.time()
                sample = {"timestamp_unix": now, "sample_started_unix": now,
                          "sample_ended_unix": now, "monitor_backend": "nvml",
                          "sample_error": sampler_error}
                samples.append(sample)
                monitor.write(json.dumps(sample, sort_keys=True) + "\n")
                process.wait()
            else:
                while True:
                    cycle = time.monotonic()
                    try:
                        sample = sampler.sample(process.pid)
                    except Exception as exc:
                        now = time.time()
                        sample = {"timestamp_unix": now, "sample_started_unix": now,
                                  "sample_ended_unix": now, "monitor_backend": "nvml",
                                  "sample_error": f"{type(exc).__name__}: {exc}"}
                    samples.append(sample)
                    monitor.write(json.dumps(sample, sort_keys=True) + "\n")
                    monitor.flush()
                    if process.poll() is not None:
                        break
                    time.sleep(max(0, MONITOR_INTERVAL_SECONDS - (time.monotonic() - cycle)))
            exit_code = process.wait()
        finally:
            if sampler is not None:
                sampler.close()
    return exit_code, samples


def monitor_summary(samples, report, monitor_path):
    start = report.get("timing_started_unix")
    end = report.get("timing_ended_unix")
    valid_window = isinstance(start, (int, float)) and isinstance(end, (int, float)) \
        and end > start
    successful = [sample for sample in samples if "sample_error" not in sample]
    before = (max((sample for sample in successful
                   if sample["sample_ended_unix"] <= start),
                  key=lambda sample: sample["sample_ended_unix"], default=None)
              if valid_window else None)
    after = (min((sample for sample in successful
                  if sample["sample_started_unix"] >= end),
                 key=lambda sample: sample["sample_started_unix"], default=None)
             if valid_window else None)
    inside = ([sample for sample in successful
               if sample["sample_started_unix"] >= start and
               sample["sample_ended_unix"] <= end]
              if valid_window else [])
    bracketed = before is not None and after is not None
    coverage_samples = [before, *inside, after] if bracketed else []
    monitored = ([sample for sample in samples
                  if sample["sample_started_unix"] <= after["sample_ended_unix"] and
                  sample["sample_ended_unix"] >= before["sample_started_unix"]]
                 if bracketed else
                 [sample for sample in samples if valid_window and
                  sample["sample_started_unix"] <= end and
                  sample["sample_ended_unix"] >= start])
    gaps = ([next_sample["sample_started_unix"] - prior["sample_ended_unix"]
             for prior, next_sample in zip(coverage_samples, coverage_samples[1:])]
            if bracketed else [])
    errors_in_window = [sample for sample in monitored if "sample_error" in sample]
    max_gap = max(gaps) if gaps else None
    durations = [sample["sample_ended_unix"] - sample["sample_started_unix"]
                 for sample in monitored]
    max_duration = max(durations, default=None)
    covered = bool(inside) and bracketed and not errors_in_window and \
        max_gap is not None and max_gap <= MAX_MONITOR_GAP_SECONDS and \
        max_duration is not None and max_duration <= MAX_MONITOR_GAP_SECONDS and \
        all(duration >= 0 for duration in durations)
    return {"log": monitor_path.name, "sample_count": len(samples),
            "monitor_backend": "nvml", "timed_window_samples": len(inside),
            "timed_window_covered": covered,
            "sampling_interval_seconds": MONITOR_INTERVAL_SECONDS,
            "max_allowed_gap_seconds": MAX_MONITOR_GAP_SECONDS,
            "max_sample_gap_seconds": max_gap,
            "max_sample_duration_seconds": max_duration,
            "bracket_before_seconds": (start - before["sample_ended_unix"]
                                       if before is not None else None),
            "bracket_after_seconds": (after["sample_started_unix"] - end
                                      if after is not None else None),
            "sample_errors": [sample["sample_error"] for sample in samples
                              if "sample_error" in sample],
            "external_pids_during_process": sorted(set().union(*(
                set(sample.get("external_pids", [])) for sample in samples))),
            "external_pids_during_timing": sorted(set().union(*(
                set(sample.get("external_pids", [])) for sample in monitored)))}


def child_env(backend, gpu_uuid, sglang_cuda_home=None):
    env = os.environ.copy()
    extra_static_pythonpath = env.get("QWEN38_STATIC_EXTRA_PYTHONPATH")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    env["QWEN38_BENCH_GPU_UUID"] = gpu_uuid
    if backend in ("sglang", "sglang_hooked"):
        # Stock must remain stock even if the parent shell has unrelated
        # static-layer hooks enabled. The hooked variant installs only ours.
        for key in list(env):
            if key.startswith(("DSA_STATIC_", "KDA_STATIC_", "MOE_STATIC_",
                               "STATIC_L2_", "MIRAGE_STATIC_", "QWEN38_STATIC_")):
                env.pop(key)
            if key.startswith("MPK_QWEN38_B8_"):
                env.pop(key)
        env.pop("PYTHONPATH", None)
        if backend == "sglang_hooked":
            env["PYTHONPATH"] = str(STATIC_GDN_BOOT) + (
                os.pathsep + extra_static_pythonpath if extra_static_pythonpath else ""
            )
            env["QWEN38_STATIC_GDN_LAYERS"] = "all"
            env["QWEN38_STATIC_TOKENS"] = "8"
            env["QWEN38_STATIC_GDN_MODE"] = "hybrid"
        if sglang_cuda_home is not None:
            nvcc = sglang_cuda_home / "bin" / "nvcc"
            env["CUDA_HOME"] = str(sglang_cuda_home)
            env["CUDA_PATH"] = str(sglang_cuda_home)
            env["CUDACXX"] = str(nvcc)
            env["PATH"] = str(nvcc.parent) + os.pathsep + env.get("PATH", "")
    else:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(REPO / "python") + (os.pathsep + existing if existing else "")
    return env


def run_backend(args, backend, batch, gpu_uuid, inputs):
    python = args.mpk_python if backend == "mpk" else args.sglang_python
    path = args.output_dir / f"b{batch}-{backend}.json"
    command = [str(python), str(WORKER), "--backend", backend,
               "--model", str(args.model), "--inputs", str(inputs),
               "--batch-size", str(batch), "--output", str(path),
               "--sglang-mem-fraction", str(args.sglang_mem_fraction),
               "--warmups", str(args.warmups), "--repeats", str(args.repeats)]
    if args.checkpoint_manifest:
        command.extend(["--checkpoint-manifest", str(args.checkpoint_manifest)])
    if backend == "mpk" and args.compiled_root:
        command.extend(["--compiled-root", str(args.compiled_root)])
    if backend == "mpk" and args.mpk_compiled_dir:
        command.extend(["--compiled-dir", str(args.mpk_compiled_dir)])
    if backend == "mpk" and args.allow_other_b200_compiled:
        command.append("--allow-other-b200-compiled")
    if args.stream_timing and backend in ("sglang", "sglang_hooked"):
        command.append("--stream-timing")
    if args.cuda_trace and backend in ("sglang", "sglang_hooked"):
        command.append("--cuda-trace")
        prefix = args.output_dir / f"nsys-b{batch}-{backend}"
        command = [args.nsys, "profile", "--trace=cuda,nvtx",
                   "--capture-range=cudaProfilerApi",
                   "--capture-range-end=stop",
                   "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
                   "--sample=none",
                   "--cpuctxsw=none", "--output", str(prefix)] + command
    print(f"B{batch} {backend}: {args.repeats} timed generation(s) on {gpu_uuid}",
          flush=True)
    monitor_path = args.output_dir / f"b{batch}-{backend}-gpu-monitor.jsonl"
    env = child_env(backend, gpu_uuid, args.sglang_cuda_home)
    nsys_tmp = None
    if args.cuda_trace and backend in ("sglang", "sglang_hooked"):
        # SGLang also uses TMPDIR for IPC sockets, whose Unix path is limited
        # to 107 bytes. Keep Nsight's writable scratch directory short.
        nsys_tmp = Path(tempfile.mkdtemp(prefix="q38-nsys-"))
        env["TMPDIR"] = str(nsys_tmp)
    if backend in ("sglang", "sglang_hooked"):
        # SGLang JIT invokes the ninja executable from its own environment.
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    log_path = args.output_dir / f"b{batch}-{backend}.log"
    try:
        exit_code, samples = run_monitored_command(
            command, env, log_path, monitor_path, gpu_uuid)
    finally:
        if nsys_tmp is not None:
            shutil.rmtree(nsys_tmp, ignore_errors=True)
    if not path.exists():
        return {"status": "error", "error": f"Worker exited {exit_code} without a report",
                "gpu_monitor": monitor_summary(samples, {}, monitor_path)}
    report = json.loads(path.read_text())
    report["gpu_monitor"] = monitor_summary(samples, report, monitor_path)
    if args.sglang_static_gdn and backend in ("sglang", "sglang_hooked"):
        report["static_gdn_capture_markers"] = capture_markers(log_path)
    if args.cuda_trace and backend in ("sglang", "sglang_hooked"):
        report["nsys_reports"] = sorted(
            path.name for path in args.output_dir.glob(f"nsys-b{batch}-{backend}*.nsys-rep"))
    if exit_code != 0:
        report["worker_exit_code"] = exit_code
    save_summary(path, report)
    return report


def save_summary(path, summary):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def completion_status(cases):
    return ("completed" if all(case["status"] == "ok" for case in cases)
            else "completed_with_invalid_case")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mpk-python", type=Path,
                        help="Required for MPK comparison; optional for static-GDN comparison")
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="auto", help="B200 UUID/index or auto-select an idle B200")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        help="Defaults to 1 4 16 for MPK, or 8 for static GDN; MPK also supports 8")
    parser.add_argument("--sglang-static-gdn", action="store_true",
                        help="Compare stock and static-GDN SGLang once each at B8")
    parser.add_argument("--stream-timing", action="store_true",
                        help="Record client-observed first-token and decode-tail times for the SGLang B8 pair")
    parser.add_argument("--cuda-trace", action="store_true",
                        help="Capture B8 prefill and decode CUDA Graph nodes with Nsight Systems")
    parser.add_argument("--nsys", default="nsys", help="Nsight Systems executable for --cuda-trace")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--compiled-root", type=Path,
                        help="Validated MPK compile runs b1/, b4/, b8/, b16/; otherwise compile fresh")
    parser.add_argument("--mpk-compiled-dir", type=Path,
                        help="Experimental precompiled B8 MPK graph directory")
    parser.add_argument("--allow-other-b200-compiled", action="store_true")
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.6)
    parser.add_argument("--sglang-cuda-home", type=Path,
                        help="CUDA toolkit for the SGLang child only, e.g. /usr/local/cuda-13.0")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats positive")
    if (args.sglang_static_gdn or args.stream_timing or args.cuda_trace) and \
            (args.warmups != 0 or args.repeats != 1):
        parser.error("Repeated trials support only synchronous MPK versus stock SGLang")
    if args.compiled_root and args.mpk_compiled_dir:
        parser.error("Use either --compiled-root or --mpk-compiled-dir")
    if args.mpk_compiled_dir and args.batch_sizes not in (None, [8]):
        parser.error("--mpk-compiled-dir currently supports B8 only")
    if args.stream_timing and not args.sglang_static_gdn:
        parser.error("--stream-timing requires --sglang-static-gdn")
    if args.cuda_trace and not args.sglang_static_gdn:
        parser.error("--cuda-trace requires --sglang-static-gdn")
    if args.cuda_trace and args.stream_timing:
        parser.error("Choose either --cuda-trace or --stream-timing for one comparison")
    if args.cuda_trace and shutil.which(args.nsys) is None:
        parser.error(f"Nsight Systems executable not found: {args.nsys}")
    if args.sglang_static_gdn:
        args.batch_sizes = [8] if args.batch_sizes is None else args.batch_sizes
        if args.batch_sizes != [8]:
            parser.error("--sglang-static-gdn requires --batch-sizes 8")
        if not (STATIC_GDN_BOOT / "sitecustomize.py").is_file():
            parser.error(f"Static GDN hook bootstrap is missing: {STATIC_GDN_BOOT / 'sitecustomize.py'}")
    else:
        args.batch_sizes = [1, 4, 16] if args.batch_sizes is None else args.batch_sizes
        if args.mpk_python is None:
            parser.error("--mpk-python is required for MPK comparison")
        if not args.batch_sizes or len(args.batch_sizes) != len(set(args.batch_sizes)) or \
                any(batch not in (1, 4, 8, 16) for batch in args.batch_sizes):
            parser.error("--batch-sizes must be a unique subset of 1 4 8 16")
    if args.prompt_tokens < 2 or args.new_tokens < 2 or \
            args.prompt_tokens + args.new_tokens - 1 > 1024:
        parser.error("Prompt plus generation must fit MPK's 1024-token capacity")
    args.model = args.model.resolve(strict=True)
    if args.sglang_cuda_home is not None:
        args.sglang_cuda_home = args.sglang_cuda_home.resolve(strict=True)
        nvcc = args.sglang_cuda_home / "bin" / "nvcc"
        if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
            parser.error(f"SGLang CUDA toolkit has no executable nvcc: {nvcc}")
    prepare_python = args.mpk_python or args.sglang_python
    probe_backend = "sglang" if args.sglang_static_gdn else "mpk"
    backends = (("sglang", "sglang_hooked") if args.sglang_static_gdn
                else ("sglang", "mpk"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "model_path": str(args.model),
               "comparison": ("sglang_static_gdn" if args.sglang_static_gdn
                              else "mpk_sglang"),
               "static_gdn_mode": "hybrid" if args.sglang_static_gdn else None,
               "batch_sizes": args.batch_sizes, "prompt_tokens": args.prompt_tokens,
               "new_tokens": args.new_tokens,
               "timed_generations_per_backend_batch": args.repeats,
               "warmup_generations": args.warmups,
               "stream_timing": args.stream_timing,
               "cuda_trace": args.cuda_trace,
               "measurement": ("streamed request wall time and client-observed token timeline; "
                               "first token includes prefill, scheduling and sampling; excludes model startup"
                               if args.stream_timing else
                               "Nsight node-level trace of one generation; timings are perturbed and not a speed benchmark"
                               if args.cuda_trace else
                               "full synchronous request wall time; includes prefill and decode, excludes model startup"),
               "caveat": ("Exploratory single-run measurement"
                          if args.warmups == 0 or args.repeats < 3 else
                          "Repeated warmed requests; compare outputs and isolation before claiming speedup"),
               "sglang_cuda_home": str(args.sglang_cuda_home) if args.sglang_cuda_home else None,
               "gpu_probes": [],
               "cases": []}
    summary_path = args.output_dir / "summary.json"
    save_summary(summary_path, summary)
    explicit_gpu = None
    if args.gpu != "auto":
        try:
            explicit_gpu = select_usable_gpu(
                gpu_inventory(), args.gpu,
                lambda gpu: probe_cuda(prepare_python, gpu, args.output_dir,
                                       "startup", probe_backend, args.sglang_cuda_home),
                summary["gpu_probes"])
            summary["explicit_gpu_uuid"] = explicit_gpu["uuid"]
            save_summary(summary_path, summary)
        except Exception as exc:
            summary.update(status="gpu_probe_failed", error=f"{type(exc).__name__}: {exc}")
            save_summary(summary_path, summary)
            return 1
    inputs = args.output_dir / "input_ids.json"
    prepare_command = [str(prepare_python), str(WORKER), "--backend", "prepare",
                       "--model", str(args.model), "--output", str(inputs),
                       "--prompt-tokens", str(args.prompt_tokens),
                       "--new-tokens", str(args.new_tokens), "--batch-sizes",
                       *[str(batch) for batch in args.batch_sizes]]
    prepare_env = child_env(probe_backend, "")
    prepare_env["CUDA_VISIBLE_DEVICES"] = ""
    if run_command(prepare_command, prepare_env, args.output_dir / "prepare.log") != 0:
        summary.update(status="prepare_error", error="See prepare.log")
        save_summary(summary_path, summary)
        return 1
    try:
        for batch in args.batch_sizes:
            if explicit_gpu is None:
                gpu = select_usable_gpu(
                    gpu_inventory(), "auto",
                    lambda candidate: probe_cuda(
                        prepare_python, candidate, args.output_dir, f"b{batch}",
                        probe_backend, args.sglang_cuda_home),
                    summary["gpu_probes"])
            else:
                gpu = choose_gpu(gpu_inventory(), explicit_gpu["uuid"])
            save_summary(summary_path, summary)
            reports = {}
            for backend in backends:
                # Check admission again before the second backend; do not run
                # on a GPU that another user made busy in the meantime.
                choose_gpu(gpu_inventory(), gpu["uuid"])
                reports[backend] = run_backend(args, backend, batch, gpu["uuid"], inputs)
            case = (summarize_static_gdn_pair(batch, reports["sglang"],
                                              reports["sglang_hooked"])
                    if args.sglang_static_gdn else
                    summarize_pair(batch, reports["mpk"], reports["sglang"]))
            if args.cuda_trace:
                # Node-level CUPTI tracing changes execution overlap and latency.
                case["profiled_not_benchmark"] = True
                case["sglang_hooked_vs_stock_speedup"] = None
            case["gpu_uuid"] = gpu["uuid"]
            case["artifacts"] = {
                backend: {"report": f"b{batch}-{backend}.json",
                          "log": f"b{batch}-{backend}.log",
                          "gpu_monitor": f"b{batch}-{backend}-gpu-monitor.jsonl"}
                for backend in backends}
            summary["cases"].append(case)
            save_summary(summary_path, summary)
            speedup_key = ("sglang_hooked_vs_stock_speedup" if args.sglang_static_gdn
                           else "mpk_vs_sglang_speedup")
            print(f"B{batch}: {case['status']}; {speedup_key} "
                  f"{case.get(speedup_key)}", flush=True)
        summary["status"] = completion_status(summary["cases"])
        save_summary(summary_path, summary)
        return 0 if summary["status"] == "completed" else 2
    except Exception as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        save_summary(summary_path, summary)
        raise


if __name__ == "__main__":
    sys.exit(main())
