"""One Qwen3.8-27B generation in one backend process on one visible B200.

The ``prepare`` mode is CPU-only. Import GPU libraries only in the selected
backend so MPK and stock SGLang can use their separate Python environments.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[2]
PROMPT_BODY = (
    "A research team is comparing language model inference on one GPU. "
    "They record latency, throughput, memory usage and reproducibility. "
    "Requests have separate histories and must never share their cache. "
    "Explain the tradeoffs clearly and give a practical example. "
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_value(*args):
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(model, backend, manifest):
    model = Path(model)
    return {
        "backend": backend,
        "model_path": str(model.resolve()),
        "config_sha256": sha256(model / "config.json"),
        "checkpoint_manifest_sha256": sha256(manifest) if manifest else None,
        "repo_commit": git_value("rev-parse", "HEAD"),
        "repo_status": git_value("status", "--porcelain"),
        "worker_sha256": sha256(__file__),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_gpu_uuid": os.environ.get("QWEN38_BENCH_GPU_UUID"),
    }


def prompt_rows(tokenizer, length, batch):
    """Same distinct, rectangular token rows as the existing HF/MPK suite."""
    tail = tokenizer(PROMPT_BODY, add_special_tokens=False)["input_ids"]
    if not tail:
        raise ValueError("Tokenizer returned an empty prompt body")
    rows = []
    for row in range(batch):
        head = tokenizer(f"Benchmark request {row + 1}. ",
                         add_special_tokens=False)["input_ids"]
        rows.append((head + tail * math.ceil(length / len(tail)))[:length])
    if any(len(row) != length for row in rows) or len({tuple(row) for row in rows}) != batch:
        raise ValueError("Prompt rows must have the requested length and distinct IDs")
    return rows


def prepare(args):
    from transformers import AutoTokenizer

    config = json.loads((args.model / "config.json").read_text())
    text = config.get("text_config", config)
    if (text.get("num_hidden_layers"), text.get("hidden_size"),
            text.get("intermediate_size"), text.get("vocab_size")) != \
            (64, 5120, 17408, 248320):
        raise ValueError("Expected the Qwen3.8-27B text checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = {str(batch): prompt_rows(tokenizer, args.prompt_tokens, batch)
             for batch in args.batch_sizes}
    write_json(args.output, {
        "model_path": str(args.model.resolve()),
        "config_sha256": sha256(args.model / "config.json"),
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "batch_sizes": args.batch_sizes,
        "prompt_ids": cases,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_json_sha256": sha256(args.model / "tokenizer.json")
        if (args.model / "tokenizer.json").exists() else None,
    })


def gpu_info():
    uuid = os.environ.get("QWEN38_BENCH_GPU_UUID")
    if not uuid:
        raise ValueError("Set QWEN38_BENCH_GPU_UUID to the single selected B200")
    if "," in os.environ.get("CUDA_VISIBLE_DEVICES", "") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Expose exactly one GPU in CUDA_VISIBLE_DEVICES")
    output = subprocess.check_output([
        "nvidia-smi", "-i", uuid,
        "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()
    fields = [field.strip() for field in output.split(",")]
    if len(fields) != 5 or fields[0].lower() != uuid.lower() or "B200" not in fields[1]:
        raise RuntimeError(f"Selected GPU is not the expected B200: {output}")
    return dict(zip(("uuid", "name", "driver", "memory_used_mib", "utilization_percent"), fields))


def validate_inputs(args):
    inputs = json.loads(args.inputs.read_text())
    batch = args.batch_size
    rows = inputs["prompt_ids"][str(batch)]
    length = inputs["prompt_tokens"]
    tokens = inputs["new_tokens"]
    if batch not in inputs["batch_sizes"] or len(rows) != batch or \
            any(len(row) != length or any(type(token) is not int or token < 0 for token in row)
                for row in rows):
        raise ValueError("Invalid shared input token IDs")
    if length < 2 or tokens < 2 or length + tokens > 1024:
        raise ValueError("MPK's 1024-token capacity would be exceeded")
    if inputs["config_sha256"] != sha256(args.model / "config.json") or \
            Path(inputs["model_path"]).resolve() != args.model.resolve():
        raise ValueError("Input set belongs to a different checkpoint")
    return inputs, rows


def run_mpk(args, rows, new_tokens):
    import torch
    from mirage.mpk.models.qwen3_5.model import Qwen35Model

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError("MPK requires exactly one visible B200")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    expected_uuid = os.environ["QWEN38_BENCH_GPU_UUID"].lower().removeprefix("gpu-")
    if actual_uuid != expected_uuid:
        raise RuntimeError("MPK selected the wrong GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    warmups, repeats = getattr(args, "warmups", 0), getattr(args, "repeats", 1)
    batch = len(rows)
    if batch == 8 and not (args.compiled_dir or args.compiled_root):
        return run_mpk_offline_b8(args, rows, new_tokens, warmups, repeats)
    model = Qwen35Model.from_pretrained(args.model, weight_policy="hf_bf16",
                                        attn_implementation="eager", batch_size=batch)
    try:
        if args.compiled_dir:
            if batch != 8:
                raise ValueError("--compiled-dir currently supports only the B8 experiment")
            model.pk.load_mpk_kernel(str(args.compiled_dir))
            model._compiled = True
            so_files = list(args.compiled_dir.glob("mpk_launcher_rank0*.so"))
            if len(so_files) != 1:
                raise ValueError("Expected one compiled B8 MPK launcher")
            compile_info = {"mode": "direct_reuse", "path": str(args.compiled_dir.resolve()),
                            "launcher_sha256": sha256(so_files[0])}
        elif args.compiled_root:
            sys.path.insert(0, str(REPO / "tests" / "qwen3_5"))
            from prefill_test_utils import load_compiled_decode
            compiled_run = args.compiled_root / f"b{batch}"
            load_compiled_decode(model, compiled_run,
                                 allow_other_b200=args.allow_other_b200_compiled)
            compile_info = {"mode": "reused", "path": str(compiled_run.resolve()),
                            "report_sha256": sha256(compiled_run / "report.json")}
        else:
            compiled_run = args.output.parent / f"compiled-b{batch}"
            model.compile(compiled_run)
            compile_info = {"mode": "compiled", "path": str(compiled_run.resolve())}

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        expected = None
        started_unix = ended_unix = None
        for trial in range(-warmups, repeats):
            torch.cuda.synchronize()
            trial_started = time.time()
            start = time.perf_counter()
            model.reset()
            model.prefill(rows[0] if batch == 1 else rows, mode="block")
            generated = [[] for _ in range(batch)]
            for step in range(new_tokens):
                next_ids = model.tensors["next_token"].tolist()
                for row, token in zip(generated, next_ids):
                    row.append(int(token))
                if step + 1 < new_tokens:
                    model.step(next_ids[0] if batch == 1 else next_ids)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            trial_ended = time.time()
            if expected is None:
                expected = generated
            elif generated != expected:
                raise RuntimeError(f"MPK output changed between repeated trials at {trial}")
            if trial >= 0:
                if started_unix is None:
                    started_unix = trial_started
                ended_unix = trial_ended
                samples.append(elapsed)
        elapsed = statistics.median(samples)
        peak_bytes = torch.cuda.max_memory_allocated()
        return generated, elapsed, started_unix, ended_unix, {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "peak_torch_allocated_bytes": peak_bytes,
            "compiled": compile_info, "prefill": "pytorch_block",
            "decode": "fixed_batch_mpk", "attention": "eager", "dtype": "bfloat16",
            "tf32": False, "cuda_graphs": False,
            "trial_seconds": samples, "warmup_generations": warmups,
        }
    finally:
        model.close()


def run_mpk_offline_b8(args, rows, new_tokens, warmups, repeats):
    """Use the production offline scheduler for the complete B8 model."""
    import torch
    from mirage.mpk.models.qwen3_5.builder import Qwen38Builder
    from mirage.mpk.models.qwen3_5.offline import create_offline_runtime, offline_generated_ids
    from mirage.mpk.models.qwen3_5.weight_loader import Qwen35WeightLoader

    prompt_length = len(rows[0])
    config = Qwen35WeightLoader(args.model).config
    mpk, plan, metadata = create_offline_runtime(config, rows, new_tokens, ignore_eos=True)
    try:
        builder = Qwen38Builder(mpk, kv_plan=plan)
        builder.build_from_model(model_path=args.model)
        compiled_run = args.output.parent / "compiled-offline-b8"
        mpk.compile(output_dir=str(compiled_run))
        compile_info = {"mode": "compiled_offline", "path": str(compiled_run.resolve())}

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        expected = None
        started_unix = ended_unix = None
        for trial in range(-warmups, repeats):
            metadata["step"].zero_()
            metadata["tokens"][:, prompt_length:].zero_()
            builder.kv_pool.zero_()
            for layer in builder.model.layers:
                if "state" in layer.mixer.tensors:
                    layer.mixer.tensors["state"].zero_()
                    layer.mixer.tensors["conv_state"].zero_()
            mpk.init_request_func()
            torch.cuda.synchronize()
            trial_started = time.time()
            start = time.perf_counter()
            mpk()
            torch.cuda.synchronize()
            generated = offline_generated_ids(metadata, prompt_length, new_tokens)
            elapsed = time.perf_counter() - start
            trial_ended = time.time()
            if expected is None:
                expected = generated
            elif generated != expected:
                raise RuntimeError(f"MPK offline output changed between repeated trials at {trial}")
            if trial >= 0:
                if started_unix is None:
                    started_unix = trial_started
                ended_unix = trial_ended
                samples.append(elapsed)
        return expected, statistics.median(samples), started_unix, ended_unix, {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "compiled": compile_info, "prefill": "offline_mpk_token_stream",
            "decode": "offline_mpk_one_call", "attention": "eager", "dtype": "bfloat16",
            "tf32": False, "cuda_graphs": False,
            "trial_seconds": samples, "warmup_generations": warmups,
        }
    finally:
        mpk.finalize()


def run_hf(args, rows, new_tokens):
    """Time native Transformers generation with the benchmark's prepared IDs."""
    import copy
    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError("HF requires exactly one visible B200")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    expected_uuid = os.environ["QWEN38_BENCH_GPU_UUID"].lower().removeprefix("gpu-")
    if actual_uuid != expected_uuid:
        raise RuntimeError("HF selected the wrong GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.hf_attention,
        local_files_only=True, device_map="cuda").eval()
    try:
        inputs = torch.tensor(rows, dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(inputs)
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        generation_config.stop_strings = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started_unix = time.time()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs, attention_mask=attention_mask,
                generation_config=generation_config, do_sample=False,
                max_new_tokens=new_tokens, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ended_unix = time.time()
        generated = output[:, inputs.shape[1]:].tolist()
        return generated, elapsed, started_unix, ended_unix, {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "attention": args.hf_attention,
            "dtype": "bfloat16", "tf32": False, "use_cache": True,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    finally:
        del model
        torch.cuda.empty_cache()


class BatchStreamTimeline:
    """Track client-visible B8 token arrivals without assuming stream order."""

    def __init__(self, start, batch_size, new_tokens):
        self.start = start
        self.new_tokens = new_tokens
        self.rows = [dict(output_ids=[], request_id=None, first_observation_tokens=None,
                          first_token_seconds=None, last_token_seconds=None)
                     for _ in range(batch_size)]
        self.request_ids = {}
        self.meta_info = [None] * batch_size

    def update(self, chunk, now):
        index = chunk.get("index")
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError(f"Invalid B8 streaming response index: {index!r}")
        meta = chunk.get("meta_info")
        if not isinstance(meta, dict):
            raise ValueError("B8 streaming response has no meta_info")
        request_id = meta.get("id")
        count = meta.get("completion_tokens")
        ids = chunk.get("output_ids")
        if not isinstance(request_id, str) or not request_id or \
                type(count) is not int or not 0 <= count <= self.new_tokens or \
                not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError("Invalid B8 streaming request ID, token count, or output IDs")
        row = self.rows[index]
        if row["request_id"] is None:
            if request_id in self.request_ids:
                raise ValueError("B8 streaming request ID appeared in two rows")
            row["request_id"] = request_id
            self.request_ids[request_id] = index
        elif row["request_id"] != request_id:
            raise ValueError("B8 streaming response changed request ID for a row")
        previous = row["output_ids"]
        if len(ids) == count and ids[:len(previous)] == previous:
            cumulative = ids
        elif len(previous) + len(ids) == count:
            cumulative = previous + ids
        else:
            raise ValueError("B8 streaming output is neither cumulative nor incremental")
        if len(cumulative) > len(previous):
            if row["first_observation_tokens"] is None:
                row["first_observation_tokens"] = len(cumulative)
                if len(cumulative) == 1:
                    row["first_token_seconds"] = now - self.start
            row["last_token_seconds"] = now - self.start
            row["output_ids"] = list(cumulative)
        self.meta_info[index] = meta

    def finish(self, now):
        complete = all(len(row["output_ids"]) == self.new_tokens for row in self.rows)
        first_exact = all(row["first_observation_tokens"] == 1 for row in self.rows)
        valid = complete and first_exact
        reason = (None if valid else "incomplete_stream" if not complete
                  else "coalesced_first_event")
        first = (max(row["first_token_seconds"] for row in self.rows)
                 if valid else None)
        completion = (max(row["last_token_seconds"] for row in self.rows)
                      if complete else None)
        per_row = [dict(index=index, request_id=row["request_id"],
                        first_observation_tokens=row["first_observation_tokens"],
                        first_token_seconds=row["first_token_seconds"],
                        last_token_seconds=row["last_token_seconds"],
                        decode_tail_seconds=(row["last_token_seconds"] -
                                             row["first_token_seconds"])
                        if row["first_token_seconds"] is not None and complete else None)
                   for index, row in enumerate(self.rows)]
        return {"valid": valid, "invalid_reason": reason,
                "timing_scope": "client-visible streamed token arrivals; includes scheduling and IPC",
                "batch_ttft_seconds": first,
                "decode_tail_seconds": completion - first if valid else None,
                "completion_seconds": completion,
                "total_seconds": now - self.start,
                "per_row": per_row}


def run_sglang(args, rows, new_tokens):
    import sglang as sgl

    warmups, repeats = getattr(args, "warmups", 0), getattr(args, "repeats", 1)
    batch = len(rows)
    engine_settings = {
        "model_path": str(args.model), "tp_size": 1,
        "dtype": "bfloat16",
        "max_running_requests": batch, "cuda_graph_max_bs_decode": batch,
        # SGLang needs room beyond the final token when the request reaches
        # MPK's 1024-token limit (for example, 64 prompt + 960 generated).
        "context_length": 2048, "mem_fraction_static": args.sglang_mem_fraction,
        "kv_cache_dtype": "bfloat16", "disable_radix_cache": True,
        "skip_tokenizer_init": True, "log_level": "warning",
    }
    if args.stream_timing:
        engine_settings["stream_interval"] = 1
    if os.environ.get("QWEN38_DIAG_DISABLE_CUDA_GRAPH") == "1":
        engine_settings["disable_cuda_graph"] = True
    sampling = {"max_new_tokens": new_tokens, "ignore_eos": True,
                "temperature": 0.0}
    engine = sgl.Engine(**engine_settings)
    try:
        cuda_trace = getattr(args, "cuda_trace", False)
        if cuda_trace:
            engine.start_profile(activities=["CUDA_PROFILER"],
                                 profile_by_stage=False,
                                 with_stack=False, record_shapes=False,
                                 detailed_annotations=True)
        try:
            samples = []
            expected = None
            started_unix = ended_unix = None
            for trial in range(-warmups, repeats):
                # With streaming, record each row by original-input index.
                trial_started = time.time()
                start = time.perf_counter()
                if args.stream_timing:
                    timeline = BatchStreamTimeline(start, batch, new_tokens)
                    for chunk in engine.generate(input_ids=rows, sampling_params=sampling,
                                                 stream=True):
                        timeline.update(chunk, time.perf_counter())
                    finish = time.perf_counter()
                    stream_timing = timeline.finish(finish)
                    generated = [row["output_ids"] for row in timeline.rows]
                    meta_info = timeline.meta_info
                else:
                    # Synchronous return means all outputs have arrived.
                    outputs = engine.generate(input_ids=rows, sampling_params=sampling)
                    if batch == 1 and isinstance(outputs, dict):
                        outputs = [outputs]
                    if not isinstance(outputs, list) or len(outputs) != batch:
                        raise RuntimeError("SGLang returned an unexpected response count")
                    generated = [output["output_ids"] for output in outputs]
                    meta_info = [output.get("meta_info") for output in outputs]
                    finish = time.perf_counter()
                elapsed = finish - start
                trial_ended = time.time()
                if expected is None:
                    expected = generated
                elif generated != expected:
                    raise RuntimeError(f"SGLang output changed between repeated trials at {trial}")
                if trial >= 0:
                    if started_unix is None:
                        started_unix = trial_started
                    ended_unix = trial_ended
                    samples.append(elapsed)
            elapsed = statistics.median(samples)
        finally:
            if cuda_trace:
                engine.stop_profile()
        details = {
            "sglang": importlib.metadata.version("sglang"),
            "engine": {key: value for key, value in engine_settings.items()
                       if key != "model_path"},
            "sampling": sampling, "response_meta_info": meta_info,
            "trial_seconds": samples, "warmup_generations": warmups,
        }
        if args.stream_timing:
            details["stream_timing"] = stream_timing
        return generated, elapsed, started_unix, ended_unix, details
    finally:
        engine.shutdown()


def benchmark(args):
    result = provenance(args.model, args.backend, args.checkpoint_manifest)
    timing_scope = ("streamed generation through final response"
                    if args.stream_timing else "synchronous generation call")
    result.update(status="running", inputs_sha256=sha256(args.inputs),
                  batch_size=args.batch_size, timing_scope=timing_scope,
                  timed_generations=args.repeats, warmup_generations=args.warmups,
                  excluded="model loading, MPK compilation, SGLang engine startup, tokenization")
    try:
        inputs, rows = validate_inputs(args)
        new_tokens = inputs["new_tokens"]
        result["prompt_tokens"] = inputs["prompt_tokens"]
        result["new_tokens"] = new_tokens
        result["gpu_before"] = gpu_info()
        if args.backend == "mpk":
            generated, seconds, started_unix, ended_unix, details = run_mpk(
                args, rows, new_tokens)
        elif args.backend == "hf":
            generated, seconds, started_unix, ended_unix, details = run_hf(
                args, rows, new_tokens)
        else:
            generated, seconds, started_unix, ended_unix, details = run_sglang(
                args, rows, new_tokens)
        if len(generated) != args.batch_size or any(
                len(row) != new_tokens or any(type(token) is not int for token in row)
                for row in generated):
            raise RuntimeError("Backend did not produce the requested fixed token count")
        result.update(status="ok", elapsed_seconds=seconds,
                      timing_started_unix=started_unix,
                      timing_ended_unix=ended_unix,
                      output_tokens_per_second=args.batch_size * new_tokens / seconds,
                      output_ids=generated, details=details, gpu_after=gpu_info())
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("prepare", "mpk", "sglang", "sglang_hooked", "hf"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--compiled-root", type=Path)
    parser.add_argument("--compiled-dir", type=Path,
                        help="Experimental precompiled B8 graph directory")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-other-b200-compiled", action="store_true")
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.6)
    parser.add_argument("--hf-attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--stream-timing", action="store_true",
                        help="Record B8 SGLang first-token and decode-tail timings")
    parser.add_argument("--cuda-trace", action="store_true",
                        help="Capture one B8 SGLang generation with CUDA profiler")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats positive")
    if (args.stream_timing or args.cuda_trace or args.backend == "hf") and \
            (args.warmups != 0 or args.repeats != 1):
        parser.error("Repeated trials currently support synchronous MPK and stock SGLang only")
    if args.compiled_root and args.compiled_dir:
        parser.error("Use either --compiled-root or --compiled-dir")
    if args.stream_timing and (args.backend not in ("sglang", "sglang_hooked") or
                               args.batch_size != 8):
        parser.error("--stream-timing requires B8 stock or hooked SGLang")
    if args.cuda_trace and (args.backend not in ("sglang", "sglang_hooked") or
                            args.batch_size != 8):
        parser.error("--cuda-trace requires B8 stock or hooked SGLang")
    if args.backend == "prepare":
        if args.prompt_tokens < 2 or args.new_tokens < 2 or \
                args.prompt_tokens + args.new_tokens - 1 > 1024 or \
                not args.batch_sizes or any(batch not in (1, 4, 8, 16, 32) for batch in args.batch_sizes):
            parser.error("Use B1/B4/B8/B16/B32 and a sequence within 1024-token capacity")
        prepare(args)
    else:
        allowed = (1, 4, 8, 16)
        if args.batch_size not in allowed or not args.inputs:
            parser.error(f"A prepared input file and a supported {args.backend} batch are required")
        if not 0 < args.sglang_mem_fraction < 1:
            parser.error("--sglang-mem-fraction must be between 0 and 1")
        benchmark(args)


if __name__ == "__main__":
    main()
