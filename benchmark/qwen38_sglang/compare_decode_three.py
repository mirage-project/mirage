"""Fixed-batch decode comparison for Qwen3.8-27B on one B200.

Run each backend in a separate environment/process with the same prepared
``worker.py --backend prepare`` input file. Timings cover the 63 token
intervals after each request's first generated token; prefill, model load,
graph compile, and engine startup are excluded. Streaming engines include
host response delivery, so these are exploratory wall times.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_mismatch(left, right):
    for row, (a, b) in enumerate(zip(left, right)):
        for step, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return {"row": row, "step": step, "left": x, "right": y}
    return None


def streamed_record(ids, first, last, t0, timing_scope):
    if any(len(row) != 64 for row in ids) or any(t is None for t in first):
        raise RuntimeError("Missing tokens or a coalesced first streaming event")
    return {
        "decode_seconds": max(last) - max(first),
        "first_token_spread_ms": 1000 * (max(first) - min(first)),
        "timing_scope": timing_scope,
        "output_ids": ids,
    }


def run_mpk(args, rows):
    import torch
    from mirage.mpk.models.qwen3_5.model import Qwen35Model

    batch = len(rows)
    model = Qwen35Model.from_pretrained(args.model, weight_policy="hf_bf16",
                                        attn_implementation="eager", batch_size=batch)
    try:
        if args.compiled:
            model.pk.load_mpk_kernel(str(args.compiled))
            model._compiled = True
        else:
            model.compile(args.output.parent / f"compiled-b{batch}")
        torch.cuda.synchronize()
        records = []
        for trial in range(-args.warmups, args.repeats):
            model.reset()
            model.prefill(rows, mode="block")
            torch.cuda.synchronize()
            ids = [[] for _ in rows]
            first = model.tensors["next_token"].tolist()
            for row, token in zip(ids, first):
                row.append(int(token))
            start = time.perf_counter()
            for _ in range(63):
                model.step([row[-1] for row in ids])
                nxt = model.tensors["next_token"].tolist()
                for row, token in zip(ids, nxt):
                    row.append(int(token))
            torch.cuda.synchronize()
            record = {"decode_seconds": time.perf_counter() - start,
                      "timing_scope": "host-observed fixed-batch MPK steps",
                      "output_ids": ids}
            if trial >= 0:
                records.append(record)
        return records, {"torch": torch.__version__, "cuda": torch.version.cuda}
    finally:
        model.close()


def run_hf(args, rows):
    """Time native eager BF16 forwards after an untimed cache-building prefill."""
    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, device_map="cuda").eval()
    try:
        batch = len(rows)
        prompt = torch.tensor(rows, dtype=torch.long, device="cuda")
        attention = torch.ones((batch, len(rows[0]) + 64), dtype=torch.long, device="cuda")
        records = []
        with torch.inference_mode():
            for trial in range(-args.warmups, args.repeats):
                output = model(input_ids=prompt, attention_mask=attention[:, :prompt.shape[1]],
                               use_cache=True)
                cache = output.past_key_values
                token = output.logits[:, -1].argmax(dim=-1)
                ids = [[int(value)] for value in token.tolist()]
                torch.cuda.synchronize()
                start = time.perf_counter()
                for step in range(63):
                    output = model(input_ids=token[:, None],
                                   attention_mask=attention[:, :prompt.shape[1] + step + 1],
                                   past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    token = output.logits[:, -1].argmax(dim=-1)
                    for row, value in zip(ids, token.tolist()):
                        row.append(int(value))
                torch.cuda.synchronize()
                record = {"decode_seconds": time.perf_counter() - start,
                          "timing_scope": "host-observed eager HF cached forwards and token reads",
                          "output_ids": ids}
                if trial >= 0:
                    records.append(record)
        return records, {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "transformers": transformers.__version__, "attention": "eager"}
    finally:
        del model
        torch.cuda.empty_cache()


def run_sglang(args, rows):
    import sglang as sgl
    from worker import BatchStreamTimeline

    batch = len(rows)
    settings = dict(model_path=str(args.model), tp_size=1, dtype="bfloat16",
                    kv_cache_dtype="bfloat16", max_running_requests=batch,
                    cuda_graph_max_bs_decode=batch, context_length=2048,
                    mem_fraction_static=args.memory_fraction,
                    disable_radix_cache=True, skip_tokenizer_init=True,
                    stream_interval=1, log_level="warning")
    engine = sgl.Engine(**settings)
    try:
        records = []
        sampling = {"max_new_tokens": 64, "ignore_eos": True, "temperature": 0.0}
        for trial in range(-args.warmups, args.repeats):
            start = time.perf_counter()
            timeline = BatchStreamTimeline(start, batch, 64)
            for chunk in engine.generate(input_ids=rows, sampling_params=sampling, stream=True):
                timeline.update(chunk, time.perf_counter())
            timing = timeline.finish(time.perf_counter())
            if not timing["valid"]:
                raise RuntimeError(f"SGLang stream timing invalid: {timing['invalid_reason']}")
            record = {"decode_seconds": timing["decode_tail_seconds"],
                      "first_token_spread_ms": 1000 * (
                          max(x["first_token_seconds"] for x in timing["per_row"]) -
                          min(x["first_token_seconds"] for x in timing["per_row"])),
                      "timing_scope": timing["timing_scope"],
                      "output_ids": [row["output_ids"] for row in timeline.rows]}
            if trial >= 0:
                records.append(record)
        return records, {"sglang": importlib.metadata.version("sglang"),
                         "engine_settings": {k: v for k, v in settings.items()
                                             if k != "model_path"}}
    finally:
        engine.shutdown()


async def vllm_trial(engine, rows, sampling, trial):
    start = time.perf_counter()
    ids = [[] for _ in rows]
    first = [None] * len(rows)
    last = [None] * len(rows)

    async def one(index, prompt_ids):
        async for output in engine.generate(
                request_id=f"decode-{trial}-{index}",
                prompt={"prompt_token_ids": prompt_ids}, sampling_params=sampling):
            now = time.perf_counter()
            for completion in output.outputs:
                delta = list(completion.token_ids)
                if not delta:
                    continue
                ids[index].extend(delta)
                if first[index] is None:
                    if len(delta) != 1:
                        raise RuntimeError("vLLM coalesced its first token event")
                    first[index] = now - start
                last[index] = now - start

    await asyncio.gather(*(one(index, row) for index, row in enumerate(rows)))
    return streamed_record(ids, first, last, start,
                           "client-visible vLLM streamed token arrivals")


async def run_vllm_async(args, rows):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    settings = dict(model=str(args.model), tensor_parallel_size=1,
                    dtype="bfloat16", kv_cache_dtype="bfloat16",
                    gpu_memory_utilization=args.memory_fraction,
                    max_model_len=2048, max_num_seqs=len(rows),
                    enable_prefix_caching=False,
                    trust_remote_code=False)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**settings))
    try:
        sampling = SamplingParams(max_tokens=64, temperature=0.0,
                                  ignore_eos=True,
                                  output_kind=RequestOutputKind.DELTA)
        records = []
        for trial in range(-args.warmups, args.repeats):
            record = await vllm_trial(engine, rows, sampling, trial)
            if trial >= 0:
                records.append(record)
        return records, {"vllm": importlib.metadata.version("vllm"),
                         "engine_settings": {k: v for k, v in settings.items()
                                             if k != "model"}}
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mpk", "hf", "sglang", "vllm"), required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be >= 0 and repeats >= 1")
    inputs = json.loads(args.inputs.read_text())
    if args.batch not in (4, 8, 16, 32):
        parser.error("--batch must be 4, 8, 16, or 32")
    rows = inputs["prompt_ids"].get(str(args.batch))
    if (inputs["prompt_tokens"], inputs["new_tokens"]) != (128, 64) or \
            rows is None or len(rows) != args.batch or any(len(row) != 128 for row in rows):
        parser.error("expected the selected batch with 128 input and 64 output tokens")
    if inputs["config_sha256"] != digest(args.model / "config.json"):
        parser.error("input file belongs to another checkpoint")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        parser.error("expose exactly one GPU using CUDA_VISIBLE_DEVICES")
    if args.backend == "mpk":
        records, environment = run_mpk(args, rows)
    elif args.backend == "hf":
        records, environment = run_hf(args, rows)
    elif args.backend == "sglang":
        records, environment = run_sglang(args, rows)
    else:
        records, environment = asyncio.run(run_vllm_async(args, rows))
    repeat_mismatches = [first_mismatch(record["output_ids"],
                                        records[0]["output_ids"])
                         for record in records[1:]]
    result = {"backend": args.backend, "model_config_sha256": digest(args.model / "config.json"),
              "inputs_sha256": digest(args.inputs), "batch": args.batch, "prompt_tokens": 128,
              "output_tokens": 64, "decode_steps": 63,
              "gpu_visible": os.environ["CUDA_VISIBLE_DEVICES"],
              "median_decode_seconds": statistics.median(r["decode_seconds"] for r in records),
              "stable_outputs": not any(repeat_mismatches),
              "repeat_mismatches": repeat_mismatches,
              "records": records, "environment": environment}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
