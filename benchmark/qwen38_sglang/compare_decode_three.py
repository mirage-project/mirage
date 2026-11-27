"""Fixed-batch decode comparison for Qwen3.8-27B on one B200.

Run each backend in a separate environment/process with the same prepared
``--backend prepare`` input file. Timings cover the 63 token
intervals after each request's first generated token; prefill, model load,
graph compile, and engine startup are excluded. Streaming engines include
host response delivery, so these are exploratory wall times.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs(args):
    """Write one shared set of distinct token-ID prompts for all backends."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    body = ("A research team is comparing language model inference on one GPU. "
            "They record latency, throughput, memory usage and reproducibility. "
            "Requests have separate histories and must never share their cache. "
            "Explain the tradeoffs clearly and give a practical example. ")
    tail = tokenizer(body, add_special_tokens=False)["input_ids"]
    if not tail:
        raise ValueError("Tokenizer returned an empty prompt body")
    cases = {}
    for batch in (4, 8, 16, 32):
        rows = []
        for index in range(batch):
            head = tokenizer(f"Benchmark request {index + 1}. ",
                             add_special_tokens=False)["input_ids"]
            rows.append((head + tail * math.ceil(128 / len(tail)))[:128])
        if len({tuple(row) for row in rows}) != batch:
            raise ValueError("Prompt rows are not distinct")
        cases[str(batch)] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config_sha256": digest(args.model / "config.json"),
        "prompt_tokens": 128, "new_tokens": 64, "prompt_ids": cases,
    }, indent=2) + "\n")


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


class BatchStreamTimeline:
    """Track client-visible token arrivals without assuming stream order."""

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
            raise ValueError(f"Invalid SGLang streaming response index: {index!r}")
        meta = chunk.get("meta_info")
        if not isinstance(meta, dict):
            raise ValueError("SGLang streaming response has no meta_info")
        request_id = meta.get("id")
        count = meta.get("completion_tokens")
        ids = chunk.get("output_ids")
        if not isinstance(request_id, str) or not request_id or \
                type(count) is not int or not 0 <= count <= self.new_tokens or \
                not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError("Invalid SGLang streaming request ID, token count, or output IDs")
        row = self.rows[index]
        if row["request_id"] is None:
            if request_id in self.request_ids:
                raise ValueError("SGLang streaming request ID appeared in two rows")
            row["request_id"] = request_id
            self.request_ids[request_id] = index
        elif row["request_id"] != request_id:
            raise ValueError("SGLang streaming response changed request ID for a row")
        previous = row["output_ids"]
        if len(ids) == count and ids[:len(previous)] == previous:
            cumulative = ids
        elif len(previous) + len(ids) == count:
            cumulative = previous + ids
        else:
            raise ValueError("SGLang streaming output is neither cumulative nor incremental")
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


def run_sglang(args, rows):
    import sglang as sgl

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
    parser.add_argument("--backend", choices=("prepare", "mpk", "hf", "sglang", "vllm"), required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    args = parser.parse_args()
    if args.backend == "prepare":
        prepare_inputs(args)
        return
    if args.inputs is None:
        parser.error("--inputs is required for a timed backend")
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
