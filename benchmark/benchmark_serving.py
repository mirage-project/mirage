#!/usr/bin/env python3
"""
Benchmark a serving endpoint on the ShareGPT dataset.

Measures throughput (tokens/s), TTFT (Time To First Token), and TPOT
(Time Per Output Token) under configurable request-per-second rates.
Requests are sent at a uniform pace — the next request is dispatched
every ``1/rps`` seconds regardless of how many are still in-flight.

Usage::

    # 1. Start MPK server:
    python -m mirage.engine.launch_server \
        --model Qwen/Qwen3-8B --max-num-batched-tokens 4 \
        --max-num-batched-requests 4 --max-seq-length 1024 --port 8000 \
        --request-timeout 7200

    # 2. Benchmark it:
    python benchmark/benchmark_serving.py \
        --url http://127.0.0.1:8000 \
        --num-prompts 200 \
        --rps 1 \
        -o mpk_results.json

    # 3. Stop MPK, start vLLM, benchmark it:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B --max-num-batched-tokens 4 \
        --max-num-seqs 4 --max-model-len 1024 --port 8000
    python benchmark/benchmark_serving.py \
        --url http://127.0.0.1:8000 \
        --num-prompts 200 \
        --rps 1 \
        -o vllm_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)


# ── Data loading ───────────────────────────────────────────────────────────────


def load_sharegpt(path: Optional[str] = None) -> list[dict]:
    """Load ShareGPT dataset, returning list of prompt dicts."""
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        print("Downloading ShareGPT dataset...")
        try:
            from datasets import load_dataset
            ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                              "ShareGPT_V3_unfiltered_cleaned_split",
                              split="train")
            data = list(ds)
        except Exception:
            print("  'datasets' library unavailable, downloading with urllib...")
            import io
            req = urllib.request.Request(DEFAULT_SHAREGPT_URL)
            with urllib.request.urlopen(req) as resp:
                data = json.load(io.BytesIO(resp.read()))

    prompts = []
    skipped = 0
    for item in data:
        conversations = item.get("conversations", [])
        if not conversations:
            skipped += 1
            continue
        first = conversations[0]
        if first.get("from") != "human":
            skipped += 1
            continue
        prompt = first.get("value", "").strip()
        if not prompt:
            skipped += 1
            continue
        prompts.append({"prompt": prompt})

    if skipped:
        print(f"  Skipped {skipped} entries with no usable human prompt")
    return prompts


def _approx_token_count(text: str) -> int:
    """Rough token count estimate using the chars/4 rule of thumb for English."""
    return max(1, len(text) // 4)


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    """Per-request timing metrics."""
    index: int
    prompt: str
    ttft: float          # Time to first token (seconds)
    itl: list[float]     # Inter-token latencies (seconds)
    total_time: float    # End-to-end wall clock (seconds)
    input_tokens: int
    output_tokens: int
    success: bool = True
    error: str = ""


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results for a single RPS level."""
    url: str
    label: str
    rps: float
    num_requests: int
    num_successful: int
    total_input_tokens: int
    total_output_tokens: int
    total_time: float
    throughput_tokens_per_sec: float
    actual_rps: float
    ttft_mean: float
    ttft_median: float
    ttft_p95: float
    ttft_p99: float
    tpot_mean: float
    tpot_median: float
    tpot_p95: float
    tpot_p99: float
    e2e_latency_mean: float
    e2e_latency_median: float
    e2e_latency_p95: float
    e2e_latency_p99: float
    results: list[RequestResult] = field(default_factory=list, repr=False)


# ── HTTP client (OpenAI-compatible streaming) ───────────────────────────────────


async def _stream_request(
    session,
    url: str,
    prompt: str,
    index: int,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> RequestResult:
    """Send one streaming chat-completion request, measure per-token timing."""
    import aiohttp

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": True,
    }

    result = RequestResult(
        index=index, prompt=prompt, ttft=0.0,
        itl=[], total_time=0.0, input_tokens=0, output_tokens=0,
    )

    t_send = time.monotonic()
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    token_count = 0

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=timeout_obj,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.success = False
                result.error = f"HTTP {resp.status}: {body[:200]}"
                return result

            buffer = ""
            async for chunk in resp.content.iter_chunked(512):
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = obj.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    now = time.monotonic()
                    if content:
                        if first_token_ts is None:
                            first_token_ts = now
                            usage = obj.get("usage", {})
                            result.input_tokens = usage.get("prompt_tokens", 0)
                        else:
                            result.itl.append(now - last_token_ts)
                        last_token_ts = now
                        token_count += 1

        if result.input_tokens == 0:
            result.input_tokens = len(prompt.split())

        result.output_tokens = token_count
        result.total_time = time.monotonic() - t_send
        if first_token_ts is not None:
            result.ttft = first_token_ts - t_send
        result.itl = result.itl[1:] if len(result.itl) > 1 else result.itl

    except asyncio.TimeoutError:
        result.success = False
        result.error = "timeout"
    except Exception as e:
        result.success = False
        result.error = str(e)[:200]

    return result


# ── Benchmark runner ───────────────────────────────────────────────────────────


def _compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of floats."""
    if not values:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _percentile(p: float) -> float:
        if n == 0:
            return 0.0
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    return {
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "p95": _percentile(95),
        "p99": _percentile(99),
    }


async def _benchmark(
    url: str,
    label: str,
    prompts: list[dict],
    rps: float,
    temperature: float = 0.0,
    timeout: float = 300.0,
    warmup_count: int = 3,
) -> BenchmarkReport:
    """Benchmark by sending requests at a uniform rate of *rps* requests/s.

    Each request is staggered by ``1/rps`` seconds, regardless of whether
    earlier requests have finished.
    """
    import aiohttp

    interval = 1.0 / rps

    print(f"\n{'='*60}")
    print(f"  {label}  |  rps={rps} (interval={interval*1000:.0f}ms)  |  requests={len(prompts)}")
    print(f"{'='*60}")

    # Warmup — send a few requests back-to-back
    if warmup_count > 0:
        print(f"  Warming up ({warmup_count} requests)...")
        warmup_prompts = prompts[:min(warmup_count, len(prompts))]
        connector = aiohttp.TCPConnector(limit=warmup_count + 1)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                _stream_request(session, url, p["prompt"], -1, temperature, timeout)
                for p in warmup_prompts
            ]
            await asyncio.gather(*tasks)
        print("  Warmup complete.")

    # Main benchmark — pace requests at the target RPS
    connector = aiohttp.TCPConnector(limit=len(prompts) + 10)
    results: list[RequestResult] = []
    start_times: list[float] = []

    async with aiohttp.ClientSession(connector=connector) as session:

        async def _paced_request(idx: int, prompt: str, delay: float) -> RequestResult:
            await asyncio.sleep(delay)
            start_times.append(time.monotonic())
            return await _stream_request(
                session, url, prompt, idx, temperature, timeout,
            )

        tasks = []
        for i, p in enumerate(prompts):
            delay = i * interval
            tasks.append(asyncio.create_task(_paced_request(i, p["prompt"], delay)))

        t_bench_start = time.monotonic()

        pbar = tqdm(total=len(tasks), desc="Requests", unit="req")
        for task in asyncio.as_completed(tasks):
            try:
                r = await task
                results.append(r)
            except Exception as e:
                results.append(RequestResult(
                    index=-1, prompt="", ttft=0,
                    itl=[], total_time=0, input_tokens=0, output_tokens=0,
                    success=False, error=str(e),
                ))
            ok = sum(1 for r in results if r.success)
            pbar.set_postfix_str(f"{ok}/{len(results)} ok")
            pbar.update(1)
        pbar.close()

    t_bench_end = time.monotonic()
    total_time = t_bench_end - t_bench_start

    results.sort(key=lambda r: r.index)
    successful = [r for r in results if r.success]
    num_failed = len(results) - len(successful)
    actual_rps = len(results) / total_time if total_time > 0 else 0.0

    print(f"\n  {len(successful)}/{len(results)} succeeded"
          + (f"  ({num_failed} failed)" if num_failed else ""))

    if not successful:
        print("  No successful requests — aborting.")
        return BenchmarkReport(
            url=url, label=label, rps=rps,
            num_requests=len(prompts), num_successful=0,
            total_input_tokens=0, total_output_tokens=0,
            total_time=total_time, throughput_tokens_per_sec=0.0,
            actual_rps=actual_rps,
            ttft_mean=0, ttft_median=0, ttft_p95=0, ttft_p99=0,
            tpot_mean=0, tpot_median=0, tpot_p95=0, tpot_p99=0,
            e2e_latency_mean=0, e2e_latency_median=0, e2e_latency_p95=0, e2e_latency_p99=0,
            results=results,
        )

    total_input = sum(r.input_tokens for r in successful)
    total_output = sum(r.output_tokens for r in successful)
    throughput = total_output / total_time if total_time > 0 else 0.0

    ttfts = [r.ttft for r in successful if r.ttft > 0]
    tpot_values = []
    for r in successful:
        tpot_values.extend(r.itl)
    e2e_latencies = [r.total_time for r in successful]

    ttft_stats = _compute_stats(ttfts)
    tpot_stats = _compute_stats(tpot_values)
    e2e_stats = _compute_stats(e2e_latencies)

    return BenchmarkReport(
        url=url, label=label, rps=rps,
        num_requests=len(prompts), num_successful=len(successful),
        total_input_tokens=total_input, total_output_tokens=total_output,
        total_time=total_time, throughput_tokens_per_sec=throughput,
        actual_rps=actual_rps,
        ttft_mean=ttft_stats["mean"], ttft_median=ttft_stats["median"],
        ttft_p95=ttft_stats["p95"], ttft_p99=ttft_stats["p99"],
        tpot_mean=tpot_stats["mean"], tpot_median=tpot_stats["median"],
        tpot_p95=tpot_stats["p95"], tpot_p99=tpot_stats["p99"],
        e2e_latency_mean=e2e_stats["mean"], e2e_latency_median=e2e_stats["median"],
        e2e_latency_p95=e2e_stats["p95"], e2e_latency_p99=e2e_stats["p99"],
        results=results,
    )


# ── Reporting ──────────────────────────────────────────────────────────────────


def _print_report(report: BenchmarkReport) -> None:
    print(f"\n  ┌─ {report.label} (target rps={report.rps})")
    print(f"  ├── Actual rps:      {report.actual_rps:.1f} requests/s")
    print(f"  ├── Requests:        {report.num_successful}/{report.num_requests} succeeded")
    print(f"  ├── Throughput:      {report.throughput_tokens_per_sec:,.1f} tokens/s")
    print(f"  ├── Total time:      {report.total_time:.2f}s")
    print(f"  ├── Input tokens:    {report.total_input_tokens:,}")
    print(f"  ├── Output tokens:   {report.total_output_tokens:,}")
    print(f"  ├──")
    print(f"  ├── TTFT ──────────────────────────────────")
    print(f"  │   Mean:   {report.ttft_mean*1000:7.1f} ms")
    print(f"  │   Median: {report.ttft_median*1000:7.1f} ms")
    print(f"  │   P95:    {report.ttft_p95*1000:7.1f} ms")
    print(f"  │   P99:    {report.ttft_p99*1000:7.1f} ms")
    print(f"  ├──")
    print(f"  ├── TPOT (inter-token latency) ────────────")
    print(f"  │   Mean:   {report.tpot_mean*1000:7.1f} ms")
    print(f"  │   Median: {report.tpot_median*1000:7.1f} ms")
    print(f"  │   P95:    {report.tpot_p95*1000:7.1f} ms")
    print(f"  │   P99:    {report.tpot_p99*1000:7.1f} ms")
    print(f"  ├──")
    print(f"  ├── E2E Latency ───────────────────────────")
    print(f"  │   Mean:   {report.e2e_latency_mean*1000:7.1f} ms")
    print(f"  │   Median: {report.e2e_latency_median*1000:7.1f} ms")
    print(f"  │   P95:    {report.e2e_latency_p95*1000:7.1f} ms")
    print(f"  │   P99:    {report.e2e_latency_p99*1000:7.1f} ms")
    print(f"  └{'─' * 44}")


# ── Main ───────────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> None:
    # 1. Load dataset
    print("Loading ShareGPT dataset...")
    all_prompts = load_sharegpt(args.dataset_path)
    print(f"  Loaded {len(all_prompts)} prompts")

    if args.max_prompt_tokens > 0:
        all_prompts = [p for p in all_prompts
                       if _approx_token_count(p["prompt"]) <= args.max_prompt_tokens]
        print(f"  Filtered to {len(all_prompts)} prompts (max {args.max_prompt_tokens} tokens)")

    num_prompts = min(args.num_prompts, len(all_prompts))
    prompts = all_prompts[:num_prompts]
    print(f"  Using {len(prompts)} prompts for benchmark")

    # 2. Health check — use the base URL derived from the endpoint URL
    from urllib.parse import urlparse
    parsed = urlparse(args.url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    try:
        urllib.request.urlopen(f"{base_url}/docs", timeout=5)
    except Exception:
        print(f"Error: Cannot reach {base_url} — is the server running?")
        sys.exit(1)

    label = args.label or args.url
    # 3. Run benchmark
    report = await _benchmark(
        url=args.url,
        label=label,
        prompts=prompts,
        rps=args.rps,
        temperature=args.temperature,
        timeout=args.timeout,
        warmup_count=min(args.warmup, num_prompts),
    )
    _print_report(report)

    # 4. Save results
    if args.output:
        r = report
        d = {
                "url": r.url,
                "label": r.label,
                "rps": r.rps,
                "num_requests": r.num_requests,
                "num_successful": r.num_successful,
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
                "total_time_s": r.total_time,
                "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                "actual_rps": r.actual_rps,
                "ttft_mean_ms": r.ttft_mean * 1000,
                "ttft_median_ms": r.ttft_median * 1000,
                "ttft_p95_ms": r.ttft_p95 * 1000,
                "ttft_p99_ms": r.ttft_p99 * 1000,
                "tpot_mean_ms": r.tpot_mean * 1000,
                "tpot_median_ms": r.tpot_median * 1000,
                "tpot_p95_ms": r.tpot_p95 * 1000,
                "tpot_p99_ms": r.tpot_p99 * 1000,
                "e2e_latency_mean_ms": r.e2e_latency_mean * 1000,
                "e2e_latency_median_ms": r.e2e_latency_median * 1000,
                "e2e_latency_p95_ms": r.e2e_latency_p95 * 1000,
                "e2e_latency_p99_ms": r.e2e_latency_p99 * 1000,
                "request_details": [
                    {
                        "index": rr.index,
                        "success": rr.success,
                        "error": rr.error,
                        "ttft_ms": rr.ttft * 1000,
                        "tpot_ms": (sum(rr.itl) / len(rr.itl) * 1000) if rr.itl else 0,
                        "e2e_ms": rr.total_time * 1000,
                        "input_tokens": rr.input_tokens,
                        "output_tokens": rr.output_tokens,
                        "prompt_preview": rr.prompt[:80],
                    }
                    for rr in r.results
                ],
            }
        with open(args.output, "w") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a serving endpoint on ShareGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="Serving endpoint URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--label", default="",
                        help="Label for this benchmark run (default: use --url)")

    parser.add_argument("--dataset-path", default=None,
                        help="Local path to ShareGPT JSON (downloads from HF if omitted)")
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Number of prompts to benchmark (default: 200)")
    parser.add_argument("--max-prompt-tokens", type=int, default=128,
                        help="Max prompt length in tokens, approximate (0 = no filter)")

    parser.add_argument("--rps", type=float, default=1.0,
                        help="Requests per second (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--timeout", type=float, default=7200.0,
                        help="Per-request timeout in seconds (default: 7200)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup requests (default: 3)")

    parser.add_argument("--output", "-o", default="",
                        help="Save results to JSON file")

    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", message=".*Unclosed.*")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
