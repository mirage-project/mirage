"""Test sequential or concurrent online serving and report metrics."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL = "Qwen/Qwen3-8B"
REQUEST_COUNT = 12
BURST_REQUESTS = 10
FOLLOW_UP_REQUESTS = 2
RING_CAPACITY = 8
MAX_SEQ_LENGTH = 512
REQUEST_TIMEOUT = 300

assert REQUEST_COUNT > RING_CAPACITY
assert BURST_REQUESTS + FOLLOW_UP_REQUESTS == REQUEST_COUNT
assert BURST_REQUESTS > RING_CAPACITY


@dataclass
class Sample:
    index: int
    stream: bool
    marker: str
    status: int = 0
    text: str = ""
    answer: str = ""
    error: str = ""
    correct: bool = False
    output_tokens: int = 0
    e2e_ms: float = 0.0
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    done_count: int = 0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("MIRAGE_HOME", str(root))
    cuda_home = environment.get("CUDA_HOME") or environment.get("CUDA_PATH")
    if not cuda_home and Path("/usr/local/cuda/bin/nvcc").is_file():
        cuda_home = "/usr/local/cuda"
    if cuda_home:
        environment.update({"CUDA_HOME": cuda_home, "CUDA_PATH": cuda_home})
        environment["PATH"] = f"{cuda_home}/bin:{environment.get('PATH', '')}"
    return environment


def wait_until_ready(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/openapi.json", timeout=5) as response:
                spec = json.load(response)
            if "/v1/chat/completions" in spec.get("paths", {}):
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become ready")


def send_chat(
    url: str,
    index: int,
    barrier: threading.Barrier | None = None,
) -> Sample:
    marker = f"MIRAGE_REQUEST_{index:04d}"
    sample = Sample(index=index, stream=index % 2 == 1, marker=marker)
    body = json.dumps({
        "messages": [{
            "role": "user",
            "content": f"Reply with exactly {marker} and nothing else.",
        }],
        "stream": sample.stream,
    }).encode()
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        if barrier is not None:
            barrier.wait(timeout=30)
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            sample.status = response.status
            if not sample.stream:
                payload = json.load(response)
                sample.text = payload["choices"][0]["message"]["content"]
            else:
                chunks = []
                itls = []
                last_token_at = None
                for raw_line in response:
                    data = raw_line.decode(errors="replace").strip()
                    if not data.startswith("data: "):
                        continue
                    data = data[6:]
                    if data == "[DONE]":
                        sample.done_count += 1
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    text = event["choices"][0]["delta"].get("content", "")
                    chunks.append(text)
                    if text:
                        now = time.monotonic()
                        if sample.ttft_ms is None:
                            sample.ttft_ms = (now - started) * 1000
                        elif last_token_at is not None:
                            itls.append((now - last_token_at) * 1000)
                        last_token_at = now
                sample.text = "".join(chunks)
                if itls:
                    sample.tpot_ms = statistics.mean(itls)
    except Exception as exc:
        sample.error = f"{type(exc).__name__}: {exc}"
    sample.e2e_ms = (time.monotonic() - started) * 1000
    return sample


def run_requests(url: str, concurrent: bool) -> tuple[list[Sample], float]:
    started = time.monotonic()
    if not concurrent:
        samples = [send_chat(url, index) for index in range(REQUEST_COUNT)]
        return samples, time.monotonic() - started

    barrier = threading.Barrier(BURST_REQUESTS)
    with ThreadPoolExecutor(max_workers=BURST_REQUESTS) as executor:
        futures = [
            executor.submit(send_chat, url, index, barrier)
            for index in range(BURST_REQUESTS)
        ]
        samples = [future.result() for future in futures]
    samples.extend(
        send_chat(url, index)
        for index in range(BURST_REQUESTS, REQUEST_COUNT)
    )
    return sorted(samples, key=lambda sample: sample.index), time.monotonic() - started


def validate(sample: Sample, markers: list[str], tokenizer) -> None:
    errors = [sample.error] if sample.error else []
    sample.answer = sample.text.rsplit("</think>", 1)[-1].strip()
    sample.output_tokens = len(
        tokenizer.encode(sample.text, add_special_tokens=False)
    )
    if sample.status != 200:
        errors.append(f"HTTP status is {sample.status}")
    if sample.answer != sample.marker:
        errors.append("final answer does not match the request marker")
    if any(marker in sample.text for marker in markers if marker != sample.marker):
        errors.append("response contains another request marker")
    if sample.stream and sample.done_count != 1:
        errors.append(f"received {sample.done_count} [DONE] events")
    sample.error = "; ".join(errors)
    sample.correct = not errors


def latency(values: list[float]) -> dict[str, float | int | None]:
    values.sort()
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None}
    p95_index = round(0.95 * (len(values) - 1))
    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": values[p95_index],
    }


def stop_server(
    process: subprocess.Popen, log_path: Path
) -> tuple[bool, int | None, float]:
    started = time.monotonic()
    if process.poll() is not None:
        return False, process.returncode, 0.0
    process.terminate()
    try:
        exit_code = process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return False, process.returncode, time.monotonic() - started
    log = log_path.read_text(encoding="utf-8", errors="replace")
    clean = (
        exit_code in (0, -15)
        and "Application shutdown complete." in log
        and "Finished server process" in log
    )
    return clean, exit_code, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("sequential", "concurrent"),
        default="sequential",
    )
    args = parser.parse_args()
    concurrent = args.scenario == "concurrent"
    request_rows = 2 if concurrent else 1
    batched_tokens = 4 if concurrent else 1
    root = Path(__file__).resolve().parents[2]
    output_path = root / f"outputs/online_serving_{args.scenario}_metrics.json"
    log_path = root / f"logs/online_server_{args.scenario}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m", "mirage.engine.launch_server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", MODEL,
        "--max-num-batched-requests", str(request_rows),
        "--max-num-batched-tokens", str(batched_tokens),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--request-timeout", str(REQUEST_TIMEOUT),
    ]
    process = None
    samples = []
    request_time = 0.0
    shutdown_clean, exit_code, shutdown_time = False, None, 0.0
    harness_error = ""
    startup_started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=server_environment(root),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wait_until_ready(process, url)
        startup_time = time.monotonic() - startup_started

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        markers = [f"MIRAGE_REQUEST_{index:04d}" for index in range(REQUEST_COUNT)]
        samples, request_time = run_requests(url, concurrent)
        for sample in samples:
            validate(sample, markers, tokenizer)
    except Exception as exc:
        startup_time = time.monotonic() - startup_started
        harness_error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            shutdown_clean, exit_code, shutdown_time = stop_server(
                process, log_path
            )

    successful = sum(sample.status == 200 for sample in samples)
    correct = sum(sample.correct for sample in samples)
    output_tokens = sum(sample.output_tokens for sample in samples)
    streaming = [sample for sample in samples if sample.stream]
    passed = (
        not harness_error
        and len(samples) == REQUEST_COUNT
        and correct == REQUEST_COUNT
        and shutdown_clean
    )
    metrics = {
        "requests_per_second": successful / request_time if request_time else 0,
        "output_tokens_per_second": (
            output_tokens / request_time if request_time else 0
        ),
        "output_tokens": output_tokens,
        "e2e": latency([sample.e2e_ms for sample in samples]),
        "streaming_ttft": latency([
            sample.ttft_ms for sample in streaming if sample.ttft_ms is not None
        ]),
        "streaming_tpot": latency([
            sample.tpot_ms for sample in streaming if sample.tpot_ms is not None
        ]),
        "server_startup_s": startup_time,
        "server_shutdown_s": shutdown_time,
    }
    report = {
        "passed": passed,
        "harness_error": harness_error,
        "configuration": {
            "model": MODEL,
            "scenario": args.scenario,
            "requests": REQUEST_COUNT,
            "sequential": not concurrent,
            "burst_requests": BURST_REQUESTS if concurrent else 0,
            "follow_up_requests": FOLLOW_UP_REQUESTS if concurrent else 0,
            "request_rows": request_rows,
            "ring_capacity": RING_CAPACITY,
        },
        "checks": {
            "http_success": successful,
            "correct": correct,
            "shutdown_clean": shutdown_clean,
        },
        "metrics": metrics,
        "server_exit_code": exit_code,
        "results": [asdict(sample) for sample in samples],
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Online serving ({args.scenario}): {'PASS' if passed else 'FAIL'}")
    print(f"  Correct: {correct}/{REQUEST_COUNT}; HTTP: {successful}/{REQUEST_COUNT}")
    print(json.dumps(metrics, indent=2))
    print(f"  Clean shutdown: {shutdown_clean}")
    if harness_error:
        print(f"  Harness error: {harness_error}")
    for sample in samples:
        if not sample.correct:
            print(f"  Request {sample.index}: {sample.error}")
    print(f"  Metrics: {output_path}")
    if not passed and log_path.exists():
        print("\nServer log tail")
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-100:]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
