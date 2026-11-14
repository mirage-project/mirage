"""Demo: online serving via HTTP API — single, concurrent, and streaming.

Start the server first::

    python -m mirage.engine.launch_server --model Qwen/Qwen3-0.6B

Then run the demo::

    python demo/qwen3/demo_online.py --model Qwen/Qwen3-0.6B
    python demo/qwen3/demo_online.py --model Qwen/Qwen3-0.6B --concurrent 3
    python demo/qwen3/demo_online.py --model Qwen/Qwen3-0.6B --stream
    python demo/qwen3/demo_online.py --stream --model Qwen/Qwen3-0.6B \
        --temperature .8 --top-p .95 --top-k 20

Requests generate at most 128 tokens by default; prompt plus output must fit
the server's context capacity.

"""

import argparse
import json
import sys
import time
import threading
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"


def chat(prompt: str, stream: bool = False, timeout: int = 300, *,
         base: str = BASE, model: str, max_tokens: int = 128,
         sampling: dict | None = None):
    """Send one chat-completion request.  Returns decoded text for non-stream,
    or yields (text, is_final) tuples for stream."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }
    payload.update(sampling or {})
    body = json.dumps(payload).encode()

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    if stream:
        return _read_sse(req, timeout)
    else:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]


def _read_sse(req: urllib.request.Request, timeout: int):
    """Generator that yields (text, is_final) from an SSE stream."""
    finished = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Read complete SSE lines without waiting for an arbitrary byte count.
        # Decoding each complete line also preserves split UTF-8 codepoints.
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                if not finished:
                    raise RuntimeError("Stream ended without a finish reason")
                yield ("", True)
                return
            obj = json.loads(payload)
            if "error" in obj:
                raise RuntimeError(obj["error"])
            choices = obj.get("choices", [])
            if not choices:  # final usage event
                continue
            choice = choices[0]
            finished |= choice.get("finish_reason") is not None
            text = choice.get("delta", {}).get("content", "")
            if text:
                yield (text, False)
    raise RuntimeError("Incomplete stream: missing [DONE]")


# ── runners ──────────────────────────────────────────────────────────────────


def run_single(client, prompt: str) -> bool:
    t0 = time.monotonic()
    text = client(prompt)
    elapsed = time.monotonic() - t0
    print(f"  {text[:300]}...")
    print(f"\n  Finished in {elapsed:.1f}s")
    ok = bool(text and text.strip())
    if not ok:
        print("  ERROR: empty response")
    return ok


def run_concurrent(client, prompts: list[str]) -> bool:
    results: list[tuple[int, str] | None] = [None] * len(prompts)

    def _worker(idx: int, prompt: str):
        try:
            results[idx] = (idx, client(prompt))
        except Exception as e:
            results[idx] = (idx, f"ERROR: {e}")

    threads = [threading.Thread(target=_worker, args=(i, p)) for i, p in enumerate(prompts)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
        time.sleep(0.001)  # stagger arrivals by ~1 ms
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    ok = True
    for r in results:
        if r is None:
            ok = False
            continue
        i, text = r
        print(f"  [{i}]  →  {text[:150].replace(chr(10), ' ')}...")
        if text.startswith("ERROR:") or not text.strip():
            ok = False
    print(f"\n  {len(prompts)} requests finished in {elapsed:.1f}s")
    return ok


def run_stream(client, prompt: str) -> bool:
    print(f"\n[stream] {prompt!r}\n")
    got = 0
    for text, is_final in client(prompt, stream=True):
        if is_final:
            print("\n\n--- DONE ---")
        else:
            print(text, end="", flush=True)
            if text:
                got += 1
    ok = got > 0
    if not ok:
        print("\n  ERROR: stream produced no tokens")
    return ok


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMEngine HTTP API demo")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--concurrent", type=int, default=0, help="Number of concurrent requests")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Served model name")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frequency-penalty", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    args = parser.parse_args()
    if args.concurrent < 0:
        parser.error("--concurrent must be nonnegative")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    base = f"http://127.0.0.1:{args.port}"

    sampling = {name: value for name in (
        "temperature", "top_p", "top_k", "seed", "frequency_penalty",
        "presence_penalty", "repetition_penalty")
        if (value := getattr(args, name)) is not None}
    def client(prompt, stream=False):
        return chat(prompt, stream=stream, base=base, model=args.model,
                    timeout=args.timeout, max_tokens=args.max_tokens,
                    sampling=sampling)

    if args.stream:
        ok = run_stream(client, "Explain what a GPU is in one sentence.")
    elif args.concurrent > 0:
        prompts = [
            "Introduce yourself in one sentence.",
            "Explain what matrix multiplication does.",
            "Why is the sky blue?",
            "Name three programming languages.",
        ]
        prompts = [prompts[i % len(prompts)] for i in range(args.concurrent)]
        print(f"\n=== {len(prompts)} concurrent requests ===\n")
        ok = run_concurrent(client, prompts)
    else:
        print("\n=== Single request ===\n")
        ok = run_single(client, "Say hello.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)
