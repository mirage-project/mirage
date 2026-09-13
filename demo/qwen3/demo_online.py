"""Demo: online serving via HTTP API — single, concurrent, and streaming.

Start the server first::

    python -m mirage.engine.launch_server --max-num-batched-tokens 4

Then run the demo::

    python demo/qwen3/demo_online.py                  # single request
    python demo/qwen3/demo_online.py --concurrent 3   # 3 concurrent requests
    python demo/qwen3/demo_online.py --stream          # streaming (SSE)

"""

import argparse
import json
import sys
import time
import threading
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"


def chat(prompt: str, stream: bool = False, timeout: int = 300):
    """Send one chat-completion request.  Returns decoded text for non-stream,
    or yields (text, is_final) tuples for stream."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
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
    resp = urllib.request.urlopen(req, timeout=timeout)
    buffer = ""
    while True:
        chunk = resp.read(256).decode("utf-8", errors="replace")
        if not chunk:
            break
        buffer += chunk
        while "\n\n" in buffer:
            line, buffer = buffer.split("\n\n", 1)
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                raise RuntimeError(obj["error"])
            text = obj["choices"][0]["delta"].get("content", "")
            yield (text, False)
    yield ("", True)


# ── runners ──────────────────────────────────────────────────────────────────


def run_single(prompt: str, timeout: int = 300) -> bool:
    t0 = time.monotonic()
    text = chat(prompt, timeout=timeout)
    elapsed = time.monotonic() - t0
    print(f"  {text[:300]}...")
    print(f"\n  Finished in {elapsed:.1f}s")
    ok = bool(text and text.strip())
    if not ok:
        print("  ERROR: empty response")
    return ok


def run_concurrent(prompts: list[str], timeout: int = 300) -> bool:
    results: list[tuple[int, str] | None] = [None] * len(prompts)

    def _worker(idx: int, prompt: str):
        try:
            results[idx] = (idx, chat(prompt, timeout=timeout))
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


def run_stream(prompt: str, timeout: int = 300) -> bool:
    print(f"\n[stream] {prompt!r}\n")
    got = 0
    for text, is_final in chat(prompt, stream=True, timeout=timeout):
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
    args = parser.parse_args()

    global BASE
    BASE = f"http://127.0.0.1:{args.port}"

    # Quick health check
    try:
        urllib.request.urlopen(f"{BASE}/docs", timeout=5)
    except Exception:
        print(f"Cannot reach server at {BASE}. Start it first:")
        print(f"  python -m mirage.engine.launch_server --max-num-batched-tokens 4")
        return

    if args.stream:
        ok = run_stream("Explain what a GPU is in one sentence.", timeout=args.timeout)
    elif args.concurrent > 0:
        prompts = [
            "Introduce yourself.",
            "How to implement GEMM kernel at nvidia blackwell gpu, please explain in detail.",
            "Explain the difference between lpl and lck.",
            "What is buggy in CMU? CMU means Carnegie Mellon University",
            "Lebron James and Steven Curry, who is the goat?",
            "Do you think Attack on Titan really have a good end?"
        ][: args.concurrent]
        print(f"\n=== {len(prompts)} concurrent requests ===\n")
        ok = run_concurrent(prompts, timeout=args.timeout)
    else:
        print("\n=== Single request ===\n")
        ok = run_single("Say hello.", timeout=args.timeout)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
