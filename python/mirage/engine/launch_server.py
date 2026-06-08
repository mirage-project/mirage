"""Launch the Mirage LLM Engine as an OpenAI-compatible HTTP server.

Usage::

    python -m mirage.engine.launch_server \\
        --model Qwen/Qwen3-8B \\
        --max-num-batched-requests 4 \\
        --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .model_runner import ModelRunner, RunnerConfig
from .llm_engine import LLMEngine


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: RunnerConfig = app.state.runner_config
    runner = ModelRunner(config)
    engine = LLMEngine(runner)
    app.state.engine = engine
    yield
    engine.close()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MPK LLM Engine", lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _parse_json(request: Request) -> dict:
    """Parse JSON body, returning 400 on empty or malformed input."""
    try:
        return await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or empty JSON body")


def _extract_prompt(messages: list[dict]) -> str:
    """Pull the last user message from an OpenAI chat messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


async def _stream_bridge(
    engine: LLMEngine, prompt: str, timeout: float,
) -> AsyncGenerator[str, None]:
    """Bridge a synchronous streaming generator to async SSE chunks.

    Each request gets its own daemon thread so concurrent requests are never
    gated by the default ``ThreadPoolExecutor`` pool size.  Items produced by
    the thread are handed to the event loop via ``call_soon_threadsafe`` so
    that the asyncio queue is accessed only from the event-loop thread.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _put(text: str, is_final: bool, error: str | None) -> None:
        """Called on the event-loop thread; safe to touch the asyncio queue."""
        queue.put_nowait((text, is_final, error))

    def _run() -> None:
        try:
            gen = engine.submit(prompt, stream=True, timeout=timeout)
            for text, is_final in gen:
                loop.call_soon_threadsafe(_put, text, is_final, None)
        except BaseException as exc:
            loop.call_soon_threadsafe(_put, "", True, str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        text, is_final, error = await queue.get()

        if error:
            yield "data: " + json.dumps({"error": error}) + "\n\n"
            break

        chunk = json.dumps({
            "choices": [{"delta": {"content": text}, "index": 0}],
        })
        yield f"data: {chunk}\n\n"
        if is_final:
            break

    thread.join()
    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await _parse_json(request)
    prompt = _extract_prompt(body.get("messages", []))
    stream = body.get("stream", False)
    timeout = request.app.state.request_timeout

    if stream:
        return StreamingResponse(
            _stream_bridge(request.app.state.engine, prompt, timeout),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: request.app.state.engine.submit(
                prompt, timeout=timeout),
        )
        return {
            "id": "chatcmpl-0",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
        }


@app.post("/v1/completions")
async def completions(request: Request):
    body = await _parse_json(request)
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)
    timeout = request.app.state.request_timeout

    if stream:
        return StreamingResponse(
            _stream_bridge(request.app.state.engine, prompt, timeout),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: request.app.state.engine.submit(
                prompt, timeout=timeout),
        )
        return {
            "id": "cmpl-0",
            "object": "text_completion",
            "choices": [{
                "index": 0,
                "text": result["text"],
                "finish_reason": "stop",
            }],
        }


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Mirage LLM Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--model-path", default=None, help="Path to local model")
    parser.add_argument("--max-num-batched-requests", default=4, type=int)
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--output-dir", default=None, help="Output directory for compiled artifacts")
    parser.add_argument("--request-timeout", default=7200.0, type=float,
                        help="Per-request timeout in seconds (default: 7200)")
    args = parser.parse_args()

    config = RunnerConfig(
        model=args.model,
        model_path=args.model_path,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_seq_length=args.max_seq_length,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        output_dir=args.output_dir,
    )
    app.state.runner_config = config
    app.state.request_timeout = args.request_timeout
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
