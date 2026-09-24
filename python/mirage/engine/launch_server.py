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
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .model_runner import ModelRunner, RunnerConfig
from .llm_engine import LLMEngine
from .protocol import ChatRequest, TextRequest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner = ModelRunner(app.state.runner_config)
    engine = LLMEngine(runner)
    app.state.engine = engine
    try:
        yield
    finally:
        engine.close()


app = FastAPI(title="MPK LLM Engine", lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────


def error_response(message, status=400, param=None):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": "invalid_request_error" if status < 500 else "server_error",
        "param": param, "code": None}})


def _decode_output(tokens, tokenizer):
    """Yield text deltas, finish reasons, and token counts for HTTP responses."""
    ids, emitted = [], ""
    for token, reason in tokens:
        if token is not None:
            ids.append(token)
        text = tokenizer.decode(ids)
        if not reason:
            # Keep incomplete UTF-8 out of streamed text.
            text = text.rstrip("\ufffd")
        yield text[len(emitted):], reason, len(ids)
        emitted = text
        if reason:
            break


async def _stream_bridge(tokens, tokenizer):
    """Bridge the engine's synchronous stream to the HTTP event loop."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def run():
        try:
            for item in _decode_output(tokens, tokenizer):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def complete(request: Request, chat: bool):
    engine = request.app.state.engine
    model = request.app.state.served_model
    try:
        body = await request.json()
        req = (ChatRequest if chat else TextRequest).model_validate(body)
        params = req.sampling_params()
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        return error_response(first["msg"], param=".".join(map(str, first["loc"])))
    except ValueError:
        return error_response("Invalid or empty JSON body")
    if req.model != model:
        return error_response(f"Model '{req.model}' is not served", 404, "model")
    try:
        tokenizer = engine.tokenizer_manager
        ids = (tokenizer.tokenize_messages([m.template_message() for m in req.messages])
               if chat else tokenizer.tokenize_raw(req.prompt))
        tokens = await asyncio.to_thread(
            engine.submit, ids, stream=True, sampling_params=params,
            return_token_ids=True, timeout=request.app.state.request_timeout)
    except ValueError as exc:
        return error_response(str(exc))
    except Exception:
        logger.exception("Failed to submit generation")
        return error_response("Unable to start generation", 503)

    response_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
    created = int(time.time())

    def usage(count):
        return dict(prompt_tokens=len(ids), completion_tokens=count, total_tokens=len(ids) + count)

    def response(text="", reason=None, *, role=False, usage=None):
        choice = dict(index=0, finish_reason=reason)
        if chat:
            content = {} if req.stream and reason else {"content": text}
            if role or not req.stream:
                content["role"] = "assistant"
            choice["delta" if req.stream else "message"] = content
        else:
            choice["text"] = text
        result = dict(id=response_id, created=created, model=model, choices=[choice],
                      object=("chat.completion.chunk" if req.stream else "chat.completion")
                             if chat else "text_completion")
        if usage is not None:
            result["usage"] = usage
            if req.stream:
                result["choices"] = []
        return result

    events = _stream_bridge(tokens, tokenizer)
    if req.stream:
        async def sse():
            def encode(value):
                return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"
            try:
                if chat:
                    yield encode(response(role=True))
                async for text, reason, count in events:
                    if text:
                        yield encode(response(text))
                    if reason:
                        yield encode(response(reason=reason))
                        if req.stream_options and req.stream_options.include_usage:
                            yield encode(response(usage=usage(count)))
            except Exception:
                logger.exception("Streaming generation failed")
                yield encode({"error": {"message": "Generation failed", "type": "server_error",
                                        "param": None, "code": None}})
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    try:
        result = ""
        async for text, reason, count in events:
            result += text
        return response(result, reason, usage=usage(count))
    except TimeoutError:
        return error_response("Generation timed out", 504)
    except Exception:
        logger.exception("Generation failed")
        return error_response("Generation failed", 500)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": app.state.served_model, "object": "model",
                                       "created": 0, "owned_by": "mirage"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await complete(request, chat=True)


@app.post("/v1/completions")
async def completions(request: Request):
    return await complete(request, chat=False)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Mirage LLM Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--model-path", default=None, help="Path to local model")
    parser.add_argument("--served-model-name")
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
    app.state.served_model = args.served_model_name or args.model
    app.state.request_timeout = args.request_timeout
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
