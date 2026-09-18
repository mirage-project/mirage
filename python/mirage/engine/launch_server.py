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
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .protocol import ChatRequest, TextRequest
from .output import GenerationEvent

DISCONNECT_POLL_INTERVAL = 0.05

logger = logging.getLogger(__name__)


def error_response(message, status=400, param=None, code=None):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": "invalid_request_error" if status < 500 else "server_error",
        "param": param, "code": code}})


def encode_sse(value):
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


@dataclass(frozen=True)
class CompletionResponse:
    model: str
    chat: bool
    created: int = field(default_factory=lambda: int(time.time()))
    suffix: str = field(default_factory=lambda: uuid.uuid4().hex)

    def envelope(self, choices, *, stream=False, usage=None):
        result = dict(
            id=("chatcmpl-" if self.chat else "cmpl-") + self.suffix,
            created=self.created, model=self.model, choices=choices,
            object=("chat.completion.chunk" if stream else "chat.completion")
                   if self.chat else "text_completion",
        )
        if usage is not None:
            result["usage"] = usage
        return result

    def choice(self, text, reason=None, *, stream=False, role=False):
        result = dict(index=0, finish_reason=reason)
        if self.chat:
            content = {} if stream and reason else {"content": text}
            if role or not stream:
                content["role"] = "assistant"
            result["delta" if stream else "message"] = content
        else:
            result["text"] = text
        return result

    def initial_chunk(self):
        return self.envelope([self.choice("", stream=True, role=True)], stream=True)

    def chunks(self, event: GenerationEvent, include_usage=False):
        if event.text:
            yield self.envelope([self.choice(event.text, stream=True)], stream=True)
        if event.finish_reason is not None:
            yield self.envelope([self.choice("", event.finish_reason, stream=True)], stream=True)
            if include_usage:
                yield self.envelope([], stream=True, usage=event.usage)

    def completed(self, text, event: GenerationEvent):
        return self.envelope([self.choice(text, event.finish_reason)], usage=event.usage)


@asynccontextmanager
async def lifespan(app):
    from .model_runner import ModelRunner
    from .llm_engine import LLMEngine
    app.state.engine = LLMEngine(ModelRunner(app.state.runner_config))
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.engine.close)


app = FastAPI(title="MPK LLM Engine", lifespan=lifespan)


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


async def complete(request, chat):
    try:
        body = await request.json()
        if isinstance(body, dict):
            body = {**request.app.state.sampling_defaults, **body}
        
        req = (ChatRequest if chat else TextRequest).model_validate(body)
        params = req.sampling_params()

    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        return error_response(first["msg"], param=".".join(map(str, first["loc"])))
    except (ValueError, UnicodeDecodeError):
        return error_response("Invalid or empty JSON body")
    model = request.app.state.served_model
    if req.model != model:
        return error_response(f"Model '{req.model}' is not served", 404, "model", "model_not_found")
    engine = request.app.state.engine
    
    try:
        kwargs = {"messages": [m.template_message() for m in req.messages]} if chat else {"prompt": req.prompt}
        prepared = await asyncio.to_thread(engine.prepare, params=params, **kwargs)
        submission = asyncio.create_task(asyncio.to_thread(
            engine.generate, prepared, request.app.state.request_timeout, poll=True))
        try:
            session = await asyncio.shield(submission)
        except asyncio.CancelledError:
            # Publishing may already be running in a worker thread. Recover its
            # handle before propagating cancellation so no request is orphaned.
            try:
                abandoned = await submission
                abandoned.close()
            finally:
                raise
    except ValueError as exc:
        return error_response(str(exc))
    except OverflowError as exc:
        return error_response(str(exc), 429, code="server_overloaded")
    except Exception:
        logger.exception("Failed to submit generation")
        return error_response("Unable to start generation", 503)

    response = CompletionResponse(model, chat)

    async def events():
        try:
            for event in session:
                if event is None:
                    await asyncio.sleep(0.002)
                else:
                    yield event
        finally:
            # Nonblocking iteration keeps next() and close() on the event loop.
            session.close()

    if req.stream:
        async def sse():
            try:
                if chat:
                    yield encode_sse(response.initial_chunk())
                include_usage = bool(req.stream_options and req.stream_options.include_usage)
                async for event in events():
                    for chunk in response.chunks(event, include_usage):
                        yield encode_sse(chunk)
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.exception("Streaming generation failed")
                message = "Generation timed out" if isinstance(exc, TimeoutError) else "Generation failed"
                yield encode_sse({"error": {"message": message, "type": "server_error", "param": None, "code": None}})
                yield "data: [DONE]\n\n"
            finally:
                session.close()
        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def collect():
        result = ""
        final = None
        async for event in events():
            result += event.text
            final = event
        if final is None or final.finish_reason is None:
            raise RuntimeError("generation ended without a terminal event")
        return response.completed(result, final)

    task = asyncio.create_task(collect())
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=DISCONNECT_POLL_INTERVAL)
            if await request.is_disconnected():
                task.cancel()
                return error_response("Client disconnected", 499)
        return await task
    except TimeoutError:
        return error_response("Generation timed out", 504)
    except Exception:
        logger.exception("Generation failed")
        return error_response("Generation failed", 500)
    finally:
        session.close()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def main():
    import uvicorn
    from .model_runner import RunnerConfig

    parser = argparse.ArgumentParser(description="Mirage LLM Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--model-path", default=None, help="Path to local model")
    parser.add_argument("--served-model-name")
    parser.add_argument("--developer-role", choices=["system", "native", "reject"], default="system")
    parser.add_argument("--pinned-ring-capacity", type=int, default=8)
    parser.add_argument("--max-pending-requests", type=int, default=128)
    parser.add_argument("--no-use-cutlass-kernel", action="store_false", dest="use_cutlass_kernel")
    parser.add_argument("--max-num-batched-requests", default=4, type=int)
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--output-dir", default=None, help="Output directory for compiled artifacts")
    parser.add_argument("--request-timeout", default=7200.0, type=float,
                        help="Per-request timeout in seconds (default: 7200)")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true",
                        help="Set sampling defaults for omitted request fields")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", "--top_p", type=float, default=0.95)
    parser.add_argument("--top-k", "--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-topk-max", type=int, default=32)
    args = parser.parse_args()
    if args.do_sample and args.temperature <= 0.0:
        parser.error("--do-sample needs --temperature > 0")

    config = RunnerConfig(
        model=args.model,
        model_path=args.model_path,
        developer_role=args.developer_role,
        pinned_ring_capacity=args.pinned_ring_capacity,
        max_pending_requests=args.max_pending_requests,
        use_cutlass_kernel=args.use_cutlass_kernel,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_seq_length=args.max_seq_length,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        output_dir=args.output_dir,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sampling_seed=args.seed,
        sampling_topk_max=args.sampling_topk_max,
    )
    app.state.runner_config = config
    app.state.sampling_defaults = config.sampling_defaults()
    app.state.served_model = args.served_model_name or args.model
    app.state.request_timeout = args.request_timeout
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
