"""OpenAI-compatible text generation server backed by Mirage's persistent kernel."""
from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .protocol import ChatRequest, TextRequest
from .responses import CompletionResponse, encode_sse
from .config import DEFAULT_MODEL, DEFAULT_REQUEST_TIMEOUT, RunnerConfig

DISCONNECT_POLL_INTERVAL = 0.05

logger = logging.getLogger(__name__)


def error_response(message, status=400, param=None, code=None):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": "invalid_request_error" if status < 500 else "server_error",
        "param": param, "code": code}})


@asynccontextmanager
async def lifespan(app):
    from .model_runner import ModelRunner
    from .llm_engine import LLMEngine
    app.state.engine = LLMEngine(ModelRunner(app.state.runner_config))
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.engine.close)


def create_app(engine=None, *, model=None, request_timeout=DEFAULT_REQUEST_TIMEOUT, sampling_defaults=None):
    app = FastAPI(title="Mirage OpenAI API", lifespan=lifespan if engine is None else None)
    app.state.engine = engine
    app.state.served_model = model
    app.state.sampling_defaults = dict(sampling_defaults or {})
    app.state.request_timeout = request_timeout

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": app.state.served_model, "object": "model",
                                           "created": 0, "owned_by": "mirage"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await complete(request, chat=True)

    @app.post("/v1/completions")
    async def text(request: Request):
        return await complete(request, chat=False)

    return app


def _next(iterator):
    # StopIteration must not propagate through an asyncio Future.
    return next(iterator, None)


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
            engine.generate, prepared, request.app.state.request_timeout))
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
    iterator = iter(session)

    async def events():
        try:
            while True:
                event = await asyncio.to_thread(_next, iterator)
                if event is None:
                    break
                yield event
        finally:
            # Session cancellation is thread-safe, even while next() is blocked.
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
            except asyncio.CancelledError:
                raise
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


app = create_app()


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    runner_defaults = RunnerConfig(model=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path")
    parser.add_argument("--served-model-name")
    for name in ("max_num_batched_requests", "max_num_batched_tokens", "max_seq_length",
                 "max_num_pages", "page_size", "pinned_ring_capacity", "max_pending_requests"):
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=getattr(runner_defaults, name))
    parser.add_argument("--developer-role", choices=["system", "native", "reject"], default=runner_defaults.developer_role,
                        help="Explicit model adapter; system maps developer messages to system messages")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-use-cutlass-kernel", action="store_false",
                        dest="use_cutlass_kernel",
                        help="Use Mirage's PTX linear kernels (needed when a CUTLASS task exceeds the GPU shared-memory limit)")
    parser.set_defaults(use_cutlass_kernel=runner_defaults.use_cutlass_kernel)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--do-sample", action="store_true",
                        help="Use startup sampling defaults for omitted request fields; explicit request values override them")
    parser.add_argument("--temperature", type=float, default=runner_defaults.temperature)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=runner_defaults.top_p)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=runner_defaults.top_k)
    parser.add_argument("--seed", dest="sampling_seed", type=int, default=runner_defaults.sampling_seed)
    parser.add_argument("--sampling-topk-max", type=int, default=runner_defaults.sampling_topk_max,
                        help="Compatibility option for the SM100 graph sampler; does not limit per-request HTTP sampling")
    args = parser.parse_args()
    config_keys = RunnerConfig.__dataclass_fields__
    try:
        app.state.runner_config = RunnerConfig(**{k: v for k, v in vars(args).items() if k in config_keys})
    except ValueError as exc:
        parser.error(str(exc))
    app.state.sampling_defaults = app.state.runner_config.sampling_defaults()
    app.state.served_model = args.served_model_name or args.model
    app.state.request_timeout = args.request_timeout
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
