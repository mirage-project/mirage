"""CPU checks for the upstream SM100 / per-request serving integration."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from mirage.engine.config import RunnerConfig
from mirage.engine.launch_server import create_app, main
from mirage.engine.llm_engine import LLMEngine
from mirage.engine.output import GenerationEvent
from mirage.engine.sampling import SamplingParams
from mirage.mpk.models.qwen3.builder import Qwen3Builder


@pytest.mark.parametrize("chat", [False, True])
def test_startup_defaults_and_request_overrides(chat):
    config = RunnerConfig(model="test", do_sample=True, temperature=.8,
                          top_p=.9, top_k=50, sampling_seed=42)
    engine = MagicMock()
    engine.generate.side_effect = lambda *args: iter_session()
    app = create_app(engine, model="test", sampling_defaults=config.sampling_defaults())
    payload = dict(model="test", **(
        {"messages": [{"role": "user", "content": "hi"}]} if chat else {"prompt": "hi"}))
    endpoint = "/v1/chat/completions" if chat else "/v1/completions"
    with TestClient(app) as client:
        response = client.post(endpoint, json=payload)
        assert response.status_code == 200, response.text
        params = engine.prepare.call_args.kwargs["params"]
        assert (params.temperature, params.top_p, params.top_k, params.seed) == (.8, .9, 50, 42)
        response = client.post(endpoint, json={**payload, "temperature": 0,
                                               "top_p": 1, "top_k": 0, "seed": 7})
        assert response.status_code == 200, response.text
        params = engine.prepare.call_args.kwargs["params"]
        assert (params.temperature, params.top_p, params.top_k, params.seed) == (0, 1, 0, 7)
        assert client.post(endpoint, json={**payload, "top_p": 0}).status_code == 400


def iter_session():
    class Session:
        def __iter__(self):
            yield GenerationEvent("hello", "length", 2, 1)

        def close(self):
            pass
    return Session()


@pytest.mark.parametrize("options", [
    {"temperature": 0}, {"temperature": float("nan")}, {"top_p": 0},
    {"top_k": -1}, {"sampling_seed": -1},
])
def test_invalid_startup_sampling(options):
    with pytest.raises(ValueError):
        RunnerConfig(model="test", do_sample=True, **options)


def test_python_engine_defaults_and_explicit_parameters():
    engine = LLMEngine.__new__(LLMEngine)
    config = RunnerConfig(model="test", do_sample=True, top_k=50)
    engine.model_runner = SimpleNamespace(config=config)
    engine.tokenizer_manager = SimpleNamespace(tokenize=lambda *args: [1, 2])
    engine.vocab_size, engine.eos_ids = 100, [99]
    prepared = engine.prepare(prompt="hi")
    assert prepared.params.temperature == .8
    assert prepared.params.top_k == 50
    explicit = SamplingParams(temperature=0, seed=7)
    assert engine.prepare(prompt="hi", params=explicit).params is explicit
    config.do_sample = False
    assert config.sampling_defaults() == {}
    assert engine.prepare(prompt="hi").params.temperature == 0


@pytest.mark.parametrize("top_k_flag,top_p_flag", [("--top_k", "--top_p"), ("--top-k", "--top-p")])
def test_cli_sampling_defaults(monkeypatch, top_k_flag, top_p_flag):
    import uvicorn
    from mirage.engine import launch_server

    app = create_app()
    monkeypatch.setattr(launch_server, "app", app)
    monkeypatch.setattr(uvicorn, "run", MagicMock())
    monkeypatch.setattr("sys.argv", ["launch_server", "--model", "test", "--do-sample",
                                    "--temperature", ".7", top_k_flag, "50", top_p_flag,
                                    ".8", "--seed", "17", "--sampling-topk-max", "32"])
    main()
    assert app.state.sampling_defaults == dict(temperature=.7, top_k=50, top_p=.8, seed=17)
    assert app.state.runner_config.sampling_topk_max == 32
    uvicorn.run.assert_called_once()


@pytest.mark.parametrize("mode,do_sample,expected", [
    ("online_pinned", False, "serving"),
    ("online_pinned", True, "serving"),
    ("offline", False, "greedy"),
    ("offline", True, "sm100"),
])
def test_qwen3_sampling_dispatch_and_buffers(monkeypatch, mode, do_sample, expected):
    """Build the real Qwen3 graph with CPU tensors and recording task methods."""
    mpk = MagicMock()
    mpk.mode, mpk.do_sample = mode, do_sample
    mpk.max_num_pages, mpk.page_size, mpk.world_size, mpk.mpi_rank = 4, 64, 1, 0
    mpk.max_num_batched_tokens, mpk.num_workers = 2, 2
    mpk.temperature, mpk.top_p, mpk.top_k = .8, .9, 20
    mpk.sampling_seed, mpk.sampling_topk_max = 42, 32
    mpk.meta_tensors = dict(input_tokens=torch.zeros(2, 1), output_tokens=torch.zeros(2, 1))
    mpk.new_tensor.side_effect = lambda dims, **kwargs: SimpleNamespace(dim=lambda i: dims[i])
    mpk.attach_input.side_effect = lambda torch_tensor, **kwargs: SimpleNamespace(
        dim=lambda i: torch_tensor.shape[i])
    builder = Qwen3Builder(mpk)
    builder.hidden_size, builder.intermediate_size = 64, 128
    builder.num_local_q_heads, builder.head_dim = 1, 64
    builder.fused_outdim_1, builder.fused_outdim_2 = 192, 256
    builder.vocab_size, builder.padded_vocab_size = 96, 128
    builder.position_embeddings = (torch.zeros(1, 16, 64), torch.zeros(1, 16, 64))
    builder.build_layers = lambda state: None
    original_full = torch.full
    monkeypatch.setattr(torch, "full", lambda *args, **kwargs: original_full(
        *args, **{**kwargs, "device": "cpu"}))
    builder.build_from_dict({"lm_head.weight": torch.zeros(96, 64),
                             "model.embed_tokens.weight": torch.zeros(96, 64),
                             "model.norm.weight": torch.ones(64)}, with_lm_head=True)
    names = {call.kwargs["name"] for call in mpk.new_tensor.call_args_list}
    assert mpk.serving_sampling_layer.call_count == int(expected == "serving")
    assert mpk.sampling_partial_layer.call_count == int(expected == "sm100")
    assert mpk.sampling_reduce_layer.call_count == int(expected == "sm100")
    assert mpk.argmax_partial_layer.call_count == int(expected == "greedy")
    assert mpk.argmax_reduce_layer.call_count == int(expected == "greedy")
    assert ("sampling_part_value" in names) == (expected == "sm100")
    assert ("argmax_part_value" in names) == (expected == "greedy")
    if expected == "sm100":
        assert mpk.sampling_reduce_layer.call_args.kwargs["seed"] == 42
        assert mpk.sampling_reduce_layer.call_args.kwargs["top_k"] == 20
