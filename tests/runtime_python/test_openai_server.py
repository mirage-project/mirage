"""Fast HTTP checks; CUDA sampling is tested by test_serving_sampling_layer.py."""

import json

import pytest
from fastapi.testclient import TestClient

from mirage.engine.launch_server import app


class FakeEngine:
    def __init__(self):
        self.tokenizer_manager = self
        self.params = None

    def tokenize_raw(self, prompt):
        return [1, 2]

    def tokenize_messages(self, messages):
        return [1, 2]

    def decode(self, ids):
        return "".join({3: "hello", 4: " world"}[token] for token in ids)

    def submit(self, ids, **kwargs):
        self.params = kwargs["sampling_params"]
        return iter([(3, None), (4, "length")])


@pytest.fixture
def client(monkeypatch):
    engine = FakeEngine()
    for key, value in dict(engine=engine, served_model="test", sampling_defaults={},
                           request_timeout=120).items():
        monkeypatch.setattr(app.state, key, value, raising=False)
    client = TestClient(app)
    client.engine = engine
    yield client
    client.close()


@pytest.mark.parametrize("chat", [False, True])
def test_completion(client, chat):
    path = "/v1/chat/completions" if chat else "/v1/completions"
    prompt = {"messages": [{"role": "user", "content": "hi"}]} if chat else {"prompt": "hi"}
    response = client.post(path, json=dict(model="test", **prompt, temperature=.7,
                                            top_k=3, top_p=.8, seed=17))
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert (choice["message"]["content"] if chat else choice["text"]) == "hello world"
    assert choice["finish_reason"] == "length"
    assert response.json()["usage"] == dict(prompt_tokens=2, completion_tokens=2,
                                              total_tokens=4)
    params = client.engine.params
    assert (params.temperature, params.top_k, params.top_p, params.seed) == (.7, 3, .8, 17)


@pytest.mark.parametrize("chat", [False, True])
def test_stream(client, chat):
    path = "/v1/chat/completions" if chat else "/v1/completions"
    prompt = {"messages": [{"role": "user", "content": "hi"}]} if chat else {"prompt": "hi"}
    response = client.post(path, json=dict(model="test", **prompt, stream=True,
                                            stream_options={"include_usage": True}))
    assert response.status_code == 200, response.text
    lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines.pop() == "[DONE]"
    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["usage"]["completion_tokens"] == 2
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    if chat:
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(chunk["choices"][0]["delta"].get("content", "")
                       for chunk in chunks if chunk["choices"])
    else:
        text = "".join(chunk["choices"][0]["text"]
                       for chunk in chunks if chunk["choices"])
    assert text == "hello world"


@pytest.mark.parametrize("field,value", [
    ("temperature", -1), ("top_p", 0), ("top_p", 1.1),
    ("top_k", -1), ("top_k", 1.5), ("seed", -1),
    ("logit_bias", {"3": 101}),
])
def test_invalid_sampling_params(client, field, value):
    response = client.post("/v1/completions", json={
        "model": "test", "prompt": "hi", field: value})
    assert response.status_code == 400
    assert response.json()["error"]["param"].startswith(field)
    assert client.engine.params is None


def test_wrong_model(client):
    response = client.post("/v1/completions", json={"model": "other", "prompt": "hi"})
    assert response.status_code == 404
    assert response.json()["error"]["param"] == "model"


def test_sampling_defaults(client, monkeypatch):
    monkeypatch.setattr(app.state, "sampling_defaults", {"temperature": .8, "top_k": 2})
    client.post("/v1/completions", json={"model": "test", "prompt": "hi"})
    assert (client.engine.params.temperature, client.engine.params.top_k) == (.8, 2)
    client.post("/v1/completions", json={"model": "test", "prompt": "hi",
                                                 "temperature": 0, "top_k": 1})
    assert (client.engine.params.temperature, client.engine.params.top_k) == (0, 1)
