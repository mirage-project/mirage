"""Synthetic logits through the production online_pinned serving sampling task."""

from collections import Counter
from types import SimpleNamespace
import shutil

import mirage
import numpy as np
import pytest
import torch

from mirage.engine.model_runner import ModelRunner
from mirage.engine.sampling import SamplingParams
from mirage.mpk.online_pinned_runtime import OnlinePinnedRuntime
from mirage.mpk.persistent_kernel import PersistentKernel


REAL_VOCAB = 8
LOGITS = np.array([-1, -1.4, -1.8, -2.2] + [-30] * 4 + [0] * 8,
                  dtype=np.float32)


@pytest.fixture(scope="module")
def sample():
    if not torch.cuda.is_available() or not shutil.which("nvcc"):
        pytest.skip("CUDA and nvcc are required")

    device = torch.cuda.current_device()
    workers, schedulers = mirage.get_configurations_from_gpu(device)
    capacity = 32
    config = SimpleNamespace(max_num_batched_requests=capacity,
                             max_num_batched_tokens=capacity,
                             max_seq_length=16, max_num_pages=capacity,
                             pinned_ring_capacity=64)
    meta = ModelRunner._allocate_meta_tensors(config)
    params = PersistentKernel.get_default_init_parameters()
    params.update(mode="online_pinned", test_mode=True, num_workers=workers,
                  num_local_schedulers=schedulers,
                  max_num_batched_requests=capacity,
                  max_num_batched_tokens=capacity,
                  max_seq_length=config.max_seq_length,
                  max_num_pages=config.max_num_pages, page_size=16,
                  pinned_ring_capacity=config.pinned_ring_capacity,
                  meta_tensors=meta)
    pk = PersistentKernel(**params)
    logits = torch.tensor(LOGITS, dtype=torch.bfloat16, device=device).repeat(capacity, 1)
    output = meta["output_tokens"]
    pk.serving_sampling_layer(pk.attach_input(logits, name="logits"),
                              pk.attach_input(output, name="output"))
    pk.compile()

    adapter = SimpleNamespace(metadata=SimpleNamespace(mode="online_pinned"),
                              pinned_ring_capacity=config.pinned_ring_capacity,
                              total_num_requests=capacity, persistent_kernel=pk,
                              **meta)
    runtime = OnlinePinnedRuntime(adapter)
    runtime.reset()
    runtime.start()
    pk()
    next_rid = 0

    def draw(settings, seeds=range(512), history=(), prompt_len=0, max_new_tokens=1):
        nonlocal next_rid
        # The online path requires a nonempty prompt.
        prompt = list(history[:prompt_len]) or [REAL_VOCAB - 1]
        seeds = list(seeds)
        requests = []
        for seed in seeds:
            rid = next_rid
            next_rid += 1
            sampling = SamplingParams(**settings, seed=seed,
                                      max_new_tokens=max_new_tokens)
            payload = sampling.pack(len(prompt), config.max_seq_length, REAL_VOCAB, [])
            runtime.submit(rid, torch.tensor(prompt, dtype=torch.int64),
                           generation_config=payload)
            requests.append(rid)
        result = []
        for rid in requests:
            row, step = runtime.wait_for_request(rid, timeout=60)
            assert runtime.finish_reason(row) == "length"
            result.append(int(runtime.read_tokens_at_row(row, step)[-1]))
            assert runtime.release_request(rid)
        return result

    try:
        yield draw
    finally:
        runtime.request_shutdown()
        pk.wait()
        runtime.stop()
        pk.finalize()


def test_greedy_and_seed(sample):
    assert sample({"temperature": 0}, range(8)) == [0] * 8
    assert sample({"temperature": 1, "top_k": 1}, range(8)) == [0] * 8
    assert sample({"temperature": 1}, [17]) == sample({"temperature": 1}, [17])


@pytest.mark.parametrize("settings", [
    {"temperature": 1},
    {"temperature": .5},
    {"temperature": 1, "top_k": 3},
    {"temperature": 1, "top_p": .75},
    {"temperature": 1, "top_k": 3, "top_p": .75},
])
def test_sampling_distribution(sample, settings):
    scores = torch.tensor(LOGITS[:REAL_VOCAB], dtype=torch.bfloat16).float()
    scores /= settings["temperature"]
    order = torch.argsort(scores, descending=True)[:settings.get("top_k", REAL_VOCAB)]
    probabilities = torch.softmax(scores[order], 0)
    if "top_p" in settings:
        count = int((probabilities.cumsum(0) >= settings["top_p"]).nonzero()[0]) + 1
        order, probabilities = order[:count], probabilities[:count]
        probabilities /= probabilities.sum()
    expected = torch.zeros(REAL_VOCAB)
    expected[order] = probabilities

    draws = sample(settings)
    counts = Counter(draws)
    assert set(counts) <= set(range(REAL_VOCAB))
    for token, probability in enumerate(expected.tolist()):
        actual = counts[token] / len(draws)
        tolerance = 5 * (probability * (1 - probability) / len(draws)) ** .5 + .01
        assert abs(actual - probability) < tolerance, (settings, token, actual, probability)


def test_prompt_penalties(sample):
    assert sample({"temperature": 0, "repetition_penalty": 2}, [1], [0], 1) == [1]
    assert sample({"temperature": 0, "frequency_penalty": 2,
                   "presence_penalty": 2}, [1], [0], 1) == [0]
    assert sample({"temperature": 0, "frequency_penalty": 2}, [1], [0], 1,
                  max_new_tokens=2) == [1]
