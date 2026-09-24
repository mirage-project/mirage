"""Synthetic-logit checks for the CUDA sampler used by serving_sampling_layer."""

from collections import Counter
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from mirage.engine.sampling import SamplingParams


REAL_VOCAB = 8
LOGITS = np.array([-1, -1.4, -1.8, -2.2] + [-30] * 4 + [0] * 8,
                  dtype=np.float32)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    if not torch.cuda.is_available() or not shutil.which("nvcc"):
        pytest.skip("CUDA and nvcc are required")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("serving_sampling") / "sampling.so"
    major, minor = torch.cuda.get_device_capability()
    subprocess.run([
        "nvcc", "-O3", "-std=c++17", f"-arch=sm_{major}{minor}",
        "-shared", "-Xcompiler=-fPIC", "-use_fast_math",
        f"-I{root / 'include'}", str(root / "tests/engine/sampling_cuda.cu"),
        "-o", str(output),
    ], check=True)
    kernel = ctypes.CDLL(str(output)).sample_test
    kernel.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    def draw(settings, seeds=range(512), history=(), prompt_len=0):
        seeds = list(seeds)
        logits = np.tile(LOGITS, (len(seeds), 1))
        configs = np.array([
            SamplingParams(**settings, seed=seed).pack(1, 32, REAL_VOCAB, [])
            for seed in seeds], dtype=np.int64)
        history = np.asarray(history, dtype=np.int64)
        output_ids = np.empty(len(seeds), dtype=np.int64)
        error = kernel(0, logits.ctypes.data, configs.ctypes.data,
                       history.ctypes.data, len(history), prompt_len, 0,
                       len(LOGITS), len(seeds), output_ids.ctypes.data, 128, 1)
        assert error == 0, f"CUDA error {error}"
        return output_ids.tolist()

    return draw


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
    # BF16 is the serving logits dtype. Only the first eight tokens are real;
    # the padded logits are zero and would win if vocab_size were ignored.
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


def test_prompt_and_generated_penalties(sample):
    assert sample({"temperature": 0, "repetition_penalty": 2}, [1], [0], 1) == [1]
    assert sample({"temperature": 0, "frequency_penalty": 2,
                   "presence_penalty": 2}, [1], [0], 1) == [0]
    assert sample({"temperature": 0, "frequency_penalty": 2}, [1], [0], 0) == [1]
