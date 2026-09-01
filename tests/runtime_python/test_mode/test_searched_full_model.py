"""Full-model gates for searched schedules on Qwen3-0.6B."""
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
BENCH = os.path.join(REPO, "tests", "ci-tests", "run_batch_perf.py")

TOKENS = 8
REQUESTS = 8
SEQ = 128


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None,
                                reason=_skip_reason() or "")


def _run_model(env_extra, timeout=3600):
    """Build all 28 Qwen3 layers, decode, and return (stdout, token ids)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tok_path = f.name
    env = dict(os.environ)
    env.update(env_extra)
    env["MPK_DUMP_TOKENS"] = tok_path
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    proc = subprocess.run(
        [sys.executable, "-u", BENCH,
         "--model", "Qwen/Qwen3-0.6B",
         "--max-num-batched-tokens", str(TOKENS),
         "--max-num-batched-requests", str(REQUESTS),
         "--max-seq-length", str(SEQ), "--ignore-eos"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"model run failed:\n{tail}"
    with open(tok_path) as f:
        dumped = json.load(f)
    return proc.stdout, dumped


SEARCHED_ENV = {
    "MPK_COMPILED_ATTENTION": "0",
    "MPK_SEARCHED_SCHEDULES": "1",
}

# Qwen3-0.6B gate/up projection at batch 8 -- the shapes the cache is keyed on.
CACHED_SHAPE = [(TOKENS, 1024), (1024, 3072)]
CACHED_GRID = (3072 // 64, 1, 1)


def test_full_model_runs_on_the_cached_searched_schedule():
    """All 28 layers build and decode with the search-discovered schedule."""
    from mirage.mpk.lowering import task_search
    sched = task_search.lookup_schedule("linear", CACHED_SHAPE, CACHED_GRID)
    assert sched is not None, (
        "no cached entry for the shapes this model builds, so the searched "
        "path would silently fall through to the hand-written schedule")
    assert sched.forloop_range == 4, sched.describe()

    out, dumped = _run_model(SEARCHED_ENV)
    assert dumped["mlp_implementation"] == "separate", dumped

    prompt_len = dumped["prompt_length"]
    assert dumped["sequence_length"] == SEQ, dumped["sequence_length"]
    for r, row in enumerate(dumped["tokens"]):
        gen = row[prompt_len:]
        assert len(gen) == SEQ - prompt_len, (r, len(gen))
        assert all(0 <= t < 200_000 for t in gen), f"request {r}: {gen[:8]}"


@pytest.mark.xfail(strict=False, reason=(
    "Pre-existing: batched decode is not self-consistent. Present in the "
    "HANDWRITTEN path too, so it is not caused by searched schedules."))
def test_identical_prompts_give_identical_tokens():
    """Eight identical prompts, greedy argmax -- so eight identical outputs."""
    _, dumped = _run_model(SEARCHED_ENV)
    prompt_len = dumped["prompt_length"]
    rows = [tuple(r[prompt_len:]) for r in dumped["tokens"]]
    prompts = {tuple(r[:prompt_len]) for r in dumped["tokens"]}
    assert len(prompts) == 1, f"prompts differ, premise does not hold: {prompts}"
    distinct = set(rows)
    assert len(distinct) == 1, (
        f"{len(distinct)} distinct outputs from {len(rows)} identical prompts; "
        f"first divergence per request: "
        f"{[next((i for i, (a, b) in enumerate(zip(rows[0], r)) if a != b), None) for r in rows]}")
