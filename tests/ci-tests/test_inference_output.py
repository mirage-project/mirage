import json
import os
import pytest

DEFAULT_OUTPUT_DIR = os.path.join("outputs", "qwen3")
TORCH_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, "torch_output.json")
MPK_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, "mpk_output.json")
NUM_TOKENS_TO_COMPARE = 50


def _load_tokens(path: str):
    if not os.path.exists(path):
        pytest.fail(f"Missing output file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    tokens = data.get("token_ids")
    if not isinstance(tokens, list):
        pytest.fail(f"'token_ids' missing or not a list in {path}")
    return tokens, data


def test_qwen3_torch_vs_mpk_tokens():
    torch_tokens, torch_meta = _load_tokens(TORCH_OUTPUT)
    mpk_tokens, mpk_meta = _load_tokens(MPK_OUTPUT)

    n = min(NUM_TOKENS_TO_COMPARE, len(torch_tokens), len(mpk_tokens))
    if n == 0:
        pytest.fail(f"No tokens to compare (torch={len(torch_tokens)}, mpk={len(mpk_tokens)})")

    torch_slice = torch_tokens[:n]
    mpk_slice = mpk_tokens[:n]

    if torch_slice != mpk_slice:
        # Find first mismatch to make debugging easier
        mismatch_idx = next((i for i, (a, b) in enumerate(zip(torch_slice, mpk_slice)) if a != b), None)
        pytest.fail(
            f"Token mismatch at position {mismatch_idx} (0-based): "
            f"torch={torch_slice[mismatch_idx]}, mpk={mpk_slice[mismatch_idx]}; "
            f"compared first {n} tokens. "
            f"torch_generate_len={torch_meta.get('generate_length')}, "
            f"mpk_generate_len={mpk_meta.get('generate_length')}"
        )


def test_qwen3_perf_comparison():
    """Print Torch vs MPK latency comparison (informational, never fails)."""
    _, torch_meta = _load_tokens(TORCH_OUTPUT)
    _, mpk_meta = _load_tokens(MPK_OUTPUT)

    torch_lat = torch_meta.get("latency_ms_per_token")
    mpk_lat = mpk_meta.get("latency_ms_per_token")
    torch_len = torch_meta.get("generate_length", "?")
    mpk_len = mpk_meta.get("generate_length", "?")

    if torch_lat is None or mpk_lat is None:
        pytest.skip("latency_ms_per_token missing in output JSON")

    speedup = torch_lat / mpk_lat if mpk_lat > 0 else float("inf")

    print("")
    print("==================== Performance Comparison ====================")
    print(f"  Torch:  {torch_lat:.3f} ms/token  (generated {torch_len} tokens)")
    print(f"  MPK:    {mpk_lat:.3f} ms/token  (generated {mpk_len} tokens)")
    print(f"  Speedup: {speedup:.2f}x")
    print("===============================================================")
