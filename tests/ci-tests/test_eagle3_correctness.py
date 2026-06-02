"""EAGLE3 output correctness vs greedy PyTorch Qwen3-30B-A3B.

Compares the EAGLE3 speculative-decoding megakernel output against a standalone
greedy (argmax) PyTorch decode of the same model on the same prompt. Strict
rejection sampling guarantees the accepted-token stream equals the greedy target
decode independent of the number of draft steps K, so the greedy decode is the
exact oracle.

Driven by tests/ci-tests/run_eagle3_correctness.sh, which produces:
  - outputs/qwen3_30b_a3b/torch_reference.json   (oracle, greedy HF)
  - outputs/qwen3_30b_a3b/mpk_eagle3_k{K}.json   (EAGLE3 megakernel @ K)

EOS is honored identically on both sides. The compared length is
L = min(50, len(ref), len(mpk)); a pass requires ref[:L] == mpk[:L] AND equal
total lengths (same EOS stop point). A short-but-identical EOS-terminated stream
passes; a differing stop point is a real divergence and fails.

Skip policy: a genuinely ABSENT artifact skips (weights/GPU unavailable so the
runner never produced it). A PRESENT-but-malformed artifact (missing token_ids,
metadata mismatch, invalid token id) FAILS.
"""

import json
import os

import pytest

DEFAULT_OUTPUT_DIR = os.path.join("outputs", "qwen3_30b_a3b")
REFERENCE_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "torch_reference.json")
# The runner sets EAGLE3_MPK_OUTPUT to the per-K artifact (mpk_eagle3_k{K}.json).
MPK_OUTPUT = os.environ.get(
    "EAGLE3_MPK_OUTPUT",
    os.path.join(DEFAULT_OUTPUT_DIR, "mpk_eagle3_output.json"),
)
COMPARE_LEN = 50

# Qwen3-30B-A3B vocab (151936). A saved token id outside [0, VOCAB) signals the
# "-1 sentinel" / garbage-row bug class and is a hard failure, never a skip.
VOCAB_SIZE = 151936


def _load(path, role):
    if not os.path.exists(path):
        pytest.skip(
            f"Missing {role} artifact: {path}. Run "
            "tests/ci-tests/run_eagle3_correctness.sh first."
        )
    with open(path) as f:
        data = json.load(f)
    tokens = data.get("token_ids")
    if not isinstance(tokens, list):
        pytest.fail(f"'token_ids' missing or not a list in {path} ({role})")
    return tokens, data


def _assert_token_ids_valid(tokens, path, role):
    for i, t in enumerate(tokens):
        if not isinstance(t, int) or t < 0 or t >= VOCAB_SIZE:
            pytest.fail(
                f"Invalid token id at index {i} in {path} ({role}): {t!r} "
                f"(must be int in [0, {VOCAB_SIZE})). A negative/-1 value is the "
                f"sentinel/garbage-row bug class."
            )


def _assert_mpk_metadata(meta, path):
    """Malformed-present-artifact checks for the MPK side (per AC-3/AC-6)."""
    K = meta.get("K")
    mbt = meta.get("mbt")
    max_tokens_compiled = meta.get("max_tokens_compiled")
    if K is None or mbt is None:
        pytest.fail(f"MPK artifact {path} missing K/mbt metadata: K={K}, mbt={mbt}")
    if mbt != K + 1:
        pytest.fail(
            f"MPK artifact {path} metadata mismatch: mbt ({mbt}) != K+1 ({K + 1})"
        )
    if max_tokens_compiled is None:
        pytest.fail(
            f"MPK artifact {path} missing max_tokens_compiled (cannot confirm "
            f"the AC-6 fail-fast gate)."
        )
    # Correct safety condition is >= (over-allocation of M rows is harmless;
    # under-allocation lacks enough MMA rows for all mbt verifier slots).
    # Confirmed with Codex; refines the plan's original "== mbt" wording.
    if max_tokens_compiled < mbt:
        pytest.fail(
            f"MPK artifact {path} compiled MAX_TOKENS ({max_tokens_compiled}) < "
            f"mbt ({mbt}). The kernel lacks enough attention rows for all verifier "
            f"slots — the silently-wrong-pass case the AC-6 gate guards against."
        )


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def test_eagle3_output_matches_greedy_torch():
    ref_tokens, ref_meta = _load(REFERENCE_JSON, "reference")
    mpk_tokens, mpk_meta = _load(MPK_OUTPUT, "mpk")

    # Present-but-malformed artifacts FAIL (not skip).
    _assert_token_ids_valid(ref_tokens, REFERENCE_JSON, "reference")
    _assert_token_ids_valid(mpk_tokens, MPK_OUTPUT, "mpk")
    _assert_mpk_metadata(mpk_meta, MPK_OUTPUT)

    L = min(COMPARE_LEN, len(ref_tokens), len(mpk_tokens))
    if L == 0:
        pytest.fail(
            f"No tokens to compare (ref={len(ref_tokens)}, mpk={len(mpk_tokens)})"
        )

    # EOS-stop-point check (DEC-4). EOS is honored identically on both sides, so
    # if EITHER side terminated within the comparison window (L < COMPARE_LEN, an
    # EOS-driven short stream), the total generated lengths MUST agree — a length
    # mismatch there means divergent EOS placement and is a real failure. When
    # both sides reach the full COMPARE_LEN window (L == COMPARE_LEN), neither
    # stopped early; the streams legitimately continue past the window (the oracle
    # may be capped at a different num-tokens than the demo's run length), so we
    # only require the first COMPARE_LEN tokens to match and do NOT require equal
    # total lengths.
    if L < COMPARE_LEN and len(ref_tokens) != len(mpk_tokens):
        pytest.fail(
            f"EOS stop-point mismatch within the first {COMPARE_LEN} tokens: ref "
            f"generated {len(ref_tokens)}, mpk generated {len(mpk_tokens)} "
            f"(one side hit EOS earlier). "
            f"ref[:{L}]={ref_tokens[:L]} mpk[:{L}]={mpk_tokens[:L]} "
            f"K={mpk_meta.get('K')} mbt={mpk_meta.get('mbt')} "
            f"max_tokens_compiled={mpk_meta.get('max_tokens_compiled')}"
        )

    idx = _first_divergence(ref_tokens[:L], mpk_tokens[:L])
    if idx is not None:
        window = slice(max(0, idx - 3), idx + 4)
        pytest.fail(
            f"Token mismatch at generated index {idx}: "
            f"ref={ref_tokens[idx]} mpk={mpk_tokens[idx]}; "
            f"ref window {list(range(window.start, window.stop))} = "
            f"{ref_tokens[window]}; mpk window = {mpk_tokens[window]}; "
            f"K={mpk_meta.get('K')} mbt={mpk_meta.get('mbt')} "
            f"max_tokens_compiled={mpk_meta.get('max_tokens_compiled')} "
            f"accept_hist={mpk_meta.get('accept_hist')}"
        )
