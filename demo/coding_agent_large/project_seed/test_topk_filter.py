"""Correctness + performance tests for batched_topk_filter.

Correctness is checked against a slow reference (kept here as ground truth) on
distinct-valued inputs (no boundary ties). Two tests specifically require the
numpy optimization: the result must be an ndarray, and a large input must run
fast — both fail for the pure-Python reference.
"""
import time

import numpy as np

from topk_filter import batched_topk_filter


def _reference(scores, k):
    out = []
    for row in scores:
        pairs = [(i, x) for i, x in enumerate(row)]
        pairs.sort(key=lambda p: p[1], reverse=True)
        keep = set(i for i, _ in pairs[:k])
        out.append([x if i in keep else float("-inf") for i, x in enumerate(row)])
    return out


def _check(scores, k):
    got = np.asarray(batched_topk_filter(scores, k), dtype=float)
    ref = np.asarray(_reference(scores, k), dtype=float)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_basic():
    _check([[0.1, 0.9, 0.3, 0.7], [5.0, 1.0, 2.0, 4.0]], 2)


def test_k_equals_width_keeps_all():
    got = np.asarray(batched_topk_filter([[3.0, 1.0, 2.0]], 3), dtype=float)
    assert np.array_equal(got, np.asarray([[3.0, 1.0, 2.0]], dtype=float))


def test_random_matches_reference():
    rng = np.random.default_rng(0)
    scores = rng.permutation(64 * 200).reshape(64, 200).astype(float)  # distinct
    for k in (1, 5, 50, 199):
        _check(scores.tolist(), k)


def test_returns_ndarray():
    out = batched_topk_filter([[1.0, 2.0, 3.0]], 1)
    assert isinstance(out, np.ndarray), "optimize with numpy: return an ndarray"


def test_performance_is_vectorized():
    rng = np.random.default_rng(1)
    scores = rng.permutation(150 * 10000).reshape(150, 10000).astype(float)

    t = time.perf_counter()
    batched_topk_filter(scores, 50)
    dt = time.perf_counter() - t

    # Same input through the slow reference, so the output reports a speedup.
    ref_in = scores.tolist()
    t = time.perf_counter()
    _reference(ref_in, 50)
    dt_ref = time.perf_counter() - t

    print(f"\n[PERF] batched_topk_filter {scores.shape} k=50: "
          f"impl {dt * 1000:7.1f} ms  vs  reference {dt_ref * 1000:7.1f} ms  "
          f"({dt_ref / dt:5.1f}x speedup)")
    assert dt < 0.15, f"too slow ({dt:.3f}s) — vectorize with numpy, no Python loops"
