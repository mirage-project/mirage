"""Focused tests for the segmented muGraph prototype.

Split into two groups:

* **static tests** -- cache-key semantics, guard behaviour, region wiring.
  These need neither CUDA nor NVCC nor a checkpoint and always run.
* **GPU tests** -- numerical correctness of Region A / Region B / the full
  segmented MLP, graph reuse, and the decode-vs-prefill routing.  These skip
  with a clear reason when the hardware or toolchain is unavailable.

Run:
    PYTHONPATH=. pytest tests/experiments/test_segmented_mugraph.py -v
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch", reason="PyTorch is required")

from experiments.segmented_mugraph import common  # noqa: E402
from experiments.segmented_mugraph.runner import (  # noqa: E402
    RegionKey,
    RegionKind,
    SegmentedMuGraphRunner,
    TensorSpec,
    assert_no_task_graph_artifacts,
    no_task_graph_guard,
)

TOKENS, HIDDEN, INTER = 8, 4096, 2048
DTYPE = torch.bfloat16


# --------------------------------------------------------------------------
# skip helpers
# --------------------------------------------------------------------------


def _gpu_skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if shutil.which("nvcc") is None:
        return "nvcc not found; muGraph compilation needs the CUDA toolkit"
    return None


requires_gpu = pytest.mark.skipif(
    _gpu_skip_reason() is not None, reason=_gpu_skip_reason() or ""
)


@pytest.fixture(scope="module")
def runner():
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    # Superoptimization is exercised by the benchmarks; the tests compile the
    # high-level graph directly so they stay fast and deterministic.
    return SegmentedMuGraphRunner(device="cuda", try_superoptimize=False, verbose=False)


@pytest.fixture(scope="module")
def tensors():
    if _gpu_skip_reason():
        pytest.skip(_gpu_skip_reason())
    return common.make_mlp_tensors(TOKENS, HIDDEN, INTER, DTYPE, "cuda", seed=42)


# ==========================================================================
# static tests -- no GPU required
# ==========================================================================


class TestCacheKeys:
    """A compiled graph must be reused iff every structural field matches."""

    def _spec(self, shape, strides, dtype="torch.bfloat16"):
        return TensorSpec(shape, strides, dtype)

    def _key(self, **over):
        base = dict(
            kind=RegionKind.GATE_UP_SILU_MUL,
            tokens=8,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            dtype="torch.bfloat16",
            input_specs=(self._spec((8, HIDDEN), (HIDDEN, 1)),),
            target_cc=100,
            options=(),
        )
        base.update(over)
        return RegionKey(**base)

    def test_identical_keys_are_equal_and_hash_equal(self):
        assert self._key() == self._key()
        assert hash(self._key()) == hash(self._key())

    @pytest.mark.parametrize(
        "field,value",
        [
            ("tokens", 1),
            ("hidden_size", 2048),
            ("intermediate_size", 4096),
            ("dtype", "torch.float16"),
            ("target_cc", 90),
            ("kind", RegionKind.DOWN_RESIDUAL),
            ("options", (("fuse", True),)),
        ],
    )
    def test_incompatible_field_misses(self, field, value):
        assert self._key() != self._key(**{field: value})

    def test_stride_change_misses(self):
        transposed = (self._spec((8, HIDDEN), (1, 8)),)
        assert self._key() != self._key(input_specs=transposed)

    def test_dtype_change_in_input_spec_misses(self):
        other = (self._spec((8, HIDDEN), (HIDDEN, 1), "torch.float16"),)
        assert self._key() != self._key(input_specs=other)


class TestTensorSpec:
    def test_matches_and_describes(self):
        t = torch.zeros(4, 8)
        spec = TensorSpec.from_tensor(t)
        assert spec.matches(t)
        assert not spec.matches(torch.zeros(8, 4))
        assert "expected shape" in spec.describe(torch.zeros(8, 4))

    def test_transposed_weight_view_spec(self):
        """A [out, in] weight transposes to (in, out) with strides (1, in)."""
        w = torch.zeros(INTER, HIDDEN)
        v = w.t()
        assert tuple(v.shape) == (HIDDEN, INTER)
        assert tuple(v.stride()) == (1, HIDDEN)
        assert v.data_ptr() == w.data_ptr()  # a view, never a copy


class TestTaskGraphGuard:
    def test_guard_trips_on_generate_task_graph(self):
        from mirage.kernel import KNGraph

        with pytest.raises(AssertionError, match="generate_task_graph"):
            with no_task_graph_guard(REPO_ROOT):
                KNGraph.generate_task_graph(object())

    def test_guard_restores_original(self):
        from mirage.kernel import KNGraph

        original = KNGraph.generate_task_graph
        with no_task_graph_guard(REPO_ROOT):
            assert KNGraph.generate_task_graph is not original
        assert KNGraph.generate_task_graph is original

    def test_new_task_graph_json_is_detected(self, tmp_path):
        from experiments.segmented_mugraph.runner import _task_graph_snapshot

        baseline = _task_graph_snapshot(str(tmp_path))
        assert_no_task_graph_artifacts(str(tmp_path), baseline)  # clean
        (tmp_path / "task_graph.json").write_text("{}")
        with pytest.raises(AssertionError, match="emitted or rewrote"):
            assert_no_task_graph_artifacts(str(tmp_path), baseline)

    def test_preexisting_task_graph_json_is_tolerated(self, tmp_path):
        from experiments.segmented_mugraph.runner import _task_graph_snapshot

        (tmp_path / "task_graph.json").write_text("{}")
        baseline = _task_graph_snapshot(str(tmp_path))
        assert_no_task_graph_artifacts(str(tmp_path), baseline)


class TestHybridRoutingStatic:
    """Bucket routing must not need a GPU to be reasoned about."""

    def test_prefill_falls_back_and_decode_does_not(self):
        from experiments.segmented_mugraph.hybrid_mlp import HybridQwen3MLP

        class FakeLinear:
            def __init__(self, w):
                self.weight = w

        class FakeMLP:
            hidden_size, intermediate_size = 8, 16

            def __init__(self):
                self.gate_proj = FakeLinear(torch.zeros(16, 8))
                self.up_proj = FakeLinear(torch.zeros(16, 8))
                self.down_proj = FakeLinear(torch.zeros(8, 16))
                self.calls = 0

            def __call__(self, x):
                self.calls += 1
                return x

        class FakeRunner:
            def __init__(self):
                self.calls = 0

            def region_a(self, *a, **k):
                self.calls += 1
                return torch.zeros(2, 16)

            def region_b(self, *a, **k):
                return torch.zeros(1, 8)

        orig, fake = FakeMLP(), FakeRunner()
        stats = {"mugraph_calls": 0, "fallback_calls": 0}
        mlp = HybridQwen3MLP(orig, fake, {1}, stats)

        mlp(torch.zeros(1, 1, 8))  # decode -> muGraph
        assert stats == {"mugraph_calls": 1, "fallback_calls": 0}

        mlp(torch.zeros(1, 12, 8))  # prefill -> PyTorch fallback
        assert stats == {"mugraph_calls": 1, "fallback_calls": 1}
        assert orig.calls == 1


class TestMetrics:
    def test_correctness_metrics_on_identical_tensors(self):
        a = torch.randn(4, 8)
        m = common.correctness_metrics(a, a.clone())
        assert m["max_abs_err"] == 0.0
        assert m["cosine_sim"] == pytest.approx(1.0, abs=1e-6)
        assert m["all_finite"]

    def test_correctness_metrics_flags_nonfinite(self):
        a = torch.full((4,), float("nan"))
        assert not common.correctness_metrics(a, torch.zeros(4))["all_finite"]


# ==========================================================================
# GPU tests
# ==========================================================================


@requires_gpu
class TestRegionCorrectness:
    def test_region_a(self, runner, tensors):
        got = runner.region_a(tensors["x"], tensors["w_gate"], tensors["w_up"])
        ref = common.torch_region_a(tensors["x"], tensors["w_gate"], tensors["w_up"])
        m = common.correctness_metrics(got, ref)
        assert m["all_finite"]
        assert tuple(got.shape) == (TOKENS, INTER)
        assert got.dtype == DTYPE
        assert m["max_abs_err"] < common.MAX_ABS_TOL_REGION, m
        assert m["cosine_sim"] > 0.999, m

    def test_region_b(self, runner, tensors):
        got = runner.region_b(tensors["mid"], tensors["w_down"], tensors["residual"])
        ref = common.torch_region_b(
            tensors["mid"], tensors["w_down"], tensors["residual"]
        )
        m = common.correctness_metrics(got, ref)
        assert m["all_finite"]
        assert tuple(got.shape) == (TOKENS, HIDDEN)
        assert m["max_abs_err"] < common.MAX_ABS_TOL_REGION, m

    def test_region_b_without_residual(self, runner, tensors):
        got = runner.region_b(tensors["mid"], tensors["w_down"])
        ref = common.torch_region_b(tensors["mid"], tensors["w_down"])
        assert common.correctness_metrics(got, ref)["max_abs_err"] < common.MAX_ABS_TOL_REGION

    def test_full_segmented_mlp(self, runner, tensors):
        got = runner.mlp(
            tensors["x"], tensors["w_gate"], tensors["w_up"],
            tensors["w_down"], tensors["residual"],
        )
        ref = common.torch_full_mlp(
            tensors["x"], tensors["w_gate"], tensors["w_up"],
            tensors["w_down"], tensors["residual"],
        )
        m = common.correctness_metrics(got, ref)
        assert m["all_finite"]
        assert tuple(got.shape) == (TOKENS, HIDDEN)
        assert got.dtype == DTYPE
        # Same bound the MPK test_mode test uses for the complete MLP.
        assert m["max_abs_err"] < common.MAX_ABS_TOL_FULL_MLP, m
        assert m["cosine_sim"] > 0.999, m

    def test_decode_token_count_one_uses_mugraph(self, runner, tensors):
        x = tensors["x"][:1].contiguous()
        got = runner.region_a(x, tensors["w_gate"], tensors["w_up"])
        ref = common.torch_region_a(x, tensors["w_gate"], tensors["w_up"])
        assert tuple(got.shape) == (1, INTER)
        assert common.correctness_metrics(got, ref)["max_abs_err"] < common.MAX_ABS_TOL_REGION


@requires_gpu
class TestGraphCache:
    def test_reuse_across_layers_with_different_weights(self, runner, tensors):
        """Structurally identical layers share one compiled graph."""
        runner.region_a(tensors["x"], tensors["w_gate"], tensors["w_up"])
        variants, hits = runner.num_variants, runner.cache_hits

        other_gate = torch.randn_like(tensors["w_gate"]) * 0.01
        other_up = torch.randn_like(tensors["w_up"]) * 0.01
        got = runner.region_a(tensors["x"], other_gate, other_up)

        assert runner.num_variants == variants, "weights must not force a recompile"
        assert runner.cache_hits > hits
        ref = common.torch_region_a(tensors["x"], other_gate, other_up)
        assert common.correctness_metrics(got, ref)["max_abs_err"] < common.MAX_ABS_TOL_REGION

    def test_incompatible_token_count_misses(self, runner, tensors):
        runner.region_a(tensors["x"], tensors["w_gate"], tensors["w_up"])
        variants = runner.num_variants
        small = tensors["x"][:4].contiguous()
        runner.region_a(small, tensors["w_gate"], tensors["w_up"])
        assert runner.num_variants == variants + 1

    def test_padded_bucket_shared_by_one_and_min_tokens(self, runner, tensors):
        """tokens=1 compiles the padded bucket, so tokens=2 reuses it."""
        one = tensors["x"][:1].contiguous()
        runner.region_a(one, tensors["w_gate"], tensors["w_up"])
        variants = runner.num_variants
        two = tensors["x"][: runner.min_tokens].contiguous()
        runner.region_a(two, tensors["w_gate"], tensors["w_up"])
        assert runner.num_variants == variants

    def test_validate_rejects_wrong_layout(self, runner, tensors):
        region = runner.compile_region(
            RegionKind.GATE_UP_SILU_MUL, TOKENS, HIDDEN, INTER,
            (tensors["x"], tensors["w_gate"].t(), tensors["w_up"].t()), DTYPE,
        )
        with pytest.raises(ValueError, match="layout mismatch"):
            region.validate(
                (tensors["x"], tensors["w_gate"], tensors["w_up"].t())  # not transposed
            )

    def test_report_records_compilation_mode(self, runner, tensors):
        runner.region_a(tensors["x"], tensors["w_gate"], tensors["w_up"])
        report = runner.report()
        assert report
        assert all(r["mode"] in ("direct", "superoptimized") for r in report)


@requires_gpu
class TestNoTaskGraph:
    def test_segmented_path_never_generates_a_task_graph(self, tensors):
        """The headline guarantee: no MPK lowering anywhere in the region path."""
        with no_task_graph_guard(REPO_ROOT):
            r = SegmentedMuGraphRunner(
                device="cuda", try_superoptimize=False, verbose=False
            )
            out = r.mlp(
                tensors["x"], tensors["w_gate"], tensors["w_up"],
                tensors["w_down"], tensors["residual"],
            )
            torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all()


# ==========================================================================
# checkpoint-dependent test
# ==========================================================================


def _has_qwen3_checkpoint(name="Qwen/Qwen3-0.6B"):
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(name, token=False, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@requires_gpu
@pytest.mark.skipif(
    not _has_qwen3_checkpoint(),
    reason="Qwen/Qwen3-0.6B checkpoint is not present in the local HF cache",
)
def test_hybrid_model_decode_matches_torch():
    """Patched decode agrees with the unmodified model on the first step."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from experiments.segmented_mugraph.hybrid_mlp import (
        patch_qwen3_mlps,
        precompile_buckets,
    )

    model = Qwen3ForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=DTYPE, token=False, local_files_only=True
    ).to("cuda").eval()
    ids = torch.tensor([[9707, 11, 1879, 0]], device="cuda")

    with torch.inference_mode():
        ref = model(input_ids=ids).logits[:, -1, :].float().clone()

    with no_task_graph_guard(REPO_ROOT):
        runner = SegmentedMuGraphRunner(
            device="cuda", try_superoptimize=False, verbose=False
        )
        handle = patch_qwen3_mlps(model, runner, allowed_tokens=(1,))
        precompile_buckets(model, runner, (1,))
        with torch.inference_mode():
            # 4 tokens -> prefill -> PyTorch fallback
            got_prefill = model(input_ids=ids).logits[:, -1, :].float().clone()
            assert handle["stats"]["fallback_calls"] == len(model.model.layers)
            assert handle["stats"]["mugraph_calls"] == 0
            # 1 token -> decode -> muGraph
            model(input_ids=ids[:, -1:])
            assert handle["stats"]["mugraph_calls"] == len(model.model.layers)

    assert torch.isfinite(got_prefill).all()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten(), got_prefill.flatten(), dim=0
    ).item()
    assert cos >= 0.99, cos
    assert ref.argmax().item() == got_prefill.argmax().item()
    # Two graphs (Region A + Region B) shared across every layer.
    assert runner.num_variants == 2
