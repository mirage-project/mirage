"""SegmentedMuGraphRunner -- ordinary muGraph compilation for model regions.

This module owns the *segmented compilation* mechanism: a model region (a small
subgraph such as "gate/up projection + SiLU + multiply") is expressed with
high-level ``KNGraph`` operators and compiled through Mirage's ordinary muGraph
compiler.  The result is a plain CUDA shared object launched from Python.

Deliberately **not** used anywhere in this file:
  * ``KNGraph.generate_task_graph``
  * ``KNGraph.register_task``
  * ``PersistentKernel`` (construction, ``compile``, or any ``*_layer`` call)

Weight-layout contract
----------------------
Hugging Face / PyTorch ``nn.Linear`` weights are stored as ``[out, in]``.
Mirage's ``matmul(A, B)`` wants ``B`` shaped ``[in, out]``.  We therefore feed
``w.t()`` -- a *strided view*, never a per-iteration contiguous copy -- and
declare the muGraph input as ``dims=(in, out), strides=(1, in)`` which is
exactly the stride pattern of ``w.t()`` for a row-major ``w``.
"""

from __future__ import annotations

import contextlib
import glob
import io
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

import mirage as mi

# --------------------------------------------------------------------------
# dtype plumbing
# --------------------------------------------------------------------------

_TORCH_TO_MI = {
    torch.bfloat16: mi.bfloat16,
    torch.float16: mi.float16,
}


def _mi_dtype(torch_dtype: torch.dtype):
    try:
        return _TORCH_TO_MI[torch_dtype]
    except KeyError:
        raise NotImplementedError(
            f"segmented muGraph regions support {list(_TORCH_TO_MI)}, got {torch_dtype}"
        )


# --------------------------------------------------------------------------
# Guards: this prototype must never touch the MPK task graph
# --------------------------------------------------------------------------

_TASK_GRAPH_GLOBS = ("task_graph*.json", "**/task_graph*.json")


def _task_graph_snapshot(root: str = ".") -> Dict[str, float]:
    """Map of existing task-graph JSON path -> mtime under *root*."""
    snap: Dict[str, float] = {}
    for pattern in _TASK_GRAPH_GLOBS:
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            try:
                snap[os.path.realpath(path)] = os.path.getmtime(path)
            except OSError:  # vanished between glob and stat
                pass
    return snap


def assert_no_task_graph_artifacts(
    root: str = ".", baseline: Optional[Dict[str, float]] = None
) -> None:
    """Raise if a task-graph JSON was *created or rewritten* under *root*.

    The segmented path must neither emit nor consume an MPK task graph.  We
    compare against a *baseline* snapshot rather than asserting the tree is
    empty, because unrelated MPK runs legitimately leave task graphs lying
    around; only files this run touched are a violation.
    """
    baseline = {} if baseline is None else baseline
    current = _task_graph_snapshot(root)
    offending = sorted(
        path
        for path, mtime in current.items()
        if path not in baseline or baseline[path] != mtime
    )
    if offending:
        raise AssertionError(
            "segmented muGraph path emitted or rewrote an MPK task graph: " f"{offending}"
        )


@contextlib.contextmanager
def no_task_graph_guard(root: str = "."):
    """Fail loudly if ``generate_task_graph`` is called, or a task graph appears."""
    from mirage.kernel import KNGraph

    calls: List[Tuple] = []
    original = KNGraph.generate_task_graph

    def _tripwire(self, *args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "generate_task_graph() was called inside the segmented muGraph path"
        )

    baseline = _task_graph_snapshot(root)
    KNGraph.generate_task_graph = _tripwire
    try:
        yield calls
    finally:
        KNGraph.generate_task_graph = original
    assert_no_task_graph_artifacts(root, baseline)


@contextlib.contextmanager
def _suppress_native_stdout(enabled: bool = True):
    """Silence the C++ search/transpiler chatter at the file-descriptor level."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """Shape / stride / dtype identity of one runtime tensor."""

    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "TensorSpec":
        return cls(tuple(t.shape), tuple(t.stride()), str(t.dtype))

    def matches(self, t: torch.Tensor) -> bool:
        return (
            tuple(t.shape) == self.shape
            and tuple(t.stride()) == self.strides
            and str(t.dtype) == self.dtype
        )

    def describe(self, t: torch.Tensor) -> str:
        return (
            f"expected shape={self.shape} strides={self.strides} dtype={self.dtype}; "
            f"got shape={tuple(t.shape)} strides={tuple(t.stride())} dtype={t.dtype}"
        )


class RegionKind(str, Enum):
    """Which muGraph region a compiled artifact implements."""

    #: gate proj + up proj + SiLU + multiply -> silu(x@Wg^T) * (x@Wu^T)
    GATE_UP_SILU_MUL = "A_gate_up_silu_mul"
    #: down proj + residual add -> a@Wd^T + residual
    DOWN_RESIDUAL = "B_down_residual"
    #: down proj only (residual left to the surrounding model) -> a@Wd^T
    DOWN_ONLY = "B_down_only"


@dataclass(frozen=True)
class RegionKey:
    """Full cache identity of a compiled region.

    Two regions share a compiled graph only when *every* field matches, so a
    different token count, hidden/intermediate size, dtype, input stride
    pattern, GPU target or compiler option forces a recompile.  Weight *values*
    are deliberately absent: weights are runtime inputs, so structurally
    identical transformer layers reuse one compiled graph.
    """

    kind: RegionKind
    tokens: int
    hidden_size: int
    intermediate_size: int
    dtype: str
    input_specs: Tuple[TensorSpec, ...]
    target_cc: int
    options: Tuple[Tuple[str, Any], ...]


# --------------------------------------------------------------------------
# Region graph construction
# --------------------------------------------------------------------------


@dataclass
class _RegionBuild:
    graph: Any
    input_specs: Tuple[TensorSpec, ...]


def _weight_view_spec(w: torch.Tensor) -> TensorSpec:
    """Spec of the transposed *view* of a ``[out, in]`` PyTorch weight."""
    return TensorSpec.from_tensor(w.t())


def build_region_a(
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    torch_dtype: torch.dtype,
    act_strides: Tuple[int, int],
    gate_strides: Tuple[int, int],
    up_strides: Tuple[int, int],
):
    """Region A: ``silu(x @ Wg^T) * (x @ Wu^T)``.

    Inputs: activation ``[tokens, hidden]``, gate weight view ``[hidden, inter]``,
    up weight view ``[hidden, inter]``.  Output: ``[tokens, inter]``.
    """
    dt = _mi_dtype(torch_dtype)
    g = mi.new_kernel_graph()
    x = g.new_input(dims=(tokens, hidden_size), strides=act_strides, dtype=dt)
    wg = g.new_input(dims=(hidden_size, intermediate_size), strides=gate_strides, dtype=dt)
    wu = g.new_input(dims=(hidden_size, intermediate_size), strides=up_strides, dtype=dt)
    out = g.mul(g.silu(g.matmul(x, wg)), g.matmul(x, wu))
    g.mark_output(out)
    return g


def build_region_b(
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    torch_dtype: torch.dtype,
    act_strides: Tuple[int, int],
    down_strides: Tuple[int, int],
    residual_strides: Optional[Tuple[int, int]],
):
    """Region B: ``a @ Wd^T`` (+ ``residual`` when *residual_strides* is given)."""
    dt = _mi_dtype(torch_dtype)
    g = mi.new_kernel_graph()
    a = g.new_input(dims=(tokens, intermediate_size), strides=act_strides, dtype=dt)
    wd = g.new_input(dims=(intermediate_size, hidden_size), strides=down_strides, dtype=dt)
    out = g.matmul(a, wd)
    if residual_strides is not None:
        res = g.new_input(dims=(tokens, hidden_size), strides=residual_strides, dtype=dt)
        out = g.add(out, res)
    g.mark_output(out)
    return g


# --------------------------------------------------------------------------
# Compiled region
# --------------------------------------------------------------------------


@dataclass
class CompiledRegion:
    """One compiled muGraph region plus its reusable launch buffers."""

    key: RegionKey
    graph: Any
    mode: str  # "superoptimized" | "direct"
    fallback_reason: Optional[str]
    search_time_s: float
    compile_time_s: float
    output_shape: Tuple[int, ...]
    output_strides: Tuple[int, ...]
    output_alloc: int
    buf_size: int
    profiler_buf_size: int
    device: torch.device
    torch_dtype: torch.dtype

    _buffer: torch.Tensor = field(repr=False, default=None)
    _profiler: torch.Tensor = field(repr=False, default=None)
    _output: torch.Tensor = field(repr=False, default=None)
    call_count: int = 0

    @property
    def output_spec(self) -> TensorSpec:
        return TensorSpec(self.output_shape, self.output_strides, str(self.torch_dtype))

    def new_output(self) -> torch.Tensor:
        base = torch.empty(self.output_alloc, dtype=self.torch_dtype, device=self.device)
        return base.as_strided(self.output_shape, self.output_strides)

    def validate(self, tensors: Sequence[torch.Tensor]) -> None:
        specs = self.key.input_specs
        if len(tensors) != len(specs):
            raise ValueError(
                f"region {self.key.kind.value}: expected {len(specs)} inputs, got {len(tensors)}"
            )
        for i, (spec, t) in enumerate(zip(specs, tensors)):
            if not spec.matches(t):
                raise ValueError(
                    f"region {self.key.kind.value} input {i} layout mismatch: {spec.describe(t)}"
                )

    def __call__(
        self,
        tensors: Sequence[torch.Tensor],
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
        validate: bool = True,
    ) -> torch.Tensor:
        """Launch the region.

        Reuses cached scratch/output buffers so the steady-state path performs
        no per-call allocation.  Does **not** synchronize -- the caller decides
        where the synchronization boundaries are.
        """
        if validate:
            self.validate(tensors)
        if out is None:
            out = self._output
        elif not self.output_spec.matches(out):
            raise ValueError(
                f"region {self.key.kind.value} output layout mismatch: "
                f"{self.output_spec.describe(out)}"
            )
        if stream is None:
            stream = torch.cuda.current_stream()
        self.graph.run(
            [t.data_ptr() for t in tensors],
            [out.data_ptr()],
            self._buffer.data_ptr(),
            stream.cuda_stream,
            self._profiler.data_ptr(),
        )
        self.call_count += 1
        return out


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


class SegmentedMuGraphRunner:
    """Owns and caches compiled muGraph regions.

    Compilation is lazy (first ``run_region_*`` call) unless ``compile_region``
    is called explicitly.  Compiled graphs are cached by :class:`RegionKey`, so
    every structurally identical transformer layer reuses one compiled graph
    while its weights stay ordinary runtime inputs.
    """

    #: Smallest token dimension the cuBLAS lowering handles correctly.
    #:
    #: With a single row the output strides degenerate to ``(1, 1)`` and
    #: ``kn::gemm``'s ``trans_C = (stride_n_C == 1)`` test misclassifies the
    #: layout, ending up with ``ldc=1`` where cuBLAS requires ``ldc >= n``.
    #: cuBLAS then rejects the call with ``CUBLAS_STATUS_INVALID_VALUE``.  We
    #: side-step it by compiling the next larger token bucket and padding, which
    #: costs nothing measurable: these GEMMs are weight-bandwidth bound, so
    #: M=1 and M=2 take the same time.  See DESIGN.md.
    MIN_TOKENS = 2

    def __init__(
        self,
        device: torch.device | str = "cuda",
        try_superoptimize: bool = True,
        superoptimize_config: str = "mlp",
        backend: str = "cuda",
        verbose: bool = True,
        quiet_native: bool = True,
        warmup_iters: int = 4,
        profile_iters: int = 32,
        min_tokens: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.try_superoptimize = try_superoptimize
        self.superoptimize_config = superoptimize_config
        self.backend = backend
        self.verbose = verbose
        self.quiet_native = quiet_native
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.min_tokens = self.MIN_TOKENS if min_tokens is None else min_tokens
        self._pad_buffers: Dict[Tuple, torch.Tensor] = {}

        major, minor = torch.cuda.get_device_capability(self.device)
        self.target_cc = major * 10 + minor

        self._cache: Dict[RegionKey, CompiledRegion] = {}
        #: how many times ``get`` found an existing compiled region
        self.cache_hits = 0
        self.cache_misses = 0
        #: regions the caller asked for but which fell back to PyTorch
        self.fallback_calls = 0
        self.mugraph_calls = 0

    # -- introspection ----------------------------------------------------

    @property
    def compiled_regions(self) -> Dict[RegionKey, CompiledRegion]:
        return dict(self._cache)

    @property
    def num_variants(self) -> int:
        return len(self._cache)

    def report(self) -> List[Dict[str, Any]]:
        out = []
        for key, region in self._cache.items():
            out.append(
                {
                    "kind": key.kind.value,
                    "tokens": key.tokens,
                    "hidden_size": key.hidden_size,
                    "intermediate_size": key.intermediate_size,
                    "dtype": key.dtype,
                    "target_cc": key.target_cc,
                    "mode": region.mode,
                    "fallback_reason": region.fallback_reason,
                    "search_time_s": region.search_time_s,
                    "compile_time_s": region.compile_time_s,
                    "call_count": region.call_count,
                    "input_specs": [
                        {"shape": list(s.shape), "strides": list(s.strides), "dtype": s.dtype}
                        for s in key.input_specs
                    ],
                    "output": {
                        "shape": list(region.output_shape),
                        "strides": list(region.output_strides),
                    },
                }
            )
        return out

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[segmented-mugraph] {msg}", flush=True)

    # -- compilation ------------------------------------------------------

    def _superoptimize(self, graph) -> Tuple[Optional[Any], Optional[str], float]:
        """Try ``KNGraph.superoptimize``; return (graph|None, reason, elapsed).

        ``superoptimize`` raises ``AttributeError`` when the search produced
        candidates but none of them transpiled/compiled (it dereferences a
        ``None`` best graph).  We translate any failure into an explicit
        fallback reason rather than letting it escape.
        """
        t0 = time.perf_counter()
        try:
            with _suppress_native_stdout(self.quiet_native):
                best = graph.superoptimize(
                    config=self.superoptimize_config,
                    backend=self.backend,
                    warmup_iters=self.warmup_iters,
                    profile_iters=self.profile_iters,
                    use_cached_graphs=False,  # do not litter cwd with checkpoints
                )
            elapsed = time.perf_counter() - t0
            if best is None:
                return None, "superoptimize returned no graph", elapsed
            return best, None, elapsed
        except Exception as exc:  # noqa: BLE001 - fallback must never escape
            elapsed = time.perf_counter() - t0
            reason = (
                "superoptimize found no compilable muGraph "
                f"({type(exc).__name__}: {str(exc)[:160]})"
            )
            return None, reason, elapsed

    def compile_region(
        self,
        kind: RegionKind,
        tokens: int,
        hidden_size: int,
        intermediate_size: int,
        sample_inputs: Sequence[torch.Tensor],
        torch_dtype: torch.dtype = torch.bfloat16,
        options: Optional[Dict[str, Any]] = None,
    ) -> CompiledRegion:
        """Compile (or fetch from cache) one region.

        *sample_inputs* are representative runtime tensors -- already in the
        exact layout that will be used at run time (weights passed as ``w.t()``
        views).  Their shapes/strides/dtypes become part of the cache key and
        are validated against the compiled graph.
        """
        options = dict(options or {})
        input_specs = tuple(TensorSpec.from_tensor(t) for t in sample_inputs)
        key = RegionKey(
            kind=kind,
            tokens=tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=str(torch_dtype),
            input_specs=input_specs,
            target_cc=self.target_cc,
            options=tuple(sorted(options.items())),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        act_strides = input_specs[0].strides
        if kind is RegionKind.GATE_UP_SILU_MUL:
            graph = build_region_a(
                tokens, hidden_size, intermediate_size, torch_dtype,
                act_strides, input_specs[1].strides, input_specs[2].strides,
            )
        elif kind in (RegionKind.DOWN_RESIDUAL, RegionKind.DOWN_ONLY):
            res_strides = (
                input_specs[2].strides if kind is RegionKind.DOWN_RESIDUAL else None
            )
            graph = build_region_b(
                tokens, hidden_size, intermediate_size, torch_dtype,
                act_strides, input_specs[1].strides, res_strides,
            )
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown region kind {kind}")

        mode = "direct"
        fallback_reason: Optional[str] = None
        search_time = 0.0
        chosen = graph

        if self.try_superoptimize:
            best, fallback_reason, search_time = self._superoptimize(graph)
            if best is not None:
                chosen, mode = best, "superoptimized"
            else:
                self._log(
                    f"{kind.value} tokens={tokens}: falling back to direct KNGraph.compile "
                    f"-- {fallback_reason}"
                )

        t0 = time.perf_counter()
        with _suppress_native_stdout(self.quiet_native):
            results = chosen.compile(inputs=list(sample_inputs), target_cc=self.target_cc)
        compile_time = time.perf_counter() - t0

        if results is None or not chosen._valid_cuda_kernels:
            err = chosen.get_error_message() if chosen._is_compiled else "unknown"
            raise RuntimeError(
                f"region {kind.value} (tokens={tokens}) failed to compile: {err}"
            )
        if len(results["output_directives"]) != 1:
            raise RuntimeError(
                f"region {kind.value} expected exactly 1 output, got "
                f"{len(results['output_directives'])}"
            )

        directive = results["output_directives"][0]
        region = CompiledRegion(
            key=key,
            graph=chosen,
            mode=mode,
            fallback_reason=fallback_reason,
            search_time_s=search_time,
            compile_time_s=compile_time,
            output_shape=tuple(directive["shape"]),
            output_strides=tuple(directive["strides"]),
            output_alloc=int(directive["alloc_size"]),
            buf_size=int(results["buf_size"]),
            profiler_buf_size=int(results["profiler_buf_size"]),
            device=self.device,
            torch_dtype=torch_dtype,
        )
        region._buffer = torch.empty(
            max(region.buf_size, 1), dtype=torch.uint8, device=self.device
        )
        region._profiler = torch.empty(
            max(region.profiler_buf_size, 1), dtype=torch.uint64, device=self.device
        )
        region._output = region.new_output()

        self._cache[key] = region
        self._log(
            f"compiled {kind.value} tokens={tokens} mode={mode} "
            f"search={search_time:.2f}s nvcc={compile_time:.2f}s "
            f"out={region.output_shape}/{region.output_strides}"
        )
        return region

    # -- convenience wrappers --------------------------------------------

    def padded_tokens(self, tokens: int) -> int:
        """Token bucket actually compiled for a logical *tokens* count."""
        return max(tokens, self.min_tokens)

    def _pad(self, tag: str, t: torch.Tensor, rows: int) -> torch.Tensor:
        """Return *t* widened to *rows* rows, reusing a cached zero buffer."""
        if t.shape[0] == rows:
            return t
        key = (tag, rows, tuple(t.shape[1:]), t.dtype)
        buf = self._pad_buffers.get(key)
        if buf is None:
            buf = torch.zeros(rows, *t.shape[1:], dtype=t.dtype, device=self.device)
            self._pad_buffers[key] = buf
        buf[: t.shape[0]].copy_(t)
        return buf

    def region_a(
        self,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        keep_padding: bool = False,
    ) -> torch.Tensor:
        """``silu(x @ w_gate^T) * (x @ w_up^T)`` with ``[out, in]`` weights.

        The returned tensor aliases a buffer owned by this region and is
        overwritten by the next call for the same shape bucket -- consume it (or
        clone it) before re-invoking.  With *keep_padding* the full padded
        bucket is returned so it can be chained straight into Region B.
        """
        logical, hidden = x.shape
        inter = w_gate.shape[0]
        rows = self.padded_tokens(logical)
        xp = self._pad("A_in", x, rows)
        tensors = (xp, w_gate.t(), w_up.t())
        region = self.compile_region(
            RegionKind.GATE_UP_SILU_MUL, rows, hidden, inter, tensors, x.dtype
        )
        self.mugraph_calls += 1
        result = region(tensors, out=out)
        return result if keep_padding else result[:logical]

    def region_b(
        self,
        a: torch.Tensor,
        w_down: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        logical_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """``a @ w_down^T`` (+ residual) with a ``[out, in]`` weight."""
        rows_in, inter = a.shape
        hidden = w_down.shape[0]
        logical = rows_in if logical_tokens is None else logical_tokens
        rows = self.padded_tokens(rows_in)
        ap = self._pad("B_in", a, rows)
        if residual is None:
            kind, tensors = RegionKind.DOWN_ONLY, (ap, w_down.t())
        else:
            kind = RegionKind.DOWN_RESIDUAL
            tensors = (ap, w_down.t(), self._pad("B_res", residual, rows))
        region = self.compile_region(kind, rows, hidden, inter, tensors, a.dtype)
        self.mugraph_calls += 1
        return region(tensors, out=out)[:logical]

    def mlp(
        self,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full segmented MLP: Region A then Region B, no sync in between.

        Padding is applied once on the way in and stripped once on the way out,
        so the intermediate never round-trips through a contiguous copy.
        """
        logical = x.shape[0]
        mid = self.region_a(x, w_gate, w_up, keep_padding=True)
        return self.region_b(mid, w_down, residual, logical_tokens=logical)

    # -- explicit warm-up -------------------------------------------------

    def precompile_mlp(
        self,
        tokens: int,
        hidden_size: int,
        intermediate_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        with_residual: bool = False,
    ) -> Tuple[CompiledRegion, CompiledRegion]:
        """Compile both regions for one shape bucket using dummy tensors."""
        dev, dt = self.device, torch_dtype
        tokens = self.padded_tokens(tokens)
        x = torch.zeros(tokens, hidden_size, dtype=dt, device=dev)
        wg = torch.zeros(intermediate_size, hidden_size, dtype=dt, device=dev)
        wu = torch.zeros(intermediate_size, hidden_size, dtype=dt, device=dev)
        wd = torch.zeros(hidden_size, intermediate_size, dtype=dt, device=dev)
        a = self.compile_region(
            RegionKind.GATE_UP_SILU_MUL, tokens, hidden_size, intermediate_size,
            (x, wg.t(), wu.t()), dt,
        )
        mid = a.new_output()
        if with_residual:
            res = torch.zeros(tokens, hidden_size, dtype=dt, device=dev)
            b = self.compile_region(
                RegionKind.DOWN_RESIDUAL, tokens, hidden_size, intermediate_size,
                (mid, wd.t(), res), dt,
            )
        else:
            b = self.compile_region(
                RegionKind.DOWN_ONLY, tokens, hidden_size, intermediate_size,
                (mid, wd.t()), dt,
            )
        return a, b

    def has_bucket(
        self,
        tokens: int,
        hidden_size: int,
        intermediate_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        with_residual: bool = False,
    ) -> bool:
        """True when both regions for this shape bucket are already compiled."""
        tokens = self.padded_tokens(tokens)
        kinds = (
            RegionKind.GATE_UP_SILU_MUL,
            RegionKind.DOWN_RESIDUAL if with_residual else RegionKind.DOWN_ONLY,
        )
        present = {
            (k.kind, k.tokens, k.hidden_size, k.intermediate_size, k.dtype)
            for k in self._cache
        }
        return all(
            (kind, tokens, hidden_size, intermediate_size, str(torch_dtype)) in present
            for kind in kinds
        )
