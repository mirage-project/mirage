"""A model as one inspectable graph, ahead of any decision about tasks."""
from __future__ import annotations

import dataclasses
from typing import Optional

OPS: dict[str, int] = {
    "matmul": 2,
    "add": 2,
    "mul": 2,
    "div": 2,
    "silu": 1,
    "gelu": 1,
    "relu": 1,
    "exp": 1,
    "sqrt": 1,
    "square": 1,
    "rms_norm": 1,
    "reduction": 1,
}

# The OTHER half of the vocabulary: a node the graph cannot model, computed by
# a hand-written MPK task. name -> (arity, number of outputs). A model builds
# its graph out of OPS and these; lowering/opaque.py supplies the task that
# computes each one, and asserts it covers this table.
#
# They are declared here, beside OPS, because together the two ARE the set of
# nodes a model may use -- and because an arity checked at graph construction
# is a mismatch caught before the operand order reaches a kernel.
OPAQUE_OPS: dict[str, tuple[int, int]] = {
    "embedding": (2, 1),        # (tokens, table) -- a gather
    "attention": (7, 1),        # the whole of attention as one task
    "attn_prep": (7, 4),        # qk-norm, RoPE, cache append, staging
    "attn_finalize": (1, 1),    # pack the padded core output
    "argmax": (1, 3),           # a reduction to indices: token, value, index
}

# A node whose op starts with this is not a muGraph op at all
OPAQUE = "opaque:"


def is_opaque(op: str) -> bool:
    return op.startswith(OPAQUE)


@dataclasses.dataclass(frozen=True)
class Value:
    """An SSA tensor. `producer` is None for a graph input (weight or feed)."""
    name: str
    dims: tuple[int, ...]
    producer: Optional[int] = None      # index into ModelGraph.nodes
    role: str = "activation"            # "activation" | "weight" | "feed"
    # None means the graph's default (bfloat16). Only an opaque task needs
    # anything else: argmax reduces through an INDEX buffer, and reading an
    # int64 index out of a bf16 allocation is a misaligned access, not a
    # wrong number -- the run dies in cudaDeviceSynchronize with no hint.
    dtype: Optional[str] = None
    # A value a task only PARTIALLY writes needs a defined starting state.
    # Attention prep writes mask[pos]=0 to mark one position valid and never
    # touches the rest, so the buffer must start at -30000: cudaMalloc leaves
    # it undefined, and zero means "every position is valid".
    init: Optional[float] = None

    def __repr__(self) -> str:
        return f"{self.name}{list(self.dims)}"


@dataclasses.dataclass
class Node:
    op: str
    inputs: tuple[Value, ...]
    output: Value
    attrs: dict = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.output.name} = {self.op}({', '.join(v.name for v in self.inputs)})"


class ModelGraph:
    """Nodes in topological order. Build with new_input / op methods."""

    def __init__(self, name: str = "model"):
        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []
        self._n = 0

    def new_input(self, dims, name: str, role: str = "weight") -> Value:
        v = Value(name=name, dims=tuple(dims), producer=None, role=role)
        self.inputs.append(v)
        return v

    def mark_output(self, v: Value) -> None:
        self.outputs.append(v)

    def _fresh(self, dims, dtype=None, init=None) -> Value:
        self._n += 1
        return Value(name=f"v{self._n}", dims=tuple(dims),
                     producer=len(self.nodes), dtype=dtype, init=init)

    def opaque(self, name: str, inputs, dims, **attrs) -> Value:
        """One node for a task the graph cannot model. Its result is a real
        Value, so everything downstream stays connected."""
        return self.opaque_multi(name, inputs, [dims], **attrs)[0]

    def opaque_multi(self, name: str, inputs, dims_list, dtypes=None,
                     inits=None, **attrs) -> list:
        """The same, for a task that writes SEVERAL tensors -- attention prep
        stages q/k^T/v/mask for the generated core. Every output is a real
        Value, so the reader's dependency on the writer stays visible to MPK.

        Arity is checked here, at construction, not at lowering.
        """
        arity, n_out = OPAQUE_OPS.get(name, (None, None))
        if arity is None:
            raise ValueError(f"unknown opaque task {name!r}; OPAQUE_OPS "
                             f"declares {sorted(OPAQUE_OPS)}")
        if len(inputs) != arity:
            raise ValueError(f"opaque {name!r} takes {arity} inputs, got "
                             f"{len(inputs)}")
        if len(dims_list) != n_out:
            raise ValueError(f"opaque {name!r} writes {n_out} tensors, got "
                             f"{len(dims_list)}")
        outs = [self._fresh(d, t, i)
                for d, t, i in zip(dims_list, dtypes or [None] * n_out,
                                   inits or [None] * n_out)]
        self._append(OPAQUE + name, inputs, outs[0],
                     dict(attrs, extra_outputs=tuple(outs[1:])))
        return outs

    def _append(self, op: str, inputs, out: Value, attrs: dict) -> None:
        self.nodes.append(Node(op=op, inputs=tuple(inputs), output=out,
                               attrs=attrs))

    def _emit(self, op: str, inputs, dims, **attrs) -> Value:
        if len(inputs) != OPS[op]:
            raise ValueError(f"{op} takes {OPS[op]} inputs, got {len(inputs)}")
        out = self._fresh(dims)
        self._append(op, inputs, out, attrs)
        return out

    def matmul(self, a: Value, b: Value) -> Value:
        if a.dims[-1] != b.dims[-2]:
            raise ValueError(f"matmul shape: {a} @ {b}")
        return self._emit("matmul", (a, b), a.dims[:-1] + (b.dims[-1],))

    def rms_norm(self, x: Value, normalized_shape=None) -> Value:
        ns = tuple(normalized_shape) if normalized_shape else (x.dims[-1],)
        return self._emit("rms_norm", (x,), x.dims, normalized_shape=ns)

    def _binary(self, op: str, a: Value, b: Value) -> Value:
        # Broadcasting is the caller's business; the shape is the wider operand.
        dims = a.dims if len(a.dims) >= len(b.dims) else b.dims
        return self._emit(op, (a, b), dims)

    def add(self, a, b): return self._binary("add", a, b)
    def mul(self, a, b): return self._binary("mul", a, b)
    def div(self, a, b): return self._binary("div", a, b)

    def reduction(self, x: Value, dim: int) -> Value:
        """Sum over `dim`, keeping it at extent 1 -- the softmax denominator."""
        dims = list(x.dims)
        dims[dim] = 1
        return self._emit("reduction", (x,), tuple(dims), dim=dim)

    def silu(self, x): return self._emit("silu", (x,), x.dims)
    def gelu(self, x): return self._emit("gelu", (x,), x.dims)
    def relu(self, x): return self._emit("relu", (x,), x.dims)
    def exp(self, x): return self._emit("exp", (x,), x.dims)
    def sqrt(self, x): return self._emit("sqrt", (x,), x.dims)
    def square(self, x): return self._emit("square", (x,), x.dims)

    def consumers(self, v: Value) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if v in n.inputs]

    def __len__(self) -> int:
        return len(self.nodes)

    def describe(self) -> str:
        return "\n".join(f"  [{i:3d}] {n!r}" for i, n in enumerate(self.nodes))
