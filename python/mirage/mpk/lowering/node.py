"""A model as one inspectable graph, ahead of any decision about tasks."""
from __future__ import annotations

import contextlib
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

    def __repr__(self) -> str:
        return f"{self.name}{list(self.dims)}"

@dataclasses.dataclass
class Node:
    op: str
    inputs: tuple[Value, ...]
    output: Value
    layer: Optional[int] = None         # which transformer layer, for replication
    tag: str = ""                       # human label, e.g. "mlp.gate"
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
        self._layer: Optional[int] = None
        self._tag: str = ""

    def new_input(self, dims, name: str, role: str = "weight") -> Value:
        v = Value(name=name, dims=tuple(dims), producer=None, role=role)
        self.inputs.append(v)
        return v

    def mark_output(self, v: Value) -> None:
        self.outputs.append(v)

    def _fresh(self, dims) -> Value:
        self._n += 1
        base = f"{self._tag}." if self._tag else ""
        return Value(name=f"{base}v{self._n}", dims=tuple(dims),
                     producer=len(self.nodes))

    def opaque_multi(self, name: str, inputs, dims_list, **attrs) -> list:
        """An opaque task that writes SEVERAL tensors -- attention prep stages
        q/k^T/v/mask for the generated core. Every output is a real Value, so
        the reader's dependency on the writer stays visible to MPK."""
        outs = [self._fresh(d) for d in dims_list]
        self.nodes.append(Node(op=OPAQUE + name, inputs=tuple(inputs),
                               output=outs[0], layer=self._layer, tag=self._tag,
                               attrs=dict(attrs, extra_outputs=tuple(outs[1:]))))
        return outs

    def opaque(self, name: str, inputs, dims, **attrs) -> Value:
        """A hand-written task the graph does not model. Its result is a real
        value, so everything downstream stays connected."""
        out = self._fresh(dims)
        self.nodes.append(Node(op=OPAQUE + name, inputs=tuple(inputs),
                               output=out, layer=self._layer, tag=self._tag,
                               attrs=attrs))
        return out

    def _emit(self, op: str, inputs, dims, **attrs) -> Value:
        assert len(inputs) == OPS[op], \
            f"{op} takes {OPS[op]}, got {len(inputs)}"
        out = self._fresh(dims)
        self.nodes.append(Node(op=op, inputs=tuple(inputs), output=out,
                               layer=self._layer, tag=self._tag, attrs=attrs))
        return out

    def matmul(self, a: Value, b: Value) -> Value:
        assert a.dims[-1] == b.dims[-2], f"matmul shape: {a} @ {b}"
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

    @contextlib.contextmanager
    def scope(self, layer: Optional[int] = None, tag: str = ""):
        """Label the nodes emitted inside, so a partition found on one layer
        can be replicated to the rest."""
        prev = (self._layer, self._tag)
        self._layer, self._tag = layer, tag
        try:
            yield self
        finally:
            self._layer, self._tag = prev

    def consumers(self, v: Value) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if v in n.inputs]

    def __len__(self) -> int:
        return len(self.nodes)

    def describe(self) -> str:
        return "\n".join(f"  [{i:3d}] {n!r}" for i, n in enumerate(self.nodes))
