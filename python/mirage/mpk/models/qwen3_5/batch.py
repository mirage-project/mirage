"""Static batches: shared GEMMs and a request axis for private-state tasks.

Each customized operation is one graph node with grid=(heads, requests, 1).
Request partitioning offsets pointers to disjoint state rows. Decode launches
one persistent graph for the entire batch.
"""
from numbers import Integral
import os

import torch


def validate_batch_size(batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an integer in [1, 32]")


def validate_offline_batch_size(batch_size):
    """Keep the numerically unverified B8 graph behind an explicit experiment flag."""
    validate_batch_size(batch_size)
    if batch_size not in (1, 8):
        raise ValueError("Qwen offline currently has B1 and experimental B8 graphs")
    if batch_size == 8 and os.environ.get("MPK_QWEN_EXPERIMENTAL_B8") != "1":
        raise ValueError("B8 exact-token validation is incomplete; set "
                         "MPK_QWEN_EXPERIMENTAL_B8=1 to run the experimental graph")


def linear_grid(width, batch_size, default_cols=64, *, optimized_b8=False):
    """Choose output partitions for optimized fixed-batch graphs and probes."""
    override = os.environ.get(f"MPK_QWEN38_B{batch_size}_LINEAR_COLS")
    if override is None:
        cols = 128 if optimized_b8 and batch_size in (8, 16) and width % 128 == 0 else default_cols
    else:
        if override not in ("64", "128") or batch_size not in (8, 16):
            raise ValueError(f"MPK_QWEN38_B{batch_size}_LINEAR_COLS must be 64 or 128 for B8/B16")
        # The unpadded GDN a/b projection has 96 outputs and retains its
        # existing 32-column partition. All divisible projections use 128.
        cols = int(override) if width % int(override) == 0 else default_cols
    if width % cols != 0:
        raise ValueError(f"linear width {width} is not divisible by {cols}")
    return (width // cols, 1, 1)


class BatchRows:
    def __init__(self, pk, batch_size, tensors, dt, prefix):
        validate_batch_size(batch_size)
        self.pk, self.batch_size = pk, batch_size
        self.tensors, self.dt, self.prefix = tensors, dt, prefix
        self.shapes = {}

    def alias(self, name, row_shape):
        self.shapes[name] = tuple(row_shape)

    def shape(self, row_shape):
        shape = tuple(row_shape)
        if self.batch_size > 1:
            shape = (self.batch_size,) + (shape[1:] if shape[0] == 1 else shape)
        return shape

    def empty(self, name, row_shape, dtype=torch.bfloat16):
        tensor = torch.zeros(self.shape(row_shape), dtype=dtype, device="cuda")
        self.tensors[name] = tensor
        self.dt[name] = self.pk.attach_input(tensor, name=self.prefix + name)
        self.alias(name, row_shape)
        return self.dt[name]

    def view(self, name, source, shape, *, narrow=None):
        view = self.dt[source]
        if narrow is not None:
            view = self.pk.narrow(view, *narrow)
        self.dt[name] = self.pk.view(view, list(self.shape(shape)))
        self.alias(name, shape)

    def task(self, name, names, partitions, tasks=1, params=None):
        from mirage.core import CyTBGraph
        from mirage.kernel import TBGraph

        graph = TBGraph(CyTBGraph((tasks, self.batch_size, 1), (256, 1, 1), 1, 64))
        tensors = [self.dt[n] for n in names]
        for n, tensor, dim in zip(names, tensors, partitions):
            batched = self.batch_size > 1 and n in self.shapes
            if batched and dim >= 0 and self.shapes[n][0] != 1:
                dim += 1
            graph.new_input(tensor, (dim, 0 if batched else -1, -1), -1, True)
        self.pk.kn_graph.customized(tensors, graph)
        self.pk.kn_graph.register_task(graph, name, [] if params is None else params)
