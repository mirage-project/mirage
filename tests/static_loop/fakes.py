"""Stand-ins for the loop's tests on a machine without a GPU or the built Mirage extension:

  import_mpk()   registers the packages mirage and mirage.mpk by path (their __init__ loads the extension), so
                 mirage.mpk.compiler, search, solver, cost_records, static_schedule and mirage.mpk.loop.* import alone
  FakePk         what compiler.graph_nodes reads of a StaticMegakernel: kn_graph.get_task_info / regrid, _static_nodes,
                 _tensor_names, num_gpus, _num_sms()
  small_graph()  a 4-node graph shaped like the MoE layer: a GEMM with K parts (grid y), its reader (adds the K parts), a pool
                 (one task per SM) and a node after the pool
"""
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def import_mpk():
    for name, sub in (("mirage", "python/mirage"), ("mirage.mpk", "python/mirage/mpk")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(ROOT, sub)]
            sys.modules[name] = m
    import mirage.mpk.compiler  # noqa: F401  (checks the packages resolve)
    return sys.modules["mirage.mpk"]


class Tensor:
    _next = 0

    def __init__(self, *dims):
        Tensor._next += 1
        self.guid = Tensor._next
        self._dims = list(dims)
        self.num_dims = len(dims)
        self.dtype = "bf16"   # as a graph tensor has one (compiler.node_view writes it to the schedule files)

    def dim(self, i):
        return self._dims[i]


class FakeKn:
    def __init__(self):
        self.ops = []   # [name, params, grid, inputs, outputs]

    def get_task_info(self, idx):
        name, params, grid, ins, outs = self.ops[idx]
        return name, params, grid, len(ins), len(outs), ins + outs

    def regrid(self, idx, grid, params):
        self.ops[idx][2] = tuple(grid)
        self.ops[idx][1] = list(params)

    def get_num_operators(self):
        return len(self.ops)


class FakePk:
    def __init__(self, num_gpus=2, num_sms=8):
        self.num_gpus, self.num_sms = num_gpus, num_sms
        self.kn_graph = FakeKn()
        self._static_nodes, self._tensor_names = {}, {}
        self.tokens = 4

    def _num_sms(self):
        return self.num_sms

    def name(self, t, name):
        self._tensor_names[t.guid] = name
        return t

    def add(self, name, grid, inputs, outputs, maps, params=(), **info):
        idx = len(self.kn_graph.ops)
        self.kn_graph.ops.append([name, list(params), tuple(grid), list(inputs), list(outputs)])
        d = dict(name=name, input_maps=list(maps), changeable_grid_dims={}, rows_split_over_gpus=False, gpu_split_blocks=grid[0],
                 one_task_per_sm=False, concurrent_along_axis=None, cluster_pair_params=None, buf=-1, out_buffer=None,
                 exchange=False, counter=-1, buf_slots=[], map_slots=[], dry=False, cost_inputs=[], variants={},
                 variant_defaults={}, stage0_inputs=False)
        d.update(info)
        d["variant_defaults"] = {k: list(params)[i] for k, (i, _) in d["variants"].items()}
        self._static_nodes[idx] = d
        return idx


def small_graph(num_gpus=2, num_sms=8):
    """gemm (4 row blocks x K parts 1/2/4) -> reader (4 tasks, adds the K parts) -> pool (one per SM) -> tail (2 tasks)."""
    pk = FakePk(num_gpus, num_sms)
    x, w = pk.name(Tensor(4, 512), "x"), pk.name(Tensor(512, 512), "w")
    a = pk.name(Tensor(4, 512), "gemm_out")
    b = pk.name(Tensor(4, 512), "reader_out")
    c = pk.name(Tensor(4, 512), "pool_out")
    d = pk.name(Tensor(4, 512), "tail_out")
    pk.add("gemm_tile", (4, 1, 1), [x, w], [a], [(-1, 1, -1), (0, 1, -1), (1, -1, -1)], [0, 512, 512, 0],
           changeable_grid_dims={1: (128, 1, None)})
    pk.add("reader", (4, 1, 1), [a], [b], [(1, -1, -1), (1, -1, -1)], cost_inputs=[0], dry=True)
    pk.add("pool", (num_sms, 1, 1), [b], [c], [(-1, -1, -1), (-1, -1, -1)], one_task_per_sm=True)
    pk.add("tail", (2, 1, 1), [c], [d], [(1, -1, -1), (1, -1, -1)])
    return pk
