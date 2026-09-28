"""Exercise the actual Python layer without requiring CUDA or compiled Mirage.

Extract only the method AST and supply recording graph objects. This verifies
public validation and graph construction; it does not test C++ code generation.
"""

import ast
import itertools
from pathlib import Path
import unittest


class Tensor:
    ids = itertools.count(1)

    def __init__(self, shape, dtype="bf16", stride=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.num_dims = len(shape)
        self.guid = next(self.ids)
        self.base_guid = 0
        size = 1
        strides = []
        for d in reversed(shape):
            strides.append(size)
            size *= d
        self.stride = tuple(reversed(strides)) if stride is None else stride

    def dim(self, i):
        return self.shape[i]


class ThreadblockGraph:
    def __init__(self, config):
        self.config = config
        self.tensors = []

    def new_input(self, tensor, *args):
        self.tensors.append(tensor)


class Graph:
    def __init__(self):
        self.tasks = []

    def customized(self, tensors, tb_graph):
        if tensors != tb_graph.tensors:
            raise AssertionError("graph arguments disagree with registered tensors")

    def register_task(self, tb_graph, name, params):
        self.tasks.append((name, tb_graph, params))


def load_layer():
    repo = Path(__file__).resolve().parents[4]
    source = repo / "python/mirage/mpk/persistent_kernel.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PersistentKernel")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_mla_layer")
    env = {"DTensor": Tensor, "TBGraph": ThreadblockGraph,
           "CyTBGraph": lambda *args: args, "bfloat16": "bf16",
           "float32": "fp32", "int32": "int32"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), env)
    return env["sparse_mla_layer"]


class Kernel:
    sparse_mla_layer = load_layer()

    def __init__(self):
        self.target_cc = 100
        self.max_num_batched_tokens = 3
        self.max_num_pages = 4
        self.page_size = 64
        self.max_num_batched_requests = 2
        self.kn_graph = Graph()
        self.allocated_names = []

    def new_tensor(self, dims, dtype, name):
        self.allocated_names.append(name)
        return Tensor(dims, dtype)


class SparseMLALayerContractTests(unittest.TestCase):
    def inputs(self, rope=64, heads=8):
        return [Tensor((3, heads, 512 + rope)), Tensor((4, 64, 512 + rope)),
                Tensor((3, 2304), "int32"), Tensor((3,), "int32"),
                Tensor((3, heads, 512))]

    def test_unsplit_has_direct_output(self):
        pk = Kernel()
        args = self.inputs()
        pk.sparse_mla_layer(*args, softmax_scale=0.0625)
        self.assertEqual(len(pk.kn_graph.tasks), 1)
        name, graph, params = pk.kn_graph.tasks[0]
        self.assertEqual(name, "sparse_mla_sm100")
        self.assertIs(graph.tensors[-1], args[-1])
        self.assertEqual(graph.config[:2], ((3, 1, 1), (256, 1, 1)))
        self.assertEqual(params[:7], [8, 64, 64, 2304, 1, 2, 4])
        self.assertEqual(pk.allocated_names, [])

    def test_split_reduce_consumes_both_producer_outputs(self):
        for rope in (0, 64):
            for heads in (8, 16, 32, 64):
                for splits in (2, 4, 8):
                    with self.subTest(rope=rope, heads=heads, splits=splits):
                        pk = Kernel()
                        args = self.inputs(rope, heads)
                        pk.sparse_mla_layer(*args, softmax_scale=0.0625, num_splits=splits)
                        producer, consumer = pk.kn_graph.tasks
                        groups = (heads + 15) // 16
                        self.assertEqual(producer[0], "sparse_mla_sm100")
                        self.assertEqual(producer[1].config[:2],
                                         ((3, groups, splits), (256, 1, 1)))
                        self.assertEqual(producer[1].tensors[:4], args[:4])
                        self.assertEqual(producer[2][:7], [heads, rope, 64, 2304, splits, 2, 4])
                        self.assertEqual(consumer[0], "sparse_mla_reduce_sm100")
                        self.assertEqual(consumer[1].config[:2],
                                         ((3, groups, 1), (256, 1, 1)))
                        self.assertEqual(consumer[2], producer[2])
                        partial, lse = producer[1].tensors[-2:]
                        self.assertEqual(partial.shape, (3, splits, heads, 512))
                        self.assertEqual(lse.shape, (3, splits, heads))
                        self.assertEqual((partial.dtype, lse.dtype), ("fp32", "fp32"))
                        self.assertEqual(consumer[1].tensors, [partial, lse, args[-1]])

    def test_multi_head_group_downstream_consumer(self):
        for heads in (32, 64):
            for splits in (1, 2, 4, 8):
                with self.subTest(heads=heads, splits=splits):
                    pk = Kernel()
                    args = self.inputs(rope=0, heads=heads)
                    pk.sparse_mla_layer(*args, softmax_scale=0.0625, num_splits=splits)
                    first_output = pk.kn_graph.tasks[-1][1].tensors[-1]
                    self.assertIs(first_output, args[-1])
                    second_output = Tensor((3, heads, 512))
                    pk.sparse_mla_layer(first_output, *args[1:4], second_output,
                                        softmax_scale=0.125)
                    self.assertEqual(len(pk.kn_graph.tasks), 2 if splits == 1 else 3)
                    first, second = pk.kn_graph.tasks[0], pk.kn_graph.tasks[-1]
                    self.assertEqual(first[1].config[0], (3, heads // 16, splits))
                    self.assertEqual(second[0], "sparse_mla_sm100")
                    self.assertEqual(second[1].config[0], (3, heads // 16, 1))
                    self.assertEqual(second[1].tensors,
                                     [first_output, *args[1:4], second_output])
                    self.assertEqual(second[2][:5], [heads, 0, 64, 2304, 1])
                    self.assertEqual(len(pk.allocated_names), 0 if splits == 1 else 2)

    def test_tile_and_split_boundary_capacities(self):
        for capacity in (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513):
            for splits in (1, 2, 4, 8):
                with self.subTest(capacity=capacity, splits=splits):
                    pk = Kernel()
                    args = self.inputs(heads=32)
                    args[2] = Tensor((3, capacity), "int32")
                    pk.sparse_mla_layer(*args, softmax_scale=0.0625, num_splits=splits)
                    self.assertEqual(len(pk.kn_graph.tasks), 1 if splits == 1 else 2)
                    producer = pk.kn_graph.tasks[0]
                    self.assertEqual(producer[2][3:5], [capacity, splits])
                    self.assertEqual(producer[1].config[0], (3, 2, splits))
                    self.assertIs(pk.kn_graph.tasks[-1][1].tensors[-1], args[-1])

    def test_scale_float32_bits_are_preserved(self):
        for scale, bits in ((2.0 ** -149, 0x00000001),
                            (2.0 ** -126, 0x00800000),
                            (0.0625, 0x3D800000), (0.1, 0x3DCCCCCD),
                            (0.125, 0x3E000000), (1.0, 0x3F800000),
                            (float.fromhex("0x1.fffffep+127"), 0x7F7FFFFF)):
            for splits in (1, 2, 4, 8):
                with self.subTest(scale=scale, splits=splits):
                    pk = Kernel()
                    pk.sparse_mla_layer(*self.inputs(), softmax_scale=scale,
                                        num_splits=splits)
                    for _, _, params in pk.kn_graph.tasks:
                        self.assertEqual(params[-1], bits)

    def test_invalid_public_arguments(self):
        for scale in (0, -1, float("nan"), float("inf"), float("-inf"),
                      1e100, 1e-100, 2.0 ** -150, 2.0 ** 128):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                Kernel().sparse_mla_layer(*self.inputs(), softmax_scale=scale)
        for splits in (0, 3, 16, True):
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                Kernel().sparse_mla_layer(*self.inputs(), softmax_scale=0.0625, num_splits=splits)
        pk = Kernel()
        pk.target_cc = 90
        with self.assertRaisesRegex(ValueError, "SM100"):
            pk.sparse_mla_layer(*self.inputs(), softmax_scale=0.0625)

    def test_shape_dtype_stride_and_alias_checks(self):
        for index, tensor in ((0, Tensor((3, 7, 576))),
                              (1, Tensor((4, 64, 512))),
                              (2, Tensor((3, 2048), "bf16")),
                              (3, Tensor((2,), "int32")),
                              (4, Tensor((3, 8, 256))),
                              (0, Tensor((3, 8, 576), stride=(5000, 576, 1)))):
            args = self.inputs()
            args[index] = tensor
            with self.subTest(index=index, shape=tensor.shape), self.assertRaises(ValueError):
                Kernel().sparse_mla_layer(*args, softmax_scale=0.0625)
        args = self.inputs(rope=0)
        args[-1].base_guid = args[0].guid
        with self.assertRaisesRegex(ValueError, "alias"):
            Kernel().sparse_mla_layer(*args, softmax_scale=0.0625)


if __name__ == "__main__":
    unittest.main()
