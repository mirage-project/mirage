"""Native graph metadata regression: reshape must not reuse slice strides.

No GPU allocations or launches; requires a built Mirage native extension.
"""
import unittest
import torch
from mirage.core import CyKNGraph, bfloat16
from mirage.kernel import KNGraph


class ViewLayoutTests(unittest.TestCase):
    def setUp(self):
        self.graph = KNGraph(CyKNGraph(disable_fingerprint=True))

    def tensor(self, shape):
        return self.graph.new_input(shape, dtype=bfloat16)

    def test_gate_heads_same_rank_reshape(self):
        x = self.tensor((1, 6144))
        y = self.graph.view(x, [48, 128])
        self.assertEqual(y.stride, torch.empty(1, 6144).view(48, 128).stride())
        # Each head advances 128 BF16 elements, not the original row's 6144.
        self.assertEqual(y.stride[0] * 47, 6016)

    def test_flatten_heads_same_rank(self):
        x = self.tensor((48, 128))
        self.assertEqual(self.graph.view(x, [1, 6144]).stride, (6144, 1))

    def test_reshape_contiguous_narrow_with_singleton_row(self):
        x = self.tensor((1, 128))
        ab = self.graph.narrow(x, 1, 0, 96)
        self.assertEqual(ab.stride, (128, 1))
        self.assertEqual(self.graph.view(ab, [2, 48]).stride, (48, 1))

    def test_different_rank_reshape(self):
        x = self.tensor((1, 10240))
        self.assertEqual(self.graph.view(x, [1, 80, 128]).stride, (10240, 128, 1))

    def test_strided_slice_preserved_and_unsafe_reshape_rejected(self):
        x = self.tensor((2, 64))
        part = self.graph.narrow(x, 1, 0, 32)
        self.assertEqual(part.stride, (64, 1))
        self.assertEqual(self.graph.view(part, [2, 32]).stride, (64, 1))
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            self.graph.view(part, [4, 16])
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            self.graph.view(part, [64])


if __name__ == '__main__':
    unittest.main()
