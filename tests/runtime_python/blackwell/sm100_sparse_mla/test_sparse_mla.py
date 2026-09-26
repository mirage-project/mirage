"""GPU correctness tests. Requires the standalone extension built with setup.py."""

import unittest
import numpy as np
from reference import make_case, torch_reference

try:
    import torch
except ImportError:
    torch = None

HAS_SM100 = (torch is not None and torch.cuda.is_available() and
             torch.cuda.get_device_capability() == (10, 0))


def to_cuda(case):
    return [torch.tensor(x, device="cuda", dtype=torch.bfloat16 if i < 2 else torch.int32)
            for i, x in enumerate(case)]


def assert_close(actual, expected):
    assert torch.isfinite(actual).all(), "nonfinite kernel output"
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)
    rms = (actual.float() - expected.float()).square().mean().sqrt()
    relative_rms = rms / expected.float().square().mean().sqrt().clamp_min(1e-8)
    assert relative_rms.item() <= 0.02, f"relative RMS={relative_rms.item()}"


def allocate_workspace(q, splits):
    t, h = q.shape[:2]
    output = torch.full((t, h, 512), float("nan"), device=q.device, dtype=q.dtype)
    partial = torch.full((t, splits, h, 512), float("nan"), device=q.device)
    lse = torch.full((t, splits, h), float("nan"), device=q.device)
    return output, partial, lse


@unittest.skipUnless(HAS_SM100, "requires SM100 and CUDA PyTorch")
class SparseMLAGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # If a GPU exists but the extension was not built, fail rather than
        # silently reporting skipped correctness tests.
        import runtime_kernel_sparse_mla
        cls.extension = runtime_kernel_sparse_mla
        torch.backends.cuda.matmul.allow_tf32 = False

    def check_case(self, case):
        tensors = to_cuda(case)
        q = tensors[0]
        expected = torch_reference(*tensors, 0.0625)
        for splits in (1, 2, 4, 8):
            workspace = allocate_workspace(q, splits)
            self.extension.run(*tensors, *workspace, 0.0625, splits)
            torch.cuda.synchronize()
            assert_close(workspace[0], expected)
            if splits > 1:
                self.assertFalse(torch.isnan(workspace[1]).any())
                self.assertFalse(torch.isnan(workspace[2]).any())
            # Reuse the SAME workspaces after all selections become empty.
            saved_counts = tensors[3].clone()
            tensors[3].zero_()
            self.extension.run(*tensors, *workspace, 0.0625, splits)
            torch.cuda.synchronize()
            self.assertEqual(torch.count_nonzero(workspace[0]).item(), 0)
            if splits > 1:
                self.assertTrue(torch.isneginf(workspace[2]).all())
                self.assertEqual(torch.count_nonzero(workspace[1]).item(), 0)
            tensors[3].copy_(saved_counts)

    def test_shapes_and_query_regimes(self):
        for rope in (0, 64):
            for heads in (8, 16, 32, 64):
                for query_lengths, seq_lengths in (((1, 1), (129, 67)),
                                                  ((3, 5), (3, 5)),
                                                  ((2, 3), (193, 131))):
                    with self.subTest(rope=rope, heads=heads, q=query_lengths, s=seq_lengths):
                        self.check_case(make_case(rope, heads, query_lengths, seq_lengths))

    def test_index_boundaries_and_invalid_entries(self):
        for page_size in (64, 128):
            for count in (0, 1, 63, 64, 65, 129, 2048, 2101):
                with self.subTest(page_size=page_size, count=count):
                    case = make_case(query_lengths=(2,), seq_lengths=(2201,),
                                     page_size=page_size, capacity=2304)
                    case[3][:] = count
                    if count >= 4:
                        case[2][0, :4] = [-1, 99999, 2200, -2]
                    self.check_case(case)

    def test_order_invariance_and_inactive_query(self):
        case = list(make_case(rope_dim=0, query_lengths=(1,), seq_lengths=(193,)))
        for i in (0, 2, 3):
            case[i] = np.concatenate([case[i], case[i]], axis=0)
        self.check_case(case)
        case[2][0, :case[3][0]] = case[2][0, :case[3][0]][::-1]
        self.check_case(case)


if __name__ == "__main__":
    unittest.main()
