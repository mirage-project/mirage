"""CPU semantic tests: python -m unittest discover -s THIS_DIRECTORY -p test_reference.py."""

import unittest
import numpy as np
from reference import make_case, numpy_reference


class SparseMLAReferenceTests(unittest.TestCase):
    def test_full_selection_matches_dense_causal(self):
        for rope in (0, 64):
            case = make_case(rope_dim=rope, query_lengths=(1, 3), seq_lengths=(17, 5))
            q, cache, _, _, qo, ki, pages, last = case
            actual = numpy_reference(*case, 0.0625)
            for b in range(len(last)):
                sequence = cache[pages[ki[b]:ki[b + 1]]].reshape(-1, cache.shape[-1])
                seq_len = (ki[b + 1] - ki[b] - 1) * cache.shape[1] + last[b]
                for t in range(qo[b], qo[b + 1]):
                    end = seq_len - (qo[b + 1] - qo[b]) + t - qo[b] + 1
                    logits = q[t] @ sequence[:end].T * 0.0625
                    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
                    expected = (probs / probs.sum(axis=-1, keepdims=True)) @ sequence[:end, :512]
                    np.testing.assert_allclose(actual[t], expected, atol=1e-6, rtol=1e-5)

    def test_padding_future_indices_and_empty_selection(self):
        case = make_case(query_lengths=(2,), seq_lengths=(5,), capacity=8)
        case[2][:] = -1
        case[2][0, :4] = [0, 4, 999, -1]  # query 0 is at position 3
        case[3][:] = [4, 0]
        out = numpy_reference(*case, 0.0625)
        np.testing.assert_array_equal(out[0], np.broadcast_to(case[1][case[6][0], 0, :512], out[0].shape))
        np.testing.assert_array_equal(out[1], 0)
        self.assertTrue(np.isfinite(out).all())

    def test_all_invalid_indices_with_positive_counts(self):
        for rope in (0, 64):
            for count in (1, 63, 64, 65, 129, 513):
                with self.subTest(rope=rope, count=count):
                    case = make_case(rope_dim=rope, query_lengths=(2,),
                                     seq_lengths=(130,), capacity=count + 1)
                    case[2][0, :count] = np.resize([-1, -2, 130, 99999, 129], count)
                    case[2][1, :count] = np.resize([-1, -2, 130, 99999], count)
                    case[2][:, count] = 0
                    case[3][:] = count
                    out = numpy_reference(*case, 0.0625)
                    self.assertTrue(np.isfinite(out).all())
                    np.testing.assert_array_equal(out, 0)

    def test_tile_and_split_boundary_prefixes(self):
        for rope in (0, 64):
            for page_size in (64, 128):
                for count in (0, 1, 63, 64, 65, 127, 128, 129,
                              255, 256, 257, 511, 512, 513):
                    with self.subTest(rope=rope, page_size=page_size, count=count):
                        case = make_case(rope_dim=rope, query_lengths=(1,),
                                         seq_lengths=(515,), page_size=page_size,
                                         capacity=count + 1)
                        q, cache, indices, counts, _, _, pages, _ = case
                        q.fill(0)
                        indices[0, :count] = np.arange(count)[::-1]
                        indices[0, count] = 514
                        counts[:] = count
                        out = numpy_reference(*case, 0.0625)
                        sequence = cache[pages].reshape(-1, cache.shape[-1])
                        expected = (sequence[:count, :512].mean(axis=0, dtype=np.float64)
                                    if count else np.zeros(512))
                        self.assertTrue(np.isfinite(out).all())
                        np.testing.assert_allclose(
                            out[0], np.broadcast_to(expected, out[0].shape),
                            atol=1e-6, rtol=1e-5)

    def test_explicit_scale_matches_two_token_softmax(self):
        for rope in (0, 64):
            case = make_case(rope_dim=rope, query_lengths=(1,),
                             seq_lengths=(2,), capacity=2)
            q, cache, indices, counts, _, _, pages, _ = case
            q.fill(0)
            q[:, :, -1] = 1
            cache.fill(0)
            cache[pages[0], 1, :512] = 1
            cache[pages[0], 1, -1] = 1
            indices[0] = [0, 1]
            counts[:] = 2
            for scale in (0.0001, 0.0625, 0.1, 0.125, 1.0, 16.0):
                with self.subTest(rope=rope, scale=scale):
                    out = numpy_reference(*case, scale)
                    self.assertTrue(np.isfinite(out).all())
                    np.testing.assert_allclose(
                        out, 1 / (1 + np.exp(-scale)), atol=1e-7, rtol=1e-6)

    def test_order_invariance_and_remote_token(self):
        case = make_case(query_lengths=(1,), seq_lengths=(130,), capacity=8)
        case[2][0] = [0, 64, 129, -1, -1, -1, -1, -1]
        case[3][0] = 3
        original = numpy_reference(*case, 0.0625)
        case[2][0, :3] = [129, 0, 64]
        np.testing.assert_allclose(numpy_reference(*case, 0.0625), original, atol=1e-6)
        case[2][0, :3] = [127, 128, 129]
        self.assertGreater(np.max(np.abs(numpy_reference(*case, 0.0625) - original)), 0.1)

    def test_duplicates_and_invalid_counts_are_rejected(self):
        case = make_case(query_lengths=(1,), seq_lengths=(4,), capacity=8)
        case[2][0, :2] = [0, 0]
        case[3][0] = 2
        with self.assertRaisesRegex(ValueError, "unique"):
            numpy_reference(*case, 0.0625)
        case[3][0] = 9
        with self.assertRaisesRegex(ValueError, "prefix"):
            numpy_reference(*case, 0.0625)

    def test_capacity_over_2048_and_padded_query(self):
        case = list(make_case(rope_dim=0, query_lengths=(1,), seq_lengths=(2101,), capacity=2304))
        case[0] = np.concatenate([case[0], case[0]], axis=0)
        case[2] = np.concatenate([case[2], case[2]], axis=0)
        case[3] = np.concatenate([case[3], case[3]], axis=0)
        out = numpy_reference(*case, 0.0625)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(np.linalg.norm(out[0]), 0)
        np.testing.assert_array_equal(out[1], 0)


if __name__ == "__main__":
    unittest.main()
