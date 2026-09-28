"""MPK graph integration, including a downstream consumer and split reduction.

Run on SM100 after building/installing Mirage. Offline test mode schedules
one pure-prefill chunk. Historical decode and mixed request metadata are
covered by the standalone launcher, which accepts explicit page metadata.
"""

import tempfile
import unittest
from reference import make_case, torch_reference
from test_sparse_mla import HAS_SM100, assert_close, to_cuda, torch


@unittest.skipUnless(HAS_SM100, "requires SM100, CUDA PyTorch and built Mirage")
class SparseMLAMPKTests(unittest.TestCase):
    def run_case(self, rope, query_len, splits, heads=8):
        import mirage
        from mirage.mpk.persistent_kernel import PersistentKernel

        case = make_case(rope_dim=rope, heads=heads, query_lengths=(query_len,),
                         seq_lengths=(query_len,), capacity=128)
        q, cache, indices, counts, qo, ki, pages, last = to_cuda(case)
        # Use one page; the offline page allocator starts with physical page 0.
        self.assertEqual(cache.shape[0], 1)
        workers, schedulers = mirage.get_configurations_from_gpu(0)
        params = PersistentKernel.get_default_init_parameters()
        params.update(
            test_mode=True, num_workers=workers, num_local_schedulers=schedulers,
            max_seq_length=128, max_num_batched_requests=1,
            max_num_batched_tokens=query_len, max_num_pages=1, page_size=64,
            meta_tensors={"prompt_lengths": torch.tensor(
                [query_len], dtype=torch.int32, device="cuda")},
        )
        pk = PersistentKernel(**params)
        try:
            out = torch.full((query_len, heads, 512), float("nan"),
                             dtype=torch.bfloat16, device="cuda")
            q_dt = pk.attach_input(q, name="sparse_q")
            cache_dt = pk.attach_input(cache, name="sparse_cache")
            indices_dt = pk.attach_input(indices, name="sparse_indices")
            counts_dt = pk.attach_input(counts, name="sparse_counts")
            out_dt = pk.attach_input(out, name="sparse_output")
            pk.sparse_mla_layer(q_dt, cache_dt, indices_dt, counts_dt, out_dt,
                                softmax_scale=0.0625, num_splits=splits)

            # With R=0, latent output can be used as a query by another task.
            # This checks that output->consumer dependencies survive codegen.
            second = None
            if rope == 0:
                second = torch.full_like(out, float("nan"))
                second_dt = pk.attach_input(second, name="sparse_second_output")
                pk.sparse_mla_layer(out_dt, cache_dt, indices_dt, counts_dt, second_dt,
                                    softmax_scale=0.125, num_splits=1)
            expected = torch_reference(q, cache, indices, counts, qo, ki, pages, last, 0.0625)
            with tempfile.TemporaryDirectory(prefix="mirage_sparse_mla_") as build_dir:
                pk.compile(output_dir=build_dir)
                pk()
                torch.cuda.synchronize()
                assert_close(out, expected)
                for start in range(0, heads, 16):
                    assert_close(out[:, start:start + 16], expected[:, start:start + 16])
                if second is not None:
                    expected_second = torch_reference(
                        out, cache, indices, counts, qo, ki, pages, last, 0.125)
                    assert_close(second, expected_second)
                    for start in range(0, heads, 16):
                        assert_close(second[:, start:start + 16],
                                     expected_second[:, start:start + 16])
        finally:
            pk.finalize()

    def test_decode_sized_query(self):
        for rope in (0, 64):
            for splits in (1, 8):
                with self.subTest(rope=rope, splits=splits):
                    self.run_case(rope, 1, splits)

    def test_multiple_queries_and_consumer(self):
        for rope in (0, 64):
            for splits in (1, 4):
                with self.subTest(rope=rope, splits=splits):
                    self.run_case(rope, 5, splits)

    def test_multiple_head_groups_and_consumer(self):
        for rope in (0, 64):
            for heads in (32, 64):
                for splits in (1, 2, 4, 8):
                    with self.subTest(rope=rope, heads=heads, splits=splits):
                        self.run_case(rope, 5, splits, heads=heads)


if __name__ == "__main__":
    unittest.main()
