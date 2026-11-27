"""CPU-only request-lifecycle boundary tests; GPU parity is tested separately."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/mirage/mpk/models"))
from qwen3_5.model import Qwen35Model
from qwen3_5.gdn import Qwen35GDNProbe
from qwen3_5.mpk_prefill import flash_split_config


class RequestContractTests(unittest.TestCase):
    def test_offline_rejects_unwired_sdpa_before_gpu_allocation(self):
        with self.assertRaisesRegex(ValueError, "eager attention only"):
            Qwen35Model(None, None, offline=True, attn_implementation="sdpa")

    def test_offline_requires_every_attention_cache_page(self):
        config = SimpleNamespace(num_hidden_layers=64, hidden_size=5120,
                                 intermediate_size=17408, vocab_size=248320,
                                 attention_layers=(3, 7))
        pk = SimpleNamespace(test_mode=False, mode="offline", world_size=1,
                             max_num_batched_tokens=1, max_num_batched_requests=1,
                             max_tokens_per_request=1, max_seq_length=32,
                             max_num_pages=1, page_size=1024)
        with self.assertRaisesRegex(ValueError, "Every offline attention layer"):
            Qwen35Model(config, None, pk=pk, offline=True, kv_cache_pairs={3: (None, None)})

    def test_invalid_attention_backend_rejected_before_loading_or_gpu_use(self):
        for value in (None, "flash", "SDPA", 1, [], True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "attn_implementation"):
                    Qwen35Model.from_pretrained("/does-not-exist", attn_implementation=value)
                with self.assertRaisesRegex(ValueError, "attn_implementation"):
                    Qwen35Model(None, None, attn_implementation=value)

    def test_sdpa_rejects_disabled_flash_before_allocating_a_kernel(self):
        config = SimpleNamespace(num_hidden_layers=64, hidden_size=5120,
                                 intermediate_size=17408, vocab_size=248320)
        with patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_capability", return_value=(9, 0)), \
             patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(multi_processor_count=114)), \
             patch("torch.backends.cuda.flash_sdp_enabled", return_value=False):
            with self.assertRaisesRegex(ValueError, "FlashAttention backend"):
                Qwen35Model(config, None, attn_implementation="sdpa")

    def model(self):
        model = Qwen35Model.__new__(Qwen35Model)
        model.offline = False
        model.config = SimpleNamespace(vocab_size=248320, eos_token_id=248044)
        model.position = 0
        model._compiled, model._closed = True, False
        model._prefill_graph = None
        model.tensors = {"next_token": torch.tensor([42])}
        model.calls = []
        model.block_calls = []
        def step(token):
            model.calls.append(token)
            model.position += 1
            return torch.tensor([token])
        model.step = step
        def block(tokens):
            model.block_calls.append(list(tokens))
            model.calls.extend(tokens)
            model.position += len(tokens)
            return torch.tensor([tokens[-1]])
        model._prefill_block = block
        return model

    def test_default_prefill_dispatches_one_whole_chunk(self):
        model = self.model()
        model.prefill([7, 8, 9])
        self.assertEqual(model.block_calls, [[7, 8, 9]])
        self.assertEqual(model.position, 3)

    def test_mpk_prefill_requires_compiled_graph_before_mutation(self):
        model = self.model()
        with self.assertRaisesRegex(RuntimeError, "prefill_chunk_size=64"):
            model.prefill([7, 8], mode="mpk")
        self.assertEqual(model.calls, [])
        self.assertEqual(model.position, 0)

    def test_mpk_prefill_dispatch_and_one_token_decode_bridge(self):
        model = self.model()
        model._prefill_graph = Mock()
        model.prefill([7, 8], mode="mpk")
        model._prefill_graph.run.assert_called_once_with([7, 8])
        self.assertEqual(model.calls, [])
        model.prefill([9], mode="mpk")
        self.assertEqual(model.calls, [9])

    def test_mpk_generate_passes_mode_without_torch_prefill(self):
        model = self.model()
        model._prefill_graph = Mock()
        self.assertEqual(model.generate([7, 8], 1, prefill_mode="mpk"), [42])
        model._prefill_graph.run.assert_called_once_with([7, 8])
        self.assertEqual(model.block_calls, [])

    def test_invalid_compile_tile_rejected_without_gpu_or_compile(self):
        for size in (0, 32, 128, True, 64.0, "64"):
            model = self.model()
            model._compiled = False
            model.pk = Mock()
            with self.assertRaisesRegex(ValueError, "prefill_chunk_size"):
                model.compile("unused", prefill_chunk_size=size)
            model.pk.compile.assert_not_called()

    def test_close_finalizes_both_graphs_once(self):
        model = self.model()
        model.pk, model._prefill_graph = Mock(), Mock()
        model.close()
        model.close()
        model.pk.finalize.assert_called_once()
        model._prefill_graph.close.assert_called_once()

    def test_flash_split_ranges_depend_on_full_prompt(self):
        expected = [(1,1),(2,1),(3,1),(4,1),(5,1),(3,2),(4,2),(3,3),
                    (3,3),(5,2),(2,6),(2,6),(2,7),(2,7),(5,3),(8,2)]
        for blocks, config in enumerate(expected, 1):
            self.assertEqual(flash_split_config(blocks*64, blocks*64, 114), config)
        self.assertEqual(flash_split_config(2, 1024, 114), (16,1))

    def test_compile_failure_closes_both_graphs(self):
        model = self.model()
        model._compiled = False
        model.pk = Mock()
        model.pk.compile.side_effect = RuntimeError("injected compile failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            model.compile("unused")
        self.assertTrue(model._closed)
        model.pk.finalize.assert_called_once()

    def test_stream_prefill_is_explicit_and_can_follow_block(self):
        model = self.model()
        model.prefill([7, 8])
        model.prefill([9, 10], mode="stream")
        self.assertEqual(model.block_calls, [[7, 8]])
        self.assertEqual(model.calls, [7, 8, 9, 10])
        self.assertEqual(model.position, 4)

    def test_generate_forwards_prefill_mode(self):
        for mode in ("block", "stream"):
            model = self.model()
            self.assertEqual(model.generate([7, 8], 1, prefill_mode=mode), [42])
            self.assertEqual(model.block_calls, [[7, 8]] if mode == "block" else [])

    def test_unknown_prefill_mode_is_atomic_including_zero_generation(self):
        for mode in (None, "blocks", 1):
            for method in ("prefill", "generate"):
                model = self.model()
                with self.assertRaisesRegex(ValueError, "prefill mode"):
                    if method == "prefill":
                        model.prefill([7, 8], mode=mode)
                    else:
                        model.generate([7, 8], 0, prefill_mode=mode)
                self.assertEqual(model.calls, [])
                self.assertEqual(model.position, 0)

    def test_prefill_requires_live_compiled_model(self):
        for compiled, closed in ((False, False), (True, True)):
            model = self.model()
            model._compiled, model._closed = compiled, closed
            with self.assertRaisesRegex(RuntimeError, "Compile"):
                model.prefill([7, 8])
            self.assertEqual(model.calls, [])

    def test_invalid_prompt_is_atomic(self):
        for prompt in ([], [1, -1], [1, 248320], [1, True], [1, 2.5]):
            model = self.model()
            with self.assertRaises(ValueError):
                model.prefill(prompt)
            self.assertEqual(model.calls, [])

    def test_prefill_overflow_is_atomic(self):
        model = self.model()
        model.position = 1023
        with self.assertRaises(ValueError):
            model.prefill([0, 1])
        self.assertEqual(model.calls, [])
        model.prefill([0])
        self.assertEqual(model.position, 1024)

    def test_generated_last_token_does_not_need_a_cache_slot(self):
        model = self.model()
        self.assertEqual(model.generate([1] * 1024, 1), [42])
        self.assertEqual(len(model.calls), 1024)
        model = self.model()
        with self.assertRaises(ValueError):
            model.generate([1] * 1024, 2)
        self.assertEqual(model.calls, [])

    def test_eos_stops_without_feeding_it_back(self):
        model = self.model()
        self.assertEqual(model.generate([7], 10, eos_token_ids=[42]), [42])
        self.assertEqual(model.calls, [7])

    def test_default_eos_comes_from_checkpoint(self):
        model = self.model()
        model.config.eos_token_id = 42
        self.assertEqual(model.generate([7], 10), [42])
        self.assertEqual(model.calls, [7])

    def test_none_uses_checkpoint_eos(self):
        model = self.model()
        model.config.eos_token_id = 42
        self.assertEqual(model.generate([7], 10, eos_token_ids=None), [42])
        self.assertEqual(model.calls, [7])

    def test_empty_eos_disables_default_stopping(self):
        for eos in ([], ()):
            with self.subTest(eos=eos):
                model = self.model()
                model.config.eos_token_id = 42
                self.assertEqual(model.generate([7], 3, eos_token_ids=eos), [42] * 3)
                self.assertEqual(model.calls, [7, 42, 42])
                self.assertEqual(model.config.eos_token_id, 42)

    def test_eos_override_replaces_default_without_mutating_config(self):
        model = self.model()
        model.config.eos_token_id = 42
        self.assertEqual(model.generate([7], 3, eos_token_ids=[43]), [42] * 3)
        self.assertEqual(model.calls, [7, 42, 42])
        self.assertEqual(model.config.eos_token_id, 42)

    def test_integer_and_multiple_eos_overrides(self):
        for eos in (42, [41, 42], (42, 42)):
            with self.subTest(eos=eos):
                model = self.model()
                self.assertEqual(model.generate([7], 3, eos_token_ids=eos), [42])
                self.assertEqual(model.calls, [7])

    def test_invalid_eos_is_atomic(self):
        for eos in (-1, 248320, True, 42.0, "42", [True], [42, -1],
                    [42, 248320], [42, 1.5], [42, None], [[42]]):
            with self.subTest(eos=eos):
                model = self.model()
                with self.assertRaisesRegex(ValueError, "EOS"):
                    model.generate([7], 3, eos_token_ids=eos)
                self.assertEqual(model.calls, [])
                self.assertEqual(model.position, 0)

    def test_eos_in_prompt_does_not_stop_generation(self):
        model = self.model()
        model.config.eos_token_id = 7
        self.assertEqual(model.generate([7], 2), [42, 42])
        self.assertEqual(model.calls, [7, 42])

    def test_new_request_requires_reset(self):
        model = self.model()
        model.prefill([7])
        with self.assertRaisesRegex(ValueError, "Reset"):
            model.generate([8], 1)
        self.assertEqual(model.calls, [7])

    def test_zero_generation_does_not_mutate(self):
        model = self.model()
        self.assertEqual(model.generate([7], 0), [])
        self.assertEqual(model.calls, [])

    def test_step_updates_position_for_both_mixer_types(self):
        model = Qwen35Model.__new__(Qwen35Model)
        model.offline = False
        model.config = SimpleNamespace(vocab_size=248320)
        model._compiled, model._closed, model.position = True, False, 0
        model.tensors = {"token": torch.zeros(1, dtype=torch.int64), "logits": torch.zeros(1)}
        gdn = SimpleNamespace(tensors={"position": torch.tensor([-1])}, position=0)
        attention = SimpleNamespace(tensors={"position": torch.tensor([-1]), "rope": torch.zeros(2)},
                                    rope_table=torch.tensor([[1.,2.],[3.,4.]]), position=0)
        model.layers = [SimpleNamespace(mixer=gdn), SimpleNamespace(mixer=attention)]
        observed = []
        class Kernel:
            def init_request_func(self):
                pass
            def __call__(self):
                observed.append(tuple(int(layer.mixer.tensors["position"]) for layer in model.layers))
        model.pk = Kernel()
        with patch("torch.cuda.synchronize"):
            model.step(7)
            model.step(8)
        self.assertEqual(observed, [(0,0),(1,1)])
        self.assertEqual((gdn.position,attention.position,model.position), (2,2,2))
        torch.testing.assert_close(attention.tensors["rope"],attention.rope_table[1])

    def test_gdn_reset_rearms_first_token_after_replay(self):
        probe = Qwen35GDNProbe.__new__(Qwen35GDNProbe)
        probe.config = SimpleNamespace(hidden_size=5120)
        probe.batch_size = 1
        probe._owns_kernel, probe.position, probe.max_seq_length = True, 0, 1024
        probe.tensors = {"input": torch.zeros(1,5120,dtype=torch.bfloat16),
                         "output": torch.zeros(1,5120,dtype=torch.bfloat16),
                         "position": torch.zeros(1,dtype=torch.int64),
                         "state": torch.zeros(1), "conv_state": torch.zeros(1)}
        observed = []
        class Kernel:
            def init_request_func(self):
                pass
            def __call__(self):
                observed.append(int(probe.tensors["position"]))
                probe.tensors["state"].fill_(1)
                probe.tensors["conv_state"].fill_(1)
        probe.pk = Kernel()
        hidden = torch.zeros(1,5120,dtype=torch.bfloat16)
        with patch("torch.cuda.synchronize"):
            probe.step(hidden)
            probe.step(hidden)
            probe.reset()
            self.assertEqual(probe.position,0)
            for name in ("state","conv_state","position"):
                self.assertEqual(int(probe.tensors[name].count_nonzero()),0)
            probe.step(hidden)
        self.assertEqual(observed,[0,1,0])


if __name__ == "__main__":
    unittest.main()
