"""CPU tests for fixed-batch validation, dispatch and per-request EOS."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python/mirage/mpk/models'))
from qwen3_5.model import Qwen35Model
from qwen3_5.batch import linear_grid, validate_batch_size, validate_offline_batch_size


def model(batch=3):
    m = Qwen35Model.__new__(Qwen35Model)
    m.offline = False
    m.batch_size, m.position = batch, 0
    m.config = SimpleNamespace(vocab_size=100, eos_token_id=99)
    m._compiled, m._closed, m._prefill_graph = True, False, None
    m.tensors = {'token': torch.zeros(batch, dtype=torch.int64),
                 'next_token': torch.zeros(batch, dtype=torch.int64),
                 'logits': torch.zeros(batch, 100)}
    m.layers = []
    m.pk = Mock()
    return m


@pytest.mark.parametrize('value', [0, -1, 33, True, 2.0, '2', None])
def test_invalid_batch_size_before_checkpoint_loading(value):
    with pytest.raises(ValueError, match='batch_size'):
        Qwen35Model.from_pretrained('/nonexistent', batch_size=value)


@pytest.mark.parametrize('value', [1, 2, 3, 4, 8, 16, 32])
def test_batch_sizes(value):
    validate_batch_size(value)


def test_offline_b8_requires_explicit_experiment_flag(monkeypatch):
    monkeypatch.delenv("MPK_QWEN_EXPERIMENTAL_B8", raising=False)
    validate_offline_batch_size(1)
    with pytest.raises(ValueError, match="exact-token validation is incomplete"):
        validate_offline_batch_size(8)
    monkeypatch.setenv("MPK_QWEN_EXPERIMENTAL_B8", "1")
    validate_offline_batch_size(8)


def test_b16_linear_partition_is_opt_in(monkeypatch):
    monkeypatch.delenv("MPK_QWEN38_B16_LINEAR_COLS", raising=False)
    assert linear_grid(5120, 16) == (80, 1, 1)
    assert linear_grid(5120, 16, optimized_b8=True) == (40, 1, 1)
    assert linear_grid(96, 16, 32, optimized_b8=True) == (3, 1, 1)
    monkeypatch.setenv("MPK_QWEN38_B16_LINEAR_COLS", "64")
    assert linear_grid(5120, 16, optimized_b8=True) == (80, 1, 1)


@pytest.mark.parametrize('tokens', [[], [1, 2, 3], [[1], [2]], [[1], [2, 3], [4]],
                                  [[1], [True], [3]], [[1], [100], [3]], [[1], [-1], [3]]])
def test_bad_prefill_is_atomic(tokens):
    m = model()
    m._prefill_block = Mock()
    with pytest.raises(ValueError):
        m.prefill(tokens)
    m.pk.assert_not_called()
    m._prefill_block.assert_not_called()
    assert m.position == 0


def test_block_and_stream_dispatch():
    m = model()
    m._prefill_block = Mock(return_value=m.tensors['logits'])
    m.prefill([[1, 2], [3, 4], [5, 6]])
    m._prefill_block.assert_called_once_with([[1, 2], [3, 4], [5, 6]])
    with patch('torch.cuda.synchronize'):
        m.prefill([[1, 2], [3, 4], [5, 6]], mode='stream')
    assert m.pk.call_count == 2  # One persistent launch per column, not per row.
    assert m.position == 2
    assert m.tensors['token'].tolist() == [2, 4, 6]


def test_step_capacity_and_shape_are_atomic():
    m = model()
    for tokens in ([1, 2], 1, [1, 2, True], [1, 2, 100]):
        with pytest.raises(ValueError): m.step(tokens)
    m.position = 1024
    with pytest.raises(ValueError, match='capacity'): m.step([1, 2, 3])
    m.pk.assert_not_called()
    assert m.tensors['token'].tolist() == [0, 0, 0]


def test_batch_eos_is_independent():
    m = model()
    m.prefill = Mock()
    m.tensors['next_token'][:] = torch.tensor([99, 5, 7])
    columns = iter([[8, 99, 8], [9, 9, 99]])
    def step(tokens): m.tensors['next_token'][:] = torch.tensor(next(columns))
    m.step = Mock(side_effect=step)
    assert m.generate([[1, 2], [3, 4], [5, 6]], 8) == [[99], [5, 99], [7, 8, 99]]
    assert m.step.call_count == 2


def test_capacity_generation_and_zero_length():
    m = model()
    assert m.generate([[1], [2], [3]], 0) == [[], [], []]
    with pytest.raises(ValueError, match='capacity'):
        m.generate([[1]*1024]*3, 2)
    with pytest.raises(ValueError, match='equal prompt lengths'):
        m.generate([[1], [2, 3], [4]], 0)
    with pytest.raises(ValueError, match='batch_size=1'):
        m.generate([[1], [2], [3]], 0, prefill_mode='mpk')
    m.pk.assert_not_called()
