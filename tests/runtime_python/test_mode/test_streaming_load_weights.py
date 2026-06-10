import weakref
import pytest
import torch
import torch.nn as nn
from mirage.mpk.layers._base import MPKModule


class _Leaf(MPKModule):
    def __init__(self, prefix=""):
        super().__init__(prefix=prefix)
        self.weight = nn.Parameter(torch.empty(4, 4))


class _Model(MPKModule):
    def __init__(self, prefix=""):
        super().__init__(prefix=prefix)
        self.a = _Leaf()
        self.b = _Leaf()


def _tracking_iter(n, alive):
    refs = []
    for i in range(n):
        t = torch.ones(4, 4)
        refs.append(weakref.ref(t))
        yield f"{'a' if i % 2 == 0 else 'b'}.weight", t
        alive.append(sum(1 for r in refs if r() is not None))


def test_streaming_releases_each_tensor():
    m = _Model()
    alive = []
    consumed = m.load_weights(_tracking_iter(2, alive))
    assert consumed == {"a.weight", "b.weight"}
    assert max(alive) <= 1, f"held {max(alive)} tensors live (expected streaming)"


def test_missing_key_raises():
    m = _Model()
    with pytest.raises(ValueError, match="a.weight"):
        m.load_weights(iter([("b.weight", torch.ones(4, 4))]))


def test_unexpected_key_raises():
    m = _Model()
    with pytest.raises(ValueError, match="zzz.weight"):
        m.load_weights(iter([
            ("a.weight", torch.ones(4, 4)),
            ("b.weight", torch.ones(4, 4)),
            ("zzz.weight", torch.ones(4, 4)),
        ]))
