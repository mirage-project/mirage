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


def test_missing_required_weight_raises():
    m = _Model()
    with pytest.raises(ValueError, match=r"a\.weight"):
        m.load_weights(iter([("b.weight", torch.ones(4, 4))]))


def test_unexpected_key_raises():
    m = _Model()
    with pytest.raises(ValueError, match=r"zzz\.weight"):
        m.load_weights(iter([
            ("a.weight", torch.ones(4, 4)),
            ("b.weight", torch.ones(4, 4)),
            ("zzz.weight", torch.ones(4, 4)),
        ]))


def test_skip_weight_is_consumed_and_not_required():
    from mirage.mpk.layers._base import SKIP_WEIGHT

    class _Skipper(MPKModule):
        def __init__(self, prefix=""):
            super().__init__(prefix=prefix)
            self.weight = nn.Parameter(torch.empty(4, 4))
            # weight is filled elsewhere, so don't require it to be loaded here.
            self._optional_param_paths = frozenset({"weight"})

        def resolve_weight(self, name, params):
            if name == "weight":
                return SKIP_WEIGHT
            return super().resolve_weight(name, params)

    m = _Skipper()
    consumed = m.load_weights(iter([("weight", torch.ones(4, 4))]))
    assert consumed == {"weight"}        # skipped keys still count as consumed


def test_qwen3_qnorm_knorm_remap():
    from mirage.mpk.models.qwen3.modeling import _remap_qwen3_hf_key
    assert _remap_qwen3_hf_key("model.layers.3.self_attn.q_norm.weight") == \
        "model.layers.3.self_attn.attn.q_norm"
    assert _remap_qwen3_hf_key("model.layers.3.self_attn.k_norm.weight") == \
        "model.layers.3.self_attn.attn.k_norm"
    # non-matching keys pass through unchanged
    assert _remap_qwen3_hf_key("model.layers.3.self_attn.q_proj.weight") == \
        "model.layers.3.self_attn.q_proj.weight"
    assert _remap_qwen3_hf_key("lm_head.weight") == "lm_head.weight"
