import torch
from mirage.mpk.layers.moe.w13 import MoEW13BF16
from mirage.mpk.layers.moe.w2 import MoEW2BF16


def test_local_expert_param_shape():
    w = MoEW13BF16(num_experts=8, num_experts_per_tok=2, hidden_size=8,
                   intermediate_size=4, ep_size=2, ep_rank=1)
    assert w.weight.shape[0] == 4            # 8 experts / ep_size 2 = 4 local
    assert w.num_local_experts == 4
    assert w.local_expert_start == 4         # rank 1 owns experts 4..7


def test_loader_writes_local_skips_remote():
    w = MoEW13BF16(num_experts=8, num_experts_per_tok=2, hidden_size=8,
                   intermediate_size=4, ep_size=2, ep_rank=0)  # owns 0..3
    inter, hidden = 4, 8
    src = torch.full((inter, hidden), 9.0, dtype=torch.bfloat16)
    assert w.weight_loader(w.weight, src, expert_id=5, slot="gate") is False   # non-local
    assert w.weight_loader(w.weight, src, expert_id=2, slot="gate") is True     # local
    assert torch.equal(w.weight[2, :inter], src)


def test_w2_local_loader():
    w = MoEW2BF16(num_experts=8, num_experts_per_tok=2, hidden_size=8,
                  intermediate_size=4, ep_size=2, ep_rank=0)
    assert w.weight.shape[0] == 4
    down = torch.full((8, 4), 3.0, dtype=torch.bfloat16)   # (hidden, intermediate)
    assert w.weight_loader(w.weight, down, expert_id=7) is False  # non-local on rank 0
    assert w.weight_loader(w.weight, down, expert_id=1) is True
    assert torch.equal(w.weight[1], down)
