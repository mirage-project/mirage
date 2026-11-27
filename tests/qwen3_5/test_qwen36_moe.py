"""CPU shape and MoE arithmetic checks; GPU parity is a separate gate."""

import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/mirage/mpk/models"))
from qwen3_5.moe_config import Qwen36MoeTextConfig
from qwen3_5.weight_loader import Qwen36WeightLoader, expected_text_shapes
from qwen3_5.moe_reference import moe_forward, sum_expert_slots


def config_dict():
    data = json.loads((Path(__file__).parent / "fixtures/qwen3_8_27b_config.json").read_text())
    data["model_type"] = "qwen3_5_moe"
    text = data["text_config"]
    text.update(model_type="qwen3_5_moe_text", hidden_size=2048, num_hidden_layers=40,
                layer_types=["full_attention" if i % 4 == 3 else "linear_attention"
                             for i in range(40)], num_attention_heads=16,
                num_key_value_heads=2, linear_num_value_heads=32,
                moe_intermediate_size=512, shared_expert_intermediate_size=512,
                num_experts=256, num_experts_per_tok=8)
    text.pop("intermediate_size", None)
    return data


def test_official_moe_geometry_and_checkpoint_names(tmp_path):
    data = config_dict()
    (tmp_path / "config.json").write_text(json.dumps(data))
    c = Qwen36WeightLoader(tmp_path).config
    assert isinstance(c, Qwen36MoeTextConfig)
    assert (len(c.gdn_layers), len(c.attention_layers)) == (30, 10)
    shapes = expected_text_shapes(c)
    assert shapes["layers.0.linear_attn.in_proj_qkv.weight"] == (8192, 2048)
    assert shapes["layers.3.self_attn.q_proj.weight"] == (8192, 2048)
    assert shapes["layers.3.self_attn.k_proj.weight"] == (512, 2048)
    assert shapes["layers.0.mlp.gate.weight"] == (256, 2048)
    assert shapes["layers.0.mlp.experts.gate_up_proj"] == (256, 1024, 2048)
    assert shapes["layers.0.mlp.experts.down_proj"] == (256, 2048, 512)
    assert shapes["layers.0.mlp.shared_expert_gate.weight"] == (1, 2048)
    assert "layers.0.mlp.gate_proj.weight" not in shapes


def test_moe_config_rejects_invalid_routing():
    data = config_dict()
    data["text_config"]["num_experts_per_tok"] = 257
    with pytest.raises(ValueError, match="exceeds"):
        Qwen36MoeTextConfig.from_dict(data)


def test_moe_reference_routes_and_gates_shared_expert():
    torch.manual_seed(36)
    hidden = (torch.randn(3, 8) * 0.2).bfloat16()
    rand = lambda *shape: (torch.randn(*shape) * 0.1).bfloat16()
    w = {"gate.weight": rand(4, 8),
         "experts.gate_up_proj": rand(4, 12, 8),
         "experts.down_proj": rand(4, 8, 6),
         "shared_expert.gate_proj.weight": rand(6, 8),
         "shared_expert.up_proj.weight": rand(6, 8),
         "shared_expert.down_proj.weight": rand(8, 6),
         "shared_expert_gate.weight": rand(1, 8)}
    actual, indices, values = moe_forward(hidden, w, top_k=2)
    # Independently gather by token, reduce slots in FP32, and add the shared MLP.
    logits = F.linear(hidden, w["gate.weight"])
    probabilities = torch.softmax(logits.float(), -1)
    reference_indices = probabilities.argsort(dim=-1, descending=True)[:, :2]
    selected = probabilities.gather(1, reference_indices)
    reference_values = (selected / selected.sum(-1, keepdim=True)).bfloat16()
    torch.testing.assert_close(indices, reference_indices)
    torch.testing.assert_close(values, reference_values, rtol=0, atol=0)
    expected = []
    for row in range(3):
        contributions = []
        for slot, expert in enumerate(indices[row].tolist()):
            gate_part, up_part = F.linear(hidden[row], w["experts.gate_up_proj"][expert]).chunk(2)
            down = F.linear(F.silu(gate_part) * up_part, w["experts.down_proj"][expert])
            contributions.append(down * values[row, slot])
        routed = torch.stack(contributions).sum(dim=0)
        shared = F.linear(F.silu(F.linear(hidden[row], w["shared_expert.gate_proj.weight"])) *
                          F.linear(hidden[row], w["shared_expert.up_proj.weight"]),
                          w["shared_expert.down_proj.weight"])
        scale = torch.sigmoid(F.linear(hidden[row], w["shared_expert_gate.weight"]))
        expected.append(routed + scale * shared)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)


def test_moe_reduction_uses_grouped_fp32_slot_sum():
    # Expert-major BF16 additions would round 256 + 1 back to 256 before
    # subtracting 256. grouped_mm sums the already weighted slots in FP32.
    outputs = torch.tensor([[[256.0], [-256.0], [1.0]]], dtype=torch.bfloat16)
    scores = torch.ones((1, 3), dtype=torch.float32)
    routing = torch.tensor([[1], [3], [2]], dtype=torch.int32)
    shared = torch.zeros((1, 1), dtype=torch.bfloat16)
    result = sum_expert_slots(outputs, scores, routing, shared)
    assert result.item() == 1.0
    eager = torch.zeros(1, dtype=torch.bfloat16)
    for slot in (0, 2, 1):
        eager += outputs[0, slot]
    assert eager.item() == 0.0
