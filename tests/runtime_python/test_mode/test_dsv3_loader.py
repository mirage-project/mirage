"""Pure CPU unit tests for the DeepSeek V3 streaming loader helpers (Task C2).

We only test the *pure* pieces that don't need a live PersistentKernel:

  * ``_remap_dsv3_key`` — the HF-key -> catalog ``named_parameters()`` path map.
  * ``_parse_expert_key`` — the routed-expert key regex parser.
  * ``_is_out_of_range_layer_key`` — detect MTP / reduced-layer keys to skip.

The full ``DeepseekV3ForCausalLM.load_weights`` needs a constructed model
(live PK + CUDA), so it is exercised by the demo, not here.
"""

import os
import sys

import pytest

# Make ``import mirage`` resolve to the in-repo package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_PKG_ROOT = os.path.join(_REPO_ROOT, "python")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from mirage.mpk.models.deepseek_v3.modeling import (  # noqa: E402
    _is_out_of_range_layer_key,
    _parse_expert_key,
    _remap_dsv3_key,
)


# ---------------------------------------------------------------------------
# _remap_dsv3_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hf_key, catalog_path",
    [
        # --- Global tensors (pass through unchanged: child-module .weight). ---
        ("model.embed_tokens.weight", "model.embed_tokens.weight"),
        ("model.norm.weight", "model.norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
        # --- Decoder-layer layernorms (child-module .weight, unchanged). ---
        (
            "model.layers.3.input_layernorm.weight",
            "model.layers.3.input_layernorm.weight",
        ),
        (
            "model.layers.3.post_attention_layernorm.weight",
            "model.layers.3.post_attention_layernorm.weight",
        ),
        # --- MLA layernorms (raw nn.Parameter: drop trailing .weight). ---
        (
            "model.layers.2.self_attn.q_a_layernorm.weight",
            "model.layers.2.self_attn.q_a_layernorm",
        ),
        (
            "model.layers.2.self_attn.kv_a_layernorm.weight",
            "model.layers.2.self_attn.kv_a_layernorm",
        ),
        # --- MLA directly-written projections (raw nn.Parameter: _proj.weight -> _proj_weight). ---
        (
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.q_a_proj_weight",
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.kv_a_proj_with_mqa_weight",
        ),
        # --- MoE router gate matrix (raw nn.Parameter). ---
        ("model.layers.5.mlp.gate.weight", "model.layers.5.mlp.gate_weight"),
        # --- MoE shared experts (raw nn.Parameter: shared_<proj>_proj_weight). ---
        (
            "model.layers.5.mlp.shared_experts.gate_proj.weight",
            "model.layers.5.mlp.shared_gate_proj_weight",
        ),
        (
            "model.layers.5.mlp.shared_experts.up_proj.weight",
            "model.layers.5.mlp.shared_up_proj_weight",
        ),
        (
            "model.layers.5.mlp.shared_experts.down_proj.weight",
            "model.layers.5.mlp.shared_down_proj_weight",
        ),
        # --- Dense-MLP projections (raw nn.Parameter: <proj>_proj_weight). ---
        (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_proj_weight",
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj_weight",
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.down_proj_weight",
        ),
    ],
)
def test_remap_dsv3_key(hf_key, catalog_path):
    assert _remap_dsv3_key(hf_key) == catalog_path


# ---------------------------------------------------------------------------
# _parse_expert_key
# ---------------------------------------------------------------------------


def test_parse_expert_key_gate():
    assert _parse_expert_key(
        "model.layers.5.mlp.experts.37.gate_proj.weight"
    ) == (5, 37, "gate")


def test_parse_expert_key_up_and_down():
    assert _parse_expert_key(
        "model.layers.10.mlp.experts.0.up_proj.weight"
    ) == (10, 0, "up")
    assert _parse_expert_key(
        "model.layers.60.mlp.experts.255.down_proj.weight"
    ) == (60, 255, "down")


@pytest.mark.parametrize(
    "non_expert_key",
    [
        # Shared-expert keys are NOT routed-expert keys.
        "model.layers.5.mlp.shared_experts.gate_proj.weight",
        # Router gate is not an expert weight.
        "model.layers.5.mlp.gate.weight",
        # MLA / layernorm / global keys.
        "model.layers.5.self_attn.q_b_proj.weight",
        "model.layers.5.input_layernorm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        # The expert's fp8 scale companion is not the weight itself.
        "model.layers.5.mlp.experts.37.gate_proj.weight_scale_inv",
    ],
)
def test_parse_expert_key_none(non_expert_key):
    assert _parse_expert_key(non_expert_key) is None


# ---------------------------------------------------------------------------
# _is_out_of_range_layer_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        # MTP keys live at model.layers.<num_hidden_layers>.* (here: 61);
        # with built_layers=4 they are out of range.
        "model.layers.61.eh_proj.weight",
        "model.layers.61.enorm.weight",
        "model.layers.61.hnorm.weight",
        "model.layers.61.shared_head.norm.weight",
        # Reduced-layer run: checkpoint has layer 5 but only 4 were built.
        "model.layers.5.mlp.gate.weight",
        "model.layers.4.self_attn.q_a_proj.weight",  # boundary: idx == built
        # fp8 scale companion for an out-of-range layer.
        "model.layers.61.eh_proj.weight_scale_inv",
        "model.layers.5.mlp.experts.0.gate_proj.weight",
    ],
)
def test_is_out_of_range_layer_key_skip(key):
    assert _is_out_of_range_layer_key(key, built_layers=4) is True


@pytest.mark.parametrize(
    "key",
    [
        # In-range built layers (built_layers=4 -> valid indices 0..3).
        "model.layers.0.input_layernorm.weight",
        "model.layers.2.self_attn.q_a_proj.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.experts.255.down_proj.weight",
        # Non-layer keys are never out-of-range (global tensors).
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ],
)
def test_is_out_of_range_layer_key_keep(key):
    assert _is_out_of_range_layer_key(key, built_layers=4) is False
