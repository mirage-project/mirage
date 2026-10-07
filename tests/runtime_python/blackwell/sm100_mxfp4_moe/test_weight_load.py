"""Weight-load smoke test for a GPT-OSS MXFP4 checkpoint.

The released 120B file (openai/gpt-oss-120b, shard model-00009 header) stores
expert blocks and scales as uint8 and expert biases as bf16. Attention,
router, embeddings and lm_head are bf16. ``torch_dtype: bfloat16`` on a
config, and ``torch.set_default_dtype(bfloat16)`` in the demo, must not
make the loader treat those packed experts as a bf16 GEMM.
"""

import json
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "python")))

from mirage.mpk.models.gpt_oss.mxfp4 import (
    Mxfp4Checkpoint,
    checkpoint_expert_format,
    convert_expert_weight,
)

# openai/gpt-oss-120b model-00009-of-00014.safetensors header.
RELEASED_GATE_UP_BLOCKS = (128, 5760, 90, 16)  # U8, 90 * 16 = 2880 / 2
RELEASED_GATE_UP_SCALES = (128, 5760, 90)  # U8
RELEASED_GATE_UP_BIAS = (128, 5760)  # BF16
RELEASED_DOWN_BLOCKS = (128, 2880, 90, 16)  # U8
RELEASED_DOWN_SCALES = (128, 2880, 90)  # U8
RELEASED_DOWN_BIAS = (128, 2880)  # BF16


def _gpt_oss_config(**extra):
    cfg = {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "num_hidden_layers": 2,
        "num_local_experts": 4,
        "vocab_size": 201088,
        "hidden_size": 64,
        "intermediate_size": 64,
        "head_dim": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "sliding_window": 128,
        "num_experts_per_tok": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "eos_token_id": 200002,
        "rope_theta": 150000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
    }
    cfg.update(extra)
    return cfg


def _write_config(directory, **extra):
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(_gpt_oss_config(**extra), f)


def _write_sharded(directory, tensors):
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(directory, shard))
    index = {"metadata": {}, "weight_map": {key: shard for key in tensors}}
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


def test_released_120b_shapes():
    assert RELEASED_GATE_UP_BLOCKS == (128, 5760, 90, 16)
    assert RELEASED_GATE_UP_SCALES == (128, 5760, 90)
    assert RELEASED_GATE_UP_BIAS == (128, 5760)
    assert RELEASED_DOWN_BLOCKS == (128, 2880, 90, 16)
    assert RELEASED_DOWN_SCALES == (128, 2880, 90)
    assert RELEASED_DOWN_BIAS == (128, 2880)
    # 90 scale blocks * 16 bytes = K/2, K = hidden = intermediate = 2880.
    assert 90 * 16 == 2880 // 2


def _tiny_experts():
    # Same rank as the release: blocks [E, rows, K/32, 16], scales [E, rows, K/32].
    e, rows, k_blocks = 2, 4, 1
    blocks = torch.zeros(e, rows, k_blocks, 16, dtype=torch.uint8)
    blocks[0, 0, 0, 0] = 0x1A  # even row, even K
    blocks[0, 1, 0, 0] = 0x2B  # odd row
    scales = torch.full((e, rows, k_blocks), 127, dtype=torch.uint8)
    bias = torch.arange(e * rows, dtype=torch.bfloat16).reshape(e, rows)
    return blocks, scales, bias


def test_mxfp4_config_stays_packed_under_bf16_default():
    torch.set_default_dtype(torch.bfloat16)
    blocks, scales, bias = _tiny_experts()
    down_blocks = blocks.clone()
    down_scales = scales.clone()
    down_bias = bias.clone()
    with tempfile.TemporaryDirectory() as directory:
        _write_config(
            directory,
            torch_dtype="bfloat16",
            quantization_config={
                "modules_to_not_convert": [
                    "model.layers.*.self_attn",
                    "model.layers.*.mlp.router",
                    "model.embed_tokens",
                    "lm_head",
                ],
                "quant_method": "mxfp4",
            },
        )
        prefix = "model.layers.0.mlp.experts."
        _write_sharded(directory, {
            prefix + "gate_up_proj_blocks": blocks,
            prefix + "gate_up_proj_scales": scales,
            prefix + "gate_up_proj_bias": bias,
            prefix + "down_proj_blocks": down_blocks,
            prefix + "down_proj_scales": down_scales,
            prefix + "down_proj_bias": down_bias,
        })
        config = AutoConfig.from_pretrained(directory)
        assert config.dtype == torch.bfloat16
        assert checkpoint_expert_format(config, directory) == "mxfp4"

        loaded = Mxfp4Checkpoint(directory)
        gu_b, gu_s, gu_bias = convert_expert_weight(
            loaded[prefix + "gate_up_proj_blocks"],
            loaded[prefix + "gate_up_proj_scales"],
            loaded[prefix + "gate_up_proj_bias"],
            deinterleave=True,
        )
        dn_b, dn_s, dn_bias = convert_expert_weight(
            loaded[prefix + "down_proj_blocks"],
            loaded[prefix + "down_proj_scales"],
            loaded[prefix + "down_proj_bias"],
            deinterleave=False,
        )
        assert gu_b.dtype == torch.uint8 and gu_s.dtype == torch.uint8
        assert dn_b.dtype == torch.uint8 and dn_s.dtype == torch.uint8
        assert gu_bias.dtype == torch.bfloat16 and dn_bias.dtype == torch.bfloat16
        # rows 0,2 then 1,3. The odd-row byte moves to row index 2.
        assert gu_b.shape == (2, 4, 16)
        assert int(gu_b[0, 0, 0]) == 0x1A
        assert int(gu_b[0, 2, 0]) == 0x2B
        assert int(dn_b[0, 0, 0]) == 0x1A
        assert int(dn_b[0, 1, 0]) == 0x2B
        if torch.cuda.is_available():
            on_gpu = gu_b.contiguous().to("cuda")
            assert on_gpu.dtype == torch.uint8


def test_packed_files_without_quant_config_are_not_bf16():
    """A dropped quantization_config must not select the bf16 AutoModel path."""
    torch.set_default_dtype(torch.bfloat16)
    blocks, scales, bias = _tiny_experts()
    with tempfile.TemporaryDirectory() as directory:
        _write_config(directory, torch_dtype="bfloat16")
        _write_sharded(directory, {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": blocks,
            "model.layers.0.mlp.experts.gate_up_proj_scales": scales,
            "model.layers.0.mlp.experts.gate_up_proj_bias": bias,
        })
        config = AutoConfig.from_pretrained(directory)
        assert not hasattr(config, "quantization_config") or (
            getattr(config, "quantization_config", None) in (None, {})
        )
        assert checkpoint_expert_format(config, directory) == "mxfp4"


def test_unpacked_checkpoint_stays_bf16():
    torch.set_default_dtype(torch.bfloat16)
    weight = torch.zeros(2, 8, 4, dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as directory:
        _write_config(directory, torch_dtype="bfloat16")
        _write_sharded(directory, {
            "model.layers.0.mlp.experts.gate_up_proj": weight,
        })
        config = AutoConfig.from_pretrained(directory)
        assert checkpoint_expert_format(config, directory) == "bf16"


def test_float_blocks_are_rejected():
    blocks = torch.zeros(2, 4, 16, dtype=torch.bfloat16)
    scales = torch.zeros(2, 4, 1, dtype=torch.uint8)
    bias = torch.zeros(2, 4, dtype=torch.bfloat16)
    try:
        convert_expert_weight(blocks, scales, bias, deinterleave=False)
    except TypeError as exc:
        assert "uint8" in str(exc)
    else:
        raise AssertionError("bf16 blocks were accepted as MXFP4")


def test_fp32_bias_is_cast_to_bf16():
    blocks = torch.zeros(2, 4, 1, 16, dtype=torch.uint8)
    scales = torch.zeros(2, 4, 1, dtype=torch.uint8)
    bias = torch.ones(2, 4, dtype=torch.float32)
    _, _, out_bias = convert_expert_weight(
        blocks, scales, bias, deinterleave=False)
    assert out_bias.dtype == torch.bfloat16
    assert out_bias[0, 0].item() == 1.0


if __name__ == "__main__":
    test_released_120b_shapes()
    test_mxfp4_config_stays_packed_under_bf16_default()
    test_packed_files_without_quant_config_are_not_bf16()
    test_unpacked_checkpoint_stays_bf16()
    test_float_blocks_are_rejected()
    test_fp32_bias_is_cast_to_bf16()
    print("PASS gpt-oss mxfp4 weight load")
