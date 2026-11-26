"""Validated configuration for the unquantized, dense Qwen3.5 text decoder."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Qwen35TextConfig:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    eos_token_id: int
    tie_word_embeddings: bool
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data):
        text = data.get("text_config", data)
        if text.get("model_type") not in ("qwen3_5_text", "qwen3_5"):
            raise ValueError("Expected a dense Qwen3.5 text configuration")
        if data.get("quantization_config") or text.get("quantization_config"):
            raise ValueError("Quantized checkpoints are not supported by the BF16 loader")
        if text.get("num_experts", 0) or text.get("num_experts_per_tok", 0):
            raise ValueError("MoE checkpoints require a separate adapter")
        dtype = text.get("dtype", text.get("torch_dtype"))
        if dtype != "bfloat16":
            raise ValueError(f"Expected bfloat16 weights, got {dtype!r}")
        if text.get("mamba_ssm_dtype", "float32") != "float32":
            raise ValueError("Qwen3.5 recurrent state must be float32")
        if text.get("hidden_act") != "silu" or text.get("attention_bias", False):
            raise ValueError("Only SiLU and bias-free attention are supported")
        if text.get("attn_output_gate", True) is not True:
            raise ValueError("Qwen3.5 attention output gating must be enabled")
        integer_names = (
            "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim", "max_position_embeddings",
        )
        values = {}
        for name in integer_names:
            value = text.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            values[name] = value
        layers = tuple(text.get("layer_types", ()))
        if len(layers) != values["num_hidden_layers"] or any(
            kind not in ("linear_attention", "full_attention") for kind in layers
        ):
            raise ValueError("layer_types must describe every decoder layer")
        if values["num_attention_heads"] % values["num_key_value_heads"]:
            raise ValueError("Attention query heads must be divisible by KV heads")
        if values["linear_num_value_heads"] % values["linear_num_key_heads"]:
            raise ValueError("GDN value heads must be divisible by key heads")
        rope = text.get("rope_parameters", {})
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Only default Qwen3.5 RoPE is supported")
        fraction = float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25)))
        rotary_dim = values["head_dim"] * fraction
        if not 0 < fraction <= 1 or not rotary_dim.is_integer() or int(rotary_dim) % 2:
            raise ValueError("partial_rotary_factor must give a positive, even rotary dimension")
        theta = float(rope.get("rope_theta", 10000000.0))
        epsilon = float(text.get("rms_norm_eps", 1e-6))
        if not all(math.isfinite(x) and x > 0 for x in (theta, epsilon)):
            raise ValueError("RoPE theta and RMSNorm epsilon must be finite and positive")
        eos = text.get("eos_token_id")
        if type(eos) is not int or not 0 <= eos < values["vocab_size"]:
            raise ValueError("eos_token_id must be an integer within the vocabulary")
        tied = text.get("tie_word_embeddings", data.get("tie_word_embeddings", False))
        if type(tied) is not bool or data.get("tie_word_embeddings", tied) != tied:
            raise ValueError("Inconsistent tie_word_embeddings configuration")
        return cls(
            **values, layer_types=layers, rms_norm_eps=epsilon,
            eos_token_id=eos, tie_word_embeddings=tied,
            rope_theta=theta, partial_rotary_factor=fraction,
            mrope_section=tuple(rope.get("mrope_section", ())),
            mrope_interleaved=bool(rope.get("mrope_interleaved", True)),
        )

    @property
    def gdn_layers(self):
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "linear_attention")

    @property
    def attention_layers(self):
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "full_attention")

    @property
    def conv_dim(self):
        return 2 * self.linear_num_key_heads * self.linear_key_head_dim + self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def rotary_dim(self):
        return int(self.head_dim * self.partial_rotary_factor)
