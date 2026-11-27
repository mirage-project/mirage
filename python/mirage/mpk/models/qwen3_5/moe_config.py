"""Text-only BF16 configuration for Qwen3.6-35B-A3B.

The hybrid attention geometry is shared with Qwen3.8; the expert fields are
kept separate so a dense checkpoint cannot accidentally enter the MoE path.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from .config import Qwen35TextConfig


@dataclass(frozen=True)
class Qwen36MoeTextConfig(Qwen35TextConfig):
    num_experts: int
    num_experts_per_tok: int
    shared_expert_intermediate_size: int

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data):
        text = data.get("text_config", data)
        if data.get("model_type") != "qwen3_5_moe" or text.get("model_type") != "qwen3_5_moe_text":
            raise ValueError("Expected a Qwen3.6 MoE text configuration")
        fields = ("num_experts", "num_experts_per_tok", "moe_intermediate_size",
                  "shared_expert_intermediate_size")
        for name in fields:
            if type(text.get(name)) is not int or text[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if text["num_experts_per_tok"] > text["num_experts"]:
            raise ValueError("num_experts_per_tok exceeds num_experts")
        # Run the shared attention/GDN/RoPE validator on a dense-shaped copy.
        normalized = dict(text)
        normalized["model_type"] = "qwen3_5_text"
        normalized["intermediate_size"] = text["moe_intermediate_size"]
        normalized.pop("num_experts")
        normalized.pop("num_experts_per_tok")
        base = Qwen35TextConfig.from_dict({**data, "text_config": normalized})
        return cls(**vars(base), num_experts=text["num_experts"],
                   num_experts_per_tok=text["num_experts_per_tok"],
                   shared_expert_intermediate_size=text["shared_expert_intermediate_size"])
