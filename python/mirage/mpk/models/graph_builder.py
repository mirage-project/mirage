import abc
from typing import Optional, Dict, Any

from dataclasses import dataclass
import torch

@dataclass
class MirageModelConfig:
    # model architecture
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    # kv cache
    k_cache: list[torch.Tensor]
    v_cache: list[torch.Tensor]
    # position embeddings (sin, cos)
    position_embeddings: torch.Tensor
    # model weights
    state_dict: dict | None = None
    
    def info_as_string(self):
        info = f"Hidden size: {self.hidden_size}\n"
        info += f"Intermediate size: {self.intermediate_size}\n"
        info += f"Vocab size: {self.vocab_size}\n"
        info += f"Num q heads: {self.num_q_heads}\n"
        info += f"Num kv heads: {self.num_kv_heads}\n"
        info += f"Head dim: {self.head_dim}\n"
        info += f"Num layers: {self.num_layers}\n"
        info += f"K cache [0]: {self.k_cache[0].shape}\n"
        info += f"V cache [0]: {self.v_cache[0].shape}\n"
        info += f"Position embeddings: {self.position_embeddings.shape}\n"
        info += f"State dict len: {len(self.state_dict) if self.state_dict is not None else 0}\n"
        info += "-------------------------------------------\n"
        return info


class GraphBuilder(abc.ABC):
    def __init__(self, mpk, weights: Optional[Dict[str, Any]] = None):
        self.mpk = mpk
        self.weights = weights or {}

    @abc.abstractmethod
    def build_from_model(self, model_path: str | None = None):
        raise NotImplementedError("build_from_model is not implemented")
