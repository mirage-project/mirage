import abc
from typing import Optional, Dict, Any

from dataclasses import dataclass
import torch

@dataclass
class MirageModelConfig:
    # model architecture
    hidden_size: int = None
    intermediate_size: int = None
    vocab_size: int = None
    local_num_q_heads: int = None
    local_num_kv_heads: int = None
    head_dim: int = None
    num_layers: int = None
    # kv cache
    k_cache: list[torch.Tensor] = None
    v_cache: list[torch.Tensor] = None
    # position embeddings (cos, sin)
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None
    # model weights
    state_dict: dict | None = None
    
    with_lm_head: bool = True
    
    def __str__(self):
        return self.info_as_string()
    
    def info_as_string(self):
        info = "-------------------------------------------\n"
        info += "MirageModelConfig info:"
        info += f"Hidden size: {self.hidden_size if self.hidden_size is not None else 'None'}\n"
        info += f"Intermediate size: {self.intermediate_size if self.intermediate_size is not None else 'None'}\n"
        info += f"Vocab size: {self.vocab_size if self.vocab_size is not None else 'None'}\n"
        info += f"Num q heads: {self.local_num_q_heads if self.local_num_q_heads is not None else 'None'}\n"
        info += f"Num kv heads: {self.local_num_kv_heads if self.local_num_kv_heads is not None else 'None'}\n"
        info += f"Head dim: {self.head_dim if self.head_dim is not None else 'None'}\n"
        info += f"Num layers: {self.num_layers if self.num_layers is not None else 'None'}\n"
        info += f"K cache [0]: {self.k_cache[0].shape if self.k_cache[0] is not None else 'None'}\n"
        info += f"V cache [0]: {self.v_cache[0].shape if self.v_cache[0] is not None else 'None'}\n"
        info += f"Position embeddings cos: {self.position_embeddings[0].shape if self.position_embeddings[0] is not None else 'None'}\n"
        info += f"Position embeddings sin: {self.position_embeddings[1].shape if self.position_embeddings[1] is not None else 'None'}\n"
        info += f"State dict len: {len(self.state_dict) if self.state_dict is not None else 0}\n"
        info += f"With LM head: {self.with_lm_head}\n"
        info += "-------------------------------------------\n"
        return info


class GraphBuilder(abc.ABC):
    def __init__(self, mpk, weights: Optional[Dict[str, Any]] = None):
        self.mpk = mpk
        self.weights = weights or {}

    @abc.abstractmethod
    def build_from_model(self, model_path: str | None = None):
        raise NotImplementedError("build_from_model is not implemented")
