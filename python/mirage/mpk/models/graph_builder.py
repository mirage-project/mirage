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
    local_num_q_heads: int
    local_num_kv_heads: int
    head_dim: int
    num_layers: int
    # kv cache
    k_cache: list[torch.Tensor]
    v_cache: list[torch.Tensor]
    # position embeddings (cos, sin)
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    # model weights
    state_dict: dict | None = None
    
    with_lm_head: bool = True
    
    def info_as_string(self):
        info = f"Hidden size: {self.hidden_size if self.hidden_size is not None else 'None'}\n"
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
        info += "-------------------------------------------\n"
        return info


class GraphBuilder(abc.ABC):
    def __init__(self, mpk, weights: Optional[Dict[str, Any]] = None):
        self.mpk = mpk
        self.weights = weights or {}

    @abc.abstractmethod
    def build_from_model(self, model_path: str | None = None):
        raise NotImplementedError("build_from_model is not implemented")
