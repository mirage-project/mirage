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
    # position embeddings (cos, sin)
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None
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
        info += f"Position embeddings cos: {self.position_embeddings[0].shape if self.position_embeddings[0] is not None else 'None'}\n"
        info += f"Position embeddings sin: {self.position_embeddings[1].shape if self.position_embeddings[1] is not None else 'None'}\n"
        info += f"State dict len: {len(self.state_dict) if self.state_dict is not None else 0}\n"
        info += "-------------------------------------------\n"
        return info


class GraphBuilder(abc.ABC):
    """Base for the per-model task-graph builders.

    A subclass reaches its KV caches through `self.mpk.kv_plan` -- one way in,
    not two. Whoever constructs the PersistentKernel necessarily already has
    the plan (its kv_groups and page tables come from it), so it rides on the
    kernel; a second constructor argument would only be another way to say the
    same thing, and a way for the two to disagree.
    """

    def __init__(self, mpk, weights: Optional[Dict[str, Any]] = None):
        self.mpk = mpk
        self.weights = weights or {}

    @staticmethod
    def kv_streams(config, page_size: int, world_size: int = 1):
        """This model's KV streams. EVERY builder must override this.

        Called BEFORE the PersistentKernel exists, because the plan supplies
        its kv_groups and the page-table meta tensors. Caches are then reached
        through `self.mpk.kv_plan.attach(self.mpk, layer)`.

        A list of KVStream is the normal answer; a stream whose kernel reads
        the cache flat says `paged=False`. `[]` is reserved for a model with
        no KV cache at all, and is refused until one exists. Inheriting this
        base is an error: it used to mean "fall back to KV 1.0", silently and
        indistinguishably from "no KV".
        """
        raise NotImplementedError(
            "this builder does not declare its KV streams; override "
            "kv_streams()")

    @staticmethod
    def load_config(model_name: str, model_path: str | None = None):
        """The config to hand kv_streams(). AutoConfig by default.

        A hook because AutoConfig does not know every architecture MPK
        supports -- Inkling is not in transformers' mapping, and its builder
        already reads config.json directly. Without this, declaring KV streams
        would silently require every registered model to be AutoConfig-
        loadable, which is a requirement about transformers, not about MPK.
        """
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_path or model_name)

    @abc.abstractmethod
    def build_from_model(self, model_path: str | None = None):
        raise NotImplementedError("build_from_model is not implemented")
