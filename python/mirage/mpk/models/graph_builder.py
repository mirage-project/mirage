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

    A subclass reaches its KV caches through `self.mpk.kv_plan`. Whoever
    constructs the PersistentKernel already holds the plan -- its kv_groups
    and page tables come from it -- so the plan rides on the kernel rather
    than being passed to the builder separately.
    """

    def __init__(self, mpk, weights: Optional[Dict[str, Any]] = None):
        self.mpk = mpk
        self.weights = weights or {}
        plan = getattr(mpk, "kv_plan", None)
        if plan is not None:
            assert len(plan.groups) == len(mpk.kv_groups), (
                f"mpk.kv_plan has {len(plan.groups)} KV group(s) but mpk was "
                f"built with {len(mpk.kv_groups)} -- pass "
                f"kv_groups=plan.group_specs() and this same plan together")

    @staticmethod
    def kv_streams(config, world_size: int = 1):
        """This model's KV streams. EVERY builder must override this.

        Called BEFORE the PersistentKernel exists, because the plan supplies
        its kv_groups and the page-table meta tensors. Caches are then reached
        through `self.mpk.kv_plan.attach(self.mpk, layer)`.
        """
        raise NotImplementedError(
            "this builder does not declare its KV streams; override "
            "kv_streams()")

    @staticmethod
    def load_config(model_name: str, model_path: str | None = None):
        """The config to hand kv_streams(). AutoConfig by default; override
        it for an architecture transformers does not know."""
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_path or model_name)

    @abc.abstractmethod
    def build_from_model(self, model_path: str | None = None):
        raise NotImplementedError("build_from_model is not implemented")
