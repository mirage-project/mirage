"""Qwen3.8-27B dense graph builder for MPK's offline runtime."""

import torch

from ..graph_builder import GraphBuilder
from ...kv_planner import KVSpec, plan_kv_groups
from ...model_registry import register_model_builder


def plan_qwen38_kv_cache(config):
    """A contiguous 1024-token page for each full-attention layer."""
    if (config.num_hidden_layers, config.hidden_size, config.num_key_value_heads,
            config.head_dim, len(config.attention_layers)) != (64, 5120, 4, 256, 16):
        raise ValueError("Expected Qwen3.8-27B dense attention geometry")
    per_entry_bytes = 2 * config.num_key_value_heads * config.head_dim * 2
    return plan_kv_groups([
        KVSpec("full_attention", per_entry_bytes=per_entry_bytes,
               layer_ids=config.attention_layers, preferred_block_size=1024)
    ], target_cc=90)


@register_model_builder("Qwen3.8-27B", "Qwen/Qwen3.8-27B")
class Qwen38Builder(GraphBuilder):
    def __init__(self, mpk, weights=None, *, kv_plan=None):
        super().__init__(mpk, weights)
        self.kv_plan = kv_plan

    def build_from_model(self, model_name=None, model_path=None, *, weight_policy="hf_bf16",
                         shard_roots=()):
        from .model import Qwen35Model
        from .weight_loader import Qwen35WeightLoader

        pk = self.mpk
        if model_path is None:
            raise ValueError("Supply the local BF16 checkpoint directory as model_path")
        loader = Qwen35WeightLoader(model_path, shard_roots=shard_roots)
        if pk.world_size != 1:
            raise ValueError("Qwen3.8-27B currently requires one GPU")
        if pk.test_mode:
            if (pk.max_num_batched_tokens != pk.max_num_batched_requests
                    or not 1 <= pk.max_num_batched_requests <= 32 or pk.max_seq_length != 1024):
                raise ValueError("The 27B probe requires a fixed batch of 1..32 and 1024 tokens")
            self.model = Qwen35Model(loader.config, self.weights or loader.load(weight_policy=weight_policy),
                                     pk=pk, batch_size=pk.max_num_batched_requests)
            return self.model
        batch = pk.max_num_batched_requests
        from .batch import validate_offline_batch_size
        validate_offline_batch_size(batch)
        if (pk.mode != "offline" or batch not in (1, 8) or pk.max_num_batched_tokens != batch
                or pk.max_tokens_per_request != 1
                or pk.max_num_pages != batch or pk.page_size != 1024
                or not 2 <= pk.max_seq_length <= 1024):
            raise ValueError("The offline 27B graph requires B1 or B8 and one 1024-token page per request")
        plan = self.kv_plan or plan_qwen38_kv_cache(loader.config)
        if (len(plan.groups) != 1 or plan.groups[0].block_size != 1024
                or plan.max_num_pages != pk.max_num_pages
                or len(pk.kv_groups) != 1 or pk.kv_groups[0].block_size != 1024):
            raise ValueError("Pass the same Qwen3.8 KV plan to PersistentKernel and Qwen38Builder")
        self.kv_pool, views = plan.allocate_pool({
            "full_attention": [("k", (4, 256), torch.bfloat16),
                               ("v", (4, 256), torch.bfloat16)]
        }, max_num_pages=pk.max_num_pages)
        cache_pairs = {}
        for layer in loader.config.attention_layers:
            group, slot = plan.layer_info(layer)
            plan.assert_in_pool(views[group]["k"][slot], f"layer {layer} K cache")
            plan.assert_in_pool(views[group]["v"][slot], f"layer {layer} V cache")
            cache_pairs[layer] = (views[group]["k"][slot, 0], views[group]["v"][slot, 0])
        self.model = Qwen35Model(loader.config, self.weights or loader.load(weight_policy=weight_policy),
                                 pk=pk, batch_size=batch, offline=True, kv_cache_pairs=cache_pairs)
        from .offline_contract import validate_offline_task_graph
        pk._task_graph_validator = lambda graph_json: validate_offline_task_graph(graph_json, batch)
        return self.model


# Builder discovery imports only files named builder.py.
from .moe_builder import Qwen36MoeBuilder  # noqa: E402,F401
