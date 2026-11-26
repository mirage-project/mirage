"""Qwen3.6-35B-A3B BF16 text graph on the shared hybrid decoder and MoE tasks."""

import torch

from ..graph_builder import GraphBuilder
from ...kv_planner import KVSpec, plan_kv_groups
from ...model_registry import register_model_builder


def plan_qwen36_kv_cache(config):
    if (config.num_hidden_layers, config.hidden_size, config.num_key_value_heads,
            config.head_dim, len(config.attention_layers), config.num_experts) != (
            40, 2048, 2, 256, 10, 256):
        raise ValueError("Expected Qwen3.6-35B-A3B text attention geometry")
    per_entry_bytes = 2 * config.num_key_value_heads * config.head_dim * 2
    return plan_kv_groups([
        KVSpec("full_attention", per_entry_bytes=per_entry_bytes,
               layer_ids=config.attention_layers, preferred_block_size=1024)
    ], target_cc=100)


@register_model_builder("Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B")
class Qwen36MoeBuilder(GraphBuilder):
    def __init__(self, mpk, weights=None, *, kv_plan=None):
        super().__init__(mpk, weights)
        self.kv_plan = kv_plan

    def build_from_model(self, model_name=None, model_path=None, *, weight_policy="hf_bf16",
                         shard_roots=()):
        from .model import Qwen36MoeModel
        from .weight_loader import Qwen36WeightLoader

        if model_path is None:
            raise ValueError("Supply the local BF16 checkpoint directory as model_path")
        loader = Qwen36WeightLoader(model_path, shard_roots=shard_roots)
        pk = self.mpk
        if pk.world_size != 1:
            raise ValueError("Qwen3.6 MoE currently requires one B200 GPU")
        batch = pk.max_num_batched_requests
        if pk.test_mode:
            if (pk.max_num_batched_tokens != batch or not 1 <= batch <= 32
                    or pk.max_seq_length != 1024):
                raise ValueError("The MoE probe requires a fixed batch and 1024-token capacity")
            self.model = Qwen36MoeModel(loader.config, self.weights or loader.load(weight_policy=weight_policy),
                                        pk=pk, batch_size=batch)
            return self.model
        from .batch import validate_offline_batch_size
        validate_offline_batch_size(batch)
        if (pk.mode != "offline" or batch not in (1, 8)
                or pk.max_num_batched_tokens != batch or pk.max_tokens_per_request != 1
                or pk.max_num_pages != batch or pk.page_size != 1024
                or not 2 <= pk.max_seq_length <= 1024):
            raise ValueError("The Qwen3.6 offline graph requires B1/B8 and one 1024-token page per request")
        plan = self.kv_plan or plan_qwen36_kv_cache(loader.config)
        if (len(plan.groups) != 1 or plan.groups[0].block_size != 1024
                or plan.max_num_pages != pk.max_num_pages
                or len(pk.kv_groups) != 1 or pk.kv_groups[0].block_size != 1024):
            raise ValueError("Pass the same Qwen3.6 KV plan to PersistentKernel and Qwen36MoeBuilder")
        self.kv_pool, views = plan.allocate_pool({
            "full_attention": [("k", (2, 256), torch.bfloat16),
                               ("v", (2, 256), torch.bfloat16)]
        }, max_num_pages=pk.max_num_pages)
        cache_pairs = {}
        for layer in loader.config.attention_layers:
            group, slot = plan.layer_info(layer)
            plan.assert_in_pool(views[group]["k"][slot], f"layer {layer} K cache")
            plan.assert_in_pool(views[group]["v"][slot], f"layer {layer} V cache")
            cache_pairs[layer] = (views[group]["k"][slot, 0], views[group]["v"][slot, 0])
        self.model = Qwen36MoeModel(loader.config, self.weights or loader.load(weight_policy=weight_policy),
                                    pk=pk, batch_size=batch, offline=True, kv_cache_pairs=cache_pairs)
        from .offline_contract import validate_offline_task_graph
        pk._task_graph_validator = lambda graph_json: validate_offline_task_graph(
            graph_json, batch, attention_layers=10, kv_heads=2, query_heads=16)
        return self.model
