"""Strict CPU-first loaders for BF16 Qwen3.8 dense and Qwen3.6 MoE checkpoints.

Retains the checkpoint's tensor layout. By default all text parameters use the
native HF BF16 loading precision; ``checkpoint`` preserves their stored dtypes.
Kernel-specific packing and RMSNorm's (1 + weight) belong to the graph.
"""

from contextlib import ExitStack
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .config import Qwen35TextConfig

TEXT_PREFIX = "model.language_model."
WEIGHT_POLICIES = ("hf_bf16", "checkpoint")


def expected_text_shapes(config):
    c = config
    shapes = {"embed_tokens.weight": (c.vocab_size, c.hidden_size), "norm.weight": (c.hidden_size,)}
    for i, kind in enumerate(c.layer_types):
        block = {
            "input_layernorm.weight": (c.hidden_size,),
            "post_attention_layernorm.weight": (c.hidden_size,),
        }
        if hasattr(c, "num_experts"):
            block.update({
                "mlp.gate.weight": (c.num_experts, c.hidden_size),
                "mlp.experts.gate_up_proj": (c.num_experts, 2 * c.intermediate_size, c.hidden_size),
                "mlp.experts.down_proj": (c.num_experts, c.hidden_size, c.intermediate_size),
                "mlp.shared_expert.gate_proj.weight": (c.shared_expert_intermediate_size, c.hidden_size),
                "mlp.shared_expert.up_proj.weight": (c.shared_expert_intermediate_size, c.hidden_size),
                "mlp.shared_expert.down_proj.weight": (c.hidden_size, c.shared_expert_intermediate_size),
                "mlp.shared_expert_gate.weight": (1, c.hidden_size),
            })
        else:
            block.update({
                "mlp.gate_proj.weight": (c.intermediate_size, c.hidden_size),
                "mlp.up_proj.weight": (c.intermediate_size, c.hidden_size),
                "mlp.down_proj.weight": (c.hidden_size, c.intermediate_size),
            })
        if kind == "linear_attention":
            value_dim = c.linear_num_value_heads * c.linear_value_head_dim
            block.update({
                "linear_attn.in_proj_qkv.weight": (c.conv_dim, c.hidden_size),
                "linear_attn.in_proj_z.weight": (value_dim, c.hidden_size),
                "linear_attn.in_proj_a.weight": (c.linear_num_value_heads, c.hidden_size),
                "linear_attn.in_proj_b.weight": (c.linear_num_value_heads, c.hidden_size),
                "linear_attn.conv1d.weight": (c.conv_dim, 1, c.linear_conv_kernel_dim),
                "linear_attn.A_log": (c.linear_num_value_heads,),
                "linear_attn.dt_bias": (c.linear_num_value_heads,),
                "linear_attn.norm.weight": (c.linear_value_head_dim,),
                "linear_attn.out_proj.weight": (c.hidden_size, value_dim),
            })
        else:
            block.update({
                # Each query head contains both Q and its output-gate vector.
                "self_attn.q_proj.weight": (2 * c.num_attention_heads * c.head_dim, c.hidden_size),
                "self_attn.k_proj.weight": (c.num_key_value_heads * c.head_dim, c.hidden_size),
                "self_attn.v_proj.weight": (c.num_key_value_heads * c.head_dim, c.hidden_size),
                "self_attn.o_proj.weight": (c.hidden_size, c.num_attention_heads * c.head_dim),
                "self_attn.q_norm.weight": (c.head_dim,),
                "self_attn.k_norm.weight": (c.head_dim,),
            })
        shapes.update({f"layers.{i}.{name}": shape for name, shape in block.items()})
    if not c.tie_word_embeddings:
        shapes["lm_head.weight"] = (c.vocab_size, c.hidden_size)
    return shapes


def canonical_name(name):
    if name.startswith(TEXT_PREFIX):
        return name[len(TEXT_PREFIX):]
    if name == "lm_head.weight":
        return name
    # Only these explicit auxiliary namespaces may be ignored.
    if name.startswith("model.visual.") or name.startswith("mtp."):
        return None
    raise ValueError(f"Unrecognized checkpoint tensor: {name}")


def allowed_dtypes(name):
    # Sensitive scalars and normalization weights may be stored in FP32.
    if name.endswith(("norm.weight", "A_log", "dt_bias")):
        return {"BF16", "F32"}
    return {"BF16"}


class Qwen35WeightLoader:
    def __init__(self, model_path, *, shard_roots=()):
        # Additional trusted roots are explicit, e.g. a caller-owned RAM disk.
        self.shard_roots = tuple(Path(p).resolve() for p in shard_roots)
        self.path = Path(model_path).resolve()
        self.config = Qwen35TextConfig.from_json(self.path / "config.json")
        c = self.config
        if (c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.vocab_size,
                c.num_attention_heads, c.num_key_value_heads,
                c.linear_num_key_heads, c.linear_num_value_heads,
                len(c.gdn_layers), len(c.attention_layers)) != (
                64, 5120, 17408, 248320, 24, 4, 16, 48, 48, 16):
            raise ValueError("The BF16 checkpoint loader supports Qwen3.8-27B dense only")

    def _files(self):
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text())["weight_map"]
            if not mapping:
                raise ValueError("Empty checkpoint index")
            names = sorted(set(mapping.values()))
        else:
            mapping = None
            names = ["model.safetensors"]
        files = {}
        for name in names:
            file = (self.path / name).resolve()
            if (Path(name).is_absolute() or ".." in Path(name).parts
                    or not any(file.is_relative_to(root) for root in (self.path, *self.shard_roots))
                    or not name.endswith(".safetensors")):
                raise ValueError(f"Invalid shard path: {name}")
            if not file.is_file():
                raise FileNotFoundError(file)
            files[name] = file
        return mapping, files

    def _inspect(self, stack):
        mapping, files = self._files()
        readers = {name: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                   for name, path in files.items()}
        actual = {}
        for shard, reader in readers.items():
            for name in reader.keys():
                if name in actual:
                    raise ValueError(f"Tensor appears in multiple shards: {name}")
                actual[name] = shard
        if mapping is not None and actual != mapping:
            raise ValueError("Checkpoint index does not exactly match shard contents")
        expected = expected_text_shapes(self.config)
        text = {}
        skipped = {"vision": [], "mtp": []}
        specs = {}
        for original, shard in actual.items():
            name = canonical_name(original)
            if name is None:
                skipped["vision" if original.startswith("model.visual.") else "mtp"].append(original)
                continue
            if name in text:
                raise ValueError(f"Duplicate canonical tensor: {name}")
            shape = expected.get(name)
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                shape = (self.config.vocab_size, self.config.hidden_size)
            if shape is None:
                raise ValueError(f"Unexpected text tensor for configured layer types: {original}")
            tensor_slice = readers[shard].get_slice(original)
            actual_shape = tuple(tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
            if actual_shape != shape:
                raise ValueError(f"{original}: expected shape {shape}, got {actual_shape}")
            if dtype not in allowed_dtypes(name):
                raise ValueError(f"{original}: unsupported dtype {dtype}")
            text[name] = (shard, original)
            specs[name] = {"shape": list(shape), "dtype": dtype}
        missing = sorted(set(expected) - text.keys())
        if missing:
            raise ValueError(f"Missing text tensors: {missing}")
        if self.config.tie_word_embeddings and "lm_head.weight" in text:
            def read(name):
                shard, original = text[name]
                return readers[shard].get_tensor(original)
            if not torch.equal(read("lm_head.weight"), read("embed_tokens.weight")):
                raise ValueError("Tied LM head differs from embedding weights")
        report = {
            "checkpoint_tensors": len(actual), "text_tensors": len(text),
            "skipped": {kind: sorted(names) for kind, names in skipped.items()},
            "tied_lm_head": self.config.tie_word_embeddings,
            "gdn_layers": list(self.config.gdn_layers),
            "attention_layers": list(self.config.attention_layers),
            "tensors": specs,
        }
        return readers, text, report

    def inspect(self):
        """Validate metadata without materializing the model on CPU or GPU."""
        with ExitStack() as stack:
            return self._inspect(stack)[2]

    def load(self, device="cpu", *, weight_policy="hf_bf16"):
        """Return text weights with shared storage for a tied LM head.

        ``hf_bf16`` matches the pinned HF ``from_pretrained(dtype=bfloat16)``
        parameter precision, including FP32 tensors stored in the checkpoint.
        ``checkpoint`` retains the original dtypes for legacy frozen oracles.
        This controls parameter loading only, not FP32 recurrence/accumulation.
        """
        if weight_policy not in WEIGHT_POLICIES:
            raise ValueError(f"Unknown weight policy: {weight_policy!r}; expected {WEIGHT_POLICIES}")
        dtype = torch.bfloat16 if weight_policy == "hf_bf16" else None
        with ExitStack() as stack:
            readers, text, _ = self._inspect(stack)
            weights = {
                name: readers[shard].get_tensor(original).to(device=device, dtype=dtype)
                for name, (shard, original) in text.items()
                if not (name == "lm_head.weight" and self.config.tie_word_embeddings)
            }
        if self.config.tie_word_embeddings:
            weights["lm_head.weight"] = weights["embed_tokens.weight"]
        return weights


class Qwen36WeightLoader(Qwen35WeightLoader):
    """Reuse strict shard, dtype, and namespace checks for the MoE checkpoint."""

    def __init__(self, model_path, *, shard_roots=()):
        from .moe_config import Qwen36MoeTextConfig

        self.shard_roots = tuple(Path(p).resolve() for p in shard_roots)
        self.path = Path(model_path).resolve()
        self.config = Qwen36MoeTextConfig.from_json(self.path / "config.json")
        c = self.config
        if (c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.vocab_size,
                c.num_attention_heads, c.num_key_value_heads, c.linear_num_key_heads,
                c.linear_num_value_heads, c.num_experts, c.num_experts_per_tok,
                c.shared_expert_intermediate_size, len(c.gdn_layers), len(c.attention_layers)) != (
                40, 2048, 512, 248320, 16, 2, 16, 32, 256, 8, 512, 30, 10):
            raise ValueError("The BF16 MoE loader supports Qwen3.6-35B-A3B only")
