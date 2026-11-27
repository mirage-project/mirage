"""Compose one decoder layer, standalone or attached to a shared full-model graph.

Keep BF16 rounding at both residual boundaries explicit. State belongs to a synchronized
fixed batch; this does not implement the continuous-batching scheduler.
"""
import os

import torch

from .attention import Qwen35AttentionProbe, validate_attention_implementation
from .batch import linear_grid
from .gdn import Qwen35GDNProbe


class Qwen35DecoderProbe:
    def __init__(self, config, weights, *, layer=0, pk=None, input_pair=None, name_prefix=None,
                 attn_implementation="eager", batch_size=1, position_pair=None,
                 rope_pair=None, kv_cache_pair=None, optimized_b8=False):
        validate_attention_implementation(attn_implementation)
        moe = hasattr(config, "num_experts")
        if (config.hidden_size, config.intermediate_size) != (
                (2048, 512) if moe else (5120, 17408)):
            raise ValueError("Unsupported Qwen3.8/Qwen3.6 decoder dimensions")
        if moe and (torch.cuda.get_device_capability() != (10, 0) or
                    config.num_experts != 256 or config.num_experts_per_tok != 8):
            raise ValueError("Qwen3.6 MoE currently requires SM100 and 256 top-8 experts")
        if not 0 <= layer < config.num_hidden_layers:
            raise ValueError("Invalid decoder layer")
        cls = Qwen35GDNProbe if layer in config.gdn_layers else Qwen35AttentionProbe
        options = {} if layer in config.gdn_layers else {
            "attn_implementation": attn_implementation, "rope_pair": rope_pair,
            "kv_cache_pair": kv_cache_pair}
        options["optimized_b8"] = optimized_b8
        self.mixer = mixer = cls(config, weights, layer=layer, pk=pk,
                                 input_pair=input_pair, name_prefix=name_prefix, batch_size=batch_size,
                                 position_pair=position_pair, **options)
        self.pk = pk = mixer.pk
        self.tensors, self.dt = mixer.tensors, mixer.dt
        self.batch_size = batch_size
        self.max_seq_length = mixer.max_seq_length
        self._compiled = False

        def attach(name, tensor):
            tensor = tensor.detach().to(device="cuda").contiguous()
            self.tensors[name] = tensor
            self.dt[name] = pk.attach_input(tensor, name=(name_prefix or "qwen35_decoder_") + name)
            return self.dt[name]

        def empty(name, width):
            return mixer.batch.empty(name, (1, width))

        prefix = f"layers.{layer}."
        hidden, intermediate = config.hidden_size, config.intermediate_size
        silu_rows = os.environ.get(f"MPK_QWEN38_B{batch_size}_SILU_ROWS", "1" if optimized_b8 else "0") == "1"
        add_rows = os.environ.get(f"MPK_QWEN38_B{batch_size}_ADD_ROWS", "1" if optimized_b8 else "0") == "1"
        if (silu_rows or add_rows) and batch_size not in (8, 16):
            raise ValueError("Row-partition experiments require batch_size=8 or 16")
        add_grid = (batch_size if add_rows else 1, 1, 1)
        empty("residual", hidden)
        # Separate add tasks preserve the HF BF16 projection-before-add order.
        pk.elementwise_add_layer(self.dt["input"], self.dt["output"], self.dt["residual"],
                                 add_grid, (256, 1, 1))
        attach("post_norm_weight", weights[prefix + "post_attention_layernorm.weight"].float())
        empty("post_normalized", hidden)
        mixer._task("qwen35_norm", ["residual", "post_norm_weight", "post_normalized"], [-1]*3, 1)
        if moe:
            self._build_moe(config, weights, prefix, attach, empty, optimized_b8)
            return
        gate, up = (weights[prefix + f"mlp.{name}_proj.weight"] for name in ("gate", "up"))
        if any(w.shape != (intermediate, hidden) or w.dtype != torch.bfloat16 for w in (gate, up)):
            raise ValueError("Invalid BF16 MLP gate/up weights")
        attach("gate_up_weight", torch.cat((gate, up), dim=0))
        empty("gate_up", 2 * intermediate)
        pk.linear_layer(self.dt["post_normalized"], self.dt["gate_up_weight"], self.dt["gate_up"],
                        linear_grid(2 * intermediate, batch_size, optimized_b8=optimized_b8), (256, 1, 1))
        empty("activated", intermediate)
        # Split by request row so each task still sees its complete gate/up pair.
        pk.silu_mul_layer(self.dt["gate_up"], self.dt["activated"],
                          (1, batch_size if silu_rows else 1, 1), (256, 1, 1))
        down = weights[prefix + "mlp.down_proj.weight"]
        if down.shape != (hidden, intermediate) or down.dtype != torch.bfloat16:
            raise ValueError("Invalid BF16 MLP down weight")
        attach("down_weight", down)
        empty("mlp_output", hidden)
        pk.linear_layer(self.dt["activated"], self.dt["down_weight"], self.dt["mlp_output"],
                        linear_grid(hidden, batch_size, optimized_b8=optimized_b8), (256, 1, 1))
        empty("decoder_output", hidden)
        pk.elementwise_add_layer(self.dt["residual"], self.dt["mlp_output"], self.dt["decoder_output"],
                                 add_grid, (256, 1, 1))

    def _build_moe(self, config, weights, prefix, attach, empty, optimized_b8):
        """HF top-8 routed experts plus separately gated shared expert."""
        pk = self.pk
        batch = self.batch_size
        optimize_shared = optimized_b8
        hidden, experts, topk = config.hidden_size, config.num_experts, config.num_experts_per_tok
        width, shared_width = config.intermediate_size, config.shared_expert_intermediate_size

        def checked(name, shape):
            tensor = weights[prefix + "mlp." + name]
            if tensor.shape != shape or tensor.dtype != torch.bfloat16:
                raise ValueError(f"Invalid BF16 MoE tensor {prefix}mlp.{name}")
            return tensor

        attach("router_weight", checked("gate.weight", (experts, hidden)))
        empty("router_logits", experts)
        pk.linear_layer(self.dt["post_normalized"], self.dt["router_weight"],
                        self.dt["router_logits"],
                        linear_grid(experts, batch, optimized_b8=optimize_shared), (256, 1, 1))
        attach("router_scores", torch.zeros((batch, topk), device="cuda", dtype=torch.float32))
        attach("routing_indices", torch.zeros((experts, batch), device="cuda", dtype=torch.int32))
        attach("active_experts", torch.zeros((experts + 1,), device="cuda", dtype=torch.int32))
        pk.moe_topk_softmax_routing_layer(
            self.dt["router_logits"],
            (self.dt["router_scores"], self.dt["routing_indices"], self.dt["active_experts"]),
            (1, 1, 1), (256, 1, 1), round_weights_to_bf16=True)
        attach("experts_gate_up", checked("experts.gate_up_proj", (experts, 2 * width, hidden)))
        attach("experts_down", checked("experts.down_proj", (experts, hidden, width)))
        attach("expert_mid", torch.zeros((batch, topk, 2 * width), device="cuda", dtype=torch.bfloat16))
        attach("expert_activated", torch.zeros((batch, topk, width), device="cuda", dtype=torch.bfloat16))
        attach("expert_out", torch.zeros((batch, topk, hidden), device="cuda", dtype=torch.bfloat16))
        pk.moe_w13_linear_layer(self.dt["post_normalized"], self.dt["experts_gate_up"],
                                self.dt["routing_indices"], self.dt["active_experts"],
                                self.dt["expert_mid"], (10, 2 * width // 128, 1), (256, 1, 1))
        pk.moe_silu_mul_layer(self.dt["expert_mid"], self.dt["expert_activated"],
                              (batch, topk, 1), (256, 1, 1))
        pk.moe_w2_linear_layer(self.dt["expert_activated"], self.dt["experts_down"],
                               self.dt["routing_indices"], self.dt["active_experts"],
                               self.dt["expert_out"], (8, hidden // 128, 1), (256, 1, 1))

        gate = checked("shared_expert.gate_proj.weight", (shared_width, hidden))
        up = checked("shared_expert.up_proj.weight", (shared_width, hidden))
        attach("shared_gate_up_weight", torch.cat((gate, up), dim=0))
        empty("shared_gate_up", 2 * shared_width)
        pk.linear_layer(self.dt["post_normalized"], self.dt["shared_gate_up_weight"],
                        self.dt["shared_gate_up"],
                        linear_grid(2 * shared_width, batch, optimized_b8=optimize_shared), (256, 1, 1))
        empty("shared_activated", shared_width)
        pk.silu_mul_layer(self.dt["shared_gate_up"], self.dt["shared_activated"],
                          (1, batch, 1), (256, 1, 1))
        attach("shared_down_weight", checked("shared_expert.down_proj.weight", (hidden, shared_width)))
        empty("shared_output", hidden)
        pk.linear_layer(self.dt["shared_activated"], self.dt["shared_down_weight"],
                        self.dt["shared_output"],
                        linear_grid(hidden, batch, optimized_b8=optimize_shared), (256, 1, 1))
        gate_weight = checked("shared_expert_gate.weight", (1, hidden))
        attach("shared_gate_weight", torch.cat((gate_weight, gate_weight.new_zeros((63, hidden))), dim=0))
        empty("shared_gate", 64)
        pk.linear_layer(self.dt["post_normalized"], self.dt["shared_gate_weight"],
                        self.dt["shared_gate"], (1, 1, 1), (256, 1, 1))
        empty("shared_gated", hidden)
        self.mixer._task("qwen36_shared_gate",
                         ["shared_gate", "shared_output", "shared_gated"], [-1] * 3, 1)
        empty("mlp_output", hidden)
        pk.qwen36_moe_sum_layer(self.dt["expert_out"], self.dt["router_scores"],
                                self.dt["routing_indices"], self.dt["shared_gated"],
                                self.dt["mlp_output"])
        empty("decoder_output", hidden)
        pk.elementwise_add_layer(self.dt["residual"], self.dt["mlp_output"],
                                 self.dt["decoder_output"], (batch, 1, 1), (256, 1, 1))

    @property
    def position(self):
        return self.mixer.position

    def compile(self, output_dir):
        self.mixer.compile(output_dir)
        self._compiled = True

    def step(self, hidden):
        if not self._compiled:
            raise RuntimeError("Compile the decoder probe before stepping")
        self.mixer.step(hidden)
        return self.tensors["decoder_output"]

    def reset(self, **prefix):
        self.mixer.reset(**prefix)

    def close(self):
        self.mixer.close()
