"""Grouped Qwen3.8/Qwen3.6 GDN subgraph for offline and fixed-step MPK."""

import torch

from .batch import BatchRows, linear_grid


class Qwen35GDNProbe:
    def __init__(self, config, weights, *, layer=0, pk=None, input_pair=None, name_prefix=None,
                 batch_size=1, position_pair=None, optimized_b8=False):
        c = config
        geometry = (c.hidden_size, c.linear_num_key_heads, c.linear_num_value_heads,
                    c.linear_key_head_dim, c.linear_value_head_dim,
                    c.linear_conv_kernel_dim, c.rms_norm_eps)
        if geometry not in ((5120, 16, 48, 128, 128, 4, 1e-6),
                            (2048, 16, 32, 128, 128, 4, 1e-6)):
            raise ValueError("Unsupported Qwen3.8/Qwen3.6 GDN dimensions")
        if layer not in c.gdn_layers:
            raise ValueError("Selected layer is not Gated DeltaNet")
        if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
            raise ValueError("The GDN probe requires an SM90 or SM100 GPU")
        self.config = c
        self.batch_size = batch_size
        hidden, kh, vh = c.hidden_size, c.linear_num_key_heads, c.linear_num_value_heads
        value_width, conv_width = vh * 128, c.conv_dim
        ab_width = ((2 * vh + 63) // 64) * 64 if batch_size == 1 else 2 * vh
        self.position = 0
        self.max_seq_length = 1024
        self._owns_kernel = pk is None
        if pk is None:
            from .model import create_fixed_batch_kernel
            pk = create_fixed_batch_kernel(batch_size)
        self.pk = pk
        name_prefix = name_prefix or "qwen35_"
        self.tensors = {}
        self.dt = {}
        self.batch = BatchRows(pk, batch_size, self.tensors, self.dt, name_prefix)

        def attach(name, tensor):
            tensor = tensor.detach().to(device="cuda").contiguous()
            self.tensors[name] = tensor
            self.dt[name] = pk.attach_input(tensor, name=name_prefix + name)
            return self.dt[name]

        def empty(name, shape, dtype=torch.bfloat16):
            return self.batch.empty(name, shape, dtype)

        prefix = f"layers.{layer}."
        def weight(name):
            return weights[prefix + name]

        if input_pair is None:
            empty("input", (1, hidden))
        else:
            self.tensors["input"], self.dt["input"] = input_pair
            self.batch.alias("input", (1, hidden))
        empty("normalized", (1, hidden))
        attach("input_norm_weight", weight("input_layernorm.weight").float())
        self._task("qwen35_norm", ["input", "input_norm_weight", "normalized"], [-1]*3, 1)

        for name, width in (("qkv", conv_width), ("z", value_width), ("ab", ab_width)):
            if name == "ab":
                # Keep batch rows contiguous when exposing [B, 2, heads].
                # SM100 supports a 32-column tile; the B=1 layout retains its
                # historical padding for the unchanged single-request path.
                w = torch.cat((weight("linear_attn.in_proj_a.weight"),
                               weight("linear_attn.in_proj_b.weight"),
                               weight("linear_attn.in_proj_a.weight").new_zeros(
                                   ab_width - 2 * vh, hidden)), dim=0)
            else:
                w = weight(f"linear_attn.in_proj_{name}.weight")
            if w.dtype != torch.bfloat16:
                raise ValueError("Projection weights must be BF16")
            attach(name + "_weight", w)
            empty(name, (1, width))
            if hidden == 5120 and name == "ab" and torch.cuda.get_device_capability() == (9, 0):
                self._task("qwen38_splitk", ["normalized", name+"_weight", name], [-1, 0, 1], width//64)
            else:
                pk.linear_layer(self.dt["normalized"], self.dt[name + "_weight"],
                                self.dt[name], linear_grid(width, batch_size,
                                                           32 if name == "ab" and batch_size > 1 else 64,
                                                           optimized_b8=optimized_b8),
                                (256, 1, 1))

        empty("conv", (1, conv_width))
        empty("conv_state", (conv_width, 3))
        attach("conv_weight", weight("linear_attn.conv1d.weight").reshape(conv_width, 4))
        self._task("qwen35_conv", ["qkv", "conv_weight", "conv_state", "conv"],
                   [1, 0, 0, 1], conv_width // 128)

        self.batch.view("conv_heads", "conv", [2 * kh + vh, 128])
        self.batch.view("z_heads", "z", [vh, 128])
        self.batch.view("ab_heads", "ab", [2, vh], narrow=(1, 0, 2 * vh))
        attach("parameters", torch.stack((weight("linear_attn.A_log").float(),
                                          weight("linear_attn.dt_bias").float())))
        nw = weight("linear_attn.norm.weight")
        if nw.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Gated norm weights must be BF16 or FP32")
        attach("gated_norm_weight", nw)
        empty("state", (vh, 128, 128), torch.float32)
        if position_pair is None:
            empty("position", (1,), torch.int64)
        else:
            self.tensors["position"], self.dt["position"] = position_pair
            self.batch.alias("position", (1,))
        empty("raw", (vh, 128))
        empty("gated", (vh, 128))
        names = ["conv_heads", "z_heads", "ab_heads", "parameters",
                 "gated_norm_weight", "state", "position"]
        partitions = [-1, 0, 1, 1, -1, 0, -1]
        attach("head_id", torch.arange(vh, dtype=torch.int64).reshape(vh, 1))
        names.append("head_id")
        partitions.append(0)
        self._task("qwen35_delta_grouped",
                   names + ["raw", "gated"], partitions + [0, 0], vh)
        attach("out_weight", weight("linear_attn.out_proj.weight"))
        empty("output", (1, hidden))
        pk.linear_layer(pk.view(self.dt["gated"], [batch_size, value_width]), self.dt["out_weight"],
                        self.dt["output"], linear_grid(hidden, batch_size, optimized_b8=optimized_b8), (256, 1, 1))

    def _task(self, name, names, partition_dims, tasks):
        self.batch.task(name, names, partition_dims, tasks)

    def compile(self, output_dir):
        if not self._owns_kernel:
            raise RuntimeError("Compile the owning full model, not an attached layer")
        self.pk.compile(output_dir=str(output_dir))

    @torch.inference_mode()
    def step(self, hidden):
        if not self._owns_kernel:
            raise RuntimeError("Step the owning full model, not an attached layer")
        if hidden.shape != (self.batch_size, self.config.hidden_size) or hidden.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 hidden states [batch_size, hidden_size]")
        if self.position >= self.max_seq_length:
            raise ValueError("The fixed batch has reached its 1024-token limit")
        self.tensors["input"].copy_(hidden)
        self.tensors["position"].fill_(self.position)
        # Offline test mode consumes its synthetic request in one launch.
        # Re-arm only scheduler metadata; keep the private GDN state/history
        # across tokens. The previous step has already synchronized all work.
        self.pk.init_request_func()
        self.pk()
        # State mutation belongs to one launch. Never overlap two steps/reset.
        torch.cuda.synchronize()
        self.position += 1
        return self.tensors["output"]

    @torch.inference_mode()
    def reset(self):
        torch.cuda.synchronize()
        self.tensors["conv_state"].zero_()
        self.tensors["state"].zero_()
        self.tensors["position"].zero_()
        self.position = 0

    def close(self):
        if self._owns_kernel:
            self.pk.finalize()
