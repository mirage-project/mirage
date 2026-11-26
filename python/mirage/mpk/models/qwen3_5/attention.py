"""One-token Qwen3.8/Qwen3.6 attention subgraph for offline and fixed-step MPK."""
import torch

from .batch import BatchRows, linear_grid

from .attention_reference import text_rope


def validate_attention_implementation(value):
    if not isinstance(value, str) or value not in ("eager", "sdpa"):
        raise ValueError("attn_implementation must be 'eager' or 'sdpa'")


def validate_sdpa_runtime():
    # This geometry makes PyTorch's single-request split heuristic choose one
    # 64-key block per split up to capacity 1024. MIG profiles may differ.
    if (torch.cuda.get_device_capability() != (9, 0)
            or torch.cuda.get_device_properties(0).multi_processor_count < 64):
        raise ValueError("SDPA attention requires an SM90 GPU with at least 64 SMs")
    if not torch.backends.cuda.flash_sdp_enabled():
        raise ValueError("SDPA attention requires PyTorch's FlashAttention backend enabled")


class Qwen35AttentionProbe:
    def __init__(self, config, weights, *, layer=3, pk=None, input_pair=None, name_prefix=None,
                 attn_implementation="eager", batch_size=1, position_pair=None,
                 rope_pair=None, kv_cache_pair=None, optimized_b8=False):
        validate_attention_implementation(attn_implementation)
        if kv_cache_pair is not None and attn_implementation != "eager":
            raise ValueError("Planner-backed offline attention currently supports eager only")
        self.attn_implementation = attn_implementation
        c = config
        geometry = (c.hidden_size, c.num_attention_heads, c.num_key_value_heads,
                    c.head_dim, c.rotary_dim, c.rms_norm_eps)
        if geometry not in ((5120, 24, 4, 256, 64, 1e-6),
                            (2048, 16, 2, 256, 64, 1e-6)):
            raise ValueError("Unsupported Qwen3.8/Qwen3.6 full-attention dimensions")
        if layer not in c.attention_layers:
            raise ValueError("Selected layer is not full attention")
        if c.max_position_embeddings < 1024:
            raise ValueError("The probe requires a context capacity of at least 1024")
        if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
            raise ValueError("The attention probe requires an SM90 or SM100 GPU")
        if attn_implementation == "sdpa":
            validate_sdpa_runtime()
        self.config = c
        self.batch_size = batch_size
        hidden, qh, kvh = c.hidden_size, c.num_attention_heads, c.num_key_value_heads
        self.position = 0
        self.max_seq_length = 1024
        self._compiled = False
        self._owns_kernel = pk is None
        if pk is None:
            from .model import create_fixed_batch_kernel
            pk = create_fixed_batch_kernel(batch_size)
        self.pk = pk
        if attn_implementation == "sdpa":
            pk.use_fast_math = False
        name_prefix = name_prefix or "qwen35_attn_"
        self.tensors, self.dt = {}, {}
        self.batch = BatchRows(pk, batch_size, self.tensors, self.dt, name_prefix)

        def attach(name, tensor):
            tensor = tensor.detach().to(device="cuda").contiguous()
            self.tensors[name] = tensor
            self.dt[name] = pk.attach_input(tensor, name=name_prefix + name)
            return self.dt[name]

        def empty(name, shape, dtype=torch.bfloat16):
            return self.batch.empty(name, shape, dtype)

        prefix = f"layers.{layer}."
        if input_pair is None:
            empty("input", (1, hidden))
        else:
            self.tensors["input"], self.dt["input"] = input_pair
            self.batch.alias("input", (1, hidden))
        empty("normalized", (1, hidden))
        attach("input_norm_weight", weights[prefix + "input_layernorm.weight"].float())
        self._task("qwen35_norm", ["input", "input_norm_weight", "normalized"], [-1]*3, 1)
        for name, projection, width in (("qg", "q", 2 * qh * c.head_dim),
                                        ("k", "k", kvh * c.head_dim),
                                        ("v", "v", kvh * c.head_dim)):
            w = weights[prefix + f"self_attn.{projection}_proj.weight"]
            if w.dtype != torch.bfloat16 or w.shape != (width, hidden):
                raise ValueError(f"Invalid BF16 {projection} projection")
            attach(name + "_weight", w)
            empty(name, (1, width))
            if hidden == 5120 and name in ("k", "v") and torch.cuda.get_device_capability() == (9, 0):
                self._task("qwen38_splitk", ["normalized", name+"_weight", name], [-1, 0, 1], width//64)
            else:
                pk.linear_layer(self.dt["normalized"], self.dt[name + "_weight"], self.dt[name],
                                linear_grid(width, batch_size, optimized_b8=optimized_b8), (256, 1, 1))
        self.batch.view("qg_heads", "qg", [qh, 512])
        self.batch.view("k_heads", "k", [kvh, 256])
        self.batch.view("v_heads", "v", [kvh, 256])
        for name in ("q_norm", "k_norm"):
            attach(name, weights[prefix + f"self_attn.{name}.weight"].float())
        if rope_pair is None:
            cos, sin = text_rope(c, torch.arange(1024))
            self.rope_table = torch.stack((cos[0], sin[0]), dim=1).to("cuda")
            empty("rope", (2, 64))
        else:
            self.rope_table = None
            self.tensors["rope"], self.dt["rope"] = rope_pair
            self.batch.alias("rope", (2, 64))
        empty("query", (qh, 256))
        empty("key", (kvh, 256))
        empty("gate", (qh, 256))
        self._task("qwen35_qk_rope", ["qg_heads", "k_heads", "q_norm", "k_norm", "rope",
                                      "query", "key", "gate"], [-1]*8, 1)
        if position_pair is None:
            empty("position", (1,), torch.int64)
        else:
            self.tensors["position"], self.dt["position"] = position_pair
            self.batch.alias("position", (1,))
        for name, cache in zip(("key_cache", "value_cache"), kv_cache_pair or (None, None)):
            if cache is None:
                empty(name, (1024, kvh, 256))
            else:
                if cache.shape != (1024, kvh, 256) or cache.dtype != torch.bfloat16 or not cache.is_contiguous():
                    raise ValueError("Expected a contiguous BF16 planner cache page")
                self.tensors[name] = cache
                self.dt[name] = pk.attach_input(cache, name=name_prefix + name)
                # The planner gives a page-zero view into a shared page pool.
                # The offline task selects the request's live page from the
                # runtime page table, so grid.y must not offset this pointer.
        self._task("qwen38_kv_store_offline" if kv_cache_pair else "qwen35_kv_store",
                   ["key", "v_heads", "position", "key_cache", "value_cache"],
                   [-1]*5, 1)
        attach("head_id", torch.arange(qh, dtype=torch.int64).reshape(qh, 1))
        empty("gated", (qh, 256))
        scratch = "attention_scratch" if attn_implementation == "sdpa" else "attention_weights"
        empty(scratch, (qh, 4112) if attn_implementation == "sdpa" else (qh, 1024),
              torch.float32 if attn_implementation == "sdpa" else torch.bfloat16)
        if attn_implementation == "sdpa" and qh == 24:
            # With 24 query heads, HF may put multiple KV tiles in one split.
            # Reuse the configurable FlashAttention task for decode as well.
            from .mpk_prefill import flash_split_config
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            self.split_table = torch.tensor(
                [flash_split_config(1, n, sms, qh) for n in range(1, 1025)],
                dtype=torch.int64, device="cuda")
            empty("split_config", (2,), torch.int64)
            empty("attention_positions", (qh, 1), torch.int64)
            self._task("qwen35_prefill", ["query", "gate", "attention_positions", "key_cache", "value_cache",
                       "head_id", "split_config", "gated", scratch],
                       [0, 0, 0, -1, -1, 0, -1, 0, 0], qh,
                       params=[7, 1, hidden, c.intermediate_size, qh, kvh,
                               c.linear_num_key_heads, c.linear_num_value_heads])
        else:
            attention_task = "qwen35_sdpa_attention" if attn_implementation == "sdpa" else "qwen35_attention"
            if kv_cache_pair:
                attention_task = "qwen38_attention_offline"
            self._task(attention_task, ["query", "gate", "position", "key_cache", "value_cache",
                                           "head_id", "gated", scratch],
                       [0, 0, -1, -1, -1, 0, 0, 0], qh)
        attach("out_weight", weights[prefix + "self_attn.o_proj.weight"])
        empty("output", (1, hidden))
        pk.linear_layer(pk.view(self.dt["gated"], [batch_size, qh * 256]), self.dt["out_weight"],
                        self.dt["output"], linear_grid(hidden, batch_size, optimized_b8=optimized_b8), (256, 1, 1))

    def _task(self, name, names, partitions, tasks, params=None):
        self.batch.task(name, names, partitions, tasks, params)

    def compile(self, output_dir):
        if not self._owns_kernel:
            raise RuntimeError("Compile the owning full model, not an attached layer")
        self.pk.compile(output_dir=str(output_dir))
        self._compiled = True

    @torch.inference_mode()
    def step(self, hidden):
        if not self._owns_kernel:
            raise RuntimeError("Step the owning full model, not an attached layer")
        if not self._compiled:
            raise RuntimeError("Compile the attention probe before stepping")
        if hidden.shape != (self.batch_size, self.config.hidden_size) or hidden.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 hidden states [batch_size, hidden_size]")
        if self.position >= self.max_seq_length:
            raise ValueError("Sequence capacity exceeded; start a new request")
        self.tensors["input"].copy_(hidden)
        self.tensors["rope"].copy_(self.rope_table[self.position])
        self.tensors["position"].fill_(self.position)
        if "split_config" in self.tensors:
            self.tensors["split_config"].copy_(self.split_table[self.position])
            self.tensors["attention_positions"].fill_(self.position)
        self.pk.init_request_func()
        self.pk()
        torch.cuda.synchronize()
        self.position += 1
        return self.tensors["output"]

    @torch.inference_mode()
    def reset(self, key=None, value=None):
        """Start a request, optionally from externally validated KV prefixes."""
        if (key is None) != (value is None):
            raise ValueError("Both KV prefixes must be supplied")
        past = 0
        if key is not None:
            expected = (self.config.num_key_value_heads, 256)
            valid_shape = (key.ndim == 3 and key.shape[1:] == expected) if self.batch_size == 1 else (
                key.ndim == 4 and key.shape[0] == self.batch_size and key.shape[2:] == expected)
            past = (key.shape[0] if self.batch_size == 1 else key.shape[1]) if valid_shape else 0
            if (not valid_shape or value.shape != key.shape or past > 1024
                    or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16):
                raise ValueError("Expected BF16 KV prefixes [past <= 1024, configured KV heads, 256]")
            # Clone before clearing: callers may pass views of the old cache.
            key, value = key.to("cuda").clone(), value.to("cuda").clone()
        torch.cuda.synchronize()
        self.tensors["key_cache"].zero_()
        self.tensors["value_cache"].zero_()
        scratch = "attention_scratch" if self.attn_implementation == "sdpa" else "attention_weights"
        self.tensors[scratch].zero_()
        if past:
            for name, source in (("key_cache", key), ("value_cache", value)):
                target = self.tensors[name]
                (target[:past] if self.batch_size == 1 else target[:, :past]).copy_(source)
        self.position = past

    def close(self):
        if self._owns_kernel:
            self.pk.finalize()
