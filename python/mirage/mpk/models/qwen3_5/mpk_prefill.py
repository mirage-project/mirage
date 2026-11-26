"""Single-request 64-token MPK prefill graph with transactional cache commit.

All transformer arithmetic runs as tasks in the persistent kernel. Python
only fills token/position metadata, copies state, launches tiles, and commits
the final outputs. No HF forward or PyTorch attention/GDN/linear is called.
The separate decode graph and the PyTorch reference prefill remain available.
"""
from pathlib import Path

import torch


def flash_split_config(query_length, key_length, num_sms, num_heads=8):
    """Pinned FlashAttention occupancy heuristic (batch=1, D=256)."""
    import math
    m, n = (query_length+63)//64, (key_length+63)//64
    blocks, capacity = num_heads*m, 2*num_sms
    if blocks >= .8*capacity:
        return 1, n
    candidates = [(s, (blocks*s/capacity)/math.ceil(blocks*s/capacity))
                  for s in range(1, min(128, capacity, n)+1)
                  if s == 1 or math.ceil(n/s) != math.ceil(n/(s-1))]
    best = max(efficiency for _, efficiency in candidates)
    splits = next(s for s, efficiency in candidates if efficiency >= .85*best)
    return splits, (n+splits-1)//splits


def eager_mma_config(query_length, key_length):
    """Measured pinned cuBLAS alignment schedule for batched BF16 attention.

    Short small-N kernels are represented by the 16-wide path; unaligned
    larger matrices use the native CUTLASS m16n8k8 accumulation order.
    This is a numerical schedule, not a portable cuBLAS dispatch guarantee.
    """
    return (8 if query_length > 17 and key_length % 2 else 16), key_length


class Qwen35ChunkPrefill:
    chunk_size = 64

    def __init__(self, model):
        from mirage import get_configurations_from_gpu
        from mirage.mpk.persistent_kernel import PersistentKernel

        if torch.cuda.get_device_capability() != (9, 0):
            raise ValueError("MPK chunk prefill currently requires an SM90 GPU")
        workers, schedulers = get_configurations_from_gpu(0)
        options = PersistentKernel.get_default_init_parameters()
        options.update(test_mode=True, num_workers=workers, num_local_schedulers=schedulers,
                       mpi_rank=0, world_size=1, max_num_batched_tokens=64,
                       max_num_batched_requests=1, max_seq_length=1024,
                       page_size=1024, max_num_pages=1, use_fast_math=False)
        self.model, self.pk = model, PersistentKernel(**options)
        c = model.config
        self.hidden, self.intermediate = c.hidden_size, c.intermediate_size
        self.qh, self.kvh = c.num_attention_heads, c.num_key_value_heads
        self.kh, self.vh = c.linear_num_key_heads, c.linear_num_value_heads
        if (self.hidden, self.intermediate, self.qh, self.kvh, self.kh, self.vh) != (
                5120, 17408, 24, 4, 16, 48):
            raise ValueError("MPK chunk prefill supports Qwen3.8-27B dense only")
        self.conv_width = c.conv_dim
        self.ab_width = ((2 * self.vh + 63) // 64) * 64
        self.geometry = [self.hidden, self.intermediate, self.qh, self.kvh, self.kh, self.vh]
        self.tensors, self.dt, self.layers = {}, {}, []
        self.compiled, self.closed = False, False
        self.empty("tokens", (64,), torch.int64)
        self.empty("metadata", (3,), torch.int64)
        self.empty("split_config", (2,), torch.int64)
        self.dt["length"] = self.pk.narrow(self.dt["metadata"], 0, 1, 1)
        self.dt["chunk_lengths"] = self.pk.narrow(self.dt["metadata"], 0, 1, 2)
        self.dt["query_length"] = self.pk.narrow(self.dt["metadata"], 0, 2, 1)
        self.empty("positions", (64*self.qh, 1), torch.int64)
        self.attach("head_ids", torch.arange(self.qh, device="cuda").repeat(64).reshape(64*self.qh, 1))
        self.empty("rope", (64, 2, 64))
        self.attach("embedding_weight", model.tensors["embedding"])
        self.empty("embedding", (64, self.hidden))
        self.task(0, ["tokens", "embedding_weight", "embedding"], [0, -1, 0], 64)
        previous = "embedding"
        for index, layer in enumerate(model.layers):
            prefix = f"layer{index}."
            info = dict(prefix=prefix, gdn=index in model.config.gdn_layers)
            self.layers.append(info)
            for name in ("input_norm_weight", "post_norm_weight", "gate_up_weight", "down_weight", "out_weight"):
                self.attach(prefix + name, layer.tensors[name])
            norm = self.empty(prefix + "normalized", (64, self.hidden))
            self.task(1, [previous, prefix+"input_norm_weight", "query_length", norm], [0, -1, -1, 0], 64)
            if info["gdn"]:
                mixed = self.gdn(prefix, layer.tensors, norm)
            else:
                mixed = self.attention(prefix, layer.tensors, norm)
            residual = self.empty(prefix + "residual", (64, self.hidden))
            self.task(9, [previous, mixed, residual], [0, 0, 0], 64)
            normalized = self.empty(prefix + "post_normalized", (64, self.hidden))
            self.task(1, [residual, prefix+"post_norm_weight", "query_length", normalized], [0, -1, -1, 0], 64)
            gate_up = self.linear(prefix+"gate_up", normalized, prefix+"gate_up_weight", 2*self.intermediate)
            activated = self.empty(prefix+"activated", (64, self.intermediate))
            self.task(8, [gate_up, activated], [0, 0], 64)
            mlp = self.linear(prefix+"mlp", activated, prefix+"down_weight", self.hidden)
            previous = self.empty(prefix+"output", (64, self.hidden))
            self.task(9, [residual, mlp, previous], [0, 0, 0], 64)
        self.attach("norm_weight", model.tensors["norm_weight"])
        self.empty("normalized", (64, self.hidden))
        self.task(1, [previous, "norm_weight", "query_length", "normalized"], [0, -1, -1, 0], 64)
        self.empty("last_normalized", (1, self.hidden))
        self.task(10, ["normalized", "length", "last_normalized"], [-1, -1, -1], 1)
        self.attach("lm_head", model.tensors["lm_head"])
        self.empty("logits", (1, model.config.vocab_size))
        self.pk.linear_layer(self.dt["last_normalized"], self.dt["lm_head"], self.dt["logits"],
                             (model.config.vocab_size//64, 1, 1), (256, 1, 1))
        self.empty("next_token", (1,), torch.int64)
        self.task_original("qwen35_argmax", ["logits", "next_token"])

    def attach(self, name, tensor):
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("MPK prefill tensors must be contiguous CUDA storage")
        self.tensors[name] = tensor
        self.dt[name] = self.pk.attach_input(tensor, name="qwen35_prefill_" + name.replace(".", "_"))
        return name

    def empty(self, name, shape, dtype=torch.bfloat16):
        return self.attach(name, torch.zeros(shape, dtype=dtype, device="cuda"))

    def task(self, kind, names, partitions, tasks, backend=0):
        from mirage.core import CyTBGraph
        from mirage.kernel import TBGraph
        graph = TBGraph(CyTBGraph((tasks, 1, 1), (256, 1, 1), 1, 64))
        tensors = [self.dt[name] for name in names]
        for tensor, dim in zip(tensors, partitions):
            graph.new_input(tensor, (dim, -1, -1), -1, True)
        self.pk.kn_graph.customized(tensors, graph)
        self.pk.kn_graph.register_task(graph, "qwen35_prefill", [kind, backend] + self.geometry)

    def task_original(self, kind, names):
        from mirage.core import CyTBGraph
        from mirage.kernel import TBGraph
        graph = TBGraph(CyTBGraph((1, 1, 1), (256, 1, 1), 1, 64))
        tensors = [self.dt[name] for name in names]
        for tensor in tensors:
            graph.new_input(tensor, (-1, -1, -1), -1, True)
        self.pk.kn_graph.customized(tensors, graph)
        self.pk.kn_graph.register_task(graph, kind)

    def linear(self, name, source, weight, width):
        self.empty(name, (64, width))
        self.task(2, [source, weight, "query_length", name], [-1, 0, -1, 1], width//64)
        return name

    def gdn(self, prefix, weights, normalized):
        for name in ("qkv_weight", "z_weight", "ab_weight", "parameters", "conv_weight", "gated_norm_weight"):
            self.attach(prefix+name, weights[name])
        qkv = self.linear(prefix+"qkv", normalized, prefix+"qkv_weight", self.conv_width)
        z = self.linear(prefix+"z", normalized, prefix+"z_weight", self.vh*128)
        ab = self.linear(prefix+"ab", normalized, prefix+"ab_weight", self.ab_width)
        self.empty(prefix+"conv_in", (self.conv_width, 3))
        self.empty(prefix+"conv_out", (self.conv_width, 3))
        self.empty(prefix+"conv", (64, self.conv_width))
        self.task(3, [qkv, prefix+"conv_weight", prefix+"conv_in", "length",
                      prefix+"conv", prefix+"conv_out"], [1, 0, 0, -1, 1, 0], self.conv_width//128)
        self.dt[prefix+"conv_heads"] = self.pk.view(self.dt[prefix+"conv"], [64, 2*self.kh+self.vh, 128])
        self.dt[prefix+"z_heads"] = self.pk.view(self.dt[z], [64, self.vh, 128])
        self.dt[prefix+"ab_heads"] = self.dt[ab]
        self.empty(prefix+"state_in", (self.vh, 128, 128), torch.float32)
        self.empty(prefix+"state_out", (self.vh, 128, 128), torch.float32)
        self.empty(prefix+"gated", (64, self.vh, 128))
        self.empty(prefix+"scratch", (self.vh, 9*8192+3*4096+64+64*self.vh*128//2), torch.float32)
        names = [prefix+"conv_heads", prefix+"z_heads", prefix+"ab_heads", prefix+"parameters",
                 prefix+"gated_norm_weight", prefix+"state_in", "chunk_lengths"]
        partitions = [-1, 1, -1, 1, -1, 0, -1]
        self.attach(prefix+"head_ids", torch.arange(self.vh, device="cuda", dtype=torch.int64).reshape(self.vh, 1))
        names.append(prefix+"head_ids")
        partitions.append(0)
        self.task(11,
                  names + [prefix+"gated", prefix+"state_out", prefix+"scratch"],
                  partitions + [1, 0, 0], self.vh)
        self.dt[prefix+"gated_flat"] = self.pk.view(self.dt[prefix+"gated"], [64, self.vh*128])
        return self.linear(prefix+"mixed", prefix+"gated_flat", prefix+"out_weight", self.hidden)

    def attention(self, prefix, weights, normalized):
        for name in ("qg_weight", "k_weight", "v_weight", "q_norm", "k_norm"):
            self.attach(prefix+name, weights[name])
        qg = self.linear(prefix+"qg", normalized, prefix+"qg_weight", self.qh*512)
        key = self.linear(prefix+"k", normalized, prefix+"k_weight", self.kvh*256)
        value = self.linear(prefix+"v", normalized, prefix+"v_weight", self.kvh*256)
        self.dt[prefix+"qg_heads"] = self.pk.view(self.dt[qg], [64, self.qh, 512])
        self.dt[prefix+"k_heads"] = self.pk.view(self.dt[key], [64, self.kvh, 256])
        self.empty(prefix+"query", (64, self.qh, 256))
        self.empty(prefix+"key", (64, self.kvh, 256))
        self.empty(prefix+"gate", (64, self.qh, 256))
        self.task(5, [prefix+"qg_heads", prefix+"k_heads", prefix+"q_norm", prefix+"k_norm", "rope", "query_length",
                      prefix+"query", prefix+"key", prefix+"gate"], [0, 0, -1, -1, 0, -1, 0, 0, 0], 64)
        self.empty(prefix+"key_cache", (1024, self.kvh, 256))
        self.empty(prefix+"value_cache", (1024, self.kvh, 256))
        self.task(6, [prefix+"key", value, "metadata", prefix+"key_cache", prefix+"value_cache"], [-1]*5, 1)
        self.dt[prefix+"queries"] = self.pk.view(self.dt[prefix+"query"], [64*self.qh, 256])
        self.dt[prefix+"gates"] = self.pk.view(self.dt[prefix+"gate"], [64*self.qh, 256])
        self.empty(prefix+"gated", (64*self.qh, 256))
        sdpa = self.model.attn_implementation == "sdpa"
        self.empty(prefix+"attention_scratch", (64*self.qh, 4112 if sdpa else 1536), torch.float32)
        self.task(7, [prefix+"queries", prefix+"gates", "positions", prefix+"key_cache", prefix+"value_cache",
                      "head_ids", "split_config", prefix+"gated", prefix+"attention_scratch"], [0, 0, 0, -1, -1, 0, -1, 0, 0],
                  64*self.qh, backend=int(sdpa))
        self.dt[prefix+"gated_flat"] = self.pk.view(self.dt[prefix+"gated"], [64, self.qh*256])
        return self.linear(prefix+"mixed", prefix+"gated_flat", prefix+"out_weight", self.hidden)

    def compile(self, directory):
        if self.compiled or self.closed:
            raise RuntimeError("Compile prefill exactly once before closing")
        self.pk.compile(output_dir=str(Path(directory)))
        self.compiled = True

    def launch(self):
        self.pk.init_request_func()
        self.pk()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def run(self, tokens):
        if not self.compiled or self.closed:
            raise RuntimeError("Compile MPK prefill before running")
        model, t = self.model, self.tensors
        splits = (flash_split_config(len(tokens), model.position+len(tokens),
                                     torch.cuda.get_device_properties(0).multi_processor_count, self.qh)
                  if model.attn_implementation == "sdpa" else
                  eager_mma_config(len(tokens), model.position+len(tokens)))
        t["split_config"].copy_(torch.tensor(splits, device="cuda", dtype=torch.int64))
        # Start from committed decode state. Retrying a failed operation never
        # reuses partially updated scratch from the previous attempt.
        for layer, info in zip(model.layers, self.layers):
            p = info["prefix"]
            if info["gdn"]:
                t[p+"state_in"].copy_(layer.tensors["state"])
                t[p+"conv_in"].copy_(layer.tensors["conv_state"])
            else:
                for name in ("key_cache", "value_cache"):
                    t[p+name].copy_(layer.tensors[name])
        rope = model.layers[model.config.attention_layers[0]].mixer.rope_table
        for offset in range(0, len(tokens), 64):
            chunk = tokens[offset:offset+64]
            past, length = model.position+offset, len(chunk)
            t["tokens"].zero_()
            t["tokens"][:length].copy_(torch.tensor(chunk, device="cuda", dtype=torch.int64))
            t["metadata"].copy_(torch.tensor([past, length, len(tokens)], device="cuda", dtype=torch.int64))
            positions = [[pos] for pos in range(past, past+length) for _ in range(self.qh)]
            positions += [[-1]] * ((64-length)*self.qh)
            t["positions"].copy_(torch.tensor(positions, device="cuda", dtype=torch.int64))
            t["rope"].zero_()
            t["rope"][:length].copy_(rope[past:past+length])
            self.launch()
            for info in self.layers:
                if info["gdn"]:
                    p = info["prefix"]
                    t[p+"state_in"].copy_(t[p+"state_out"])
                    t[p+"conv_in"].copy_(t[p+"conv_out"])
        torch.cuda.synchronize()
        end, last = model.position+len(tokens), (len(tokens)-1)%64
        # Commit only after all tiles have completed successfully.
        for layer, info in zip(model.layers, self.layers):
            p = info["prefix"]
            if info["gdn"]:
                layer.tensors["state"].copy_(t[p+"state_out"])
                layer.tensors["conv_state"].copy_(t[p+"conv_out"])
            else:
                for name in ("key_cache", "value_cache"):
                    layer.tensors[name][:end].copy_(t[p+name][:end])
            layer.tensors["decoder_output"].copy_(t[p+"output"][last:last+1])
            layer.mixer.tensors["position"].fill_(end-1)
        for name in ("logits", "next_token"):
            model.tensors[name].copy_(t[name])
        model.tensors["hidden"].copy_(t["embedding"][last:last+1])
        model.tensors["normalized"].copy_(t["last_normalized"])
        model.tensors["token"].fill_(tokens[-1])
        torch.cuda.synchronize()
        model.position = end
        for layer in model.layers:
            layer.mixer.position = end
        return model.tensors["logits"]

    def close(self):
        if not self.closed:
            self.pk.finalize()
            self.closed = True
