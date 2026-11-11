"""Greedy GPT-OSS-20B/120B decode on Hopper with one static megakernel per layer."""
import argparse, itertools, json, os, statistics, time
# 120B's weights fill most of an H100: the expert repacking's freed temporaries must be reusable for the next layer's
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import (StaticMegakernel, sm90_attention_parts_size, sm90_flags, sm90_moe_ring_f16_scales,
                                         sm90_moe_ring_pack)
from mirage.mpk.models.utils import shuffle_tensors
from mxfp4 import Mxfp4Checkpoint, convert_expert_weight

T, PAGE = 8, 64                       # token rows of a step, KV page size
KV_CHUNK = 128                        # full attention: the KV split in chunks of this many positions (sm90_attention_split)
# the gate's router over this many tasks a token (sm90_gate_layer), by expert count: measured on H100 (20B: 2 and 4; 120B: 4,
# 8 and 16)
ROUTER_SPLITS = {32: 4, 128: 8}
FLAGS = sm90_flags() + ["-DSTATIC_RESET_IN_KERNEL", "-DSTATIC_PDL_TRIGGER=1"]   # serving build: launches back to back


class Model:
    """The checkpoint's config and weights, converted for the static kernels (on the GPU), and the step's shared buffers."""

    def __init__(self, ckpt: str, max_seq: int):
        from transformers import AutoConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
        cfg = self.cfg = AutoConfig.from_pretrained(ckpt)
        self.H, self.I, self.E, self.TOPK = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts, cfg.num_experts_per_tok
        self.QH, self.KVH, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.L, self.types, self.window, self.V = cfg.num_hidden_layers, list(cfg.layer_types), cfg.sliding_window, cfg.vocab_size
        self.eps, self.limit = cfg.rms_norm_eps, getattr(cfg, "swiglu_limit", 7.0)
        self.max_seq = max_seq
        assert max_seq % PAGE == 0
        sd = Mxfp4Checkpoint(ckpt)
        cos, sin = GptOssRotaryEmbedding(cfg)(torch.empty(0, dtype=torch.bfloat16), torch.arange(max_seq)[None])
        bf = lambda t: t.to(torch.bfloat16).contiguous().cuda()
        zeros = lambda *shape, dtype=torch.bfloat16: torch.zeros(*shape, dtype=dtype, device="cuda")
        H, I, E, TOPK, QH, KVH, D = self.H, self.I, self.E, self.TOPK, self.QH, self.KVH, self.D
        A = TOPK                                                     # expert slots: one routed token
        self.shared = {                                              # every layer's (the step's buffers)
            "cos": bf(torch.cat([cos[0], cos[0]], -1)), "sin": bf(torch.cat([sin[0], sin[0]], -1)),
            "qo_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            "kv_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            "kv_indices": torch.arange(max_seq // PAGE, dtype=torch.int32, device="cuda"),
            "last_page_len": torch.ones(1, dtype=torch.int32, device="cuda"),
            "num_tokens": torch.ones(1, dtype=torch.int32, device="cuda"),
            "attn_parts": zeros(sm90_attention_parts_size(T, QH, D, max_seq // KV_CHUNK)),   # the split attention's chunks
            "h": zeros(T, H), "qkv32": zeros(T, (QH + 2 * KVH) * D, dtype=torch.float32),
            "qkv": zeros(T, (QH + 2 * KVH) * D), "attn": zeros(T, QH * D), "o32": zeros(T, H, dtype=torch.float32),
            "x1": zeros(T, H), "h2": zeros(T, H, dtype=torch.float16),   # h2: the MoE's f16 input
            "topk_w": zeros(T, TOPK, dtype=torch.float32), "routing": zeros(E, T, dtype=torch.int32),
            "active": zeros(E + 1, dtype=torch.int32), "act": zeros(A, T, I, dtype=torch.float16), "eo": zeros(T, TOPK, H),
        }
        self.x = [zeros(T, H), zeros(T, H)]                          # the residual stream, ping-pong between layers
        self.layers = []
        self.f16_scales = True   # every layer's MXFP4 scales fit the ring's f16 path (20B's do; 120B's need fp32)
        for i in range(self.L):
            p = f"model.layers.{i}."
            at = lambda k: sd[p + k]
            w13, s13, b13 = convert_expert_weight(at("mlp.experts.gate_up_proj_blocks"), at("mlp.experts.gate_up_proj_scales"),
                                                  at("mlp.experts.gate_up_proj_bias"), deinterleave=True)
            w2, s2, b2 = convert_expert_weight(at("mlp.experts.down_proj_blocks"), at("mlp.experts.down_proj_scales"),
                                               at("mlp.experts.down_proj_bias"), deinterleave=False)
            self.f16_scales = self.f16_scales and sm90_moe_ring_f16_scales(s13) and sm90_moe_ring_f16_scales(s2)
            q = [at(f"self_attn.{n}_proj.weight") for n in "qkv"]
            qb = [at(f"self_attn.{n}_proj.bias") for n in "qkv"]
            self.layers.append({
                "ln1": bf(at("input_layernorm.weight")), "ln2": bf(at("post_attention_layernorm.weight")),
                "w_qkv": bf(shuffle_tensors(q, KVH, 0)), "b_qkv": bf(shuffle_tensors(qb, KVH, 0)),
                "w_o": bf(at("self_attn.o_proj.weight")), "b_o": bf(at("self_attn.o_proj.bias")),
                "sinks": bf(at("self_attn.sinks").view(KVH, QH // KVH)),
                "w_router": bf(at("mlp.router.weight")), "b_router": bf(at("mlp.router.bias")),
                "w13": sm90_moe_ring_pack(w13.cuda(), s13.cuda(), 128, glu=True), "b13": bf(b13),
                "w2": sm90_moe_ring_pack(w2.cuda(), s2.cuda(), 64, glu=False), "b2": bf(b2),
                "k_cache": zeros(max_seq // PAGE, PAGE, KVH, D), "v_cache": zeros(max_seq // PAGE, PAGE, KVH, D),
            })
        self.embed = bf(sd["model.embed_tokens.weight"])
        self.head = {"norm": bf(sd["model.norm.weight"]), "lm_head": bf(sd["lm_head.weight"]),
                     "h": zeros(T, H), "logits": zeros(T, self.V, dtype=torch.float32)}

    def layer_tensors(self, i: int) -> dict:
        """Layer i's tensors by graph name: its weights and cache, the shared buffers, its residual in and out."""
        return {**self.shared, **self.layers[i], "x_in": self.x[i % 2], "x_out": self.x[(i + 1) % 2]}

    def head_tensors(self) -> dict:
        return {"x": self.x[self.L % 2], **self.head}

    def set_position(self, pos: int):
        """The page table for a step at position pos (the token's K / V go to slot pos; pages in order)."""
        n = pos // PAGE + 1
        self.shared["kv_indptr"][1:].fill_(n)
        self.shared["last_page_len"].fill_(pos + 1 - (n - 1) * PAGE)


def build_layer(m: Model, mpk: StaticMegakernel, t: dict, window: int):
    """The decoder layer: x_out = x_in + attn(norm(x_in)) + moe(norm(...))."""
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    mpk.sm90_rmsnorm_layer(A["x_in"], A["ln1"], A["h"], m.eps)
    mpk.gemm_tile_layer(input=A["h"], weight=A["w_qkv"], output=A["qkv32"], combine="store")
    qkv_bias = dict(qkv_bias=A["b_qkv"], qkv_staged=A["qkv"])   # the attention tasks add the bias (no bias node)
    if window:   # a sliding window: its two tiles in one task per KV head
        mpk.sm90_attention_layer(A["qkv32"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                 A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["attn"], num_q_heads=m.QH,
                                 window=window, sinks=A["sinks"], **qkv_bias)
    else:        # full attention: the KV in chunks over many tasks, then their merge (with the sinks)
        mpk.sm90_attention_split_layer(A["qkv32"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                       A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["attn_parts"],
                                       num_q_heads=m.QH, chunk=KV_CHUNK, **qkv_bias)
        mpk.sm90_attention_merge_layer(A["attn_parts"], A["qo_indptr"], A["kv_indptr"], A["last_page_len"], A["attn"],
                                       num_q_heads=m.QH, chunk=KV_CHUNK, chunks=m.max_seq // KV_CHUNK, page=PAGE,
                                       sinks=A["sinks"])
    mpk.gemm_tile_layer(input=A["attn"], weight=A["w_o"], output=A["o32"], combine="store")
    mpk.sm90_gate_layer(A["o32"], A["b_o"], A["x_in"], A["ln2"], A["w_router"], A["b_router"], A["num_tokens"], A["x1"],
                        A["h2"], A["topk_w"], A["routing"], A["active"], eps=m.eps,
                        router_splits=ROUTER_SPLITS.get(m.E, 1))
    mpk.sm90_moe_ring_layer(A["h2"], A["w13"], A["routing"], A["active"], A["act"], bias=A["b13"], glu=True,
                            limit=m.limit, alpha=1.702, rows_per_task=128, fp32_scales=not m.f16_scales)
    mpk.sm90_moe_ring_layer(A["act"], A["w2"], A["routing"], A["active"], A["eo"], bias=A["b2"], glu=False,
                            rows_per_task=64, fp32_scales=not m.f16_scales)
    mpk.sm90_mul_sum_add_layer(A["eo"], A["topk_w"], A["x1"], A["x_out"])


def build_head(m: Model, mpk: StaticMegakernel, t: dict):
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    mpk.sm90_rmsnorm_layer(A["x"], A["norm"], A["h"], m.eps)
    mpk.gemm_tile_layer(input=A["h"], weight=A["lm_head"], output=A["logits"], combine="store")


def round_robin_plan(mpk) -> dict:
    """Assign each node's default-grid tasks round-robin over the SMs."""
    nsm = mpk._num_sms()
    lists, k = [[] for _ in range(nsm)], 0
    grids = {n.graph_idx: tuple(n.grid) for n in compiler.graph_nodes(mpk)}
    for node, grid in grids.items():   # grid x outermost: an MoE expert slot's tasks together
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    lists[k % nsm].append((node, (x, y, z)))
                    k += 1
    return {"grids": grids, "lists": [lists]}


def place_router_splits(mpk, plan: dict) -> dict:
    """The plans place the gate's (t, 0, 0) tasks; its router splits (t, s > 0) go to the SMs that have no task from o_proj on
    before their first W13 task (each just before it), the active token's (t = 0) first, the earliest-free SMs first."""
    nodes = {n.graph_idx: n for n in compiler.graph_nodes(mpk)}
    gate = next(i for i, n in nodes.items() if n.name == "sm90_gate")
    T, S = nodes[gate].grid[0], nodes[gate].grid[1]
    if S == 1:
        return plan
    lists = [[list(sm) for sm in gpu] for gpu in plan["lists"]]
    for gpu in lists:
        def before_w13(sm):   # the nodes the SM runs before its first W13 task (the gate's next node)
            return [n for n, _ in itertools.takewhile(lambda e: e[0] != gate + 1, gpu[sm])]
        hosts = sorted((sm for sm in range(len(gpu)) if all(n < gate - 1 for n in before_w13(sm))),
                       key=lambda sm: (max(before_w13(sm), default=-1), len(before_w13(sm)), sm))
        for i, (x, y) in enumerate((x, y) for x in range(T) for y in range(1, S)):
            sm = hosts[i % len(hosts)]
            gpu[sm].insert(len(before_w13(sm)), (gate, (x, y, 0)))
    return {**plan, "lists": lists}


def compile_kernel(name, build, t, out_dir):
    """Build one static kernel for a layer kind or the model head."""
    mpk = StaticMegakernel(num_gpus=1)
    build(mpk, t)
    if name != "head" and torch.cuda.get_device_name() == "NVIDIA H100 80GB HBM3" and mpk._num_sms() == 132:
        plan = place_router_splits(mpk, compiler.plan_from_file(
            os.path.join(os.path.dirname(__file__), "plans", "h100", f"{name}_plan.json")))
    else:
        plan = round_robin_plan(mpk)
    paths, _ = compiler.compile_plan(mpk, plan, os.path.join(out_dir, name, "schedules"))
    return mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out_dir, name, "build"), extra_flags=FLAGS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--ref", default=None)
    p.add_argument("--new", type=int, default=200)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--out", default="gpt_oss_static")
    p.add_argument("--json", default=None, help="write {prompt_ids, gen, text, timing} here")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if args.ref:
        ref = json.load(open(args.ref))
        prompt = ref["prompt_ids"]
    else:
        ref = None
        text = tok.apply_chat_template([{"role": "user", "content": "Give me a short introduction to large language models."}],
                                       tokenize=False, add_generation_prompt=True)
        prompt = tok(text).input_ids
    P, NEW = len(prompt), args.new
    t0 = time.time()
    m = Model(args.ckpt, args.max_seq)
    assert P + NEW <= m.max_seq
    print(f"weights loaded and converted in {time.time() - t0:.0f} s", flush=True)
    # the kernels, and one instance per layer (layer i: instance inst[i] of kernel kind[i])
    sliding = [i for i in range(m.L) if m.types[i] == "sliding_attention"]
    full = [i for i in range(m.L) if m.types[i] != "sliding_attention"]
    k_sl = compile_kernel("sliding", lambda mpk, t: build_layer(m, mpk, t, m.window), m.layer_tensors(sliding[0]), args.out)
    k_fu = compile_kernel("full", lambda mpk, t: build_layer(m, mpk, t, 0), m.layer_tensors(full[0]), args.out)
    k_hd = compile_kernel("head", lambda mpk, t: build_head(m, mpk, t), m.head_tensors(), args.out)
    kernel_of, inst = {}, {}
    for i in range(m.L):
        kernel_of[i] = k_sl if i in sliding else k_fu
        inst[i] = kernel_of[i].add_instance(m.layer_tensors(i))
    k_hd.add_instance(m.head_tensors())

    cur = torch.zeros(1, dtype=torch.long, device="cuda")
    gen = torch.zeros(NEW, dtype=torch.long, device="cuda")
    prompt_t = torch.tensor(prompt, dtype=torch.long, device="cuda")
    x0 = m.x[0]

    def step(pos: int):
        """Position pos: the token there (the prompt's, else the last generated) through the model; the argmax into gen."""
        if pos < P:
            cur.copy_(prompt_t[pos:pos + 1])
        x0[0:1].copy_(m.embed.index_select(0, cur))
        m.set_position(pos)
        for i in range(m.L):
            kernel_of[i].launch_instances(inst[i], 1)
        k_hd.launch_instances(0, 1)
        nxt = m.head["logits"][0].argmax().view(1)
        if pos >= P - 1:
            gen[pos - P + 1:pos - P + 2].copy_(nxt)
        cur.copy_(nxt)

    for run in range(2):   # the first run warms up (and checks); the second is timed
        for layer in m.layers:
            layer["k_cache"].zero_()
            layer["v_cache"].zero_()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(P + NEW)]
        s.record()
        for pos in range(P + NEW - 1):
            step(pos)
            marks[pos].record()
        e.record()
        torch.cuda.synchronize()
        total = s.elapsed_time(e)
        per = [marks[i - 1].elapsed_time(marks[i]) for i in range(1, P + NEW - 1)]
        print(f"run {run}: {P + NEW - 1} positions in {total:.1f} ms; per step median {statistics.median(per) * 1e3:.0f} us "
              f"(min {min(per) * 1e3:.0f}, max {max(per) * 1e3:.0f})", flush=True)
    out = gen.tolist()
    text = tok.decode(out)
    print("text:", repr(text[:400]))
    if ref:
        same = next((i for i, (a, b) in enumerate(zip(out, ref["gen"])) if a != b), min(len(out), len(ref["gen"])))
        print(f"tokens equal to the reference's for the first {same} of {min(len(out), len(ref['gen']))}")
    if args.json:
        json.dump({"prompt_ids": prompt, "prompt_len": P, "gen": out, "text": text, "total_ms": total,
                   "step_us_median": statistics.median(per) * 1e3}, open(args.json, "w"))
    for k in (k_sl, k_fu, k_hd):
        k.finalize()


if __name__ == "__main__":
    main()
