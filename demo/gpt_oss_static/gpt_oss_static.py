"""Greedy GPT-OSS-20B/120B decode on Hopper with one static megakernel per layer."""
import argparse, itertools, json, os, statistics, time
# 120B's weights fill most of an H100: the expert repacking's freed temporaries must be reusable for the next layer's
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import (StaticMegakernel, sm90_attention_parts_size, sm90_flags,
                                         sm90_moe_ring_pack)
from mirage.mpk.models.utils import shuffle_tensors
from mxfp4 import Mxfp4Checkpoint, convert_expert_weight

T, PAGE = 8, 64                       # token rows of a step, KV page size
KV_CHUNK = 128                        # full attention: the KV split in chunks of this many positions (sm90_attention_split)
# the gate's router over this many tasks a token (sm90_gate_layer), by expert count: measured on H100 (20B: 2 and 4; 120B: 4,
# 8 and 16)
ROUTER_SPLITS = {32: 4, 128: 8}
L2_PREFETCH_MB = 12   # each layer's first QKV weight bytes into L2 at its start (StaticMegakernel.l2_prefetch_weights)
FLAGS = sm90_flags() + ["-DSTATIC_RESET_IN_KERNEL", "-DSTATIC_PDL_TRIGGER=1"]   # serving build: launches back to back


class Model:
    """The checkpoint's config and weights, converted for the static kernels (on the GPU), and the step's shared buffers."""

    def __init__(self, ckpt: str, max_seq: int, requests: int = 1):
        from transformers import AutoConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
        cfg = self.cfg = AutoConfig.from_pretrained(ckpt)
        self.H, self.I, self.E, self.TOPK = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts, cfg.num_experts_per_tok
        self.QH, self.KVH, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.L, self.types, self.window, self.V = cfg.num_hidden_layers, list(cfg.layer_types), cfg.sliding_window, cfg.vocab_size
        self.eps, self.limit = cfg.rms_norm_eps, getattr(cfg, "swiglu_limit", 7.0)
        self.max_seq, self.R = max_seq, requests
        R, MP = requests, max_seq // PAGE                            # requests (a token each), pages a request
        assert max_seq % PAGE == 0 and 1 <= R <= T
        sd = Mxfp4Checkpoint(ckpt)
        cos, sin = GptOssRotaryEmbedding(cfg)(torch.empty(0, dtype=torch.bfloat16), torch.arange(max_seq)[None])
        bf = lambda t: t.to(torch.bfloat16).contiguous().cuda()
        zeros = lambda *shape, dtype=torch.bfloat16: torch.zeros(*shape, dtype=dtype, device="cuda")
        H, I, E, TOPK, QH, KVH, D = self.H, self.I, self.E, self.TOPK, self.QH, self.KVH, self.D
        A = min(E, R * TOPK)                                         # expert slots: R routed tokens
        self.shared = {                                              # every layer's (the step's buffers)
            "cos": bf(torch.cat([cos[0], cos[0]], -1)), "sin": bf(torch.cat([sin[0], sin[0]], -1)),
            "qo_indptr": torch.arange(R + 1, dtype=torch.int32, device="cuda"),
            "kv_indptr": torch.arange(R + 1, dtype=torch.int32, device="cuda"),
            "kv_indices": torch.zeros(R * MP, dtype=torch.int32, device="cuda"),   # request r's pages r MP .., packed
            "last_page_len": torch.ones(R, dtype=torch.int32, device="cuda"),
            "num_tokens": torch.full((1,), R, dtype=torch.int32, device="cuda"),
            "attn_parts": zeros(sm90_attention_parts_size(T, QH, D, max_seq // KV_CHUNK)),   # the split attention's chunks
            "h": zeros(T, H), "qkv32": zeros(T, (QH + 2 * KVH) * D, dtype=torch.float32),
            "qkv": zeros(T, (QH + 2 * KVH) * D), "attn": zeros(T, QH * D), "o32": zeros(T, H, dtype=torch.float32),
            "x1": zeros(T, H), "h2": zeros(T, H, dtype=torch.float16),   # h2: the MoE's f16 input
            "topk_w": zeros(T, TOPK, dtype=torch.float32), "routing": zeros(E, T, dtype=torch.int32),
            "active": zeros(E + 1, dtype=torch.int32), "act": zeros(A, T, I, dtype=torch.float16), "eo": zeros(T, TOPK, H),
        }
        self.x = [zeros(T, H), zeros(T, H)]                          # the residual stream, ping-pong between layers
        self.x_embed = zeros(T, H)                                   # layer 0's input: the head writes the next token's row
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            at = lambda k: sd[p + k]
            w13, s13, b13 = convert_expert_weight(at("mlp.experts.gate_up_proj_blocks"), at("mlp.experts.gate_up_proj_scales"),
                                                  at("mlp.experts.gate_up_proj_bias"), deinterleave=True)
            w2, s2, b2 = convert_expert_weight(at("mlp.experts.down_proj_blocks"), at("mlp.experts.down_proj_scales"),
                                               at("mlp.experts.down_proj_bias"), deinterleave=False)
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
                "k_cache": zeros(R * MP, PAGE, KVH, D), "v_cache": zeros(R * MP, PAGE, KVH, D),
            })
        self.embed = bf(sd["model.embed_tokens.weight"])
        self.head = {"norm": bf(sd["model.norm.weight"]), "lm_head": bf(sd["lm_head.weight"]),
                     "h": zeros(T, H), "logits": zeros(T, self.V, dtype=torch.float32)}

    def layer_tensors(self, i: int) -> dict:
        """Layer i's tensors by graph name: its weights and cache, the shared buffers, its residual in and out."""
        return {**self.shared, **self.layers[i], "x_in": self.x_embed if i == 0 else self.x[i % 2],
                "x_out": self.x[(i + 1) % 2]}

    def head_tensors(self, prompt: torch.Tensor, gen: torch.Tensor) -> dict:
        """The final norm and lm_head, then the next step's tokens (prompt [R, P] i64; gen [R, NEW] i64: the generated
        tokens)."""
        return {"x": self.x[self.L % 2], **self.head, "embed": self.embed, "prompt": prompt, "gen": gen,
                "x_embed": self.x_embed, "kv_indptr": self.shared["kv_indptr"],
                "last_page_len": self.shared["last_page_len"], "kv_indices": self.shared["kv_indices"]}

    def set_position(self, pos: int):
        """The page table for a step of every request at position pos (its K / V go to slot pos; request r's pages
        r MP .., in order)."""
        n, MP = pos // PAGE + 1, self.max_seq // PAGE
        self.shared["kv_indptr"].copy_(torch.arange(self.R + 1, dtype=torch.int32, device="cuda") * n)
        self.shared["kv_indices"][:self.R * n].copy_(
            (torch.arange(self.R, device="cuda")[:, None] * MP + torch.arange(n, device="cuda")).flatten())
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
    # the MXFP4 scales in fp32 after the MMAs, both models: the f16 path's scaled decode costs more than it saves (20B, H100:
    # 2957 -> 2829 us/step)
    mpk.sm90_moe_ring_layer(A["h2"], A["w13"], A["routing"], A["active"], A["act"], bias=A["b13"], glu=True,
                            limit=m.limit, alpha=1.702, rows_per_task=128, fp32_scales=True)
    mpk.sm90_moe_ring_layer(A["act"], A["w2"], A["routing"], A["active"], A["eo"], bias=A["b2"], glu=False,
                            rows_per_task=64, fp32_scales=True)
    mpk.sm90_mul_sum_add_layer(A["eo"], A["topk_w"], A["x1"], A["x_out"])
    # the QKV weight's first 12 MB into L2 by the SMs that start while the previous layer's kernel finishes (PDL): its
    # tail leaves HBM idle (H100, same runs: 20B 2834 -> 2791, 120B 4133 -> 4060 us/step; 16 MB and up gain less: the
    # prefetch then slows that tail)
    mpk.l2_prefetch_weights(L2_PREFETCH_MB, stop_at=("sm90_attention", "sm90_attention_split"))


def build_head(m: Model, mpk: StaticMegakernel, t: dict):
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    mpk.sm90_rmsnorm_layer(A["x"], A["norm"], A["h"], m.eps)
    mpk.gemm_tile_layer(input=A["h"], weight=A["lm_head"], output=A["logits"], combine="store")
    # the next step's token (the prompt's, else the argmax), its embedding row and page table: no launch between the steps
    mpk.sm90_next_token_layer(A["logits"], A["embed"], A["prompt"], A["x_embed"], A["gen"], A["kv_indptr"],
                              A["last_page_len"], A["kv_indices"], PAGE)


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
    if (name != "head" and torch.cuda.get_device_name() == "NVIDIA H100 80GB HBM3" and mpk._num_sms() == 132
            and os.environ.get("PLAN", "file") == "file"):
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
    p.add_argument("--batch", type=int, default=1, help="requests a step (at most T)")
    p.add_argument("--prompts", default=None, help="a json list of user messages: request r's is r % len (all cut to the "
                   "shortest's tokens); default the one message")
    p.add_argument("--only", type=int, default=None, help="with --prompts: one request, message ONLY")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    R = args.batch
    if R > 1:   # the H100 plans place one request's grids (TOPK expert slots)
        os.environ.setdefault("PLAN", "rr")
    ref = json.load(open(args.ref)) if args.ref else None
    if args.prompts:
        msgs = json.load(open(args.prompts))
        ids = [tok(tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                           add_generation_prompt=True)).input_ids for c in msgs]
        P = min(len(i) for i in ids)
        ids = [i[:P] for i in ids]
        prompts = [ids[args.only]] * R if args.only is not None else [ids[r % len(ids)] for r in range(R)]
    elif ref:
        prompts = [ref["prompt_ids"]] * R
    else:
        text = tok.apply_chat_template([{"role": "user", "content": "Give me a short introduction to large language models."}],
                                       tokenize=False, add_generation_prompt=True)
        prompts = [tok(text).input_ids] * R
    P, NEW = len(prompts[0]), args.new
    t0 = time.time()
    m = Model(args.ckpt, args.max_seq, R)
    assert P + NEW <= m.max_seq
    print(f"weights loaded and converted in {time.time() - t0:.0f} s", flush=True)
    # the kernels, and one instance per layer (layer i: instance inst[i] of kernel kind[i])
    sliding = [i for i in range(m.L) if m.types[i] == "sliding_attention"]
    full = [i for i in range(m.L) if m.types[i] != "sliding_attention"]
    k_sl = compile_kernel("sliding", lambda mpk, t: build_layer(m, mpk, t, m.window), m.layer_tensors(sliding[0]), args.out)
    k_fu = compile_kernel("full", lambda mpk, t: build_layer(m, mpk, t, 0), m.layer_tensors(full[0]), args.out)
    gen = torch.zeros(R, NEW, dtype=torch.long, device="cuda")
    prompt_t = torch.tensor(prompts, dtype=torch.long, device="cuda")
    k_hd = compile_kernel("head", lambda mpk, t: build_head(m, mpk, t), m.head_tensors(prompt_t, gen), args.out)
    kernel_of, inst = {}, {}
    for i in range(m.L):
        kernel_of[i] = k_sl if i in sliding else k_fu
        inst[i] = kernel_of[i].add_instance(m.layer_tensors(i))
    k_hd.add_instance(m.head_tensors(prompt_t, gen))

    def step():
        """One position through the model; the head leaves the next position's token, embedding row and page table."""
        for i in range(m.L):
            kernel_of[i].launch_instances(inst[i], 1)
        k_hd.launch_instances(0, 1)

    for run in range(2):   # the first run warms up (and checks); the second is timed
        for layer in m.layers:
            layer["k_cache"].zero_()
            layer["v_cache"].zero_()
        m.set_position(0)   # position 0: the prompts' first tokens
        m.x_embed[:R].copy_(m.embed[prompt_t[:, 0]])
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(P + NEW)]
        s.record()
        for pos in range(P + NEW - 1):
            step()
            marks[pos].record()
        e.record()
        torch.cuda.synchronize()
        total = s.elapsed_time(e)
        per = [marks[i - 1].elapsed_time(marks[i]) for i in range(1, P + NEW - 1)]
        # as the vLLM / SGLang sweep's (T(NEW tokens) - T(NEW / 2 tokens)) / (NEW / 2): the steps of the later half's tokens
        dec = marks[P + NEW // 2 - 2].elapsed_time(marks[P + NEW - 2]) / (NEW - NEW // 2)
        print(f"run {run}: {P + NEW - 1} positions in {total:.1f} ms; per step median {statistics.median(per) * 1e3:.0f} us "
              f"(min {min(per) * 1e3:.0f}, max {max(per) * 1e3:.0f}); decode {dec * 1e3:.0f} us/step, batch {R}: "
              f"{R / dec * 1e3:.0f} tok/s", flush=True)
    out = gen.tolist()
    text = tok.decode(out[0])
    print("text:", repr(text[:400]))
    if ref:
        refs = ref["gen"] if isinstance(ref["gen"][0], list) else [ref["gen"]]
        for r in range(R):
            want = refs[r % len(refs)]
            same = next((i for i, (a, b) in enumerate(zip(out[r], want)) if a != b), min(len(out[r]), len(want)))
            print(f"request {r}: tokens equal to the reference's for the first {same} of {min(len(out[r]), len(want))}")
    if args.json:
        json.dump({"prompt_ids": prompts if R > 1 else prompts[0], "prompt_len": P, "gen": out if R > 1 else out[0],
                   "text": text, "total_ms": total, "step_us_median": statistics.median(per) * 1e3,
                   "decode_us": dec * 1e3}, open(args.json, "w"))
    for k in (k_sl, k_fu, k_hd):
        k.finalize()


if __name__ == "__main__":
    main()
