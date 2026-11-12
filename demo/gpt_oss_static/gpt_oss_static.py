"""Greedy GPT-OSS-20B/120B decode on Hopper with one static megakernel per layer."""
import argparse, json, os, statistics, subprocess, time
# 120B's weights fill most of an H100: the expert repacking's freed temporaries must be reusable for the next layer's
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from mirage.mpk import compiler, cost_records, plan_gen, search, static_schedule
from mirage.mpk.static_harness import Harness
from mirage.mpk.static_megakernel import (StaticMegakernel, sm90_attention_parts_size, sm90_flags,
                                         sm90_moe_ring_pack)
from mirage.mpk.models.utils import shuffle_tensors
from mxfp4 import Mxfp4Checkpoint, convert_expert_weight

T, PAGE = 8, 64                       # token rows of a step, KV page size
KV_CHUNK = 128                        # full attention: the KV split in chunks of this many positions (sm90_attention_split)
PLANS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plans")
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
                        A["h2"], A["topk_w"], A["routing"], A["active"], eps=m.eps)   # router splits: the plan's
    # the MXFP4 scales in fp32 after the MMAs, both models: the f16 path's scaled decode costs more than it saves (20B, H100:
    # 2957 -> 2829 us/step)
    # the routed experts as a pool (the other static models' MoE kind): the SMs share the step's units at run time, three
    # blocks a unit when a step has at least an SM's worth of them (sm90_moe_pool_layer big_units)
    mpk.sm90_moe_pool_layer(A["h2"], A["w13"], A["w2"], A["routing"], A["active"], A["act"], A["eo"], b13=A["b13"],
                            b2=A["b2"], limit=m.limit, alpha=1.702, fp32_scales=True, big_units=(3, 3))
    mpk.sm90_mul_sum_add_layer(A["eo"], A["topk_w"], A["x1"], A["x_out"])
    if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the QKV weight's first MB into L2 at the start (as the GLM demos)
        mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))


def build_head(m: Model, mpk: StaticMegakernel, t: dict):
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    mpk.sm90_rmsnorm_layer(A["x"], A["norm"], A["h"], m.eps)
    mpk.gemm_tile_layer(input=A["h"], weight=A["lm_head"], output=A["logits"], combine="store")
    # the next step's token (the prompt's, else the argmax), its embedding row and page table: no launch between the steps
    mpk.sm90_next_token_layer(A["logits"], A["embed"], A["prompt"], A["x_embed"], A["gen"], A["kv_indptr"],
                              A["last_page_len"], A["kv_indices"], PAGE)


def plan_path(m: Model, plans: str, kind: str) -> str:
    """The plan file of a kernel: one per graph (layer kind, expert count, requests)."""
    return os.path.join(plans, f"{kind}_e{m.E}_r{m.R}.json")


def load_plan(mpk, path: str):
    """The plan file at path if it exists and has lists for this GPU's SMs, else None (round robin)."""
    if not path or not os.path.exists(path):
        return None
    p = compiler.plan_from_file(path)
    if len(p["lists"]) != 1 or len(p["lists"][0]) != mpk._num_sms():
        print(f"{path}: lists for another GPU ({len(p['lists'][0])} SMs, this one {mpk._num_sms()}): round robin")
        return None
    return p


def compile_kernel(name, build, t, out_dir, plan=None, profile=False):
    """Build one static kernel for a layer kind or the model head: plan a plan file (load_plan), else round robin."""
    mpk = StaticMegakernel(num_gpus=1)
    build(mpk, t)
    p = load_plan(mpk, plan)
    print(f"{name}: plan {plan if p else 'round robin'}", flush=True)
    paths, _ = compiler.compile_plan(mpk, p or plan_gen.round_robin_plan(mpk), os.path.join(out_dir, name, "schedules"))
    return mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out_dir, name, "build"), extra_flags=FLAGS,
                              profile=profile)


def tune(m: Model, kinds: dict, out: str, plans: str, costs: str, rounds: int, time_limit_s: float, tokens: torch.Tensor,
         reps: int = 10):
    """--tune: per kernel kind, the cost collection (plan_gen.collection_plans and the current plan, each a timing build run
    reps times with L2 emptied, the page table at mid-sequence, every layer input the embedding rows of tokens, one a request,
    so that the requests route as real ones: cost_records appended to costs), then search.search_plan from the current plan
    -> plan_path; last, the span of both plans (serving build). Builds run 8 at a time (the build cache)."""
    from concurrent.futures import ThreadPoolExecutor
    os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE", os.path.join(out, "build_cache"))
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
    os.makedirs(plans, exist_ok=True)
    rows = m.embed[tokens]
    for x in (m.x_embed, *m.x):
        x.zero_()
        x[:len(tokens)].copy_(rows)

    def graph(build, t):
        mpk = StaticMegakernel(num_gpus=1)
        build(mpk, t)
        return mpk

    def compiled(build, t, plan, tag, profile, build_only=False):
        mpk = graph(build, t)
        paths, _ = compiler.compile_plan(mpk, plan, os.path.join(out, tag, "schedules"))
        sk = mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out, tag, "build"), extra_flags=FLAGS,
                                profile=profile, build_only=build_only)
        return mpk, paths, sk

    def launch(h):
        m.set_position(m.max_seq // 2)
        h.launch(256 << 20, timeout_s=30)

    for kind, (build, t) in kinds.items():
        mpk = graph(build, t)
        cur = load_plan(mpk, plan_path(m, plans, kind)) or plan_gen.round_robin_plan(mpk)
        todo = [("cur", cur)] + plan_gen.collection_plans(mpk, rounds=rounds)
        print(f"tune {kind}: {len(todo)} plans", flush=True)
        with ThreadPoolExecutor(8) as ex:   # nvcc in parallel; the runs below load from the cache
            list(ex.map(lambda np_: compiled(build, t, np_[1], f"{kind}/c_{np_[0]}", True, True), todo))
        for name, plan in todo:
            mpk, paths, sk = compiled(build, t, plan, f"{kind}/c_{name}", True)
            h, launches = Harness(sk), []
            for _ in range(reps):
                launch(h)
                launches.append((h.task_times(), h.timing()))
            cost_records.append_records(costs, cost_records.records_of_run(mpk, paths, launches, f"{kind}_{name}", machine))
            print(f"  {name}: span {h.span_us():.1f} us (the last launch)", flush=True)
            sk.finalize()
        mpk = graph(build, t)
        cost = search.Costs(costs, code=static_schedule.code_tag(), gpu=machine["gpu"])
        plan, summary = search.search_plan(mpk, cost, time_limit_s=time_limit_s, start_from=cur)
        print(f"tune {kind}: solver {({k: summary.get(k) for k in ('status', 'predicted_us', 'grids')})}", flush=True)
        spans = {"cur": [], "solved": []}
        for i, (tag, p) in enumerate((("cur", cur), ("solved", plan), ("cur", cur), ("solved", plan))):
            # a build directory each: a loaded layer's .so must not be overwritten
            _, _, sk = compiled(build, t, p, f"{kind}/span_{tag}{i}", False)
            h = Harness(sk)
            for _ in range(30):
                launch(h)
                spans[tag].append(h.span_us())
            print(f"tune {kind}: span {tag} median {statistics.median(spans[tag][-30:]):.1f} us", flush=True)
            sk.finalize()
        best = "solved" if statistics.median(spans["solved"]) < statistics.median(spans["cur"]) else "cur"
        compiler.write_plan_file(plan_path(m, plans, kind), plan if best == "solved" else cur)   # the faster, as measured
        print(f"tune {kind}: kept the {best} plan", flush=True)


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
    p.add_argument("--plans", default=os.path.join(PLANS, "h100"), help="the plan files' directory (plan_path; a missing "
                   "file: round robin)")
    p.add_argument("--tune", metavar="COSTS", default=None, help="collect cost records into COSTS, solve each kernel's "
                   "plan into --plans (tune), and exit")
    p.add_argument("--rounds", type=int, default=2, help="with --tune: random plans per candidate grid of each node")
    p.add_argument("--time", type=float, default=300, help="with --tune: the solver's time limit (s)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    R = args.batch
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
    gen = torch.zeros(R, NEW, dtype=torch.long, device="cuda")
    prompt_t = torch.tensor(prompts, dtype=torch.long, device="cuda")
    kinds = {"sliding": (lambda mpk, t: build_layer(m, mpk, t, m.window), m.layer_tensors(sliding[0])),
             "full": (lambda mpk, t: build_layer(m, mpk, t, 0), m.layer_tensors(full[0])),
             "head": (lambda mpk, t: build_head(m, mpk, t), m.head_tensors(prompt_t, gen))}
    if args.tune:
        tune(m, kinds, os.path.join(args.out, "tune"), args.plans, args.tune, args.rounds, args.time,
             prompt_t[torch.arange(R), (P // 2 + torch.arange(R)) % P])   # a distinct token a request
        return
    k_sl, k_fu, k_hd = (compile_kernel(k, b, t, args.out, plan_path(m, args.plans, k)) for k, (b, t) in kinds.items())
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
