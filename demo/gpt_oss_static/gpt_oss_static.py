"""GPT-OSS (MXFP4 checkpoint) decoding on Hopper through the static megakernel (PR 786's compiler), one request, greedy:

  layer kernels  one per attention kind (sliding window / full; the window is a template argument): the decoder layer as 12
                 nodes (static_megakernel.py: sm90_rmsnorm, gemm_tile + sm90_bias for qkv and o_proj, sm90_router,
                 sm90_attention, sm90_topk, sm90_moe_ring W13 (+ SwiGLU) and W2, sm90_mul_sum_add); each kernel runs as one
                 instance per layer of its kind (its own weights, KV cache, buffers), the layers' kernels in order on one stream,
                 each a programmatic dependent launch (PDL) of the one before
  head kernel    the final RMSNorm and lm_head (gemm_tile)
  torch          the embedding gather, the argmax, the page table (filled before each step)

A step is one token (the prompt's, then the generated ones): the kernels hold 8 token rows, row 0 is the request's.

  --plan rr             every node's default grid, tasks dealt round-robin over the SMs (no costs needed)
  --record COSTS        the timing build of the rr plan: each kernel launched alone (L2 emptied) at a mid-sequence step; its
                        per-task times appended to COSTS as cost records (cost_records.py)
  --search COSTS ...    the solver's plan (search.search_plan, OR-Tools) from those records, each kernel's own

python gpt_oss_static.py <mxfp4 checkpoint> [--ref ref.json] [--new 200] [--plan rr] [--record F | --search F ...] [--out DIR]
--ref: a run's {"prompt_ids", "gen"} (e.g. the MPK path's gptoss_e2e.py output): its prompt, and its tokens compared with ours.
"""
import argparse, json, os, statistics, subprocess, sys, time
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel, sm90_flags, sm90_moe_ring_pack
from mirage.mpk.static_harness import Harness
from mirage.mpk.models.utils import shuffle_tensors
from mirage.mpk.models.gpt_oss.mxfp4 import Mxfp4Checkpoint, convert_expert_weight

T, PAGE = 8, 64                       # token rows of a step, KV page size
FLAGS = sm90_flags() + ["-DSTATIC_RESET_IN_KERNEL", "-DSTATIC_PDL_TRIGGER=1"]   # serving build: launches back to back
FLUSH = 512 << 20


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
            "h": zeros(T, H), "qkv32": zeros(T, (QH + 2 * KVH) * D, dtype=torch.float32),
            "qkv": zeros(T, (QH + 2 * KVH) * D), "attn": zeros(T, QH * D), "o32": zeros(T, H, dtype=torch.float32),
            "x1": zeros(T, H), "h2": zeros(T, H), "logits": zeros(T, E),
            "topk_w": zeros(T, TOPK, dtype=torch.float32), "routing": zeros(E, T, dtype=torch.int32),
            "active": zeros(E + 1, dtype=torch.int32), "act": zeros(A, T, I), "eo": zeros(T, TOPK, H),
        }
        self.x = [zeros(T, H), zeros(T, H)]                          # the residual stream, ping-pong between layers
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
    mpk.sm90_bias_layer(A["qkv32"], A["b_qkv"], A["qkv"])
    mpk.sm90_attention_layer(A["qkv"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"], A["kv_indptr"],
                             A["kv_indices"], A["last_page_len"], A["attn"], num_q_heads=m.QH, window=window,
                             sinks=A["sinks"])
    mpk.gemm_tile_layer(input=A["attn"], weight=A["w_o"], output=A["o32"], combine="store")
    mpk.sm90_bias_layer(A["o32"], A["b_o"], A["x1"], residual=A["x_in"])
    mpk.sm90_rmsnorm_layer(A["x1"], A["ln2"], A["h2"], m.eps)
    mpk.sm90_router_layer(A["h2"], A["w_router"], A["b_router"], A["logits"])
    mpk.sm90_topk_layer(A["logits"], A["num_tokens"], A["topk_w"], A["routing"], A["active"])
    mpk.sm90_moe_ring_layer(A["h2"], A["w13"], A["routing"], A["active"], A["act"], bias=A["b13"], glu=True,
                            limit=m.limit, alpha=1.702, rows_per_task=128)
    mpk.sm90_moe_ring_layer(A["act"], A["w2"], A["routing"], A["active"], A["eo"], bias=A["b2"], glu=False,
                            rows_per_task=64)
    mpk.sm90_mul_sum_add_layer(A["eo"], A["topk_w"], A["x1"], A["x_out"])


def build_head(m: Model, mpk: StaticMegakernel, t: dict):
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    mpk.sm90_rmsnorm_layer(A["x"], A["norm"], A["h"], m.eps)
    mpk.gemm_tile_layer(input=A["h"], weight=A["lm_head"], output=A["logits"], combine="store")


def round_robin_plan(mpk, most_tasks: bool = False) -> dict:
    """Every node's default grid (most_tasks: the candidate grid with the most tasks, compiler.candidate_grids), its tasks
    dealt round-robin over the SMs in graph order."""
    nsm = mpk._num_sms()
    lists, k = [[] for _ in range(nsm)], 0
    grid_of = lambda n: (max(compiler.candidate_grids(n), key=lambda g: g[0] * g[1] * g[2]) if most_tasks else tuple(n.grid))
    grids = {n.graph_idx: grid_of(n) for n in compiler.graph_nodes(mpk)}
    for node, grid in grids.items():
        for z in range(grid[2]):
            for y in range(grid[1]):
                for x in range(grid[0]):
                    lists[k % nsm].append((node, (x, y, z)))
                    k += 1
    return {"grids": grids, "lists": [lists]}


def compile_kernel(name, m, build, t, args, profile=False, most_tasks=False):
    """One kernel: the graph, its plan (args.plan / args.search), the build; returns (mpk, plan paths, StaticKernel, plan)."""
    mpk = StaticMegakernel(num_gpus=1)
    build(mpk, t)
    plan = round_robin_plan(mpk, most_tasks)
    plan_file = os.path.join(args.out, f"{name}_plan.json")
    if args.search and os.path.exists(plan_file):   # this kernel's search ran already
        plan = compiler.plan_from_file(plan_file)
        print(f"{name}: the solver's plan from {plan_file}", flush=True)
    elif args.search:
        from mirage.mpk import plan_viewer, search, static_schedule
        costs = search.Costs(args.search, code=static_schedule.code_tag(), gpu=torch.cuda.get_device_name(0))
        plan, summary = search.search_plan(mpk, costs, time_limit_s=args.time, workers=args.workers, start_from=plan)
        print(f"{name}: solver {summary.get('status')} predicted {summary.get('predicted_us', summary.get('time_us'))} us",
              flush=True)
        compiler.write_plan_file(plan_file, plan)
        plan_viewer.save(os.path.join(args.out, f"{name}_plan_view.json"), plan, summary, "searched from round robin")
    paths, _ = compiler.compile_plan(mpk, plan, os.path.join(args.out, name, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(
        args.out, name, "build" + ("_prof" if profile else "") + ("_most" if most_tasks else "")),
                            extra_flags=FLAGS, profile=profile)
    return mpk, paths, sk, plan


def node_summary(mpk, plan: dict, times) -> None:
    """Per node of a timing build's launch: its tasks' median / max time and the window from its first start to its last end."""
    lists = plan["lists"][0]
    t0 = min(e[0] for sm in times for e in sm if e[0])
    names = {n.graph_idx: n.name for n in compiler.graph_nodes(mpk)}
    for node in plan["grids"]:
        d = [times[sm][i] for sm in range(len(lists)) for i, (n, _) in enumerate(lists[sm]) if n == node and times[sm][i][0]]
        if not d:
            continue
        dur = sorted((e[1] - e[0]) / 1e3 for e in d)
        print(f"  node {node:2d} {names[node]:<16} {len(d):4d} tasks, median {dur[len(dur) // 2]:6.1f} us, max {dur[-1]:6.1f} us, "
              f"window {(min(e[0] for e in d) - t0) / 1e3:6.1f} .. {(max(e[1] for e in d) - t0) / 1e3:6.1f} us")


def record_costs(args, m: Model):
    """--record: the timing build of each kernel, launched alone (L2 emptied) at a mid-sequence position, cost records out."""
    from mirage.mpk import cost_records
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
    m.set_position(m.max_seq // 2)
    m.shared["active"][m.E] = m.TOPK
    kinds = [("sliding", lambda mpk, t: build_layer(m, mpk, t, m.window), m.layer_tensors(0)),
             ("full", lambda mpk, t: build_layer(m, mpk, t, 0), m.layer_tensors(1)),
             ("head", lambda mpk, t: build_head(m, mpk, t), m.head_tensors())]
    for (name, build, t), most in [(k, most) for k in kinds for most in (False, True)]:
        mpk, paths, sk, plan = compile_kernel(name, m, build, t, args, profile=True, most_tasks=most)
        h, launches, spans = Harness(sk), [], []
        for _ in range(args.reps):
            h.launch(FLUSH, timeout_s=30)
            launches.append((h.task_times(), h.timing()))
            spans.append(h.span_us())
        print(f"{name}{' (most tasks)' if most else ''}: span median {statistics.median(spans):.1f} us over {args.reps} "
              f"launches; the last one per node:")
        node_summary(mpk, plan, launches[-1][0][0])
        records = cost_records.records_of_run(mpk, paths, launches, f"gpt_oss_{name}" + ("_most" if most else ""), machine)
        cost_records.append_records(args.record, records)
        print(f"{name}: {len(records)} cost records -> {args.record}", flush=True)
        sk.finalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--ref", default=None)
    p.add_argument("--new", type=int, default=200)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--plan", default="rr", choices=["rr"])
    p.add_argument("--record", default=None)
    p.add_argument("--search", nargs="+", default=None)
    p.add_argument("--time", type=float, default=120)
    p.add_argument("--workers", type=int, default=4, help="with --search: CP-SAT workers")
    p.add_argument("--reps", type=int, default=10)
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
    if args.record:
        record_costs(args, m)
        return

    # the kernels, and one instance per layer (layer i: instance inst[i] of kernel kind[i])
    sliding = [i for i in range(m.L) if m.types[i] == "sliding_attention"]
    full = [i for i in range(m.L) if m.types[i] != "sliding_attention"]
    *_, k_sl, _ = compile_kernel("sliding", m, lambda mpk, t: build_layer(m, mpk, t, m.window), m.layer_tensors(sliding[0]), args)
    *_, k_fu, _ = compile_kernel("full", m, lambda mpk, t: build_layer(m, mpk, t, 0), m.layer_tensors(full[0]), args)
    *_, k_hd, _ = compile_kernel("head", m, lambda mpk, t: build_head(m, mpk, t), m.head_tensors(), args)
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
