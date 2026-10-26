"""The K3 MLA layer (TP8, 8 tokens, 12 of the 96 heads per GPU) for the static compiler: the graph and its start plan
(the serving package generates its kernel from them). The math: reference.py (float64, rounded where SGLang rounds).

Graph (SGLang's MLA target-verify contract: the request's KV rows read from the framework's paged cache through its page table,
this step's 8 KV rows written at the framework's locations; the bank rows' number and strides at run time):
  x -> fused q_a / kv_a projection (gemm_tile [T, 2112], K parts in slots)
    -> mla_norm (the 2 RMSNorms: q_a [T, 1536] for q_b; the KV rows -> kv_new and the cache)
    -> mla_qb (q_b [T, 12 x 192] bf16: warp-MMA tiles, weight resident) -> mla_absorb (per head: [q_nope . w_kc | q_pe])
  x -> g_proj (gemm_tile [T, 1536], slots)
    -> mla_attn_part (contexts over thr rows; per group of HG heads x KV split: attention over the split's KV rows; the splits
       combined -> attn bf16)
    -> mla_attention (per head x KV split: shorter contexts' attention over the KV rows; split 0: (or attn) . w_vc,
       x bf16(sigmoid(gate)))
    -> o_proj (gemm_tile, warp MMAs polling its input, 64-row tiles, each tile's columns pushed to every GPU)
    -> attn_residual (aggregation 2: 2 parts per token on the 2 CTAs of a cluster; u = the new prefix out, RMSNorm).
build(input_dir=None): zero tensors of the shapes (code generation without a GPU). Options: num_sms (0: the device's; code
generation: 148), eps (0: 1e-5, K3's rms_norm_eps), thr (THR).
build(input_dir = an SGLang MLA dump directory, sglang_k3_mla dump mode: L<layer>_c0_r<g>.pt): the loop's harness run
(python/mirage/mpk/loop/run.py, part.yaml): every GPU's real weights and inputs of that layer, a synthetic KV context of `rows`
rows in pages of 64 in a shuffled pool (the same on every GPU), options layer (47) and rows
(1000); check(): the outputs vs reference.py. Shapes only (input_dir None): option
bank_rows (2) = the bank's allocated rows in the graph (16 = the harness's and SGLang's [T, 16, H], so the cost keys match the
harness's records; the kernel reads the bank by pointer with bank_info's strides, the code is the same).
"""
import os
import sys
import torch

T, H = 8, 7168
HEADS, NOPE, ROPE, KV, VD, QLORA = 12, 128, 64, 512, 128, 1536
NA = QLORA + KV + ROPE           # 2112: the fused q_a / kv_a projection
NQ = HEADS * (NOPE + ROPE)       # 2304: q_b per GPU
NG = HEADS * VD                  # 1536: g_proj, o_proj's K
D = KV + ROPE                    # 576: a KV row
EPS = 1e-5
SCALE = (NOPE + ROPE) ** -0.5
PS = 64                          # the framework's KV page size (SGLang page_size 64)
KV_ROWS = 4096                   # the shapes-only cache (the run binds the framework's buffer)
BLOCKS = 64                      # the shapes-only page table (the run binds the framework's)
FA_PARTS, G_PARTS = 4, 8      # K parts: fused_a 56 / 4, g_proj 56 / 8 K tiles per task
ATTN_SPLITS = 8                  # mla_attention's KV splits per head (split 0 combines; a short context runs in split 0 alone)
PART_SPLITS = 21                 # mla_attn_part's KV splits (each split's chunks read once for a group of heads; 21: 4000 / 8000
                                 # rows = 63 / 126 chunks = 21 x 3 / 6, no split with an extra chunk, 10-13)
HG = 2                           # heads per mla_attn_part task (2 x 8 tokens = 16 query rows: 32 accumulators per thread; with
                                 # 4 heads, 64, and the kernel spilled next to mla_attention's per-head tasks, 10-11)
THR = 0                          # contexts over THR rows: mla_attn_part; shorter: mla_attention's per-head tasks (option thr;
                                 # 10-13 one layer, tcgen05 scores: grouped 37.1-37.4 / 41.1-41.3 / 41.4-41.6 us at 37 / 200 /
                                 # 320 rows, per-head 37.8-38.0 / 42.2-42.6 / 41.0-41.1)

WEIGHTS = ["wa", "wq_norm", "wkv_norm", "wqb", "wkcT", "wvcT", "wg", "wo", "q", "gain", "ow", "layer_info"]
RUN = ["x", "bank", "prefix", "kv_cache", "loc", "page_table", "seq_k", "hidden", "new_prefix"]
DESCRIPTION = ("Kimi K3 MLA layer, TP8, 8 tokens per launch (fused q_a / kv_a projection, RMSNorms, q_b, absorbed attention over "
               "SGLang's paged KV cache, w_vc, output gate, o_proj, all-reduce, MLP-side attention residual over 1..16 bank rows; "
               "this step's KV rows written into the cache)")
CLUSTER = 2   # the launch's cluster size (attn_residual's 2 parts per token)


def serving_build():
    """The serving kernel's build (static_schedule.Build), the KDA kernel's (demo/kimi_k3_kda/part.py serving_build): resets in
    the kernel with two copies, the first GEMM stages prefilled before the PDL wait, the next kernel launched at this one's start;
    the CTAs whose first task is mla_absorb skip the PDL wait (its w_kc copy and q_b's K parts are this kernel's own)."""
    from mirage.mpk.static_schedule import Build, NodeRef
    return Build(defines=(("STATIC_RESET_IN_KERNEL", 1), ("STATIC_RESET_PARITY", 1), ("STATIC_PDL_PREFILL", 1),
                          ("STATIC_PDL_TRIGGER", 1), ("STATIC_NOWAIT_NODE", NodeRef("mla_absorb"))),
                 switches=frozenset({"task_times"}), host=False)


_RUN = {}   # the harness run's reference inputs (build -> check)


def _reference():
    """reference.py next to this file: the float64 reference and the kernel's weight layout"""
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    import reference
    return reference


def _real_tensors(input_dir: str, gpus: int, options: dict):
    """Every GPU's tensors of the harness run: the dump's weights and inputs, the synthetic KV context (check_mla.py's)."""
    kmla = _reference()
    layer, rows = int(options.get("layer") or 47), int(options.get("rows") or 1000)
    n = rows + T
    npages = (n + PS - 1) // PS
    pool = npages + 7
    g = torch.Generator().manual_seed(1234 + rows)
    cache0 = (torch.randn(pool * PS, D, generator=g) * 0.5).to(torch.bfloat16)
    perm = torch.randperm(pool, generator=g)[:npages].to(torch.int32)
    pos = torch.arange(n)
    krows = perm[pos // PS].long() * PS + pos % PS
    ws, dumps = [], []
    for r in range(gpus):
        d = torch.load(os.path.join(os.path.expanduser(input_dir), f"L{layer}_c0_r{r}.pt"), map_location="cpu", weights_only=False)
        wd = d["w"]
        dev = torch.device("cuda", r)
        w = {k: v.to(dev) for k, v in kmla.to_kernel_weights(wd["fused_qkv_a_proj_with_mqa"], wd["q_a_layernorm"], wd["kv_a_layernorm"],
                                                              wd["q_b_proj"], wd["w_kc"], wd["w_vc"], wd["g_proj"], wd["o_proj"],
                                                              wd["mlp_res_proj"], wd["mlp_res_norm"], wd["post_attention_layernorm"]).items()}
        bank = torch.zeros(T, 16, H, dtype=torch.bfloat16, device=dev)
        bank[:, :d["nvb"]] = d["bank"].to(dev)
        w.update(x=d["x"].to(dev).contiguous(), bank=bank,
                 prefix=(d["prefix"] if d["prefix"] is not None else torch.zeros_like(d["x"])).to(dev).contiguous(),
                 layer_info=kmla.layer_info(d["nvb"], bank.stride(0), bank.stride(1), dev), kv_cache=cache0.to(dev),
                 loc=krows[rows:].clone().to(dev), page_table=torch.cat([perm, torch.full((8,), -1, dtype=torch.int32)]).to(dev),
                 seq_k=torch.tensor([n], dtype=torch.int32, device=dev),
                 hidden=torch.zeros(T, H, dtype=torch.bfloat16, device=dev), new_prefix=torch.zeros(T, H, dtype=torch.bfloat16, device=dev))
        ws.append(w)
        dumps.append(d)
    torch.cuda.set_device(0)
    _RUN.update(dumps=dumps, cache0=cache0, krows=krows, rows=rows, layer=layer)
    return ws


def build(input_dir, gpus: int, tokens: int, options: dict):
    from mirage.core import bfloat16, float32
    from mirage.mpk import mla_layers as mla
    from mirage.mpk.static_megakernel import StaticMegakernel
    if tokens != T:
        raise ValueError(f"the MLA kernel runs {T} tokens, asked for {tokens}")
    bf, f32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
    z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt)
    if input_dir is None:   # shapes only (code generation)
        ws = [{"wa": z(NA, H), "wq_norm": z(QLORA), "wkv_norm": z(KV), "wqb": z(NQ, QLORA), "wkcT": z(HEADS * KV, NOPE),
               "wvcT": z(HEADS * VD, KV), "wg": z(NG, H), "wo": z(H, NG), "q": z(H), "gain": z(H), "ow": z(H),
               "x": z(T, H), "bank": z(T, int(options.get("bank_rows") or 2), H), "prefix": z(T, H),
               "layer_info": torch.tensor([2, 2 * H, H, 0, 0, 0, 0, 0], dtype=i32), "kv_cache": z(KV_ROWS, D), "loc": z(T, dt=i64),
               "page_table": z(BLOCKS, dt=i32), "seq_k": z(1, dt=i32), "hidden": z(T, H), "new_prefix": z(T, H)}]
    else:
        ws = _real_tensors(input_dir, gpus, options)
    w = ws[0]
    mpk = StaticMegakernel(num_gpus=gpus, num_sms=int(options.get("num_sms") or 0) or None)
    mpk.mla_plan = str(options.get("plan") or "")   # start_plan variant (tools/gen_kernels.py --mla-plan)
    mpk.max_bufs = 40
    t = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in w.items()}
    eps = float(options.get("eps") or EPS)
    latent = mpk.new_tensor(dims=(T, NA), dtype=float32, name="latent")
    mpk.gemm_tile_layer(input=t["x"], weight=t["wa"], output=latent, combine="slots")
    q_a = mpk.new_tensor(dims=(T, QLORA), dtype=bfloat16, name="q_a")
    kv_new = mla.norm_layer(mpk, latent, t["wq_norm"], t["wkv_norm"], q_a, t["kv_cache"], t["loc"], eps, eps)
    qp = mpk.new_tensor(dims=(T, NQ), dtype=bfloat16, name="q_b")   # (not "q": the residual weight's name)
    mla.qb_layer(mpk, q_a, t["wqb"], qp)
    gp = mpk.new_tensor(dims=(T, NG), dtype=float32, name="gate_parts")
    mpk.gemm_tile_layer(input=t["x"], weight=t["wg"], output=gp, combine="slots")
    q_full = mpk.new_tensor(dims=(T, HEADS * D), dtype=bfloat16, name="q_full")
    mla.absorb_layer(mpk, qp, t["wkcT"], q_full, heads=HEADS, nope=NOPE, rope=ROPE)
    thr = int(options.get("thr", THR))
    attn = mpk.new_tensor(dims=(T, HEADS * KV), dtype=bfloat16, name="attn")
    mla.attn_part_layer(mpk, q_full, kv_new, t["kv_cache"], t["page_table"], t["seq_k"], attn, heads=HEADS, scale=SCALE, thr=thr,
                        group=HG, splits=PART_SPLITS, page_size=PS, late_splits=1)   # the 21st split: SMs 40..45 (start_plan)
    o_in = mpk.new_tensor(dims=(T, NG), dtype=bfloat16, name="o_in")
    mla.attention_layer(mpk, q_full, gp, t["wvcT"], kv_new, t["kv_cache"], t["page_table"], t["seq_k"], attn, o_in, heads=HEADS,
                        scale=SCALE, thr=thr, page_size=PS, vd=VD, splits=ATTN_SPLITS)
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="oproj_sent")
    mpk.gemm_tile_layer(input=o_in, weight=t["wo"], output=sent, combine="push", warp_mma=True)
    mpk.attn_residual_layer(input=sent, bank=t["bank"], prefix=t["prefix"], q=t["q"], gain=t["gain"], ow=t["ow"],
                            output=t["hidden"], u_output=t["new_prefix"], bank_info=t["layer_info"], eps=eps)
    return ws, mpk


def check(h, ws, input_dir: str, out_dir: str) -> bool:
    """One launch; GPU 0's hidden and new prefix vs mla_reference.py (every GPU's o_proj partial from the reference on its own
    weights, all-reduced with the prefix, aggregation 2), the 8 KV rows written, the hidden bit-identical on all GPUs (check_mla.py's
    checks; limits in part.yaml)."""
    R = _reference()
    h.launch(536870912, timeout_s=120)
    dumps, cache0, krows, rows = _RUN["dumps"], _RUN["cache0"], _RUN["krows"], _RUN["rows"]
    ctx = cache0[krows[:rows]].double()
    cores = [R.core(d["x"], d["w"], ctx) for d in dumps]
    parts = [c["o"].to(torch.bfloat16) for c in cores]
    u = R.all_reduce_prefix(parts, dumps[0]["prefix"] if dumps[0]["prefix"] is not None else torch.zeros_like(dumps[0]["x"]))
    hid_ref = R.aggregation2(u, dumps[0]["bank"], dumps[0]["w"])
    r_h, r_p = R.rel(ws[0]["hidden"].cpu(), hid_ref), R.rel(ws[0]["new_prefix"].cpu(), u)
    r_kv = R.rel(ws[0]["kv_cache"][krows[rows:].to(ws[0]["kv_cache"].device)].cpu(), cores[0]["kv_new"])
    same = sum(torch.equal(ws[g]["hidden"].cpu().view(torch.int16), ws[0]["hidden"].cpu().view(torch.int16)) for g in range(1, len(ws)))
    ok = r_h < 2e-2 and r_p < 1e-2 and r_kv < 1e-2 and same == len(ws) - 1
    if not ok:   # which side is off: the kernel's hidden or the reference
        hk = ws[0]["hidden"].float().cpu()
        print(f"  hidden: kernel non-finite {int((~hk.isfinite()).sum())} (rows {sorted(set((~hk.isfinite()).nonzero()[:, 0].tolist()))}), "
              f"reference non-finite {int((~hid_ref.float().isfinite()).sum())}, kernel absmax {hk[hk.isfinite()].abs().max():.3g}", flush=True)
    print(f"  L{_RUN['layer']} {rows} rows (+8): hidden relRMS {r_h:.2e} | new prefix {r_p:.2e} | KV rows written {r_kv:.2e} | "
          f"hidden bit-identical on {same}/{len(ws) - 1} other GPUs -> {'ok' if ok else 'FAIL'}", flush=True)
    h.report()
    return ok


def watch(ws) -> dict:
    return {"hidden": ws[0]["hidden"]}


def start_plan(mpk, gpus: int, num_sms: int) -> dict:
    """First task on SMs 0..131 a fused_a task (its first stages' weights before the PDL wait on SMs < 100), on SMs 132..147 a
    g_proj task. Second: mla_qb tile i on SM i < 36, mla_norm token t on SM 36 + t, mla_absorb head h on SM 44 + h,
    mla_attention head h split 0 on SM 56 + h (its w_vc copied as it starts), g_proj on SMs 68..147; mla_attention's other
    splits on SMs 68..147 and 0..3, after the mla_attn_part tasks (group, split), split-major, on SMs 68..147 then 0..45, one
    per SM. Last: o_proj tile i (64 rows) on SMs 0..55 and 68..123;
    attn_residual token t's parts on SMs 124 + 2t, 125 + 2t (one cluster)."""
    from mirage.mpk.compiler import graph_nodes
    assert [n.name for n in graph_nodes(mpk)] == ["gemm_tile", "mla_norm", "mla_qb", "gemm_tile", "mla_absorb", "mla_attn_part",
                                                  "mla_attention", "gemm_tile", "attn_residual"]
    nodes = {n.name: n for n in graph_nodes(mpk) if n.name != "gemm_tile"}
    gemms = [n for n in graph_nodes(mpk) if n.name == "gemm_tile"]
    fa, gpj, op = gemms                      # fused_a, g_proj, o_proj (graph order)
    nm, qb, ab, apt, at, res = (nodes[k] for k in ("mla_norm", "mla_qb", "mla_absorb", "mla_attn_part", "mla_attention",
                                                     "attn_residual"))
    assert num_sms >= 148, "the start plan is laid out for 148 SMs"
    lists = [[] for _ in range(num_sms)]
    fa_grid, g_grid = (fa.grid[0], FA_PARTS, 1), (gpj.grid[0], G_PARTS, 1)
    i = 0
    for x in range(fa_grid[0]):
        for y in range(FA_PARTS):
            lists[i].append((fa.graph_idx, (x, y, 0))); i += 1
    assert i == 132, i
    plan = getattr(mpk, "mla_plan", "")
    if plan == "g_late":   # g_proj only on SMs 124..147 (4 each: its 22 MB off the fused_a window; the gate is read at the
        g_sms = [124 + i % 24 for i in range(96)]   # attention's end)
    else:
        g_sms = list(range(132, 148)) + list(range(68, 148))      # 16 first tasks, then 80 second ones
    gi = [(x, y) for x in range(g_grid[0]) for y in range(G_PARTS)]
    assert len(gi) == len(g_sms) == 96
    for (x, y), sm in zip(gi, g_sms):
        lists[sm].append((gpj.graph_idx, (x, y, 0)))
    assert qb.grid[0] == 36
    for x in range(qb.grid[0]):
        lists[x].append((qb.graph_idx, (x, 0, 0)))
    for t_ in range(T):
        lists[36 + t_].append((nm.graph_idx, (t_, 0, 0)))
    for h in range(HEADS):
        lists[44 + h].append((ab.graph_idx, (h, 0, 0)))
        lists[56 + h].append((at.graph_idx, (h, 0, 0)))   # (its w_vc copy and gate poll as it starts)
    # the attention tasks (group, split): after their SM's g_proj / mla_qb / mla_norm task, before its o_proj tile; split-major
    # (a short context uses the first splits); one per SM (a split waits for its group's other splits before the combine)
    parts = [(g_, s_) for s_ in range(PART_SPLITS) for g_ in range(HEADS // HG)]
    part_sms = list(range(68, 148)) + list(range(0, 46))   # (124..147: before the attention residual, idle until o_proj;
                                                           # 10-11 A/B 20 vs 16 splits: 61.3 vs 62.7 us at 8000 rows, 45.8 vs 45.3 at 1000)
    assert len(parts) <= len(part_sms), (len(parts), len(part_sms))
    for (g_, s_), sm in zip(parts, part_sms):
        lists[sm].append((apt.graph_idx, (g_, s_, 0)))
    # mla_attention's other KV splits (h, s > 0): SMs 68..147, then 0..3, after their earlier tasks (after the mla_attn_part task on
    # SMs 68..123 and 0..3: that one exits at once on a short context; the split reads attn as its declared input)
    others = [(h, s_) for s_ in range(1, ATTN_SPLITS) for h in range(HEADS)]
    split_sms = list(range(68, 148)) + list(range(0, 4))
    assert len(others) <= len(split_sms), (len(others), len(split_sms))
    for (h, s_), sm in zip(others, split_sms):
        lists[sm].append((at.graph_idx, (h, s_, 0)))
    o_sms = list(range(0, 56)) + list(range(68, 124))
    assert op.grid[0] == len(o_sms) == 112
    for r, sm in enumerate(o_sms):
        lists[sm].append((op.graph_idx, (r, 0, 0)))
    for t_ in range(res.grid[0]):
        for y_ in range(res.grid[1]):
            lists[124 + 2 * t_ + y_].append((res.graph_idx, (t_, y_, 0)))
    return {"grids": {fa.graph_idx: fa_grid, gpj.graph_idx: g_grid}, "lists": [lists for _ in range(gpus)]}
