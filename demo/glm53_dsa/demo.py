"""Run GLM-5.3 / 5.2 DSA attention decode layers (MLA, the lightning indexer and IndexShare) with the static-schedule compiler.

The demo runs one decode step of T = 8 requests on one TP shard, with one kernel for each layer. With --tp 8, the shard has 8
of the 64 heads. The A projections and the indexer are replicated, as in vLLM. A "full" layer (HF config indexer_types) runs
its indexer. It writes its top-2048 positions for each request to a graph tensor. A "shared" layer reads that tensor (GLM
IndexShare). For example, --layers 6 7 runs layer 6, then runs layer 7 with the indices of layer 6.

  graph (full layer; a shared layer has no indexer nodes and reads the indices):
    group_quant(x) -> scaled_gemm_tile [q_a | kv_a]          -> dsa_q_norm_quant (RMSNorm + quant) -> scaled_gemm_tile q_b
                   \\                                         \\-> dsa_kv_write (the latent cache row)    \\-> scaled_gemm_tile wq_b
    gemm_tile x [wk | weights_proj] -> dsa_index_k_write (the index cache row)
    dsa_q_absorb (q_nope W_UK, RoPE q_pe);  dsa_index_score -> dsa_index_topk (the indices)
    sparse_mla (splits) -> dsa_mla_merge -> dsa_o_absorb (W_UV) -> group_quant(v) -> scaled_gemm_tile o_proj ->
    allreduce_send -> sum_gpus -> residual_add (y = attention output + residual, a graph tensor)

  0. weights        Download the attention tensors of --layers (weights.py) and run check_config.
  1. state          seq_lens (--seq-lens; 0: an idle slot), a shuffled block table (pages of 64) and the RoPE table. x: --x
                    (bf16 [8, 6144] raw file) or random. The residual is zero, so y is the attention output. For each
                    layer, random history in the latent cache (and the index cache): bf16 rows, e4m3 keys with UE8M0 scales
                    (seed 1).
  2. compile_plan   Select the plan of the layer kind (full or shared) with --plan: hand (default), solver, or PREFIX (the
                    files PREFIX_full.json and PREFIX_shared.json). The plan gives the schedules. The plan files are for one
                    GPU of 148 SMs, with --tp 8, --maxctx 16384, --chunks 16 and --splits 16:
                      hand_plan_<kind>.json    The grids qkva 21 x 12 (4 K tiles per task), [wk | weights_proj] 2 x 12,
                                               q_b NH x 2 by 4 and wq_b 32 x 4. The other nodes keep the default grid of
                                               their layer. The tasks go to the SMs in turn, in graph order.
                      solver_plan_<kind>.json  The output of search.search_plan, started from the hand plan (the same
                                               grids, a different placement). The cost records come from this demo on a
                                               B200, with three --seq-lens sets: the default set, all 2048, and all 16384.
                                               For the full layer, the solver plan is slower than the hand plan on a
                                               B200, by 2.6 to 5.9 us. This holds for all four --seq-lens sets that
                                               were measured: the default set, all 2048, all 8192 and all 16384. Thus
                                               the default is the hand plan.
                                               For the shared layer, the solver found the hand plan optimal: the two
                                               plans have the same task lists.
                    kernels/<plan>_<kind>_layer.cu is the layer each one gives.
                    --search COSTS: the plan is the solver's instead, starting from --plan (search.search_plan with the cost
                    record files COSTS; needs OR-Tools) -> <out>/layer<L>/plan.json, and plan_view.json with the solver's
                    predicted timeline (python plan_viewer.py <out>/layer<L>/plan_view.json)
  3. compile_static <out>/layer<L>/build/layer.cu, built and loaded
  4. y              The first launch (not timed) gives the output y of layer L (the graph tensor attn_out, bf16 [8, 6144]).
                    --save-y DIR writes it to DIR/y_layer<L>.bin. --compare-y DIR compares it with DIR/y_layer<L>.bin
                    (relRMS).
  5. timed launches --reps launches, 512 MB read before each (empties L2); span per launch
                    --record FILE: the timing build; the per-task times become cost records (cost_records.py) appended to
                    FILE
Steps 4-5 run the layer through the test harness (python/mirage/mpk/static_harness.py).

python demo.py --checkpoint zai-org/GLM-5.3 [--layers 6 7] [--tp 8 --rank 0] [--seq-lens ...] [--plan PLAN] [--reps 20]
python demo.py ... --plan hand --record costs.json      (cost records of this GPU)
python demo.py ... --plan hand --search costs.json      (the solver's plans from these records)
"""
import argparse
import os
import statistics
import subprocess

import numpy as np
import torch
from mirage.core import bfloat16, float32, int64, uint8
from mirage.mpk import compiler
from mirage.mpk.static_harness import Harness
from mirage.mpk.static_megakernel import StaticMegakernel

try:
    from .weights import check_config, checkpoint_reader, download_layers, is_full_layer, layer_tensors
except ImportError:
    from weights import check_config, checkpoint_reader, download_layers, is_full_layer, layer_tensors

T, H, PAGE, TOPK = 8, 6144, 64, 2048
HERE = os.path.dirname(os.path.abspath(__file__))


def rope_table(positions: int, theta: float) -> torch.Tensor:
    """Return the vLLM cos_sin_cache for rotary dim 64: [positions, 64] bf16, cos then sin. theta: config rope_theta."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], -1).to(torch.bfloat16)


def make_state(seq_lens, maxctx, theta, seed=0) -> dict:
    """Return seq_lens, a shuffled block table [T, maxctx / PAGE] and the RoPE table (CPU tensors)."""
    g = torch.Generator().manual_seed(seed)
    maxp = maxctx // PAGE
    pages = torch.randperm(T * maxp, generator=g).view(T, maxp).int()
    return {"seq_lens": torch.tensor(seq_lens, dtype=torch.int32), "block_table": pages,
            "cos_sin": rope_table(maxctx, theta)}


def random_history(cache: torch.Tensor, index_cache: torch.Tensor = None, seed: int = 1):
    """Fill the caches with random history: latent rows N(0, 1) in bf16; index keys e4m3(N(0, 1) x 64) with the UE8M0 scale
    2^-6 (byte 121, 4 copies; the dsa_index_k_write page layout). The kernel writes the rows of this step."""
    g = torch.Generator(device=cache.device).manual_seed(seed)
    cache.copy_(torch.randn(cache.shape, generator=g, device=cache.device).to(torch.bfloat16))
    if index_cache is not None:
        ic = index_cache.view(-1, PAGE * 132)
        keys = (torch.randn(ic.shape[0], PAGE * 128, generator=g, device=cache.device) * 64).to(torch.float8_e4m3fn)
        ic[:, :PAGE * 128] = keys.view(torch.uint8)
        ic[:, PAGE * 128:] = 121


def kernel_tensors(w, st, bufs) -> dict:
    """Return the graph inputs by name. input_norm is not a graph input, so the function leaves it out."""
    return {k: v for k, v in {**w, **st, **bufs}.items() if k != "input_norm"}


def build_graph(mpk, w, st, bufs, full, cfg, maxctx, splits, chunks, eps=1e-5):
    """Declare the nodes of the layer. bufs: the graph tensors (x, the caches, indices, residual, attn_out) by name."""
    t = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in kernel_tensors(w, st, bufs).items()}
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    NH = w["w_uk"].shape[0]
    xq, qkva = nt((T, H), uint8, "x_fp8"), nt((T, 2688), int64, "qkva")
    mpk.group_quant_layer(input=t["x"], output=xq)
    mpk.scaled_gemm_tile_layer(input=xq, weight=t["w_qkva"], weight_scales=t["w_qkva_sf"], output=qkva, combine="add")
    if full:
        kw = nt((T, 256), int64, "idx_kw")
        mpk.gemm_tile_layer(input=t["x"], weight=t["w_kw"], output=kw, combine="add")
    qcq = nt((T, 2048), uint8, "q_c_fp8")   # q_c quantized (the node also applies q_a_layernorm)
    mpk.dsa_q_norm_quant_layer(qkva=qkva, gamma=t["q_a_norm"], output=qcq, eps=eps)
    mpk.dsa_kv_write_layer(qkva=qkva, gamma=t["kv_a_norm"], cos_sin=t["cos_sin"], seq_lens=t["seq_lens"],
                           block_table=t["block_table"], cache=t["cache"], kv0=cfg["q_lora_rank"], eps=eps)
    if full:
        mpk.dsa_index_k_write_layer(kw=kw, k_norm_weight=t["k_norm_w"], k_norm_bias=t["k_norm_b"], cos_sin=t["cos_sin"],
                                    seq_lens=t["seq_lens"], block_table=t["block_table"], index_cache=t["index_cache"])
    qb = nt((T, NH * 256), int64, "q_b")
    mpk.scaled_gemm_tile_layer(input=qcq, weight=t["w_qb"], weight_scales=t["w_qb_sf"], output=qb, combine="add")
    if full:
        iq = nt((T, 4096), int64, "idx_q")
        mpk.scaled_gemm_tile_layer(input=qcq, weight=t["w_iqb"], weight_scales=t["w_iqb_sf"], output=iq, combine="add")
    q = nt((T, NH, 576), bfloat16, "q_mla")
    mpk.dsa_q_absorb_layer(qb=qb, w_uk=t["w_uk"], cos_sin=t["cos_sin"], seq_lens=t["seq_lens"], output=q)
    if full:
        scores = nt((T, maxctx), float32, "idx_scores")
        mpk.dsa_index_score_layer(iq=iq, kw=kw, index_cache=t["index_cache"], cos_sin=t["cos_sin"], seq_lens=t["seq_lens"],
                                  block_table=t["block_table"], output=scores, chunks=chunks)
        mpk.dsa_index_topk_layer(scores=scores, seq_lens=t["seq_lens"], output=t["indices"])
    # mla_split names the sparse_mla buffer: o [T, S, NH, 512] fp32, then lse [T, S, NH] (not the declared shape)
    split = nt((T, NH, 512), float32, "mla_split")
    mpk.sparse_mla_layer(q=q, cache=t["cache"], indices=t["indices"], block_table=t["block_table"], seq_lens=t["seq_lens"],
                         output=split, softmax_scale=(cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]) ** -0.5,
                         splits=splits)
    o_lat, v, vq = nt((T, NH, 512), bfloat16, "o_lat"), nt((T, NH * 256), bfloat16, "v"), nt((T, NH * 256), uint8, "v_fp8")
    mpk.dsa_mla_merge_layer(input=split, output=o_lat)
    mpk.dsa_o_absorb_layer(input=o_lat, w_uv=t["w_uv"], output=v)
    mpk.group_quant_layer(input=v, output=vq)
    o, sent, osum = nt((T, H), float32, "o_proj"), nt((T, H), bfloat16, "o_sent"), nt((T, H), bfloat16, "o_sum")
    mpk.scaled_gemm_tile_layer(input=vq, weight=t["w_o"], weight_scales=t["w_o_sf"], output=o, combine="store")
    mpk.allreduce_send_layer(input=o, output=sent)
    mpk.sum_gpus_layer(input=sent, output=osum)
    mpk.residual_add_layer(addend=osum, residual=t["residual"], output=t["attn_out"])   # y = attention output + residual


def plan_file(plan: str, full: bool) -> str:
    """Return the plan file of a layer kind: plan = "hand", "solver" or a path prefix."""
    kind = "full" if full else "shared"
    prefix = os.path.join(HERE, f"{plan}_plan") if plan in ("solver", "hand") else plan
    return f"{prefix}_{kind}.json"


def check_plan(path: str, sms: int) -> dict:
    """Read a plan file and check that it has task lists for one GPU of `sms` SMs. Raise ValueError if not."""
    p = compiler.plan_from_file(path)
    shape = (len(p["lists"]), len(p["lists"][0]) if p["lists"] else 0)
    if shape != (1, sms):
        raise ValueError(f"{path} has lists for {shape[0]} GPU(s) of {shape[1]} SMs; the layer runs on 1 GPU of {sms} SMs. "
                         f"Give a plan file for this machine (--plan)")
    return p


def build(mpk, ws: dict, plan: str, out: str, search: list = None, costs_code: str = "current", time_limit_s: float = 300,
          profile: bool = False, extra_flags: list = None):
    """Compile the plan of the graph in mpk and load the kernel. Return the kernel and the compile_plan information, with
    "paths" (the schedule files).
    plan: a plan file.
    search: cost record files. If given, the solver searches a plan that starts from `plan` (search.search_plan). It writes
    the plan to out/plan.json and its predicted timeline to out/plan_view.json.
    costs_code: the kernel templates of the records to use: "current" (static_schedule.code_tag), "any", or a tag.
    profile: if True, build the timing build (for cost records). extra_flags: more nvcc flags (e.g. -DSTATIC_RESET_IN_KERNEL).
    The function writes the schedules to out/schedules and the compiled layer to out/build."""
    p = check_plan(plan, torch.cuda.get_device_properties(0).multi_processor_count)
    os.makedirs(out, exist_ok=True)
    if search:
        from mirage.mpk import plan_viewer, search as solver, static_schedule
        code = {"current": static_schedule.code_tag(), "any": None}.get(costs_code, costs_code)
        costs = solver.Costs(search, code=code, gpu=torch.cuda.get_device_name(0))
        p, summary = solver.search_plan(mpk, costs, time_limit_s=time_limit_s, start_from=p)
        compiler.write_plan_file(os.path.join(out, "plan.json"), p)
        plan_viewer.save(os.path.join(out, "plan_view.json"), p, summary, f"searched from {plan}")
    paths, info = compiler.compile_plan(mpk, p, os.path.join(out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[ws], out_dir=os.path.join(out, "build"), extra_flags=extra_flags,
                            profile=profile)
    return sk, dict(info, paths=paths)


def add_common_args(ap: argparse.ArgumentParser):
    """Add the arguments that other scripts share with the demo (for example a test): the checkpoint, the layers, the shapes,
    the plan."""
    ap.add_argument("--checkpoint", required=True, help="a HF model id or a local directory")
    ap.add_argument("--revision", help="with a HF model id: a branch, tag or commit (default: main)")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 7],
                    help="layers in order; a shared layer reads the indices of the last full layer before it in this list")
    ap.add_argument("--tp", type=int, default=8, help="the TP size (the heads on this GPU: 64 / tp)")
    ap.add_argument("--rank", type=int, default=0, help="the TP shard on this GPU")
    ap.add_argument("--maxctx", type=int, default=16384,
                    help="the most positions a request may have (a multiple of 1024 and of 64 x chunks)")
    ap.add_argument("--splits", type=int, default=16, help="sparse_mla tasks per request")
    ap.add_argument("--chunks", type=int, default=16, help="dsa_index_score tasks per request")
    ap.add_argument("--plan", default="hand", help="hand (default), solver, or a path prefix of PREFIX_full.json and "
                    "PREFIX_shared.json (the hand and solver plans: one GPU of 148 SMs)")
    ap.add_argument("--nvcc-flag", action="append", default=[], help="an nvcc flag for the layer (repeatable), e.g. "
                    "-DSTATIC_RESET_IN_KERNEL. With this flag, the kernel resets its buffers. The host does no memset or "
                    "sync for each launch")


def setup(args, dev):
    """Download and check the checkpoint, check the plans, and make the state. Return (cfg, the checkpoint reader, the state
    on dev, the full-layer flag of each layer, the indices tensor [T, TOPK] int32)."""
    assert max(args.seq_lens) <= args.maxctx and min(args.seq_lens) >= 0
    snapshot = download_layers(args.checkpoint, args.layers, revision=args.revision)
    cfg = check_config(snapshot)
    fulls = [is_full_layer(cfg, layer) for layer in args.layers]
    if not fulls[0]:
        raise ValueError(f"layer {args.layers[0]} is a shared layer: list a full layer before it (--layers)")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for full in set(fulls):   # check the plans before the weight load
        check_plan(plan_file(args.plan, full), sms)
    st = {k: v.to(dev) for k, v in make_state(args.seq_lens, args.maxctx, cfg["rope_parameters"]["rope_theta"]).items()}
    indices = torch.full((T, TOPK), -1, dtype=torch.int32, device=dev)
    return cfg, checkpoint_reader(snapshot), st, fulls, indices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--seq-lens", type=int, nargs=8, default=[1, 300, 2048, 2049, 4000, 8192, 12000, 16384],
                    help="the context length of each request, with the token of this step (0: an idle slot)")
    ap.add_argument("--x", help="the input x (the input_layernorm output), bf16 [8, 6144] raw file (default: random, seed 2)")
    ap.add_argument("--search", nargs="+", default=None, metavar="COSTS",
                    help="cost record files. Use the solver plan, started from --plan")
    ap.add_argument("--costs-code", default="current", help="with --search: the records of these kernel templates: current "
                    "(static_schedule.code_tag), any, or a tag")
    ap.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    ap.add_argument("--record", default=None, metavar="FILE",
                    help="use the timing build. Append the cost records of this run to FILE")
    ap.add_argument("--save-y", metavar="DIR", help="write y of each layer to DIR/y_layer<L>.bin")
    ap.add_argument("--compare-y", metavar="DIR", help="compare y of each layer with DIR/y_layer<L>.bin (relRMS)")
    ap.add_argument("--out", default="build/glm53_dsa", help="the directory for the schedules and the builds")
    ap.add_argument("--reps", type=int, default=20, help="the timed launches of each layer")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    cfg, read, st, fulls, indices = setup(args, dev)
    if args.x:
        x = torch.from_numpy(np.fromfile(args.x, dtype=np.uint16).reshape(T, H)).view(torch.bfloat16)
    else:
        x = torch.randn(T, H, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    x = x.to(dev)
    npages = T * (args.maxctx // PAGE)
    residual = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)   # y = the attention output
    launches_of = {}   # --record: per layer, the per-task times and the span ends of each timed launch
    for layer, full in zip(args.layers, fulls):
        w = layer_tensors(read, layer, tp=args.tp, rank=args.rank, device=dev, full=full)
        cache = torch.empty(npages * PAGE, 576, dtype=torch.bfloat16, device=dev)
        index_cache = torch.empty(npages, PAGE * 132, dtype=torch.uint8, device=dev) if full else None
        random_history(cache, index_cache)
        bufs = {"x": x, "cache": cache, "indices": indices, "residual": residual,
                "attn_out": torch.zeros(T, H, dtype=torch.bfloat16, device=dev)}
        if full:
            bufs["index_cache"] = index_cache
        mpk = StaticMegakernel(num_gpus=1)
        build_graph(mpk, w, st, bufs, full, cfg, args.maxctx, args.splits, args.chunks, cfg["rms_norm_eps"])
        out = os.path.join(args.out, f"layer{layer}")
        sk, info = build(mpk, kernel_tensors(w, st, bufs), plan_file(args.plan, full), out, search=args.search,
                         costs_code=args.costs_code, time_limit_s=args.time, profile=args.record is not None,
                         extra_flags=args.nvcc_flag)
        h = Harness(sk)
        h.launch(timeout_s=60)
        torch.cuda.synchronize()
        y = bufs["attn_out"].cpu()
        if args.save_y:
            os.makedirs(args.save_y, exist_ok=True)
            y.view(torch.int16).numpy().tofile(os.path.join(args.save_y, f"y_layer{layer}.bin"))
        if args.compare_y:
            f = os.path.join(args.compare_y, f"y_layer{layer}.bin")
            ref = torch.from_numpy(np.fromfile(f, dtype=np.uint16).reshape(T, H)).view(torch.bfloat16).float()
            print(f"layer {layer}: y vs {f}: relRMS {((y.float() - ref).norm() / ref.norm()).item():.2e}")
        spans, launches = [], []
        for _ in range(args.reps):
            h.launch(512 << 20, timeout_s=60)   # 512 MB read first: L2 emptied
            spans.append(h.span_us())
            if args.record:
                launches.append((h.task_times(), h.timing()))
        print(f"layer {layer} ({'full' if full else 'shared'}), grids {info['grids']}: median span "
              f"{statistics.median(spans):.1f} us over {args.reps} launches")
        if args.record:
            launches_of[layer] = (mpk, info["paths"], launches)
        sk.finalize()
    if args.record:
        from mirage.mpk import cost_records
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
        for layer, (mpk, paths, launches) in launches_of.items():
            records = cost_records.records_of_run(mpk, paths, launches,
                                                  f"{os.path.basename(os.path.normpath(args.out))}_layer{layer}", machine)
            cost_records.append_records(args.record, records)
            print(f"layer {layer} cost records: {len(records)} added to {args.record} "
                  f"(code {records[0]['code'] if records else '-'}, {machine})")


if __name__ == "__main__":
    main()
