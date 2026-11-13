"""Run one GLM-5.3 MoE layer with the static-schedule compiler.

The layer selects 8 of 256 routed experts and also runs one shared expert.
Expert weights use e4m3 values and fp32 scales for 128 x 128 blocks.
Expert activations use e4m3 values and fp32 scales for 128-value groups.
The router uses bf16 operands. This process controls all GPUs.
TP divides the expert intermediate dimension across GPUs.
With --gpus 1 --tp 8, one GPU computes one TP8 shard, not the full layer.
STATIC_TOKENS sets T, the rows of the step (8, 16 or 32; default 8). --shared-in-bank puts the shared expert in the
bank of the experts node (shared_bank: 7 layer calls).

  0. download_layer  the checkpoint (--checkpoint: a local snapshot, or a Hugging Face repo id such as zai-org/GLM-5.3: then
                     download_layer downloads only the shards of --layer) and check_config (weights.py)
  1. load_weights    the tensors of each GPU: layer --layer of the checkpoint; the expert banks in the piece layout of
                     scaled_moe_experts (weights.py, mirage.mpk.fp8_fp32scale_weights). x: --x (bf16 [T, 6144] raw file)
                     or random
  2. build_graph     the layer as 13 layer calls, as the K3 demo: router (gemm_tile), topk_route (x 2.5), group_quant (x); the
                     shared expert: scaled_gemm_tile (gate_up), silu_and_mul, group_quant (h), scaled_gemm_tile (down);
                     scaled_moe_experts; allreduce_send (shared), allreduce_send (routed); sum_gpus (routed), sum_gpus (shared), residual_add
  3. compile_plan    the plan (--plan: a plan file, default solver_plan.json) ->
                     schedules/schedule_gpu<g>.json. The two plan files are for one GPU (--gpus 1 --tp 8: one TP8 shard),
                     as measured on a B200:
                       hand_plan.json    router in 4 x 12 tasks (64 rows, 12 K parts), shared gate_up in 4 x 16 (16 K parts),
                                         the tasks dealt to the SMs in turn, in graph order (the best of the grids tried by hand)
                       solver_plan.json  search.search_plan on cost records of each task measured on a B200, started from
                                         the hand plan (the same grids; its own placement)
                     kernels/<plan>_layer.cu is the layer each one gives
                     --search COSTS: the plan is the solver's instead, starting from --plan (search.search_plan with the cost
                     record files COSTS; needs OR-Tools) -> <out>/plan.json, and <out>/plan_view.json with the solver's predicted
                     timeline (python plan_viewer.py <out>/plan_view.json)
  4. compile_static  build/layer.cu, built and loaded
  5. timed launches  --reps launches, 512 MB read on every GPU before each (empties L2); span per launch
                     --record FILE: the timing build; the per-task times become cost records (cost_records.py) appended to FILE
  6. --compare-y     y vs a reference y file (bf16 [T, 6144], e.g. SGLang's), relRMS
Steps 5-6 run the layer through the test harness (python/mirage/mpk/static_harness.py).

y = the MoE output (+ the residual: zero in this demo, so y is comparable with the MoE output of SGLang).

python demo.py --checkpoint zai-org/GLM-5.3 --gpus 1 --tp 8 [--rank 0] [--layer 3] [--plan PLAN] [--out DIR] [--reps 20]
               [--x x.bin] [--compare-y y.bin]                  (one GPU: shard 0 of TP8; y is that shard's part)
Other GPU counts or SM counts need a plan file of their own (--plan).
python demo.py ... --gpus 1 --tp 8 --plan hand_plan.json --record costs.json      (cost records of this GPU)
python demo.py ... --gpus 1 --tp 8 --plan hand_plan.json --search costs.json      (the solver's plan from these records)
"""
import argparse
import itertools
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
    from .weights import check_config, checkpoint_reader, download_layer, layer_tensors   # imported as demo.glm53_moe.demo
except ImportError:
    from weights import check_config, checkpoint_reader, download_layer, layer_tensors    # run as a script

T = int(os.environ.get("STATIC_TOKENS", "8"))   # rows per step (a token of each request): the kernel's T (8, 16 or 32)
H, I, E, TOPK, SCALE = 6144, 2048, 256, 8, 2.5   # hidden, expert intermediate, experts, top k, routed scaling


def load_weights(read, layer: int, gpus: int, x: torch.Tensor, num_experts: int = E, tp: int = None, rank: int = 0,
                 residual: torch.Tensor = None) -> list:
    """Load each GPU's expert shard and allocate its input and output tensors.

    By default, tp equals gpus and GPU g holds shard g.
    With one GPU and tp > 1, the GPU holds shard rank.
    Each GPU receives x, the router, the bias, and the residual.
    The residual has shape [T, H], uses bf16, and defaults to zero.
    """
    tp = tp or gpus
    assert tp == gpus or gpus == 1, "tp != gpus: one GPU only"
    ws, h = [], x.shape[1]
    for g in range(gpus):
        dev = torch.device("cuda", g)
        w = layer_tensors(read, layer, tp=tp, rank=g if tp == gpus else rank, num_experts=num_experts, device=dev)
        w.update(x=x.to(dev), routing_pairs=torch.empty(T, TOPK, dtype=torch.int64, device=dev),
                 routed_rows=torch.empty(T * TOPK, h, device=dev), moe_out=torch.empty(T, h, dtype=torch.bfloat16, device=dev),
                 residual=(torch.zeros(T, h, dtype=torch.bfloat16) if residual is None else residual).to(dev))
        ws.append(w)
    return ws


def shared_bank(w: dict) -> dict:
    """w with the shared expert in the expert bank, as demo/glm53_flash's MoE (topk_route_layer(shared_experts=1); the
    experts node runs it in its queue): w13 / w13_sf / w2 / w2_sf of E + 1 experts, the shared one last (at the TP shard's
    intermediate it has the routed experts' shapes); no sgu / sgu_sf / sd / sd_sf. The pairs and the routed rows (if w
    has them) get one more pair or row per token. build_graph then declares the layer without the shared expert's own
    nodes. An idle row (topk_route_live_rows) has no expert of its own: its routed pairs name the shared expert of the
    bank at weight 0, and its shared pair has weight 1."""
    w = dict(w)
    for k, s in (("w13", "sgu"), ("w13_sf", "sgu_sf"), ("w2", "sd"), ("w2_sf", "sd_sf")):
        assert w[s].numel() == w[k][0].numel(), f"shared_bank: {s} is not one expert of {k}"
        w[k] = torch.cat([w[k], w.pop(s).reshape(1, *w[k].shape[1:])])
    if "routing_pairs" in w:
        t, k = w["routing_pairs"].shape
        w["routing_pairs"] = w["routing_pairs"].new_empty(t, k + 1)
        w["routed_rows"] = w["routed_rows"].new_empty(t * (k + 1), w["routed_rows"].shape[1])
    return w


def round_robin_plan(mpk: StaticMegakernel, grids: dict) -> dict:
    """Return a round-robin plan. grids: {graph_idx: grid} for the nodes that it names; each other node keeps the grid
    of its layer call. The plan deals the tasks to the SMs in turn, in graph order. Task x of a one_task_per_sm node
    goes to SM x. Every GPU gets the same lists (as in demo/glm53_flash)."""
    per_sm, cursor = [[] for _ in range(mpk._num_sms())], 0
    for n in compiler.graph_nodes(mpk):
        for pos in itertools.product(*(range(d) for d in grids.get(n.graph_idx, n.grid))):
            if n.info["one_task_per_sm"]:
                per_sm[pos[0]].append((n.graph_idx, pos))
                continue
            per_sm[cursor % len(per_sm)].append((n.graph_idx, pos))
            cursor += 1
    return {"grids": {k: tuple(v) for k, v in grids.items()}, "lists": [per_sm] * mpk.num_gpus}


def build_kernel(gpus: int) -> StaticMegakernel:
    return StaticMegakernel(num_gpus=gpus)


def build_graph(mpk: StaticMegakernel, w: dict, next_norm=None, live=None) -> dict:
    """Declare the layer's calls: 13, or 7 with the shared expert in the bank (w from shared_bank: no sgu; the experts
    node runs it). The plan selects the grids and task placement. Returns the router's grid for a round-robin plan
    (round_robin_plan), {graph_idx: grid}. next_norm: (gamma, output, eps), torch tensors attached last ("norm_w",
    "x_next"): moe_out is the new residual and the kernel also writes RMSNorm(moe_out) gamma (row_rmsnorm_layer), the next
    sublayer's input. live: a torch tensor [T] int32 attached last ("live"): topk_route routes only the rows with live > 0
    (topk_route_live_rows; a serving batch's padded slots stream no expert of their own)."""
    E, H = w["router"].shape
    if w["w2"].shape[2] > 256:   # TP 4 (expert intermediate 512 on a GPU): the experts' counters need more than 16 lines
        mpk.max_counter_lines = 32
    if next_norm is not None:    # the norm's h32 scratch, gamma and output: more than config.cuh's 24 buffer slots
        mpk.max_bufs = max(getattr(mpk, "max_bufs", None) or 0, 32)
    t = {name: mpk.attach_input(torch_tensor=v, name=name) for name, v in w.items()}
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    logits, z_q = nt((T, E), float32, "router_logits"), nt((T, H), uint8, "x_fp8")
    mpk.gemm_tile_layer(input=t["x"], weight=t["router"], output=logits)                             # logits = x . Wg^T
    bank = "sgu" not in w   # the shared expert in the bank (shared_bank)
    if bank or T > 8:   # the experts node's per-slot counters: T K (or T (K + 1)) slots
        mpk.max_counter_lines = max(getattr(mpk, "max_counter_lines", None) or 0, 24)
    mpk.topk_route_layer(input=logits, bias=t["bias"], output=t["routing_pairs"],                    # sigmoid + bias, top 8,
                         routed_scaling_factor=SCALE, shared_experts=1 if bank else 0)               # renorm x 2.5 (+ shared)
    mpk.group_quant_layer(input=t["x"], output=z_q)                                                  # x in FP8, fp32 scales
    if bank:
        mpk.scaled_moe_experts_layer(z_q=z_q, pairs=t["routing_pairs"], w13=t["w13"], w13_scales=t["w13_sf"],
                                     w2=t["w2"], w2_scales=t["w2_sf"], output=t["routed_rows"])      # W13, SiLU, W2
        r_sent, Rsum = nt((T, H), bfloat16, "r_sent"), nt((T, H), bfloat16, "routed_summed")
        mpk.allreduce_send_layer(input=t["routed_rows"], output=r_sent)                              # a token's 9 rows added
        mpk.sum_gpus_layer(input=r_sent, output=Rsum)                                                # over the GPUs
        mpk.residual_add_layer(addend=Rsum, residual=t["residual"], output=t["moe_out"],             # y = R + res
                               counted=next_norm is not None)
    else:
        _shared_branch(mpk, t, z_q, nt, next_norm)
    if next_norm is not None:
        gamma, out = (mpk.attach_input(torch_tensor=v, name=n) for v, n in zip(next_norm[:2], ("norm_w", "x_next")))
        mpk.row_rmsnorm_layer(input=t["moe_out"], gamma=gamma, output=out, eps=next_norm[2])
    if live is not None:
        mpk.topk_route_live_rows(mpk.attach_input(torch_tensor=live, name="live"))
    router = next(n for n in compiler.graph_nodes(mpk) if n.name == "gemm_tile")
    return {router.graph_idx: (E // 64, 12, 1)}   # the router in 64-row tiles x 12 K parts, as hand_plan.json


def _shared_branch(mpk, t, z_q, nt, next_norm):
    """build_graph's calls after group_quant with the shared expert in its own nodes (as the K3 demo)."""
    H = t["router"].dim(1)
    SIR = t["sd"].dim(1)
    sgu, h_s = nt((T, 2 * SIR), int64, "shared_gate_up"), nt((T, SIR), bfloat16, "shared_act")
    h_q, S = nt((T, SIR), uint8, "shared_act_fp8"), nt((T, H), float32, "shared_down")
    mpk.scaled_gemm_tile_layer(input=z_q, weight=t["sgu"], weight_scales=t["sgu_sf"], output=sgu,
                               combine="add")                                                        # shared gate_up
    mpk.silu_and_mul_layer(input=sgu, output=h_s)                                                    # SiLU(gate) * up
    mpk.group_quant_layer(input=h_s, output=h_q)                                                     # h in FP8, fp32 scales
    mpk.scaled_gemm_tile_layer(input=h_q, weight=t["sd"], weight_scales=t["sd_sf"], output=S,
                               combine="store")                                                      # shared down: S
    mpk.scaled_moe_experts_layer(z_q=z_q, pairs=t["routing_pairs"], w13=t["w13"], w13_scales=t["w13_sf"],
                                 w2=t["w2"], w2_scales=t["w2_sf"], output=t["routed_rows"])          # W13, SiLU, W2
    # The hand plan keeps graph order on each SM.
    # Put the shared send after the experts so its wait does not delay them.
    # One send task per SM, as for R: the saved plans were measured with this grid.
    s_sent = nt((T, H), bfloat16, "s_sent")
    mpk.allreduce_send_layer(input=S, output=s_sent, num_tasks=mpk._num_sms())                       # S to every GPU
    r_sent = nt((T, H), bfloat16, "r_sent")
    mpk.allreduce_send_layer(input=t["routed_rows"], output=r_sent)                                  # a token's rows, sent
    Rsum, Ssum = nt((T, H), bfloat16, "routed_summed"), nt((T, H), bfloat16, "shared_summed")
    mpk.sum_gpus_layer(input=r_sent, output=Rsum)                                                    # R over the GPUs
    mpk.sum_gpus_layer(input=s_sent, output=Ssum)                                                    # S over the GPUs
    mpk.residual_add_layer(input=Rsum, addend=Ssum, residual=t["residual"], output=t["moe_out"],     # y = R + S + res
                           counted=next_norm is not None)


def check_plan(plan: str, gpus: int, sms: int) -> dict:
    """Read a plan file and check that it has task lists for `gpus` GPUs of `sms` SMs each. Raise ValueError if not."""
    p = compiler.plan_from_file(plan)
    shape = (len(p["lists"]), len(p["lists"][0]) if p["lists"] else 0)
    if shape != (gpus, sms):
        raise ValueError(f"{plan} has lists for {shape[0]} GPU(s) of {shape[1]} SMs; the layer runs on {gpus} GPU(s) of {sms} "
                         f"SMs. Give a plan file for this machine (--plan)")
    return p


def build(ws: list, out: str, plan: str, search: list = None, costs_code: str = "current", time_limit_s: float = 300,
          profile: bool = False, extra_flags: list = None):
    """Build the graph, compile its plan, and load the kernel for GPUs 0..len(ws)-1.

    ws contains the tensors from load_weights (or shared_bank's). plan: a saved plan file, or None: round_robin_plan.
    search: cost record files. Then the solver searches a plan, starting from the plan above (search.search_plan), and
    writes it to out/plan.json and its predicted timeline to out/plan_view.json. costs_code: the records of these kernel
    templates: "current" (static_schedule.code_tag), "any", or a tag.
    profile: the timing build (for cost records). extra_flags: more nvcc flags (e.g. -DSTATIC_RESET_IN_KERNEL).
    Write schedules to out/schedules and the compiled layer to out/build.
    Return the kernel and the information from compile_plan, with "paths" (the schedule files) and "mpk" (the graph).
    """
    gpus = len(ws)
    mpk = build_kernel(gpus)
    grids = build_graph(mpk, ws[0])
    p = check_plan(plan, gpus, torch.cuda.get_device_properties(0).multi_processor_count) if plan else \
        round_robin_plan(mpk, grids)
    if search:
        from mirage.mpk import plan_viewer, search as solver, static_schedule
        code = {"current": static_schedule.code_tag(), "any": None}.get(costs_code, costs_code)
        costs = solver.Costs(search, code=code, gpu=torch.cuda.get_device_name(0))
        p, summary = solver.search_plan(mpk, costs, time_limit_s=time_limit_s, start_from=p)
        os.makedirs(out, exist_ok=True)
        compiler.write_plan_file(os.path.join(out, "plan.json"), p)
        plan_viewer.save(os.path.join(out, "plan_view.json"), p, summary, f"searched from {plan or 'round robin'}")
    paths, info = compiler.compile_plan(mpk, p, os.path.join(out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(out, "build"), extra_flags=extra_flags,
                            profile=profile)
    return sk, dict(info, paths=paths, mpk=mpk)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="a local zai-org/GLM-5.3 snapshot, or the repo id (then only the shards of --layer are downloaded)")
    ap.add_argument("--layer", type=int, default=3, help="a MoE layer of the checkpoint (layers 0..2 are dense)")
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count(), help="GPUs; TP size = gpus unless --tp")
    ap.add_argument("--tp", type=int, help="the TP size with --gpus 1 (the kernel needs I / tp = 128, 256 or 512: tp = 4, 8 or 16)")
    ap.add_argument("--rank", type=int, default=0, help="with --gpus 1 --tp N: the shard on the GPU")
    ap.add_argument("--revision", help="with a Hugging Face repo id: a branch, tag or commit (default: main). The demo "
                                       "resolves it to a commit and downloads all files at that commit")
    ap.add_argument("--x", help="the input x, bf16 [T, 6144] raw file (default: random, seed 1)")
    ap.add_argument("--out", default="build/glm53_moe")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--compare-y", help="a reference y, bf16 [T, 6144] raw file")
    ap.add_argument("--plan", help="a plan file: solver_plan.json (default; found by search.search_plan) or hand_plan.json "
                         "(both for one GPU of 148 SMs, T = 8, of the layer with the shared expert's own nodes); with "
                         "--shared-in-bank or another T (STATIC_TOKENS) the default is round robin (round_robin_plan)")
    ap.add_argument("--shared-in-bank", action="store_true", help="the shared expert in the experts node's bank "
                    "(shared_bank), as demo/glm53_flash's MoE: 7 layer calls")
    ap.add_argument("--search", nargs="+", default=None, metavar="COSTS", help="cost record files: the solver's plan, from --plan")
    ap.add_argument("--costs-code", default="current", help="with --search: the records of these kernel templates: current "
                    "(static_schedule.code_tag), any, or a tag")
    ap.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    ap.add_argument("--record", default=None, metavar="FILE", help="the timing build; this run's cost records appended to FILE")
    ap.add_argument("--nvcc-flag", action="append", default=[], help="an nvcc flag for the layer (repeatable), e.g. "
                    "-DSTATIC_RESET_IN_KERNEL: the kernel resets its buffers itself, no host memsets or sync per launch")
    args = ap.parse_args()
    # check the GPU count, the TP size and the plan before the download and the weight load
    tp = args.tp or args.gpus
    if args.gpus < 1 or (tp != args.gpus and args.gpus != 1) or I % tp != 0 or I // tp not in (128, 256, 512):
        raise ValueError(f"--gpus {args.gpus} --tp {args.tp}: the kernel needs I / tp = 128, 256 or 512 (tp = 4, 8 or 16), "
                         f"and tp = gpus or one GPU")
    if args.plan is None and not args.shared_in_bank and T == 8:   # the plan files are of T = 8
        args.plan = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_plan.json")
    if args.plan:
        check_plan(args.plan, args.gpus, torch.cuda.get_device_properties(0).multi_processor_count)
    snapshot = download_layer(args.checkpoint, args.layer, revision=args.revision)
    check_config(snapshot, args.layer)
    if args.x:
        x = torch.from_numpy(np.fromfile(args.x, dtype=np.uint16).reshape(T, H).astype(np.int16)).view(torch.bfloat16)
    else:
        x = torch.randn(T, H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    ws = load_weights(checkpoint_reader(snapshot), args.layer, args.gpus, x, tp=args.tp, rank=args.rank)
    if args.shared_in_bank:
        ws = [shared_bank(w) for w in ws]
    sk, info = build(ws, args.out, args.plan, search=args.search,
                     costs_code=args.costs_code, time_limit_s=args.time, profile=args.record is not None,
                     extra_flags=args.nvcc_flag)
    h = Harness(sk)
    h.launch(timeout_s=60)
    y = ws[0]["moe_out"].cpu()
    for g in range(1, args.gpus):
        assert torch.equal(ws[g]["moe_out"].cpu(), y), f"y differs on GPU {g}"
    if args.compare_y:
        ref = torch.from_numpy(np.fromfile(args.compare_y, dtype=np.uint16).reshape(T, H).astype(np.int16)).view(torch.bfloat16).float()
        print(f"y vs {args.compare_y}: relRMS {((y.float() - ref).norm() / ref.norm()).item():.2e}")
    spans, launches = [], []   # launches (--record): per timed launch, the per-task times and the per-GPU span ends
    for _ in range(args.reps):
        h.launch(512 << 20, timeout_s=60)   # 512 MB read on every GPU first: L2 emptied
        spans.append(h.span_us())
        if args.record:
            launches.append((h.task_times(), h.timing()))
    print(f"{args.gpus} GPU(s), grids {info['grids']}: median span {statistics.median(spans):.1f} us over {args.reps} launches")
    if args.record:
        from mirage.mpk import cost_records
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
        records = cost_records.records_of_run(info["mpk"], info["paths"], launches, os.path.basename(os.path.normpath(args.out)),
                                              machine)
        cost_records.append_records(args.record, records)
        print(f"cost records: {len(records)} added to {args.record} (code {records[0]['code'] if records else '-'}, {machine})")
    sk.finalize()


if __name__ == "__main__":
    main()
