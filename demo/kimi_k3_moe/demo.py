"""Kimi K3 MoE layer, TP8, on the static-schedule compiler, all 8 GPUs from this process:

  1. load_weights    the real layer weights of every GPU (files written from the K3 checkpoint; the file names are in load_weights)
  2. build_graph     the layer as 15 layer calls (no grid, no K split yet)
  3. compile_plan    the plan (a plan file; default solver_plan.json) -> the graph regridded to the plan's grids,
                     schedules/schedule_gpu<g>.json                          (python/mirage/mpk/compiler.py)
  4. compile_static  build/layer.cu from the schedules, built and loaded     (python/mirage/mpk/static_schedule.py)
  5. check_launch    one launch; y vs SGLang's y, y the same on all GPUs, each GPU's partial sums vs their fp32 references
  6. timed_launches  --reps launches, 512 MB read on every GPU before each (empties L2); span per launch

python demo.py --layer-dir ~/mk/moe_layer [--plan PLAN_FILE] [--out DIR] [--reps 20] [--compare-y y_rank0.bin]
Weight files: <layer_dir>/ holds the tensors every GPU has (router, latent_down, latent_up, gamma, bias, the 8-token input x and the
residual prefix, SGLang's y), <layer_dir>_r<g>/ GPU g's shard (shared gate_up / down, expert banks).
"""
import argparse, os, socket, statistics, sys, json
from collections import Counter
import numpy as np
import torch
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk import compiler
from mirage.core import bfloat16, float32, int64, uint8
from mirage.utils import get_configurations_from_gpu

T, H, L, E, TOPK = 8, 7168, 3584, 896, 16       # tokens, hidden, latent, experts, routed experts per token
IR_LOCAL, SHR_LOCAL = 384, 768                   # routed / shared intermediate per GPU at TP8


# the layer's input tensors: (name, file, shape, file dtype, from GPU g's shard). A file holds the raw values; bf16 files are read as
# uint16 and viewed as bf16. The expert banks are in the kernel's packed tile order.
WEIGHT_FILES = [
    ("router_weight", "w_gate_bf16.bin", (E, H), "bf16", False),
    ("score_correction_bias", "gate_bias_f32.bin", (E,), np.float32, False),
    ("latent_down_weight", "w_down_bf16.bin", (L, H), "bf16", False),
    ("latent_up_weight", "w_up_bf16.bin", (H, L), "bf16", False),
    ("routed_expert_norm_weight", "gamma_bf16.bin", (L,), "bf16", False),
    ("moe_in", "x_bf16.bin", (T, H), "bf16", False),
    ("moe_prefix", "prefix_bf16.bin", (T, H), "bf16", False),
    ("shared_gate_up_weight", "w_sgu_bf16.bin", (2 * SHR_LOCAL, H), "bf16", True),
    ("shared_down_weight", "w_sd_bf16.bin", (H, SHR_LOCAL), "bf16", True),
    ("w13_blocks", "w13_blk.bin", (E, 2 * IR_LOCAL, L // 2), np.uint8, True),
    ("w13_scales", "w13_sf.bin", (E, 2 * IR_LOCAL, L // 32), np.uint8, True),
    ("w2_blocks", "w2_blk.bin", (E, L, IR_LOCAL // 2), np.uint8, True),
    ("w2_scales", "w2_sf.bin", (E, L, IR_LOCAL // 32), np.uint8, True),
]


def read_file(path: str, shape: tuple, dtype) -> torch.Tensor:
    """A tensor from a raw file; dtype "bf16": uint16 values viewed as bf16."""
    if dtype == "bf16":
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16).reshape(shape)).view(torch.bfloat16)
    return torch.from_numpy(np.fromfile(path, dtype=dtype).reshape(shape))


def load_weights(layer_dir: str, gpu: int) -> dict:
    """GPU `gpu`'s tensors on device `gpu`: WEIGHT_FILES (the shared ones from layer_dir, GPU g's shard from layer_dir_r<g>, GPU 0's
    from layer_dir), and the layer's outputs the checks read back."""
    shard_dir = f"{layer_dir}_r{gpu}" if gpu else layer_dir
    dev = torch.device("cuda", gpu)
    w = {}
    for name, file, shape, dtype, sharded in WEIGHT_FILES:
        w[name] = read_file(os.path.join(shard_dir if sharded else layer_dir, file), shape, dtype).contiguous().to(dev)
    # the outputs: y; this GPU's routed rows (one per token and routed expert: [T x TOPK][L], moe_experts' output; routed_partial adds
    # a token's TOPK); this GPU's shared-down sums before the all-reduce
    w["moe_out"] = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    w["routed_sum_partial"] = torch.zeros(T * TOPK, L, dtype=torch.float32, device=dev)
    w["shared_down_partial"] = torch.zeros(T, H, dtype=torch.float32, device=dev)
    return w


def routed_partial(w: dict) -> torch.Tensor:
    """This GPU's routed partial sum [T][L] (fp32, CPU): its TOPK rows per token added in order, as allreduce_send adds them."""
    return w["routed_sum_partial"].cpu().view(T, TOPK, L).sum(1)


def build_kernel(num_gpus: int = 8) -> StaticMegakernel:
    """The StaticMegakernel for the layer divided over num_gpus GPUs (the weight files are TP8 shards)."""
    params = StaticMegakernel.get_default_init_parameters()
    qo_indptr = torch.zeros(T + 1, dtype=torch.int32, device="cuda")
    qo_indptr[T] = T
    num_workers, num_schedulers = get_configurations_from_gpu(0)
    params.update(mode="offline", test_mode=True, world_size=1, mpi_rank=0, num_workers=num_workers, num_local_schedulers=num_schedulers,
                  max_num_batched_tokens=T, max_num_batched_requests=T, meta_tensors={"qo_indptr_buffer": qo_indptr})
    return StaticMegakernel(num_gpus=num_gpus, **params)


def build_graph(mpk: StaticMegakernel, w: dict, rmsnorm_recompute: bool = False):
    """The K3 MoE layer as 15 layer calls. No grid_dim, no K split: the plan decides them. Shared down is its own GEMM node and S is
    sent on its own (as soon as shared down is done), R after moe_experts. sum_rmsnorm: 1, 2 or 4 tasks per token (the plan),
    swapping their sums of squares (in a cluster of 2 CTAs when compile_plan can pair them), or each adding the whole row's
    (rmsnorm_recompute)."""
    a = lambda k: mpk.attach_input(torch_tensor=w[k], name=k)
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name, io_category="cuda_tensor")
    x, prefix = a("moe_in"), a("moe_prefix")
    w_router, bias = a("router_weight"), a("score_correction_bias")
    w_down, w_up, gamma = a("latent_down_weight"), a("latent_up_weight"), a("routed_expert_norm_weight")
    w_sgu, w_sd = a("shared_gate_up_weight"), a("shared_down_weight")
    w13, w13_sf, w2, w2_sf = a("w13_blocks"), a("w13_scales"), a("w2_blocks"), a("w2_scales")
    G = nt((4096,), uint8, "globals")   # the layer state G (filled by static_megakernel/host.cuh)
    M = nt((4096,), uint8, "maps")      # the tensor maps (filled by host.cuh)
    logits = nt((T, E), float32, "router_logits"); pairs = nt((T, TOPK), int64, "routing_pairs")
    z = nt((T, L), float32, "latent_z"); z_q = nt((T, L), uint8, "latent_z_mxfp8")
    sgu = nt((T, 2 * SHR_LOCAL), float32, "shared_gate_up"); h_s = nt((T, SHR_LOCAL), bfloat16, "shared_act")
    Rn = nt((T, L), bfloat16, "routed_normed"); Ssum = nt((T, H), bfloat16, "shared_summed")
    up_part = nt((T, H), float32, "latent_up_partial"); o = nt((T, H), bfloat16, "latent_up_out")   # each GPU: its own rows
    R, S, y = a("routed_sum_partial"), a("shared_down_partial"), a("moe_out")
    mpk.gemm_tile_layer(input=x, weight=w_router, globals=G, maps=M, output=logits)                       # router: logits = x . Wg^T (fp32)
    mpk.topk_route_layer(input=logits, bias=bias, globals=G, output=pairs)                                     # sigmoid + bias, top-16, renorm
    mpk.gemm_tile_layer(input=x, weight=w_down, globals=G, maps=M, output=z, rows_split_over_gpus=True)   # latent_down: z = x . Wdown^T
    mpk.sum_quant_send_layer(input=z, globals=G, output=z_q)                                                       # MXFP8 quantize + all-gather
    mpk.gemm_tile_layer(input=x, weight=w_sgu, globals=G, maps=M, output=sgu, combine="add")              # shared gate_up
    mpk.situ_and_mul_layer(input=sgu, globals=G, output=h_s)                                                      # SiTU
    r_sent, s_sent = nt((T, L), float32, "r_sent"), nt((T, H), float32, "s_sent")
    mpk.gemm_tile_layer(input=h_s, weight=w_sd, globals=G, maps=M, output=S, combine="store")          # shared down: S = h_s . Wsd^T
    mpk.allreduce_send_layer(input=S, globals=G, output=s_sent, num_tasks=56)                            # S to every GPU, early
    mpk.moe_experts_layer(z_q=z_q, pairs=pairs, h_s=h_s, w13=w13, w2=w2, globals=G, maps=M, output=(R,),  # W13 + SiTU + requant, W2
                           w13_scales=w13_sf, w2_scales=w2_sf)
    mpk.allreduce_send_layer(input=R, globals=G, output=r_sent)                                          # R to every GPU
    mpk.sum_rmsnorm_layer(input=r_sent, gamma=gamma, globals=G, output=Rn, recompute=rmsnorm_recompute)    # R over the GPUs, RMSNorm
    mpk.sum_gpus_layer(input=s_sent, globals=G, output=Ssum)                                              # S over the GPUs
    mpk.gemm_tile_layer(input=Rn, weight=w_up, globals=G, maps=M, output=up_part, rows_split_over_gpus=True,
                        input_polled=True)                                                                   # latent_up (K parts)
    mpk.sum_send_layer(input=up_part, globals=G, output=o)                                               # add the K parts, to every GPU
    mpk.residual_add_layer(input=o, addend=Ssum, residual=prefix, globals=G, output=y)                         # y = o + S + prefix
    return y


def print_nodes(mpk, title: str) -> None:
    print(title)
    for n in compiler.graph_nodes(mpk):
        ins = [compiler.tname(mpk, t) for t in n.inputs]
        outs = [compiler.tname(mpk, t) for t in n.outputs]
        print(f"  node {n.graph_idx:2d} {n.name:<16} grid {n.grid} params {n.params}   in {ins} -> out {outs}")


def relrms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def build(layer_dir: str, ngpu: int, rmsnorm_recompute: bool = False):
    """Steps 1-2: (every GPU's tensors, the StaticMegakernel holding the graph)."""
    ws = [load_weights(layer_dir, g) for g in range(ngpu)]
    torch.cuda.set_device(0)
    mpk = build_kernel(ngpu)
    build_graph(mpk, ws[0], rmsnorm_recompute)
    return ws, mpk


def launch_and_wait(sk, wait_s: float, what: str) -> None:
    """One launch (512 MB read on every GPU first: L2 emptied); a launch that does not finish: its counters, then exit."""
    sk.launch(l2_flush_bytes=512 << 20)
    try:
        sk.wait(timeout_s=wait_s)
    except (TimeoutError, RuntimeError) as e:
        print(f"KERNEL DID NOT FINISH{what}:", e, flush=True)
        sk.report()
        os._exit(2)


def worst_blocks(R: torch.Tensor, R_ref: torch.Tensor, n: int = 8) -> list:
    """The n (token, 128-row block) of R with the largest error: (token, block, error rms / reference rms, mean signed error /
    the block's reference rms)."""
    blocks = L // 128
    diff, ref = (R - R_ref).reshape(T, blocks, 128), R_ref.reshape(T, blocks, 128)
    err = diff.pow(2).mean(-1).sqrt() / ref.pow(2).mean().sqrt()
    top = torch.topk(err.flatten(), n)
    out = []
    for v, i in zip(top.values, top.indices):
        t, b = int(i) // blocks, int(i) % blocks
        out.append((t, b, round(float(v), 3), round(float(diff[t, b].mean() / ref[t, b].pow(2).mean().sqrt()), 3)))
    return out


def check_launch(sk, ws, layer_dir: str, out_dir: str, compare_y=None) -> bool:
    """Step 5: one launch, then y (GPU 0) vs SGLang's y (relRMS < 2e-2), y bit-identical on the other GPUs, and per GPU the
    routed / shared-down partial sums vs their fp32 references (relRMS < 1e-3). Writes y_gpu0.bin. Returns whether all pass."""
    ngpu = len(ws)
    launch_and_wait(sk, 60, "")
    y0 = ws[0]["moe_out"].cpu()
    y_ref = read_file(f"{layer_dir}/y_ref_bf16.bin", (T, H), "bf16")
    e_y = relrms(y0.float(), y_ref.float())
    max_diff = (y0.float() - y_ref.float()).abs().max().item()
    same = sum(torch.equal(ws[g]["moe_out"].cpu().view(torch.int16), y0.view(torch.int16)) for g in range(1, ngpu))
    ok = e_y < 2e-2 and same == ngpu - 1
    print(f"y (GPU 0) vs SGLang y: relRMS {e_y:.3e} max|d| {max_diff:.3e} | y bit-identical on {same}/{ngpu - 1} other GPUs")
    for g in range(ngpu):
        d = layer_dir if g == 0 else f"{layer_dir}_r{g}"
        R_ref = read_file(f"{d}/ref_R0_f32.bin", (T, L), np.float32)
        S_ref = read_file(f"{d}/ref_S0_f32.bin", (T, H), np.float32)
        R = routed_partial(ws[g])
        e_R, e_S = relrms(R, R_ref), relrms(ws[g]["shared_down_partial"].cpu(), S_ref)
        # 1e-3 for both (S goes through the bf16 rounding of the shared activation, and that rounding depends on how shared gate_up's
        # K is split, so S differs from its reference by more than R: up to about 1e-4)
        ok = ok and e_R < 1e-3 and e_S < 1e-3
        print(f"  GPU {g}: routed partial vs ref relRMS {e_R:.2e} | shared-down partial vs ref relRMS {e_S:.2e}")
        if e_R >= 1e-3:
            print("    worst (token, row block, err/ref rms, mean signed diff / ref block rms):", worst_blocks(R, R_ref))
    if compare_y and not os.path.exists(compare_y):
        print(f"y comparison skipped: {compare_y} does not exist")
    elif compare_y:
        yc = read_file(compare_y, (T, H), "bf16")
        eq = torch.equal(yc.view(torch.int16), y0.view(torch.int16))
        n_diff = (y0.view(torch.int16) != yc.view(torch.int16)).sum().item()
        print(f"y (GPU 0) bit-identical to {compare_y}: {eq} (relRMS {relrms(y0.float(), yc.float()):.3e}, {n_diff} of {T * H} values differ)")
    y0.view(torch.int16).numpy().tofile(os.path.join(out_dir, "y_gpu0.bin"))
    sk.report()   # the counters and per-SM stamps of the checked launch (host.cuh static_host_report)
    return ok


def timed_launches(sk, ws, reps: int, wait_s: float, after_launch=None) -> list:
    """Step 6: `reps` launches, 512 MB read on every GPU before each (L2 emptied). Per launch: the span = max over GPUs of
    (latest SM end - last SM past the start barrier), as static_host_timing measures it, and y / GPU 0's routed rows and shared sums
    compared with the first launch's (all expected the same: every sum runs in a fixed order).
    after_launch(): called after each launch (e.g. a tool that reads the timing build's stamps there). Returns the spans (us)."""
    differs = lambda a, b: a.view(torch.int32) != b.view(torch.int32)   # value by value, as bits
    spans, n_diff, rel_diff, per_token, rs_diff = [], [], [], [], []
    y0 = ws[0]["moe_out"].cpu()
    R0, S0 = ws[0]["routed_sum_partial"].cpu(), ws[0]["shared_down_partial"].cpu()
    for rep in range(reps):
        launch_and_wait(sk, wait_s, f" at timed launch {rep}")
        spans.append(sk.span_us())
        if after_launch:
            after_launch()
        y = ws[0]["moe_out"].cpu()
        y_differs = y.view(torch.int16) != y0.view(torch.int16)
        n_diff.append(int(y_differs.sum()))
        rel_diff.append(relrms(y.float(), y0.float()))
        per_token.append(tuple(y_differs.sum(1).tolist()))
        rs_diff.append((int(differs(ws[0]["routed_sum_partial"].cpu(), R0).sum()), int(differs(ws[0]["shared_down_partial"].cpu(), S0).sum())))
    print(f"y (GPU 0) of each timed launch vs the checked launch: values differing min {min(n_diff)} max {max(n_diff)} of {T * H}, "
          f"relRMS max {max(rel_diff):.3e}")
    print("  values differing per token (token 0..7) -> number of timed launches:", dict(Counter(per_token)))
    print("  GPU 0's R, S of each timed launch vs the checked launch: (R values differing, S values differing) -> number of launches:",
          dict(Counter(rs_diff)))
    return spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-dir", required=True)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--plan", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_plan.json"),
                   help="a plan file (compiler.write_plan_file): solver_plan.json (default; found by search.search_plan) or "
                        "hand_plan.json (written by hand); kernels/<plan>_layer.cu is the layer each one gives")
    p.add_argument("--wait", type=float, default=60, help="seconds to wait for one launch before reporting it as not finished")
    p.add_argument("--nvcc-flag", action="append", default=[], help="extra nvcc flag for the build (repeatable)")
    p.add_argument("--out", default="moe_static")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--rmsnorm-recompute", action="store_true", help="sum_rmsnorm: each task adds the squares of the whole row itself "
                   "(no swap between a token's tasks)")
    p.add_argument("--compare-y", default=None, help="a y file (bf16 [8, 7168], e.g. another build's y) to compare y with, value by value")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ngpu = args.gpus
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    ws, mpk = build(args.layer_dir, ngpu, args.rmsnorm_recompute)
    print_nodes(mpk, "graph nodes as written (default grids):")
    plan = compiler.plan_from_file(args.plan)
    # 63 = 64 entries per SM list (static_mk::MAX_TASKS_PER_SM, static_megakernel/config.cuh) minus the end marker
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"), ngpu, num_sms, max_tasks_per_sm=63)
    print_nodes(mpk, "graph nodes after compiling:")
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag)
    cases = [line.strip() for line in open(sk.cu_path).read().splitlines() if line.strip().startswith("case ")]
    print("generated layer:", sk.cu_path, "| kernel task loop branches:", cases)
    if not check_launch(sk, ws, args.layer_dir, args.out, args.compare_y):
        print("CHECK FAILED: no timing")
        sk.finalize()
        sys.exit(1)
    spans = timed_launches(sk, ws, args.reps, args.wait)
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)} x {ngpu}, plan {args.plan}, grids {info['grids']}, "
          f"{args.reps} launches (L2 emptied before each): "
          f"span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us "
          f"(latest SM end - last SM past the start barrier, max over GPUs)")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"plan": args.plan, "grids": info["grids"], "spans_us": spans}, f, indent=1)
    sk.finalize()


if __name__ == "__main__":
    main()
