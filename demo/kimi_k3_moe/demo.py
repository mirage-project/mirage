"""Kimi K3 MoE layer, TP8, on the static-schedule compiler, all 8 GPUs from this process:

  1. load_weights    the real layer weights of every GPU (files written from the K3 checkpoint; the file names are in load_weights)
  2. build_graph     the layer as 8 layer calls (no grid, no K split yet)
  3. compile_plan    the plan (the hand plan, the solver's plan, or a plan file) -> the graph regridded to the plan's grids,
                     schedules/schedule_gpu<g>.json                          (python/mirage/mpk/compiler.py)
  4. compile_static  build/layer.cu from the schedules, built and loaded     (python/mirage/mpk/static_schedule.py)
  5. check_launch    one launch; y vs SGLang's y, y the same on all GPUs, each GPU's partial sums vs their fp32 references
  6. timed_launches  --reps launches, 512 MB written on every GPU before each (empties L2); span per launch

python demo.py --layer-dir ~/mk/moe_layer [--plan hand|search|PLAN_FILE] [--costs FILE] [--time 60] [--out DIR] [--reps 20]
               [--compare-y y_rank0.bin]
Weight files: <layer_dir>/ holds the tensors every GPU has (router, latent_down, latent_up, gamma, bias, the 8-token input x and the
residual prefix, SGLang's y), <layer_dir>_r<g>/ GPU g's shard (shared gate_up / down, expert banks).
"""
import argparse, os, socket, statistics, sys, json
from collections import Counter
import numpy as np
import torch
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk import compiler, moe
from mirage.core import bfloat16, float32, int64, uint8
from mirage.utils import get_configurations_from_gpu

T, H, L, E, TOPK = 8, 7168, 3584, 896, 16       # tokens, hidden, latent, experts, routed experts per token
IR_LOCAL, SHR_LOCAL = 384, 768                   # routed / shared intermediate per GPU at TP8
EPI_ROUTER, EPI_LATENT, EPI_SGU = 0, 1, 2        # the gemm_tile kind: which weight and output


def load_weights(layer_dir: str, gpu: int) -> dict:
    """GPU `gpu`'s tensors on device `gpu`, shapes as the bodies consume them (expert banks in the kernel's packed tile order)."""
    rd = lambda p, shape, dt: torch.from_numpy(np.fromfile(p, dtype=dt).reshape(shape))
    bf = lambda p, shape: rd(p, shape, np.uint16).view(torch.bfloat16)
    b, r = layer_dir, f"{layer_dir}_r{gpu}" if gpu else layer_dir
    w = {"router_weight": bf(f"{b}/w_gate_bf16.bin", (E, H)), "score_correction_bias": rd(f"{b}/gate_bias_f32.bin", (E,), np.float32),
         "latent_down_weight": bf(f"{b}/w_down_bf16.bin", (L, H)), "latent_up_weight": bf(f"{b}/w_up_bf16.bin", (H, L)),
         "routed_expert_norm_weight": bf(f"{b}/gamma_bf16.bin", (L,)),
         "moe_in": bf(f"{b}/x_bf16.bin", (T, H)), "moe_prefix": bf(f"{b}/prefix_bf16.bin", (T, H)),
         "shared_gate_up_weight": bf(f"{r}/w_sgu_bf16.bin", (2 * SHR_LOCAL, H)), "shared_down_weight": bf(f"{r}/w_sd_bf16.bin", (H, SHR_LOCAL)),
         "w13_blocks": rd(f"{r}/w13_blk.bin", (E, 2 * IR_LOCAL, L // 2), np.uint8), "w13_scales": rd(f"{r}/w13_sf.bin", (E, 2 * IR_LOCAL, L // 32), np.uint8),
         "w2_blocks": rd(f"{r}/w2_blk.bin", (E, L, IR_LOCAL // 2), np.uint8), "w2_scales": rd(f"{r}/w2_sf.bin", (E, L, IR_LOCAL // 32), np.uint8)}
    dev = torch.device("cuda", gpu)
    w = {k: v.contiguous().to(dev) for k, v in w.items()}
    # the layer's outputs, read back for the checks: y, and this GPU's routed / shared partial sums before the all-reduce
    w.update(moe_out=torch.zeros(T, H, dtype=torch.bfloat16, device=dev), routed_sum_partial=torch.zeros(T, L, dtype=torch.float32, device=dev),
             shared_down_partial=torch.zeros(T, H, dtype=torch.float32, device=dev))
    return w


def build_kernel() -> StaticMegakernel:
    params = StaticMegakernel.get_default_init_parameters()
    qo_indptr = torch.zeros(T + 1, dtype=torch.int32, device="cuda"); qo_indptr[T] = T
    num_workers, num_schedulers = get_configurations_from_gpu(0)
    params.update(mode="offline", test_mode=True, world_size=1, mpi_rank=0, num_workers=num_workers, num_local_schedulers=num_schedulers,
                  max_num_batched_tokens=T, max_num_batched_requests=T, meta_tensors={"qo_indptr_buffer": qo_indptr})
    return StaticMegakernel(**params)


def build_graph(mpk: StaticMegakernel, w: dict):
    """The K3 MoE layer as 8 layer calls. No grid_dim, no K split: the plan decides them."""
    a = lambda k: mpk.attach_input(torch_tensor=w[k], name=k)
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name, io_category="cuda_tensor")
    x, prefix = a("moe_in"), a("moe_prefix")
    w_router, bias, w_down, w_sgu, w_sd = a("router_weight"), a("score_correction_bias"), a("latent_down_weight"), a("shared_gate_up_weight"), a("shared_down_weight")
    w13, w13_sf, w2, w2_sf, gamma, w_up = a("w13_blocks"), a("w13_scales"), a("w2_blocks"), a("w2_scales"), a("routed_expert_norm_weight"), a("latent_up_weight")
    G = nt((4096,), uint8, "globals"); M = nt((4096,), uint8, "maps")     # the bodies' G struct and tensor maps (filled by static_megakernel/moe_host.cuh)
    logits = nt((T, E), float32, "router_logits"); pairs = nt((T, TOPK), int64, "routing_pairs")
    z = nt((T, L), float32, "latent_z"); z_q = nt((T, L), uint8, "latent_z_mxfp8")
    sgu = nt((T, 2 * SHR_LOCAL), float32, "shared_gate_up"); h_s = nt((T, SHR_LOCAL), bfloat16, "shared_act")
    R, S, y = a("routed_sum_partial"), a("shared_down_partial"), a("moe_out")
    mpk.gemm_tile_layer(input=x, weight=w_router, globals=G, maps=M, output=logits, kind=EPI_ROUTER)      # router: logits = x . Wg^T (fp32)
    mpk.route_layer(input=logits, bias=bias, globals=G, output=pairs)                                     # sigmoid + bias, top-16, renorm
    mpk.gemm_tile_layer(input=x, weight=w_down, globals=G, maps=M, output=z, kind=EPI_LATENT)             # latent_down: z = x . Wdown^T
    mpk.quant_layer(input=z, globals=G, output=z_q)                                                       # MXFP8 quantize + all-gather
    mpk.gemm_tile_layer(input=x, weight=w_sgu, globals=G, maps=M, output=sgu, kind=EPI_SGU)               # shared gate_up
    mpk.sact_layer(input=sgu, globals=G, output=h_s)                                                      # SiTU
    mpk.expert_queue_layer(z_q=z_q, pairs=pairs, h_s=h_s, w13=w13, w2=w2, globals=G, maps=M, output=(R, S),   # W13 + SiTU + requant, W2, shared down
                              w13_scales=w13_sf, w2_scales=w2_sf, shared_down_weight=w_sd)
    mpk.tail_layer(r_partial=R, s_partial=S, prefix=prefix, gamma=gamma, w_up=w_up, globals=G, maps=M, output=y)   # [R|S] all-reduce, RMSNorm, latent_up, y
    return y


def print_nodes(mpk, title: str) -> None:
    print(title)
    for n in compiler.graph_nodes(mpk):
        print(f"  node {n.graph_idx:2d} {n.name:<16} grid {n.grid} params {n.params}   in {[compiler.tname(mpk, t) for t in n.inputs]} -> out {[compiler.tname(mpk, t) for t in n.outputs]}")


def relrms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def build(layer_dir: str, ngpu: int):
    """Steps 1-2: (every GPU's tensors, the StaticMegakernel holding the graph)."""
    ws = [load_weights(layer_dir, g) for g in range(ngpu)]
    torch.cuda.set_device(0)
    mpk = build_kernel(); build_graph(mpk, ws[0])
    return ws, mpk


def check_launch(sk, ws, layer_dir: str, out_dir: str, compare_y=None) -> bool:
    """Step 5: one launch, then y (GPU 0) vs SGLang's y (relRMS < 2e-2), y bit-identical on the other GPUs, and per GPU the
    routed / shared-down partial sums vs their fp32 references (relRMS < 1e-3). Writes y_gpu0.bin. Returns whether all pass."""
    ngpu = len(ws)
    sk.launch(l2_flush_bytes=512 << 20)
    try: sk.wait(timeout_s=60)
    except (TimeoutError, RuntimeError) as e:
        print("KERNEL DID NOT FINISH:", e, flush=True)
        sk.report()
        os._exit(2)
    y0 = ws[0]["moe_out"].cpu()
    y_ref = torch.from_numpy(np.fromfile(f"{layer_dir}/y_ref_bf16.bin", dtype=np.uint16).reshape(T, H)).view(torch.bfloat16)
    e_y = relrms(y0.float(), y_ref.float()); mx = (y0.float() - y_ref.float()).abs().max().item()
    same = sum(torch.equal(ws[g]["moe_out"].cpu().view(torch.int16), y0.view(torch.int16)) for g in range(1, ngpu))
    ok = e_y < 2e-2 and same == ngpu - 1
    print(f"y (GPU 0) vs SGLang y: relRMS {e_y:.3e} max|d| {mx:.3e} | y bit-identical on {same}/{ngpu - 1} other GPUs")
    for g in range(ngpu):
        d = layer_dir if g == 0 else f"{layer_dir}_r{g}"
        rR = torch.from_numpy(np.fromfile(f"{d}/ref_R0_f32.bin", dtype=np.float32).reshape(T, L)); rS = torch.from_numpy(np.fromfile(f"{d}/ref_S0_f32.bin", dtype=np.float32).reshape(T, H))
        eR, eS = relrms(ws[g]["routed_sum_partial"].cpu(), rR), relrms(ws[g]["shared_down_partial"].cpu(), rS)
        # 1e-3 for both (the reference kernel uses 1e-4 for S at its one split; S goes through the bf16 rounding of the shared activation,
        # whose rounding depends on the shared gate_up K split, measured up to 1.1e-4 at split 4)
        ok = ok and eR < 1e-3 and eS < 1e-3
        print(f"  GPU {g}: routed partial vs ref relRMS {eR:.2e} | shared-down partial vs ref relRMS {eS:.2e}")
    if compare_y and not os.path.exists(compare_y):
        print(f"y comparison skipped: {compare_y} does not exist")
    elif compare_y:
        yc = torch.from_numpy(np.fromfile(compare_y, dtype=np.uint16).reshape(T, H)).view(torch.bfloat16)
        eq = torch.equal(yc.view(torch.int16), y0.view(torch.int16))
        print(f"y (GPU 0) bit-identical to {compare_y}: {eq} (relRMS {relrms(y0.float(), yc.float()):.3e}, {(y0.view(torch.int16) != yc.view(torch.int16)).sum().item()} of {T * H} values differ)")
    y0.view(torch.int16).numpy().tofile(os.path.join(out_dir, "y_gpu0.bin"))
    sk.report()   # the task family's own view of the checked launch
    return ok


def timed_launches(sk, ws, reps: int, wait_s: float, after_launch=None) -> list:
    """Step 6: `reps` launches, 512 MB written on every GPU before each (L2 emptied). Per launch: the span = max over GPUs of
    (latest SM end - last SM past the start barrier), as moe_host_timing measures it, and y / GPU 0's partial
    sums compared with the first launch's (they differ a little from launch to launch: float atomic adds in varying order).
    after_launch(): called after each launch (the search tools read their stamps there). Returns the spans (us)."""
    spans, ndiff, rdiff, per_tok, rs_diff = [], [], [], [], []
    y0 = ws[0]["moe_out"].cpu()
    R0, S0 = ws[0]["routed_sum_partial"].cpu(), ws[0]["shared_down_partial"].cpu()
    for rep in range(reps):
        sk.launch(l2_flush_bytes=512 << 20)
        try: sk.wait(timeout_s=wait_s)
        except (TimeoutError, RuntimeError) as e:
            print(f"KERNEL DID NOT FINISH at timed launch {rep}:", e, flush=True)
            sk.report()
            os._exit(2)
        spans.append(sk.span_us())
        if after_launch: after_launch()
        yn = ws[0]["moe_out"].cpu(); ne = (yn.view(torch.int16) != y0.view(torch.int16))
        ndiff.append(int(ne.sum())); rdiff.append(relrms(yn.float(), y0.float())); per_tok.append(tuple(ne.sum(1).tolist()))
        rs_diff.append((int((ws[0]["routed_sum_partial"].cpu().view(torch.int32) != R0.view(torch.int32)).sum()), int((ws[0]["shared_down_partial"].cpu().view(torch.int32) != S0.view(torch.int32)).sum())))
    print(f"y (GPU 0) of each timed launch vs the checked launch: values differing min {min(ndiff)} max {max(ndiff)} of {T * H}, relRMS max {max(rdiff):.3e}")
    print("  values differing per token (token 0..7) -> number of timed launches:", dict(Counter(per_tok)))
    print("  GPU 0's R, S of each timed launch vs the checked launch: (R values differing, S values differing) -> number of launches:", dict(Counter(rs_diff)))
    return spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-dir", required=True); p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--plan", default="hand", help="hand: the hand plan (moe.hand_plan); search: the solver's plan "
                   "(search.search_plan, durations from --costs); else a plan file (compiler.write_plan_file)")
    p.add_argument("--costs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "costs", "b300.json"),
                   help="with --plan search: the cost records file (default: <repo>/costs/b300.json)")
    p.add_argument("--time", type=float, default=60, help="with --plan search: solver time limit (s)")
    p.add_argument("--wait", type=float, default=60, help="seconds to wait for one launch before reporting it as not finished")
    p.add_argument("--nvcc-flag", action="append", default=[], help="extra nvcc flag for the build (repeatable)")
    p.add_argument("--out", default="moe_static"); p.add_argument("--reps", type=int, default=20)
    p.add_argument("--compare-y", default=None, help="a y file (bf16 [8, 7168], e.g. the reference kernel's y) to compare y with")
    args = p.parse_args(); os.makedirs(args.out, exist_ok=True)
    ngpu = args.gpus; num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    ws, mpk = build(args.layer_dir, ngpu)
    print_nodes(mpk, "graph nodes as written (default grids):")

    if args.plan == "hand":
        plan = moe.hand_plan(mpk, ngpu, num_sms)
    elif args.plan == "search":
        from mirage.mpk import search      # needs OR-Tools (pip install ortools), only for --plan search
        # 61 = 64 entries per SM list (static_mk::MAX_TASKS_PER_SM) - the end marker - the expert queue and tail entries
        plan, _ = search.search_plan(mpk, moe.MoeCosts(args.costs), ngpu, num_sms, 61, time_limit_s=args.time)
        compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
    else:
        plan = compiler.plan_from_file(args.plan)
    # 63 = 64 entries per SM list (static_mk::MAX_TASKS_PER_SM, static_megakernel/config.cuh) minus the end marker
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"), ngpu, num_sms, max_tasks_per_sm=63)
    print_nodes(mpk, "graph nodes after compiling:")
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag)
    code = open(sk.cu_path).read()
    print("generated layer:", sk.cu_path, "| kernel task loop branches:", [l.strip() for l in code.splitlines() if l.strip().startswith("case ")])
    if not check_launch(sk, ws, args.layer_dir, args.out, args.compare_y):
        print("CHECK FAILED: no timing"); sk.finalize(); sys.exit(1)

    spans = timed_launches(sk, ws, args.reps, args.wait)
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)} x {ngpu}, plan {args.plan}, grids {info['grids']}, "
          f"{args.reps} launches (L2 emptied before each): span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us "
          f"(latest SM end - last SM past the start barrier, max over GPUs)")
    json.dump({"plan": args.plan, "grids": info["grids"], "spans_us": spans}, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    sk.finalize()


if __name__ == "__main__":
    main()
