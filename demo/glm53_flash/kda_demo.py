"""GLM-5.3-Flash KDA attention sublayer (one decode step, T = 8 requests) on the static-schedule compiler, TP4:

  1. weights      KDA layer --layer of the checkpoint (--model-dir), sharded over --gpus GPUs by heads (kda_reference.py)
  2. states       --warm decode steps of the PyTorch reference from zero states (inputs: the embeddings of random tokens,
                  copied into the 4 mHC streams), so the conv / recurrent states hold real history
  3. graph        7 layer calls: hc_mix -> hc_pre -> gemm_tile (q|k|v|b|f_a|g_a) -> kda_decode -> gemm_tile (o_proj)
                  -> allreduce_send -> hc_post
  4. plan         --plan FILE: hand_plan.json (default: hand-picked grids, round-robin placement) or solver_plan.json
                  (search.search_plan on cost records measured on 4 B200s); kernels/<plan>_layer.cu is the layer each one
                  gives; --search COSTS: the solver's plan from --plan
  5. build, run   one launch; R' vs the reference (relRMS < 1e-3, bit-identical on every GPU), o / states per GPU
  6. time         --reps launches, L2 emptied before each; --record [FILE]: the timing build, the stamps of each launch to
                  OUT/task_times.json, cost records appended to FILE

CUDA_VISIBLE_DEVICES=0,2,5,6 python demo/glm53_flash/kda_demo.py --model-dir /raid/catalyst/models/GLM-5.3-Flash \\
    [--layer 0] [--warm 16] [--plan PLAN] [--search COSTS ...] [--record [FILE]] [--out /tmp/kda_static] [--reps 20]
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_reference as ref  # noqa: E402

from mirage.core import bfloat16, float32  # noqa: E402
from mirage.mpk import compiler  # noqa: E402
from mirage.mpk.static_harness import Harness  # noqa: E402
from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel  # noqa: E402

T, HC = 8, 4                 # tokens (requests) per step, mHC streams
HC_PARTS, AR_TASKS = 64, 32  # hc_mix's K parts and allreduce_send's tasks (the plans' grids)
FLUSH_BYTES = 512 << 20      # read on every GPU before each launch: L2 emptied


def relrms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def make_states(ck, w, warm: int, seed: int, dev, tp: int):
    """--warm reference steps from zero states; returns (R of the tested step, conv state, recurrent state, expected)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 150000, ((warm + 1) * T,), generator=g).tolist()
    emb = ck.embed_rows(ids).view(warm + 1, T, -1).to(dev)
    H, NH = emb.shape[-1], w["num_heads"]
    conv = torch.zeros(T, 3 * NH * ref.D, 3, dtype=torch.bfloat16, device=dev)
    rec = torch.zeros(T, NH, ref.D, ref.D, dtype=torch.float32, device=dev)
    expand = lambda e: e.unsqueeze(1).expand(T, HC, H).contiguous()
    for s in range(warm):
        _, _, _, _, conv, rec = ref.ref_sublayer(expand(emb[s]), w, conv, rec, tp)
    R = expand(emb[warm])
    return R, conv, rec, ref.ref_sublayer(R, w, conv, rec, tp)


def gpu_tensors(w, R, conv, rec, rank: int, tp: int) -> dict:
    """GPU `rank`'s graph tensors, by name."""
    dev = torch.device("cuda", rank)
    s = ref.shard_kda(w, rank, tp, dev)
    cs, rs = ref.shard_states(conv, rec, rank, tp)
    HLD = s["f_b"].shape[0]
    return {
        "kda_res_in": R.reshape(T, -1).to(dev), "kda_hc_fn": s["hc_fn"], "kda_hc_small": s["hc_small"],
        "kda_gamma": s["gamma"], "kda_w_in": s["w_in"], "kda_conv_w": s["conv_w"], "kda_f_b": s["f_b"],
        "kda_g_b": s["g_b"], "kda_small": s["small"], "kda_conv_state": cs.to(dev), "kda_rec_state": rs.to(dev),
        "kda_rows": torch.arange(T, dtype=torch.int32, device=dev), "kda_w_o": s["w_o"],
        "kda_o": torch.zeros(T, HLD, dtype=torch.bfloat16, device=dev),
        "kda_res_out": torch.zeros(T, R.shape[1] * R.shape[2], dtype=torch.bfloat16, device=dev),
    }


def build_graph(mpk: StaticMegakernel, t: dict) -> None:
    """The KDA sublayer as 7 layer calls (no K split yet: the plan decides)."""
    a = lambda k: mpk.attach_input(torch_tensor=t[k], name=k)
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    R, fn, hc_small, gamma = a("kda_res_in"), a("kda_hc_fn"), a("kda_hc_small"), a("kda_gamma")
    w_in, conv_w, f_b, g_b, small = a("kda_w_in"), a("kda_conv_w"), a("kda_f_b"), a("kda_g_b"), a("kda_small")
    conv_state, rec_state, rows = a("kda_conv_state"), a("kda_rec_state"), a("kda_rows")
    w_o, o, R_out = a("kda_w_o"), a("kda_o"), a("kda_res_out")
    H = gamma.dim(0)
    mix = nt((T, HC_MIX_STRIDE), float32, "kda_hc_mix")
    x = nt((T, H), bfloat16, "kda_x")
    proj = nt((T, w_in.dim(0)), float32, "kda_proj")
    part = nt((T, H), float32, "kda_out_part")
    sent = nt((T, H), bfloat16, "kda_sent")
    mpk.hc_mix_layer(residual=R, fn=fn, output=mix, num_parts=HC_PARTS)                     # mHC pre, part 1
    mpk.hc_pre_layer(mix=mix, residual=R, hc_small=hc_small, gamma=gamma, output=x)          # mHC pre, part 2 + RMSNorm
    mpk.gemm_tile_layer(input=x, weight=w_in, output=proj, combine="slots", reader_rearms=False)   # q|k|v|b|f_a|g_a
    mpk.kda_decode_layer(proj=proj, conv_weight=conv_w, f_b=f_b, g_b=g_b, small=small, conv_state=conv_state,
                         rec_state=rec_state, rows=rows, output=o)                          # conv, gates, delta rule, o_norm
    mpk.gemm_tile_layer(input=o, weight=w_o, output=part, combine="slots", reader_rearms=False)    # o_proj (partial)
    mpk.allreduce_send_layer(input=part, output=sent, num_tasks=AR_TASKS)                    # to every GPU
    mpk.hc_post_layer(sent=sent, residual=R, x=x, output=R_out)                              # sum + mHC post


def print_nodes(mpk, title: str) -> None:
    print(title)
    for n in compiler.graph_nodes(mpk):
        ins = [compiler.tname(mpk, t) for t in n.inputs]
        outs = [compiler.tname(mpk, t) for t in n.outputs]
        print(f"  node {n.graph_idx:2d} {n.name:<15} grid {n.grid} params {n.params}   in {ins} -> out {outs}")


def check(ws, expect, tp: int) -> bool:
    """R' (every GPU) vs the reference, and per GPU o and the new states vs their shards."""
    R_exp, _, o_exp, _, conv_exp, rec_exp = expect
    R0 = ws[0]["kda_res_out"].cpu()
    e_R = relrms(R0, R_exp.reshape(T, -1))
    same = sum(torch.equal(ws[g]["kda_res_out"].cpu().view(torch.int16), R0.view(torch.int16)) for g in range(1, tp))
    ok = e_R < 1e-3 and same == tp - 1
    print(f"R' (GPU 0) vs reference: relRMS {e_R:.3e} | bit-identical on {same}/{tp - 1} other GPUs")
    HLD = o_exp.shape[1] // tp
    for g in range(tp):
        cs, rs = ref.shard_states(conv_exp, rec_exp, g, tp)
        e_o = relrms(ws[g]["kda_o"], o_exp[:, g * HLD:(g + 1) * HLD])
        conv_g = ws[g]["kda_conv_state"].cpu()
        exact = (conv_g.view(torch.int16) == cs.cpu().view(torch.int16)).float().mean().item()
        e_rec = relrms(ws[g]["kda_rec_state"], rs)
        ok = ok and e_o < 1e-2 and e_rec < 1e-3 and exact > 0.999
        print(f"  GPU {g}: o relRMS {e_o:.2e} | conv state equal {100 * exact:.3f}% | recurrent state relRMS {e_rec:.2e}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--layer", type=int, default=0, help="a linear_attention (KDA) layer")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--warm", type=int, default=16, help="reference decode steps before the tested one")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plan", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_plan.json"),
                   help="a plan file (compiler.write_plan_file): hand_plan.json or solver_plan.json")
    p.add_argument("--search", nargs="+", default=None, metavar="COSTS", help="cost record files: the solver's plan")
    p.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    p.add_argument("--record", nargs="?", const="", default=None, metavar="FILE",
                   help="the timing build: the stamps of each launch to OUT/task_times.json, cost records appended to FILE")
    p.add_argument("--out", default="/tmp/kda_static")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--wait", type=float, default=60, help="seconds before a launch is reported as not finished")
    p.add_argument("--nvcc-flag", action="append", default=[])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tp = args.gpus
    torch.cuda.set_device(0)

    ck = ref.Checkpoint(args.model_dir)
    w = ref.load_kda_layer(ck, args.layer)
    assert w["num_heads"] % tp == 0
    print(f"layer {args.layer}: {w['num_heads']} heads, {w['num_heads'] // tp} per GPU; {args.warm} warm-up steps")
    R, conv, rec, expect = make_states(ck, w, args.warm, args.seed, torch.device("cuda", 0), tp)
    ws = [gpu_tensors(w, R, conv, rec, g, tp) for g in range(tp)]

    mpk = StaticMegakernel(num_gpus=tp)
    build_graph(mpk, ws[0])
    print_nodes(mpk, "graph nodes as written:")
    plan = compiler.plan_from_file(args.plan)
    if args.search:
        from mirage.mpk import plan_viewer, search, static_schedule
        costs = search.Costs(args.search, code=static_schedule.code_tag(), gpu=torch.cuda.get_device_name(0))
        plan, summary = search.search_plan(mpk, costs, time_limit_s=args.time, start_from=plan)
        plan_viewer.save(os.path.join(args.out, "plan_view.json"), plan, summary, "kda sublayer")
    compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"))
    print_nodes(mpk, "graph nodes after compiling:")
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(args.out, "build"),
                            extra_flags=args.nvcc_flag, profile=args.record is not None)
    h = Harness(sk)
    try:
        h.launch(FLUSH_BYTES, timeout_s=args.wait)
    except (TimeoutError, RuntimeError) as e:
        print("KERNEL DID NOT FINISH:", e, flush=True)
        os._exit(2)
    if not check(ws, expect, tp):
        print("CHECK FAILED: no timing")
        sk.finalize()
        sys.exit(1)
    spans, launches = [], []
    for _ in range(args.reps):   # the states keep advancing; only the times matter here
        h.launch(FLUSH_BYTES, timeout_s=args.wait)
        spans.append(h.span_us())
        if args.record is not None:
            launches.append((h.task_times(), h.timing()))
    if args.record is not None:
        with open(os.path.join(args.out, "task_times.json"), "w") as f:   # the raw stamps, for a timeline
            json.dump([{"times": tt, "timing": tm} for tt, tm in launches], f)
    if args.record:
        from mirage.mpk import cost_records
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
        records = cost_records.records_of_run(mpk, paths, launches, os.path.basename(os.path.normpath(args.out)), machine)
        cost_records.append_records(args.record, records)
        print(f"cost records: {len(records)} added to {args.record}")
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)} x {tp}, grids {info['grids']}, {args.reps} launches: "
          f"span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"layer": args.layer, "grids": info["grids"], "spans_us": spans}, f, indent=1)
    sk.finalize()


if __name__ == "__main__":
    main()
