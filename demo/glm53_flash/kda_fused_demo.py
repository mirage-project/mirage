"""GLM-5.3-Flash KDA attention with the sublayer boundary inside the layer, as dsa_fused_demo.py for the DSA layer (Kimi K3's
fused layers, #811: o_proj, the TP all-reduce, the residual and the norm in one kernel): one decode step of T = 8 requests on
--tp GPUs in one process (static_host's multicast exchange region), one kernel per GPU.

  input    per GPU its TP shard's KDA states, random, laid out as SGLang's KDA pool (conv [slots][3][conv dim] bf16,
           temporal [slots][heads][V][K] fp32; the 8 tokens on slots out of order); the residual streams R [8, 4 x 4096]
           bf16 (random) and the attention's mHC pre before the layer (as SGLang's hc_attn_pre: x_attn, h_post, h_res)
  layer    sglang_kda/kda_static_layer.build_graph, the layer SGLang runs: in_proj -> kda_decode -> o_proj (its K
           parts added: this GPU's partial sum) -> allreduce_send (bf16, to every GPU) -> hc_post_mix (the sum over the
           GPUs in GPU order; R'_j = sum_i comb[i][j] R_i + post_j out; the FFN's mHC mixes) -> hc_pre (the FFN's mHC pre:
           x_ffn = RMSNorm(sum_i pre_i R'_i) post_attention_layernorm, the FFN's h_post / h_res finished)
  outputs  R', x_ffn, h_post_ffn, h_res_ffn on every GPU; the states updated in place
  check    the boundary from the kernel's own per-GPU partial sums (dsa_fused_demo.check_boundary's formulas: SGLang's mHC
           post / pre); every GPU's outputs equal; per GPU kda_decode's o and the new states against kda_reference.ref_kda
           on the same inputs, o_proj from the kernel's o; end to end: the reference's partial sums, then the boundary
  time     --reps launches, 512 MB read before each (L2 emptied), the states put back: the span over the GPUs

  plans    the round-robin plan (kda_static_layer.plan_of; OUT/plan.json), or --plan FILE (one GPU's lists, repeated for the
           GPUs); --search COSTS: the plan search.search_plan finds from it with the prefetch cost model
  costs    --record FILE: the timing build, this run's cost records appended to FILE; --collect FILE: every plan of
           plan_gen.collection_plans and the round-robin plan run with --record FILE by this script (--jobs at a time). On
           one GPU (--gpus 1 --tp 4) the all-reduce's wait for the other GPUs is not in the records (as dsa_fused_demo).

python demo/glm53_flash/kda_fused_demo.py --model-dir /raid/catalyst/models/GLM-5.3-Flash [--tp 4] [--gpus N] [--layer 0]
    [--o-parts 4] [--out /tmp/flash_kda_fused] [--reps 20] [--nvcc-flag=-DSTATIC_RESET_IN_KERNEL]
    [--plan FILE] [--search COSTS ...] [--record FILE] [--collect FILE [--jobs 4] [--rounds 3]]
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("KDA_STATIC_ROOT", os.path.dirname(os.path.dirname(HERE)))   # kda_static_layer: this checkout
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sglang_kda"))
import dsa_fused_demo as dfd  # noqa: E402
import kda_reference as ref  # noqa: E402
import kda_static_layer as ksl  # noqa: E402

from mirage.mpk import compiler  # noqa: E402
from mirage.mpk.static_harness import Harness  # noqa: E402
from mirage.mpk.static_megakernel import StaticMegakernel  # noqa: E402

T, HC, H = ksl.T, ksl.HC, dfd.H
SLOTS = max(16, T)   # the state slots of the T tokens (T = 32 / 64: slots 16 .. T - 1 after the 16, in a fixed shuffle)
ROWS = ([3, 5, 0, 7, 9, 11, 2, 14, 1, 4, 6, 8, 10, 12, 13, 15] + [19, 24, 16, 30, 21, 27, 17, 25, 22, 31, 18, 28, 20, 26, 23,
                                                                     29]
        + [32 + (i * 23) % 32 for i in range(32)])[:T]
FLUSH_BYTES = 512 << 20
WARM = 2   # timed launches left out of the cost records
bf, relrms = ref.bf, dfd.relrms


def run_args(args) -> list:
    """The arguments that decide the graph and the state, for this script in another process (--collect)."""
    return (["--model-dir", args.model_dir, "--layer", str(args.layer), "--tp", str(args.tp), "--gpus", str(args.gpus),
             "--o-parts", str(args.o_parts), "--seed", str(args.seed), "--iters", str(args.iters), "--reps", str(args.reps),
             "--idle", str(args.idle)] +
            [f"--nvcc-flag={f}" for f in args.nvcc_flag])


def collect(args, mpk, current: dict) -> None:
    """--collect: every plan of plan_gen.collection_plans and the current plan ("cur") written to
    OUT/plans/<name>.json and run by this script with --record (output in OUT/<name>/), --jobs at a time: the builds in
    parallel, the GPU work one at a time (the lock OUT/gpu.lock). A plan whose run left a result.json is skipped."""
    from concurrent.futures import ThreadPoolExecutor
    from mirage.mpk import plan_gen
    plans = [("cur", current)] + plan_gen.collection_plans(mpk, rounds=args.rounds, seed=args.seed)
    plan_dir = os.path.join(args.out, "plans")
    os.makedirs(plan_dir, exist_ok=True)
    print(f"collect: {len(plans)} plans -> {plan_dir}, records -> {args.collect}, {args.jobs} at a time", flush=True)

    def one(item):
        name, plan = item
        path, run_out = os.path.join(plan_dir, name + ".json"), os.path.join(args.out, name)
        compiler.write_plan_file(path, dict(plan, lists=plan["lists"][:1]))
        if os.path.exists(os.path.join(run_out, "result.json")):
            return name, "done before"
        cmd = [sys.executable, os.path.abspath(__file__)] + run_args(args) + [
            "--plan", path, "--record", args.collect, "--out", run_out, "--gpu-lock", os.path.join(args.out, "gpu.lock")]
        with open(os.path.join(plan_dir, name + ".log"), "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        return name, f"FAILED (exit {code}), log {plan_dir}/{name}.log" if code else "ok"

    failed = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for i, (name, status) in enumerate(ex.map(one, plans)):
            print(f"collect {i + 1}/{len(plans)} {name}: {status}", flush=True)
            failed += [name] if status.startswith("FAILED") else []
    print(f"collect: {len(plans) - len(failed)} of {len(plans)} plans measured" + (f", failed: {failed}" if failed else ""),
          flush=True)


def rank_weights(w: dict, rank: int, tp: int) -> dict:
    """The layer's weights restricted to this rank's heads: ref_kda on them gives this rank's o_proj partial sum."""
    D, HL = ref.D, w["num_heads"] // tp
    cs, hs = slice(rank * HL * D, (rank + 1) * HL * D), slice(rank * HL, (rank + 1) * HL)
    r = dict(w)
    for k in ("q", "k", "v", "f_b", "g_b", "q_conv", "k_conv", "v_conv", "dt_bias"):
        r[k] = w[k][cs]
    r["b"], r["A_log"], r["o_proj"], r["num_heads"] = w["b"][hs], w["A_log"][hs], w["o_proj"][:, cs], HL
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=0, help="GPUs of the layer (default --tp): GPU r holds TP shard r of --tp, the "
                    "all-reduce adds these GPUs' partial sums (fewer than --tp: a test of the mechanism on fewer GPUs)")
    ap.add_argument("--o-parts", type=int, default=ksl.O_PARTS, help="o_proj's K parts")
    ap.add_argument("--post-only", action="store_true", help="the layer ends at the attention's mHC post (R' alone)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=0, help="Sinkhorn iterations (default: the config's)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="/tmp/flash_kda_fused")
    ap.add_argument("--nvcc-flag", action="append", default=[])
    ap.add_argument("--plan", help="a plan file (compiler.write_plan_file) instead of the round-robin plan")
    ap.add_argument("--idle", type=int, default=0, help="a decode step whose last N rows are idle (slot -1, a padded batch): "
                    "their o = 0, their states and conv windows untouched")
    ap.add_argument("--search", nargs="+", metavar="COSTS", help="cost record files: the plan search.search_plan finds from "
                    "the round-robin plan (or --plan) with the prefetch cost model")
    ap.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    ap.add_argument("--costs-code", default="current", help="with --search: the records of these kernel templates: current "
                    "(static_schedule.code_tag), any, or a tag")
    ap.add_argument("--record", metavar="FILE", help="the timing build; this run's cost records appended to FILE")
    ap.add_argument("--collect", metavar="FILE", help="a cost collection: each plan run with --record FILE")
    ap.add_argument("--jobs", type=int, default=4, help="with --collect: plans run at once")
    ap.add_argument("--rounds", type=int, default=3, help="with --collect: random plans per candidate grid of each node")
    ap.add_argument("--gpu-lock", help="a lock file held around this run's GPU work (--collect gives it to its runs)")
    args = ap.parse_args()
    NL = T - args.idle   # the live rows
    tp, n = args.tp, args.gpus or args.tp
    assert n <= tp and torch.cuda.device_count() >= n, f"{torch.cuda.device_count()} GPUs visible, --gpus {n} of --tp {tp}"
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)["text_config"]
    assert cfg["layer_types"][args.layer] == "linear_attention" and cfg["hc_mult"] == HC
    eps, hc_eps, iters = cfg["rms_norm_eps"], cfg["hc_eps"], args.iters or cfg["hc_sinkhorn_iters"]

    # what every GPU shares: the mHC parameters, the streams, the attention's mHC pre (reference, CPU)
    w = ref.load_kda_layer(ref.Checkpoint(args.model_dir), args.layer)
    m = dfd.load_mhc(args.model_dir, args.layer)
    g = torch.Generator().manual_seed(args.seed + 7)
    R = bf(torch.randn(T, HC, H, generator=g))
    x_attn, post_a, comb_a = dfd.hc_pre_ref(R, m["hc_attn_fn"], m["hc_attn_base"], m["hc_attn_scale"],
                                            m["input_layernorm.weight"], eps, hc_eps, eps, iters)
    fused = {"hc_fn": m["hc_ffn_fn"], "hc_base": m["hc_ffn_base"], "hc_scale": m["hc_ffn_scale"],
             "norm": m["post_attention_layernorm.weight"]}

    # per GPU: its shard's states, graph tensors, reference (on the states before the launch)
    ts, saved, refs = [], [], []
    for r in range(n):
        dev = torch.device("cuda", r)
        wr = rank_weights(w, r, tp)
        HLD = wr["num_heads"] * ref.D
        gr = torch.Generator(device=dev).manual_seed(args.seed + 100 + r)
        conv = (torch.randn(SLOTS, 3, 3 * HLD, generator=gr, device=dev) * 0.5).to(torch.bfloat16)
        temporal = torch.randn(SLOTS, wr["num_heads"], ref.D, ref.D, generator=gr, device=dev) * 0.05
        t = ksl.layer_tensors(args.layer, args.model_dir, r, tp, conv, temporal, dev, fused)
        rows = torch.tensor(ROWS, dtype=torch.int32, device=dev)
        t["x"].copy_(x_attn)
        t["rows"].copy_(rows)
        t["R_in"].copy_(R.reshape(T, HC * H))
        t["h_post_in"].copy_(post_a)
        t["h_res_in"].copy_(comb_a)
        live = rows.long()
        out_ref, y_ref, conv_new, rec_new = ref.ref_kda(x_attn.to(dev), wr, conv[live].transpose(1, 2).contiguous(),
                                                        temporal[live])
        if args.idle:   # the idle rows: o = 0 (their reference states are not compared)
            t["rows"][T - args.idle:] = -1
            out_ref[T - args.idle:] = 0
            y_ref[T - args.idle:] = 0
        refs.append({"out": out_ref.cpu(), "y": y_ref.cpu(), "conv": conv_new, "rec": rec_new, "w_o": wr["o_proj"]})
        saved.append((conv, temporal, conv.clone(), temporal.clone()))
        ts.append(t)

    mpk = StaticMegakernel(num_gpus=n)
    grids = ksl.build_graph(mpk, ts[0], {"rms_eps": eps, "hc_eps": hc_eps, "norm_eps": eps, "iters": iters,
                                         "o_parts": args.o_parts, "post_only": args.post_only})
    if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the layer's GEMM weights into L2 at its start (core.cuh kernel_begin)
        mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))
    os.makedirs(args.out, exist_ok=True)
    plan = ksl.plan_of(mpk, grids) if not args.plan else compiler.plan_from_file(args.plan)
    if args.collect:
        del ts, saved, refs
        torch.cuda.empty_cache()
        collect(args, mpk, plan)
        return
    if args.search:
        from mirage.mpk import plan_viewer, search, static_schedule
        code = {"current": static_schedule.code_tag(), "any": None}.get(args.costs_code, args.costs_code)
        costs = search.Costs(args.search, code=code, gpu=torch.cuda.get_device_name(0), model="prefetch")
        plan, summary = search.search_plan(mpk, costs, time_limit_s=args.time, start_from=dict(plan, lists=plan["lists"][:1]))
        plan_viewer.save(os.path.join(args.out, "plan_view.json"), plan, summary, f"searched from {args.plan or 'round robin'}")
    if len(plan["lists"]) == 1 and n > 1:
        plan["lists"] = plan["lists"] * n   # every GPU: the same lists (no node splits its rows over the GPUs)
    compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
    paths, info = compiler.compile_plan(mpk, compiler.plan_from_file(os.path.join(args.out, "plan.json")),
                                        os.path.join(args.out, "schedules"))
    timing = bool(args.record)
    sk = mpk.compile_static(paths, gpu_tensors=ts, out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag,
                            profile=timing)
    if args.gpu_lock:   # held to the end of the process (the OS lets go of it at exit)
        import fcntl
        lock = open(args.gpu_lock, "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
    h = Harness(sk)
    try:
        h.launch(FLUSH_BYTES, timeout_s=60)
    except (TimeoutError, RuntimeError) as e:
        print("KERNEL DID NOT FINISH:", e, flush=True)
        os._exit(2)
    for r in range(n):
        torch.cuda.synchronize(r)
    print(f"layer {args.layer}, {n} GPUs with TP{tp} shards 0..{n - 1} ({w['num_heads'] // tp} heads per GPU), slots {ROWS}")

    # per GPU: KDA against the reference, o_proj from the kernel's o
    o = [dfd.fx_float(ts[r]["o_fx"].cpu()) for r in range(n)]
    ok = True
    for r in range(n):
        conv, temporal, conv0, tmp0 = saved[r]
        live = torch.tensor(ROWS[:NL], device=conv.device).long()   # (--idle: the last rows idle, their slots untouched)
        idle_s = torch.tensor(ROWS[NL:], device=conv.device).long()
        same = (conv[live].transpose(1, 2).contiguous().view(torch.int16) ==
                refs[r]["conv"][:NL].view(torch.int16)).float().mean().item()
        e_y = relrms(ts[r]["o"].cpu(), refs[r]["y"].reshape(T, -1))
        e_rec = relrms(temporal[live], refs[r]["rec"][:NL])
        if args.idle:
            kept = torch.equal(conv[idle_s], conv0[idle_s]) and torch.equal(temporal[idle_s], tmp0[idle_s])
            o_idle = ts[r]["o"][NL:].float().abs().max().item()
            print(f"  GPU {r}: {args.idle} idle rows: o max |o| {o_idle:.1e}, their slots untouched {kept}")
            ok = ok and kept and o_idle == 0.0
        o_node = relrms(o[r], (ts[r]["o"].float() @ refs[r]["w_o"].to(conv.device).float().t()).cpu())
        o_ref = relrms(o[r], refs[r]["out"].float())
        print(f"  GPU {r}: kda o relRMS {e_y:.2e} | conv state equal {100 * same:.3f}% | temporal relRMS {e_rec:.2e} | "
              f"o_proj from the kernel's o {o_node:.2e} | partial sum vs the reference {o_ref:.2e}")
        ok = ok and e_y < 5e-3 and same > 0.999 and e_rec < 1e-3 and o_node < 1e-3

    # the boundary from the kernel's partial sums, and end to end
    ffn = lambda Rx: dfd.hc_pre_ref(Rx, m["hc_ffn_fn"], m["hc_ffn_base"], m["hc_ffn_scale"],
                                    m["post_attention_layernorm.weight"], eps, hc_eps, eps, iters)
    keys = ("R_out", "x_ffn", "h_post_out", "h_res_out")
    outs = [{k: ts[r][k].cpu() for k in keys} for r in range(n)]
    R_ref = dfd.hc_post_ref(dfd.tp_sum(o), R, post_a, comb_a)
    R_k = outs[0]["R_out"].view(T, HC, H)
    same_gpus = all(torch.equal(outs[r][k], outs[0][k]) for r in range(1, n) for k in keys)
    eqR = (R_k.view(torch.int16) == R_ref.view(torch.int16)).float().mean().item()
    outs_ref = [x["out"].float() for x in refs]
    R_e2e = dfd.hc_post_ref(dfd.tp_sum(outs_ref), R, post_a, comb_a)
    if args.post_only:
        eR, eR2 = relrms(R_k, R_ref), relrms(R_k.float() - R.float(), R_e2e.float() - R.float())
        print(f"  boundary (post only): R' relRMS {eR:.2e} (bit-equal {100 * eqR:.3f}%) | end to end R' - R {eR2:.2e}")
        ok = ok and same_gpus and eR < 1e-3 and eR2 < 3e-2
    else:
        x_ref, post_f, comb_f = ffn(R_k)
        torch.save({"outs": outs[0], "x_ref": x_ref, "ffn_norm": ts[0]["ffn_norm"].cpu(), "R_ref": R_ref},
                   os.path.join(args.out, "outputs.pt"))
        eqx = (outs[0]["x_ffn"].view(torch.int16) == x_ref.view(torch.int16)).float().mean().item()
        e = {"R": relrms(R_k, R_ref), "x_ffn": relrms(outs[0]["x_ffn"], x_ref),
             "h_post": relrms(outs[0]["h_post_out"], post_f), "h_res": relrms(outs[0]["h_res_out"], comb_f)}
        print(f"  boundary (from the kernel's partial sums): R' relRMS {e['R']:.2e} (bit-equal {100 * eqR:.3f}%) | "
              f"x_ffn relRMS {e['x_ffn']:.2e} (bit-equal {100 * eqx:.3f}%) | h_post {e['h_post']:.2e} | h_res {e['h_res']:.2e} | "
              f"every GPU's outputs equal: {same_gpus}")
        x_e2e, post_e2e, _ = ffn(R_e2e)
        ee = {"R' - R": relrms(R_k.float() - R.float(), R_e2e.float() - R.float()), "x_ffn": relrms(outs[0]["x_ffn"], x_e2e),
              "h_post": relrms(outs[0]["h_post_out"], post_e2e)}
        print("  end to end (the reference's partial sums, then the boundary): " +
              " | ".join(f"{k} {x:.2e}" for k, x in ee.items()))
        ok = ok and same_gpus and e["R"] < 1e-3 and e["x_ffn"] < 1e-2 and e["h_post"] < 1e-4 and e["h_res"] < 1e-4 and \
            ee["x_ffn"] < 3e-2
    if not ok:
        print("CHECK FAILED: no timing")
        sk.finalize()
        sys.exit(1)
    print("CHECK PASSED")

    spans, launches = [], []
    for _ in range(args.reps):
        for conv, temporal, conv0, tmp0 in saved:
            conv.copy_(conv0)
            temporal.copy_(tmp0)
        h.launch(FLUSH_BYTES, timeout_s=60)
        spans.append(h.span_us())
        if timing:
            launches.append((h.task_times(), h.timing()))
    if args.record:
        from mirage.mpk import cost_records
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
        records = cost_records.records_of_run(mpk, paths, launches[WARM:], os.path.basename(os.path.normpath(args.out)), machine)
        cost_records.append_records(args.record, records)
        print(f"cost records: {len(records)} added to {args.record} ({machine})")
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)} x {n}, grids {info['grids']}, {args.reps} launches: "
          f"span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"plan": args.plan or "round robin", "grids": {str(k): v for k, v in info["grids"].items()}, "gpus": n,
                   "timing_build": timing, "spans_us": spans}, f, indent=1)
    sk.finalize()


if __name__ == "__main__":
    main()
