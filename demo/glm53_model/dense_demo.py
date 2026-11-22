"""Run one GLM-5.3 dense MLP layer (layers 0 .. first_k_dense_replace - 1; dense_mlp.py) with the static-schedule compiler,
on --gpus G GPUs (TP G, GPU g holds shard g; the all-reduce allreduce_send -> sum_gpus -> residual_add runs in the kernel),
on the round-robin plan of the decoder (moe_demo.round_robin_plan; T = STATIC_TOKENS). Every GPU gets the same x and
residual; y (mlp_out) must be equal on all GPUs.

  1. weights        dense_mlp.dense_tensors of --layer for each GPU (--checkpoint: a local snapshot of the full model)
  2. build          dense_mlp.build_graph (+ row_rmsnorm with --next-norm), compile_plan, compile_static
  3. y              the first launch (not timed): y = MLP(x) + residual; x random (seed 1)
  4. timed launches --reps launches, 512 MB read on every GPU before each (empties L2); median span
  --save-y DIR / --compare-y DIR: every launch's y (and x_next with --next-norm) to DIR/launches_dense<L>.pt, or compared
  with it bit for bit (old / new build A/B); --vary-x, --residual random, --next-norm: ab_launches.py.

STATIC_TOKENS=64 python demo/glm53_model/dense_demo.py --checkpoint /dev/shm/glm53 --layer 0 --gpus 8 \\
    --nvcc-flag=-DSTATIC_RESET_IN_KERNEL --save-y DIR [--vary-x --residual random --next-norm]
"""
import argparse
import json
import os
import statistics

import torch
from mirage.mpk import compiler
from mirage.mpk.static_harness import Harness
from mirage.mpk.static_megakernel import StaticMegakernel

from demo.glm53_model import ab_launches, dense_mlp
from demo.glm53_moe import demo as moe_demo
from demo.glm53_moe.weights import checkpoint_reader

T = dense_mlp.T
H = 6144


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a local GLM-5.3 snapshot (config.json, the safetensors index)")
    ap.add_argument("--layer", type=int, default=0, help="a dense layer (< first_k_dense_replace)")
    ap.add_argument("--gpus", type=int, default=8, help="GPUs = TP size")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="build/glm53_dense")
    ap.add_argument("--nvcc-flag", action="append", default=[], help="an nvcc flag for the layer (repeatable)")
    ap.add_argument("--save-y", metavar="DIR", help="write every launch's outputs to DIR/launches_dense<L>.pt")
    ap.add_argument("--compare-y", metavar="DIR", help="compare every launch's outputs with DIR/launches_dense<L>.pt")
    ab_launches.add_args(ap)
    args = ap.parse_args()
    with open(os.path.join(args.checkpoint, "config.json")) as f:
        cfg = json.load(f)
    if not 0 <= args.layer < cfg["first_k_dense_replace"]:
        raise ValueError(f"--layer {args.layer}: the dense layers are 0 .. {cfg['first_k_dense_replace'] - 1}")
    G = args.gpus
    devs = [torch.device("cuda", g) for g in range(G)]
    read = checkpoint_reader(args.checkpoint)
    x = torch.randn(T, H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    res = ab_launches.residual(args.residual, (T, H))
    ws, nn = [], []
    for g, d in enumerate(devs):
        w = dense_mlp.dense_tensors(read, args.layer, tp=G, rank=g, device=d)
        w.update(x=x.to(d), residual=res.to(d), mlp_out=torch.zeros(T, H, dtype=torch.bfloat16, device=d))
        ws.append(w)
        nn.append({"norm_w": ab_launches.norm_gamma(H).to(d), "x_next": torch.zeros(T, H, dtype=torch.bfloat16, device=d)}
                  if args.next_norm else {})
    mpk = StaticMegakernel(num_gpus=G)
    dense_mlp.build_graph(mpk, ws[0], next_norm=(nn[0]["norm_w"], nn[0]["x_next"], cfg["rms_norm_eps"])
                          if args.next_norm else None)
    plan = moe_demo.round_robin_plan(mpk, {})   # the decoder's dense plan at T > 8
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[dict(ws[g], **nn[g]) for g in range(G)],
                            out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag)
    h = Harness(sk)
    h.launch(timeout_s=60)
    outs = [dict({"y": ws[g]["mlp_out"]}, **({"x_next": nn[g]["x_next"]} if args.next_norm else {})) for g in range(G)]
    launch_outs = [ab_launches.outputs(outs, "launch 0")]   # every GPU the same bits
    spans = []
    for n in range(1, args.reps + 1):
        if args.vary_x:
            ab_launches.set_x([w["x"] for w in ws], ab_launches.varied_x(n, (T, H)))
        h.launch(512 << 20, timeout_s=60)
        spans.append(h.span_us())
        launch_outs.append(ab_launches.outputs(outs, f"launch {n}"))
    print(f"dense layer {args.layer}, {G} GPU(s), grids {info['grids']}: median span {statistics.median(spans):.1f} us "
          f"over {args.reps} launches")
    ok = True
    name = f"launches_dense{args.layer}.pt"
    if args.save_y:
        os.makedirs(args.save_y, exist_ok=True)
        ab_launches.save(os.path.join(args.save_y, name), launch_outs)
    if args.compare_y:
        ok = ab_launches.compare(os.path.join(args.compare_y, name), launch_outs, f"dense layer {args.layer}")
    sk.finalize()
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
