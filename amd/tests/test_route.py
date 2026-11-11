"""The router alone on one AMD GPU: gemm_tile (logits = x Wg^T, K parts in slots) then topk_route (sigmoid + bias, top 16,
renormalised), the K3 shapes, random weights, many launches. Each launch: the logits' K parts vs x Wg^T, and the pairs (expert sets, weights) vs top 16 of the kernel's own logits, so a wrong
pick shows itself whatever the GEMM did.

python amd/tests/test_route.py [--launches 20] [--grid 7,14] [--out DIR] [--flag -DX=1 ...] [--profile]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "amd", "tools"))

T, H, E, K = 8, 7168, 896, 16


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--launches", type=int, default=20)
    p.add_argument("--grid", default="7,14", help="the router GEMM's grid x,y (128-row tasks x K parts)")
    p.add_argument("--out", default="route_test")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--profile", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    from mirage.core import float32, int64
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import simple_plan
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    gx, gy = (int(v) for v in a.grid.split(","))
    w = {
        "moe_in": (torch.randn(T, H, generator=g, device=dev)).to(torch.bfloat16),
        "router_weight": (torch.randn(E, H, generator=g, device=dev) * H ** -0.5).to(torch.bfloat16),
        "score_correction_bias": torch.randn(E, generator=g, device=dev) * 0.05,
        "routing_pairs": torch.zeros(T, K, dtype=torch.int64, device=dev),
        "router_logits": torch.zeros(gy * T * E, dtype=torch.float32, device=dev),
    }
    mpk = StaticMegakernel(num_gpus=1)
    x = mpk.attach_input(torch_tensor=w["moe_in"], name="moe_in")
    wg = mpk.attach_input(torch_tensor=w["router_weight"], name="router_weight")
    bias = mpk.attach_input(torch_tensor=w["score_correction_bias"], name="score_correction_bias")
    logits = mpk.new_tensor(dims=(T, E), dtype=float32, name="router_logits")
    pairs = mpk.new_tensor(dims=(T, K), dtype=int64, name="routing_pairs")
    mpk.gemm_tile_layer(input=x, weight=wg, output=logits)
    mpk.topk_route_layer(input=logits, bias=bias, output=pairs)
    nodes = compiler.graph_nodes(mpk)
    plan = simple_plan(mpk, {nodes[0].graph_idx: (gx, gy, 1)})
    os.makedirs(a.out, exist_ok=True)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[w], out_dir=os.path.join(a.out, "build"), extra_flags=a.flag, profile=a.profile)
    h = Harness(sk)
    ref = w["moe_in"].double() @ w["router_weight"].double().T
    b = w["score_correction_bias"].double()
    bad_launches = 0
    for launch in range(a.launches):
        try:
            h.launch(0, timeout_s=30)
        except (TimeoutError, RuntimeError) as e:
            print(f"KERNEL DID NOT FINISH (launch {launch}): {e}", flush=True)
            os._exit(2)
        torch.cuda.synchronize()
        parts = w["router_logits"].reshape(gy, T, E)
        tot = torch.zeros(T, E, dtype=torch.float32, device=dev)
        for q in range(gy):
            tot = tot + parts[q]
        e_logit = ((tot.double() - ref).norm() / ref.norm()).item()
        sig = torch.sigmoid(tot.double())
        top = torch.topk(sig + b, K, dim=1).indices.cpu()
        pk = w["routing_pairs"].cpu()
        e_k, wt = (pk >> 32).long(), (pk & 0xFFFFFFFF).int().view(torch.float32)
        ok_tokens = sum(set(e_k[t].tolist()) == set(top[t].tolist()) for t in range(T))
        valid = bool(((e_k >= 0) & (e_k < E)).all())
        if valid:
            wref = sig.cpu().gather(1, e_k) / sig.cpu().gather(1, e_k).sum(1, keepdim=True)
            e_w = ((wt.double() - wref).norm() / wref.norm()).item()
        else:
            e_w = float("nan")
        bad = ok_tokens != T or not e_w < 1e-5
        bad_launches += bad
        print(f"launch {launch}: logits relRMS {e_logit:.1e}; tokens with the right 16 experts {ok_tokens}/{T}; weights relRMS "
              f"{e_w:.1e}{'  BAD' if bad else ''}", flush=True)
        if a.profile and launch == a.launches - 1:   # per node: task durations (us), from the timing build's stamps
            import json
            with open(paths[0]) as f:
                sched = json.load(f)
            times, names = h.task_times()[0], {n.graph_idx: n.name for n in compiler.graph_nodes(mpk)}
            per = {}
            for sm, queue in enumerate(sched["worker_task_queues"]):
                for i, tid in enumerate(queue):
                    st, en = times[sm][i][0], times[sm][i][1]
                    if st and en:
                        per.setdefault(names[sched["all_tasks"][tid]["node"]], []).append((en - st) / 1e3)
            print("task us per node:", {k: sorted(round(v, 1) for v in vs)[:: max(1, len(vs) // 8)] for k, vs in per.items()},
                  flush=True)
        if bad and bad_launches <= 2:
            rank = torch.argsort(torch.argsort(-(sig + b), dim=1), dim=1).cpu()
            for t in range(T):
                if valid and set(e_k[t].tolist()) != set(top[t].tolist()):
                    print(f"    token {t}: kernel experts {e_k[t].tolist()} ranks {sorted(rank[t, e_k[t]].tolist())}", flush=True)
            print(f"    raw pairs token 0: {[hex(v) for v in pk[0].tolist()]}", flush=True)
    print(f"{bad_launches} of {a.launches} launches wrong", flush=True)
    sk.finalize()
    sys.exit(1 if bad_launches else 0)


if __name__ == "__main__":
    main()
