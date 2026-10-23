"""The small sm90 task types against torch at GPT-OSS-20B shapes: sm90_router (logits = h . W^T + b; h bf16 and f16),
sm90_bias (fp32 + bias + residual -> bf16), sm90_mul_sum_add (residual + the top-k weighted expert outputs), sm90_rmsnorm
(bf16 in, f16 out), sm90_topk (top-4 softmax of logits of its own, NTOK = 5 of the 8 tokens routed); E experts (env E:
32, GPT-OSS-120B's 128)."""
import os
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk.static_harness import Harness
from sm90_util import SM90_FLAGS

T, H, TOPK, N, NTOK = 8, 2880, 4, 5120, 5
E = int(os.environ.get("E", 32))   # experts (GPT-OSS-20B 32, 120B 128)


def main():
    torch.manual_seed(0)
    r = lambda *s: torch.randn(*s, device="cuda")
    bufs = {"h": r(T, H).bfloat16(), "w": (r(E, H) * 0.05).bfloat16(), "b": r(E).bfloat16(),
            "logits": torch.zeros(T, E, dtype=torch.bfloat16, device="cuda"),
            "in32": r(T, N), "bias": r(N).bfloat16(), "res": r(T, N).bfloat16(),
            "out": torch.zeros(T, N, dtype=torch.bfloat16, device="cuda"),
            "eo": r(T, TOPK, H).bfloat16(), "tw": torch.softmax(r(T, TOPK), -1), "x": r(T, H).bfloat16(),
            "y": torch.zeros(T, H, dtype=torch.bfloat16, device="cuda"),
            "gamma": (1 + 0.1 * r(H)).bfloat16(), "hn": torch.zeros(T, H, dtype=torch.float16, device="cuda"),
            "logits16": torch.zeros(T, E, dtype=torch.bfloat16, device="cuda"),
            "tlogits": (r(T, E) * 2).bfloat16(), "ntok": torch.tensor([NTOK], dtype=torch.int32, device="cuda"), "topk_w": torch.zeros(T, TOPK, device="cuda"),
            "routing": torch.full((E, T), 7, dtype=torch.int32, device="cuda"),
            "active": torch.zeros(E + 1, dtype=torch.int32, device="cuda")}
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    mpk.sm90_router_layer(A["h"], A["w"], A["b"], A["logits"])
    mpk.sm90_bias_layer(A["in32"], A["bias"], A["out"], residual=A["res"])
    mpk.sm90_mul_sum_add_layer(A["eo"], A["tw"], A["x"], A["y"])
    mpk.sm90_rmsnorm_layer(A["x"], A["gamma"], A["hn"], 1e-5)
    mpk.sm90_router_layer(A["hn"], A["w"], A["b"], A["logits16"])
    mpk.sm90_topk_layer(A["tlogits"], A["ntok"], A["topk_w"], A["routing"], A["active"])
    tlogits = bufs["tlogits"].float()   # sm90_topk zeroes its logits
    nsm = mpk._num_sms()
    lists, k = [[] for _ in range(nsm)], 0
    grids = {n.graph_idx: tuple(n.grid) for n in compiler.graph_nodes(mpk)}
    for node, grid in grids.items():
        for xx in range(grid[0]):
            lists[k % nsm].append((node, (xx, 0, 0)))
            k += 1
    out_dir = os.environ.get("OUT", "build_static") + "/sm90_small"
    paths, _ = compiler.compile_plan(mpk, {"grids": grids, "lists": [lists]}, os.path.join(out_dir, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS)
    Harness(sk).launch(timeout_s=20)
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    refs = {"router": (bufs["logits"], bufs["h"].float() @ bufs["w"].float().T + bufs["b"].float()),
            "bias": (bufs["out"], bufs["in32"] + bufs["bias"].float() + bufs["res"].float()),
            "mul_sum_add": (bufs["y"], bufs["x"].float() + (bufs["tw"][..., None] * bufs["eo"].float()).sum(1))}
    xf = bufs["x"].float()
    refs["rmsnorm f16"] = (bufs["hn"], xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * bufs["gamma"].float())
    refs["router f16 in"] = (bufs["logits16"], bufs["hn"].float() @ bufs["w"].float().T + bufs["b"].float())
    ok = True
    for name, (got, want) in refs.items():
        e = rel(got, want)
        ok = ok and e < 5e-3
        print(f"CHECK {name}: rel err {e:.2e}")
    # weights renormalized, padding rows unrouted, the active list in any order
    top_p, top_e = torch.softmax(tlogits, -1).topk(TOPK, -1)
    want_routing = torch.zeros(E, T, dtype=torch.int32, device="cuda")
    for t in range(NTOK):
        for j in range(TOPK):
            want_routing[top_e[t, j], t] = j + 1
    used = sorted({int(e) for e in top_e[:NTOK].flatten()})
    act = bufs["active"].tolist()
    ew = rel(bufs["topk_w"][:NTOK], top_p[:NTOK] / top_p[:NTOK].sum(-1, keepdim=True))
    same, act_ok = torch.equal(bufs["routing"], want_routing), act[E] == len(used) and sorted(act[:act[E]]) == used
    print(f"CHECK topk: routing equal {same} | weights rel err {ew:.2e} | active list {act_ok}")
    ok = ok and same and act_ok and ew < 1e-3
    print("PASS" if ok else "FAIL")
    sk.finalize()


if __name__ == "__main__":
    main()
