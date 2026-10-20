"""The sm90_gate task type (o_proj's bias and residual, the MoE's input norm, the router and the top-k in one task a token)
against torch at GPT-OSS-20B shapes: x1, h, the top-k weights, the routing indices and the active-expert list, NTOK (default 3)
of the 8 tokens routed."""
import os
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk.static_harness import Harness
from sm90_util import SM90_FLAGS

T, H, E, TOPK, EPS = 8, 2880, 32, 4, 1e-5


def main():
    ntok = int(os.environ.get("NTOK", 3))
    torch.manual_seed(0)
    r = lambda *s: torch.randn(*s, device="cuda")
    bufs = {"o": r(T, H) * 4, "ob": r(H).bfloat16(), "x": (r(T, H) * 4).bfloat16(), "nw": (1 + 0.1 * r(H)).bfloat16(),
            "rw": (r(E, H) * 0.05).bfloat16(), "rb": r(E).bfloat16(),
            "ntok": torch.tensor([ntok], dtype=torch.int32, device="cuda"),
            "x1": torch.zeros(T, H, dtype=torch.bfloat16, device="cuda"),
            "h": torch.zeros(T, H, dtype=torch.float16, device="cuda"),
            "tw": torch.zeros(T, TOPK, device="cuda"), "routing": torch.full((E, T), 7, dtype=torch.int32, device="cuda"),
            "active": torch.zeros(E + 1, dtype=torch.int32, device="cuda")}
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    mpk.sm90_gate_layer(A["o"], A["ob"], A["x"], A["nw"], A["rw"], A["rb"], A["ntok"], A["x1"], A["h"], A["tw"],
                        A["routing"], A["active"], eps=EPS)
    nsm = mpk._num_sms()
    (node, grid), = [(n.graph_idx, tuple(n.grid)) for n in compiler.graph_nodes(mpk)]
    lists = [[] for _ in range(nsm)]
    for x in range(grid[0]):
        lists[x % nsm].append((node, (x, 0, 0)))
    out_dir = os.environ.get("OUT", "build_static") + "/sm90_gate"
    paths, _ = compiler.compile_plan(mpk, {"grids": {node: grid}, "lists": [lists]}, os.path.join(out_dir, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS,
                            profile=os.environ.get("PROFILE") == "1")
    hh = Harness(sk)
    hh.launch(timeout_s=20)
    if os.environ.get("PROFILE") == "1":   # the slowest task, L2 emptied, over 20 launches
        import statistics
        d = []
        for _ in range(20):
            hh.launch(flush_bytes=128 << 20, timeout_s=20)
            tt = hh.task_times()[0]
            d.append(max((e[1] - e[0]) / 1e3 for sm in range(nsm) for e in tt[sm][:len(lists[sm])] if e[0]))
        print(f"TIME slowest task {statistics.median(d):.2f} us")

    x1 = (bufs["o"] + bufs["ob"].float() + bufs["x"].float()).bfloat16()
    xf = x1.float()
    h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * bufs["nw"].float()).half()
    logits = (h.float() @ bufs["rw"].float().T + bufs["rb"].float()).bfloat16().float()
    probs = torch.softmax(logits, -1)
    top_p, top_e = probs.topk(TOPK, -1)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    ok = True
    e1, e2 = rel(bufs["x1"], x1), rel(bufs["h"], h)
    print(f"CHECK x1 rel err {e1:.2e} | h rel err {e2:.2e}")
    ok = ok and e1 < 1e-2 and e2 < 1e-2
    want_routing = torch.zeros(E, T, dtype=torch.int32, device="cuda")
    for t in range(ntok):
        for j in range(TOPK):
            want_routing[top_e[t, j], t] = j + 1
    same = torch.equal(bufs["routing"], want_routing)
    ew = rel(bufs["tw"][:ntok], top_p[:ntok] / top_p[:ntok].sum(-1, keepdim=True))
    used = sorted({int(e) for e in top_e[:ntok].flatten()})
    got = bufs["active"].tolist()
    act_ok = got[E] == len(used) and got[:len(used)] == used
    print(f"CHECK routing equal {same} | top-k weights rel err {ew:.2e} | active list {got[:got[E]]} (want {used}) "
          f"{act_ok} | padding rows' weights zero {bool((bufs['tw'][ntok:] == 0).all())}")
    ok = ok and same and ew < 1e-3 and act_ok
    print("PASS" if ok else "FAIL")
    sk.finalize()


if __name__ == "__main__":
    main()
