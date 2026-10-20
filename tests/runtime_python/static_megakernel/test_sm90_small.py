"""The small sm90 task types against torch at GPT-OSS-20B shapes: sm90_router (logits = h . W^T + b), sm90_bias (fp32 + bias
+ residual -> bf16), sm90_mul_sum_add (residual + the top-k weighted expert outputs)."""
import os
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk.static_harness import Harness
from sm90_util import SM90_FLAGS

T, H, E, TOPK, N = 8, 2880, 32, 4, 5120


def main():
    torch.manual_seed(0)
    r = lambda *s: torch.randn(*s, device="cuda")
    bufs = {"h": r(T, H).bfloat16(), "w": (r(E, H) * 0.05).bfloat16(), "b": r(E).bfloat16(),
            "logits": torch.zeros(T, E, dtype=torch.bfloat16, device="cuda"),
            "in32": r(T, N), "bias": r(N).bfloat16(), "res": r(T, N).bfloat16(),
            "out": torch.zeros(T, N, dtype=torch.bfloat16, device="cuda"),
            "eo": r(T, TOPK, H).bfloat16(), "tw": torch.softmax(r(T, TOPK), -1), "x": r(T, H).bfloat16(),
            "y": torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")}
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    mpk.sm90_router_layer(A["h"], A["w"], A["b"], A["logits"])
    mpk.sm90_bias_layer(A["in32"], A["bias"], A["out"], residual=A["res"])
    mpk.sm90_mul_sum_add_layer(A["eo"], A["tw"], A["x"], A["y"])
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
    ok = True
    for name, (got, want) in refs.items():
        e = rel(got, want)
        ok = ok and e < 5e-3
        print(f"CHECK {name}: rel err {e:.2e}")
    print("PASS" if ok else "FAIL")
    sk.finalize()


if __name__ == "__main__":
    main()
