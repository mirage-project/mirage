"""Check the static RMSNorm and MoE combine tasks used by GPT-OSS."""
import os

import torch

from mirage.mpk import compiler
from mirage.mpk.static_harness import Harness
from mirage.mpk.static_megakernel import StaticMegakernel
from sm90_util import SM90_FLAGS

T, H, TOPK = 8, 2880, 4


def main():
    torch.manual_seed(0)
    r = lambda *shape: torch.randn(*shape, device="cuda")
    bufs = {
        "eo": r(T, TOPK, H).bfloat16(),
        "weights": torch.softmax(r(T, TOPK), -1),
        "x": r(T, H).bfloat16(),
        "y": torch.zeros(T, H, dtype=torch.bfloat16, device="cuda"),
        "gamma": (1 + 0.1 * r(H)).bfloat16(),
        "norm": torch.zeros(T, H, dtype=torch.float16, device="cuda"),
    }
    mpk = StaticMegakernel(num_gpus=1)
    a = {name: mpk.attach_input(torch_tensor=value, name=name) for name, value in bufs.items()}
    mpk.sm90_mul_sum_add_layer(a["eo"], a["weights"], a["x"], a["y"])
    mpk.sm90_rmsnorm_layer(a["x"], a["gamma"], a["norm"], 1e-5)

    nsm = mpk._num_sms()
    lists = [[] for _ in range(nsm)]
    grids = {node.graph_idx: tuple(node.grid) for node in compiler.graph_nodes(mpk)}
    task = 0
    for node, grid in grids.items():
        for x in range(grid[0]):
            lists[task % nsm].append((node, (x, 0, 0)))
            task += 1
    out_dir = os.path.join(os.environ.get("OUT", "build_static"), "sm90_small")
    paths, _ = compiler.compile_plan(mpk, {"grids": grids, "lists": [lists]}, os.path.join(out_dir, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS)
    Harness(sk).launch(timeout_s=20)

    x = bufs["x"].float()
    want_norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * bufs["gamma"].float()
    want_y = x + (bufs["weights"][..., None] * bufs["eo"].float()).sum(1)
    for name, got, want in (("rmsnorm", bufs["norm"], want_norm), ("mul_sum_add", bufs["y"], want_y)):
        rel = ((got.float() - want).norm() / want.norm()).item()
        print(f"CHECK {name}: rel err {rel:.2e}")
        assert rel < 5e-3
    sk.finalize()


if __name__ == "__main__":
    main()
