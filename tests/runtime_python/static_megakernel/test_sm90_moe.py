"""The sm90_moe task type (MPK's Hopper MXFP4 MoE kernel in the static megakernel): routed W13 and W2, checked against
torch at GPT-OSS-20B shapes (32 experts, top-4, 2880), 8 tokens.
usage: [ROUTED=<tokens routed>] test_sm90_moe.py [reps]"""
import os, statistics, sys
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk.static_harness import Harness

T, E, TOPK, H, I = 8, 32, 4, 2880, 2880
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
from sm90_util import E2M1, SM90_FLAGS, decode


def weights(n, k):
    packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(118, 124, (E, n, k // 32), dtype=torch.uint8, device="cuda")
    return packed, scales, decode(packed, scales)


def main():
    torch.manual_seed(0)
    x = (torch.randn(T, H, device="cuda") * 0.1).bfloat16()
    w13p, w13s, w13 = weights(2 * I, H)
    w2p, w2s, w2 = weights(H, I)
    b13 = torch.randn(E, 2 * I, device="cuda").bfloat16()
    b2 = torch.randn(E, H, device="cuda").bfloat16()
    # token t < ROUTED routed to experts (3t + j) % E, j < TOPK (ROUTED = 1: a decode step, 4 experts)
    ROUTED = int(os.environ.get("ROUTED", T))
    routing = torch.zeros(E, T, dtype=torch.int32, device="cuda")
    for t in range(ROUTED):
        for j in range(TOPK):
            routing[(3 * t + j) % E, t] = j + 1
    active = [e for e in range(E) if routing[e].any()]
    mask = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    mask[:len(active)] = torch.tensor(active, dtype=torch.int32)
    mask[E] = len(active)
    mid = torch.zeros(T, TOPK, 2 * I, dtype=torch.bfloat16, device="cuda")
    x2 = (torch.randn(T, TOPK, I, device="cuda") * 0.1).bfloat16()
    out = torch.zeros(T, TOPK, H, dtype=torch.bfloat16, device="cuda")

    mpk = StaticMegakernel(num_gpus=1)
    tensors = {"x": x, "w13p": w13p, "w13s": w13s, "b13": b13, "w2p": w2p, "w2s": w2s, "b2": b2, "routing": routing,
               "mask": mask, "mid": mid, "x2": x2, "out": out}
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in tensors.items()}
    mpk.sm90_moe_layer(A["x"], A["w13p"], A["w13s"], A["routing"], A["mask"], A["mid"], bias=A["b13"], w13=True,
                       rows_per_task=192)
    mpk.sm90_moe_layer(A["x2"], A["w2p"], A["w2s"], A["routing"], A["mask"], A["out"], bias=A["b2"], w13=False,
                       rows_per_task=64)
    # the plan: every node's default grid, its tasks dealt round-robin over the SMs
    nsm = mpk._num_sms()
    lists, k = [[] for _ in range(nsm)], 0
    grids = {n.graph_idx: tuple(n.grid) for n in compiler.graph_nodes(mpk)}
    for node, grid in grids.items():
        for y in range(grid[1]):
            for xx in range(grid[0]):
                lists[k % nsm].append((node, (xx, y, 0)))
                k += 1
    out_dir = os.environ.get("OUT", "build_static") + "/sm90_moe"
    paths, info = compiler.compile_plan(mpk, {"grids": grids, "lists": [lists]}, os.path.join(out_dir, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[tensors], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS)
    h = Harness(sk)
    h.launch(timeout_s=20)


    def reference(inp, w, b, n):
        ref = torch.zeros(T, TOPK, n, device="cuda")
        for t in range(T):
            for e in range(E):
                j = int(routing[e, t]) - 1
                if j >= 0 and t < ROUTED:
                    row = inp[t] if inp.dim() == 2 else inp[t, j]
                    ref[t, j] = row.float() @ w[e].T + b[e].float()
        return ref


    for name, got, ref in (("W13", mid, reference(x, w13, b13, 2 * I)), ("W2", out, reference(x2, w2, b2, H))):
        got, ref = got[:ROUTED], ref[:ROUTED]
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        print(f"CHECK {name}: rel err {rel:.3e}")
    spans = []
    for _ in range(REPS):
        h.launch(flush_bytes=256 << 20, timeout_s=20)
        spans.append(h.span_us())
    print(f"TIME W13 + W2 ({len(active)} experts, 8 tokens): span median {statistics.median(spans):.1f} us")
    sk.finalize()


if __name__ == "__main__":
    main()
