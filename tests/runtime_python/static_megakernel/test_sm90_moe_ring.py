"""The sm90_moe_ring task type (routed MXFP4 experts streamed through the static megakernel's ring on Hopper): W13 with
the clamped SwiGLU fused, then W2, checked against torch at GPT-OSS-20B shapes (32 experts, top-4, 2880) and
timed. A decode step routes ROUTED (default 1) of the 8 tokens, to at most A = 4 ROUTED experts.
usage: [ROUTED=<tokens>] [PROFILE=1 (per-node task times)] test_sm90_moe_ring.py [reps]"""
import os, statistics, sys
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel, sm90_moe_ring_pack
from mirage.mpk.static_harness import Harness
from sm90_util import SM90_FLAGS, decode

T, E, TOPK, H, I = 8, 32, 4, 2880, 2880
LIMIT, ALPHA = 7.0, 1.702


def mxfp4(n, k):
    packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(118, 124, (E, n, k // 32), dtype=torch.uint8, device="cuda")
    return packed, scales, decode(packed, scales)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    routed = int(os.environ.get("ROUTED", 1))
    A = min(E, TOPK * routed)
    torch.manual_seed(0)
    x = (torch.randn(T, H, device="cuda") * 0.1).half()   # the MoE's input is f16 (sm90_rmsnorm's f16 output)
    w13p, w13s, w13 = mxfp4(2 * I, H)
    w2p, w2s, w2 = mxfp4(H, I)
    b13 = torch.randn(E, 2 * I, device="cuda").bfloat16()
    b2 = torch.randn(E, H, device="cuda").bfloat16()
    routing = torch.zeros(E, T, dtype=torch.int32, device="cuda")
    for t in range(routed):
        for j in range(TOPK):
            routing[(3 * t + j) % E, t] = j + 1
    active = [e for e in range(E) if routing[e].any()]
    mask = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    mask[:len(active)] = torch.tensor(active, dtype=torch.int32)
    mask[E] = len(active)
    bufs = {"x": x, "w13": sm90_moe_ring_pack(w13p, w13s, 128, glu=True), "b13": b13,
            "w2": sm90_moe_ring_pack(w2p, w2s, 64, glu=False), "b2": b2, "routing": routing,
            "mask": mask, "act": torch.zeros(A, T, I, dtype=torch.float16, device="cuda"),
            "out": torch.zeros(T, TOPK, H, dtype=torch.bfloat16, device="cuda")}
    mpk = StaticMegakernel(num_gpus=1)
    D = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    mpk.sm90_moe_ring_layer(D["x"], D["w13"], D["routing"], D["mask"], D["act"], bias=D["b13"], glu=True,
                            limit=LIMIT, alpha=ALPHA, rows_per_task=128)
    mpk.sm90_moe_ring_layer(D["act"], D["w2"], D["routing"], D["mask"], D["out"], bias=D["b2"], glu=False,
                            rows_per_task=64)
    # the plan: default grids, tasks round-robin over the SMs in graph order, an expert slot's tasks together (W2's slot a
    # waits for W13's slot a only)
    nsm = mpk._num_sms()
    lists, k = [[] for _ in range(nsm)], 0
    grids = {n.graph_idx: tuple(n.grid) for n in compiler.graph_nodes(mpk)}
    for node, grid in grids.items():
        for xx in range(grid[0]):
            for y in range(grid[1]):
                lists[k % nsm].append((node, (xx, y, 0)))
                k += 1
    out_dir = os.environ.get("OUT", "build_static") + "/sm90_moe_ring"
    paths, _ = compiler.compile_plan(mpk, {"grids": grids, "lists": [lists]}, os.path.join(out_dir, "schedules"))
    profile = os.environ.get("PROFILE") == "1"
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS,
                            profile=profile)
    h = Harness(sk)
    h.launch(timeout_s=20)

    ref = torch.zeros(T, TOPK, H, device="cuda")
    ref_act = torch.zeros(A, T, I, device="cuda")
    for a, e in enumerate(active):
        gu = x.float() @ w13[e].T + b13[e].float()
        gate, up = gu[:, :I].clamp(max=LIMIT), gu[:, I:].clamp(-LIMIT, LIMIT)
        ref_act[a] = (up + 1) * gate * torch.sigmoid(gate * ALPHA)
        y = bufs["act"][a].float() @ w2[e].T + b2[e].float()   # W2 checked on the task's own activation
        for t in range(T):
            j = int(routing[e, t]) - 1
            if j >= 0:
                ref[t, j] = y[t]
    rel13 = ((bufs["act"].float() - ref_act).norm() / ref_act.norm()).item()
    got, want = bufs["out"][:routed].float(), ref[:routed]
    rel2 = ((got - want).norm() / want.norm()).item()
    print(f"CHECK W13 + SwiGLU: rel err {rel13:.3e} | W2 (routed rows): rel err {rel2:.3e}")
    spans = []
    for _ in range(reps):
        h.launch(flush_bytes=256 << 20, timeout_s=20)
        spans.append(h.span_us())
    if profile:   # per node: its tasks' median / max time, and the window from its first start to its last end
        times = h.task_times()[0]
        t0 = min(e[0] for sm in times for e in sm if e[0])
        for node in grids:
            d = [times[sm][i] for sm in range(nsm) for i, (n, _) in enumerate(lists[sm]) if n == node]
            dur = sorted((e[1] - e[0]) / 1e3 for e in d)
            print(f"NODE {node}: {len(d)} tasks, median {dur[len(dur) // 2]:.1f} us, max {dur[-1]:.1f} us, window "
                  f"{(min(e[0] for e in d) - t0) / 1e3:.1f}..{(max(e[1] for e in d) - t0) / 1e3:.1f} us")
    print(f"TIME W13 (+ SwiGLU) + W2 ({len(active)} experts): span median {statistics.median(spans):.1f} us")
    sk.finalize()


if __name__ == "__main__":
    main()
