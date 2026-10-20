"""GPT-OSS-20B's MoE block as one static megakernel on Hopper, checked against torch:
  y = x + MoE(RMSNorm(x)): router (a gemm_tile, padded to 64 rows) + bias, top-4 softmax, W13 (MXFP4) + bias, clamped SwiGLU,
  W2 (MXFP4) + bias, the weighted combine.
usage: [ROUTED=<tokens of the 8>] test_sm90_moe_block.py [reps]"""
import os, statistics, sys
import torch
from mirage.core import bfloat16, float32, int32
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk.static_harness import Harness
from sm90_util import SM90_FLAGS, decode

T, E, TOPK, H, I = 8, 32, 4, 2880, 2880
EPS, LIMIT, ALPHA = 1e-5, 7.0, 1.702
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ROUTED = int(os.environ.get("ROUTED", T))
dev = "cuda"
torch.manual_seed(0)


def mxfp4(n, k):
    packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    scales = torch.randint(118, 122, (E, n, k // 32), dtype=torch.uint8, device=dev)
    return packed, scales, decode(packed, scales)


x = torch.randn(T, H, device=dev).bfloat16()
gamma = (1 + 0.1 * torch.randn(H, device=dev)).bfloat16()
w_router = torch.zeros(64, H, device=dev, dtype=torch.bfloat16)   # 32 experts, padded to a whole 64-row tile
w_router[:E] = (torch.randn(E, H, device=dev) * 0.05).bfloat16()
b_router = torch.randn(E, device=dev).bfloat16()
w13p, w13s, w13 = mxfp4(2 * I, H)
w2p, w2s, w2 = mxfp4(H, I)
b13 = (torch.randn(E, 2 * I, device=dev) * 0.1).bfloat16()
b2 = (torch.randn(E, H, device=dev) * 0.1).bfloat16()
num_tokens = torch.tensor([ROUTED], dtype=torch.int32, device=dev)

bufs = {"x": x, "gamma": gamma, "w_router": w_router, "b_router": b_router, "w13p": w13p, "w13s": w13s, "b13": b13,
        "w2p": w2p, "w2s": w2s, "b2": b2, "num_tokens": num_tokens,
        "h": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "logits32": torch.zeros(T, 64, dtype=torch.float32, device=dev),
        "logits": torch.zeros(T, E, dtype=torch.bfloat16, device=dev),
        "topk_w": torch.zeros(T, TOPK, dtype=torch.float32, device=dev),
        "routing": torch.zeros(E, T, dtype=torch.int32, device=dev),
        "active": torch.zeros(E + 1, dtype=torch.int32, device=dev),
        "mid": torch.zeros(T, TOPK, 2 * I, dtype=torch.bfloat16, device=dev),
        "act": torch.zeros(T, TOPK, I, dtype=torch.bfloat16, device=dev),
        "eo": torch.zeros(T, TOPK, H, dtype=torch.bfloat16, device=dev),
        "y": torch.zeros(T, H, dtype=torch.bfloat16, device=dev)}

mpk = StaticMegakernel(num_gpus=1)
A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
mpk.sm90_rmsnorm_layer(A["x"], A["gamma"], A["h"], EPS)
mpk.gemm_tile_layer(input=A["h"], weight=A["w_router"], output=A["logits32"], combine="store")
mpk.sm90_bias_layer(A["logits32"], A["b_router"], A["logits"])
mpk.sm90_topk_layer(A["logits"], A["num_tokens"], A["topk_w"], A["routing"], A["active"])
mpk.sm90_moe_layer(A["h"], A["w13p"], A["w13s"], A["routing"], A["active"], A["mid"], bias=A["b13"], w13=True,
                   rows_per_task=192)
mpk.sm90_swiglu_layer(A["mid"], A["act"], LIMIT, ALPHA)
mpk.sm90_moe_layer(A["act"], A["w2p"], A["w2s"], A["routing"], A["active"], A["eo"], bias=A["b2"], w13=False,
                   rows_per_task=64)
mpk.sm90_mul_sum_add_layer(A["eo"], A["topk_w"], A["x"], A["y"])

# the plan: the nodes' default grids, tasks dealt round-robin over the SMs in graph order (a producer's tasks before its
# consumers' on every SM)
nsm = mpk._num_sms()
lists, k = [[] for _ in range(nsm)], 0
grids = {n.graph_idx: tuple(n.grid) for n in compiler.graph_nodes(mpk)}
for node, grid in grids.items():
    for z in range(grid[2]):
        for y in range(grid[1]):
            for xx in range(grid[0]):
                lists[k % nsm].append((node, (xx, y, z)))
                k += 1
out_dir = os.environ.get("OUT", "build_static") + "/sm90_moe_block"
paths, _ = compiler.compile_plan(mpk, {"grids": grids, "lists": [lists]}, os.path.join(out_dir, "schedules"))
sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS)
h = Harness(sk)
h.launch(timeout_s=20)

# the reference, in fp32 from the same bf16 inputs
xf = x.float()
hn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).bfloat16().float() * gamma.float()
logits = hn @ w_router[:E].float().T + b_router.float()
top_v, top_i = logits.topk(TOPK, dim=-1)
top_w = torch.softmax(top_v, dim=-1)
y_ref = xf.clone()
for t in range(ROUTED):
    for j in range(TOPK):
        e = int(top_i[t, j])
        gu = hn[t] @ w13[e].T + b13[e].float()
        gate, up = gu[:I].clamp(max=LIMIT), gu[I:].clamp(-LIMIT, LIMIT)
        a = (up + 1) * gate * torch.sigmoid(gate * ALPHA)
        y_ref[t] += top_w[t, j] * (a @ w2[e].T + b2[e].float())
got = bufs["y"][:ROUTED].float()
ref = y_ref[:ROUTED]
print(f"CHECK routing: experts match {bool((bufs['routing'][top_i[:ROUTED], torch.arange(ROUTED, device=dev)[:, None]] > 0).all())}")
print(f"CHECK MoE block ({ROUTED} tokens): rel err {((got - ref).norm() / (ref - xf[:ROUTED]).norm()).item():.3e} "
      "(of the MoE's contribution)")
spans = []
for _ in range(REPS):
    h.launch(flush_bytes=256 << 20, timeout_s=20)
    spans.append(h.span_us())
print(f"TIME MoE block ({int(bufs['active'][E])} experts): span median {statistics.median(spans):.1f} us")
sk.finalize()
