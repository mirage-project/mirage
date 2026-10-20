"""The sm90_attention task type (MPK's Hopper paged attention as a static megakernel task) at GPT-OSS-20B shapes (64 q heads,
8 KV heads, head dim 64, sinks): one decode token of one request at position POS (default 300) after a cache of POS tokens, a
sliding-window node (128), a full-attention node, and a full attention split in KV chunks (sm90_attention_split + _merge,
CHUNK) in one kernel, each with its own cache, checked against torch (the output, and the new K / V appended to the cache).
The sliding-window and split nodes take the QKV GEMM's fp32 rows and add the bias themselves (qkv_bias), the full one bf16.
usage: [POS=<history>] [PROFILE=1 (each node's span over 20 launches)] test_sm90_attention.py"""
import os
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel, sm90_attention_parts_size
from mirage.mpk.static_harness import Harness
from sm90_util import SM90_FLAGS

T, QH, KVH, D, PAGE, MAX_SEQ, WINDOW, CHUNK = 8, 64, 8, 64, 64, 1024, 128, 128
QPK = QH // KVH


def rope_tables(base: float = 150000.0):
    inv = 1.0 / base ** (torch.arange(0, D, 2, device="cuda").float() / D)
    ang = torch.arange(MAX_SEQ, device="cuda").float()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.cos()], -1).bfloat16(), torch.cat([ang.sin(), ang.sin()], -1).bfloat16()


def rope(x, cos, sin):   # NeoX rotate-half, as the kernel (rotary_embedding_hopper.cuh)
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def reference(q, k_all, v_all, sinks, pos, window):
    """q [QH, D] (rotated), k_all / v_all [pos + 1, KVH, D]: the attention output [QH x D] (fp32)."""
    lo = max(0, pos - window + 1) if window else 0
    k, v = k_all[lo:pos + 1].float(), v_all[lo:pos + 1].float()
    out = torch.empty(QH, D, device="cuda")
    for h in range(QH):
        g = h // QPK
        logits = (k[:, g] @ q[h].float()) / D ** 0.5
        m = torch.maximum(logits.max(), sinks[g, h % QPK].float())
        p = (logits - m).exp()
        den = p.sum() + (sinks[g, h % QPK].float() - m).exp()
        out[h] = (p[:, None] * v[:, g]).sum(0) / den
    return out.flatten()


def main():
    pos = int(os.environ.get("POS", 300))
    torch.manual_seed(0)
    cos, sin = rope_tables()
    pages = MAX_SEQ // PAGE
    qkv32 = torch.randn(T, KVH * (QPK + 2) * D, device="cuda") * 0.5
    qkv_bias = (torch.randn(KVH * (QPK + 2) * D, device="cuda") * 0.1).bfloat16()
    qkv = (qkv32 + qkv_bias.float()).bfloat16()
    sinks = torch.randn(KVH, QPK, device="cuda").bfloat16()
    caches = {}
    for w in ("s", "f", "p"):
        for kv in ("k", "v"):
            c = torch.zeros(pages, PAGE, KVH, D, dtype=torch.bfloat16, device="cuda")
            c.view(-1, KVH, D)[:pos] = torch.randn(pos, KVH, D, device="cuda").bfloat16()
            caches[f"{kv}_{w}"] = c
    npages = pos // PAGE + 1
    bufs = dict(qkv=qkv, qkv32=qkv32, qkv_bias=qkv_bias, staged_s=torch.zeros_like(qkv), staged_p=torch.zeros_like(qkv), cos=cos, sin=sin, sinks=sinks, **caches,
                qo=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                kv_indptr=torch.tensor([0, npages], dtype=torch.int32, device="cuda"),
                kv_indices=torch.arange(pages, dtype=torch.int32, device="cuda"),
                last_len=torch.tensor([pos + 1 - (npages - 1) * PAGE], dtype=torch.int32, device="cuda"),
                out_s=torch.zeros(T, QH * D, dtype=torch.bfloat16, device="cuda"),
                out_f=torch.zeros(T, QH * D, dtype=torch.bfloat16, device="cuda"),
                out_p=torch.zeros(T, QH * D, dtype=torch.bfloat16, device="cuda"),
                parts=torch.zeros(sm90_attention_parts_size(T, QH, D, MAX_SEQ // CHUNK), dtype=torch.bfloat16,
                                  device="cuda"))
    k_hist = {w: caches[f"k_{w}"].view(-1, KVH, D)[:pos].clone() for w in ("s", "f", "p")}
    v_hist = {w: caches[f"v_{w}"].view(-1, KVH, D)[:pos].clone() for w in ("s", "f", "p")}
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    for w, window in (("s", WINDOW), ("f", 0)):
        fused = dict(qkv_bias=A["qkv_bias"], qkv_staged=A["staged_s"]) if w == "s" else {}
        mpk.sm90_attention_layer(A["qkv32" if w == "s" else "qkv"], A[f"k_{w}"], A[f"v_{w}"], A["cos"], A["sin"], A["qo"],
                                 A["kv_indptr"], A["kv_indices"], A["last_len"], A[f"out_{w}"], num_q_heads=QH,
                                 window=window, sinks=A["sinks"], **fused)
    mpk.sm90_attention_split_layer(A["qkv32"], A["k_p"], A["v_p"], A["cos"], A["sin"], A["qo"], A["kv_indptr"],
                                   A["kv_indices"], A["last_len"], A["parts"], num_q_heads=QH, chunk=CHUNK,
                                   qkv_bias=A["qkv_bias"], qkv_staged=A["staged_p"])
    mpk.sm90_attention_merge_layer(A["parts"], A["qo"], A["kv_indptr"], A["last_len"], A["out_p"], num_q_heads=QH,
                                   chunk=CHUNK, chunks=MAX_SEQ // CHUNK, page=PAGE, sinks=A["sinks"])
    nsm = mpk._num_sms()
    lists, k = [[] for _ in range(nsm)], 0
    grids = {n.graph_idx: tuple(n.grid) for n in compiler.graph_nodes(mpk)}
    for node, grid in grids.items():
        for z in range(grid[2]):
            for y in range(grid[1]):
                for xx in range(grid[0]):
                    lists[k % nsm].append((node, (xx, y, z)))
                    k += 1
    out_dir = os.environ.get("OUT", "build_static") + "/sm90_attention"
    paths, _ = compiler.compile_plan(mpk, {"grids": grids, "lists": [lists]}, os.path.join(out_dir, "schedules"))
    profile = os.environ.get("PROFILE") == "1"
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS,
                            profile=profile)
    h = Harness(sk)
    h.launch(timeout_s=20)

    # the new token: q heads, k, v of each KV head (qkv interleaved by KV head), RoPE at position pos
    x = qkv[0].float().view(KVH, QPK + 2, D)
    q = rope(x[:, :QPK].reshape(QH, D), cos[pos].float(), sin[pos].float())
    k_new = rope(x[:, QPK], cos[pos].float(), sin[pos].float())
    v_new = x[:, QPK + 1]
    ok = True
    for w, window in (("s", WINDOW), ("f", 0), ("p", 0)):
        k_all = torch.cat([k_hist[w].float(), k_new[None]])
        v_all = torch.cat([v_hist[w].float(), v_new[None]])
        ref = reference(q, k_all, v_all, sinks, pos, window)
        got = bufs[f"out_{w}"][0].float()
        rel = ((got - ref).norm() / ref.norm()).item()
        cache_k = caches[f"k_{w}"].view(-1, KVH, D)[pos].float()
        cache_v = caches[f"v_{w}"].view(-1, KVH, D)[pos].float()
        rk = ((cache_k - k_new).norm() / k_new.norm()).item()
        rv = ((cache_v - v_new).norm() / v_new.norm()).item()
        print(f"CHECK window {window}: output rel err {rel:.2e} | appended k {rk:.2e} v {rv:.2e}")
        ok = ok and rel < 2e-2 and rk < 1e-2 and rv < 1e-2
    print("PASS" if ok else "FAIL")
    if profile:   # each node's window, first start to last end (the nodes run one after another: no shared inputs)
        import statistics
        names = {n.graph_idx: n.name for n in compiler.graph_nodes(mpk)}
        spans = {node: [] for node in grids}
        for _ in range(20):
            h.launch(flush_bytes=128 << 20, timeout_s=20)
            times = h.task_times()[0]
            for node in grids:
                d = [times[sm][i] for sm in range(nsm) for i, (n, _) in enumerate(lists[sm]) if n == node and times[sm][i][0]]
                spans[node].append((max(e[1] for e in d) - min(e[0] for e in d)) / 1e3)
        for node in grids:
            print(f"TIME node {node} {names[node]}: {statistics.median(spans[node]):.1f} us")
    sk.finalize()


if __name__ == "__main__":
    main()
