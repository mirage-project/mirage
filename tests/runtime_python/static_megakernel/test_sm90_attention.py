"""The sm90_attention task type (MPK's Hopper paged attention as a static megakernel task) at GPT-OSS-20B shapes (64 q heads,
8 KV heads, head dim 64, sinks): one decode token of one request at position POS (default 300) after a cache of POS tokens, a
sliding-window node (128) and a full-attention node in one kernel, each with its own cache, checked against torch (the output,
and the new K / V the task appends to the cache).
usage: [POS=<history>] test_sm90_attention.py"""
import os
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk.static_harness import Harness
from sm90_util import SM90_FLAGS

T, QH, KVH, D, PAGE, MAX_SEQ, WINDOW = 8, 64, 8, 64, 64, 1024, 128
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
    qkv = (torch.randn(T, KVH * (QPK + 2) * D, device="cuda") * 0.5).bfloat16()
    sinks = torch.randn(KVH, QPK, device="cuda").bfloat16()
    caches = {}
    for w in ("s", "f"):
        for kv in ("k", "v"):
            c = torch.zeros(pages, PAGE, KVH, D, dtype=torch.bfloat16, device="cuda")
            c.view(-1, KVH, D)[:pos] = torch.randn(pos, KVH, D, device="cuda").bfloat16()
            caches[f"{kv}_{w}"] = c
    npages = pos // PAGE + 1
    bufs = dict(qkv=qkv, cos=cos, sin=sin, sinks=sinks, **caches,
                qo=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                kv_indptr=torch.tensor([0, npages], dtype=torch.int32, device="cuda"),
                kv_indices=torch.arange(pages, dtype=torch.int32, device="cuda"),
                last_len=torch.tensor([pos + 1 - (npages - 1) * PAGE], dtype=torch.int32, device="cuda"),
                out_s=torch.zeros(T, QH * D, dtype=torch.bfloat16, device="cuda"),
                out_f=torch.zeros(T, QH * D, dtype=torch.bfloat16, device="cuda"))
    k_hist = {w: caches[f"k_{w}"].view(-1, KVH, D)[:pos].clone() for w in ("s", "f")}
    v_hist = {w: caches[f"v_{w}"].view(-1, KVH, D)[:pos].clone() for w in ("s", "f")}
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    for w, window in (("s", WINDOW), ("f", 0)):
        mpk.sm90_attention_layer(A["qkv"], A[f"k_{w}"], A[f"v_{w}"], A["cos"], A["sin"], A["qo"], A["kv_indptr"],
                                 A["kv_indices"], A["last_len"], A[f"out_{w}"], num_q_heads=QH, window=window,
                                 sinks=A["sinks"])
    nsm = mpk._num_sms()
    lists, k = [[] for _ in range(nsm)], 0
    grids = {n.graph_idx: tuple(n.grid) for n in compiler.graph_nodes(mpk)}
    for node, grid in grids.items():
        for y in range(grid[1]):
            for xx in range(grid[0]):
                lists[k % nsm].append((node, (xx, y, 0)))
                k += 1
    out_dir = os.environ.get("OUT", "build_static") + "/sm90_attention"
    paths, _ = compiler.compile_plan(mpk, {"grids": grids, "lists": [lists]}, os.path.join(out_dir, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=os.path.join(out_dir, "build"), extra_flags=SM90_FLAGS)
    Harness(sk).launch(timeout_s=20)

    # the new token: q heads, k, v of each KV head (qkv interleaved by KV head), RoPE at position pos
    x = qkv[0].float().view(KVH, QPK + 2, D)
    q = rope(x[:, :QPK].reshape(QH, D), cos[pos].float(), sin[pos].float())
    k_new = rope(x[:, QPK], cos[pos].float(), sin[pos].float())
    v_new = x[:, QPK + 1]
    ok = True
    for w, window in (("s", WINDOW), ("f", 0)):
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
    sk.finalize()


if __name__ == "__main__":
    main()
