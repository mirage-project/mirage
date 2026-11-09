"""Run one DeepSeek-V4.1-Flash attention layer with the static-schedule compiler, as one GPU of attention TP.

The layer is the official Attention.forward of a decode step (inference/model.py L766-792): x [8, 5120] bf16 and one position per
token -> y [8, 5120] bf16. Each of the 8 tokens is its own sequence with its own caches. --tp 4 is SGLang's B200 recipe (TP4 +
EP4): the GPU holds 16 of the 64 heads, its y is a partial sum, and the all-reduce that adds the 4 GPUs' parts is not part of
the layer. --tp 1 is the DP-attention shape (all 64 heads on the GPU). --gpus 4 (with --tp 4): this process runs the layer on 4
GPUs, shard r on GPU r, and the all-reduce is in the layer (sum_store_layer with num_gpus 4: each sum_store task sends its
columns of this GPU's y to every GPU in bf16, then adds the 4 GPUs' parts in GPU order in fp32): every GPU's y is the layer's
output, and the demo checks that it is bit-identical on the 4 GPUs.

The variant follows from --layer (weights.variant_of; the nodes at TP4, one more at TP2 / TP1: act_quant before wo_b):
  swa      L0, 1             the window: sparse_attn over 128 keys                                           11 nodes
  reuse    30 layers         + the compressed KV of the group's source layer at its top-k (128 + 512 keys)   10 nodes
  reindex  L24, 28, 32, 36   + the indexer: its q and weights against L20's index K, inside L20's candidate
                             blocks -> the top-k                                                              15 nodes
  full_r2  L2, 8, 14         + the compressor (2-token pooling with state), the compressed KV and index K
                             writes, the indexer over the layer's own index K                                20 nodes
  full_r1  L20               the same with ratio 1; L20 also writes the candidate blocks                    20 nodes
What a layer reads from other layers' launches (Reuse: the compressed KV and the top-k; Reindex: L20's index K, compressed KV
and candidate blocks) are graph tensors set before the launch. The layer's own caches (the window; Full layers: the compressed
KV, the index K and the compressor state) are graph tensors kept across launches.

  0. download_layer  --checkpoint: a local snapshot, or deepseek-ai/DeepSeek-V4.1-Flash (then only the shards of --layer)
  1. load_weights    the layer's tensors for GPU --rank of attention TP --tp (weights.layer_tensors; the RoPE table is computed
                     on the GPU, as the official code does)
  2. build_graph     the variant's layer calls (grids and placement come from the plan)
  3. compile_plan    --plan: "hand" (default: hand_plan below) or a plan file; --search COSTS: the solver's plan from cost
                     records, starting from that plan (search.search_plan; OR-Tools) -> <out>/plan.json
  4. compile_static  <out>/build/layer.cu, built and loaded
  5. inputs          x random (seed 1) at position --pos for all 8 tokens; the caches filled with valid random values
  6. timed launches  --reps launches, 512 MB read on the GPU before each (empties L2); the span of each. --nvcc-flag
                     -DSTATIC_RESET_IN_KERNEL: serving mode (the kernel resets its buffers itself, no host memsets).
                     --record FILE: the timing build; the per-task times are appended to FILE as cost records
  --golden DIR       instead of 5-6: golden cases of the official implementation (DeepSeek's inference/ code on random
                     inputs, one GPU, world_size 1): L<l>_<case>/case.json (steps, state and tensor names, positions),
                     shared.safetensors (the state before the first step), step<k>.safetensors (inputs, y, intermediates,
                     the state after the step; the caches also in the kernel's quantized layout). Per case and step, the --tp
                     shards run one after another on this GPU, each from the step's state: o_q byte for byte against the
                     golden's heads, y = bf16(the shards' wo_b K parts added in fp32) against the golden y, and the state
                     each shard leaves (the window cache; the compressed KV, the index K and the compressor state; the
                     caches a Reuse or Reindex layer reads, unchanged) against the golden's state after the step. A host-reset
                     build only (a serving build's readers re-arm the node buffers the checks read)
Steps 5-6 run the layer through the test harness (python/mirage/mpk/static_harness.py).

The plan files are for one TP4 GPU at --pos 4096 on a B200 (148 SMs), per variant (--layer 0, 3, 24, 2, 20):
  plans/<variant>_tp4_hand.json     hand_plan's (--plan hand, the default, plans for any position)
  plans/<variant>_tp4_solver.json   search.search_plan on cost records of the hand plan's tasks measured on a B200, started from
                                    the hand plan (the same grids; its own placement). Measured on a B200: as fast as the hand
                                    plan on swa, reuse and reindex, slower on full_r2 and full_r1
kernels/<variant>_tp4_<plan>_layer.cu is the layer each one gives. The indexer's grid follows the context (indexer_score: chunks
of 256 compressed positions or more, score_chunks), so other positions take --plan hand.

python demo.py --layer 3 [--checkpoint DIR|REPO] [--tp 4] [--rank 0] [--pos 4096] [--plan hand|FILE] [--out DIR] [--reps 20]
python demo.py --layer 3 --tp 4 --plan hand --record costs.json             (cost records of this GPU)
python demo.py --layer 3 --tp 4 --plan hand --search costs.json             (the solver's plan from these records)
python demo.py --layer 2 --tp 4 --golden /path/to/dsv4_1_golden [--cases L2_p2047 L2_mixed]
python demo.py --layer 3 --tp 4 --gpus 4 [--golden DIR]                   (4 GPUs, the all-reduce in the layer)
python demo.py --layer 3 --tp 4 --gpus 4 --cupti --nvcc-flag=-DSTATIC_RESET_IN_KERNEL
                                                          (kernel time: the serving build lines the 4 GPUs up before each launch)
"""
import argparse
import copy
import glob
import json
import os
import statistics
import subprocess

import torch
from safetensors import safe_open
from mirage.core import bfloat16, float32, uint8
from mirage.mpk import compiler
from mirage.mpk.static_harness import Harness
from mirage.mpk.static_megakernel import MAX_BUFS, RING_STAGES, StaticMegakernel, zq_bytes

try:
    from .weights import checkpoint_reader, download_layer, layer_tensors, model_config, variant_of   # as a package
except ImportError:
    from weights import checkpoint_reader, download_layer, layer_tensors, model_config, variant_of    # as a script

T, HID, QL, D, W, OL, NC = 8, 5120, 1280, 512, 128, 1024, 512   # tokens, dim, q_lora_rank, head_dim, window, o_lora_rank, top-k
ROW, CROW, IROW = D + D // 32, D // 2 + D // 16, 64 + 4          # bytes per row: window cache, compressed KV, index K
# the variants whose wo_b quantizes wo_a's output itself (no act_quant node between them; build_graph); the SWA layer keeps the
# node (faster there)
OA_Q_FUSED_IN = ("reuse", "reindex", "full_r2", "full_r1")


def state_tensors(variant: str, ratio: int, csrc: bool, cap: int, dev) -> dict:
    """The layer's inputs, output and state as graph tensors (zeros; set by inputs() or golden_step())."""
    s = {"x": torch.zeros(T, HID, dtype=torch.bfloat16, device=dev), "positions": torch.zeros(T, dtype=torch.int32, device=dev),
         "y": torch.zeros(T, HID, dtype=torch.bfloat16, device=dev),
         "window_cache": torch.zeros(T, W, ROW, dtype=torch.uint8, device=dev),
         "window_loc": torch.zeros(T, dtype=torch.int32, device=dev), "window_rows": torch.zeros(T, W, dtype=torch.int32, device=dev)}
    if variant != "swa":
        s["compress_cache"] = torch.zeros(T, cap, CROW, dtype=torch.uint8, device=dev)
        s["compress_idx"] = torch.full((T, NC), -1, dtype=torch.int32, device=dev)
        # the page table of compress_cache [T][cap][CROW] as one list of rows (a serving layer: the framework's, expanded to
        # pages of 128 rows)
        s["compress_pages"] = (torch.arange(T, device=dev)[:, None] * (cap // 128) +
                               torch.arange(cap // 128, device=dev)).to(torch.int32)
    if variant in ("reindex", "full_r2", "full_r1"):
        s["index_k"] = torch.zeros(T, cap, IROW, dtype=torch.uint8, device=dev)
    if variant == "reindex" or csrc:
        s["candidates"] = torch.zeros(T, cap // 8, dtype=torch.uint8, device=dev)
    if variant.startswith("full") and ratio == 2:
        s["cmp_state"] = torch.zeros(T, ratio, 2, D, dtype=torch.float32, device=dev)   # per token, ring row: kv | score
        s["req_slots"] = torch.arange(T, dtype=torch.int64, device=dev)
    return s


def index_k_pages(rows: torch.Tensor) -> torch.Tensor:
    """The index K cache [T][cap][IROW] by rows -> the same bytes in the kernel's layout: pages of 128 positions, a page's value
    bytes, then their scale bytes (tasks/indexer_common.cuh; SGLang's pool)."""
    n, cap = rows.shape[:2]
    pg = rows.view(n, cap // 128, 128, IROW)
    return torch.cat([pg[..., :64].flatten(2), pg[..., 64:].flatten(2)], -1).view(n, cap, IROW)


def index_k_rows(pages: torch.Tensor) -> torch.Tensor:
    """index_k_pages' inverse."""
    n, cap = pages.shape[:2]
    pg = pages.view(n, cap // 128, 128 * IROW)
    return torch.cat([pg[..., :128 * 64].view(n, cap // 128, 128, 64), pg[..., 128 * 64:].view(n, cap // 128, 128, 4)],
                     -1).view(n, cap, IROW)


def window_tables(w: dict):
    """The window's rows for the positions in w: window_cache [T][W][ROW] as one list of rows, row t W + p % W of position p of
    token t (a serving layer rebinds the tensors to the framework's pool and its tables: the kernel reads only these)."""
    p, t = w["positions"].long(), torch.arange(T, device=w["positions"].device)
    w["window_loc"].copy_(t * W + p % W)
    w["window_rows"].copy_(t[:, None] * W + (p[:, None] - torch.arange(W, device=p.device)) % W)


def build_graph(mpk: StaticMegakernel, w: dict, variant: str, ratio: int, csrc: bool, heads: int):
    """Declare the variant's layer calls on the tensors of w (load_weights and state_tensors). num_gpus > 1: sum_store does the
    all-reduce."""
    compressed, full, reindex = variant != "swa", variant.startswith("full"), variant == "reindex"
    mpk.poll_ns = 0   # the waits spin (no sleep between polls)
    xb = int(mpk.num_gpus > 1)   # sum_store's exchange buffer
    if full or reindex:   # the indexer's tensors need more than config.cuh's default 24 slots (the layer defines STATIC_MAX_BUFS):
        mpk.max_bufs = (36 if full else 30) + xb   # exactly the slots the graph uses (each one more costs every launch)
    elif compressed and xb:
        mpk.max_bufs = 25   # a Reuse layer: config.cuh's default 24 and sum_store's exchange buffer
    NQ, G = heads * D, heads // 8
    t = {}
    a = lambda k: t.setdefault(k, mpk.attach_input(torch_tensor=w[k], name=k))   # one tensor per name: writers and readers
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    x_q, qr_q = nt((T, HID), uint8, "x_q"), nt((T, QL), uint8, "qr_q")
    mpk.act_quant_layer(input=a("x"), output=x_q)                                                   # x -> e4m3, per 32
    wq_a_part = nt((T, QL), float32, "wq_a_part")
    mpk.gemm_fp8_tile_layer(input=x_q, weight=a("wq_a_weight"), weight_scales=a("wq_a_scales"), output=wq_a_part)
    mpk.q_norm_layer(input=wq_a_part, gamma=a("q_norm_weight"), output=qr_q)                       # qr -> e4m3
    wq_b_part = nt((T, NQ), float32, "wq_b_part")                                                  # this GPU's heads
    mpk.gemm_fp8_tile_layer(input=qr_q, weight=a("wq_b_weight"), weight_scales=a("wq_b_scales"), output=wq_b_part,
                            weights_after=x_q)       # its weight loads start after x_q's first task (the start's HBM)
    pos, freqs = a("positions"), a("freqs_cis")
    wkv_part = nt((T, D), float32, "wkv_part")
    mpk.gemm_fp8_tile_layer(input=x_q, weight=a("wkv_weight"), weight_scales=a("wkv_scales"), output=wkv_part)
    mpk.kv_post_layer(input=wkv_part, gamma=a("kv_norm_weight"), positions=pos, freqs=freqs, output=a("window_cache"),
                      window_loc=a("window_loc"))
    cx = {}
    if full:   # the compressor -> latent -> the compressed KV row and the index K row
        cmp_part, latent = nt((T, w["cmp_weight"].shape[0]), float32, "cmp_part"), nt((T, D), bfloat16, "latent")
        mpk.gemm_tile_layer(input=a("x"), weight=a("cmp_weight"), output=cmp_part)
        mpk.compressor_layer(input=cmp_part, gamma=a("cmp_norm_weight"), positions=pos, output=latent, ratio=ratio,
                             state=a("cmp_state") if ratio == 2 else None, req_slots=a("req_slots") if ratio == 2 else None)
        mpk.ckv_post_layer(input=latent, positions=pos, freqs=freqs, output=a("compress_cache"), ratio=ratio,
                           pages=a("compress_pages"))
        wk_part = nt((T, 128), float32, "wk_part")
        mpk.gemm_tile_layer(input=latent, weight=a("wk"), output=wk_part)
        mpk.index_k_post_layer(input=wk_part, gamma=a("k_norm_weight"), positions=pos, freqs=freqs, output=a("index_k"),
                               ratio=ratio, pages=a("compress_pages"))
    if full or reindex:   # the indexer: q, weights, scores, top-k (every GPU computes all of it: no reduction)
        iq_part, q_idx = nt((T, 4096), float32, "iq_part"), nt((T, 4096), bfloat16, "q_idx")
        mpk.gemm_fp8_tile_layer(input=qr_q, weight=a("iwq_b_weight"), weight_scales=a("iwq_b_scales"), output=iq_part)
        mpk.indexer_q_layer(input=iq_part, positions=pos, freqs=freqs, output=q_idx)
        wp_part = nt((T, 128), float32, "wp_part")
        mpk.gemm_tile_layer(input=a("x"), weight=a("wproj"), output=wp_part)
        scores = nt((T, w["index_k"].shape[1]), bfloat16, "scores")
        sms = mpk._num_sms()
        mpk.indexer_score_layer(q=q_idx, weights=wp_part, index_k=a("index_k"), positions=pos, output=scores, ratio=ratio,
                                candidates=a("candidates") if reindex else None, pages=a("compress_pages"),
                                chunks=score_chunks(heads, w["index_k"].shape[1], sms))
        mpk.indexer_topk_layer(scores=scores, positions=pos, output=a("compress_idx"), ratio=ratio,
                               candidates_out=a("candidates") if csrc else None)
    if compressed:
        cx = {"compress_cache": a("compress_cache"), "compress_idx": a("compress_idx"), "compress_pages": a("compress_pages")}
    o_q = nt((T, NQ), uint8, "o_q")
    mpk.sparse_attn_layer(q=wq_b_part, window_cache=a("window_cache"), positions=pos, freqs=freqs, attn_sink=a("attn_sink"),
                          output=o_q, window_rows=a("window_rows"), **cx)   # q from wq_b's K parts (its RoPE done there)
    wo_a_part = nt((T, G * OL), float32, "wo_a_part")
    mpk.gemm_fp8_tile_layer(input=o_q, weight=a("wo_a_weight"), weight_scales=a("wo_a_scales"), output=wo_a_part, groups=G)
    # fused only when each of wo_b's 4 K parts (hand_grids) is at most RING_STAGES / 2 stages: TP4 (K 2048), not TP2 / TP1
    fused = variant in OA_Q_FUSED_IN and G * OL // 4 // 256 <= RING_STAGES // 2
    if fused:   # wo_b's tasks quantize wo_a's K parts themselves (gemm_fp8_tile.cuh fp8_fused_act): no oa_q hop
        oa_q = wo_a_part
        # x_q still lands in "oa_q" (act_quant's layout; build returns it with the node buffers): wo_b's row block 0 copies it
        w.setdefault("oa_q", torch.zeros(zq_bytes(T, G * OL), dtype=torch.uint8, device=w["x"].device))
    else:
        oa_q = nt((T, G * OL), uint8, "oa_q")
        mpk.act_quant_layer(input=wo_a_part, output=oa_q)
    wo_b_part = nt((T, HID), float32, "wo_b_part")
    mpk.gemm_fp8_tile_layer(input=oa_q, weight=a("wo_b_weight"), weight_scales=a("wo_b_scales"), output=wo_b_part,
                            tap=a("oa_q") if fused else None)
    if mpk.num_gpus > 1:   # y = the GPUs' parts added (sum_store's tasks send theirs to every GPU, then add)
        sent = nt((mpk.num_gpus * T, HID), bfloat16, "y_sent")
        mpk.sum_store_layer(input=wo_b_part, output=a("y"), sent=sent)                             # y on every GPU
    else:
        mpk.sum_store_layer(input=wo_b_part, output=a("y"))                                        # this GPU's y


def score_chunks(heads: int, cap: int, sms: int) -> int:
    """indexer_score's tasks a token: chunks of 256 x the fewest halves that fit its tasks in one round on the SMs without a
    sparse_attn task (H < 64: a task per pair) or an indexer_topk task. TP4 at 4096 positions: 2 halves, 9 chunks, 72 tasks on
    100 SMs, instead of 136 in 2 rounds; full_r2 there: 9 chunks of 256."""
    ch, free = cap // 256, sms - T * heads // 16 * (1 + NC // 128 if heads < 64 else 1) - T
    return -(-ch // -(-T * ch // free)) if free > 0 else ch


# the hand plan's grids, by output tensor name
def hand_grids(heads: int, cap: int, cmp_rows: int = 0) -> dict:
    G = heads // 8
    return {"x_q": (T, 2, 1), "wq_a_part": (QL // 128, 4, 1), "qr_q": (T, 1, 1), "wq_b_part": (heads * D // 128, 1, 1),
            "wkv_part": (D // 128, 2, 1), "window_cache": (T, 1, 1), "o_q": (T, heads // 16, 1),
            "wo_a_part": (G * OL // 128, max(2, 64 // (G * OL // 128)), 1), "oa_q": (T, G * OL // 2048, 1),
            "wo_b_part": (HID // 128, 4, 1), "y": (HID // 128, 1, 1), "y_sent": (HID // 128, 1, 1), "cmp_part": (max(cmp_rows, 128) // 128, 4, 1), "latent": (T, 1, 1),
            "compress_cache": (T, 1, 1), "wk_part": (1, 1, 1), "index_k": (T, 1, 1), "iq_part": (4096 // 128, 1, 1),
            "q_idx": (T, 1, 1), "wp_part": (1, 4, 1), "compress_idx": (T, 1, 1)}


# the nodes before sparse_attn, in the order the hand plan deals them (a topological order: the window's kv chain and the
# compressor's before wq_b and the indexer, which wait for q_norm)
BEFORE = ["x_q", "wq_a_part", "wkv_part", "cmp_part", "wp_part", "qr_q", "window_cache", "latent", "compress_cache", "wk_part",
          "index_k", "iq_part", "wq_b_part", "q_idx", "scores", "compress_idx"]


def hand_plan(mpk: StaticMegakernel, heads: int, cap: int, cmp_rows: int, sms: int) -> dict:
    """sparse_attn's tasks first, one on each of SMs 0..; the nodes before it dealt round robin over the other SMs in the order
    BEFORE; the nodes after it round robin over all SMs from the first SM without a sparse_attn task (wo_a's tasks not behind
    sparse_attn's: their weights load while it runs). The grids: hand_grids (wo_a in about 64 tasks, at least 2 K parts: TP4 4,
    so that a task's 1024 K of weights fit its ring before o_q is there; x's act_quant in 2 column slices per token). The same
    lists on every GPU (sum_store_layer with num_gpus > 1 needs that)."""
    nodes = compiler.graph_nodes(mpk)
    names = {n.graph_idx: compiler.tname(mpk, n.outputs[0]) for n in nodes}
    grid = hand_grids(heads, cap, cmp_rows)
    grid["o_q"] = next(n.grid for n in nodes if n.name == "sparse_attn")   # (T, heads / 16, 1 or a task per pair: the layer's)
    grid["scores"] = next((n.grid for n in nodes if n.name == "indexer_score"), None)   # (T, score_chunks: the layer's)
    grids = {idx: grid[name] for idx, name in names.items()}
    tasks = [(idx, (x, y, z)) for idx in sorted(grids) for x in range(grids[idx][0]) for y in range(grids[idx][1])
             for z in range(grids[idx][2])]
    sa = next(idx for idx, name in names.items() if name == "o_q")
    own = [tk for tk in tasks if tk[0] == sa]
    before = sorted((tk for tk in tasks if tk[0] < sa), key=lambda tk: BEFORE.index(names[tk[0]]))
    after = [tk for tk in tasks if tk[0] > sa]
    n = len(own)
    # more than 8 score chunks a token (above 2048 compressed positions): indexer_topk's tasks (one per token) next, each first
    # on an SM of its own, so that its dry pass (the first fetch of its code) runs while the scores are computed
    topk_first = cap > 2048
    first = [tk for tk in before if topk_first and names[tk[0]] == "compress_idx"]
    before = [tk for tk in before if tk not in first]
    m = n + len(first)
    lists = [[own[sm]] if sm < n else [first[sm - n]] if sm < m else before[sm - m::sms - m] for sm in range(sms)]
    for i, tk in enumerate(after):
        lists[(n + i) % sms].append(tk)
    return {"grids": grids, "lists": [copy.deepcopy(lists) for _ in range(mpk.num_gpus)]}


def build(ws: list, variant: str, ratio: int, csrc: bool, heads: int, out: str, plan: str = "hand", search: list = None,
          costs_code: str = "current", time_limit_s: float = 300, profile: bool = False, extra_flags: list = None, device: int = 0,
          free_sms: int = 0):
    """Build the graph on len(ws) GPUs (ws: each GPU's tensors), compile its plan and load the kernel. Return the kernel, each
    GPU's node buffers (by output tensor name) and the information from compile_plan, with "paths" and "mpk". free_sms: SMs left
    out of the launch (no CTA there): a framework's side stream runs on them meanwhile."""
    gpus, w = len(ws), ws[0]
    sms = torch.cuda.get_device_properties(device).multi_processor_count - free_sms
    mpk = StaticMegakernel(num_gpus=gpus, num_sms=sms)
    mpk.reset_parity = any("STATIC_RESET_PARITY" in f for f in (extra_flags or []))
    build_graph(mpk, w, variant, ratio, csrc, heads)
    for wg in ws[1:]:   # the fused act_quant's tap (build_graph made it on GPU 0)
        if "oa_q" in w:
            wg.setdefault("oa_q", torch.zeros_like(w["oa_q"], device=wg["x"].device))
    cap = w["compress_cache"].shape[1] if "compress_cache" in w else 256
    p = hand_plan(mpk, heads, cap, w["cmp_weight"].shape[0] if "cmp_weight" in w else 0, sms) if plan == "hand" \
        else compiler.plan_from_file(plan)
    if len(p["lists"]) == 1 and gpus > 1:   # a plan file of one GPU: the same lists on every GPU
        p = dict(p, lists=[copy.deepcopy(p["lists"][0]) for _ in range(gpus)])
    if search:
        if gpus > 1:
            raise ValueError("--search: one GPU only")
        from mirage.mpk import plan_viewer, search as solver, static_schedule
        code = {"current": static_schedule.code_tag(), "any": None}.get(costs_code, costs_code)
        costs = solver.Costs(search, code=code, gpu=torch.cuda.get_device_name(0))
        p, summary = solver.search_plan(mpk, costs, time_limit_s=time_limit_s, start_from=p)
        os.makedirs(out, exist_ok=True)
        compiler.write_plan_file(os.path.join(out, "plan.json"), p)
        plan_viewer.save(os.path.join(out, "plan_view.json"), p, summary, f"searched from {plan}")
    paths, info = compiler.compile_plan(mpk, p, os.path.join(out, "schedules"))
    nodes = compiler.graph_nodes(mpk)
    # STATIC_NO_START_BARRIER: the host allocates the buffers reset at each launch (two sets)
    nb = any("STATIC_NO_START_BARRIER" in f for f in (extra_flags or []))
    given = lambda n: n.info["out_buffer"] is not None and not n.info["exchange"] and \
        not (nb and n.info["out_buffer"](n)[1] >= 0 and not n.info["out_buffer"](n)[2])
    bufs = [{compiler.tname(mpk, n.outputs[0]): torch.zeros(n.info["out_buffer"](n)[0], dtype=torch.uint8, device=wg["x"].device)
             for n in nodes if given(n)} for wg in ws]
    for wg, bg in zip(ws, bufs):
        if "oa_q" in wg and "oa_q" not in bg:   # fused act_quant: wo_b's x_q tap (build_graph), with the node buffers
            bg["oa_q"] = wg["oa_q"]
    sk = mpk.compile_static(paths, gpu_tensors=[{**bg, **wg} for wg, bg in zip(ws, bufs)], out_dir=os.path.join(out, "build"),
                            extra_flags=extra_flags, profile=profile, device=device)
    return sk, bufs, dict(info, paths=paths, mpk=mpk, plan=p)


def random_inputs(w: dict, variant: str, ratio: int, pos: int, seed: int = 1):
    """x random, every token at position pos, the caches and state filled with valid random values (no NaN encodings)."""
    gen = torch.Generator().manual_seed(seed)
    dev = w["x"].device
    rnd = lambda *shape: torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8).to(dev)
    w["x"].copy_(torch.randn(T, HID, generator=gen).bfloat16())
    w["positions"].fill_(pos)
    window_tables(w)
    e4m3 = rnd(T, W, D) & 0xFE                       # e4m3 bytes, never 0x7F / 0xFF (NaN)
    w["window_cache"].copy_(torch.cat([e4m3, 118 + rnd(T, W, D // 32) % 8], -1))   # scales 2^-9 .. 2^-2
    n = (pos + 1) // max(ratio, 1)
    if "compress_cache" in w:
        cap = w["compress_cache"].shape[1]
        w["compress_cache"].copy_(torch.cat([rnd(T, cap, D // 2), 0x18 + rnd(T, cap, D // 16) % 0x20], -1))   # e4m3 scales
    if "index_k" in w:
        cap = w["index_k"].shape[1]
        w["index_k"].copy_(index_k_pages(torch.cat([rnd(T, cap, 64), 120 + rnd(T, cap, 4) % 8], -1)))
    if "candidates" in w:
        w["candidates"].fill_(1)
    if "cmp_state" in w:
        w["cmp_state"].copy_(torch.randn(*w["cmp_state"].shape, generator=gen))
    if variant == "reuse":   # the source layer's top-k: 512 distinct positions below n, in order
        idx = torch.full((T, NC), -1, dtype=torch.int32)
        for t in range(T):
            k = min(NC, n)
            idx[t, :k] = torch.randperm(n, generator=gen)[:k].sort().values.int()
        w["compress_idx"].copy_(idx)


def from_buffer(buf: torch.Tensor, K: int):
    """act_quant's layout (o_q, oa_q): e4m3 bytes [T, K], then per 128 columns a 512-B scale chunk -> bytes, scales [T, K / 32]."""
    o = buf.cpu()
    return o[:T * K].view(T, K), o[T * K:].view(K // 128, 32, 16)[:, :T, :4].permute(1, 0, 2).reshape(T, K // 32)


def added(buf: torch.Tensor, n: int) -> torch.Tensor:
    """A GEMM's K parts in slots [P][T][n] fp32 -> added in part order [T, n]."""
    s = buf.view(torch.float32).view(-1, T, n)
    out = s[0].clone()
    for i in range(1, s.shape[0]):
        out += s[i]
    return out


def golden_state(w: dict, g, src, pre: str, variant: str, ratio: int, csrc: bool):
    """Set the graph tensors to golden step g's inputs and the state before it: src's tensors named pre + name (the case's
    shared state, pre "", before its first step; the step before's, pre "state_after.")."""
    cat = lambda src, name, row: torch.cat([src.get_tensor(name + (".fp8" if row == D else ".fp4")).view(torch.uint8)
                                            .reshape(T, -1, row), src.get_tensor(name + ".scale").view(torch.uint8)], -1)
    w["x"].copy_(g.get_tensor("x"))
    w["positions"].copy_(g.get_tensor("positions"))
    if "window_loc" in w:   # (not in check_golden's expected state)
        window_tables(w)
    w["window_cache"].copy_(cat(src, pre + "window_kv_cache", D))
    if variant.startswith("full"):
        new = {"compress_cache": cat(src, pre + "compress_kv_cache", D // 2), "index_k": cat(src, pre + "indexer.k_cache", 64)}
        if ratio == 2:
            w["cmp_state"][:, :, 0].copy_(src.get_tensor(pre + "compressor.kv_state"))
            w["cmp_state"][:, :, 1].copy_(src.get_tensor(pre + "compressor.score_state"))
    elif variant in ("reuse", "reindex"):
        new = {"compress_cache": cat(g, "in.compress_kv", D // 2)}
        if variant == "reindex":
            new["index_k"] = cat(g, "in.index_k", 64)
            cb = g.get_tensor("in.candidates")
            nb = (cb.shape[1] + 7) // 8
            w["candidates"].zero_()
            w["candidates"][:, :nb].copy_(torch.nn.functional.pad(cb, (0, nb * 8 - cb.shape[1])).view(T, nb, 8).any(-1).byte())
        else:
            ti = g.get_tensor("topk_idxs")[:, W:]
            ids = torch.full((T, NC), -1, dtype=torch.int32)
            ids[:, :ti.shape[1]] = torch.where(ti >= 0, ti - W, ti)
            w["compress_idx"].copy_(ids)
    else:
        new = {}
    for name, v in new.items():
        w[name].zero_()
        w[name][:, :v.shape[1]].copy_(v)
    if "index_k" in new:
        w["index_k"].copy_(index_k_pages(w["index_k"]))
    if variant != "reuse" and "compress_idx" in w:
        w["compress_idx"].fill_(-7)   # written by indexer_topk: a value it did not write shows
    if csrc:
        w["candidates"].fill_(0xAB)


# the quantized caches the state check compares: per row its value bytes (then its scale bytes), e2m1 values (else e4m3)
CACHES = {"window_cache": (D, False), "compress_cache": (D // 2, True), "index_k": (64, True)}


def cache_diff(got: torch.Tensor, exp: torch.Tensor, nv: int, fp4: bool) -> tuple:
    """A quantized cache (per row nv value bytes, then the scale bytes) against the golden's: (rows, rows equal, values one
    code step off, values further off, rows whose scale bytes differ). A value's code: its e4m3 byte (fp4: e2m1, two a byte)
    as a signed integer in value order; the values are compared in the rows whose scale bytes are equal."""
    got, exp = got.reshape(-1, got.shape[-1]), exp.reshape(-1, exp.shape[-1])
    ne = (got != exp).any(-1)
    g, e = got[ne], exp[ne]
    sdiff = (g[:, nv:] != e[:, nv:]).any(-1)

    def code(b):
        if fp4:
            b = torch.stack([b & 15, b >> 4], -1).flatten(-2)
        sign, mag = (b >> 3, b & 7) if fp4 else (b >> 7, b & 127)
        return torch.where(sign.bool(), -mag.int(), mag.int())

    d = (code(g[~sdiff, :nv]) - code(e[~sdiff, :nv])).abs()
    return got.shape[0], int((~ne).sum()), int((d == 1).sum()), int((d > 1).sum()), int(sdiff.sum())


def check_golden(args, ws: list, shards: list, variant: str, ratio: int, csrc: bool, heads: int):
    """--golden: per case and step, o_q and y against the golden, and the state each shard leaves against the golden's state after
    the step (golden_state with its state_after tensors). One GPU (ws: one): the shards one after another on it, each from the
    step's state; y = bf16(the shards' wo_b K parts added in fp32). --gpus = --tp: one launch per step, shard r on GPU r (the
    all-reduce in the layer): y of GPU 0 (y of every GPU must be equal), also against y_ref = bf16(the shards' wo_b K parts
    added in fp32), which is the one-GPU check's y."""
    multi, w = len(ws) > 1, ws[0]
    cases = args.cases or sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(os.path.join(args.golden, f"L{args.layer}_*/")))
    state = [k for k in ("window_cache", "compress_cache", "index_k", "cmp_state", "candidates", "compress_idx") if k in w]
    stats = {"steps": 0, "o_q heads equal": 0, "o_q heads": 0, "y values within 2^-7 relative": 0, "y values": 0,
             "y relRMS max": 0.0}
    if multi:
        stats.update({"steps with y equal on every GPU": 0, "y values equal to y_ref": 0, "y vs y_ref relRMS max": 0.0})
    cst = {n: [0] * 5 for n in CACHES if n in w}   # cache_diff's counts, added over the steps and shards
    sst = {"values equal": 0, "values": 0, "relRMS max": 0.0}   # the compressor state
    for case in cases:
        src, pre = safe_open(os.path.join(args.golden, case, "shared.safetensors"), "pt"), ""
        nsteps = json.load(open(os.path.join(args.golden, case, "case.json")))["steps"]
        for k in range(nsteps):
            g = safe_open(os.path.join(args.golden, case, f"step{k}.safetensors"), "pt")
            for wg in ws:
                golden_state(wg, g, src, pre, variant, ratio, csrc)
            snapshot = {n: w[n].clone() for n in state}
            exp = {n: w[n].clone() for n in ("x", "positions", *state)}   # the state the step should leave
            golden_state(exp, g, g, "state_after.", variant, ratio, csrc)
            ysum = torch.zeros(T, HID, dtype=torch.float32, device=w["x"].device)   # GPU 0 (a launch leaves another current)
            g_oq = g.get_tensor("o_q").view(torch.uint8).view(T, 64, D)
            g_os = g.get_tensor("o_s").view(torch.uint8).view(T, 64, D // 32)
            if multi:
                args.h.launch(0, timeout_s=60)
            for r in range(len(shards)):
                wg, bg = (ws[r], args.bufs[r]) if multi else (w, args.bufs[0])
                if not multi:
                    for n in state:
                        w[n].copy_(snapshot[n])
                    for n, v in shards[r].items():
                        w[n].copy_(v)
                    args.h.launch(0, timeout_s=60)
                oq, os_ = from_buffer(bg["o_q"], heads * D)
                hs = slice(heads * r, heads * (r + 1))
                eq = (oq.view(T, heads, D) == g_oq[:, hs]).all(-1) & (os_.view(T, heads, D // 32) == g_os[:, hs]).all(-1)
                stats["o_q heads equal"] += int(eq.sum())
                stats["o_q heads"] += eq.numel()
                ysum += added(bg["wo_b_part"], HID).to(ysum.device)
                for n, acc in cst.items():
                    view = index_k_rows if n == "index_k" else (lambda c: c)
                    for i, v in enumerate(cache_diff(view(wg[n].to(exp[n].device)), view(exp[n]), *CACHES[n])):
                        acc[i] += v
                if "cmp_state" in w:
                    got, ref = wg["cmp_state"].to(exp["cmp_state"].device), exp["cmp_state"]
                    sst["values equal"] += int((got == ref).sum())
                    sst["values"] += got.numel()
                    sst["relRMS max"] = max(sst["relRMS max"], ((got - ref).norm() / ref.norm().clamp(min=1e-30)).item())
            y, gold = ysum.bfloat16().float(), g.get_tensor("y").to(ysum.device).float()
            if multi:   # the layer's own all-reduce
                y_ref, y = y, w["y"].float()
                stats["steps with y equal on every GPU"] += int(all(torch.equal(wg["y"].cpu(), w["y"].cpu()) for wg in ws))
                stats["y values equal to y_ref"] += int((y == y_ref).sum())
                stats["y vs y_ref relRMS max"] = max(stats["y vs y_ref relRMS max"], ((y - y_ref).norm() / y_ref.norm()).item())
            near = (gold.abs() * 2.0 ** -7).clamp(min=2.0 ** -133)   # 1 to 2 bf16 steps at the golden's magnitude
            stats["y values within 2^-7 relative"] += int(((y - gold).abs() <= near).sum())
            stats["y values"] += y.numel()
            stats["y relRMS max"] = max(stats["y relRMS max"], ((y - gold).norm() / gold.norm()).item())
            stats["steps"] += 1
            src, pre = g, "state_after."
    gp = f", {len(ws)} GPUs" if multi else ""
    print(f"{variant} L{args.layer} tp {args.tp}{gp}, {len(cases)} cases: " + ", ".join(
        f"{k}: {v:.2e}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()))
    print(f"{variant} L{args.layer} tp {args.tp}{gp}, the state each shard left vs the golden's state after the step: " + "; ".join(
        [f"{n}: rows equal {a[1]} / {a[0]}, values 1 step off {a[2]}, further off {a[3]}, rows with other scales {a[4]}"
         for n, a in cst.items()]
        + ([f"cmp_state: values equal {sst['values equal']} / {sst['values']}, relRMS max {sst['relRMS max']:.2e}"]
           if "cmp_state" in w else [])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="deepseek-ai/DeepSeek-V4.1-Flash",
                    help="a local snapshot, or the repo id (then only the shards of --layer are downloaded)")
    ap.add_argument("--revision", help="with a repo id: a branch, tag or commit (default: main)")
    ap.add_argument("--layer", type=int, default=3, help="a layer of the backbone (0..39); its variant follows from the config")
    ap.add_argument("--tp", type=int, default=4, help="attention TP size: 4 (16 heads per GPU), 2 (32) or 1 (64 heads: DP attention)")
    ap.add_argument("--rank", type=int, default=0, help="the GPU's rank in attention TP (its heads)")
    ap.add_argument("--gpus", type=int, default=1, help="1 (one shard: --rank), or 4 with --tp 4: shard r on GPU r, the "
                    "all-reduce in the layer")
    ap.add_argument("--pos", type=int, default=4096, help="the position of the 8 tokens (random inputs)")
    ap.add_argument("--out", default="build/dsv41_attn")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--plan", default="hand", help="'hand' (hand_plan) or a plan file")
    ap.add_argument("--search", nargs="+", default=None, metavar="COSTS", help="cost record files: the solver's plan, from --plan")
    ap.add_argument("--costs-code", default="current", help="with --search: the records of these kernel templates: current "
                    "(static_schedule.code_tag), any, or a tag")
    ap.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    ap.add_argument("--record", default=None, metavar="FILE", help="the timing build; this run's cost records appended to FILE")
    ap.add_argument("--nvcc-flag", action="append", default=[], help="an nvcc flag for the layer (repeatable), e.g. "
                    "-DSTATIC_RESET_IN_KERNEL (serving mode)")
    ap.add_argument("--cupti", action="store_true", help="also time --reps launches with the torch profiler: the kernel's "
                    "duration in its trace (CUPTI), as SGLang's kernels are timed from their profiler trace")
    ap.add_argument("--save-plan", default=None, metavar="FILE", help="write the plan the layer was built with to FILE")
    ap.add_argument("--golden", help="a golden dir (the official implementation's cases): check the layer against its cases "
                    "of --layer instead")
    ap.add_argument("--cases", nargs="*", default=None, help="with --golden: these cases only")
    args = ap.parse_args()
    if args.golden and any("STATIC_RESET_IN_KERNEL" in f for f in args.nvcc_flag):
        raise ValueError("--golden reads node buffers that a serving build's readers re-arm to 0xFF (sum_store re-arms wo_b's "
                         "K parts): use a host-reset build (no -DSTATIC_RESET_IN_KERNEL)")
    if args.tp not in (1, 2, 4) or not 0 <= args.rank < args.tp:
        raise ValueError(f"--tp {args.tp} --rank {args.rank}: tp 1, 2 or 4 (sparse_attn takes 16 heads a task)")
    if not (args.gpus == 1 or (args.gpus == 4 and args.tp == 4)) or args.gpus > torch.cuda.device_count():
        raise ValueError(f"--gpus {args.gpus} --tp {args.tp}: --gpus 1, or --gpus 4 with --tp 4 (the GPUs visible: "
                         f"{torch.cuda.device_count()})")
    multi = args.gpus > 1
    snapshot = download_layer(args.checkpoint, args.layer, revision=args.revision)
    cfg = model_config(snapshot)
    variant, ratio = variant_of(cfg, args.layer), cfg["compress_ratios"][args.layer]
    csrc, heads = args.layer == cfg["candidate_source_layer"], 64 // args.tp
    read, dev = checkpoint_reader(snapshot), torch.device("cuda", 0)
    if args.golden:   # positions up to the cases' last, the caches as large as theirs need
        steps = [f for c in glob.glob(os.path.join(args.golden, f"L{args.layer}_*/")) for f in glob.glob(os.path.join(c, "step*"))]
        top = max(int(safe_open(f, "pt").get_tensor("positions").max()) for f in steps)
    else:
        top = args.pos
    rows = (top + 2 + 255) // 256 * 256                      # the RoPE table's positions
    # compressed positions (indexer_score: whole 256-position chunks; at least the top-k's NC)
    cap = max(((top + 1) // max(ratio, 1) + 255) // 256 * 256, NC)
    ranks = range(args.tp) if args.golden or multi else [args.rank]
    devs = [torch.device("cuda", r) for r in ranks] if multi else [dev] * len(ranks)
    shards = [layer_tensors(read, cfg, args.layer, args.tp, r, rows, d) for r, d in zip(ranks, devs)]
    split = ("wq_b_weight", "wq_b_scales", "attn_sink", "wo_a_weight", "wo_a_scales", "wo_b_weight", "wo_b_scales")
    if multi:   # GPU r: shard r, its own state
        ws = [{**s, **state_tensors(variant, ratio, csrc, cap, d)} for s, d in zip(shards, devs)]
    else:
        ws = [{**{k: v.clone() for k, v in shards[0].items()}, **state_tensors(variant, ratio, csrc, cap, dev)}]
    w = ws[0]
    sk, bufs, info = build(ws, variant, ratio, csrc, heads, args.out, args.plan, search=args.search, costs_code=args.costs_code,
                           time_limit_s=args.time, profile=args.record is not None, extra_flags=args.nvcc_flag)
    h = Harness(sk)
    gp = f", {args.gpus} GPUs" if multi else ""
    print(f"{variant} L{args.layer}, tp {args.tp} ({heads} heads){gp}, grids {info['grids']}")
    if args.save_plan:
        compiler.write_plan_file(args.save_plan, info["plan"])
    if args.golden:
        args.h, args.bufs = h, bufs
        check_golden(args, ws, [{k: s[k] for k in split} for s in shards], variant, ratio, csrc, heads)
        sk.finalize()
        return
    for wg in ws:   # the same inputs on every GPU (seed 1)
        random_inputs(wg, variant, ratio, args.pos)
    h.launch(timeout_s=60)   # the same inputs each launch: every later launch writes the same cache rows again
    y0 = w["y"].cpu()
    if multi:   # the all-reduce's result: the same bits on every GPU
        diff = [g for g, wg in enumerate(ws) if not torch.equal(wg["y"].cpu().view(torch.int16), y0.view(torch.int16))]
        if diff:
            raise RuntimeError(f"y differs from GPU 0's on GPUs {diff}")
        print(f"y bit-identical on the {len(ws)} GPUs", flush=True)
    spans, events, launches = [], [], []
    serving = bool(h.layout.get("serving"))
    for _ in range(args.reps):
        h.launch(512 << 20, timeout_s=60)     # 512 MB read first: L2 emptied
        spans.append(h.span_us())
        if serving:   # the kernel's time as a server sees it (CUDA events around the launch)
            events.append(max(h.elapsed_us()))
        if args.record:
            launches.append((h.task_times(), h.timing()))
    if multi:   # the timed launches (the exchange region's two sets in turn) give the first launch's y again, on every GPU
        diff = [g for g, wg in enumerate(ws) if not torch.equal(wg["y"].cpu().view(torch.int16), y0.view(torch.int16))]
        if diff:
            raise RuntimeError(f"after {args.reps} launches, y differs from the first launch's on GPUs {diff}")
        print(f"y after {args.reps} launches: bit-identical to the first launch's on the {len(ws)} GPUs", flush=True)
    print(f"position {args.pos}: median span {statistics.median(spans):.1f} us over {args.reps} launches"
          + (f"; serving mode, events: median {statistics.median(events):.1f} us" if serving else ""))
    if args.cupti:   # the same launches timed by CUPTI: the kernel's duration as a profiler trace reports it
        import tempfile
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.reps):
                h.launch(512 << 20, timeout_s=60)
        path = os.path.join(tempfile.mkdtemp(), "trace.json")
        prof.export_chrome_trace(path)
        ks = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel" and "layer_kernel" in e.get("name", "")]
        per = {}   # per GPU, in launch order
        for e in sorted(ks, key=lambda e: e["ts"]):
            per.setdefault(e.get("args", {}).get("device", 0), []).append(e["dur"])
        n = min(len(v) for v in per.values())
        durs = [max(v[i] for v in per.values()) for i in range(n)]   # a launch: its slowest GPU
        print(f"CUPTI layer_kernel duration, {n} launches: median {statistics.median(durs):.1f} us" + (
            f" (the slowest of {len(per)} GPUs: the first launched waits for the others at the start barrier); the fastest GPU "
            f"(the last launched): median {statistics.median(min(v[i] for v in per.values()) for i in range(n)):.1f} us"
            if len(per) > 1 else ""))
    if args.record:
        from mirage.mpk import cost_records
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
        records = cost_records.records_of_run(info["mpk"], info["paths"], launches, os.path.basename(os.path.normpath(args.out)),
                                              machine)
        cost_records.append_records(args.record, records)
        print(f"cost records: {len(records)} added to {args.record}")
    sk.finalize()


if __name__ == "__main__":
    main()
