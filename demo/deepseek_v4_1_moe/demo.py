"""Run one DeepSeek-V4.1-Flash MoE layer, MoE TP4, with the static-schedule compiler.

The layer is the MoE of a decode step (inference/model.py MoE.forward): x [8, 5120] bf16 (the ffn_norm output) -> y [8, 5120]
bf16. Every V4.1 layer has the same MoE: 384 routed experts (MXFP4: e2m1 values, an e8m0 scale per 32), top 6 by sqrt(softplus)
scores, 1 shared expert (e4m3, 32 x 32 block scales), intermediate 2304, SwiGLU clamped at 10. MoE TP divides every expert's
intermediate across the GPUs: with --tp 4 each GPU holds 576 columns. This process controls all GPUs. With --gpus 1 one GPU
computes one rank's part (--rank), not the full layer.

  0. download_layer  --checkpoint: a local snapshot, or deepseek-ai/DeepSeek-V4.1-Flash (then only the shards of --layer)
  1. load_weights    GPU g's tensors: rank g's columns (weights.py: the routed experts packed in swiglu_moe_experts' pieces, the
                     shared expert sliced and padded). x: --x (bf16 [8, 5120] raw file) or random (seed 1), the same on every GPU
  2. build_graph     8 nodes:
                       router              gemm_tile: logits = x . gate.weight^T, fp32 K parts
                       topk_route          the sqrtsoftplus gate: the top 6 of scores + bias, weights = scores / (sum + 1e-20)
                                           x 1.5, a token's pairs sorted by expert id
                       act_quant           x -> x_q: e4m3, a power-of-2 scale per 32 columns
                       gemm_fp8_tile       the shared expert's gate | up, fp32 K parts
                       act_quant           SwiGLU: gate and up rounded to bf16, clamped, h = bf16(silu(gate) up) -> hs_q,
                                           zero-padded to shared down's K (TP4: 576 -> 768)
                       gemm_fp8_tile       shared down, fp32 K parts
                       swiglu_moe_experts  the routed experts, one task per SM: h_q = act_quant(bf16(silu(gate) up)), then
                                           R = weight x (h_q . w2), rows (t, k) [48, 5120] fp32 (the weight multiplies W2's
                                           result, as SGLang's and vLLM's trtllm-gen MoE do)
                       sum_store           y = bf16(the token's 6 R rows, then shared down's parts); with --gpus 4 each GPU
                                           sends that bf16 partial to every GPU and adds the 4 in fp32 in rank order (as
                                           SGLang's all-reduce): y on every GPU
  3. compile_plan    --plan: hand (hand_plan below) or a plan file. The plan files are for one GPU of TP4 on a 148-SM B200 (with
                     --gpus 4 every GPU gets the same lists):
                       hand_plan.json    the hand plan
                       solver_plan.json  search.search_plan on cost records of the hand plan, started from it
                     kernels/<plan>_layer.cu is the layer each one gives (a serving build).
                     --search COSTS: the solver's plan instead, from --plan (search.search_plan with the cost record files
                     COSTS; needs OR-Tools; one GPU) -> <out>/plan.json; --save-plan FILE writes the plan the layer was built with
  4. compile_static  <out>/build/layer.cu, built and loaded (--nvcc-flag=-DSTATIC_RESET_IN_KERNEL: the serving build, whose
                     kernel resets its own buffers)
  5. check_launch    one launch: y the same on all GPUs; --compare-y: y vs a reference y file (bf16 [8, 5120], e.g. SGLang's),
                     relRMS and max |d|. Not in a serving build (its readers re-arm the buffers they read): each GPU's fp32
                     partial (the R rows and shared down's parts as sum_store adds them) vs fp64 from its own h_q and hs_q
                     (relRMS < 1e-3), and y vs the partials as sum_store adds them (one GPU: bf16 of its partial; --gpus 4: the
                     GPUs' bf16 partials added in fp32 in rank order), bit for bit
  6. timed launches  --reps launches, 512 MB read on every GPU before each (empties L2); the span of each (and the CUDA events
                     around the kernel in a serving build). --record FILE: the timing build; a per-node table, and the per-task
                     times appended to FILE as cost records (cost_records.py)
Steps 5-6 run the layer through the test harness (python/mirage/mpk/static_harness.py).

python demo.py --layer 3 --gpus 4 [--checkpoint DIR|REPO] [--plan hand|FILE] [--x x.bin] [--compare-y y.bin]   (TP4 on 4 GPUs)
python demo.py --layer 3 [--rank 0] [--x x.bin] [--compare-y y.bin]   (one GPU: rank 0's part; compare with that rank's part)
python demo.py --layer 3 --plan solver_plan.json --nvcc-flag=-DSTATIC_RESET_IN_KERNEL --reps 50   (serving build)
python demo.py --layer 3 --record costs.json              (cost records of this GPU)
python demo.py --layer 3 --search costs.json              (the solver's plan from these records)
"""
import argparse
import copy
import json
import os
import statistics

import numpy as np
import torch
from mirage.core import bfloat16, float32, int64, uint8
from mirage.mpk import compiler
from mirage.mpk.static_harness import Harness
from mirage.mpk.static_megakernel import StaticMegakernel

try:
    from .weights import checkpoint_reader, download_layer, layer_tensors, model_config, routed_tensors, unpack_pieces
except ImportError:   # as a script
    from weights import checkpoint_reader, download_layer, layer_tensors, model_config, routed_tensors, unpack_pieces

T, HID, E, K, INTER = 8, 5120, 384, 6, 2304   # tokens, dim, routed experts, experts per token, expert intermediate
ROUTE_SCALE, SWIGLU_LIMIT = 1.5, 10.0          # model.py Gate.forward's route_scale; Expert's clamp
ROUTER_PARTS = 8   # the router's K parts: 5 K tiles of 128 each, a task's K fits the ring's 5 stages
HQ, HQ_SF = "swiglu_moe_experts.h_q", "swiglu_moe_experts.h_q_scales"   # the expert node's scratch (given: the check reads h_q)


def build_graph(mpk: StaticMegakernel, w: dict) -> None:
    """Declare the layer calls on the tensors of w (load_weights and the inputs)."""
    t = {}
    a = lambda k: t.setdefault(k, mpk.attach_input(torch_tensor=w[k], name=k))   # one tensor per name: writers and readers
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    I, KO = w["sgu_weight"].shape[0] // 2, w["sd_weight"].shape[1]
    logit_part, pairs = nt((T, E), float32, "logit_part"), nt((T, K), int64, "pairs")
    mpk.gemm_tile_layer(input=a("x"), weight=a("gate_weight"), output=logit_part)                    # logits, fp32 K parts
    mpk.topk_route_layer(input=logit_part, bias=a("gate_bias"), output=pairs, routed_scaling_factor=ROUTE_SCALE,
                         score_func="sqrtsoftplus")                                                 # top 6, sorted by id
    x_q = nt((T, HID), uint8, "x_q")
    mpk.act_quant_layer(input=a("x"), output=x_q)                                                   # x -> e4m3, per 32
    sgu_part = nt((T, 2 * I), float32, "sgu_part")
    mpk.gemm_fp8_tile_layer(input=x_q, weight=a("sgu_weight"), weight_scales=a("sgu_scales"), output=sgu_part)
    hs_q = nt((T, KO), uint8, "hs_q")
    mpk.act_quant_layer(input=sgu_part, output=hs_q, swiglu_limit=SWIGLU_LIMIT)                     # SwiGLU -> e4m3
    sd_part = nt((T, HID), float32, "sd_part")
    mpk.gemm_fp8_tile_layer(input=hs_q, weight=a("sd_weight"), weight_scales=a("sd_scales"), output=sd_part)
    routed = nt((T * K, HID), float32, "routed_rows")
    mpk.swiglu_moe_experts_layer(x_q=x_q, pairs=pairs, w13=a("w13"), w13_scales=a("w13_scales"), w2=a("w2"),
                                 w2_scales=a("w2_scales"), output=routed, swiglu_limit=SWIGLU_LIMIT)      # the routed experts
    if "streams" in w and "h_comb" not in w:     # the sublayer's end in the kernel: the sums land in mHC's post (StaticMegakernel.dsv41_mhc_nodes)
        sent = nt((mpk.num_gpus * T, HID), bfloat16, "y_sent")
        mpk.sum_store_layer(input=sd_part, routed=routed, output=None, sent=sent)
        mpk.dsv41_mhc_nodes(a, w, sent)
    elif mpk.num_gpus > 1:   # y = the GPUs' parts added (each sum_store task sends its bf16 sums to every GPU, then adds them)
        sent = nt((mpk.num_gpus * T, HID), bfloat16, "y_sent")
        mpk.sum_store_layer(input=sd_part, routed=routed, output=a("y"), sent=sent)                 # y on every GPU
    else:
        mpk.sum_store_layer(input=sd_part, routed=routed, output=a("y"))                            # y = routed + shared
    if "h_comb" in w:      # only mHC's coefficients beside the sublayer (the framework runs the all-reduce and the post)
        mpk.dsv41_mhc_stats_nodes(a, w)
    if "eg_wkv" in w or "eg_kv" in w:   # (11-20) the next layer's Engram on this post's streams (engram_nodes)
        engram_nodes(mpk, a, w)
    if "head_w" in w:      # (11-20, DSV41_HEAD=2) the last layer: the LM head on its post's next input (head_nodes)
        mb = mpk.max_bufs
        mpk.max_bufs = 64
        head_nodes(mpk, a, w, "x_out")
        mpk.max_bufs = max(mb, len(mpk._bufs))


EG_KP = int(os.environ.get("DSV41_EG_KP", 3))       # the wkv GEMM's K parts (6144 = 24 pieces of 256)
EG_SEND = int(os.environ.get("DSV41_EG_SEND", 25))  # the gathering sends' tasks


HEAD_KP = int(os.environ.get("DSV41_HEAD_KP", 1))        # (head_nodes) the GEMM's K parts (5120 = 40 pieces of 128)
HEAD_SEND = int(os.environ.get("DSV41_HEAD_SEND", 148))  # the sends' tasks
HEAD_GQ = int(os.environ.get("DSV41_HEAD_GQ", 16))       # the gather's tasks a token


def head_nodes(mpk: StaticMegakernel, a, w: dict, x: str = "x") -> None:
    """(11-20) DeepSeek-V4.1's LM head (deepseek_v4_1_sglang/head_demo.py): a(x) [T, 5120] bf16 (the final norm of the streams'
    collapse) -> gemm_tile x . head_w^T (this GPU's vocab rows, padded to whole 128-row blocks; fp32 K parts in slots) ->
    allreduce_send (each GPU's row, bf16) -> gather_f32 -> a("logits") [T, vocab] fp32 on every GPU (with head_ids in w, also
    each token's greedy next token: a("head_ids") [T] int32, as torch.argmax of its logits)."""
    from mirage.core import bfloat16 as bf16, float32 as f32
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    Vp = w["head_w"].shape[0]
    if x == "x_out":   # the post writes x through a buffer slot: the GEMM waits for the post (its output streams_out)
        compiler.PRODUCER_ALIAS[a("x_out").guid] = a("streams_out").guid
    part = nt((T, Vp), f32, "head_part")
    mpk.gemm_tile_layer(input=a(x), weight=a("head_w"), output=part)
    sent = nt((T, Vp), bf16, "head_sent")
    mpk.allreduce_send_layer(input=part, output=sent, num_tasks=HEAD_SEND)
    mpk.gather_f32_layer(input=sent, output=a("logits"), parts=HEAD_GQ, ids=a("head_ids") if "head_ids" in w else None)


EG_ROWS_STRIDE = 48   # (11-20, DSV41_EG_ROWS=1) the hash ids' row stride: [T, Engram layers 2, columns 24] int64


def engram_kv_nodes(mpk: StaticMegakernel, a, w: dict):
    """(11-20) Engram's kv: eg_emb (bf16 [T, 6144]) -> act_quant -> the wkv GEMM on this GPU's W columns (EG_KP K parts) ->
    allreduce_send: into the exchange region (returns its output, for a gate in this kernel), or with eg_kv in w
    (DSV41_ENGRAM=2: an attention kernel before the gate's MoE kernel) into the graph tensor eg_kv [GPUs T, W] through
    its multicast address eg_kv_mc (returns None). With eg_tw in w (DSV41_EG_ROWS=1): eg_emb is looked up in this kernel
    (engram_rows: eg_ids, the table's rows eg_tw and scales eg_ts, into eg_emb by its multicast address eg_emb_mc), the
    act_quant polls it."""
    from mirage.core import bfloat16 as bf16, float32 as f32, uint8 as u8
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    W, KE = w["eg_wkv"].shape[0], w["eg_emb"].shape[1]
    eg_q = nt((T, KE), u8, "eg_q")
    if "eg_tw" in w:
        peer = w["eg_tw"].shape[0] if w["eg_tw"].dtype == torch.int64 else 0   # (SGLang's PEER_READ layout: w_bias [P])
        mpk.engram_rows_layer(ids=a("eg_ids"), weight=a("eg_tw"), scale=a("eg_ts"), output=a("eg_emb"), output_mc=a("eg_emb_mc"),
                              starts=a("eg_starts"), ids_stride=EG_ROWS_STRIDE, peer=peer)
    mpk.act_quant_layer(input=a("eg_emb"), output=eg_q, poll="eg_tw" in w)
    eg_part = nt((T, W), f32, "eg_part")
    mpk.gemm_fp8_tile_layer(input=eg_q, weight=a("eg_wkv"), weight_scales=a("eg_wkv_scales"), output=eg_part)
    if "eg_kv" in w:
        mpk.allreduce_send_layer(input=eg_part, output=a("eg_kv"), num_tasks=EG_SEND, dest_mc=a("eg_kv_mc"))
        return None
    eg_sent = nt((T, W), bf16, "eg_sent")   # (its buffer: [GPUs][T][W] in the exchange region)
    mpk.allreduce_send_layer(input=eg_part, output=eg_sent, num_tasks=EG_SEND)
    return eg_sent


def engram_nodes(mpk: StaticMegakernel, a, w: dict) -> None:
    """(11-20) DeepSeek-V4.1's Engram of the next layer (inference/model.py Engram.forward), in this MoE kernel after its mHC
    post: eg_emb (the layer's n-gram rows, bf16 [T, 6144], SGLang's gather) -> act_quant -> the wkv GEMM on this GPU's W
    columns of the 25600 (TP: W = 25600 / GPUs, the whole K: each column as the official computes it, in EG_KP K parts) ->
    allreduce_send (each GPU's columns to every GPU: kv gathered) -> engram_gate (token t: R' of the post and kv -> R'' =
    streams_eg). SGLang then reads R'' instead of running its Engram (wkv replicated: 157 MB a GPU)."""
    mpk.max_bufs = 64   # (until the slots are known, below)
    # DSV41_ENGRAM=2: kv gathered by the attention kernel before this one (eg_kv, a graph tensor), only the gate here
    kv = engram_kv_nodes(mpk, a, w) if "eg_wkv" in w else a("eg_kv")
    nx = {"pre": a("h_pre"), "gamma": a("eg_gamma"), "x": a("x_out")} if "eg_gamma" in w else {}   # (DSV41_EG_X=1)
    mpk.engram_gate_layer(kv=kv, streams=a("streams_out"), weight=a("eg_qk"), output=a("streams_eg"), **nx)
    mpk.max_bufs = len(mpk._bufs)


def hand_plan(mpk: StaticMegakernel, sms: int) -> dict:
    """One task per SM. The router in 3 row blocks x ROUTER_PARTS K parts; shared gate_up in about 36 tasks (its row blocks x K
    parts: TP4 9 x 4, 5 stages of 256 each, a task's K fits the ring); shared down one K part (TP4: 3 stages). topk_route's and
    the SwiGLU act_quant's tasks first, on SMs of their own (their dry pass runs while their inputs are computed), then x's
    act_quant, the router, gate_up, shared down. The routed experts on every SM (task x on SM x), then sum_store's tasks (one
    per 128 columns), one on each of SMs 0, 1, ..."""
    nodes = compiler.graph_nodes(mpk)
    names = {n.graph_idx: compiler.tname(mpk, n.outputs[0]) for n in nodes}
    blocks = next(n.inputs[1].dim(0) for n in nodes if compiler.tname(mpk, n.outputs[0]) == "sgu_part") // 128
    grid = {"logit_part": (E // 128, ROUTER_PARTS, 1), "pairs": (T, 1, 1), "x_q": (T, 1, 1), "sgu_part": (blocks, 36 // blocks, 1),
            "hs_q": (T, 1, 1), "sd_part": (HID // 128, 1, 1), "routed_rows": (sms, 1, 1), "y": (HID // 128, 1, 1),
            "y_sent": (HID // 128, 1, 1)}
    grid.update({names[n.graph_idx]: n.grid for n in nodes if n.name in ("hc_coef", "hc_post_combine")})
    grid["hc_mix"] = (1, 32, 1)   # the mixes as a gemm_tile: K 20480 in 32 parts of 5 K tiles
    eg = "eg_part" in names.values()   # (11-20) Engram's nodes (engram_nodes)
    if eg:
        egw = next(n.inputs[1].dim(0) for n in nodes if compiler.tname(mpk, n.outputs[0]) == "eg_part")
        grid.update({"eg_q": (T, 1, 1), "eg_part": (egw // 128, EG_KP, 1), "eg_sent": (EG_SEND, 1, 1)})
    if "streams_eg" in names.values():
        grid["streams_eg"] = (4, T, 1)
    hd = "head_part" in names.values()   # (11-20, DSV41_HEAD=2) the LM head's nodes (head_nodes)
    if hd:
        vp = next(n.inputs[1].dim(0) for n in nodes if compiler.tname(mpk, n.outputs[0]) == "head_part")
        grid.update({"head_part": (vp // 128, HEAD_KP, 1), "head_sent": (HEAD_SEND, 1, 1), "logits": (HEAD_GQ, T, 1),
                     "logits_am": (HEAD_GQ, T, 1)})   # (logits_am: the gather with the argmax)
    grids = {idx: grid[name] for idx, name in names.items()}
    tasks = lambda name: [(idx, (x, y, 0)) for idx in grids if names[idx] == name
                          for x in range(grids[idx][0]) for y in range(grids[idx][1])]
    order = [tk for name in ("pairs", "hs_q", "x_q", "logit_part", "sgu_part", "sd_part") for tk in tasks(name)]
    # more tasks than SMs (STATIC_FREE_SMS_MOE): the last ones behind the first short tasks that do not wait for them (x_q's, the router's)
    assert len(order) <= sms + (sms - 16), f"{len(order)} tasks, {sms} SMs"
    ex = next(idx for idx, name in names.items() if name == "routed_rows")
    ys = tasks("y_sent" if mpk.num_gpus > 1 else "y")
    lists = [([order[sm]] if sm < len(order) else []) for sm in range(sms)]
    for i, tk in enumerate(order[sms:]):
        lists[16 + i].append(tk)
    # mHC's coefficients before the experts' queue opens; the sublayer's end after sum_store's tasks
    idle = max(sms - len(order), 1)   # the SMs with no task before the experts: the mixes run there first (their input is ready)
    for i, tk in enumerate(tasks("hc_mix")):
        lists[len(order) % sms + i % idle].append(tk)
    for i, tk in enumerate(tasks("hc_coef")):
        lists[sms - 1 - i].append(tk)
    lists = [lists[sm] + [(ex, (sm, 0, 0))] + ys[sm:sm + 1] for sm in range(sms)]
    for i, tk in enumerate(tasks("streams_out")):
        lists[(len(ys) + i) % sms].append(tk)
    if eg or "streams_eg" in names.values():   # (11-20) Engram: eg_emb's act_quant before the experts on SMs with no front task of their own (their experts wait
        # for the router anyway); the wkv GEMM after the experts, first on the SMs with no sum_store or post task (then the
        # others, behind theirs); the sends after the GEMM tasks, the gate last (it needs the post and every GPU's kv)
        busy = len(ys) + len(tasks("streams_out"))
        for i, tk in enumerate(tasks("eg_q")):
            sm = (len(order) + i) % sms
            lists[sm].insert(lists[sm].index((ex, (sm, 0, 0))), tk)
        free = list(range(busy, sms))   # the SMs with nothing after their expert task
        for i, tk in enumerate(tasks("eg_part")):
            lists[free[i % len(free)]].append(tk)
        for i, tk in enumerate(tasks("eg_sent")):
            lists[free[i % len(free)]].append(tk)
        posts = sorted({sm for sm in range(sms) for tk in lists[sm] if tk in set(tasks("streams_out"))})
        # the gate: with the wkv GEMM here (DSV41_ENGRAM=1) behind a post task (the free SMs run the GEMM, which it waits for);
        # kv from the attention kernel (DSV41_ENGRAM=2) on free SMs behind their expert task: its dry pass (its code's first
        # fetch) runs while the post is still on its way
        gsms = posts if eg else free
        for i, tk in enumerate(tasks("streams_eg")):
            lists[gsms[i % len(gsms)]].append(tk)
    if hd:   # the head behind everything else: its GEMM tasks first on the SMs free after their expert task (their weights
        # stream while the post computes x: build with -DGEMM_RELAID_PREFILL), then over all SMs in turn; the sends, the gathers
        busy = len(ys) + len(tasks("streams_out"))
        turn = list(range(busy, sms)) + list(range(busy))
        k = 0
        for name in ("head_part", "head_sent", "logits", "logits_am"):
            for tk in tasks(name):
                lists[turn[k % sms]].append(tk)
                k += 1
    return {"grids": grids, "lists": [copy.deepcopy(lists) for _ in range(mpk.num_gpus)]}   # the same plan on every GPU


def x_flag_modes(mpk: StaticMegakernel, prefill: bool = False) -> dict:
    """H1 stage A (docs/h1_design.md; build with -DSTATIC_X_FLAG): the kernel loop's mode per node. The front (the readers of x
    and the nodes that read only weights, x and this kernel's own outputs) polls the attention post's flag ("poll"): it starts
    when x is stored instead of when the attention's grid has ended (each CTA polls once); every other node waits for the grid
    first ("wait": the streams' readers, sum_store, the post). prefill (DSV41_X_FLAG=2, a separate switch, the user 11-19):
    only x's readers poll, the other front nodes start as their CTA enters (their weights load during the attention's end)."""
    nodes = compiler.graph_nodes(mpk)
    names = {n.graph_idx: compiler.tname(mpk, n.outputs[0]) for n in nodes}
    readers = {n.graph_idx for n in nodes if any(compiler.tname(mpk, t) == "x" for t in n.inputs)}
    if sorted(names[i] for i in readers) != ["logit_part", "x_q"]:
        raise ValueError(f"x_flag_modes: x's readers are {sorted(names[i] for i in readers)}, not the router and x_q")
    safe = {"pairs", "sgu_part", "hs_q", "sd_part", "routed_rows"}
    return {i: "poll" if i in readers or (nm in safe and not prefill) else "wait" for i, nm in names.items()
            if i in readers or nm not in safe or not prefill}


def build(ws: list, out: str, profile: bool = False, extra_flags: list = None, plan: str = "hand", search: list = None,
          costs_code: str = "current", time_limit_s: float = 300, device: int = 0, num_gpus: int = None, rank: int = None,
          exchange_pool=None, x_flag: bool = False, x_flag_prefill: bool = False):
    """Build the graph on len(ws) GPUs (ws: each GPU's tensors), compile its plan (plan "hand": hand_plan; else a plan file;
    search: the solver's plan from these cost record files, starting from that plan -> <out>/plan.json) and load the kernel.
    Return the kernel, each GPU's node buffers (by output tensor name) and the information from compile_plan, with "paths",
    "mpk" and "plan"."""
    gpus, w = num_gpus or len(ws), ws[0]   # (num_gpus, rank, exchange_pool: this process is GPU rank of a layer on num_gpus)
    # STATIC_FREE_SMS_MOE: SMs left out of the launch (no CTA there): a framework's side stream runs on them meanwhile
    sms = torch.cuda.get_device_properties(0).multi_processor_count - int(os.environ.get("STATIC_FREE_SMS_MOE", 0))
    mpk = StaticMegakernel(num_gpus=gpus, num_sms=sms)
    mpk.moe_rows_poll = "-DMOE_ROWS_POLL" in (extra_flags or [])   # (C12, 11-18: sum_store polls the routed rows; default off)
    build_graph(mpk, w)
    if x_flag:   # H1 stage A: x's flag and the deferred wait (x_flag_modes)
        mpk.x_flag_modes = x_flag_modes(mpk, x_flag_prefill)
    p = hand_plan(mpk, sms) if plan == "hand" \
        else compiler.plan_from_file(plan)
    if len(p["lists"]) == 1 and gpus > 1:   # a plan file of one GPU: the same lists on every GPU
        p = dict(p, lists=[copy.deepcopy(p["lists"][0]) for _ in range(gpus)])
    if search and gpus > 1:
        raise ValueError("--search: one GPU only")
    if search:
        from mirage.mpk import plan_viewer, search as solver, static_schedule
        code = {"current": static_schedule.code_tag(), "any": None}.get(costs_code, costs_code)
        costs = solver.Costs(search, code=code, gpu=torch.cuda.get_device_name(0))
        p, summary = solver.search_plan(mpk, costs, time_limit_s=time_limit_s, start_from=p)
        os.makedirs(out, exist_ok=True)
        compiler.write_plan_file(os.path.join(out, "plan.json"), p)
        plan_viewer.save(os.path.join(out, "plan_view.json"), p, summary, f"searched from {plan}")
    paths, info = compiler.compile_plan(mpk, p, os.path.join(out, "schedules"))
    nodes = compiler.graph_nodes(mpk)
    ex = next(n for n in nodes if n.name == "swiglu_moe_experts")
    bufs = []
    for wg in ws:   # each GPU's node buffers (not the exchange region's) and the expert node's scratch, on that GPU
        d = wg["x"].device
        # STATIC_NO_START_BARRIER: the host allocates the buffers reset at each launch (two sets)
        nb = any("STATIC_NO_START_BARRIER" in f for f in (extra_flags or []))
        bg = {compiler.tname(mpk, n.outputs[0]): torch.zeros(n.info["out_buffer"](n)[0], dtype=torch.uint8, device=d)
              for n in nodes if n.info["out_buffer"] is not None and not n.info["exchange"] and
              not (nb and n.info["out_buffer"](n)[1] >= 0 and not n.info["out_buffer"](n)[2])}
        for slot, name in zip(ex.info["buf_slots"], (HQ, HQ_SF)):   # zeros, as the host allocates them (h_q's padding stays 0)
            bg[name] = torch.zeros(mpk._bufs[slot]["fill"](ex)[0], dtype=torch.uint8, device=d)
        bufs.append(bg)
    try:
        sk = mpk.compile_static(paths, gpu_tensors=[{**wg, **bg} for wg, bg in zip(ws, bufs)], out_dir=os.path.join(out, "build"),
                                extra_flags=extra_flags, profile=profile, device=device,
                                **({} if rank is None else {"rank": rank, "exchange": exchange_pool.take(mpk.exchange_region_bytes())}))
    finally:   # (head_nodes' alias holds for this graph only: guids are per process, other graphs must not see it)
        compiler.PRODUCER_ALIAS.clear()
    # every GPU's init stores (host.cuh: the exchange region's zero-fill and the buffers' first values, cudaMemset on the
    # legacy stream, asynchronous) land before any launch: at the first launch the serving-mode harness's align kernel
    # (static_host_prelude) stores into the other GPUs' hello slots, and a zero-fill still queued on a GPU would wipe the
    # hello (that GPU's align then waits for ever and its layer never starts; seen 11-18 with and without
    # -DSTATIC_NO_START_BARRIER)
    for wg in ws:
        torch.cuda.synchronize(wg["x"].device)
    print(f"grids {info['grids']}")
    return sk, bufs, dict(info, paths=paths, mpk=mpk, plan=p)


# ---- reading the node buffers, the reference ----

def pairs_of(bufs: dict):
    """topk_route's pairs (expert << 32 | weight bits) -> ids [T, K] int32, weights [T, K] fp32 (CPU)."""
    words = bufs["pairs"].cpu().numpy().view(np.uint32).reshape(T, K, 2)   # little-endian: [weight bits, expert]
    return torch.from_numpy(words[..., 1].astype(np.int32)), torch.from_numpy(words[..., 0].copy().view(np.float32))


def from_buffer(buf: torch.Tensor, n: int):
    """act_quant's layout: e4m3 bytes [T, n], then per 128 columns a 512-B scale chunk -> bytes [T, n], scale bytes [T, n / 32]
    (CPU)."""
    o = buf.cpu()
    return o[:T * n].view(T, n), o[T * n:].view(n // 128, 32, 16)[:, :T, :4].permute(1, 0, 2).reshape(T, n // 32)


def w8_dequant(wb: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """An e4m3 weight [N, K] (bytes) and its scale chunks [N / 128, K / 128, 512] (w8_scale_chunks: the scale of row r and K
    columns 32 j .. + 31 of a tile at byte 16 (r % 32) + 4 (r // 32) + j) -> the weight's values, fp64 [N, K] (on wb's device)."""
    N, Kw = wb.shape
    n, k32 = torch.arange(N, device=wb.device), torch.arange(Kw // 32, device=wb.device)
    byte = 16 * (n % 32)[:, None] + 4 * ((n % 128) // 32)[:, None] + (k32 % 4)[None, :]
    sc = chunks[(n // 128)[:, None], (k32 // 4)[None, :], byte].double() - 127
    return wb.view(torch.float8_e4m3fn).double() * torch.exp2(sc).repeat_interleave(32, 1)


def relrms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm()).item()


def partial_reference(bufs: dict, w: dict, I: int) -> torch.Tensor:
    """This GPU's partial of y in fp64 from its own h_q and hs_q (what sum_store adds, without its fp32 roundings): per token
    its K routed rows (dequantized h_q x the packed w2 of the expert) and shared down (hs_q x its w2 columns) [T, HID]."""
    IRP, KT2, OT2, KO = -(-I // 128) * 128, -(-I // 128), HID // 128, w["sd_weight"].shape[1]
    dev = w["x"].device
    ids, wts = pairs_of(bufs)
    hq_all, hsf_all = bufs[HQ].view(T * K * T, IRP), bufs[HQ_SF].view(T * K, KT2, 32, 16)
    ref = torch.zeros(T, HID, dtype=torch.float64, device=dev)
    for si, e in enumerate(torch.unique(ids).tolist()):   # build_table's slots: the experts used, in ascending id
        W2 = unpack_pieces(w["w2"][e].view(OT2, KT2, 128, 64), w["w2_scales"][e].view(OT2, KT2, 512))
        W2 = W2.transpose(1, 2).reshape(HID, IRP)
        for t, k in (ids == e).nonzero().tolist():
            q, sc = hq_all[si * T + t], hsf_all[si, :, t, :4].reshape(-1)
            ref[t] += float(wts[t, k]) * (W2 @ (q.view(torch.float8_e4m3fn).double() *
                                                torch.exp2(sc.double() - 127).repeat_interleave(32)))
    q, sc = from_buffer(bufs["hs_q"], KO)
    hs = (q.view(torch.float8_e4m3fn).double() * torch.exp2(sc.double() - 127).repeat_interleave(32, 1)).to(dev)
    return ref + hs @ w8_dequant(w["sd_weight"], w["sd_scales"]).T


def node_table(info: dict, launches: list) -> None:
    """--record: per node, the median over its tasks and the launches of a task's own time (cost_records.launch_durations: its end
    minus the later of the previous task's end on its SM and its inputs' latest end); the expert node, whose SMs share a queue: its
    work per launch (the sum over the SMs from the queue's opening) and that per SM."""
    from mirage.mpk.compiler import producer_of
    from mirage.mpk.cost_records import launch_durations
    mpk = info["mpk"]
    schedules = [json.load(open(p)) for p in info["paths"]]
    nodes = {n.graph_idx: n for n in compiler.graph_nodes(mpk)}
    pool = next(k for k, n in nodes.items() if n.info["one_task_per_sm"])
    gating = {q.graph_idx for q in (producer_of(t, list(nodes.values())) for t in nodes[pool].inputs) if q}
    per, work = {}, []
    for times, timing in launches:
        placed, w, *_ = launch_durations(schedules, times, [b for b, _ in timing], pool, gating)
        work.append(w[0])
        for t, us in placed[0].items():
            per.setdefault(schedules[0]["all_tasks"][t]["node"], []).append(us)
    print(f"per node, timing build, {len(launches)} launches (a task's time excludes its wait for its inputs):")
    for k in sorted(nodes):
        n = nodes[k]
        name = compiler.tname(mpk, n.outputs[0])
        if k == pool:
            wk = statistics.median(work)
            print(f"  {n.name:<20} {name:<12} {n.grid[0]} SMs share a queue: {wk:.0f} us of work per launch, {wk / n.grid[0]:.1f} per SM")
        elif k in per:
            print(f"  {n.name:<20} {name:<12} grid {str(tuple(n.grid)):<12} {statistics.median(per[k]):5.1f} us (median task)")


def read_bf16(path: str) -> torch.Tensor:
    """A bf16 [T, HID] tensor from a raw file (CPU)."""
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).reshape(T, HID)).view(torch.bfloat16)


def gpu_partial(bufs: dict) -> torch.Tensor:
    """This GPU's fp32 partial of y, in sum_store's order: a token's K routed rows in order, then shared down's K parts."""
    R = bufs["routed_rows"].view(torch.float32).view(T, K, HID)
    sd = bufs["sd_part"].view(torch.float32).view(-1, T, HID)
    p = R[:, 0].clone()
    for k in range(1, K):
        p += R[:, k]
    for part in sd:
        p += part
    return p


def check_launch(ws: list, h: Harness, bufs: list, serving: bool, compare_y: str = None) -> bool:
    """Step 5 (the module comment). Returns whether the checks pass (y vs --compare-y is reported, not judged)."""
    h.launch(timeout_s=60)
    y = ws[0]["y"].cpu()
    ok = True
    if len(ws) > 1:
        same = sum(torch.equal(wg["y"].cpu().view(torch.int16), y.view(torch.int16)) for wg in ws[1:])
        ok &= same == len(ws) - 1
        print(f"y bit-identical to GPU 0's on {same}/{len(ws) - 1} other GPUs")
    if compare_y:
        ref = read_bf16(compare_y)
        print(f"y vs {compare_y}: relRMS {relrms(y, ref):.2e}, max |d| {(y.double() - ref.double()).abs().max().item():.2e}")
    if serving:
        print("a serving build: the partial checks are skipped (its readers re-arm the buffers they read)")
    else:
        I = ws[0]["sgu_weight"].shape[0] // 2
        parts = [gpu_partial(b) for b in bufs]
        rel = [relrms(p, partial_reference(b, wg, I)) for p, b, wg in zip(parts, bufs, ws)]
        acc = parts[0].bfloat16().float().cpu()   # the exchange: each GPU's partial sent as bf16, added in fp32 in rank order
        for p in parts[1:]:
            acc += p.bfloat16().float().cpu()
        exact = torch.equal(acc.bfloat16().view(torch.int16), y.view(torch.int16))
        ok &= max(rel) < 1e-3 and exact
        print(f"each GPU's fp32 partial vs fp64 from its own h_q and hs_q: relRMS {', '.join(f'{r:.1e}' for r in rel)} (< 1e-3) | "
              f"y vs the partials as sum_store adds them: {'bit-identical' if exact else 'DIFFERENT'}")
    print(f"check {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="deepseek-ai/DeepSeek-V4.1-Flash",
                    help="a local snapshot, or the repo id (then only the shards of --layer are downloaded)")
    ap.add_argument("--revision", help="with a repo id: a branch, tag or commit (default: main)")
    ap.add_argument("--layer", type=int, default=0, help="a layer of the backbone (0..39)")
    ap.add_argument("--gpus", type=int, default=1, help="1 (one rank's part: --rank), or 4 with --tp 4: rank g on GPU g, the "
                    "all-reduce in the layer (sum_store's exchange)")
    ap.add_argument("--tp", type=int, default=4, help="MoE TP size: 4 (576 intermediate columns per GPU); with --gpus 1 also 2 or 1")
    ap.add_argument("--rank", type=int, default=0, help="with --gpus 1: the rank whose columns the GPU holds")
    ap.add_argument("--x", help="the input x, bf16 [8, 5120] raw file (default: random, seed 1)")
    ap.add_argument("--compare-y", help="a reference y, bf16 [8, 5120] raw file (with --gpus 1: the same rank's part)")
    ap.add_argument("--out", default="build/dsv41_moe")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--plan", default="hand", help="'hand' (hand_plan) or a plan file")
    ap.add_argument("--search", nargs="+", default=None, metavar="COSTS", help="cost record files: the solver's plan, from --plan")
    ap.add_argument("--costs-code", default="current", help="with --search: the records of these kernel templates: current "
                    "(static_schedule.code_tag), any, or a tag")
    ap.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    ap.add_argument("--record", default=None, metavar="FILE", help="the timing build; this run's cost records appended to FILE, "
                    "and a per-node table")
    ap.add_argument("--save-plan", default=None, metavar="FILE", help="write the plan the layer was built with to FILE")
    ap.add_argument("--nvcc-flag", action="append", default=[], help="an nvcc flag for the layer (repeatable), e.g. "
                    "--nvcc-flag=-DSTATIC_RESET_IN_KERNEL (the serving build; note the '=')")
    args = ap.parse_args()
    if not (args.gpus == 1 or (args.gpus == 4 and args.tp == 4)) or args.gpus > torch.cuda.device_count():
        raise ValueError(f"--gpus {args.gpus} --tp {args.tp}: --gpus 1, or --gpus 4 with --tp 4 (the GPUs visible: "
                         f"{torch.cuda.device_count()})")
    if args.tp not in (1, 2, 4) or not 0 <= args.rank < args.tp:
        raise ValueError(f"--tp {args.tp} --rank {args.rank}: tp 1, 2 or 4 (whole 128-row blocks of gate | up)")
    snapshot = download_layer(args.checkpoint, args.layer, revision=args.revision)
    cfg = model_config(snapshot)
    assert (cfg["n_routed_experts"], cfg["n_activated_experts"], cfg["score_func"], cfg["route_scale"], cfg["moe_inter_dim"],
            cfg["swiglu_limit"]) == (E, K, "sqrtsoftplus", ROUTE_SCALE, INTER, SWIGLU_LIMIT), \
        "the checkpoint's MoE is not the one this demo builds"
    read = checkpoint_reader(snapshot)
    ranks = list(range(args.gpus)) if args.gpus > 1 else [args.rank]
    devs = [torch.device("cuda", g) for g in range(len(ranks))]
    ws = [layer_tensors(read, args.layer, args.tp, r, d) for r, d in zip(ranks, devs)]
    for wg, rt in zip(ws, routed_tensors(read, args.layer, args.tp, ranks, devs)):   # the routed experts, packed
        wg.update(rt)
    x = read_bf16(args.x) if args.x else torch.randn(T, HID, generator=torch.Generator().manual_seed(1)).bfloat16()
    for wg, d in zip(ws, devs):   # the same x on every GPU
        wg["x"] = x.to(d).contiguous()
        wg["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=d)
    sk, bufs, info = build(ws, args.out, profile=args.record is not None, extra_flags=args.nvcc_flag, plan=args.plan,
                           search=args.search, costs_code=args.costs_code, time_limit_s=args.time)
    if args.save_plan:
        compiler.write_plan_file(args.save_plan, info["plan"])
    h = Harness(sk)
    serving = bool(h.layout.get("serving"))
    if not check_launch(ws, h, bufs, serving, args.compare_y):
        sk.finalize()
        raise SystemExit(1)
    spans, launches, events = [], [], []
    for _ in range(args.reps):
        h.launch(512 << 20, timeout_s=60)     # 512 MB read first: L2 emptied
        spans.append(h.span_us())
        if serving:   # the kernel's time as a server sees it (CUDA events around the launch)
            events.append(max(h.elapsed_us()))
        if args.record:
            launches.append((h.task_times(), h.timing()))
    who = f"{args.gpus} GPUs" if args.gpus > 1 else f"rank {args.rank}"
    print(f"L{args.layer}, tp {args.tp} {who}: median span {statistics.median(spans):.1f} us over {args.reps} launches"
          + (f"; serving build, events median {statistics.median(events):.1f} us" if serving else ""))
    if args.record:   # the timing build's per-task times -> a per-node table and cost records (search.search_plan reads them)
        node_table(info, launches)
        import subprocess
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
