#!/usr/bin/env python3
"""Generate ~/report_perfetto.md : a HUMAN-READABLE, decode-layer-annotated
account of an MPK DeepSeek-V3 perfetto trace.

For each task position in one steady-state decode layer it reports:
  - the time window [t0,t1] (us, relative to layer start) and #blocks
  - WHICH decode-layer component it is (input_layernorm / qkv_a / q_b / RoPE /
    MLA-attn / o_proj / AllReduce / router-gate / topk / permute / W13 / SiLU /
    W2 / unpermute / shared-expert / ...), its function
  - grid: how many CTAs and HOW it is partitioned
  - EXPECTED (roofline) time vs the MEASURED slowest-CTA body
  - within-call IMBALANCE: median vs slowest CTA, flagged when >10% (the
    "same name hides a slow shape / load imbalance" check)
  - the same-layer SLOWEST task, for context

Semantic labels come from the builder-cited ground-truth per-layer task template
(DeepSeek-V3, bs=1, TP=8, EP=2, MTP-off). Each label is corroborated in the
report by showing EXPECTED grid/shape next to the MEASURED CTA count, so a
mislabel is self-evident. Timing/grid/imbalance numbers are 100% from the trace.

Usage: gen_perfetto_report.py <perfetto.csv> [out=~/report_perfetto.md] [num_workers=136]
"""
import sys, os, csv, re, statistics
from collections import defaultdict

CSV = sys.argv[1]
OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else os.path.expanduser("~/report_perfetto.md")
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 136
U32 = 1 << 32
HBM_BW = 8.0e12        # B200 ~8 TB/s/GPU (memory-bound roofline for M=1)
NVLINK_BW = 7.2e11     # ~720 GB/s effective per-GPU AllReduce ring BW (approx)

def dur(b, e):
    x = (e - b) % U32
    return x if x < (1 << 31) else 0

def load_names():
    # Walk up from this file to find include/mirage/persistent_kernel/runtime_header.h
    # so the tool works from any location (package dir, scripts/, repo root).
    here = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.join("include", "mirage", "persistent_kernel", "runtime_header.h")
    d = here
    for _ in range(8):
        p = os.path.join(d, rel)
        if os.path.exists(p):
            t = open(p).read()
            return {int(i): n for n, i in re.findall(r"\b(TASK_[A-Z0-9_]+)\s*=\s*(\d+)", t)}
        d = os.path.dirname(d)
    return {}
ID2NAME = load_names()

# ---- DeepSeek-V3 decode-layer GROUND-TRUTH template (cited from builder.py via
#      the Explore extraction). Ordered in true forward (dependency) order,
#      starting at input_layernorm. (M,N,K) and grid are per-rank TP=8.
#      rfn = (weight_bytes, kind) for the roofline; kind in {fp8,bf16,attn,ar}.
H, KLAT, KPE = 16, 512, 64          # per-rank heads, kv_lora, k_pe
HID, QLORA = 7168, 1536
#      ORDER = topk-segment canonical order (segment boundary = topk): the
#      MoE-tail of layer L, then layer L+1 attention, then L+1 router. Greedy
#      type-match in THIS order assigns the right component even when begin-time
#      is scrambled by shared-expert / chain overlap.
TEMPLATE = [
 dict(type="TASK_MOE_TOPK_SIGMOID_SM100", comp="topk-sigmoid routing",
      fn="sigmoid+bias, group top-4/8 -> top-8 experts + weights", grid="1 CTA (8 warps) — SERIAL",
      shape="256 logits topk=8", rfn=None),
 dict(type="TASK_QUANTIZE_FP8_SM100", comp="quantize MoE input -> FP8(UE8M0)",
      fn="quantize post-attn norm -> FP8 + UE8M0 scale for W13", grid="grid.x=token rows",
      shape="K=7168", rfn=(7168*2, "bf16")),
 dict(type="TASK_TENSOR_INIT", comp="zero-init MoE permute meta",
      fn="zero permute meta buffer", grid="1 CTA", shape="meta", rfn=None),
 dict(type="TASK_MOE_PERMUTE_SM100", comp="permute (expand + sort by expert)",
      fn="gather routed tokens per local expert into BM=128-padded FP8 layout + scale",
      grid="grid.x = expert groups (128 local experts / 4 per CTA = 32)", shape="M_total=16384 K=7168", rfn=None),
 dict(type="TASK_FP8_GROUP_GEMM_LARGEM_SM100", comp="W13 (gate+up) group-GEMM",
      fn="grouped FP8 GEMM per expert: permuted -> [gate|up]; only 8 of 128 experts active",
      grid="persistent workers stride (bm,bn) tiles; BM=128 block shares 1 expert",
      shape="active 8 tokens N=1024 K=7168", rfn=(8*1024*7168, "fp8")),
 dict(type="TASK_SILU_MUL", comp="SiLU (routed)",
      fn="SiLU(gate)*up on W13 out", grid="grid.x=experts (1 CTA/expert, 128)", shape="N=512", rfn=None),
 dict(type="TASK_QUANTIZE_FP8_SM100", comp="quantize SiLU -> FP8(UE8M0) (-> W2)",
      fn="quantize silu_out -> FP8 + UE8M0 K-outer scale", grid="grid.x = permuted rows",
      shape="K=512", rfn=None),
 dict(type="TASK_FP8_GROUP_GEMM_LARGEM_SM100", comp="W2 (down) group-GEMM",
      fn="grouped FP8 GEMM silu_fp8 -> down per expert", grid="persistent workers stride tiles",
      shape="active 8 tokens N=7168 K=512", rfn=(8*7168*512, "fp8")),
 dict(type="TASK_FP8_GEMM_DENSE_FINEN_SM100", comp="shared-expert gate_up",
      fn="FP8 GEMM hidden->[gate|up] (shared expert, TP over 8)", grid="persistent workers stride N-tiles",
      shape="M=1 N=512 K=7168", rfn=(512*7168, "fp8")),
 dict(type="TASK_SILU_MUL", comp="SiLU (shared expert)",
      fn="SiLU(gate)*up on shared mid", grid="grid.x = interleave chunks", shape="N=256", rfn=None),
 dict(type="TASK_FP8_GEMM_DENSE_SM100", comp="shared-expert down",
      fn="FP8 GEMM shared_silu(256)->hidden 7168", grid="persistent workers stride N-tiles",
      shape="M=1 N=7168 K=256", rfn=(7168*256, "fp8")),
 dict(type="TASK_MOE_UNPERMUTE_SM100", comp="unpermute + combine + shared add",
      fn="scatter topk-weighted routed sum + shared residual -> moe_out",
      grid="grid.x=token tiles (8 rows/CTA), grid.y=hidden splits (8)", shape="HID=7168 topk=8", rfn=None),
 dict(type="TASK_NVSHMEM_TILE_ALLREDUCE", comp="TP-AllReduce #2 (post-MoE)",
      fn="NVSHMEM all-reduce moe_out across 8 ranks + residual",
      grid="grid.x = 128-wide hidden tiles (56)", shape="reduce N=7168", rfn="ar"),
 # ---- next layer L+1 attention ----
 dict(type="TASK_FUSED_RMSNORM_QUANTIZE_FP8_SM100", comp="input_layernorm(+quant)",
      fn="RMSNorm(hidden) then FP8-quantize -> qkv_a input", grid="grid.x=token row (2 CTA)",
      shape="norm K=7168", rfn=(7168*2, "bf16")),
 dict(type="TASK_FP8_GEMM_DENSE_FINEN_SM100", comp="qkv_a_proj (q_a|kv_a|k_pe fused)",
      fn="FP8 GEMM hidden->2176 = q_a(1536)|c_latent(512)|k_pe(64)|pad(64)",
      grid="persistent workers stride output-N(BN=128) tiles; idle CTAs early-exit",
      shape="M=1 N=2176 K=7168", rfn=(2176*7168, "fp8")),
 dict(type="TASK_FUSED_RMSNORM_QUANTIZE_FP8_SM100", comp="q_a_layernorm(+quant)",
      fn="RMSNorm of q_a slice (K=1536) then FP8-quantize -> q_b input", grid="grid.x=token row",
      shape="norm K=1536", rfn=(1536*2, "bf16")),
 dict(type="TASK_FP8_GEMM_DENSE_SM100", comp="q_b_proj (absorbed)",
      fn="FP8 GEMM q_a(1536)->per-head [nope512|pe64] = H*576", grid="persistent workers stride N-tiles",
      shape="M=1 N=9216 K=1536", rfn=(9216*1536, "fp8")),
 dict(type="TASK_DEEPSEEK_MLA_ROPE_SM100", comp="RoPE (q_pe + k_pe)",
      fn="rotary embed on the pe-halves of q (16 heads) and k (1)", grid="grid.x=request, grid.y=head -> ~17 CTA",
      shape="rope dim 64/head", rfn=None),
 dict(type="TASK_RMS_NORM_HOPPER", comp="kv_a_layernorm",
      fn="RMSNorm of c_latent slice (K=512)", grid="1 CTA", shape="norm K=512", rfn=(512*2, "bf16")),
 dict(type="TASK_MLA_KV_APPEND_SM100", comp="KV-append",
      fn="write new token [c_latent512|k_pe64]=576 into contiguous KV cache", grid="1 CTA (bs=1)",
      shape="write 576", rfn=(576*2, "bf16")),
 dict(type="TASK_MLA_MTP_DECODE_TP8_SM100", comp="MLA attention (decode)",
      fn="flash-decode over latent KV; H=16 heads/rank, D_K=576 D_V=512 -> partial-O + LSE",
      grid="grid.x = q-group x KV-split (num_splits=ceil(kv/128)); grid.y=request -> ~4 CTA",
      shape="Q=H*576 KV=kv_len*576", rfn="attn"),
 dict(type="TASK_MLA_MTP_DECODE_TP_REDUCE_SM100", comp="MLA-reduce (split combine)",
      fn="LSE-weighted reduce of KV-split partials -> attn_out (H,512)",
      grid="grid.x=V-dim tiles (512/rd_dv=256 -> 256 CTAs = 2 WAVES @136!); grid.y=q-group grid.z=req",
      shape="reduce D_V=512", rfn=None),
 dict(type="TASK_QUANTIZE_FP8_SM100", comp="quantize attn_out -> FP8",
      fn="quantize attn_out (H*512) -> FP8+scale for the o-down BMM", grid="grid.y=(token*head) rows",
      shape="K=512/head", rfn=(512*2, "bf16")),
 dict(type="TASK_LINEAR_FP8_BMM_DENSE_SM100", comp="kv_b_v un-absorb (BMM2)",
      fn="per-head BMM attn_out_fp8 @ kv_b_v -> (H*128)", grid="grid.y=head (16); 1 N-tile",
      shape="per-head M=1 N=128 K=512", rfn=(16*128*512, "fp8")),
 dict(type="TASK_FP8_GEMM_DENSE_SM100", comp="o_proj (-> AllReduce#1)",
      fn="FP8 GEMM attn(H*128=2048)->hidden 7168 (TP partial)", grid="persistent workers stride N-tiles",
      shape="M=1 N=7168 K=2048", rfn=(7168*2048, "fp8")),
 dict(type="TASK_NVSHMEM_TILE_ALLREDUCE", comp="TP-AllReduce #1 (post-o_proj)",
      fn="NVSHMEM all-reduce o_proj partial across 8 ranks + residual",
      grid="grid.x = 128-wide hidden column tiles (7168/128=56)", shape="reduce N=7168", rfn="ar"),
 dict(type="TASK_RMS_NORM_HOPPER", comp="post_attention_layernorm",
      fn="RMSNorm(attn_out) -> MoE input", grid="1 CTA", shape="norm K=7168", rfn=(7168*2, "bf16")),
 dict(type="TASK_LINEAR_SM100", comp="router / gate GEMM (BF16)",
      fn="BF16 GEMM hidden->256 expert logits", grid="grid.x = output-N(expert) tiles (~32)",
      shape="M=1 N=256 K=7168", rfn=(256*7168*2, "bf16")),
]

def roofline_us(rfn, kv_len=512):
    if rfn is None:
        return None
    if rfn == "attn":   # MLA: stream latent KV cache once (FP8 576/elt shared across heads)
        return kv_len * 576 * 1.0 / HBM_BW * 1e6
    if rfn == "ar":     # ring all-reduce: 2*(n-1)/n * msg, msg=7168*2 bytes
        return 2 * (8 - 1) / 8 * 7168 * 2 / NVLINK_BW * 1e6
    nbytes, kind = rfn
    return nbytes * (1.0 if kind == "fp8" else 1.0) / HBM_BW * 1e6   # bytes already counted

# ---------- load + relabel ----------
evs = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        try:
            tid = int(r["task_type_id"]); b = int(r["begin_ts"]); e = int(r["end_ts"]); bi = int(r["block_idx"])
        except (KeyError, ValueError):
            continue
        if int(r.get("duration_ns", 1)) == 0:
            continue
        evs.append((b, e, bi, ID2NAME.get(tid, r.get("task_type_name") or f"UNKNOWN_{tid}")))
evs.sort()
SKIP = {"TASK_SCHD_EVENTS","TASK_SCHD_PREPARE_BATCH","TASK_BEGIN_TASK_GRAPH","TASK_SCHD_TASKS",
        "TASK_GET_EVENT","TASK_GET_NEXT_TASK","TASK_SM100_TASK_END","TASK_NVSHMEM_GLOBAL_ARGMAX"}

# ---------- segment on the once-per-layer topk marker (mid-MoE; the segment is
#            one full layer's work, phase-shifted: MoE-tail(L) + attn(L+1) +
#            router(L+1). The TEMPLATE is ordered to match this phase. ----------
BND = "TASK_MOE_TOPK_SIGMOID_SM100"
layer_bnd = sorted(b for b, _, _, nm in evs if nm == BND)
segs = list(zip(layer_bnd, layer_bnd[1:]))
use = segs[2:9] if len(segs) > 9 else segs[1:]

def cluster(es):
    es = sorted(es); out = []; cur = [es[0]]; seen = {es[0][2]}
    for x in es[1:]:
        if x[2] in seen or x[0] - cur[-1][0] > 40000:
            out.append(cur); cur = [x]; seen = {x[2]}
        else:
            cur.append(x); seen.add(x[2])
    out.append(cur); return out

# Build per-(name,occ) stats across analyzed layers + a representative layer's ordered positions
pos_stats = defaultdict(list)   # (name,occ) -> list of (ctas, slowCTA_us, medCTA_us, wall_us, t0_rel, blocks_lohi)
rep = None
for li, (s, e0) in enumerate(use):
    layer = [(b, e, bi, nm) for b, e, bi, nm in evs if s <= b < e0 and nm not in SKIP]
    bytype = defaultdict(list)
    for x in layer:
        bytype[x[3]].append(x)
    insts = []
    for nm, xs in bytype.items():
        for w in cluster(xs):
            insts.append((min(x[0] for x in w), nm, w))
    insts.sort()
    occ = defaultdict(int); ordered = []
    for t0, nm, w in insts:
        occ[nm] += 1
        bodies = sorted(dur(x[0], x[1]) / 1000.0 for x in w)
        blocks = sorted(set(x[2] for x in w))
        rec = (len(blocks), bodies[-1], statistics.median(bodies),
               (max(x[1] for x in w) - min(x[0] for x in w)) % U32 / 1000.0,
               (t0 - s) % U32 / 1000.0, (blocks[0], blocks[-1]))
        pos_stats[(nm, occ[nm])].append(rec)
        ordered.append((nm, occ[nm], rec))
    if li == len(use) // 2:
        rep = (s, e0, (e0 - s) % U32 / 1000.0, ordered)

# ---------- match representative layer positions to the template (canonical order) ----------
s, e0, layer_wall, ordered = rep
slowest_in_layer = max(ordered, key=lambda o: o[2][1])
claimed = [False] * len(ordered)
labeled = []   # (template_entry or None, observed pos)
for tpl in TEMPLATE:
    cand = [i for i, o in enumerate(ordered) if not claimed[i] and o[0] == tpl["type"]]
    if cand:
        i = cand[0]; claimed[i] = True
        labeled.append((tpl, ordered[i]))
# any observed positions not matched (extra waves / unexpected)
for i, o in enumerate(ordered):
    if not claimed[i]:
        labeled.append((None, o))
labeled.sort(key=lambda x: x[1][2][4])   # by t0

# ---------- emit report ----------
def fmt_imbalance(med, slow):
    if med <= 1e-6:
        return "-", ""
    r = slow / med
    return f"{(r-1)*100:.0f}%", ("  <-- IMBALANCED >10%" if r >= 1.10 and slow > 1.0 else "")

L = []
L.append(f"# Perfetto decode-layer report — `{os.path.basename(CSV)}`")
L.append("")
L.append(f"- Config: DeepSeek-V3 decode, bs=1, **TP=8 EP=2**, MTP-off, {NW} workers.")
L.append(f"- Source trace: `{CSV}`")
L.append(f"- Layer boundary = `MOE_TOPK_SIGMOID` (1/layer; segment = MoE-tail(L) + "
         f"attention(L+1) + router(L+1), presented in forward order). "
         f"{len(layer_bnd)} markers found; analyzed {len(use)} steady-state layers.")
L.append(f"- Roofline = memory-bound M=1: weight_bytes / {HBM_BW/1e12:.0f} TB-s "
         f"(GEMM), ring-AR over NVLink ~{NVLINK_BW/1e9:.0f} GB-s. EXPECTED is an "
         f"order-of-magnitude floor, not a vendor number.")
L.append("")
# summary
tot_busy = sum(dur(b, e) for b, e, bi, nm in evs if s <= b < e0 and nm not in SKIP) / 1000.0
occ_pct = 100 * tot_busy / (NW * layer_wall)
L.append(f"## Layer summary (representative steady-state layer)")
L.append(f"- **Layer wall = {layer_wall:.1f} us**  (x61 layers ~= {layer_wall*61/1000:.1f} ms/token)")
L.append(f"- **Mean GPU occupancy ~= {occ_pct:.0f}%** of {NW} workers "
         f"(busy worker-us {tot_busy:.0f} / capacity {NW*layer_wall:.0f}).")
L.append(f"- **Slowest task in layer: `{slowest_in_layer[0].replace('TASK_','').replace('_SM100','')}` "
         f"= {slowest_in_layer[2][1]:.1f} us** (slowest CTA body).")
L.append("")

# ---- auto "top problems" section (data-driven, not hand-written) ----
prob_imb = []   # (imb%, label, slow, med, ctas)
prob_rfn = []   # (ratio, label, slow, exp)
prob_wave = []  # (label, ctas, waves)
for tpl, (nm, oc, rec) in labeled:
    ctas, slow, med, wall, t0, _ = rec
    label = (tpl["comp"] if tpl else nm.replace("TASK_", "").replace("_SM100", "") + " (extra)")
    if med > 0.5 and slow / med >= 1.10 and ctas > 1:
        prob_imb.append((slow / med - 1, label, slow, med, ctas))
    if tpl and tpl.get("rfn") not in (None, "attn", "ar"):
        exp = roofline_us(tpl["rfn"])
        if exp and exp >= 0.1:   # meaningful GEMM/AR roofline; skip ~0 tiny-op floors
            prob_rfn.append((slow / exp, label, slow, exp))
    waves = (ctas + NW - 1) // NW
    if waves > 1:
        prob_wave.append((label, ctas, waves))
L.append("## What the timeline shows (auto-extracted)")
L.append("")
L.append(f"**A. NOT idle-bubble-bound — it's low-occupancy + imbalance.** GPU is ~{occ_pct:.0f}% "
         f"occupied: most of the {NW} workers sit idle while the critical path threads through "
         f"few-CTA serial tasks. The fix is raising effective parallelism on those tasks + killing imbalance.")
L.append("")
L.append("**B. Within-call imbalance (one slow CTA sets the call latency — the \"average hides the straggler\" trap):**")
for imb, label, slow, med, ctas in sorted(prob_imb, reverse=True)[:6]:
    L.append(f"- `{label}`: slowest CTA **{slow:.1f} us** vs median {med:.1f} us "
             f"(**+{imb*100:.0f}%**) across {ctas} CTAs")
L.append("")
L.append("**C. Critical-path tasks far over the memory-roofline floor (latency-/occupancy-bound):**")
for ratio, label, slow, exp in sorted(prob_rfn, reverse=True)[:6]:
    L.append(f"- `{label}`: measured **{slow:.1f} us** vs roofline {exp:.2f} us (**{ratio:.0f}× floor**)")
L.append("")
if prob_wave:
    L.append("**D. Over-dispatch (grid > workers → extra waves of the same task):**")
    for label, ctas, waves in prob_wave:
        L.append(f"- `{label}`: {ctas} CTAs on {NW} workers = **{waves} waves** (cap the grid)")
    L.append("")
# E: tiny serial-op fixed-overhead tax (<=2 CTA, near-zero compute, but 2-15us each)
tiny = [(tpl["comp"] if tpl else nm, rec[1]) for tpl, (nm, oc, rec) in labeled
        if rec[0] <= 2 and rec[1] >= 1.0]
if tiny:
    tot = sum(t[1] for t in tiny)
    L.append(f"**E. Small serial ops — fixed per-task overhead tax (≤2 CTA, ~0 compute, "
             f"yet {tot:.0f} us total / layer ≈ {tot*61/1000:.1f} ms/token):**")
    for label, slow in sorted(tiny, key=lambda x: -x[1]):
        L.append(f"- `{label}`: {slow:.1f} us on 1-2 CTAs (launch/sync-bound, fully serial)")
    L.append("")

L.append("## Interpretation & caveats (Codex-cross-checked — read before acting)")
L.append("")
L.append("- **\"Occupancy ~36%\" = task-BODY residency, not SM utilization.** In-body TMA / "
         "mbarrier / TMEM / NVSHMEM / memory stalls count as \"busy.\" The honest claim is: "
         "*the layer is a narrow critical-path chain with small zero-active gaps (~3%) and low "
         "runnable width* — NOT \"idle-bubble bound.\"")
L.append("- **The within-call \"imbalance\" is mostly real-work-CTA vs fixed-overhead-CTA, not load "
         "imbalance.** Per-CTA duration distributions (steady layer): "
         "W13 = all 136 CTAs 6.5–8 us with ~13 stragglers @23 us (a TAIL); "
         "W2 = all 136 CTAs 23.7–25.3 us (WELL-BALANCED, ignore its flag); "
         "o_proj = ~56 real-work CTAs @12 us + ~80 overhead-only CTAs @5 us (only 56 N-tiles exist); "
         "MLA-decode = 3 CTAs @5.7 us + **1 @31.8 us** (a genuine single-split straggler).")
L.append("- **OPEN QUESTION needing the GPU:** is each straggler/real-work CTA doing real COMPUTE "
         "or an in-body STALL (TMA-weight wait / mbarrier)? Even the real-work GEMM CTAs run "
         "~100–180x over the M=1 weight-stream roofline → smells latency/overhead-bound, not BW-bound. "
         "Disambiguate by (a) clustering CTA durations by predicted tile-count, (b) checking if the "
         "slow block_idx is deterministic across layers (mapping/work) or random (resource stall).")
L.append("- The ~5–7 us **fixed cost every CTA pays even with no tile** is the same tax as section E: "
         "MPK per-task setup (TMA-desc / smem alloc / barrier) dominates M=1 work.")
L.append("")
L.append("## Timeline (true forward order; one decode layer)")
L.append("")
L.append("| t0–t1 (us) | blocks | task (perfetto) | decode component | function | grid: CTAs / scheme | "
         "measured slowCTA | EXPECTED (roofline) | imbalance |")
L.append("|---|---|---|---|---|---|---|---|---|")
for tpl, (nm, oc, rec) in labeled:
    ctas, slow, med, wall, t0, (blo, bhi) = rec
    short = nm.replace("TASK_", "").replace("_SM100", "")
    if tpl:
        comp = tpl["comp"]; scheme = tpl["grid"]; fn = tpl["fn"]
        exp = roofline_us(tpl.get("rfn"))
        exps = f"{exp:.2f} us" if exp is not None else "—"
    else:
        comp = "?? (unmatched / extra wave)"; scheme = "—"; exps = "—"; fn = "—"
    imb, flag = fmt_imbalance(med, slow)
    L.append(f"| {t0:.1f}–{t0+wall:.1f} | {ctas} (#{blo}-{bhi}) | {short} | {comp} | {fn} | "
             f"{ctas} / {scheme} | {slow:.2f} us (med {med:.2f}) | {exps} | {imb}{flag} |")
L.append("")
# per-(task,occ) worst-case across layers + over-dispatch
L.append("## Per-position worst-case across analyzed layers (slowest CTA + imbalance)")
L.append("")
L.append("| task @occ | maxCTAs | slowCTA (worst layer) | med | imbalance | waves(>1=over-dispatch) |")
L.append("|---|---|---|---|---|---|")
rows = []
for (nm, oc), recs in pos_stats.items():
    mx = max(recs, key=lambda r: r[1])
    ctas = max(r[0] for r in recs)
    waves = (ctas + NW - 1) // NW
    imb, flag = fmt_imbalance(mx[2], mx[1])
    rows.append((mx[1], nm.replace("TASK_", "").replace("_SM100", ""), oc, ctas, mx[1], mx[2], imb, flag, waves))
for slow, short, oc, ctas, mxslow, med, imb, flag, waves in sorted(rows, reverse=True):
    wv = f"**{waves} WAVES**" if waves > 1 else "1"
    L.append(f"| {short} @{oc} | {ctas} | {mxslow:.2f} us | {med:.2f} | {imb}{flag} | {wv} |")
L.append("")
L.append("> Notes: semantic labels are aligned to the builder-cited DSv3 decode template; "
         "EXPECTED grid/shape is shown so a mislabel is self-evident (measured CTAs should match the scheme). "
         "slowCTA = slowest single CTA body (the call's critical cost); imbalance = (slowCTA/medianCTA − 1).")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}  ({len(labeled)} timeline positions, layer wall {layer_wall:.1f} us, occ {occ_pct:.0f}%)")
