"""Run DeepSeek-V4.1-Flash's decode steps of an SGLang server on the static megakernel, or check the layers against SGLang's.

boot/sitecustomize.py imports this module when DSV41_STATIC_ATTN or DSV41_STATIC_MOE is set (layer ids, or "all"). It patches:
  ModelRunner.init_cuda_graphs  first, builds this rank's chosen layers (dsv41_static_layer)
  MQALayer.forward              a decode step of exactly T rows: one launch instead of SGLang's attention, returns
                                mhc.AttnOutput(y), this rank's partial: SGLang's all-reduce and mHC follow as usual. A prefill
                                of T prompts of the same length runs SGLang's attention, then replays the layer over it
  DeepseekV2MoE.forward         a decode step of exactly T rows: one launch instead of SGLang's MoE, returns y as merged rows
DSV41_STATIC_AR=1: the layers do the TP all-reduce themselves (dsv41_static_layer ExchangePool): the attention returns a plain tensor
(SGLang's transport then owes no sum) and the MoE's y passes MoeTransport.prepare_output (patched) as it is.
DSV41_STATIC_POOL=1 (the default; SGLANG_DSV4_KV_LAYOUT=v41_new): the layers read and write SGLang's own pools (dsv41_static_layer
bind_pools): no replay, and a step of any other row count runs SGLang's layer on the same state. DSV41_STATIC_POOL=0: the layers
keep their own caches, so every step after the prefill must have T rows and a step that would fall back raises instead.

check (DSV41_STATIC_CHECK=1; eager runs): SGLang's layer first, then the static one on the same input; printed by rank 0: y relRMS
static vs SGLang. DSV41_STATIC_KEEP=sgl (the default in check mode) passes SGLang's output on, static the static layer's.
Env: DSV41_STATIC_ATTN, DSV41_STATIC_MOE, DSV41_STATIC_CHECK, DSV41_STATIC_KEEP, DSV41_STATIC_PDL (0 / 1),
DSV41_STATIC_CKPT (a directory with inference/config.json and the shards, default the server's model path),
DSV41_STATIC_OUT (build directory), DSV41_STATIC_MAX_POS (the largest position, default 2047), DSV41_STATIC_ROOT,
DSV41_STATIC_PROF (a file: the kernels' own stamps of the last step when the driver rewrites it, see dsv41_static_prof.py).
"""
import os
import time

import torch
import dsv41_static_layer as _L8   # noqa: E402  (the layers of T = 8; larger T: its copies, layer_module)
from dsv41_static_layer import ROOT, ExchangePool, T, StaticHead, attn_weights, layer_module   # noqa: E402
from sglang.srt.model_executor import model_runner as _mr
from sglang.srt.models import deepseek_v2 as _dv2
from sglang.srt.models import deepseek_v4 as _v4
from sglang.srt.models import deepseek_v4_boundary as _bd
from sglang.srt.models import deepseek_v4_mhc as _mhc


def _ids(name):
    v = os.environ.get(name, "").strip()
    return set() if not v else None if v == "all" else {int(x) for x in v.split(",") if x.strip()}


ATTN, MOE = _ids("DSV41_STATIC_ATTN"), _ids("DSV41_STATIC_MOE")   # None: every layer
CHECK = os.environ.get("DSV41_STATIC_CHECK", "0") != "0"
CHECK_STEPS = int(os.environ.get("DSV41_STATIC_CHECK_STEPS", "3"))   # check mode, mHC in the kernels: steps printed per sublayer
CHECK_BITS = os.environ.get("DSV41_STATIC_CHECK_BITS", "0") != "0"   # and the static outputs' integer sums with each line
PROBE = os.environ.get("DSV41_STATIC_PROBE", "0") != "0"
POOL = os.environ.get("DSV41_STATIC_POOL", "1") != "0"
AR = os.environ.get("DSV41_STATIC_AR", "0") != "0"   # the all-reduce in the kernels: y is the sum, SGLang's own one is skipped
assert not (AR and CHECK), "check mode compares this rank's partial y: DSV41_STATIC_AR=0"
SEAM = os.environ.get("DSV41_STATIC_SEAM", "0") != "0"   # no all-reduce in the kernels; the MoE takes the attention's seam kernel
assert not (AR and SEAM), "DSV41_STATIC_SEAM: the all-reduce stays in SGLang (DSV41_STATIC_AR=0)"
MHC = os.environ.get("DSV41_STATIC_MHC", "0") != "0"   # mHC and the all-reduce in the kernels: a sublayer ends in its kernel
# H1 (DSV41_X_FLAG, dsv41_static_layer): the flags pair a MoE launch with the attention launch before it, so both must run in
# the same forwards: they do (decode, T rows, _active(); one launch each, check mode included), except the attention's prefill
# replay without SGLang's pools (DSV41_STATIC_POOL=0), which launches the attention alone
assert os.environ.get("DSV41_X_FLAG", "0") == "0" or (MHC and os.environ.get("DSV41_STATIC_POOL", "1") != "0"), \
    "DSV41_X_FLAG: DSV41_STATIC_MHC=1 and SGLang's pools (DSV41_STATIC_POOL=1)"
assert not (MHC and (AR or SEAM)), "DSV41_STATIC_MHC: no DSV41_STATIC_AR, no DSV41_STATIC_SEAM"
STATS = os.environ.get("DSV41_STATIC_STATS", "0") != "0"   # mHC's coefficients in the kernels; the seam stays SGLang's, no SMs left free
assert not STATS or (SEAM and not MHC), "DSV41_STATIC_STATS needs DSV41_STATIC_SEAM=1 and no DSV41_STATIC_MHC"
DUMP = os.environ.get("DSV41_STATIC_DUMP")   # a directory: the static MoE layers' x of every step
# (11-20, default off) DSV41_ENGRAM=1 (with DSV41_STATIC_MHC=1): an Engram layer's gate in the kernel of the MoE layer before it
# (dsv41_static_layer StaticMoE engram=True: moe_demo.engram_nodes, wkv column-sharded over the GPUs and gathered by multicast);
# SGLang's gather of the n-gram rows runs before that MoE kernel (its own code, issued earlier: EngramEmbedding.forward via
# Engram.embed_rows, on the step's hash ids, which EngramHasher.forward (patched) leaves in _STATE), and Engram.forward
# (patched) passes the kernel's R'' on instead of its quantize, wkv (replicated, 157 MB a GPU) and gate. Check mode: SGLang's
# Engram on the kernel's R' as well, R'' compared
ENGRAM = os.environ.get("DSV41_ENGRAM", "0") != "0"
# DSV41_ENGRAM=2: the wkv GEMM and the gather of kv in the attention kernel of the layer before (its idle SMs while sparse_attn's
# inputs are computed), into a symmetric buffer (dsv41_static_layer.engram_kv_buffer); the MoE kernel after it only gates
ENGRAM2 = os.environ.get("DSV41_ENGRAM", "0") == "2"
assert not ENGRAM or MHC, "DSV41_ENGRAM: DSV41_STATIC_MHC=1"
# (11-20, default off) DSV41_HEAD=1 (with DSV41_STATIC_MHC=1, every layer static): the step's end in our kernels: the last MoE
# kernel's post computes the head's input (the streams' collapse with its FFN pre, the final norm: as it computes a next
# attention's input), and a static LM head kernel (dsv41_static_layer StaticHead, head_demo.py) the logits [T, vocab] fp32 into
# SGLang's next_token_logits_buffer: SGLang's mhc_combine and final norm (patched) pass that input on, and
# LogitsProcessor._get_logits (patched) launches the head instead of its LM head GEMM, gather and fp32 copy
HEAD = os.environ.get("DSV41_HEAD", "0") != "0"
# DSV41_HEAD=2: the head's nodes in the last MoE kernel itself (StaticMoE head=True, moe_demo.head_nodes): no kernel boundary,
# and its GEMM tasks on the SMs free after their experts stream their weights while the post computes the input
HEAD2 = os.environ.get("DSV41_HEAD", "0") == "2"
HEAD_CHECK = os.environ.get("DSV41_HEAD_CHECK", "0") != "0"   # eager steps: SGLang's LM head on the same x as well, compared
# (11-20, with DSV41_HEAD=2) DSV41_HEAD_ARGMAX=1: the head's gather also stores each token's greedy next token (torch.argmax's
# rules over its own logits); the decode graphs' replays hand them to the sampler (SGLang's in-graph greedy helper's rules)
HEAD_ARGMAX = HEAD2 and os.environ.get("DSV41_HEAD_ARGMAX", "0") != "0"
# (11-20, DSV41_ENGRAM=2) DSV41_EG_EARLY=1: SGLang's gathers of the n-gram rows of every Engram layer (its table lookup and
# the TP all-reduce) before the step's first host kernel (attention L0), not between the kernels before each host: the PDL
# chain of the later host (e.g. L1 with DSV41_EG_KV14=1) is not broken by them
EG_EARLY = os.environ.get("DSV41_EG_EARLY", "0") != "0"
# (11-20, DSV41_ENGRAM=2) DSV41_EG_ROWS=1: the n-gram rows looked up in the host attention kernel (engram_rows: each rank's rows
# of SGLang's own table, by multicast into every rank's buffer; the act_quant polls them): no SGLang gather or all-reduce
EG_ROWS = os.environ.get("DSV41_EG_ROWS", "0") != "0"
# (11-21) DSV41_STATIC_ROWS=1,2,4: decode steps of n < T rows run the static layers too (GLM-5.3 #821's way): SGLang's own CUDA
# graphs of those batch sizes, unchanged; the static layers of T rows read the n rows, their idle rows as SGLang pads a graph
# (StepTables.fill: position 0, request slot 0, KV slot 0); the MoE layers get a build of n live rows for each n listed (its idle
# rows not routed: no expert outside the live rows'); a step of n rows without one runs the T build, its idle rows routed. A
# step of T rows runs exactly the T build as before
ROWS = sorted({int(x) for x in os.environ.get("DSV41_STATIC_ROWS", "").split(",") if x.strip()})
assert all(0 < r < 8 for r in ROWS), f"DSV41_STATIC_ROWS {ROWS}: live-row builds below T"
SMALL = bool(ROWS)
# (11-21) DSV41_STATIC_T=16,32,64: the static layers also of these T (GLM-5.3 #821's way): a decode step of n > 8 rows runs
# the layers of the smallest T >= n, the rows past n idle (padded as SGLang pads a CUDA graph, StepTables.fill); the same
# weights, each T its own kernels, buffers and exchange regions (dsv41_static_layer.layer_module: the modules of that T)
TS = sorted({int(x) for x in os.environ.get("DSV41_STATIC_T", "").split(",") if x.strip()})
assert all(t in (16, 32, 64) for t in TS), f"DSV41_STATIC_T {TS}: of 16, 32, 64"
_STATE_SETS = {}   # T > 8 -> its state (as _STATE's tables, first, head_moe, head, eg_hosts, ids_pad, eg_early)


def _T_of(rows: int):
    """The T of the layers a decode step of `rows` rows runs on (the smallest T >= rows), or None"""
    return next((t for t in [T] + TS if rows <= t), None)


def _pick(sub, rows: int):
    """sub's static layer for a decode step of `rows` rows (sub: a decoder layer's self_attn or mlp), or None"""
    t = _T_of(rows)
    return getattr(sub, "_dsv41", None) if t == T else getattr(sub, "_dsv41_t", {}).get(t)


def _set(tok: int) -> dict:
    """The per-T state of the layers of T = tok"""
    return _STATE if tok == T else _STATE_SETS[tok]
# (a 8-row build and its live-row variants alternate from step to step: H1's x flags count each kernel's launches, so the
# chain would wait for ever; DSV41_ENGRAM=1's gather in the MoE forward takes T rows only)
assert not (SMALL and os.environ.get("DSV41_X_FLAG", "0") != "0"), "DSV41_STATIC_ROWS: not with DSV41_X_FLAG (H1)"
assert not (SMALL and ENGRAM and not ENGRAM2), "DSV41_STATIC_ROWS: with DSV41_ENGRAM=2 (or no Engram), not 1"


def _ok_rows(rows: int) -> bool:
    """A decode step of this many rows runs the static layers: T, or with DSV41_STATIC_ROWS (and not in check mode) 1 .. T;
    with DSV41_STATIC_T up to its largest T (check mode: exactly one of them)."""
    return rows == T or (SMALL and not CHECK and 0 < rows < T) or \
        (T < rows <= (TS[-1] if TS else 0) and (not CHECK or rows in TS))
assert not HEAD or MHC, "DSV41_HEAD: DSV41_STATIC_MHC=1"
KEEP = os.environ.get("DSV41_STATIC_KEEP", "sgl" if CHECK else "static")
PDL = os.environ.get("DSV41_STATIC_PDL", "0") != "0"
OUT = os.environ.get("DSV41_STATIC_OUT", os.path.join(ROOT, "build", "sgl_static_dsv41"))
MAX_POS = int(os.environ.get("DSV41_STATIC_MAX_POS", "2047"))
os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE", os.path.join(OUT, "cache"))
_orig_init = _mr.ModelRunner.init_cuda_graphs
_orig_attn = _v4.MQALayer.forward
_orig_moe = _dv2.DeepseekV2MoE.forward
_STATE = {"tp_group": None, "started": False, "serving": False}   # serving: past init_cuda_graphs (SGLang's warmup and
                                                                  # autotune run dummy prefill and decode steps there)


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _log(msg):
    print(f"[dsv41_static rank {_rank()} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _relrms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _active():
    """Past SGLang's warmup and autotune, or a CUDA graph being captured (a capture may run inside init_cuda_graphs)."""
    return _STATE["serving"] or torch.cuda.is_current_stream_capturing()


def _chosen(ids, layer):
    return ids is None or layer in ids


def _put(sub, name: str, tok: int, st):
    """sub.<name> = st for T = 8 (as before); for a larger T sub.<name>_t[tok] (DSV41_STATIC_T)"""
    if tok == T:
        setattr(sub, name, st)
    else:
        if not hasattr(sub, name + "_t"):
            setattr(sub, name + "_t", {})
        getattr(sub, name + "_t")[tok] = st


def _build_set(self, Lm, S: dict, layers: list, cfg: dict, ckpt: str, rank: int, tp: int, dev, base: dict):
    """This rank's chosen static layers of one T (11-21): Lm = dsv41_static_layer (T = 8) or its copy of a larger T
    (layer_module, DSV41_STATIC_T); S its state (_STATE for T = 8, _STATE_SETS[T] else); base {layer: (attention, MoE) of
    T = 8} (a larger T takes their weights) or {}"""
    tok = Lm.T
    tag = "" if tok == T else f" (T {tok})"
    sources = Lm.Sources(cfg)
    attn_pool = Lm.ExchangePool(_STATE["tp_group"], dev, len(layers)) if AR or MHC else None
    moe_pool = Lm.ExchangePool(_STATE["tp_group"], dev, len(layers) * (1 + (len(ROWS) if tok == T else 0))) \
        if AR or MHC else None   # (+ the live-row variants)
    self._dsv41_pools = self._dsv41_pools + (attn_pool, moe_pool)   # the exchange regions: alive as long as the layers
    steps = None
    if POOL:   # the caps by ratio as StaticAttn's
        caps = {r: max(((MAX_POS + 1) // r + 255) // 256 * 256, 512) for r in set(cfg["compress_ratios"]) if r in (1, 2)}
        steps = S["tables"] = Lm.StepTables(dev, caps)
        S["first"] = next((lay.layer_id for lay in layers if _chosen(ATTN, lay.layer_id)), None)
    egkv = {}    # (DSV41_ENGRAM=2) the MoE layer before an Engram layer -> its gathered kv (view, multicast address)
    eghost = {}  # the attention layer that computes an Engram layer's kv -> (view, multicast address, Engram layer)
    if ENGRAM2:
        # DSV41_EG_KV14=A: layer 14's kv in attention layer A's kernel (default 13, the layer before; 11-20: a reuse
        # kernel has no idle SMs for it, +21 us at L13, a swa kernel about +10)
        host = {14: int(os.environ.get("DSV41_EG_KV14", "13"))}
        for lay in layers:
            L = lay.layer_id
            if L + 1 < len(layers) and layers[L + 1].engram is not None and _chosen(ATTN, L) and _chosen(MOE, L):
                view, keep, mc = Lm.engram_kv_buffer(tp, dev, _STATE["tp_group"])
                egkv[L] = (view, mc)
                a = host.get(L + 1, L)
                assert a <= L and a not in eghost and _chosen(ATTN, a), f"DSV41_EG_KV14: attention layer {a}"
                eghost[a] = (view, mc, L + 1)
                if EG_ROWS:   # (collective: every rank in the same order)
                    rows = Lm.engram_rows_tables(layers[L + 1].engram, tp, dev, _STATE["tp_group"])
                    if rows["peer"]:   # (each rank reads every row in place: its own buffer, 0xFF for the polls)
                        ev = torch.full((tok, 6144), -1, dtype=torch.int16, device=dev).view(torch.bfloat16)
                        rows.update(emb=ev, emb_mc=ev.data_ptr())
                    else:
                        ev, ekeep, emc = Lm.engram_emb_buffer(tp, dev, _STATE["tp_group"])
                        rows.update(emb=ev, emb_mc=emc)
                        self._dsv41_pools = self._dsv41_pools + (ekeep,)
                    eghost[a] = (view, mc, L + 1, rows)
                    _log(f"Engram L{L + 1}{tag}: its rows looked up in attention L{a}'s kernel "
                         f"({'PEER_READ: in place' if rows['peer'] else 'sharded: multicast'})")
                self._dsv41_pools = self._dsv41_pools + (keep,)
    for lay in layers:
        L = lay.layer_id
        if _chosen(ATTN, L):
            sa = Lm.StaticAttn(L, ckpt, rank, tp, dev, OUT, MAX_POS, sources, PDL, attn_pool, steps, mhc=MHC, stats=STATS,
                               engram_kv=eghost.get(L), weights_from=base.get(L, (None, None))[0])
            _put(lay.self_attn, "_dsv41", tok, sa)
            if L in eghost:
                sa.eg_mod = layers[eghost[L][2]].engram
                S.setdefault("eg_hosts", []).append(sa)
                _log(f"attention L{L}{tag}: layer {eghost[L][2]}'s Engram kv in its kernel")
            _ATTNS.append(sa)
            if POOL:
                sa.bind_pools(self.token_to_kv_pool)
            _log(f"attention L{L}{tag} ({sa.variant}) ready, {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
        if _chosen(MOE, L):
            ex = lay.mlp.experts
            if int(getattr(ex, "moe_ep_size", 1)) != 1:
                raise RuntimeError("the static MoE is one TP shard of every expert: run SGLang with ep_size 1")
            # the next sublayer's input in the kernel unless Engram rewrites the streams first or the head reads them
            # (DSV41_STATIC_NORM_NEXT=0, measurement 2026-11-17: never; the MoE kernel ends at its mHC post and SGLang
            # computes the next attention's input, as GLM-5.3-Flash does at bs 8 (#821))
            norm_next = MHC and L + 1 < len(layers) and layers[L + 1].engram is None \
                and os.environ.get("DSV41_STATIC_NORM_NEXT", "1") != "0"
            eg = ENGRAM and L + 1 < len(layers) and layers[L + 1].engram is not None
            head = HEAD and L + 1 == len(layers)   # the last layer: its post computes the head's input
            pool = moe_pool
            if head and HEAD2:   # (its exchange region holds the head's rows too: larger than the pool's slots)
                pool = Lm.ExchangePool(_STATE["tp_group"], dev, 1)
                self._dsv41_pools = self._dsv41_pools + (pool,)
            sm = Lm.StaticMoE(L, ckpt, int(ex.moe_tp_rank), int(ex.moe_tp_size), dev, OUT, PDL, pool, mhc=MHC,
                              norm_next=norm_next or head, stats=STATS, engram=eg, gamma_name="norm.weight" if head else None,
                              engram_kv=egkv[L][0] if eg and L in egkv else None, head=head and HEAD2,
                              weights_from=base.get(L, (None, None))[1])
            _put(lay.mlp, "_dsv41", tok, sm)
            if head:
                S["head_moe"] = sm
            if eg:   # the next layer's Engram module: its gather runs before this kernel, its forward passes R'' on
                sm.eg_mod = layers[L + 1].engram
                _put(layers[L + 1].engram, "_dsv41_moe", tok, sm)
                _log(f"MoE L{L}{tag}: layer {L + 1}'s Engram gate in its kernel")
            for r in (ROWS if tok == T else ()):   # (DSV41_STATIC_ROWS) its decode builds of r live rows, the same tensors
                vpool = pool
                if head and HEAD2:
                    vpool = Lm.ExchangePool(_STATE["tp_group"], dev, 1)
                    self._dsv41_pools = self._dsv41_pools + (vpool,)
                sm.variants[r] = sm.rows_variant(r, vpool)
            if ROWS and L == 0 and tok == T:
                _log(f"MoE: decode builds of {ROWS} live rows beside the {T}-row one")
            _MOES.append(sm)
            if AR or SEAM or MHC:
                _OUTS.add(sm.w["y"].data_ptr())
            _log(f"MoE L{L}{tag} ready, shared expert TP1 {getattr(lay.mlp, '_shared_expert_tp1', '?')}, "
                 f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")


def _init_cuda_graphs(self, *args, **kwargs):
    if not getattr(self, "_dsv41_built", False) and not getattr(self, "is_draft_worker", False):
        self._dsv41_built = True
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.runtime_context import get_parallel
        ckpt = os.environ.get("DSV41_STATIC_CKPT", self.server_args.model_path)
        cfg = attn_weights.model_config(ckpt)
        rank, tp = int(get_parallel().attn_tp_rank), int(get_parallel().attn_tp_size)
        dev = torch.device("cuda", torch.cuda.current_device())
        _STATE["tp_group"] = get_tp_group().device_group
        layers = sorted((m for m in self.model.modules() if isinstance(m, _v4.DeepseekV4CudaDecoderLayer)),
                        key=lambda m: m.layer_id)
        _STATE["meta"] = {"variants": {lay.layer_id: attn_weights.variant_of(cfg, lay.layer_id) for lay in layers},
                          "engram": [lay.layer_id for lay in layers if lay.engram is not None], "tp": tp}   # (for dsv41_static_prof)
        self._dsv41_pools = ()   # the exchange regions: alive as long as the layers
        _build_set(self, _L8, _STATE, layers, cfg, ckpt, rank, tp, dev, {})
        base = {lay.layer_id: (getattr(lay.self_attn, "_dsv41", None), getattr(lay.mlp, "_dsv41", None)) for lay in layers}
        for tok in TS:   # (DSV41_STATIC_T) the layers of each larger T: the same weights, their own kernels and buffers
            _build_set(self, layer_module(tok), _STATE_SETS.setdefault(tok, {}), layers, cfg, ckpt, rank, tp, dev, base)
        if HEAD:
            if _STATE.get("head_moe") is None or ATTN is not None or MOE is not None:
                raise RuntimeError("DSV41_HEAD: every layer static (DSV41_STATIC_ATTN=all, DSV41_STATIC_MOE=all)")
            if HEAD2:   # (the head's nodes in the last MoE kernel: its logits buffer)
                for S in [_STATE] + [_STATE_SETS[t] for t in TS]:
                    S["head"] = S["head_moe"]
            else:
                assert not TS, "DSV41_HEAD=1 (its own kernel) with DSV41_STATIC_T: T = 8 only"
                head_pool = ExchangePool(_STATE["tp_group"], dev, 1)
                self._dsv41_pools = self._dsv41_pools + (head_pool,)
                _STATE["head"] = StaticHead(ckpt, rank, tp, dev, OUT, PDL, head_pool)
            fn = getattr(getattr(self.model, "model", None), "norm", None)
            if fn is None:
                raise RuntimeError("DSV41_HEAD: no final norm at model.model.norm")
            _patch_final_norm(fn)
            _log(f"LM head static, {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
        torch.cuda.synchronize()
        if AR or MHC:   # every rank's layers built (each fills its copy of the exchange regions) before any first launch
            torch.distributed.barrier(group=_STATE["tp_group"])
    out = _orig_init(self, *args, **kwargs)
    _STATE["serving"] = True
    return out


def _plus0(v):
    """e2m1 bytes (two nibbles) with each -0 (0b1000) as +0."""
    lo, hi = v & 0x0F, v >> 4
    return torch.where(lo == 8, 0, lo) | (torch.where(hi == 8, 0, hi) << 4)


def _probe(st, positions, forward_batch):
    """DSV41_STATIC_PROBE=1, check mode, rank 0: read SGLang's pools by the addressing the kernels would use and compare the rows
    byte by byte with the caches the static layer filled itself."""
    pool, be = _v4.get_token_to_kv_pool(), _v4.get_attn_backend()
    fm = be.forward_metadata
    core = getattr(fm, "core_attn_metadata", None) or fm.core_metadata
    L, w, pos = st.layer, st.w, positions.view(-1).long()
    p0 = int(pos[0])
    win = w["window_cache"].cpu()
    raw = pool.get_swa_raw_buffer(L)
    idx, lens = core.swa_page_indices, core.swa_topk_lengths
    out = core.swa_out_cache_loc
    pt = core.page_table
    _log(f"probe L{L} p {p0}: layout {pool.kv_layout}, swa buffer {tuple(raw.shape)} {raw.dtype}, swa page {pool.get_swa_key_page_size()}, "
         f"bytes per token {pool.get_swa_key_bytes_per_token()}; swa_page_indices {tuple(idx.shape)} {idx.dtype} lens {lens[:2].tolist()} "
         f"out_loc {None if out is None else out[:2].tolist()}; row 0 list head {idx[0][:4].tolist()} tail "
         f"{idx[0][max(int(lens[0]) - 3, 0):int(lens[0]) + 2].tolist()}; page_table {tuple(pt.shape)} {pt.dtype} row 0 {pt[0][:4].tolist()}; "
         f"req {forward_batch.req_pool_indices[:3].tolist()}")
    flat = raw.view(-1).view(torch.uint8).cpu()
    idc, ln = idx.cpu(), lens.cpu()
    for name, order in (("oldest first", lambda t, i, n: int(pos[t]) - n + 1 + i), ("newest first", lambda t, i, n: int(pos[t]) - i)):
        eq, rows = 0, 0
        for t in range(T):
            n = int(ln[t])
            for i in range(n):
                slot = int(idc[t, i])
                eq += int(torch.equal(flat[slot * 528:slot * 528 + 528], win[t, order(t, i, n) % 128]))
                rows += 1
        _log(f"probe L{L} window, list {name}: rows byte-equal {eq} / {rows}")
    if "compress_cache" not in w:
        return
    cc = w["compress_cache"].cpu()
    kv = pool.get_extra_key_buffer(L)
    ps = pool.get_extra_key_page_size(L)
    ratio = st.ratio
    _log(f"probe L{L} compressed: layout {pool.get_extra_key_layout(L)}, buffer {tuple(kv.shape)} {kv.dtype}, page {ps}, ratio {ratio}, "
         f"c{ratio}_out_loc {getattr(core, f'c{ratio}_out_loc', None)[:2].tolist() if getattr(core, f'c{ratio}_out_loc', None) is not None else None}")
    f2, ptc = kv.view(-1).view(torch.uint8).cpu(), pt.cpu()
    eq, rows, st_ptrs = 0, 0, []
    for t in range(T):
        n = (int(pos[t]) + 1) // ratio
        for j in range(n):
            slot = int(ptc[t, j // ps]) * ps + j % ps
            eq += int(torch.equal(f2[slot * 288:slot * 288 + 288], cc[t, j]))
            rows += 1
    _log(f"probe L{L} compressed rows via page_table: byte-equal {eq} / {rows}")
    if "index_k" not in w or st.variant == "reindex":
        return
    ib = pool.get_index_k_with_scale_buffer(L).view(torch.uint8).cpu()
    ips = pool.get_index_k_page_size(ratio)
    ik = w["index_k"].cpu()
    pr = ib.shape[1] // ips
    eq, eq_i, rows, nb = 0, 0, 0, 0
    z = _plus0
    for t in range(T):
        n = (int(pos[t]) + 1) // ratio
        for j in range(n):
            pg = int(ptc[t, j // ps]) * (ps // ips) + (j % ps) // ips
            o = j % ips
            a = torch.cat([z(ib[pg, o * 64:o * 64 + 64]), ib[pg, ips * 64 + o * 4:ips * 64 + o * 4 + 4]])
            b = torch.cat([z(ik[t, j, :64]), ik[t, j, 64:]])
            eq += int(torch.equal(a, b))
            eq_i += int(torch.equal(z(ib[pg, o * pr:o * pr + 64]), b[:64])) if pr == 68 else 0
            nb += int((a != b).sum())
            rows += 1
    _log(f"probe L{L} index K: buffer {tuple(ib.shape)}, page {ips}, -0 as +0: split layout rows equal {eq} / {rows} ({nb} of "
         f"{rows * 68} bytes differ), row layout data equal {eq_i} / {rows}")
    if "cmp_state" in w:
        cs = pool.get_attention_compress_states(L)
        ks = cs.kv_score_buffer.kv_score
        ring = cs.ring_size
        rpi = forward_batch.req_pool_indices.long().cpu()
        p = int(pos[0])
        k = ks[int(rpi[0]) * ring + p % ring].float().cpu()
        a, b = w["cmp_state"][0, 0, p % ring].float().cpu(), w["cmp_state"][0, 1, p % ring].float().cpu()
        _log(f"probe L{L} state: {tuple(ks.shape)} {ks.dtype}, ring {ring}; ours {tuple(w['cmp_state'].shape)}; kv relRMS "
             f"{_relrms(a, k[:512]):.2e}, score relRMS {_relrms(b, k[512:]):.2e}; p {p}")


def _attn_forward(self, x, positions, forward_batch, x_quant=None, defer_all_reduce=False):
    st = getattr(self, "_dsv41", None)
    if st is None or not _active():
        return _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce)
    mode, rows = forward_batch.forward_mode, x.shape[0]
    sd = _pick(self, rows) if mode.is_decode() and rows > T else st   # (DSV41_STATIC_T: the layer of the smallest T >= rows)
    if sd is not None and mode.is_decode() and _ok_rows(rows) and (rows == sd.T or POOL) and defer_all_reduce:
        st, S = sd, _set(sd.T)
        rpi = forward_batch.req_pool_indices[:rows]
        if not POOL:
            if st.filled == 0 and st.steps == 0:
                raise RuntimeError(f"L{st.layer}: a decode step before any prefill of {T} prompts of the same length")
            if getattr(st, "req", None) is not None and not torch.equal(st.req, rpi):
                raise RuntimeError(f"L{st.layer}: the rows changed: prefill {st.req.tolist()}, decode {rpi.tolist()}")
        if POOL and st.layer == S["first"]:
            fm = _v4.get_attn_backend().forward_metadata
            S["tables"].fill(getattr(fm, "core_attn_metadata", None) or fm.core_metadata, positions, rpi, rows)
        if not _STATE["started"]:
            capturing = torch.cuda.is_current_stream_capturing()
            _log(f"first static decode step (L{st.layer}): " + ("in a CUDA graph capture" if capturing else
                                                                f"positions {positions.view(-1).tolist()}"))
        _STATE["started"] = True
        if MHC:   # the kernel ends the sublayer: run_attn_post (patched) takes its streams and next input
            out = _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce) if CHECK else None
            emb = None
            eg_ids = None
            if getattr(st, "engram", False) and EG_ROWS:   # (DSV41_EG_ROWS=1) the kernel looks the rows up: only the ids
                ids = _STATE.get("hash_ids")
                assert ids is not None and ids.shape[0] == rows, "DSV41_EG_ROWS: no hash ids of this step (EngramHasher.forward)"
                if rows < st.T:   # (DSV41_STATIC_ROWS / _T) the ids of T rows, the idle rows' 0 (a row of the table)
                    pad = S.get("ids_pad")
                    if pad is None or pad.shape[1:] != ids.shape[1:]:
                        pad = S["ids_pad"] = torch.zeros(st.T, *ids.shape[1:], dtype=ids.dtype, device=ids.device)
                    pad[:rows].copy_(ids)
                    ids = pad
                eg_ids = ids[:, st.eg_mod.layer_hash_index]
            elif getattr(st, "engram", False):   # (DSV41_ENGRAM=2) the next layer's n-gram rows by SGLang's own gather, now
                ids = _STATE.get("hash_ids")
                assert ids is not None and ids.shape[0] == rows, "DSV41_ENGRAM=2: no hash ids of this step (EngramHasher.forward)"
                gather = lambda h: h.eg_mod.embed_rows(ids[:, h.eg_mod.layer_hash_index], forward_batch, False).flatten(
                    -2).to(torch.bfloat16).contiguous()
                if EG_EARLY:   # every host's rows before the step's first host kernel (the hash ids are the step's)
                    pre = S.setdefault("eg_early", {})
                    if st.layer not in pre:
                        pre.clear()
                        pre.update({h.layer: gather(h) for h in S["eg_hosts"]})
                    emb = pre.pop(st.layer)
                else:
                    emb = gather(st)
                if emb.shape[0] < st.T:   # (DSV41_STATIC_ROWS / _T) T rows, the idle rows' 0
                    emb = torch.cat([emb, emb.new_zeros(st.T - emb.shape[0], emb.shape[1])])
            y = st.run(x, positions, st.R, emb=emb, ids=eg_ids)
            st.R = None
            _STATE["pending"] = st
            return out if CHECK else _mhc.AttnOutput(y)
        if CHECK:
            out = _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce)
            ref = out.partial if isinstance(out, _mhc.AttnOutput) else out
            y = _run_stats(st, x, positions)
            torch.cuda.synchronize()
            if _rank() == 0 and st.steps <= 4:
                _log(f"check attn L{st.layer} ({st.variant}) step {st.steps} p {positions.view(-1)[:2].tolist()}: y relRMS "
                     f"static vs SGLang {_relrms(y, ref):.3e} (norms {y.float().norm():.4g}, {ref.float().norm():.4g})")
            if PROBE and _rank() == 0 and st.steps <= 2 and st.layer in (0, 2, 3):
                try:
                    _probe(st, positions, forward_batch)
                except Exception as e:   # a probe must not end the run
                    _log(f"probe L{st.layer} failed: {type(e).__name__}: {e}")
            return out if KEEP == "sgl" else _mhc.AttnOutput(y)
        y = _run_stats(st, x, positions)
        return y if AR else _mhc.AttnOutput(y)
    if POOL:
        return _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce)
    if _STATE["started"] and mode.is_decode() and rows > 0:
        raise RuntimeError(f"L{st.layer}: a decode step of {rows} rows after static steps: SGLang's pools lack their rows")
    out = _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce)
    if mode.is_extend() and rows > 0:
        if rows % T:
            _log(f"L{st.layer}: prefill of {rows} rows, not T prompts of the same length: not replayed")
        else:
            torch.cuda.synchronize()
            st.req = forward_batch.req_pool_indices[:T].clone()
            st.replay(x, positions)
            if st.layer < 2:
                _log(f"L{st.layer}: replayed the prefill, {rows} rows, positions {positions.view(-1)[:3].tolist()}...")
    return out


def _run_stats(st, *args):
    """st.run; DSV41_STATIC_STATS: with this sublayer's streams, its triplet for the post (patched run_*_post)."""
    if not STATS:
        return st.run(*args)
    y = st.run(*args, st.R)
    st.R = None
    _STATE["pending"] = st
    return y


def _moe_forward(self, hidden_states, forward_batch=None, *args, **kwargs):
    st = getattr(self, "_dsv41", None)
    rows = hidden_states.shape[0]
    if st is not None and rows > T:   # (DSV41_STATIC_T: the layer of the smallest T >= rows)
        st = _pick(self, rows)
    if st is None or not _active() or forward_batch is None or not forward_batch.forward_mode.is_decode() or \
            not _ok_rows(rows) or (rows < st.T and not MHC):
        return _orig_moe(self, hidden_states, forward_batch, *args, **kwargs)
    if not _ATTNS and not _STATE.get("moe_started"):   # (no static attention on this rank: the MoE layers report the first step)
        _STATE["moe_started"] = True
        _log(f"first static decode step (L{st.layer}): " + ("in a CUDA graph capture" if torch.cuda.is_current_stream_capturing()
                                                             else "eager") + " [MoE layers only]")
    if MHC:   # the kernel ends the sublayer: run_moe_post (patched) takes its streams and next input
        x = hidden_states.clone() if CHECK else hidden_states   # (SGLang's experts may write their output into x)
        out = _orig_moe(self, hidden_states, forward_batch, *args, **kwargs) if CHECK else None
        emb = None
        if st.engram and "eg_emb" in st.w:   # (DSV41_ENGRAM=1) the next layer's n-gram rows by SGLang's own gather, now
            ids = _STATE.get("hash_ids")
            assert ids is not None and ids.shape[0] == st.T, "DSV41_ENGRAM: no hash ids of this step (EngramHasher.forward)"
            m = st.eg_mod
            emb = m.embed_rows(ids[:, m.layer_hash_index], forward_batch, False).flatten(-2).to(torch.bfloat16).contiguous()
        v = st.variants.get(rows, st) if rows < st.T else st   # (DSV41_STATIC_ROWS: the build of these live rows, if any)
        y = v.run(x, st.R, emb=emb)
        if st.engram:
            st.eg_fresh = True   # (this forward's R'': the next Engram layer's forward passes it on, check mode compares it)
        st.R = None
        _STATE["pending"] = v
        return out if CHECK else y
    if CHECK:
        out = _orig_moe(self, hidden_states, forward_batch, *args, **kwargs)
        from sglang.srt.layers.layer_boundary.output import DeferredFinalize, UnreducedOutput, complete_owed
        from sglang.srt.layers.moe.utils import sum_post_experts_output
        if isinstance(out, (DeferredFinalize, UnreducedOutput)):
            ref = complete_owed(out)
        elif isinstance(out, _dv2.MoEOutput):
            c = lambda v: v.clone() if torch.is_tensor(v) else v
            ref = sum_post_experts_output(out._replace(routed=c(out.routed), shared=c(out.shared)).get_merged())
        else:
            ref = out if self.reduce_results else sum_post_experts_output(out.clone())
        torch.cuda.synchronize()   # (everything SGLang queued up to here has finished: a fault below is the static layer's)
        if DUMP:
            os.makedirs(DUMP, exist_ok=True)
            torch.save(hidden_states.cpu(), os.path.join(DUMP, f"x_r{_rank()}_L{st.layer}_s{st.steps}.pt"))
        y = _run_stats(st, hidden_states)
        try:
            torch.cuda.synchronize()
        except Exception as e:
            _log(f"the static MoE L{st.layer} step {st.steps} faulted: {type(e).__name__}")
            raise
        s = y.float().clone()
        torch.distributed.all_reduce(s, group=_STATE["tp_group"])
        if _rank() == 0 and st.steps <= 4:
            _log(f"check MoE L{st.layer} step {st.steps}: y relRMS static vs SGLang {_relrms(s.bfloat16(), ref):.3e} "
                 f"(norms {s.norm():.4g}, {ref.float().norm():.4g})")
        if KEEP == "sgl":
            return out
    else:
        y = _run_stats(st, hidden_states)
    return y if AR or SEAM else _dv2.MergedMoEOutput(y, None, self.experts, 1.0, False)


_OUTS = set()   # data_ptr() of the static MoE layers' y with the all-reduce in the kernel (AR)
_MOES = []      # the static MoE layers of this rank
_ATTNS = []     # the static attention layers of this rank


def report_experts():
    """The last step's routing, per MoE layer: how many distinct experts its 8 tokens used."""
    if not _MOES or "pairs" not in _MOES[0].bufs[0]:   # (STATIC_NO_START_BARRIER: the host allocates the two sets of pairs)
        return
    from deepseek_v4_1_moe import demo as moe_demo
    n = sorted(len(set(moe_demo.pairs_of(m.bufs[0])[0].flatten().tolist())) for m in _MOES)
    print(f"experts last step, {len(n)} MoE layers: min {n[0]} median {n[len(n) // 2]} max {n[-1]}", flush=True)
    from mirage.mpk.static_harness import Harness
    e = [len(set(moe_demo.pairs_of(m.bufs[0])[0].flatten().tolist())) for m in _MOES]
    print("moe_spans", [(ex, round(Harness(m.sk).span_us(), 1)) for ex, m in zip(e, _MOES)], flush=True)   # (experts, span) per layer, the last launch
    print("attn_spans", [round(Harness(a.sk).span_us(), 1) for a in _ATTNS], flush=True)


_orig_moe_prepare = _bd.MoeTransport.prepare_output


def _moe_prepare(self, hidden_states):
    """MoeTransport.prepare_output: a static layer's y is already the sum."""
    if isinstance(hidden_states, torch.Tensor) and hidden_states.data_ptr() in _OUTS:
        self.input = None
        return hidden_states
    return _orig_moe_prepare(self, hidden_states)


if ENGRAM:
    from sglang.srt.layers import engram as _eg
    _orig_hasher = _eg.EngramHasher.forward
    _orig_engram = _eg.Engram.forward

    def _hasher_forward(self, input_ids, forward_batch):
        out = _orig_hasher(self, input_ids, forward_batch)
        _STATE["hash_ids"] = out   # (this step's: the static MoE before an Engram layer gathers its rows with them)
        return out

    def _engram_forward(self, x, hash_ids, forward_batch=None, *, input_ids, cp_all_tokens=False):
        st = getattr(self, "_dsv41_moe", None)
        for c in getattr(self, "_dsv41_moe_t", {}).values():   # (DSV41_STATIC_T: the layer whose R'' this x is)
            if x.data_ptr() == c.w["streams_eg"].data_ptr():
                st = c
        fresh = st is not None and getattr(st, "eg_fresh", False)
        if st is not None:
            st.eg_fresh = False
        if fresh and CHECK and st.steps <= CHECK_STEPS and x.shape[0] == st.T:   # SGLang's Engram on the kernel's R'
            # (every rank: its gather may be a collective), R'' compared on rank 0
            ref = _orig_engram(self, st.w["streams_out"].view(st.T, -1, x.shape[-1]).clone(), hash_ids, forward_batch,
                               input_ids=input_ids, cp_all_tokens=cp_all_tokens)
            torch.cuda.synchronize()
            got = st.w["streams_eg"].view(ref.shape)
            if _rank() == 0:
                _log(f"check Engram L{st.layer + 1} step {st.steps}: R'' relRMS static vs SGLang {_relrms(got, ref):.3e}, "
                     f"equal bits {(got.view(torch.int16) == ref.view(torch.int16)).float().mean().item():.4f}, "
                     f"|R'' - R'| relRMS {_relrms(got, st.w['streams_out'].view(ref.shape)):.3e} (SGLang's: "
                     f"{_relrms(ref, st.w['streams_out'].view(ref.shape)):.3e})")
                if "eg_gamma" in st.w:   # (DSV41_EG_X) the gate's x against SGLang's combine of the gate's own R''
                    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
                    xr = hc_combine_norm(got.reshape(st.T, -1).contiguous(), st.w["h_pre"].view(st.T, -1), st.w["eg_gamma"],
                                         1e-20)
                    _log(f"check Engram x L{st.layer + 1} step {st.steps}: x relRMS {_relrms(st.w['x_out'], xr):.3e}, equal bits "
                         f"{(st.w['x_out'].view(torch.int16) == xr.view(torch.int16)).float().mean().item():.4f}")
        if st is None or x.data_ptr() != st.w["streams_eg"].data_ptr() or not (fresh or torch.cuda.is_current_stream_capturing()):
            # (not the kernel's R'' of this forward: SGLang's Engram)
            return _orig_engram(self, x, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=cp_all_tokens)
        return x

    _eg.EngramHasher.forward = _hasher_forward
    _eg.Engram.forward = _engram_forward
    # (DSV41_EG_X=1) the gate also computed the Engram layer's attention input from R'': HcState.with_residual (SGLang drops
    # the input when Engram rewrites the residual) keeps it when the new residual is that R''
    _orig_with_residual = _mhc.HcState.with_residual

    def _with_residual(self, residual):
        if isinstance(self.input, _mhc.HcNormed) and torch.is_tensor(residual):
            for st in _MOES:
                if "eg_gamma" in st.w and residual.data_ptr() == st.w["streams_eg"].data_ptr() and \
                        self.input.rows.data_ptr() == st.w["x_out"].data_ptr():
                    return _mhc.HcState(residual, self.pre, self.input)
        return _orig_with_residual(self, residual)

    _mhc.HcState.with_residual = _with_residual
def _head_of(t):
    """(DSV41_HEAD) the state (_STATE or a larger T's) whose last MoE kernel's streams (mhc_combine) or head input (the final
    norm) t is, or None"""
    for S in [_STATE] + list(_STATE_SETS.values()):
        st = S.get("head_moe")
        if st is not None and t.data_ptr() in (st.w["streams_out"].data_ptr(), st.w["x_out"].data_ptr()) and \
                _ok_rows(t.shape[0]) and t.shape[0] <= st.T:
            return S
    return None


def _head_x(t):
    """(DSV41_HEAD) the last MoE kernel's head input when t is its streams (mhc_combine) or that input (the final norm)."""
    S = _head_of(t)
    return None if S is None else S["head_moe"].w["x_out"][:t.shape[0]]


def _head_by_logits(lg):
    """(DSV41_HEAD) the head layer (of any T) whose logits buffer lg is, or None"""
    for S in [_STATE] + list(_STATE_SETS.values()):
        hd = S.get("head")
        if hd is not None and lg.data_ptr() == hd.w["logits"].data_ptr():
            return hd
    return None


def _patch_final_norm(norm):
    orig = norm.forward

    def forward(x, *args, **kwargs):
        hx = _head_x(x)
        if hx is not None and x.data_ptr() == hx.data_ptr() and not args and not kwargs:
            return x   # (the kernel's post applied the final norm)
        return orig(x, *args, **kwargs)
    norm.forward = forward


if HEAD:
    from sglang.kernels.ops.layernorm import mhc as _mhcops
    from sglang.srt.layers import logits_processor as _lp
    _orig_mhc_combine = _mhcops.mhc_combine
    _orig_get_logits = _lp.LogitsProcessor._get_logits

    def _mhc_combine(x_flat, pre, hc_mult, dtype, *args, **kwargs):
        hx = _head_x(x_flat)
        if hx is not None and x_flat.data_ptr() != hx.data_ptr():
            if not _STATE.get("head_seen"):
                _STATE["head_seen"] = True
                _log("the head's input from the last MoE kernel (mhc_combine and the final norm passed)")
            return hx
        return _orig_mhc_combine(x_flat, pre, hc_mult, dtype, *args, **kwargs)

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
        S = _head_of(hidden_states)
        hd, hx = (None, None) if S is None else (S.get("head"), _head_x(hidden_states))
        if hd is None or hx is None or hidden_states.data_ptr() != hx.data_ptr():
            return _orig_get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
        assert embedding_bias is None and self.logit_scale is None and not self.final_logit_softcapping, \
            "DSV41_HEAD: no logit scale, bias or softcap"
        buf = logits_metadata.next_token_logits_buffer if use_logits_buffer else None
        if buf is None or tuple(buf.shape) != tuple(hd.w["logits"].shape) or buf.dtype != torch.float32 or \
                not buf.is_contiguous():
            buf = None
        out = hd.w["logits"][:hidden_states.shape[0]] if HEAD2 else hd.run(hidden_states, buf)   # (HEAD2: computed by the
        # last MoE kernel; n rows of a smaller step)
        if HEAD_CHECK and not torch.cuda.is_current_stream_capturing() and hd.steps <= 8:   # SGLang's LM head on the same x
            got = out.clone()
            ref = _orig_get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias, False).float()
            torch.cuda.synchronize()
            if _rank() == 0:
                am = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
                top2 = ref.topk(2, -1).values
                _log(f"check head step {hd.steps}: logits relRMS static vs SGLang {_relrms(got, ref):.3e}, argmax equal {am:.3f}, "
                     f"SGLang's top-2 gap min {(top2[:, 0] - top2[:, 1]).min().item():.3e}" +
                     (f", the kernel's ids equal torch.argmax of its logits "
                      f"{bool(torch.equal(hd.w['head_ids'].long(), got.argmax(-1)))}" if "head_ids" in hd.w else ""))
            out.copy_(got)
        return out

    _mhcops.mhc_combine = _mhc_combine
    _lp.LogitsProcessor._get_logits = _get_logits
_orig_run_moe_post = _mhc.run_moe_post
_orig_run_attn_post = _mhc.run_attn_post
_orig_mix_stats = _mhc.mix_stats
_orig_read = _bd.HcReadCall.read


def _read(self, residual, norm, quant_format="", post_residual_addition=None):
    """HcReadCall.read: the streams this sublayer starts from, for its static layer: only of a decode step it runs (holding a
    prefill's streams would keep 1.3 GB per sublayer alive)."""
    x, state = _orig_read(self, residual, norm, quant_format, post_residual_addition)
    sub = self.layer.self_attn if self.attention else self.layer.mlp
    st = getattr(sub, "_dsv41", None)
    if st is not None and isinstance(state, _bd.ReadState) and self.batch.forward_mode.is_decode() and \
            _ok_rows(state.streams.shape[0]):
        if state.streams.shape[0] > T:   # (DSV41_STATIC_T)
            st = _pick(sub, state.streams.shape[0])
        st.R = state.streams
    return x, state


def _mhc_end(st, orig, hc, out, residual, stats_stream, next, world_size, precomputed, tag):
    """The end of a sublayer whose kernel ran mHC: its state; in check mode SGLang's post too, the two compared, SGLang's kept."""
    if not CHECK:
        return st.hc_state(next, _mhc)
    state = orig(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size, precomputed=precomputed)
    torch.cuda.synchronize()
    if _rank() == 0 and st.steps <= CHECK_STEPS:
        w, rows = st.w, getattr(state.input, "rows", None)
        msg = f"check mHC {tag} L{st.layer} step {st.steps}: R' relRMS {_relrms(w['streams_out'], state.streams.flatten(1)):.2e}, " \
              f"pre {_relrms(w['h_pre'], state.pre):.2e}"
        if "gamma_next" in w and rows is not None:
            msg += f", x {_relrms(w['x_out'], rows):.2e}"
        if CHECK_BITS:   # the static outputs' bits as integer sums: two builds of the same math print the same numbers
            msg += " bits " + " ".join(str(int(w[k].contiguous().view(torch.int16).sum(dtype=torch.int64)))
                                       for k in ("streams_out", "h_pre", "x_out") if k in w)
        _log(msg + f" (next {'yes' if next is not None else 'no'})")
    return state if KEEP == "sgl" else st.hc_state(next, _mhc)


def _stats_triplet(st, hc, residual, tag):
    """The kernel's (pre, post, comb) for SGLang's post; in check mode compared with SGLang's own, and SGLang's kept."""
    tri = st.triplet()
    if not CHECK:
        return tri
    torch.cuda.synchronize()
    if _rank() == 0 and st.steps <= 3:
        ref = _orig_mix_stats(hc, residual, None)
        torch.cuda.synchronize()
        _log(f"check stats {tag} L{st.layer} step {st.steps}: pre {_relrms(tri[0], ref[0]):.2e}, post {_relrms(tri[1], ref[1]):.2e}, "
             f"comb {_relrms(tri[2], ref[2].reshape(tri[2].shape)):.2e}")
    return None if KEEP == "sgl" else tri


def _run_attn_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
    st = _STATE.pop("pending", None) if MHC or STATS else None
    if st is not None and STATS:
        precomputed, st = _stats_triplet(st, hc, residual, "attn"), None
    if st is None:
        return _orig_run_attn_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                                   precomputed=precomputed)
    return _mhc_end(st, _orig_run_attn_post, hc, out, residual, stats_stream, next, world_size, precomputed, "attn")


def _run_moe_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
    """deepseek_v4_mhc.run_moe_post (DSV41_STATIC_SEAM): this rank's part of the static MoE's y takes the attention's seam, one
    SGLang kernel for the all-reduce, the mHC post and the next sublayer's input; where the next sublayer's input is not computed
    there (Engram rewrites the streams first, or the head reads them), the sum, then SGLang's plain post."""
    st = _STATE.pop("pending", None) if MHC or STATS else None
    if st is not None and MHC:
        return _mhc_end(st, _orig_run_moe_post, hc, out, residual, stats_stream, next, world_size, precomputed, "moe")
    if st is not None:
        precomputed = _stats_triplet(st, hc, residual, "moe")
    if SEAM and torch.is_tensor(out) and out.data_ptr() in _OUTS:
        if next is not None:
            return _orig_run_attn_post(hc, _mhc.AttnOutput(out), residual, stats_stream=stats_stream, next=next,
                                      world_size=world_size, precomputed=precomputed)
        from sglang.srt.layers.layer_boundary.output import UnreducedOutput, complete_owed
        from sglang.srt.runtime_context import get_parallel
        out = complete_owed(UnreducedOutput(out, group=get_parallel().tp_group))
    return _orig_run_moe_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                              precomputed=precomputed)


if os.environ.get("DSV41_STATIC_PROF"):   # the static kernels' stamps of the last step, on the driver's request (dsv41_static_prof)
    import dsv41_static_prof as _prof
    _orig_forward = _mr.ModelRunner.forward

    def _forward(self, *args, **kwargs):
        _prof.maybe_sample(_ATTNS, _MOES, _rank(), _STATE.get("meta"))
        return _orig_forward(self, *args, **kwargs)

    _mr.ModelRunner.forward = _forward
_mhc.run_moe_post = _run_moe_post
if MHC or STATS:
    _mhc.run_attn_post = _run_attn_post
    _bd.HcReadCall.read = _read
_bd.MoeTransport.prepare_output = _moe_prepare
_mr.ModelRunner.init_cuda_graphs = _init_cuda_graphs
_v4.MQALayer.forward = _attn_forward
_dv2.DeepseekV2MoE.forward = _moe_forward
# S2-1 (11-18; default off): DSV41_S2_ARGMAX=1: the greedy sampler's torch.argmax(logits, -1) by fast_argmax (the same index,
# torch.argmax's rules: the first maximum, a NaN as the maximum; 2 kernels, 4.3 us GPU at [8, 129280] on a B200 against
# torch.argmax's 25.0 us reduce_kernel). Only sglang.srt.layers.sampler's torch.argmax calls change (its module's `torch`)
S2_ARGMAX = bool(os.environ.get("DSV41_S2_ARGMAX"))
if S2_ARGMAX:
    import fast_argmax as _fa
    import sglang.srt.layers.sampler as _smp
    _fa.install(_smp)
# S2-2 (11-19; default off): DSV41_S2_DEFER=1: greedy decode without the vocab all-gather and the fp32 copy. DSv4.1 has
# no final softcap: stock gathers the bf16 lm head shards (the multimem all-gather, about 10 us) and casts them to fp32
# (6.4 us), then the sampler's argmax. Here decode keeps this rank's shard (cast to fp32: exact), and the decode CUDA graph
# records the greedy ids from the shards the way SGLang's in-graph greedy helper (UocrGreedyGraph) records an argmax: each
# rank its row maxima and first indices (fast_argmax: torch.argmax's rules, a NaN the maximum; nan_to_num first when
# SGLANG_SANITIZE_NAN_LOGITS is on, as the sampler), 2 floats a row all-gathered, then the first rank with the maximum: the
# lowest index among the maxima, torch.argmax over the full row. The sampler takes those ids when the batch replayed the graph
# and is all greedy with nothing that changes the logits (that helper's rules); any other batch gathers its shards first
# (the same fp32 logits as stock). The same tokens by construction
S2_DEFER = os.environ.get("DSV41_S2_DEFER", "0") != "0"
assert not (S2_DEFER and HEAD_ARGMAX), "DSV41_S2_DEFER and DSV41_HEAD_ARGMAX: one source of greedy ids"
if HEAD_ARGMAX:
    from sglang.srt.environ import envs as _envs3
    from sglang.srt.models.unlimited_ocr_v2_support.greedy_graph import UocrGreedyGraph as _UGG3
    _ha_used = [0, 0]   # (batches from the kernel's ids, batches the sampler ran)

    class _HeadGreedy(_UGG3):
        """The greedy ids of a decode graph's replay from the last MoE kernel (the head's gather: hd.w["head_ids"], int32);
        SGLang's in-graph greedy helper's rules for when they are what the sampler would pick (a replayed decode batch, all
        greedy, no logprobs, nothing that changes the logits), and the logits must be the kernel's own"""
        def __init__(self, **kw):
            super().__init__(**kw)
            self.recorded = False

        def capture_tail_hook(self, runner, out, forward_batch, num_tokens):
            lg = None if out is None else out.next_token_logits
            hd = None if lg is None else _head_by_logits(lg)
            if hd is not None and "head_ids" in hd.w:
                self.recorded = True

        def before_forward(self, forward_batch, *, replays_graph, decode_graph_runner):
            self._replayed_batch = forward_batch if replays_graph else None

        def recorded_ids(self, logits_output, forward_batch, *, watermark_state):
            lg = logits_output.next_token_logits
            hd = None if lg is None else _head_by_logits(lg)
            si = forward_batch.sampling_info
            rows = 0 if lg is None or lg.dim() != 2 else lg.shape[0]   # (DSV41_STATIC_ROWS: a step of fewer rows)
            ok = (self.recorded and hd is not None and lg is not None and lg.dim() == 2 and _ok_rows(rows)
                  and lg.data_ptr() == hd.w["logits"].data_ptr() and forward_batch is self._replayed_batch
                  and forward_batch.forward_mode.is_decode() and si.is_all_greedy and not forward_batch.return_logprob
                  and not (watermark_state is not None and si.has_watermark_candidates)
                  and si.logit_bias is None and si.grammar_mask is None and not si.has_custom_logit_processor
                  and not (si.penalizer_orchestrator is not None and si.penalizer_orchestrator.is_required)
                  and not _envs3.SGLANG_SANITIZE_NAN_LOGITS.get())
            _ha_used[0 if ok else 1] += 1
            if sum(_ha_used) in (1, 16, 256):
                _log(f"head argmax: {_ha_used[0]} batches from the kernel's ids, {_ha_used[1]} by the sampler")
            return hd.w["head_ids"][:rows].clone() if ok else None   # (int32: the sampler's contract; a copy: the next replay
            # rewrites the buffer)

    _orig_init_ugg3 = _mr.ModelRunner.maybe_init_uocr_greedy_graph

    def _maybe_init_ugg3(self):
        _orig_init_ugg3(self)
        if self.uocr_greedy_graph is None:
            self.uocr_greedy_graph = _HeadGreedy(max_bs=max([T] + TS), device=self.device)
            self.capture_tail_hooks.append(self.uocr_greedy_graph.capture_tail_hook)
            _log("head argmax: the decode graphs' replays take the last MoE kernel's greedy ids")
    _mr.ModelRunner.maybe_init_uocr_greedy_graph = _maybe_init_ugg3
if S2_DEFER:
    import sglang.srt.layers.logits_processor as _lpm
    from sglang.srt.environ import envs as _envs
    from sglang.srt.models.unlimited_ocr_v2_support.greedy_graph import UocrGreedyGraph as _UGG
    import fast_argmax as _fa2
    _orig_get_logits = _lpm.LogitsProcessor._get_logits
    _s2_seen = []
    _s2_on = []   # (set once this runner's recorder is installed: before it, and without it, decode gathers as before)

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
        # decode with the plain TP gather and no softcap (DSv4.1): this rank's lm head shard only, as fp32
        if (_s2_on and logits_metadata.forward_mode.is_decode() and self.do_tensor_parallel_all_gather and not self.use_attn_tp_group
                and not self.do_tensor_parallel_all_gather_dp_attn and not self.final_logit_softcapping
                and not self.use_tp_lm_head_all_to_all and hidden_states.is_cuda):
            hidden_states, _ = self._gather_dp_attn_hidden_states(hidden_states, logits_metadata)
            with _lpm.get_forward().scoped(lora_batch_layout=_lpm.LoRABatchLayout.DP_LOCAL):
                logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias, out=None)
            if self.logit_scale is not None:
                logits.mul_(self.logit_scale)
            logits = logits.float().contiguous()
            if not _s2_seen:
                _s2_seen.append(1)
                print(f"[dsv41_static rank {_rank()}] S2-2: decode keeps the vocab shard {tuple(logits.shape)}", flush=True)
            return logits
        return _orig_get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
    _lpm.LogitsProcessor._get_logits = _get_logits

    def _shard_argmax(lp, lg):
        """torch.argmax over the full fp32 row, from this rank's shard lg [rows, w] (rank r: columns r w ..)"""
        group = _lpm.get_parallel().tp_group
        rows, w = lg.shape
        off = group.rank_in_group * w
        x = lg[:, : max(1, min(w, lp.vocab_size - off))]
        if _envs.SGLANG_SANITIZE_NAN_LOGITS.get():
            x = torch.nan_to_num(x, nan=-1e30, posinf=1e30, neginf=-1e30)
        idx = _fa2.fast_argmax(x, -1).long()
        local = torch.stack([x.gather(1, idx[:, None]).squeeze(1), (idx + off).float()], dim=1).contiguous()
        g = torch.empty((group.world_size, rows, 2), dtype=torch.float32, device=lg.device)
        group.all_gather_into_tensor(g.view(-1, 2), local)
        best = g[..., 0].argmax(dim=0, keepdim=True)   # (torch.argmax: the first rank with the maximum, a NaN the maximum)
        return g[..., 1].gather(0, best).squeeze(0).long()

    def _shard_gather(lp, lg):
        """the full fp32 logits [rows, vocab] from the shards: stock's cast(gather(bf16)), the cast exact"""
        group = _lpm.get_parallel().tp_group
        rows, w = lg.shape
        g = torch.empty((group.world_size, rows, w), dtype=torch.float32, device=lg.device)
        group.all_gather_into_tensor(g.view(-1, w), lg.contiguous())
        return g.permute(1, 0, 2).reshape(rows, group.world_size * w)[:, : lp.vocab_size].contiguous()

    class _DeferredGreedy(_UGG):
        """UocrGreedyGraph's recorder and rules, with the greedy ids from the vocab shards; they count only once a decode
        graph's capture recorded them"""
        def __init__(self, lp, **kw):
            super().__init__(**kw)
            self.lp = lp
            self.recorded = False

        def capture_tail_hook(self, runner, out, forward_batch, num_tokens):
            if out is None:
                return
            lg = out.next_token_logits
            if lg is not None and lg.dim() == 2 and lg.shape[-1] < self.lp.vocab_size:
                self.ids[: lg.shape[0]].copy_(_shard_argmax(self.lp, lg))
                self.recorded = True

        def before_forward(self, forward_batch, *, replays_graph, decode_graph_runner):
            self._replayed_batch = forward_batch if replays_graph else None

        def recorded_ids(self, logits_output, forward_batch, *, watermark_state):
            lg = logits_output.next_token_logits
            if not self.recorded or lg is None or lg.dim() != 2 or lg.shape[-1] >= self.lp.vocab_size:
                return None   # (full logits: the sampler's own argmax)
            return super().recorded_ids(logits_output, forward_batch, watermark_state=watermark_state)

    _orig_init_ugg = _mr.ModelRunner.maybe_init_uocr_greedy_graph

    def _maybe_init_ugg(self):
        _orig_init_ugg(self)
        if self.uocr_greedy_graph is None:
            bs = int(_mr.get_schedule().max_running_requests or 0) or 64
            self.uocr_greedy_graph = _DeferredGreedy(self.model.logits_processor, max_bs=bs, device=self.device)
            self.capture_tail_hooks.append(self.uocr_greedy_graph.capture_tail_hook)
            _s2_on.append(1)
            print(f"[dsv41_static rank {_rank()}] S2-2: the decode graphs record the deferred greedy ids (max bs {bs})",
                  flush=True)
    _mr.ModelRunner.maybe_init_uocr_greedy_graph = _maybe_init_ugg

    _orig_sample = _mr.ModelRunner.sample
    _s2_used = [0, 0]   # (batches from the recorded ids, batches gathered)

    def _sample(self, logits_output, forward_batch):
        g = self.uocr_greedy_graph
        lg = logits_output.next_token_logits
        if isinstance(g, _DeferredGreedy) and lg is not None and lg.dim() == 2 and lg.shape[-1] < g.lp.vocab_size:
            if g.recorded_ids(logits_output, forward_batch, watermark_state=self.watermark_state) is None:
                logits_output.next_token_logits = _shard_gather(g.lp, lg)   # (not the recorded case: full logits)
                _s2_used[1] += 1
            else:
                _s2_used[0] += 1
            if sum(_s2_used) in (1, 16, 256):
                print(f"[dsv41_static rank {_rank()}] S2-2: {_s2_used[0]} batches from the recorded ids, {_s2_used[1]} gathered",
                      flush=True)
        return _orig_sample(self, logits_output, forward_batch)
    _mr.ModelRunner.sample = _sample
print(f"[dsv41_static] patched SGLang: attention {'all' if ATTN is None else sorted(ATTN)}, MoE "
      f"{'all' if MOE is None else sorted(MOE)} ({'check, keep ' + KEEP if CHECK else 'replace'}, PDL {PDL}"
      f"{', S2 argmax' if S2_ARGMAX else ''}{', S2 deferred greedy' if S2_DEFER else ''}"
      f"{', head argmax' if HEAD_ARGMAX else ''})", flush=True)
