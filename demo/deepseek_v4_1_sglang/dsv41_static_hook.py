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
from dsv41_static_layer import ROOT, ExchangePool, StepTables, T, Sources, StaticAttn, StaticMoE, attn_weights
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
assert not (MHC and (AR or SEAM)), "DSV41_STATIC_MHC: no DSV41_STATIC_AR, no DSV41_STATIC_SEAM"
STATS = os.environ.get("DSV41_STATIC_STATS", "0") != "0"   # mHC's coefficients in the kernels; the seam stays SGLang's, no SMs left free
assert not STATS or (SEAM and not MHC), "DSV41_STATIC_STATS needs DSV41_STATIC_SEAM=1 and no DSV41_STATIC_MHC"
DUMP = os.environ.get("DSV41_STATIC_DUMP")   # a directory: the static MoE layers' x of every step
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
        sources = Sources(cfg)
        _STATE["meta"] = {"variants": {lay.layer_id: attn_weights.variant_of(cfg, lay.layer_id) for lay in layers},
                          "engram": [lay.layer_id for lay in layers if lay.engram is not None], "tp": tp}   # (for dsv41_static_prof)
        attn_pool = ExchangePool(_STATE["tp_group"], dev, len(layers)) if AR or MHC else None
        moe_pool = ExchangePool(_STATE["tp_group"], dev, len(layers)) if AR or MHC else None
        self._dsv41_pools = (attn_pool, moe_pool)   # the exchange regions: alive as long as the layers
        steps = None
        if POOL:   # the caps by ratio as StaticAttn's
            caps = {r: max(((MAX_POS + 1) // r + 255) // 256 * 256, 512) for r in set(cfg["compress_ratios"]) if r in (1, 2)}
            steps = _STATE["tables"] = StepTables(dev, caps)
            first = next((lay.layer_id for lay in layers if _chosen(ATTN, lay.layer_id)), None)
            _STATE["first"] = first
        for lay in layers:
            L = lay.layer_id
            if _chosen(ATTN, L):
                lay.self_attn._dsv41 = StaticAttn(L, ckpt, rank, tp, dev, OUT, MAX_POS, sources, PDL, attn_pool, steps, mhc=MHC, stats=STATS)
                _ATTNS.append(lay.self_attn._dsv41)
                if POOL:
                    lay.self_attn._dsv41.bind_pools(self.token_to_kv_pool)
                _log(f"attention L{L} ({lay.self_attn._dsv41.variant}) ready, {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
            if _chosen(MOE, L):
                ex = lay.mlp.experts
                if int(getattr(ex, "moe_ep_size", 1)) != 1:
                    raise RuntimeError("the static MoE is one TP shard of every expert: run SGLang with ep_size 1")
                # the next sublayer's input in the kernel unless Engram rewrites the streams first or the head reads them
                # (DSV41_STATIC_NORM_NEXT=0, measurement 2026-11-17: never; the MoE kernel ends at its mHC post and SGLang
                # computes the next attention's input, as GLM-5.3-Flash does at bs 8 (#821))
                norm_next = MHC and L + 1 < len(layers) and layers[L + 1].engram is None \
                    and os.environ.get("DSV41_STATIC_NORM_NEXT", "1") != "0"
                lay.mlp._dsv41 = StaticMoE(L, ckpt, int(ex.moe_tp_rank), int(ex.moe_tp_size), dev, OUT, PDL, moe_pool, mhc=MHC,
                                           norm_next=norm_next, stats=STATS)
                _MOES.append(lay.mlp._dsv41)
                if AR or SEAM or MHC:
                    _OUTS.add(lay.mlp._dsv41.w["y"].data_ptr())
                _log(f"MoE L{L} ready, shared expert TP1 {getattr(lay.mlp, '_shared_expert_tp1', '?')}, "
                     f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
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
    if mode.is_decode() and rows == T and defer_all_reduce:
        rpi = forward_batch.req_pool_indices[:T]
        if not POOL:
            if st.filled == 0 and st.steps == 0:
                raise RuntimeError(f"L{st.layer}: a decode step before any prefill of {T} prompts of the same length")
            if getattr(st, "req", None) is not None and not torch.equal(st.req, rpi):
                raise RuntimeError(f"L{st.layer}: the rows changed: prefill {st.req.tolist()}, decode {rpi.tolist()}")
        if POOL and st.layer == _STATE["first"]:
            fm = _v4.get_attn_backend().forward_metadata
            _STATE["tables"].fill(getattr(fm, "core_attn_metadata", None) or fm.core_metadata, positions, rpi)
        if not _STATE["started"]:
            capturing = torch.cuda.is_current_stream_capturing()
            _log(f"first static decode step (L{st.layer}): " + ("in a CUDA graph capture" if capturing else
                                                                f"positions {positions.view(-1).tolist()}"))
        _STATE["started"] = True
        if MHC:   # the kernel ends the sublayer: run_attn_post (patched) takes its streams and next input
            out = _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce) if CHECK else None
            y = st.run(x, positions, st.R)
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
    if st is None or not _active() or forward_batch is None or not forward_batch.forward_mode.is_decode() or \
            hidden_states.shape[0] != T:
        return _orig_moe(self, hidden_states, forward_batch, *args, **kwargs)
    if not _ATTNS and not _STATE.get("moe_started"):   # (no static attention on this rank: the MoE layers report the first step)
        _STATE["moe_started"] = True
        _log(f"first static decode step (L{st.layer}): " + ("in a CUDA graph capture" if torch.cuda.is_current_stream_capturing()
                                                             else "eager") + " [MoE layers only]")
    if MHC:   # the kernel ends the sublayer: run_moe_post (patched) takes its streams and next input
        x = hidden_states.clone() if CHECK else hidden_states   # (SGLang's experts may write their output into x)
        out = _orig_moe(self, hidden_states, forward_batch, *args, **kwargs) if CHECK else None
        y = st.run(x, st.R)
        st.R = None
        _STATE["pending"] = st
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


_orig_run_moe_post = _mhc.run_moe_post
_orig_run_attn_post = _mhc.run_attn_post
_orig_mix_stats = _mhc.mix_stats
_orig_read = _bd.HcReadCall.read


def _read(self, residual, norm, quant_format="", post_residual_addition=None):
    """HcReadCall.read: the streams this sublayer starts from, for its static layer: only of a decode step it runs (holding a
    prefill's streams would keep 1.3 GB per sublayer alive)."""
    x, state = _orig_read(self, residual, norm, quant_format, post_residual_addition)
    st = getattr(self.layer.self_attn if self.attention else self.layer.mlp, "_dsv41", None)
    if st is not None and isinstance(state, _bd.ReadState) and self.batch.forward_mode.is_decode() and state.streams.shape[0] == T:
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
print(f"[dsv41_static] patched SGLang: attention {'all' if ATTN is None else sorted(ATTN)}, MoE "
      f"{'all' if MOE is None else sorted(MOE)} ({'check, keep ' + KEEP if CHECK else 'replace'}, PDL {PDL})", flush=True)
