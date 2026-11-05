"""Run chosen GLM-5.3-Flash DSA layers of an SGLang server on the static megakernel, or check them against SGLang's.

boot/sitecustomize.py (on PYTHONPATH in the SGLang runs) imports this module when DSA_STATIC_LAYERS is set (layer ids, e.g.
"3" or "3,7", or "all": every DSA layer). The static layer goes on from the attention's input to the FFN's input (as Kimi K3's
fused layers, #811; dsa_static_layer: the lightning indexer and MLA on SGLang's own state of the layer, o_proj, the TP
all-reduce in the kernel through symmetric memory over the attention TP group, the attention's mHC post, the FFN's mHC pre
with post_attention_layernorm). It patches:
  ModelRunner.init_cuda_graphs      first, for each chosen layer of this rank and each T of DSA_STATIC_TOKENS, builds the static
                                    layer on SGLang's state of the layer (the latent pool and req_to_token, the index-K
                                    buffer of the pooled keys and the tail buffers; its weights: this rank's heads and the
                                    indexer, from the checkpoint); with MTP (the server's speculative_num_draft_tokens) also
                                    the verify layer of T / 8 requests' draft tokens
  Glm5NextDecoderLayer.forward      a step of exactly T rows (a decode step, or a target-verify step of T / 8 requests): after
                                    attn_boundary.prepare (SGLang's attention mHC pre: x, the streams in the forward's
                                    residual stream, h_post / h_res in the layer's MHCState), one launch replaces self_attn +
                                    attn_boundary.finish + ffn_boundary.prepare: the stream gets R', MHCState the FFN's
                                    h_post / h_res, the MLP x_ffn (PDL unless DSA_STATIC_PDL=0; a CUDA graph captures it).
                                    Other steps (prefill, other batch sizes): SGLang's layer; the static layer reads and
                                    writes SGLang's own state, so the two take turns on it
  MHCState.__init__                 lets each decoder layer find its own
check (DSA_STATIC_CHECK=1, the default; eager runs): SGLang's path first (attention, all-reduce, post, pre), then the static
layer on the same inputs; printed: x_ffn, R' - R, h_post, h_res relRMS static vs SGLang, and SGLang's boundary against the
formulas the static layer follows (the static layer's outputs are kept).
Env: DSA_STATIC_LAYERS; DSA_STATIC_CHECK (1 / 0); DSA_STATIC_PDL (1 / 0); DSA_STATIC_TOKENS (8, 16, 32: the T of the layers);
DSA_STATIC_FUSED_PLAN, DSA_STATIC_FUSED_PLAN16, DSA_STATIC_FUSED_PLAN32 (the decode layers' plan files by T: dsa_fused_demo
--search, e.g. dsa_fused_plan_long.json here, or rr, the default; the verify layers run round robin); DSA_STATIC_OUT (build
directory, default <checkout>/build/sgl_static_dsa); DSA_STATIC_BUILD_CACHE (the build cache, static_schedule.compile_static:
every layer and rank has the same layer.cu, compiled once; default <checkout>/build/static_build_cache_dsa); DSA_STATIC_ROOT
(the mirage checkout, default this one).
"""
import itertools
import os

import torch
from dsa_static_layer import ROOT, StaticDSA  # boot/sitecustomize.py put this directory on sys.path
from sglang.srt.model_executor import model_runner as _mr
from sglang.srt.models import deepseek_v2 as _dv2

LAYERS = None if os.environ["DSA_STATIC_LAYERS"].strip() == "all" else \
    {int(x) for x in os.environ["DSA_STATIC_LAYERS"].split(",") if x.strip()}   # None: every DSA layer
CHECK = os.environ.get("DSA_STATIC_CHECK", "1") != "0"
PDL = os.environ.get("DSA_STATIC_PDL", "1") != "0"
FUSED_PLAN = {8: os.environ.get("DSA_STATIC_FUSED_PLAN", "rr"),      # the layers' plans by T (a plan file is of one T's
              16: os.environ.get("DSA_STATIC_FUSED_PLAN16", "rr"),   # graph: the grids differ)
              32: os.environ.get("DSA_STATIC_FUSED_PLAN32", "rr")}
TOKENS = sorted({int(x) for x in os.environ.get("DSA_STATIC_TOKENS", "8").split(",") if x.strip()})   # the layers' T (8, 16, 32):
                                                                                                      # a step of T rows runs
                                                                                                      # the T layer
assert TOKENS and all(t in (8, 16, 32) for t in TOKENS), f"DSA_STATIC_TOKENS {TOKENS}: of 8, 16, 32"
os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE",
                      os.environ.get("DSA_STATIC_BUILD_CACHE", os.path.join(ROOT, "build", "static_build_cache_dsa")))
OUT = os.environ.get("DSA_STATIC_OUT", os.path.join(ROOT, "build", "sgl_static_dsa"))
_orig_init_cuda_graphs = _mr.ModelRunner.init_cuda_graphs


def _pick(layers: dict, rows: int):
    """The layer of exactly `rows` tokens, or None."""
    return layers.get(rows) if isinstance(layers, dict) else None


def _log(msg):
    print(f"[dsa_static rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] {msg}", flush=True)


def _relrms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _fused_args(lay, group) -> dict:
    """StaticDSA's fused argument of decoder layer `lay`: its FFN mHC parameters and post_attention_layernorm."""
    cfg = lay.config
    return {"group": group, "hc_fn": lay.hc_ffn_fn.data, "hc_base": lay.hc_ffn_base.data, "hc_scale": lay.hc_ffn_scale.data,
            "norm": lay.post_attention_layernorm.weight.data, "norm_eps": float(lay.post_attention_layernorm.variance_epsilon),
            "rms_eps": float(cfg.rms_norm_eps), "hc_eps": float(cfg.hc_eps), "iters": int(cfg.hc_sinkhorn_iters)}


def _init_cuda_graphs(self, *args, **kwargs):
    if not getattr(self, "_dsa_static_built", False) and not getattr(self, "is_draft_worker", False):
        self._dsa_static_built = True
        from sglang.srt.distributed.parallel_state import get_attn_tp_group
        from sglang.srt.models import glm5_next as _g5
        from sglang.srt.runtime_context import get_parallel
        rank, tp = int(get_parallel().attn_tp_rank), int(get_parallel().attn_tp_size)
        pool, r2t = self.token_to_kv_pool, self.req_to_token_pool.req_to_token
        group = get_attn_tp_group().device_group
        assert torch.distributed.get_world_size(group) == tp
        layer_of = {id(lay.self_attn): lay for lay in self.model.modules() if isinstance(lay, _g5.Glm5NextDecoderLayer)}
        draft = getattr(self.server_args, "speculative_num_draft_tokens", None)
        for m in self.model.modules():
            if isinstance(m, _dv2.DeepseekV2AttentionMLA) and getattr(m, "use_dsa", False) and not m.is_nextn and \
                    (LAYERS is None or m.layer_id in LAYERS):
                kv = pool._get_key_buffer(m.layer_id) if hasattr(pool, "_get_key_buffer") else pool.get_key_buffer(m.layer_id)
                tk, ts = pool.get_compress_tail_buffers(m.layer_id)
                idx = {"pcache": pool.get_index_k_with_scale_buffer(m.layer_id), "tail_k": tk, "tail_score": ts}
                _log(f"building the static DSA layer {m.layer_id} (TP {rank}/{tp}): latent pool {tuple(kv.shape)} {kv.dtype}, "
                     f"req_to_token {tuple(r2t.shape)}, index-K {tuple(idx['pcache'].shape)}, tails {tuple(idx['tail_k'].shape)}")
                lay = layer_of.get(id(m))
                if lay is None:
                    raise RuntimeError(f"layer {m.layer_id}: no Glm5NextDecoderLayer holds it")
                sts, svs, first = {}, {}, None
                for tok in TOKENS:   # one layer per T, the checkpoint tensors shared
                    st = StaticDSA(m.layer_id, self.server_args.model_path, rank, tp, kv, r2t, OUT, idx, _fused_args(lay, group),
                                   pdl=PDL, fused_plan=FUSED_PLAN[tok], tokens=tok, weights=None if first is None else first.w)
                    first = first or st
                    sts[tok] = st
                    if draft:   # MTP: the verify layer of tok / 8 requests' draft tokens (bs = tok / 8)
                        svs[tok] = StaticDSA(m.layer_id, self.server_args.model_path, rank, tp, kv, r2t, OUT, idx,
                                             _fused_args(lay, group), pdl=PDL, verify=int(draft), tokens=tok,
                                             requests=tok // 8, weights=first.w)
                        _log(f"static DSA layer {m.layer_id}: the verify build of {tok} rows ({tok // 8} requests x {draft} "
                             f"draft tokens)")
                lay._dsa_fused = sts   # the decoder layer's forward runs it; the attention module keeps SGLang's
                lay._dsa_verify = svs or None
                plans = ", ".join(f"{t} rows: plan {os.path.basename(s.plan)}" for t, s in sts.items())
                _log(f"static DSA layer {m.layer_id} ready ({plans})")
        # every rank's layers built (each fills its copy of the exchange regions) before any first launch
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
    return _orig_init_cuda_graphs(self, *args, **kwargs)


def _dsa_meta():
    """The DSA attention backend's metadata of this step (DSAMetadata: cache_seqlens_int32, page_table_1,
    pooled_real_page_table; in a CUDA graph, its static buffers), or None."""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    b = get_attn_backend()
    for _ in range(4):   # wrappers (the hybrid KDA / DSA backend, ...) keep it an attribute down
        m = getattr(b, "forward_metadata", None)
        if m is not None and hasattr(m, "page_table_1") and hasattr(m, "cache_seqlens_int32"):
            return m
        b = next((getattr(b, a) for a in ("full_attn_backend", "attn_backend", "backend", "primary")
                  if getattr(b, a, None) is not None), None)
        if b is None:
            break
    return None


_STEP_KEYS = itertools.count(1)


def _step_key(fb) -> int:
    """A key of this forward (ForwardBatch): the verify layers of a forward share their per-token metadata by it."""
    key = getattr(fb, "_mk_step_key", None)
    if key is None:
        key = fb._mk_step_key = next(_STEP_KEYS)
    return key


def _layer_forward(self, positions, hidden_states, forward_batch, zero_allocator=None, gemm_output_zero_allocator=None,
                   prev_topk_indices=None, capture_output=None):
    """Glm5NextDecoderLayer.forward (SGLang 0.5.22's, the same steps) with the static layer after attn_boundary.prepare
    (the module docstring)."""
    from contextlib import nullcontext
    from sglang.srt.layers.layer_boundary.residual import access as residual_access
    from sglang.srt.layers.layer_boundary.residual.batch import stream_of
    from sglang.srt.models import glm5_next as _g5
    hidden_states_orig = residual_access.buffer(hidden_states)
    hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch, capture=capture_output)
    fb = forward_batch
    capturing = torch.cuda.is_current_stream_capturing()
    mhc = getattr(self, "_dsa_mhc", None)
    rows = hidden_states.shape[0] if isinstance(hidden_states, torch.Tensor) else -1
    st = _pick(getattr(self, "_dsa_fused", None), rows)   # the layer of exactly these rows (a decode step of T requests)
    fused = st is not None and mhc is not None and fb.forward_mode.is_decode() and not (CHECK and capturing)
    # a speculative verify step of bs requests: their draft tokens as rows (the verify build of 8 bs rows)
    bs = int(fb.batch_size) if fb.forward_mode.is_target_verify() else 0
    stv = _pick(getattr(self, "_dsa_verify", None), 8 * bs) if bs else None
    verify = stv is not None and mhc is not None and rows % bs == 0 and 0 < rows // bs <= stv.verify and \
        not (CHECK and capturing)
    if verify:
        st = stv
    topk_indices = None
    if fused or verify:
        T = st.T
        stream = stream_of(fb)
        R, hp, hr = stream.residual, mhc.h_post, mhc.h_res
        seq_lens, rpi = fb.seq_lens[:T], fb.req_pool_indices[:T]
        if CHECK:   # SGLang's path first, on copies of its inputs
            R, hp, hr, x_in = R.clone(), hp.clone(), hr.clone(), hidden_states.clone()
            y = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=fb,
                               zero_allocator=zero_allocator, input_on_attn_tp_slices=self.attn_boundary.input_on_attn_tp_slices,
                               prev_topk_indices=prev_topk_indices)
            y = y[0] if isinstance(y, tuple) else y
            o_sum = y[:rows].float().clone()   # SGLang's attention output summed over the ranks (fp32, then bf16)
            torch.distributed.all_reduce(o_sum, group=st.group)
            o_sum = o_sum.to(torch.bfloat16)
            _g5.get_attn_tp_context().clear_attn_inputs()
            x_ref = self.ffn_boundary.prepare(self.attn_boundary.finish(y, fb), fb).clone()
            R_ref, hp_ref, hr_ref = stream_of(fb).residual.clone(), mhc.h_post.clone(), mhc.h_res.clone()
            hidden_states = x_in
        if verify:
            x_ffn, R2, hp2, hr2 = stv.run_verify(hidden_states, R, hp, hr, fb.seq_lens[:bs], fb.req_pool_indices[:bs],
                                                 step_key=_step_key(fb))
        else:
            x_ffn, R2, hp2, hr2 = st.run_fused(hidden_states, R, hp, hr, seq_lens, rpi, _dsa_meta())
        stream_of(fb).write(R2)
        mhc.h_post, mhc.h_res = hp2, hr2
        if CHECK:
            torch.cuda.synchronize()
            R2, hp2, hr2, x_ffn = R2[:rows], hp2[:rows], hr2[:rows], x_ffn[:rows]   # a verify step: the layer's used rows
            R0 = R[:rows].float().reshape(R2.shape)
            R_ref, hp_ref, hr_ref, x_ref = R_ref[:rows], hp_ref[:rows], hr_ref[:rows], x_ref[:rows]
            # SGLang's boundary against the formulas the static layer follows (mhc.py _mhc_post_torch / _mhc_pre_torch,
            # RMSNorm.forward_native), on SGLang's own summed attention output and its own R': the boundary's share of the
            # differences above (the rest: the attention's)
            import dsa_fused_demo as dfd
            f = st.fused_args
            n = rows
            Rf = dfd.hc_post_ref(o_sum[:n], R[:n].reshape(n, 4, -1), hp[:n].reshape(n, 4), hr[:n].reshape(n, 16))
            xf, pf, cf = dfd.hc_pre_ref(R_ref[:n].reshape(n, 4, -1), f["hc_fn"].float(), f["hc_base"].float(),
                                        f["hc_scale"].float(), f["norm"], f["rms_eps"], f["hc_eps"], f["norm_eps"], f["iters"])
            Rr = R_ref[:n].reshape(n, 4, -1)
            eqR = (Rf.view(torch.int16) == Rr.view(torch.int16)).float().mean().item()
            eqx = (xf.view(torch.int16) == x_ref[:n].view(torch.int16)).float().mean().item()
            _log(f"check layer {st.layer} ({f'verify, {n} rows of {bs}' if verify else f'fused, {n} rows'}): x_ffn relRMS "
                 f"static-SGLang {_relrms(x_ffn, x_ref):.2e} | R' - R "
                 f"{_relrms(R2.float() - R0, R_ref.reshape(R2.shape).float() - R0):.2e} | h_post "
                 f"{_relrms(hp2, hp_ref.reshape(hp2.shape)):.2e} | h_res {_relrms(hr2, hr_ref.reshape(hr2.shape)):.2e} | "
                 f"SGLang's boundary vs the formulas on its own sums: R' {_relrms(Rf, Rr):.2e} (bit-equal {100 * eqR:.2f}%), "
                 f"x_ffn {_relrms(xf, x_ref[:n]):.2e} (bit-equal {100 * eqx:.2f}%), h_post "
                 f"{_relrms(pf, hp_ref[:n].reshape(n, 4)):.2e}, h_res {_relrms(cf, hr_ref[:n].reshape(n, 16)):.2e} | "
                 f"seq_lens {fb.seq_lens[:fb.batch_size].tolist()}")
        else:
            run = st
            seen = run.__dict__.setdefault("_logged", set())
            if capturing not in seen:
                seen.add(capturing)
                _log(f"layer {run.layer}: the {'verify' if verify else 'fused'} static layer of {T} rows replaces SGLang's "
                     f"attention + all-reduce + mHC ({rows} rows, PDL {run.pdl}{', CUDA graph capture' if capturing else ', eager'}; "
                     f"bound: {', '.join(k for k, v in run.binds.items() if v) or 'none'})")
        _g5.get_attn_tp_context().clear_attn_inputs()
        hidden_states = x_ffn
    else:
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch,
                                       zero_allocator=zero_allocator,
                                       input_on_attn_tp_slices=(self.attn_boundary.input_on_attn_tp_slices),
                                       prev_topk_indices=prev_topk_indices)
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        _g5.get_attn_tp_context().clear_attn_inputs()
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
    if isinstance(self.mlp, _g5.Glm5NextMLP):
        gemm_output_zero_allocator = None
    if isinstance(self.mlp, _g5.Glm5NextMoE) and not self.mlp.experts.moe_runner_config.inplace and \
            not torch.compiler.is_compiling() and hidden_states_orig.shape[0] == hidden_states.shape[0]:
        from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx
        _mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
    else:
        _mlp_ctx = nullcontext()
    with _mlp_ctx:
        hidden_states = self.mlp(hidden_states, forward_batch, gemm_output_zero_allocator)
    hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)
    return (hidden_states, topk_indices)


_mr.ModelRunner.init_cuda_graphs = _init_cuda_graphs
from sglang.srt.layers.layer_boundary.residual import mhc as _mhc  # noqa: E402
from sglang.srt.models import glm5_next as _g5  # noqa: E402
_orig_mhc_init = _mhc.MHCState.__init__


def _mhc_init(self, *a, **k):
    """Each decoder layer finds its MHCState (its hc_attn_pre's owner)."""
    _orig_mhc_init(self, *a, **k)
    owner = getattr(self.hc_attn_pre, "__self__", None)
    if owner is not None:
        owner._dsa_mhc = self


_mhc.MHCState.__init__ = _mhc_init
_g5.Glm5NextDecoderLayer.forward = _layer_forward
print(f"[dsa_static] patched SGLang: static DSA layers {'all' if LAYERS is None else sorted(LAYERS)} to the FFN input "
      f"({'check' if CHECK else 'replace'}, PDL {PDL}, T {TOKENS})", flush=True)
