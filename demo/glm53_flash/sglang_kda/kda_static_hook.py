"""Run chosen GLM-5.3-Flash KDA layers of an SGLang server on the static megakernel, or check them against SGLang's.

kda_static_boot.py (imported in every process of the SGLang venv through kda_static_boot.pth) imports this module when
KDA_STATIC_LAYERS is set (layer ids, e.g. "0" or "0,1,2", or "all": every KDA layer). The static layer goes on from the
attention's input to the FFN's input (kda_static_layer, as the DSA hook's, sglang_dsa/dsa_static_hook.py; Kimi K3's #811:
in_proj, kda_decode on SGLang's own KDA pool (conv [slots][3][conv dim], temporal [slots][H][V][K] fp32), o_proj, the TP
all-reduce in the kernel through symmetric memory over the attention TP group, the attention's mHC post, the FFN's mHC pre
with post_attention_layernorm). It patches:
  ModelRunner.init_cuda_graphs      first, for each chosen layer of this rank and each T of KDA_STATIC_TOKENS, builds the static
                                    layer (its weights: this rank's heads, read from the checkpoint, kda_reference); with MTP
                                    (the server's speculative_num_draft_tokens set) also the verify layer (kda_static_layer
                                    verify: T / 8 requests' draft tokens as rows, SGLang's KDA target-verify contract with
                                    ReplaySSM: the pools read only, the rings and the per-token conv windows written; the
                                    server needs --enable-linear-replayssm-spec)
  Glm5NextDecoderLayer.forward      wrapped when the layers are built (so it wraps whatever forward is in place then, e.g. the
                                    DSA hook's): a layer with a static KDA layer, in a step of exactly T rows (a decode step,
                                    or a target-verify step of T / 8 requests), runs SGLang's attn_boundary.prepare, then one
                                    launch replaces self_attn + attn_boundary.finish + ffn_boundary.prepare (the stream gets
                                    R', MHCState the FFN's h_post / h_res, the MLP x_ffn; PDL unless KDA_STATIC_PDL=0; a CUDA
                                    graph captures it); other layers and steps: the forward wrapped
  MHCState.__init__                 lets each decoder layer find its own
check (KDA_STATIC_CHECK=1; eager runs): SGLang's path first (attention, all-reduce, post, pre) on the same states (put back
after), then the static layer; printed: x_ffn, R' - R, h_post, h_res relRMS static vs SGLang, the states (the static layer's
kept); a verify step also the ring (v, k, the decay, beta) and the windows against SGLang's.
Env: KDA_STATIC_LAYERS; KDA_STATIC_CHECK (0 / 1); KDA_STATIC_PDL (1 / 0); KDA_STATIC_TOKENS (8, 16, 32: the T of the layers);
KDA_STATIC_PLAN, KDA_STATIC_PLAN16, KDA_STATIC_PLAN32 (the decode layers' plan files by T, compiler.write_plan_file; default
their round robin); KDA_STATIC_OUT (build directory, default <checkout>/build/sgl_static_kda); KDA_STATIC_BUILD_CACHE (the
layers' build cache, static_schedule.compile_static: every KDA layer and rank has the same layer.cu, compiled once; default
<checkout>/build/static_build_cache_kda); KDA_STATIC_ROOT (the mirage checkout, needed by kda_static_boot).
"""
import os

import torch
from kda_static_layer import ROOT, StaticKDA  # kda_static_boot put this directory on sys.path
from sglang.srt.model_executor import model_runner as _mr
from sglang.srt.models import glm5_next as _g5

LAYERS = None if os.environ["KDA_STATIC_LAYERS"].strip() == "all" else \
    {int(x) for x in os.environ["KDA_STATIC_LAYERS"].split(",") if x.strip()}   # None: every KDA layer
os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE",
                      os.environ.get("KDA_STATIC_BUILD_CACHE", os.path.join(ROOT, "build", "static_build_cache_kda")))
OUT = os.environ.get("KDA_STATIC_OUT", os.path.join(ROOT, "build", "sgl_static_kda"))
PDL = os.environ.get("KDA_STATIC_PDL", "1") != "0"
CHECK = os.environ.get("KDA_STATIC_CHECK", "0") == "1"
PLAN = {8: os.environ.get("KDA_STATIC_PLAN") or None,      # the layers' plan files by T (a plan file is of one T's graph)
        16: os.environ.get("KDA_STATIC_PLAN16") or None,
        32: os.environ.get("KDA_STATIC_PLAN32") or None}
TOKENS = sorted({int(x) for x in os.environ.get("KDA_STATIC_TOKENS", "8").split(",") if x.strip()})   # the layers' T (8, 16, 32):
                                                                                                      # a step of T rows runs
                                                                                                      # the T layer
assert TOKENS and all(t in (8, 16, 32) for t in TOKENS), f"KDA_STATIC_TOKENS {TOKENS}: of 8, 16, 32"


def _pick(layers: dict, rows: int):
    """The layer of exactly `rows` tokens, or None."""
    return layers.get(rows) if isinstance(layers, dict) else None


def _plan_names(plans: dict) -> str:
    """'8: a.json, 16: round robin' of the plan files by T."""
    return ", ".join(f"{t}: {os.path.basename(p) if p else 'round robin'}" for t, p in plans.items() if t in TOKENS)


def _log(msg):
    print(f"[kda_static rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] {msg}", flush=True)


def _kda_backend(forward_batch):
    from sglang.srt.model_executor.forward_context import get_attn_backend
    b = get_attn_backend()   # the step's backend (RadixLinearAttention reads it the same way)
    for _ in range(4):   # wrappers (two-batch overlap, ...) keep the hybrid backend one attribute down
        if hasattr(b, "linear_attn_backend"):
            return b.linear_attn_backend
        b = getattr(b, "backend", None) or getattr(b, "primary", None)
        if b is None:
            break
    raise RuntimeError("kda_static: no linear attention backend on the forward batch")


_orig_init_cuda_graphs = _mr.ModelRunner.init_cuda_graphs


def _head_shard():
    """(rank, size) of the group SGLang splits the KDA heads over (Glm5NextLinearAttention: linear_attn_parallel_group)."""
    from sglang.srt.runtime_context import get_parallel, linear_attn_parallel_group, linear_attn_tp_size
    size = linear_attn_tp_size()
    rank = getattr(get_parallel(), f"{linear_attn_parallel_group()}_rank", None)
    if rank is None:   # no such property: plain TP, one process per rank
        rank = torch.distributed.get_rank() % size
    return int(rank), int(size)


def _fused_args(lay, group) -> dict:
    """StaticKDA's fused argument of decoder layer `lay`: its FFN mHC parameters and post_attention_layernorm (as the fused
    DSA hook's)."""
    cfg = lay.config
    return {"group": group, "hc_fn": lay.hc_ffn_fn.data, "hc_base": lay.hc_ffn_base.data, "hc_scale": lay.hc_ffn_scale.data,
            "norm": lay.post_attention_layernorm.weight.data, "norm_eps": float(lay.post_attention_layernorm.variance_epsilon),
            "rms_eps": float(cfg.rms_norm_eps), "hc_eps": float(cfg.hc_eps), "iters": int(cfg.hc_sinkhorn_iters)}


def _verify_buffers(cache, draft: int, layer: int) -> dict:
    """StaticKDA's verify argument from this layer's Mamba cache: its ReplaySSM rings and its per-token conv windows
    (intermediate_conv_window [rows, draft, 3, 3 HL D] contiguous: causal_conv1d's SAVE_INTERMEDIATE layout, the window after
    token t at (t, 0 .. 2) = its 2 previous inputs and its own)."""
    rings = {k: getattr(cache, f"replayssm_{v}", None) for k, v in (("ring_v", "rawv"), ("ring_k", "rawk"), ("ring_g", "g"),
                                                                     ("ring_beta", "beta"))}
    if any(v is None for v in rings.values()):
        raise RuntimeError(f"layer {layer}: no ReplaySSM rings in the Mamba cache (run SGLang with "
                           f"--enable-linear-replayssm-spec)")
    icw = cache.intermediate_conv_window[0]   # [rows, draft, 3, dim]
    assert icw.dim() == 4 and icw.shape[1] == draft and icw.shape[2] == 3 and icw.is_contiguous(), \
        f"layer {layer}: conv windows {tuple(icw.shape)} strides {icw.stride()}: not [rows, draft, 3, dim] contiguous"
    return {"draft": int(draft), "icw": icw, **{k: v.contiguous() if v.is_contiguous() else v for k, v in rings.items()}}


def _init_cuda_graphs(self, *args, **kwargs):
    if not getattr(self, "_kda_static_built", False) and not getattr(self, "is_draft_worker", False):
        self._kda_static_built = True
        from sglang.srt.distributed.parallel_state import get_attn_tp_group
        rank, tp = _head_shard()
        group = get_attn_tp_group().device_group
        assert torch.distributed.get_world_size(group) == tp, "the KDA heads split over the attention TP group"
        layer_of = {id(lay.self_attn): lay for lay in self.model.modules() if isinstance(lay, _g5.Glm5NextDecoderLayer)}
        draft = getattr(self.server_args, "speculative_num_draft_tokens", None)
        for m in self.model.modules():
            if isinstance(m, _g5.Glm5NextLinearAttention) and (LAYERS is None or m.layer_idx in LAYERS):
                cache = self.req_to_token_pool.mamba2_layer_cache(m.layer_idx)
                lay = layer_of.get(id(m))
                if lay is None:
                    raise RuntimeError(f"layer {m.layer_idx}: no Glm5NextDecoderLayer holds it")
                _log(f"building the static KDA layer {m.layer_idx} (TP {rank}/{tp}, plans {_plan_names(PLAN)})")
                sts, svs, first = {}, {}, None
                for tok in TOKENS:   # one layer per T, the weights shared
                    st = StaticKDA(m.layer_idx, self.server_args.model_path, rank, tp, cache.conv[0], cache.temporal, OUT,
                                   _fused_args(lay, group), pdl=PDL, plan=PLAN[tok], tokens=tok,
                                   weights=None if first is None else first.t)
                    first = first or st
                    st.cache = cache
                    sts[tok] = st
                    if draft:   # MTP: the verify layer of tok / 8 requests' draft tokens (bs = tok / 8)
                        sv = {**_verify_buffers(cache, draft, m.layer_idx), "requests": tok // 8}
                        svs[tok] = StaticKDA(m.layer_idx, self.server_args.model_path, rank, tp, cache.conv[0],
                                             cache.temporal, OUT, _fused_args(lay, group), pdl=PDL,
                                             plan=PLAN[8] if tok == 8 else None,
                                             # (the T >= 16 plans are the decode graph's: kda_decode [16] vs the verify's [2, 16])
                                             verify=sv, weights=first.t, tokens=tok)
                        svs[tok].cache = cache
                        _log(f"static KDA layer {m.layer_idx}: the verify build of {tok} rows ({tok // 8} requests x "
                             f"{draft} draft tokens, ring {sv['ring_v'].shape[2]}, windows {tuple(sv['icw'].shape)})")
                lay._kda_fused = sts   # the decoder layer's forward runs it; the attention module keeps SGLang's
                lay._kda_verify = svs or None
                m._kda_static_cache = cache
                _log(f"static KDA layer {m.layer_idx} ready ({', '.join(f'{t} rows' for t in sts)})")
        # every rank's layers built (each fills its copy of the exchange regions) before any first launch
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        _wrap_layer_forward()
    return _orig_init_cuda_graphs(self, *args, **kwargs)


def _relrms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _wrap_layer_forward():
    """Glm5NextDecoderLayer.forward -> _layer_forward around the forward in place now (the module docstring)."""
    prev = _g5.Glm5NextDecoderLayer.forward
    if getattr(prev, "_kda_fused_wrapper", False):
        return

    def forward(self, positions, hidden_states, forward_batch, zero_allocator=None, gemm_output_zero_allocator=None,
                prev_topk_indices=None, capture_output=None):
        if getattr(self, "_kda_fused", None) is None or getattr(self, "_kda_mhc", None) is None:
            return prev(self, positions, hidden_states, forward_batch, zero_allocator, gemm_output_zero_allocator,
                        prev_topk_indices, capture_output)
        return _layer_forward(self, positions, hidden_states, forward_batch, zero_allocator, gemm_output_zero_allocator,
                              prev_topk_indices, capture_output)
    forward._kda_fused_wrapper = True
    _g5.Glm5NextDecoderLayer.forward = forward


def _layer_forward(self, positions, hidden_states, forward_batch, zero_allocator=None, gemm_output_zero_allocator=None,
                   prev_topk_indices=None, capture_output=None):
    """Glm5NextDecoderLayer.forward (SGLang 0.5.22's, the same steps) of a layer with a static KDA layer: the static layer
    after attn_boundary.prepare in a step of T rows (the module docstring)."""
    from contextlib import nullcontext
    from sglang.srt.layers.layer_boundary.residual import access as residual_access
    from sglang.srt.layers.layer_boundary.residual.batch import stream_of
    hidden_states_orig = residual_access.buffer(hidden_states)
    hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch, capture=capture_output)
    mhc, fb = self._kda_mhc, forward_batch
    capturing = torch.cuda.is_current_stream_capturing()
    rows = hidden_states.shape[0] if isinstance(hidden_states, torch.Tensor) else -1
    st = _pick(self._kda_fused, rows)   # the layer of exactly these rows (a decode step of T requests)
    fused = st is not None and fb.forward_mode.is_decode() and not (CHECK and capturing)
    # a speculative verify step of bs requests: their draft tokens as rows (the verify build of 8 bs rows)
    bs = int(fb.batch_size) if fb.forward_mode.is_target_verify() else 0
    stv = _pick(getattr(self, "_kda_verify", None), 8 * bs) if bs else None
    verify = stv is not None and rows % bs == 0 and 0 < rows // bs <= stv.draft and not (CHECK and capturing)
    if verify:
        st = stv
    topk_indices = None
    if fused or verify:
        T = st.T
        stream = stream_of(fb)
        R, hp, hr = stream.residual, mhc.h_post, mhc.h_res
        backend = _kda_backend(fb)
        idx = backend.forward_metadata.mamba_cache_indices
        if CHECK:   # SGLang's path first, on copies of its inputs and of the states
            conv, temporal = st.cache.conv[0], st.cache.temporal
            sel = idx[:(bs if verify else T)].long()
            live = sel[sel >= 0]
            c0, s0 = conv[live].clone(), temporal[live].clone()
            R, hp, hr, x_in = R.clone(), hp.clone(), hr.clone(), hidden_states.clone()
            y = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=fb,
                               zero_allocator=zero_allocator, input_on_attn_tp_slices=self.attn_boundary.input_on_attn_tp_slices,
                               prev_topk_indices=prev_topk_indices)
            y = y[0] if isinstance(y, tuple) else y
            _g5.get_attn_tp_context().clear_attn_inputs()
            x_ref = self.ffn_boundary.prepare(self.attn_boundary.finish(y, fb), fb).clone()
            R_ref, hp_ref, hr_ref = stream_of(fb).residual.clone(), mhc.h_post.clone(), mhc.h_res.clone()
            c_ref, s_ref = conv[live].clone(), temporal[live].clone()
            if verify:   # SGLang's ring and windows of the first request's slot (verify leaves the pools as they are)
                sv = stv.t
                ring_ref = {k: sv[k][live[0]].clone() for k in ("ring_v", "ring_k", "ring_g", "ring_beta")}
                icw_ref = sv["icw"][int(backend.verify_intermediate_state_indices[0])].clone()
            conv[live], temporal[live] = c0, s0
            hidden_states = x_in
        if verify:
            x_ffn, R2, hp2, hr2 = stv.run_verify(hidden_states, R, hp, hr, idx, backend.verify_intermediate_state_indices)
        else:
            x_ffn, R2, hp2, hr2 = st.run_fused(hidden_states, R, hp, hr, idx)
        stream_of(fb).write(R2)
        mhc.h_post, mhc.h_res = hp2, hr2
        if CHECK:
            torch.cuda.synchronize()
            R0 = R[:rows].float().reshape(R2.shape)
            same = (conv[live].view(torch.int16) == c_ref.view(torch.int16)).float().mean().item()
            extra = ""
            if verify:   # the first request's rows
                sv, s_ = stv.t, live[0]
                n = rows // bs
                eqk = (sv["ring_k"][s_, :, :n].view(torch.int16) == ring_ref["ring_k"][:, :n].view(torch.int16)).float().mean().item()
                eqv = (sv["ring_v"][s_, :, :n].view(torch.int16) == ring_ref["ring_v"][:, :n].view(torch.int16)).float().mean().item()
                icw = sv["icw"][int(backend.verify_intermediate_state_indices[0])]
                eqw = (icw[:n + 2].view(torch.int16) == icw_ref[:n + 2].view(torch.int16)).float().mean().item()
                extra = (f" | ring vs SGLang's: k bit-equal {100 * eqk:.2f}%, v {100 * eqv:.2f}%, decay relRMS "
                         f"{_relrms(sv['ring_g'][s_, :, :n], ring_ref['ring_g'][:, :n]):.2e}, beta "
                         f"{_relrms(sv['ring_beta'][s_, :, :n], ring_ref['ring_beta'][:, :n]):.2e} | conv windows bit-equal "
                         f"{100 * eqw:.2f}%")
            _log(f"check layer {st.layer} ({f'verify, {rows} rows of {bs}' if verify else f'fused, {rows} rows'}): x_ffn relRMS "
                 f"static-SGLang {_relrms(x_ffn, x_ref):.2e} | R' - R "
                 f"{_relrms(R2.float() - R0, R_ref[:rows].reshape(R2.shape).float() - R0):.2e} | h_post "
                 f"{_relrms(hp2, hp_ref[:rows].reshape(hp2.shape)):.2e} | h_res {_relrms(hr2, hr_ref[:rows].reshape(hr2.shape)):.2e} | "
                 f"temporal relRMS {_relrms(temporal[live], s_ref):.2e} | conv equal {100 * same:.3f}% | "
                 f"slots {live.tolist()}{extra}")
        else:
            run = st
            seen = run.__dict__.setdefault("_logged_modes", set())
            if capturing not in seen:
                seen.add(capturing)
                _log(f"layer {st.layer}: the {'verify' if verify else 'fused'} static KDA layer of {T} rows replaces SGLang's "
                     f"attention + all-reduce + mHC ({rows} rows, PDL {run.pdl}{', CUDA graph capture' if capturing else ', eager'}; "
                     f"inputs {run.inputs})")
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
_orig_mhc_init = _mhc.MHCState.__init__


def _mhc_init(self, *a, **k):
    """Each decoder layer finds its MHCState (its hc_attn_pre's owner)."""
    _orig_mhc_init(self, *a, **k)
    owner = getattr(self.hc_attn_pre, "__self__", None)
    if owner is not None:
        owner._kda_mhc = self


_mhc.MHCState.__init__ = _mhc_init
print(f"[kda_static] patched SGLang: static KDA layers {'all' if LAYERS is None else sorted(LAYERS)} to the FFN input, "
      f"PDL {PDL}, check {CHECK}, T {TOKENS}", flush=True)
