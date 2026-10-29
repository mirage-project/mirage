"""Run chosen GLM-5.3-Flash MoE layers of an SGLang server on the static megakernel, or check them against SGLang's.

sglang_dsa/boot/sitecustomize.py (on PYTHONPATH in the SGLang runs) imports this module when MOE_STATIC_LAYERS is set
(layer ids, e.g. "3" or "3,10", or "all": every MoE layer but the MTP one). The static layer runs to the end of the sublayer
(Kimi K3's scope, PR #811; moe_static_layer): the experts with this rank's own copy of its expert shards from the checkpoint
(about 1.8 GB per layer: run SGLang with a small KV pool, max_total_tokens, so they fit), the TP all-reduce in the kernel
(symmetric memory over the TP group) and the FFN's mHC post with the coefficients SGLang's FFN mHC pre left (MHCState h_post
/ h_res). It patches:
  ModelRunner.init_cuda_graphs   first, for each chosen layer of this rank and each T of MOE_STATIC_TOKENS, builds its GPU of
                                 the TP layer (the layers' exchange regions in one allocation, ExchangePool); with MTP also
                                 the verify build of T / 8 requests x the draft tokens live rows
  DeepseekV2MoE.forward          a step of at most T rows (a decode step, or a target-verify step): one launch on (x, the
                                 residual stream, h_post, h_res) returns the new streams (a CUDA graph captures it; PDL
                                 unless MOE_STATIC_PDL=0); other steps (prefill, other batch sizes): SGLang's layer
  StageBoundary.finish           (the FFN's exit) does with the new streams what SGLang's exit does after its all-reduce and
                                 MHCState.apply_post: the coefficients cleared (the last layer: the streams contracted,
                                 hc_contract), the residual stream written. Only where SGLang's exit is that plain path
                                 (_plain_exit: completed in the layer, all-reduce then the mHC update)
check (MOE_STATIC_CHECK=1, the default; eager runs): SGLang's MoE and its own exit first (on the same inputs; then the stream
and the coefficients put back), then the static layer; printed: the MoE output's share of the new streams (R'_j - sum_i
comb[i][j] R_i = post_j out) static vs SGLang and each vs the exact reference (moe_demo.reference with exact activations,
every rank's shard, fp64, summed over the group: two FP8 MoEs with different activation rounding differ by a few % on their
own, so the exact reference is the yardstick), and R' static vs SGLang.
MOE_STATIC_END=mhc_pre (beyond #811): the kernel also runs the NEXT layer's attention mHC pre (hc_post_mix + hc_pre with the
next decoder layer's hc_attn_* and input_layernorm: moe_demo.build_graph attn_next) and the hook carries its x_attn, h_res,
h_post to the next layer's MHCState (read_attn_input patched: the carry instead of SGLang's hc_attn_pre), so no SGLang kernel
runs between the two layers' kernels; the last layer's MoE ends as mhc. check: also x_attn, h_res, h_post against SGLang's
hc_attn_pre on its own streams.
Env: MOE_STATIC_LAYERS; MOE_STATIC_CHECK (1 / 0); MOE_STATIC_PDL (1 / 0); MOE_STATIC_END (mhc / mhc_pre); MOE_STATIC_TOKENS
(8, 16, 32: the T of the layers); MOE_STATIC_PLAN, MOE_STATIC_PLAN16, MOE_STATIC_PLAN32 (the layers' plan files by T, moe_demo
--search of the same --end, one GPU's lists, e.g. moe_plan_mhc_pre.json here; default: round robin of moe_demo's grids);
MOE_STATIC_OUT (build directory, default <checkout>/build/sgl_static_moe); MOE_STATIC_BUILD_CACHE (the build cache: every
layer and rank has the same layer.cu, compiled once; default <checkout>/build/static_build_cache_moe); MOE_STATIC_ROOT (the
mirage checkout, default this one).
"""
import functools
import os

import torch
from moe_static_layer import ROOT, ExchangePool, StaticMoE  # boot/sitecustomize.py put this directory on sys.path
import moe_demo  # noqa: E402  (moe_static_layer put demo/glm53_flash on sys.path)
from sglang.srt.layers.layer_boundary import stage as _stage
from sglang.srt.layers.layer_boundary.residual import mhc as _mhc
from sglang.srt.layers.layer_boundary.residual.batch import stream_of
from sglang.srt.model_executor import model_runner as _mr
from sglang.srt.models import deepseek_v2 as _dv2

LAYERS = None if os.environ["MOE_STATIC_LAYERS"].strip() == "all" else \
    {int(x) for x in os.environ["MOE_STATIC_LAYERS"].split(",") if x.strip()}   # None: every MoE layer
CHECK = os.environ.get("MOE_STATIC_CHECK", "1") != "0"
PDL = os.environ.get("MOE_STATIC_PDL", "1") != "0"
END = os.environ.get("MOE_STATIC_END", "mhc")
PLAN_FILE = {8: os.environ.get("MOE_STATIC_PLAN") or None,      # the layers' plan files by T (a plan file is of one T's
             16: os.environ.get("MOE_STATIC_PLAN16") or None,   # graph: the grids differ)
             32: os.environ.get("MOE_STATIC_PLAN32") or None}
TOKENS = sorted({int(x) for x in os.environ.get("MOE_STATIC_TOKENS", "8").split(",") if x.strip()})   # the layers' T (8, 16, 32):
                                                                                                      # a step of n rows runs
                                                                                                      # the smallest T >= n
assert TOKENS and all(t in (8, 16, 32) for t in TOKENS), f"MOE_STATIC_TOKENS {TOKENS}: of 8, 16, 32"


def _pick(layers: dict, rows: int):
    """The layer of the smallest T >= rows (its first rows; the others idle), or None."""
    if not isinstance(layers, dict) or rows <= 0:
        return None
    return next((layers[t] for t in sorted(layers) if rows <= t), None)


def _plan_names(plans: dict) -> str:
    """'8: a.json, 16: round robin' of the plan files by T."""
    return ", ".join(f"{t}: {os.path.basename(p) if p else 'round robin'}" for t, p in plans.items() if t in TOKENS)
assert END in ("mhc", "mhc_pre"), f"MOE_STATIC_END {END}: mhc or mhc_pre"
os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE",
                      os.environ.get("MOE_STATIC_BUILD_CACHE", os.path.join(ROOT, "build", "static_build_cache_moe")))
OUT = os.environ.get("MOE_STATIC_OUT", os.path.join(ROOT, "build", "sgl_static_moe"))
_orig_init_cuda_graphs = _mr.ModelRunner.init_cuda_graphs
_orig_forward = _dv2.DeepseekV2MoE.forward
_orig_finish = _stage.StageBoundary.finish
_OUTS = set()   # data_ptr() of the static layers' new-streams tensors (run_mhc returns their first rows): the FFN exit's
                # static path


def _log(msg):
    print(f"[moe_static rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] {msg}", flush=True)


def _relrms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _init_cuda_graphs(self, *args, **kwargs):
    if not getattr(self, "_moe_static_built", False) and not getattr(self, "is_draft_worker", False):
        self._moe_static_built = True
        dev = torch.device("cuda", torch.cuda.current_device())
        mods = [m for m in self.model.modules() if isinstance(m, _dv2.DeepseekV2MoE) and not getattr(m, "is_nextn", False)
                and (LAYERS is None or m.layer_id in LAYERS)]
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.models import glm5_next as _g5
        group = get_tp_group().device_group
        decoder = sorted((lay for lay in self.model.modules() if isinstance(lay, _g5.Glm5NextDecoderLayer)),
                         key=lambda lay: lay.layer_id)
        layer_of = {id(lay.mlp): lay for lay in decoder}
        after = {id(a): b for a, b in zip(decoder, decoder[1:])}   # the decoder layer after each
        draft = getattr(self.server_args, "speculative_num_draft_tokens", None)   # MTP: verify builds too (below)
        # the layers' exchange regions in one allocation of symmetric memory over the TP group (the first take, the largest
        # T, sizes the slots)
        pool = ExchangePool(group, dev, len(mods) * len(TOKENS) * (2 if draft else 1))
        self._moe_exchange_pool = pool   # kept alive with the model runner: the layers' exchange regions are its memory
        for m in mods:
            ex = m.experts
            rank, tp = int(ex.moe_tp_rank), int(ex.moe_tp_size)
            assert int(getattr(ex, "moe_ep_size", 1)) == 1, "the static layer is one TP shard of every expert: no EP"
            assert not getattr(m, "_shared_expert_tp1", False), "the shared expert replicated (TP1): not this layer's sum"
            assert (torch.distributed.get_world_size(group), torch.distributed.get_rank(group)) == (tp, rank), \
                "the TP group's ranks are the MoE's"
            lay = layer_of.get(id(m))
            if lay is None:
                raise RuntimeError(f"layer {m.layer_id}: no Glm5NextDecoderLayer holds the MoE (its FFN exit)")
            m.__dict__["_moe_boundary"] = lay.ffn_boundary   # not a submodule
            free0 = torch.cuda.mem_get_info()[0]
            nxt = after.get(id(lay)) if END == "mhc_pre" else None
            attn_next = None
            if nxt is not None:   # the next layer's attention mHC pre in this kernel
                cfg = nxt.config
                attn_next = {"hc_fn": nxt.hc_attn_fn.data, "hc_base": nxt.hc_attn_base.data, "hc_scale": nxt.hc_attn_scale.data,
                             "norm": nxt.input_layernorm.weight.data,
                             "norm_eps": float(nxt.input_layernorm.variance_epsilon), "rms_eps": float(cfg.rms_norm_eps),
                             "hc_eps": float(cfg.hc_eps), "iters": int(cfg.hc_sinkhorn_iters)}
            sts, svs, first = {}, {}, None
            for tok in sorted(TOKENS, reverse=True):   # one layer per T (the largest first: the pool's slot size), the
                                                       # expert bank shared
                st = StaticMoE(m.layer_id, self.server_args.model_path, rank, tp, dev, OUT, group, pdl=PDL,
                               keep_weights=CHECK, pool=pool, attn_next=attn_next,
                               # a plan file is of one graph: with END=mhc_pre the last layer's graph ends as mhc
                               # (no next layer), another node numbering -> round robin there
                               plan=PLAN_FILE[tok] if END != "mhc_pre" or nxt is not None else None, tokens=tok,
                               bank=None if first is None else first.bank, weights=None if first is None else first.w)
                first = first or st
                st.next_layer = nxt
                sts[tok] = st
                _OUTS.add(st.t["streams_out"].data_ptr())
                if draft and (tok // 8) * draft < tok:   # MTP: the verify build of tok / 8 requests x draft live rows (the
                                                         # rest idle: not routed, so no expert of their own is streamed)
                    rows = (tok // 8) * draft
                    sv = StaticMoE(m.layer_id, self.server_args.model_path, rank, tp, dev, OUT, group, pdl=PDL,
                                   keep_weights=CHECK, pool=pool, attn_next=attn_next,
                                   plan=PLAN_FILE[8] if (END != "mhc_pre" or nxt is not None) and tok == 8 else None,
                                   tokens=tok, bank=first.bank, weights=first.w, rows=rows)
                    sv.next_layer = nxt
                    svs[rows] = sv
                    _OUTS.add(sv.t["streams_out"].data_ptr())
            m._moe_static = sts
            m._moe_static_verify = svs or None   # by live rows (bs x draft)
            torch.cuda.synchronize()
            _log(f"static MoE layer {m.layer_id} ready ({', '.join(f'{t} rows' for t in sorted(sts))}"
                 f"{''.join(f', verify {r} of {sv.T} rows' for r, sv in sorted(svs.items()))}; TP {rank}/{tp}, end "
                 f"{END}, SGLang's fused shared experts "
                 f"{getattr(m, 'num_fused_shared_experts', '?')}): {(free0 - torch.cuda.mem_get_info()[0]) / 2**30:.2f} GB, "
                 f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
        # every rank's layers built (each fills its copy of the exchange regions) before any launch
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
    return _orig_init_cuda_graphs(self, *args, **kwargs)


def _plain_exit(b, fb) -> str:
    """'' when the FFN exit `b` (a StageBoundary) completes this step's output as the static layer's end does (in the layer,
    not deferred: the sum over the group, no transform or move over DP, then the mHC update), else what differs. As FfnExit
    decides: its ExitPolicy (b.plan.output) and that policy's plan."""
    pol = b.plan.output
    steps = pol.plan.path_for(fb)
    upd = steps.output.update
    dec = pol._decide(fb, steps)
    c = dec.complete
    for bad, what in ((b.plan.fusions is not None, "fusions"), (not isinstance(upd, _mhc._FfnUpdate), f"update {upd!r}"),
                      (dec.defer_moe_finalize, "deferred finalize"),
                      (not (isinstance(c, functools.partial) and c.func.__name__ == "_complete_now"), "the sum deferred"),
                      (isinstance(c, functools.partial) and not c.keywords.get("owes_sum"), "no sum owed"),
                      (isinstance(c, functools.partial) and c.keywords.get("dp_step") is not None, "a DP step"),
                      (steps.output.transform is not None, "a transform"), (steps.returns_over_dp, "returns over DP"),
                      (steps.writes_at_handoff, "a pipeline handoff")):
        if bad:
            return what
    return ""


def _finish(self, hidden_states, forward_batch):
    """StageBoundary.finish: for a static layer's new streams, what SGLang's FFN exit does after its sum
    and MHCState.apply_post (exit.py _complete_now, _record_output); else SGLang's."""
    if not (isinstance(hidden_states, torch.Tensor) and hidden_states.data_ptr() in _OUTS):
        return _orig_finish(self, hidden_states, forward_batch)
    state = self.plan.output.plan.path_for(forward_batch).output.update.state
    state.clear_coefficients()
    if state.is_last_layer:
        from sglang.kernels.ops.layernorm.mhc import hc_contract
        hidden_states = hc_contract(hidden_states, state.hc_mult)
    stream = stream_of(forward_batch)
    stream.residual = None
    return stream.write(hidden_states)


def _forward_mhc(self, st, hidden_states, fb, capturing, args, kwargs):
    """A step of the static layer: the new streams (_finish takes them), or SGLang's output where its exit is not the plain
    path."""
    b = self.__dict__["_moe_boundary"]
    why = _plain_exit(b, fb)
    if why:
        if not getattr(st, "_why_logged", False):
            st._why_logged = True
            _log(f"layer {st.layer}: SGLang's FFN exit is not the plain one ({why}): SGLang's layer")
        return _orig_forward(self, hidden_states, fb, *args, **kwargs)
    stream, state = stream_of(fb), b.plan.output.plan.path_for(fb).output.update.state
    R, hp, hr = stream.residual, state.h_post, state.h_res
    nxt = st.next_layer
    if not CHECK:
        y = st.run_mhc(hidden_states, R, hp, hr)
        if nxt is not None:   # the next layer's attention input and coefficients, for its read_attn_input
            y, x_n, hp_n, hr_n = y
            nxt._mk_mhc._mk_carry = (x_n, hr_n, hp_n)
        seen = st.__dict__.setdefault("_logged", set())   # one line for an eager launch, one for a capture
        if capturing not in seen:
            seen.add(capturing)
            _log(f"layer {st.layer}: the static layer{f' of {st.rows} live rows of {st.T}' if st.rows < st.T else ''} "
                 f"replaces SGLang's MoE + all-reduce + mHC post (PDL {st.pdl}"
                 f"{', CUDA graph capture' if capturing else ', eager'}; bound: "
                 f"{', '.join(k for k, v in st.binds.items() if v) or 'none'}; copied: "
                 f"{', '.join(k for k, v in st.binds.items() if not v) or 'none'})")
        return y
    # SGLang's MoE and its own exit on the same inputs, then the stream and the coefficients put back
    x = hidden_states.clone()   # SGLang's experts may write their output into x
    y_part = _orig_forward(self, hidden_states, fb, *args, **kwargs)
    R_sgl = _orig_finish(b, y_part, fb).clone()
    stream.write(R)
    state.h_post, state.h_res = hp, hr
    y = st.run_mhc(x, R, hp, hr)
    torch.cuda.synchronize()
    n = x.shape[0]   # the step's rows (a verify step: fewer than T)
    pre = ""
    if nxt is not None:   # the next layer's attention pre: ours against SGLang's hc_attn_pre on its own new streams
        y, x_n, hp_n, hr_n = y
        nm = nxt.input_layernorm
        x_ref, hr_ref, hp_ref, _ = nxt._mk_mhc.hc_attn_pre(R_sgl.reshape(y.shape).contiguous(), nm.weight,
                                                           float(nm.variance_epsilon))
        eqx = (x_n.view(torch.int16) == x_ref[:n].view(torch.int16)).float().mean().item()
        pre = (f" | next attention pre: x_attn relRMS {_relrms(x_n, x_ref[:n]):.2e} (bit-equal {100 * eqx:.2f}%), h_post "
               f"{_relrms(hp_n, hp_ref[:n].reshape(hp_n.shape)):.2e}, h_res {_relrms(hr_n, hr_ref[:n].reshape(hr_n.shape)):.2e}")
        nxt._mk_mhc._mk_carry = (x_n, hr_n, hp_n)
    Rf = R[:n].float().reshape(n, moe_demo.HC, -1)
    comb, post = hr[:n].float().reshape(n, moe_demo.HC, moe_demo.HC), hp[:n].float().reshape(n, moe_demo.HC)
    mix = torch.einsum("tij,tih->tjh", comb, Rf)   # sum_i comb[i][j] R_i
    exact = moe_demo.reference(x, st.w, act_fp8=False)
    out_exact = (exact["R"].sum(1) + exact["S"]).double()   # this rank's shard, fp64, summed over the group
    torch.distributed.all_reduce(out_exact, group=st.group)
    last = state.is_last_layer
    if last:   # the streams contracted (hc_contract): compare those
        assert nxt is None
        from sglang.kernels.ops.layernorm.mhc import hc_contract
        mine, ref = hc_contract(y, state.hc_mult), R_sgl
        _log(f"check layer {st.layer} (the last: contracted streams): relRMS static-SGLang {_relrms(mine, ref):.2e}")
        return y
    share = lambda v: v[:n].float().reshape(n, moe_demo.HC, -1) - mix   # post_j out
    want = post[..., None].double() * out_exact[:n, None, :]
    eq = (y.view(torch.int16) == R_sgl.reshape(y.shape).view(torch.int16)).float().mean().item()
    _log(f"check layer {st.layer}{f' ({fb.forward_mode.name.lower()}, {n} rows)' if n != st.T else ''}: MoE share of R' "
         f"(post_j out) relRMS static-SGLang {_relrms(share(y), share(R_sgl)):.2e}, "
         f"static-exact {_relrms(share(y), want):.2e}, SGLang-exact {_relrms(share(R_sgl), want):.2e} | R' static-SGLang "
         f"{_relrms(y, R_sgl.reshape(y.shape)):.2e} (bit-equal {100 * eq:.1f}%) | x rms {x.float().pow(2).mean().sqrt().item():.3f}"
         f" | bound: {', '.join(k for k, v in st.binds.items() if v) or 'none'}{pre}")
    return y


def _forward(self, hidden_states, forward_batch=None, *args, **kwargs):
    rows = hidden_states.shape[0] if isinstance(hidden_states, torch.Tensor) else -1
    st = _pick(getattr(self, "_moe_static", None), rows)   # the layer of the smallest T >= rows
    capturing = torch.cuda.is_current_stream_capturing()
    # a decode step (its rows padded to the CUDA graph's batch size: at most T) or a speculative verify step (bs x the draft
    # tokens, e.g. 1 x 6 or 2 x 6 rows; fewer than T rows: the layer's first rows, the others idle)
    mode = forward_batch.forward_mode if forward_batch is not None else None
    if st is None or (CHECK and capturing) or mode is None or not (mode.is_decode() or mode.is_target_verify()) or \
            kwargs.get("return_moe_output"):
        return _orig_forward(self, hidden_states, forward_batch, *args, **kwargs)
    if mode.is_target_verify():   # the verify build of exactly these live rows (bs x draft), when there is one
        st = (getattr(self, "_moe_static_verify", None) or {}).get(rows, st)
    return _forward_mhc(self, st, hidden_states, forward_batch, capturing, args, kwargs)


def _read_attn_input(self, hidden_states, out_norm=None):
    """MHCState.read_attn_input: the carry from the previous layer's static MoE (MOE_STATIC_END=mhc_pre: this layer's x_attn,
    h_res, h_post computed in that kernel), else SGLang's hc_attn_pre."""
    carry = self.__dict__.pop("_mk_carry", None)
    if carry is None:
        return _orig_read_attn_input(self, hidden_states, out_norm)
    x, self.h_res, self.h_post = carry
    return x, hidden_states


_mr.ModelRunner.init_cuda_graphs = _init_cuda_graphs
_dv2.DeepseekV2MoE.forward = _forward
_stage.StageBoundary.finish = _finish
if END == "mhc_pre":   # each decoder layer finds its MHCState (its hc_attn_pre's owner); its read takes the carry
    _orig_mhc_init = _mhc.MHCState.__init__

    def _mhc_init(self, *a, **k):
        _orig_mhc_init(self, *a, **k)
        owner = getattr(self.hc_attn_pre, "__self__", None)
        if owner is not None:
            owner._mk_mhc = self
    _mhc.MHCState.__init__ = _mhc_init
    _orig_read_attn_input = _mhc.MHCState.read_attn_input
    _mhc.MHCState.read_attn_input = _read_attn_input
print(f"[moe_static] patched SGLang: static MoE layers {'all' if LAYERS is None else sorted(LAYERS)} "
      f"({'check' if CHECK else 'replace'}, PDL {PDL}, end {END}, plans {_plan_names(PLAN_FILE)})", flush=True)
