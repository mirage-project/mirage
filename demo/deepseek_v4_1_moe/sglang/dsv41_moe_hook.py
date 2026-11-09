"""Run the MoE layers of DeepSeek-V4.1-Flash's decode steps in an SGLang server on the static megakernel, or check them.

deepseek_v4_1_csa2/sglang/boot/sitecustomize.py imports this module when DSV41_STATIC_MOE is set (layer ids, or "all"). It patches:
  ModelRunner.init_cuda_graphs   first, builds this rank's chosen layers (dsv41_moe_layer)
  DeepseekV2MoE.forward          a decode step of exactly T rows: one launch instead of SGLang's MoE, returns y, this rank's part
  MoeTransport.prepare_output    passes that y on as it is
  deepseek_v4_mhc.run_moe_post   that y takes the attention's seam: one SGLang kernel for the all-reduce, the mHC post and the next
                                 sublayer's input. Where the next sublayer's input is not computed there (Engram rewrites the
                                 streams first, or the head reads them): the sum, then SGLang's plain post

check (DSV41_STATIC_CHECK=1; eager runs): SGLang's layer first, then the static one on the same input; printed by rank 0: y relRMS
static vs SGLang; SGLang's output goes on.
Env: DSV41_STATIC_MOE, DSV41_STATIC_CHECK, DSV41_STATIC_PDL (0: no programmatic launch), DSV41_STATIC_CKPT (a directory with
inference/config.json and the shards, default the server's model path), DSV41_STATIC_OUT (build directory), DSV41_STATIC_ROOT.
"""
import os
import time

import torch
from dsv41_moe_layer import ROOT, StaticMoE, T
from sglang.srt.model_executor import model_runner as _mr
from sglang.srt.models import deepseek_v2 as _dv2
from sglang.srt.models import deepseek_v4 as _v4
from sglang.srt.models import deepseek_v4_boundary as _bd
from sglang.srt.models import deepseek_v4_mhc as _mhc

_v = os.environ["DSV41_STATIC_MOE"].strip()
LAYERS = None if _v == "all" else {int(x) for x in _v.split(",") if x.strip()}   # None: every layer
CHECK = os.environ.get("DSV41_STATIC_CHECK", "0") != "0"
PDL = os.environ.get("DSV41_STATIC_PDL", "1") != "0"
OUT = os.environ.get("DSV41_STATIC_OUT", os.path.join(ROOT, "build", "sgl_static_dsv41"))
os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE", os.path.join(OUT, "cache"))
_orig_init = _mr.ModelRunner.init_cuda_graphs
_orig_moe = _dv2.DeepseekV2MoE.forward
_orig_moe_prepare = _bd.MoeTransport.prepare_output
_orig_run_moe_post = _mhc.run_moe_post
_STATE = {"tp_group": None, "serving": False}   # serving: past init_cuda_graphs (SGLang's warmup and autotune run dummy steps there)
_OUTS = set()   # data_ptr() of the static MoE layers' y


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _log(msg):
    print(f"[dsv41_static rank {_rank()} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _relrms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _init_cuda_graphs(self, *args, **kwargs):
    if not getattr(self, "_dsv41_moe_built", False) and not getattr(self, "is_draft_worker", False):
        self._dsv41_moe_built = True
        from sglang.srt.distributed.parallel_state import get_tp_group
        ckpt = os.environ.get("DSV41_STATIC_CKPT", self.server_args.model_path)
        dev = torch.device("cuda", torch.cuda.current_device())
        _STATE["tp_group"] = get_tp_group().device_group
        layers = sorted((m for m in self.model.modules() if isinstance(m, _v4.DeepseekV4CudaDecoderLayer)),
                        key=lambda m: m.layer_id)
        for lay in layers:
            L = lay.layer_id
            if LAYERS is None or L in LAYERS:
                ex = lay.mlp.experts
                if int(getattr(ex, "moe_ep_size", 1)) != 1:
                    raise RuntimeError("the static MoE is one TP shard of every expert: run SGLang with ep_size 1")
                lay.mlp._dsv41 = StaticMoE(L, ckpt, int(ex.moe_tp_rank), int(ex.moe_tp_size), dev, OUT, PDL)
                _OUTS.add(lay.mlp._dsv41.w["y"].data_ptr())
                _log(f"MoE L{L} ready, {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
        torch.cuda.synchronize()
    out = _orig_init(self, *args, **kwargs)
    _STATE["serving"] = True
    return out


def _moe_forward(self, hidden_states, forward_batch=None, *args, **kwargs):
    st = getattr(self, "_dsv41", None)
    active = _STATE["serving"] or torch.cuda.is_current_stream_capturing()   # (a capture may run inside init_cuda_graphs)
    if st is None or not active or forward_batch is None or not forward_batch.forward_mode.is_decode() or \
            hidden_states.shape[0] != T:
        return _orig_moe(self, hidden_states, forward_batch, *args, **kwargs)
    if not CHECK:
        return st.run(hidden_states)
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
    y = st.run(hidden_states)
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
    return out


def _moe_prepare(self, hidden_states):
    """MoeTransport.prepare_output: a static layer's y goes on as it is."""
    if isinstance(hidden_states, torch.Tensor) and hidden_states.data_ptr() in _OUTS:
        self.input = None
        return hidden_states
    return _orig_moe_prepare(self, hidden_states)


def _run_moe_post(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
    """deepseek_v4_mhc.run_moe_post (the module comment)."""
    if torch.is_tensor(out) and out.data_ptr() in _OUTS:
        if next is not None:
            return _mhc.run_attn_post(hc, _mhc.AttnOutput(out), residual, stats_stream=stats_stream, next=next,
                                      world_size=world_size, precomputed=precomputed)
        from sglang.srt.layers.layer_boundary.output import UnreducedOutput, complete_owed
        from sglang.srt.runtime_context import get_parallel
        out = complete_owed(UnreducedOutput(out, group=get_parallel().tp_group))
    return _orig_run_moe_post(hc, out, residual, stats_stream=stats_stream, next=next, world_size=world_size,
                              precomputed=precomputed)


_mhc.run_moe_post = _run_moe_post
_bd.MoeTransport.prepare_output = _moe_prepare
_mr.ModelRunner.init_cuda_graphs = _init_cuda_graphs
_dv2.DeepseekV2MoE.forward = _moe_forward
print(f"[dsv41_static] patched SGLang's MoE: {'all' if LAYERS is None else sorted(LAYERS)} "
      f"({'check' if CHECK else 'replace'}, PDL {PDL})", flush=True)
