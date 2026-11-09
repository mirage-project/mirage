"""Run the attention layers of DeepSeek-V4.1-Flash's decode steps in an SGLang server on the static megakernel, or check them.

boot/sitecustomize.py imports this module when DSV41_STATIC_ATTN is set (layer ids, or "all"). It patches:
  ModelRunner.init_cuda_graphs  first, builds this rank's chosen layers (dsv41_attn_layer) on SGLang's KV pools
  MQALayer.forward              a decode step of exactly T rows: one launch instead of SGLang's attention, returns
                                mhc.AttnOutput(y), this rank's partial: SGLang's all-reduce and mHC follow as usual
A prefill and a step of any other row count run SGLang's layer on the same pools. Run SGLang with SGLANG_DSV4_KV_LAYOUT=v41_new.

check (DSV41_STATIC_CHECK=1; eager runs): SGLang's layer first, then the static one on the same input; printed by rank 0: y relRMS
static vs SGLang; SGLang's output goes on.
Env: DSV41_STATIC_ATTN, DSV41_STATIC_CHECK, DSV41_STATIC_PDL (0: no programmatic launch), DSV41_STATIC_CKPT (a directory with
inference/config.json and the shards, default the server's model path), DSV41_STATIC_OUT (build directory), DSV41_STATIC_MAX_POS (the
largest position, default 2047), DSV41_STATIC_ROOT.
"""
import os
import time

import torch
from dsv41_attn_layer import ROOT, StaticAttn, StepTables, Sources, T, attn_weights
from sglang.srt.model_executor import model_runner as _mr
from sglang.srt.models import deepseek_v4 as _v4
from sglang.srt.models import deepseek_v4_mhc as _mhc

_v = os.environ["DSV41_STATIC_ATTN"].strip()
LAYERS = None if _v == "all" else {int(x) for x in _v.split(",") if x.strip()}   # None: every layer
CHECK = os.environ.get("DSV41_STATIC_CHECK", "0") != "0"
PDL = os.environ.get("DSV41_STATIC_PDL", "1") != "0"
OUT = os.environ.get("DSV41_STATIC_OUT", os.path.join(ROOT, "build", "sgl_static_dsv41"))
MAX_POS = int(os.environ.get("DSV41_STATIC_MAX_POS", "2047"))
os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE", os.path.join(OUT, "cache"))
_orig_init = _mr.ModelRunner.init_cuda_graphs
_orig_attn = _v4.MQALayer.forward
_STATE = {"started": False, "serving": False}   # serving: past init_cuda_graphs (SGLang's warmup and autotune run dummy steps there)


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _log(msg):
    print(f"[dsv41_static rank {_rank()} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _relrms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _chosen(layer):
    return LAYERS is None or layer in LAYERS


def _init_cuda_graphs(self, *args, **kwargs):
    if not getattr(self, "_dsv41_attn_built", False) and not getattr(self, "is_draft_worker", False):
        self._dsv41_attn_built = True
        from sglang.srt.runtime_context import get_parallel
        ckpt = os.environ.get("DSV41_STATIC_CKPT", self.server_args.model_path)
        cfg = attn_weights.model_config(ckpt)
        rank, tp = int(get_parallel().attn_tp_rank), int(get_parallel().attn_tp_size)
        dev = torch.device("cuda", torch.cuda.current_device())
        layers = sorted((m for m in self.model.modules() if isinstance(m, _v4.DeepseekV4CudaDecoderLayer)),
                        key=lambda m: m.layer_id)
        sources = Sources(cfg)
        caps = {r: max(((MAX_POS + 1) // r + 255) // 256 * 256, 512) for r in set(cfg["compress_ratios"]) if r in (1, 2)}
        tables = _STATE["tables"] = StepTables(dev, caps)
        _STATE["first"] = next((lay.layer_id for lay in layers if _chosen(lay.layer_id)), None)
        for lay in layers:
            L = lay.layer_id
            if _chosen(L):
                lay.self_attn._dsv41 = StaticAttn(L, ckpt, rank, tp, dev, OUT, MAX_POS, sources, PDL, tables)
                lay.self_attn._dsv41.bind_pools(self.token_to_kv_pool)
                _log(f"attention L{L} ({lay.self_attn._dsv41.variant}) ready, {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
        torch.cuda.synchronize()
    out = _orig_init(self, *args, **kwargs)
    _STATE["serving"] = True
    return out


def _attn_forward(self, x, positions, forward_batch, x_quant=None, defer_all_reduce=False):
    st = getattr(self, "_dsv41", None)
    active = _STATE["serving"] or torch.cuda.is_current_stream_capturing()   # (a capture may run inside init_cuda_graphs)
    if st is None or not active or not forward_batch.forward_mode.is_decode() or x.shape[0] != T or not defer_all_reduce:
        return _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce)
    if st.layer == _STATE["first"]:
        fm = _v4.get_attn_backend().forward_metadata
        _STATE["tables"].fill(getattr(fm, "core_attn_metadata", None) or fm.core_metadata, positions,
                              forward_batch.req_pool_indices[:T])
    if not _STATE["started"]:
        _log(f"first static decode step (L{st.layer}): " + ("in a CUDA graph capture" if torch.cuda.is_current_stream_capturing()
                                                             else f"positions {positions.view(-1).tolist()}"))
    _STATE["started"] = True
    if not CHECK:
        return _mhc.AttnOutput(st.run(x))
    out = _orig_attn(self, x, positions, forward_batch, x_quant, defer_all_reduce)
    ref = out.partial if isinstance(out, _mhc.AttnOutput) else out
    y = st.run(x)
    torch.cuda.synchronize()
    if _rank() == 0 and st.steps <= 4:
        _log(f"check attn L{st.layer} ({st.variant}) step {st.steps} p {positions.view(-1)[:2].tolist()}: y relRMS "
             f"static vs SGLang {_relrms(y, ref):.3e} (norms {y.float().norm():.4g}, {ref.float().norm():.4g})")
    return out


_mr.ModelRunner.init_cuda_graphs = _init_cuda_graphs
_v4.MQALayer.forward = _attn_forward
print(f"[dsv41_static] patched SGLang's attention: {'all' if LAYERS is None else sorted(LAYERS)} "
      f"({'check' if CHECK else 'replace'}, PDL {PDL})", flush=True)
