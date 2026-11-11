"""Our Kimi K3 layer kernels (MoE, KDA, MLA; per-rank libraries, amd/tools/gen_rank_lib.py) inside SGLang ROCm v0.5.21's
kimi_k3.py, one TP rank per process. Hooked by amd/serving/sglang_patch.py:
  prepare(model)                      end of KimiK3LinearForCausalLM.load_weights (before SGLang's post-load processing and
                                      the graph capture): the libraries (each a collective: exchange regions over HIP IPC),
                                      the attention layers' weights; for the MoE, SGLang's routed-expert processing of each
                                      layer is replaced by ours (our layout made from its checkpoint-layout parameters, its
                                      own copy freed)
  h = attn(layer, h, forward_batch)   KimiK3DecoderLayer.forward after attn_boundary.prepare (aggregation 1): on a target-
                                      verify step of one request with 8 tokens, ours runs the attention part through
                                      aggregation 2 and returns the MLP input (the stream's residual set to the new prefix,
                                      as ffn_boundary.prepare leaves it); None: SGLang's path
  y = moe(layer, h, prefix, fb)       in place of self.mlp(h, prefix_sum=prefix) on a latent-MoE layer: y = bf16(bf16(moe(h))
                                      + prefix), the next stream; None: SGLang's path
Environment:
  MPK_K3_MOE / MPK_K3_KDA / MPK_K3_MLA   "run": ours (unset: SGLang's)
  MPK_K3_MOEAGG                          "run": on a target-verify step the MoE kernel also computes the next layer's
                                      aggregation 1 (tasks/moe_agg1.cuh; the bank row write of a block-write layer too), and
                                      SGLang's _aggregate_hip call for it returns ours (needs MPK_K3_MOE; the last layer and
                                      every other call: the plain MoE library)
  MPK_K3_KDAMOEAGG / MPK_K3_MLAMOEAGG    "run": the same with the next layer's aggregation 1 at the kernel's end (the
                                      fused libraries built from test_kda_moe_layer.py --agg NB,WRITE,2 with MPK_SPLIT_LOOP);
                                      a call whose bank geometry the kernel does not take (or the last layer): SGLang's
                                      attention and our plain MoE
  MPK_K3_MOEAGGKDA / MPK_K3_MOEAGGMLA    "run": a MoE layer whose next layer is a KDA / MLA layer of ours: the MoE, the
                                      next layer's aggregation 1 and that layer's attention part in one kernel
                                      (test_kda_moe_layer.py --order moe); moe() runs it, the next layer's attn() returns its
                                      hidden (a call that does not match: that layer's attention kernel runs again)
  MPK_K3_KDAMOE / MPK_K3_MLAMOE          "run": the attention part and its layer's MoE in one kernel (the fused libraries,
                                      amd/tests/test_kda_moe_layer.py); on a target-verify step attn() runs both, moe()
                                      returns that kernel's y; needs MPK_K3_MOE (the MoE of every other call)
  MPK_AMD_LIB_MOE / _KDA / _MLA / _KDAMOE / _MLAMOE / _MOEAGG / ...   the libraries (lib.so)
  MPK_K3_LAYERS                          comma-separated layer ids (default: all)
Plain decode (no speculative decoding), per batch size its own libraries:
  MPK_K3_TOKENS                          comma-separated batch sizes (SGLang's captured decode sizes) with libraries
  MPK_AMD_DEC_KDA / _MLA / _MOE          the libraries' paths with {T} for the size (kinds kdad, mla, moe; unset: SGLang's
                                      part); a decode step of a listed size (is_decode(), rows == batch_size) runs ours
  MPK_AMD_DEC_FKDA / _FMLA               the same for a MoE layer, the next layer's aggregation 1 and its KDA / MLA decode
                                      part in one kernel (kinds moeaggkdad / moeaggmla; needs the three above)
The MoE replaces SGLang's experts on every listed layer for every call (SGLang's routed-expert weights are freed): token counts
other than 8 run in chunks of 8 (the last padded with zeros).
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rank_lib import RankKernel  # noqa: E402
import k3_moe_weights as mw  # noqa: E402

KINDS = ("moe", "kda", "mla", "kdamoe", "mlamoe", "moeagg", "kdamoeagg", "mlamoeagg", "moeaggkda",
         "moeaggmla", "moeaggkdamix")   # (moeaggkdamix: moeaggkda + the pre-norm mixture, MPK_AGG_MIX=1's build, at the
                                        # dspark capture layers only)
MODE = {k: os.environ.get(f"MPK_K3_{k.upper()}", "") for k in KINDS}
LIB = {k: os.environ.get(f"MPK_AMD_LIB_{k.upper()}", "") for k in KINDS}
T, H = 8, 7168
DEC_T = sorted({int(x) for x in os.environ.get("MPK_K3_TOKENS", "").split(",") if x})
DEC_LIB = {k: os.environ.get(f"MPK_AMD_DEC_{k.upper()}", "") for k in ("kda", "mla", "moe", "fkda", "fmla", "magg")}
# MPK_AMD_DEC_FKDA_ALT / _FMLA_ALT (path with {T}) + MPK_ALT_PARITY (0 / 1): a paired A / B in one server - the fused MoE + attention
# layers whose MoE layer id has that parity use the ALT library, the others the main one (a second run with the other parity cancels the
# layers' own differences); the trace's per-kernel times split by position parity (amd/serving kdur_alt_an.py)
DEC_ALT = {k: os.environ.get(f"MPK_AMD_DEC_{k.upper()}_ALT", "") for k in ("fkda", "fmla")}
ALT_PARITY = int(os.environ.get("MPK_ALT_PARITY", "0"))
DEC_FUSED = ("fkda", "fmla")   # (kinds moeaggkdad / moeaggmla: a MoE layer + the next layer's aggregation 1 + its decode part)
# MPK_AMD_DEC_MAGG (path with {T}, kind moeagg: test_layer_one_gpu.py --agg 3,0,2): the LAST MoE layer with the model's final
# read (the bank aggregation with output_attn_res_proj / _norm, then the final norm: moe_agg1's math) at the kernel's end;
# SGLang's final _aggregate_hip call returns its output (the forward's last _agg_kernel)
# MPK_K3_DEC_CHECK=1 (debug): an MLA layer's first eager decode call of each size runs ours, keeps its output aside and lets
# SGLang's attention run; moe() then compares SGLang's hidden and prefix with ours (relRMS in the log)
DEC_CHECK = os.environ.get("MPK_K3_DEC_CHECK") == "1"
# MPK_K3_COUNT_EXPERTS=1 (measurement, SGLang's own MoE, eager: --disable-cuda-graph): per MoE call the distinct experts its
# rows picked (topk_ids), logged per row count every 930 calls (median / mean / min / max)
COUNT_EXPERTS = os.environ.get("MPK_K3_COUNT_EXPERTS") == "1"


class _S:
    rank = -1
    kern = {}
    dkern = {}     # (part, T) -> the decode library of that batch size (part: kda, mla, moe)
    dzeros = {}    # T -> zeros [T, H] (the prefix of a layer without one)
    moe = {}
    kda = {}
    mla = {}
    zeros = None
    pending = {}   # layer id -> the fused kernel's y of this step (attn() -> moe())
    attn_pending = {}   # layer id -> the MoE + next KDA kernel's hidden / new prefix of this step (moe() -> attn())
    copy_ok = {}   # layer id -> its next aggregation's prefix_sum (a copy of the MoE's y) checked equal to y
    copy_ok_h = {}   # layer id -> its attention's input (a copy of aggregation 1's output) checked equal to it
    captures = set()   # the layers after which SGLang's dspark captures the stream (dspark_layers_to_capture)
    mix_ready = {}   # MPK_AGG_MIX: layer id -> the MoE kernel's pre-norm mixture of this step (its dspark capture's value)
    mix_ok = {}      # layer id -> that mixture checked equal to SGLang's aggregate_stream in the layer's first eager call
    layers = {}    # layer id -> KimiK3DecoderLayer (MPK_K3_MOEAGG: the next layer's aggregation-1 weights)
    model = None   # the model (MPK_AMD_DEC_MAGG: its final read's weights)
    agg_pending = None   # MPK_K3_MOEAGG: the next _aggregate_hip call's result, computed by the MoE kernel
    logged = set()
    freed = 0
    ecount = {}    # MPK_K3_COUNT_EXPERTS: rows -> distinct experts per MoE call


S = _S()


def _log(msg):
    if S.rank <= 0:
        print(f"[mpk_amd k3] {msg}", file=sys.stderr, flush=True)


def _once(key, msg):
    if key not in S.logged:
        S.logged.add(key)
        _log(msg)


def _wanted(lid):
    spec = os.environ.get("MPK_K3_LAYERS", "")
    return not spec or lid in {int(x) for x in spec.split(",") if x}


def shuffle(w: torch.Tensor) -> torch.Tensor:
    """gemm_tile's shuffled layout of a [N, K] bf16 weight (mirage.mpk.static_megakernel.shuffle_gemm_weight)"""
    N, K = w.shape
    assert N % 16 == 0 and K % 32 == 0, (N, K)
    return w.reshape(N // 16, 16, K // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().reshape(N, K)


# ---------------------------------------------------------------- MoE
def _moe_weights(moe) -> dict:
    """The layer's weights other than the routed experts, from SGLang's modules (bf16 GEMM weights shuffled)."""
    sh = moe.shared_experts
    sgu = sh.gate_up_proj.weight.data
    inv = mw.shared_inverse_order(sgu.shape[0] // 2, sgu.device)
    return {"router_weight": shuffle(moe.gate.weight.data.to(torch.bfloat16)),
            "score_correction_bias": moe.gate.e_score_correction_bias.data.float().contiguous(),
            "latent_down_weight": shuffle(moe.routed_expert_down_proj.weight.data),
            "latent_up_weight": shuffle(moe.routed_expert_up_proj.weight.data),
            "routed_expert_norm_weight": moe.routed_expert_norm.weight.data.contiguous(),
            "shared_gate_up_weight": shuffle(sgu[inv]),
            "shared_down_weight": shuffle(sh.down_proj.weight.data)}


def _free_big(mod) -> int:
    """SGLang's routed-expert tensors of one layer (parameters, buffers, tensor attributes over 64 MB) replaced by empty ones"""
    freed = 0
    for _, t in list(mod.named_parameters(recurse=True)) + list(mod.named_buffers(recurse=True)):
        if t.is_cuda and t.numel() * t.element_size() > (64 << 20):
            freed += t.numel() * t.element_size()
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
    for name, t in list(vars(mod).items()):
        if isinstance(t, torch.Tensor) and t.is_cuda and t.numel() * t.element_size() > (64 << 20):
            freed += t.numel() * t.element_size()
            setattr(mod, name, torch.empty(0, dtype=t.dtype, device=t.device))
    return freed


# MPK_MOE_AITER=1: our libraries built with STATIC_MOE_AITER read SGLang's own expert tensors after AITER's post-load shuffle (the
# same bytes as ours in another order): one copy, SGLang's experts kept, and every non-decode MoE call (prefill) runs SGLang's own
# path (AITER) instead of our decode library in 64-row chunks (MI355X 10-24 srvd61 / 62: that made prefill ~7x slower)
MOE_AITER = os.environ.get("MPK_MOE_AITER") == "1"


def _moe_swap(lid, layer_mod, ex):
    """Our MoE layer from SGLang's loaded (checkpoint-layout) parameters; SGLang's routed experts freed. MOE_AITER: from SGLang's
    tensors after its post-load shuffle, kept."""
    t0 = time.time()
    w = _moe_weights(layer_mod.mlp)
    if MOE_AITER:
        w.update({"w13_blocks": ex.w13_weight.data, "w13_scales": ex.w13_weight_scale.data, "w2_blocks": ex.w2_weight.data,
                  "w2_scales": ex.w2_weight_scale.data})
    else:
        w.update(mw.routed_experts(ex.w13_weight.data, ex.w13_weight_scale.data, ex.w2_weight.data, ex.w2_weight_scale.data))
    lay = S.kern["moe"].create_layer(w) if "moe" in S.kern else None
    dev = w["router_weight"].device
    dec = {t: S.dkern[("moe", t)].create_layer(w) for t in DEC_T if ("moe", t) in S.dkern}
    tc = T if lay is not None else max(dec)   # (the chunk size of the other calls: the verify library's, else the largest)
    S.moe[lid] = dict(layer=lay if lay is not None else dec[tc], tc=tc, w=w,
                      y=torch.empty(tc, H, dtype=torch.bfloat16, device=dev),
                      xpad=torch.zeros(tc, H, dtype=torch.bfloat16, device=dev),
                      ppad=torch.zeros(tc, H, dtype=torch.bfloat16, device=dev),
                      dec=dec, dy={t: torch.empty(t, H, dtype=torch.bfloat16, device=dev) for t in dec})
    _agg_setup(lid, w, dev)
    _fagg_setup(lid, w, dev)
    torch.cuda.synchronize()
    if not MOE_AITER:
        S.freed += _free_big(ex)
        torch.cuda.empty_cache()
    _fuse(lid)
    _fuse_next(lid)
    _once(("moe", lid), f"MoE layer {lid}: ours ({time.time() - t0:.1f} s); SGLang's experts freed ({S.freed / 2**30:.1f} GiB "
                        f"so far), free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB")


AGG_NBMAX = 8   # tasks/moe_agg1.cuh AGG1_NBMAX


def _fagg_setup(lid, w, dev):
    """MPK_AMD_DEC_MAGG: the last MoE layer (no layer after it) with the model's final read in the kernel (kind moeagg per
    batch size): agg_q / agg_gain / agg_ow = output_attn_res_proj / output_attn_res_norm / norm"""
    ts = [t for t in DEC_T if ("magg", t) in S.dkern]
    if not ts or (lid + 1) in S.layers or S.model is None:
        return
    need = ("output_attn_res_proj", "output_attn_res_norm", "norm")
    inner = next((m for m in S.model.modules() if all(hasattr(m, n) for n in need)), None)
    if inner is None:
        _once(("fagg", lid), f"layer {lid}: no final read in the MoE kernel (the model lacks {need})")
        return
    eps = (float(inner.output_attn_res_norm.variance_epsilon), float(inner.norm.variance_epsilon))
    if eps != (1e-5, 1e-5):
        _once(("fagg", lid), f"layer {lid}: no final read in the MoE kernel (eps {eps}; the kernel's are 1e-5)")
        return
    bf = torch.bfloat16
    aw = {"agg_q": inner.output_attn_res_proj.weight.data.reshape(H).to(bf).contiguous(),
          "agg_gain": inner.output_attn_res_norm.weight.data.reshape(H).to(bf).contiguous(),
          "agg_ow": inner.norm.weight.data.reshape(H).to(bf).contiguous()}
    d = S.moe[lid]
    d.update(fagg={t: S.dkern[("magg", t)].create_layer({**w, **aw}) for t in ts}, fproj=inner.output_attn_res_proj,
             fout={t: torch.empty(t, H, dtype=bf, device=dev) for t in ts}, finfo={}, faw=aw)
    _once(("fagg", lid), f"layer {lid}: MoE + the model's final read in one kernel (sizes {ts})")


def _fagg_run(layer, d, h, p, forward_batch, n):
    """the last MoE layer's decode step with the final read (MPK_AMD_DEC_MAGG); None: not here"""
    lid = layer.layer_idx
    st = layer._mpk_bank.require()
    br, nvb = st.block_residual, int(st.num_valid_blocks)
    if br.shape[0] != n or tuple(br.stride()) != (NB_ALLOC * H, H, 1) or not 1 <= nvb <= AGG_NBMAX \
            or getattr(st, "written_on_slice", False):
        _once(("faggno", lid, n), f"layer {lid}: plain MoE in decode, no final read (bank {tuple(br.shape)} rows {nvb})")
        return None
    info = d["finfo"].get(nvb)
    if info is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        info = d["finfo"][nvb] = torch.tensor([nvb, NB_ALLOC * H, H, 0, 0, 0, 0, 0], dtype=torch.int32, device=h.device)
    y = d["dy"][n]
    d["fagg"][n].run({"moe_in": h.contiguous(), "moe_prefix": p.contiguous(), "moe_out": y, "agg_bank": br, "agg_info": info,
                      "agg_out": d["fout"][n]})
    S.agg_pending = dict(y=y.data_ptr(), y_t=y, bank=br.data_ptr(), nvb=nvb, write=False, proj=d["fproj"], out=d["fout"][n],
                         lid=lid, n=n)
    if not torch.cuda.is_current_stream_capturing():
        _once(("faggran", lid, n), f"layer {lid}: MoE + the final read ran in one kernel ({n} rows, {nvb} bank rows)")
    return y


def _agg_setup(lid, w, dev):
    """MPK_K3_MOEAGG: layer lid's MoE with the next layer's aggregation 1 (its score proj / norm and input_layernorm)"""
    nxt = S.layers.get(lid + 1)
    if (not any(k in S.kern for k in ("moeagg", "kdamoeagg", "mlamoeagg", "moeaggkda", "moeaggmla"))
            and not any(p in DEC_FUSED for p, _ in S.dkern)) or nxt is None:
        return
    need = ("self_attention_res_proj", "self_attention_res_norm", "input_layernorm", "is_block_write_layer")
    if not all(hasattr(nxt, n) for n in need):
        _once(("agg", lid), f"layer {lid}: no aggregation 1 in the MoE kernel (layer {lid + 1} lacks {need})")
        return
    eps = (float(nxt.self_attention_res_norm.variance_epsilon), float(nxt.input_layernorm.variance_epsilon))
    if eps != (1e-5, 1e-5):
        _once(("agg", lid), f"layer {lid}: no aggregation 1 in the MoE kernel (eps {eps}; the kernel's are 1e-5)")
        return
    bf = torch.bfloat16
    aw = {"agg_q": nxt.self_attention_res_proj.weight.data.reshape(H).to(bf).contiguous(),
          "agg_gain": nxt.self_attention_res_norm.weight.data.reshape(H).to(bf).contiguous(),
          "agg_ow": nxt.input_layernorm.weight.data.reshape(H).to(bf).contiguous()}
    d = S.moe[lid]
    d.update(agg=S.kern["moeagg"].create_layer({**w, **aw}) if "moeagg" in S.kern else None, aw=aw, next=nxt, agg_info={},
             agg_out=torch.empty(T, H, dtype=bf, device=dev), agg_prefix=torch.empty(T, H, dtype=bf, device=dev),
             agg_bare=torch.empty(T, H, dtype=bf, device=dev),
             agg_mix=None,
             # (plain decode, per size: the fused kernel's aggregation 1 output, the next attention's prefix, y without it)
             dagg={t: dict(out=torch.empty(t, H, dtype=bf, device=dev), prefix=torch.empty(t, H, dtype=bf, device=dev),
                           bare=torch.empty(t, H, dtype=bf, device=dev), info={}) for t in DEC_T})


def _agg_run(layer, d, h, p, forward_batch):
    """the MoE kernel with the next layer's aggregation 1 on a target-verify step of one request; None: not here"""
    a = _agg_io(layer, d, h, p, forward_batch)
    if a is None:
        return None
    d["agg"].run(a[0])
    S.agg_pending = a[1]
    return d["y"]


def _agg_io(layer, d, h, p, forward_batch, y=None):
    """(the run tensors, the pending aggregation) of the MoE kernel with the next layer's aggregation 1; None: not here
    (y: the output the kernel writes, default the MoE's own; a fused attention + MoE kernel passes its own)"""
    if (d.get("agg") is None and y is None) or d.get("aw") is None or forward_batch is None \
            or not forward_batch.forward_mode.is_target_verify() or forward_batch.batch_size != 1:
        return None
    y = d["y"] if y is None else y
    lid, nxt = layer.layer_idx, d["next"]
    st = layer._mpk_bank.require()
    br, nvb, write = st.block_residual, int(st.num_valid_blocks), bool(nxt.is_block_write_layer)
    if br.shape[0] != T or tuple(br.stride()) != (NB_ALLOC * H, H, 1) or not 1 <= nvb <= AGG_NBMAX \
            or (write and nvb >= br.shape[1]) or getattr(st, "written_on_slice", False):
        _once(("aggno", lid), f"layer {lid}: plain MoE (bank {tuple(br.shape)} strides {br.stride()} rows {nvb} write {write})")
        return None
    mdst = _capture_slice(forward_batch, lid) if d.get("mixed") else None   # (the dspark capture's packed buffer slice)
    key = (nvb, write, 0 if mdst is None else mdst.stride(0))
    info = d["agg_info"].get(key)
    if info is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        # (a capture layer: both row strides of its mixture made here, eagerly: the packed buffer's slice is seen first
        # in the graph capture, where no tensor can be made)
        for ms in {key[2], len(S.captures) * H} if d.get("mixed") else {key[2]}:
            d["agg_info"][(nvb, write, ms)] = torch.tensor([nvb, NB_ALLOC * H, H, int(write), 0, ms, 0, 0],
                                                           dtype=torch.int32, device=h.device)
        info = d["agg_info"][key]
    d["mix_dst"] = mdst
    return ({"moe_in": h.contiguous(), "moe_prefix": p.contiguous(), "moe_out": y, "agg_bank": br, "agg_info": info,
             "agg_out": d["agg_out"]},
            dict(y=y.data_ptr(), y_t=y, bank=br.data_ptr(), nvb=nvb, write=write, proj=nxt.self_attention_res_proj,
                 out=d["agg_out"], lid=lid))


def _agg_hook():
    """SGLang's _aggregate_hip (attn_residual.py): the call the MoE kernel already did returns its result"""
    import sglang.srt.layers.attn_residual as ar
    if getattr(ar, "_mpk_agg_hooked", False):
        return
    orig = ar._aggregate_hip

    def hooked(prefix_sum, addend, bank, nvb, score_proj, score_norm, out_norm, write_bank_row=False):
        pend, S.agg_pending = S.agg_pending, None
        if pend is not None and pend.get("bare") is not None and addend is not None and addend.data_ptr() == pend["bare"] \
                and prefix_sum.data_ptr() == pend["res"] and bank.data_ptr() == pend["bank"] and int(nvb) == pend["nvb"] \
                and (bool(write_bank_row) == pend["write"] or (pend["write"] and not write_bank_row)) \
                and score_proj is pend["proj"] and prefix_sum.shape[0] == pend.get("n", T):
            # (the residual + this MLP's output: the kernel's y; a block-write layer's row: the kernel wrote it, and
            # SGLang's unfused path writes the same values after this returns)
            _once(("aggadd", pend["lid"]), f"layer {pend['lid'] + 1}: aggregation 1 (residual + addend) from the MoE kernel")
            return pend["out"], pend["y_t"]
        if pend is not None and addend is None and prefix_sum.data_ptr() != pend["y"] and bank.data_ptr() == pend["bank"] \
                and int(nvb) == pend["nvb"] and (bool(write_bank_row) == pend["write"] or (pend["write"] and not write_bank_row)) \
                and score_proj is pend["proj"] and tuple(prefix_sum.shape) == (pend.get("n", T), H) and prefix_sum.dtype == torch.bfloat16:
            # (5 layers of K3 (8, 24, 52, 68, 84): SGLang aggregates a copy of the MoE's y. Its values checked equal to the
            # kernel's y in the layer's first eager call; a captured call takes the layer's result only if they were)
            lid_ = pend["lid"]
            if not torch.cuda.is_current_stream_capturing() and lid_ not in S.copy_ok:
                S.copy_ok[lid_] = bool(torch.equal(prefix_sum, pend["y_t"]))
                _log(f"layer {lid_ + 1}: aggregation 1 of a copy of y: values {'equal' if S.copy_ok[lid_] else 'DIFFER'}")
            if S.copy_ok.get(lid_):
                return pend["out"], prefix_sum
        if pend is not None and addend is None and prefix_sum.data_ptr() == pend["y"] and bank.data_ptr() == pend["bank"] \
                and int(nvb) == pend["nvb"] and bool(write_bank_row) == pend["write"] and score_proj is pend["proj"] \
                and prefix_sum.shape[0] == pend.get("n", T):
            _once(("agghit", pend["lid"]), f"layer {pend['lid'] + 1}: aggregation 1 from the MoE kernel of layer {pend['lid']}")
            return pend["out"], prefix_sum
        if pend is not None and os.environ.get("MPK_K3_AGG_DEBUG") and ("aggdbg", pend["lid"]) not in S.logged:
            S.logged.add(("aggdbg", pend["lid"]))
            known = {}
            for l_, d_ in list(S.kda.items()) + list(S.mla.items()):
                for n_ in ("new_prefix", "hidden"):
                    known[d_[n_].data_ptr()] = f"attn{l_}.{n_}"
            for l_, d_ in S.moe.items():
                for n_ in ("y", "agg_out", "agg_bare", "agg_prefix"):
                    if d_.get(n_) is not None:
                        known[d_[n_].data_ptr()] = f"moe{l_}.{n_}"
            nm = lambda t: "None" if t is None else known.get(t.data_ptr(), f"other {tuple(t.shape)} {t.dtype}")
            _log(f"[agg debug] layer {pend['lid'] + 1}: prefix_sum = {nm(prefix_sum)}, addend = {nm(addend)}, nvb {nvb}, "
                 f"write {write_bank_row}; pending y = {known.get(pend['y'], pend['y'])}")
        if pend is not None:
            _once(("aggmiss", pend["lid"]), f"layer {pend['lid'] + 1}: aggregation 1 by SGLang (the MoE kernel's did not "
                                            f"match: ptr {prefix_sum.data_ptr() == pend['y']} addend {addend is None} nvb "
                                            f"{nvb}/{pend['nvb']} write {write_bank_row}/{pend['write']})")
        return orig(prefix_sum, addend, bank, nvb, score_proj, score_norm, out_norm, write_bank_row=write_bank_row)
    ar._aggregate_hip = hooked
    ar._mpk_agg_hooked = True


def _capture_slice(forward_batch, lid):
    """layer lid's column slice of SGLang's packed dspark capture buffer (AuxHiddenStatePacker: [T, captures x H] bf16, the
    decode graph runner's), where the MoE kernel writes its pre-norm mixture so the packer has nothing to copy; None: no
    such buffer (eager steps: the packer's own)"""
    buf = getattr(forward_batch, "aux_hidden_states_buffer", None)
    caps = sorted(S.captures)
    if buf is None or lid not in caps or buf.dtype != torch.bfloat16 or tuple(buf.shape) != (T, len(caps) * H) \
            or buf.stride(1) != 1:
        return None
    j = caps.index(lid)
    return buf[:, j * H:(j + 1) * H]


def _mix_refresh():
    """MPK_K3_MOEAGGKDAMIX: the dspark capture layers' MoE + KDA kernels from the build with the pre-norm mixture (agg_mix);
    the capture layers come from SGLang's set_dspark_layers_to_capture, after the model's load"""
    if "moeaggkdamix" not in S.kern:
        return
    for lid in sorted(S.captures):
        m, k = S.moe.get(lid), S.kda.get(lid + 1)
        if m is None or k is None or m.get("mk") is None or m.get("mixed"):
            continue
        m["agg_mix"] = torch.empty(T, H, dtype=torch.bfloat16, device=m["agg_out"].device)
        m["mk"] = S.kern["moeaggkdamix"].create_layer({**m["w"], **m["aw"], **k["w"]})
        m["mixed"] = True
        _once(("mixk", lid), f"layer {lid}: the MoE + KDA kernel with the pre-norm mixture (a dspark capture layer)")


def _dspark_hook():
    """MPK_AGG_MIX: SGLang's dspark capture of the stream after layer i (the next layer's pre-norm aggregation 1,
    aggregate_stream: its snapshot copy, an elementwise op and an _agg_kernel, ~20 us at 5 layers per step) returns the
    MoE kernel's agg_mix when that kernel ran for layer i; its values are checked equal to SGLang's in the layer's first
    eager call, and a captured call takes it only if they were"""
    import sglang.srt.models.kimi_k3 as k3m
    for cls in [c for c in vars(k3m).values() if isinstance(c, type) and "_dspark_capture_stream" in vars(c)]:
        if getattr(cls, "_mpk_dspark_hooked", False):
            continue
        orig = cls._dspark_capture_stream

        def hooked(self, layer_idx, hidden_states, forward_batch, _orig=orig):
            mix = S.mix_ready.pop(layer_idx, None)
            if mix is not None:
                if not torch.cuda.is_current_stream_capturing() and layer_idx not in S.mix_ok:
                    ref = _orig(self, layer_idx, hidden_states, forward_batch)
                    S.mix_ok[layer_idx] = bool(torch.equal(ref, mix))
                    _log(f"layer {layer_idx + 1}: dspark capture = the MoE kernel's pre-norm mixture: values "
                         f"{'equal' if S.mix_ok[layer_idx] else 'DIFFER'}")
                    return ref
                if S.mix_ok.get(layer_idx):
                    _once(("mixhit", layer_idx), f"layer {layer_idx + 1}: dspark capture from the MoE kernel")
                    return mix
            return _orig(self, layer_idx, hidden_states, forward_batch)
        cls._dspark_capture_stream = hooked
        cls._mpk_dspark_hooked = True
    import sglang.srt.layers.aux_hidden_states as ahs
    if not getattr(ahs.AuxHiddenStatePacker, "_mpk_hooked", False):   # (the kernel wrote the slice: no copy)
        orig_append = ahs.AuxHiddenStatePacker.append

        def append_hooked(self, hidden, _orig=orig_append):
            buf, fs = self._buffer, int(hidden.shape[-1])
            if buf is not None and self._feature_size in (None, fs) and tuple(buf.shape[:-1]) == tuple(hidden.shape[:-1]) \
                    and buf.shape[-1] == fs * self._num_captures and buf.dtype == hidden.dtype:
                dst = buf[..., self._idx * fs:(self._idx + 1) * fs]
                if hidden.data_ptr() == dst.data_ptr() and hidden.stride() == dst.stride():
                    self._feature_size = fs
                    self._idx += 1
                    _once(("packin", self._idx), f"dspark capture {self._idx}: written in place by the MoE kernel")
                    return
            return _orig(self, hidden)
        ahs.AuxHiddenStatePacker.append = append_hooked
        ahs.AuxHiddenStatePacker._mpk_hooked = True
    for cls in [c for c in vars(k3m).values() if isinstance(c, type) and "set_dspark_layers_to_capture" in vars(c)]:
        if getattr(cls, "_mpk_capset_hooked", False):
            continue
        orig_set = cls.set_dspark_layers_to_capture

        def set_hooked(self, layer_ids, _orig=orig_set):
            r = _orig(self, layer_ids)
            S.captures = {int(c) for c in layer_ids}
            _log(f"dspark capture layers {sorted(S.captures)}: the MoE + KDA kernels with the pre-norm mixture there")
            _mix_refresh()
            return r
        cls.set_dspark_layers_to_capture = set_hooked
        cls._mpk_capset_hooked = True


def _moe_prepare(model):
    S.layers = {m.layer_idx: m for m in model.modules() if type(m).__name__ == "KimiK3DecoderLayer"}
    S.model = model
    if any(k in S.kern for k in ("moeagg", "kdamoeagg", "mlamoeagg", "moeaggkda", "moeaggmla")) or \
            any(p in DEC_FUSED + ("magg",) for p, _ in S.dkern):
        _agg_hook()
        for mod in model.modules():   # (the dspark capture layers: the model's dspark_layers_to_capture)
            caps = getattr(mod, "dspark_layers_to_capture", None)
            if caps is not None:
                S.captures = {int(c) for c in caps}
        if "moeaggkdamix" in S.kern:
            _dspark_hook()
    n = 0
    for m in model.modules():
        if type(m).__name__ != "KimiK3DecoderLayer" or not _wanted(m.layer_idx):
            continue
        if not getattr(m, "_ffn_writes_stream", False) or type(m.mlp).__name__ != "KimiK3MoE":
            continue
        ex = m.mlp.experts
        qm = getattr(ex, "quant_method", None)
        if qm is None:
            _moe_swap(m.layer_idx, m, ex)
        else:   # SGLang's post-load processing (AITER's shuffle) of this layer replaced by ours
            ex._mpk_layer = (m.layer_idx, m)
            if not getattr(qm, "_mpk_wrapped", False):
                orig = qm.process_weights_after_loading

                def wrapped(module, _orig=orig):
                    tag = getattr(module, "_mpk_layer", None)
                    if tag is None:
                        return _orig(module)
                    if MOE_AITER:   # (AITER's shuffle first: our kernels read its layout)
                        _orig(module)
                    _moe_swap(tag[0], tag[1], module)
                qm.process_weights_after_loading = wrapped
                qm._mpk_wrapped = True
        n += 1
    _log(f"MoE: {n} layers set to ours (at SGLang's post-load step)")


def moe(layer, h, prefix, forward_batch):
    """y = bf16(bf16(moe(h)) + prefix) for every call of a swapped layer (8-token chunks), else None."""
    lid = getattr(layer, "layer_idx", -1)
    chk = getattr(S, "pending_chk", None)
    if chk is not None and chk["lid"] == lid:   # (MPK_K3_DEC_CHECK: SGLang's attention output vs ours, this layer)
        S.pending_chk = None
        rel = lambda a, b: float((a.float() - b.float()).norm() / (b.float().norm() + 1e-30))
        _log(f"[dec check] layer {lid}: hidden relRMS ours vs SGLang {rel(chk['hidden'], h):.3e}; prefix "
             f"{rel(chk['new_prefix'], prefix) if prefix is not None else float('nan'):.3e}; ours[0,:4] "
             f"{chk['hidden'][0, :4].float().tolist()} SGLang {h[0, :4].float().tolist()}")
    f = S.pending.pop(lid, None)
    if f is not None and h.data_ptr() == f["hidden"].data_ptr():   # the fused kernel of this step ran the MoE
        return f["y"]
    d = S.moe.get(lid)
    if d is None:
        return None
    n = h.shape[0]
    if n in d["dec"] and forward_batch is not None and forward_batch.forward_mode.is_decode() \
            and forward_batch.batch_size == n:   # a decode step of a size with its own library
        p = prefix if prefix is not None else S.dzeros[n]
        if (d.get("mkd") or {}).get(n) is not None:   # (the MoE + the next layer's aggregation 1 and decode part)
            if prefix is None:   # (as the verify step: SGLang's next aggregation adds the residual itself)
                from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
                res = residual_batch.written_residual(forward_batch)
                if res is not None and tuple(res.shape) == (n, H) and res.dtype == torch.bfloat16:
                    if _mk_run_dec(layer, d, h, res, forward_batch, n, addend=res) is not None:
                        return d["dagg"][n]["bare"]
            y = _mk_run_dec(layer, d, h, p, forward_batch, n)
            if y is not None:
                return y
        if (d.get("fagg") or {}).get(n) is not None:   # (the last MoE layer with the model's final read)
            y = _fagg_run(layer, d, h, p, forward_batch, n)
            if y is not None:
                return y
        d["dec"][n].run({"moe_in": h.contiguous(), "moe_prefix": p.contiguous(), "moe_out": d["dy"][n]})
        if not torch.cuda.is_current_stream_capturing():
            _once(("dmoe", lid, n), f"layer {lid}: MoE ours (decode, {n} rows)")
        return d["dy"][n]
    if MOE_AITER:   # (prefill and other non-decode calls: SGLang's own MoE on its kept experts)
        if not torch.cuda.is_current_stream_capturing():
            _once(("amoe", lid), f"layer {lid}: MoE SGLang (AITER) for a non-decode call ({n} rows)")
        return None
    p = prefix if prefix is not None else (S.zeros if n == T or d["tc"] == T else S.dzeros[d["tc"]])
    if n == T and d["tc"] == T:
        if prefix is None and d.get("mk") is not None and forward_batch is not None:
            # (an MLP that does not write the stream: SGLang's next aggregation adds the stream's residual to this output
            # itself. The MoE + attention kernel takes that residual as its prefix, so its y, aggregation 1, bank row and
            # the next attention's prefix are SGLang's; this call returns the output without the prefix, agg_bare: if the
            # aggregation call does not match, SGLang's residual + agg_bare is still right)
            from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
            res = residual_batch.written_residual(forward_batch)
            if res is not None and tuple(res.shape) == (T, H) and res.dtype == torch.bfloat16:
                y = _mk_run(layer, d, h, res, forward_batch, addend=res)
                if y is not None:
                    return d["agg_bare"]
        y = _mk_run(layer, d, h, p, forward_batch)
        if y is not None:
            return y
        y = _agg_run(layer, d, h, p, forward_batch)
        if y is not None:
            return y
        d["layer"].run({"moe_in": h.contiguous(), "moe_prefix": p.contiguous(), "moe_out": d["y"]})
        return d["y"]
    out = torch.empty(n, H, dtype=torch.bfloat16, device=h.device)
    tc = d["tc"]
    if p.shape[0] != n:   # (the zeros of a layer without a prefix: tc rows)
        p = torch.zeros(n, H, dtype=torch.bfloat16, device=h.device)
    for i in range(0, n, tc):
        k = min(tc, n - i)
        x, q = h[i:i + k], p[i:i + k]
        if k < tc:
            d["xpad"][:k].copy_(x)
            d["xpad"][k:].zero_()
            d["ppad"][:k].copy_(q)
            d["ppad"][k:].zero_()
            x, q = d["xpad"], d["ppad"]
        d["layer"].run({"moe_in": x.contiguous(), "moe_prefix": q.contiguous(), "moe_out": d["y"]})
        out[i:i + k].copy_(d["y"][:k])
    return out


# ---------------------------------------------------------------- attention parts (KDA, MLA)
NB_ALLOC = 8   # SGLang's bank rows per token ([T, 8, H]: strides 8 H, H)


def _kda_weights(layer) -> dict:
    sa = layer.self_attn
    if float(getattr(sa.attn, "lower_bound", -5.0)) != -5.0:
        raise RuntimeError("KDA gate lower bound: the kernel's is -5")
    bf = torch.bfloat16
    qkvg, fa, b = sa.fused_qkvg_proj.weight.data, sa.f_a_proj.weight.data, sa.b_proj.weight.data
    dev = qkvg.device
    wproj = torch.zeros(6400, H, dtype=bf, device=dev)
    wproj[:6144], wproj[6144:6272], wproj[6272:6284] = qkvg.to(bf), fa.to(bf), b.to(bf)
    conv = sa.qkv_conv1d.weight.data
    prm = torch.cat([sa.A_log.data.reshape(-1).float(), sa.dt_bias.data.reshape(-1).float(),
                     sa.o_norm.weight.data.reshape(-1).float()]).contiguous()
    assert prm.numel() == 12 + 1536 + 128, prm.numel()
    return {"wproj_k": shuffle(wproj), "conv_w": conv.reshape(4608, 4).float().contiguous(),
            "wfb": sa.f_b_proj.weight.data.to(bf).contiguous(), "prm": prm, "oproj_k": shuffle(sa.o_proj.weight.data.to(bf)),
            **_res_weights(layer)}


def _mla_weights(layer) -> dict:
    a = layer.self_attn
    bf = torch.bfloat16
    if abs(float(a.scaling) - 192 ** -0.5) > 1e-9:
        raise RuntimeError(f"MLA scaling {a.scaling}: the kernel's is 1/sqrt(192)")
    wa = a.fused_qkv_a_proj_with_mqa.weight.data
    wa_p = torch.zeros(2176, H, dtype=bf, device=wa.device)
    wa_p[:wa.shape[0]] = wa
    return {"wa_k": shuffle(wa_p), "wq_norm": a.q_a_layernorm.weight.data.to(bf).contiguous(),
            "wkv_norm": a.kv_a_layernorm.weight.data.to(bf).contiguous(), "wqb_k": shuffle(a.q_b_proj.weight.data.to(bf)),
            # (w_kc / w_vc are SGLang's transposed views: the kernel takes [h][l][n] and [h][vd][l], their memory layout)
            "wkc_k": a.w_kc.to(bf).transpose(1, 2).contiguous().reshape(12 * 512, 128),
            "wvc_k": a.w_vc.to(bf).transpose(1, 2).contiguous().reshape(12 * 128, 512),
            "wg_k": shuffle(a.g_proj.weight.data.to(bf)), "wo_k": shuffle(a.o_proj.weight.data.to(bf)), **_res_weights(layer)}


def _res_weights(layer) -> dict:
    bf = torch.bfloat16
    return {"q": layer.mlp_res_proj.weight.data.reshape(H).to(bf).contiguous(),
            "gain": layer.mlp_res_norm.weight.data.reshape(H).to(bf).contiguous(),
            "ow": layer.post_attention_layernorm.weight.data.reshape(H).to(bf).contiguous()}


def _attn_prepare(model):
    for m in model.modules():
        if type(m).__name__ != "KimiK3DecoderLayer" or not _wanted(m.layer_idx):
            continue
        kind = {"KimiK3DeltaAttention": "kda", "KimiK3MLAAttention": "mla"}.get(type(m.self_attn).__name__)
        dts = [t for t in DEC_T if (kind, t) in S.dkern]
        if kind is None or (kind not in S.kern and kind + "moe" not in S.kern and kind + "moeagg" not in S.kern and not dts):
            continue
        fused = (kind + "moe" in S.kern or kind + "moeagg" in S.kern) and type(m.mlp).__name__ == "KimiK3MoE"
        w = _kda_weights(m) if kind == "kda" else _mla_weights(m)
        dev = w["q"].device
        d = dict(kind=kind, layer=None if fused or kind not in S.kern else S.kern[kind].create_layer(w), w=w, fused=fused,
                 dec={t: S.dkern[(kind, t)].create_layer(w) for t in dts},
                 dh={t: torch.empty(t, H, dtype=torch.bfloat16, device=dev) for t in dts},
                 dnp={t: torch.empty(t, H, dtype=torch.bfloat16, device=dev) for t in dts},
                 hidden=torch.empty(T, H, dtype=torch.bfloat16, device=dev),
                 new_prefix=torch.empty(T, H, dtype=torch.bfloat16, device=dev), info={},
                 y=torch.empty(T, H, dtype=torch.bfloat16, device=dev) if fused else None)
        (S.kda if kind == "kda" else S.mla)[m.layer_idx] = d
        if fused:
            _fuse(m.layer_idx)
    for lid in list(S.moe):   # (MoE layers swapped before the attention parts were set up)
        _fuse_next(lid)
    _log(f"attention parts set to ours: KDA layers {len(S.kda)}, MLA layers {len(S.mla)} (fused with the MoE: "
         f"{sum(d['fused'] for d in (*S.kda.values(), *S.mla.values()))})")


def _fuse(lid):
    """the fused layer (attention part + MoE) once both weight sets are there (the MoE's come at SGLang's post-load step)"""
    d, m = S.kda.get(lid) or S.mla.get(lid), S.moe.get(lid)
    if d is None or not d["fused"] or d["layer"] is not None or m is None:
        return
    k_agg = d["kind"] + "moeagg"
    if k_agg in S.kern and m.get("aw") is not None:
        d["layer"] = S.kern[k_agg].create_layer({**d["w"], **m["w"], **m["aw"]})
        d["fused_agg"] = True
    elif d["kind"] + "moe" in S.kern:
        d["layer"] = S.kern[d["kind"] + "moe"].create_layer({**d["w"], **m["w"]})
    elif d["kind"] in S.kern:   # (the last layer with only the agg library: the attention part alone)
        d["layer"], d["fused"] = S.kern[d["kind"]].create_layer(d["w"]), False
        return
    else:
        return
    _once(("fused", lid), f"layer {lid}: {d['kind']} + MoE{' + aggregation 1' if d.get('fused_agg') else ''} in one kernel")


def _fuse_next(lid):
    """MPK_K3_MOEAGGKDA / _MOEAGGMLA: MoE layer lid with the next layer's aggregation 1 and attention part in one kernel,
    once the MoE's weights (SGLang's post-load step) and the attention part's are both there"""
    m, k = S.moe.get(lid), S.kda.get(lid + 1) or S.mla.get(lid + 1)
    if m is not None and k is not None and m.get("aw") is not None and not m.get("mkd"):   # (plain decode, per size)
        ts = [t for t in DEC_T if ("f" + k["kind"], t) in S.dkern and t in k["dec"] and t in m["dec"]]
        if ts:
            # (MLA layers are every 4th layer, all one parity: alternate them by lid // 4)
            sel = (lid // 4 if k["kind"] == "mla" else lid) % 2
            alt = lambda t: ("f" + k["kind"] + "_alt", t) if sel == ALT_PARITY and ("f" + k["kind"] + "_alt", t) in S.dkern \
                else ("f" + k["kind"], t)
            m["mkd"] = {t: S.dkern[alt(t)].create_layer({**m["w"], **m["aw"], **k["w"]}) for t in ts}
            if any(alt(t)[0].endswith("_alt") for t in ts):
                _once(("mkd_alt", lid), f"layer {lid}: the ALT library (MPK_ALT_PARITY {ALT_PARITY})")
            _once(("mkd", lid), f"layer {lid}: MoE + aggregation 1 + layer {lid + 1}'s {k['kind'].upper()} decode part in "
                                f"one kernel (sizes {ts})")
    if m is None or k is None or m.get("aw") is None or m.get("mk") is not None or "moeagg" + k["kind"] not in S.kern:
        return
    m["mk"] = S.kern["moeagg" + k["kind"]].create_layer({**m["w"], **m["aw"], **k["w"]})
    _mix_refresh()
    _once(("mk", lid), f"layer {lid}: MoE + aggregation 1 + layer {lid + 1}'s {k['kind'].upper()} part in one kernel")


def _mk_run(layer, m, h, p, forward_batch, addend=None):
    """the MoE + next layer's attention kernel on a target-verify step of one request; None: not here (addend: the stream's
    residual, the prefix p, where SGLang's aggregation adds it to this MLP's output itself)"""
    S.attn_pending.pop(layer.layer_idx + 1, None)
    if m.get("mk") is None or m.get("mk_off"):
        return None
    a = _agg_io(layer, m, h, p, forward_batch)
    if a is None:
        return None
    lid = layer.layer_idx
    k, nxt = S.kda.get(lid + 1) or S.mla[lid + 1], S.layers[lid + 1]
    nvb = a[1]["nvb"] + int(a[1]["write"])   # (the next layer's aggregation 1 writes its bank row first)
    io_k = (_kda_io if k["kind"] == "kda" else _mla_io)(nxt, k, forward_batch, nvb, nxt._mpk_bank.require().block_residual)
    if io_k is None:
        return None
    # (agg_prefix: the next attention's prefix, y or zero after a block write: SGLang's AttnBankState attn_readout)
    mix = m.get("mix_dst") if m.get("mix_dst") is not None else m.get("agg_mix")
    m["mk"].run({**a[0], "agg_prefix": m["agg_prefix"], "agg_bare": m["agg_bare"],
                 **({"agg_mix": mix} if m.get("agg_mix") is not None else {}),
                 **{n: v for n, v in io_k.items() if n not in ("x", "prefix")}})
    if m.get("agg_mix") is not None:   # (the dspark capture after this layer: _dspark_hook)
        S.mix_ready[lid] = mix
    if addend is not None:   # (the aggregation call: prefix_sum = the residual, addend = agg_bare; it returns y)
        a[1].update(res=addend.data_ptr(), bare=m["agg_bare"].data_ptr(), y_t=m["y"])
    S.agg_pending = a[1]
    S.attn_pending[lid + 1] = dict(h=m["agg_out"].data_ptr(), h_t=m["agg_out"], nvb=nvb)
    _once(("mkran", lid), f"layer {lid}: MoE + layer {lid + 1}'s {k['kind'].upper()} part ran in one kernel ({nvb} bank rows)")
    return m["y"]


def _kda_io(layer, d, forward_batch, nvb, br):
    """the KDA kernel's run tensors other than x / prefix (the state pools, rings and windows of this step); None: not here"""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    lid = layer.layer_idx
    lb = getattr(get_attn_backend(), "linear_attn_backend", get_attn_backend())
    cache = lb.req_to_token_pool.mamba2_layer_cache(lid)
    slot, row = lb.forward_metadata.mamba_cache_indices, lb.verify_intermediate_state_indices
    rings = [getattr(cache, n, None) for n in ("replayssm_rawk", "replayssm_rawv", "replayssm_g", "replayssm_beta")]
    if any(r is None for r in rings) or slot.dtype != torch.int32 or row.dtype != torch.int32:
        _once(("kda", lid), f"layer {lid}: SGLang's path (ReplaySSM rings off, or index dtypes {slot.dtype} {row.dtype})")
        return None
    icw = cache.intermediate_conv_window[0]
    info = _info(d, nvb, (rings[0].shape[2], icw.shape[1]))
    if info is None:
        return None
    return dict(bank=br, new_prefix=d["new_prefix"], hidden=d["hidden"], info=info, slot=slot, icw_row=row,
                conv_pool=cache.conv[0], state_pool=cache.temporal, icw=icw, ring_k=rings[0], ring_v=rings[1],
                ring_g=rings[2], ring_beta=rings[3])


def _mla_io(layer, d, forward_batch, nvb, br):
    """the MLA kernel's run tensors other than x / prefix (the KV cache, this step's rows); None: not here"""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    be = get_attn_backend()
    fb = getattr(be, "full_attn_backend", be)
    kv = fb.token_to_kv_pool.get_key_buffer(layer.self_attn.attn_mqa.layer_id)
    rt = fb.req_to_token_pool.req_to_token
    info = _info(d, nvb, (rt.shape[1],))
    if info is None:
        return None
    return dict(bank=br, new_prefix=d["new_prefix"], hidden=d["hidden"], info=info, cache=kv.view(-1, kv.shape[-1]),
                req_to_token=rt, seq_lens=forward_batch.seq_lens, req_idx=forward_batch.req_pool_indices,
                out_loc=forward_batch.out_cache_loc)


def _info(d, nvb, extra):
    """the layer's int32 [8] info for nvb bank rows (made in an eager call: the capture replays it)"""
    key = (nvb, extra)
    t = d["info"].get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        t = torch.tensor([nvb, NB_ALLOC * H, H, *extra, 0, 0, 0, 0][:8], dtype=torch.int32, device=d["hidden"].device)
        d["info"][key] = t
    return t


def attn(layer, h, forward_batch):
    """Ours in place of SGLang's attention part (_run_self_attn, attn_boundary.finish, ffn_boundary.prepare) on a
    target-verify step of one request with 8 tokens; None: SGLang's."""
    lid = getattr(layer, "layer_idx", -1)
    mk = S.attn_pending.pop(lid, None)   # (MPK_K3_MOEAGGKDA: this step's MoE + this KDA kernel of the layer before)
    d = S.kda.get(lid) or S.mla.get(lid)
    if d is not None and forward_batch is not None and h.shape[0] in d["dec"] and forward_batch.forward_mode.is_decode() \
            and forward_batch.batch_size == h.shape[0]:
        n = h.shape[0]
        if mk is not None and mk.get("n") == n:   # (this step's MoE + this decode part of the layer before)
            from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
            nvb = int(layer._mpk_bank.require().num_valid_blocks)
            if mk["h"] != h.data_ptr() and mk["nvb"] == nvb:   # (a copy of aggregation 1's output: checked once, eagerly)
                if not torch.cuda.is_current_stream_capturing() and (lid, n) not in S.copy_ok_h:
                    S.copy_ok_h[(lid, n)] = bool(torch.equal(h, mk["h_t"]))
                    _log(f"layer {lid} ({n} rows): the attention's input is a copy of aggregation 1's output: values "
                         f"{'equal' if S.copy_ok_h[(lid, n)] else 'DIFFER'}")
                if S.copy_ok_h.get((lid, n)):
                    mk = dict(mk, h=h.data_ptr())
            if mk["h"] == h.data_ptr() and mk["nvb"] == nvb:
                residual_batch.stream_of(forward_batch).write(d["dnp"][n])
                return d["dh"][n]
            S.moe[lid - 1]["mk_off"] = True
            _once(("mkdmiss", lid, n), f"layer {lid}: the fused decode kernel's result not used (input {mk['h'] == h.data_ptr()}, "
                                       f"bank rows {mk['nvb']} / {nvb}): this layer's decode kernel, and apart from now on")
        return _attn_decode(layer, d, h, forward_batch)
    if d is None or d["layer"] is None or forward_batch is None or h.shape[0] != T or not forward_batch.forward_mode.is_target_verify() \
            or forward_batch.batch_size != 1:
        return None
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.model_executor.forward_context import get_attn_backend
    bank = layer._mpk_bank.require()
    br, nvb = bank.block_residual, int(bank.num_valid_blocks)
    if br.shape[0] != T or tuple(br.stride()) != (NB_ALLOC * H, H, 1) or not 1 <= nvb <= 16:
        _once(("geo", lid), f"layer {lid}: SGLang's path (bank {tuple(br.shape)} strides {br.stride()} rows {nvb})")
        return None
    stream = residual_batch.stream_of(forward_batch)
    if mk is not None and mk["h"] != h.data_ptr() and mk["nvb"] == nvb and tuple(h.shape) == (T, H):
        # (the layers where SGLang aggregates a copy of y: the attention's input is a copy of the kernel's aggregation 1
        # output too; checked equal in the layer's first eager call, then taken)
        if not torch.cuda.is_current_stream_capturing() and lid not in S.copy_ok_h:
            S.copy_ok_h[lid] = bool(torch.equal(h, mk["h_t"]))
            _log(f"layer {lid}: the attention's input is a copy of aggregation 1's output: values "
                 f"{'equal' if S.copy_ok_h[lid] else 'DIFFER'}")
        if S.copy_ok_h.get(lid):
            mk = dict(mk, h=h.data_ptr())
    if mk is not None and mk["h"] == h.data_ptr() and mk["nvb"] == nvb:   # the MoE + this KDA kernel of the layer before ran it
        stream.write(d["new_prefix"])
        return d["hidden"]
    if mk is not None:   # (from now on the two kernels apart for that layer pair: the eager calls before the capture see it)
        S.moe[lid - 1]["mk_off"] = True
        _once(("mkmiss", lid), f"layer {lid}: the MoE + attention kernel's result not used (input {mk['h'] == h.data_ptr()}, "
                               f"bank rows {mk['nvb']} / {nvb}): its attention kernel again, and apart from now on")
    prefix = stream.residual if stream.residual is not None else S.zeros
    io = {"x": h.contiguous(), "bank": br, "prefix": prefix.contiguous(), "new_prefix": d["new_prefix"],
          "hidden": d["hidden"]}
    be = get_attn_backend()
    if d["kind"] == "kda":
        io_k = _kda_io(layer, d, forward_batch, nvb, br)
        if io_k is None:
            return None
        io.update(io_k)
    else:
        io_m = _mla_io(layer, d, forward_batch, nvb, br)
        if io_m is None:
            return None
        io.update(io_m)
    pend = None
    if d["fused"]:
        if d.get("fused_agg"):
            a = _agg_io(layer, S.moe[lid], d["hidden"], d["new_prefix"], forward_batch, y=d["y"])
            if a is None:   # (a geometry the kernel does not take: SGLang's attention, then our plain MoE)
                return None
            io.update({k: v for k, v in a[0].items() if k not in ("moe_in", "moe_prefix")})
            pend = a[1]
        io["moe_out"] = d["y"]
        S.pending[lid] = d
    d["layer"].run(io)
    if pend is not None:
        S.agg_pending = pend
    stream.write(d["new_prefix"])
    if not torch.cuda.is_current_stream_capturing():
        _once(("ran", lid), f"layer {lid} ({d['kind']}): ours (first eager call, {nvb} bank rows)")
    return d["hidden"]


def _attn_dec_io(layer, d, forward_batch, nvb, br, n):
    """a decode part's run tensors other than x / prefix: the bank, this size's outputs, info, and the pools at the slots
    (KDA) or the KV cache and this step's rows (MLA); None: not here"""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    lid = layer.layer_idx
    io = {"bank": br, "new_prefix": d["dnp"][n], "hidden": d["dh"][n]}
    if d["kind"] == "kda":
        be = get_attn_backend()
        lb = getattr(be, "linear_attn_backend", be)
        cache = lb.req_to_token_pool.mamba2_layer_cache(lid)
        slots = lb.forward_metadata.mamba_cache_indices
        conv, temporal = cache.conv[0], cache.temporal
        if slots.dtype != torch.int32 or slots.shape[0] < n or conv.dtype != torch.bfloat16 or temporal.dtype != torch.float32 \
                or tuple(conv.shape[1:]) != (3, 4608) or tuple(temporal.shape[1:]) != (12, 128, 128) \
                or not conv.is_contiguous() or not temporal.is_contiguous():
            _once(("dkda", lid), f"layer {lid}: SGLang's path in decode (slots {slots.dtype} {tuple(slots.shape)}, conv "
                                 f"{conv.dtype} {tuple(conv.shape)}, temporal {temporal.dtype} {tuple(temporal.shape)})")
            return None
        info = _info(d, nvb, (0, 0))
        if info is None:
            return None
        io.update(info=info, slots=slots, conv_pool=conv, state_pool=temporal)
    else:
        io_m = _mla_io(layer, d, forward_batch, nvb, br)
        if io_m is None:
            return None
        # (not _mla_io's outputs: the verify step's 8-row buffers; this size's are in io)
        io.update({k: v for k, v in io_m.items() if k not in ("bank", "new_prefix", "hidden")})
    return io


def _attn_decode(layer, d, h, forward_batch):
    """a plain-decode step of a batch size with its own library (tasks/short_conv_decode.cuh + delta_rule_decode.cuh, or
    mla_attn_decode.cuh): the pools / KV cache at the per-request slots / rows; None: SGLang's"""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    lid, n = layer.layer_idx, h.shape[0]
    bank = layer._mpk_bank.require()
    br, nvb = bank.block_residual, int(bank.num_valid_blocks)
    if br.shape[0] != n or tuple(br.stride()) != (NB_ALLOC * H, H, 1) or not 1 <= nvb <= 16:
        _once(("dgeo", lid, n), f"layer {lid}: SGLang's path in decode (bank {tuple(br.shape)} strides {br.stride()} rows {nvb})")
        return None
    stream = residual_batch.stream_of(forward_batch)
    prefix = stream.residual if stream.residual is not None else S.dzeros[n]
    io_a = _attn_dec_io(layer, d, forward_batch, nvb, br, n)
    if io_a is None:
        return None
    io = {"x": h.contiguous(), "prefix": prefix.contiguous(), **io_a}
    if not torch.cuda.is_current_stream_capturing() and ("dinfo", d["kind"], n) not in S.logged:
        S.logged.add(("dinfo", d["kind"], n))
        _log(f"decode {d['kind']} inputs ({n} rows): " + "; ".join(
            f"{k} {v.dtype} {tuple(v.shape)} st {tuple(v.stride())}" + (f" {v.flatten()[:4].tolist()}" if v.numel() <= 64 * 1024
                                                                         and v.dtype in (torch.int32, torch.int64) else "")
            for k, v in io.items()))
    dbg = DEC_CHECK and d["kind"] == "mla" and not torch.cuda.is_current_stream_capturing() and ("dchk", lid, n) not in S.logged
    d["dec"][n].run(io)
    if dbg:   # (SGLang's attention runs too; moe() compares)
        S.logged.add(("dchk", lid, n))
        S.pending_chk = dict(lid=lid, hidden=d["dh"][n].clone(), new_prefix=d["dnp"][n].clone())
        return None
    stream.write(d["dnp"][n])
    if not torch.cuda.is_current_stream_capturing():
        _once(("dran", lid, n), f"layer {lid} ({d['kind']}): ours in decode ({n} rows, {nvb} bank rows)")
    return d["dh"][n]


def _agg_io_dec(layer, d, h, p, forward_batch, n):
    """_agg_io for a plain-decode step of n rows: (the fused kernel's MoE + aggregation 1 run tensors, the pending
    aggregation); None: not here"""
    if d.get("aw") is None or forward_batch is None or not forward_batch.forward_mode.is_decode() \
            or forward_batch.batch_size != n:
        return None
    y, g = d["dy"][n], d["dagg"][n]
    lid, nxt = layer.layer_idx, d["next"]
    st = layer._mpk_bank.require()
    br, nvb, write = st.block_residual, int(st.num_valid_blocks), bool(nxt.is_block_write_layer)
    if br.shape[0] != n or tuple(br.stride()) != (NB_ALLOC * H, H, 1) or not 1 <= nvb <= AGG_NBMAX \
            or (write and nvb >= br.shape[1]) or getattr(st, "written_on_slice", False):
        _once(("daggno", lid, n), f"layer {lid}: plain MoE in decode (bank {tuple(br.shape)} rows {nvb} write {write})")
        return None
    key = (nvb, write)
    info = g["info"].get(key)
    if info is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        info = g["info"][key] = torch.tensor([nvb, NB_ALLOC * H, H, int(write), 0, 0, 0, 0], dtype=torch.int32, device=h.device)
    return ({"moe_in": h.contiguous(), "moe_prefix": p.contiguous(), "moe_out": y, "agg_bank": br, "agg_info": info,
             "agg_out": g["out"]},
            dict(y=y.data_ptr(), y_t=y, bank=br.data_ptr(), nvb=nvb, write=write, proj=nxt.self_attention_res_proj,
                 out=g["out"], lid=lid, n=n))


def _mk_run_dec(layer, m, h, p, forward_batch, n, addend=None):
    """the MoE + the next layer's aggregation 1 + its decode part in one kernel (a plain-decode step of n rows); None: not
    here (addend: as _mk_run)"""
    lid = layer.layer_idx
    S.attn_pending.pop(lid + 1, None)
    if m.get("mk_off"):
        return None
    a = _agg_io_dec(layer, m, h, p, forward_batch, n)
    if a is None:
        return None
    k, nxt = S.kda.get(lid + 1) or S.mla.get(lid + 1), S.layers[lid + 1]
    nvb = a[1]["nvb"] + int(a[1]["write"])   # (the next layer's aggregation 1 writes its bank row first)
    br = nxt._mpk_bank.require().block_residual
    if br.shape[0] != n or tuple(br.stride()) != (NB_ALLOC * H, H, 1):
        return None
    io_k = _attn_dec_io(nxt, k, forward_batch, nvb, br, n)
    if io_k is None:
        return None
    g = m["dagg"][n]
    m["mkd"][n].run({**a[0], "agg_prefix": g["prefix"], "agg_bare": g["bare"], **io_k})
    if addend is not None:   # (the aggregation call: prefix_sum = the residual, addend = agg_bare; it returns y)
        a[1].update(res=addend.data_ptr(), bare=g["bare"].data_ptr(), y_t=m["dy"][n])
    S.agg_pending = a[1]
    S.attn_pending[lid + 1] = dict(h=g["out"].data_ptr(), h_t=g["out"], nvb=nvb, n=n)
    if not torch.cuda.is_current_stream_capturing():
        _once(("mkdran", lid, n), f"layer {lid}: MoE + layer {lid + 1}'s {k['kind'].upper()} decode part ran in one kernel "
                                  f"({n} rows, {nvb} bank rows)")
    return m["dy"][n]


# ---------------------------------------------------------------- setup
def _count_ids(ids):
    n = int(ids.shape[0])
    c = S.ecount.setdefault(n, [])
    c.append(int(torch.unique(ids).numel()))
    if len(c) % 930 == 0:
        w = sorted(c[-930:])
        _log(f"[experts] {n} rows: distinct experts per MoE call over the last 930 calls: median {w[465]}, mean "
             f"{sum(w) / 930:.1f}, min {w[0]}, max {w[-1]} ({len(c)} calls)")


def _count_prepare(model):
    n = 0
    for m in model.modules():
        if type(m).__name__ != "KimiK3MoE" or getattr(m.topk, "_mpk_count", False):
            continue
        orig = m.topk.forward

        def fwd(*a, _orig=orig, **k):
            r = _orig(*a, **k)
            _count_ids(r.topk_ids)
            return r
        m.topk.forward = fwd
        m.topk._mpk_count = True
        mod = sys.modules[type(m).__module__]
        if not getattr(mod, "_mpk_count", False):
            ob = mod.build_precomputed_topk_output

            def bpt(*a, _orig=ob, **k):
                r = _orig(*a, **k)
                _count_ids(r.topk_ids)
                return r
            mod.build_precomputed_topk_output = bpt
            mod._mpk_count = True
        n += 1
    _log(f"[experts] counting the distinct experts of {n} MoE layers")


def prepare(model):
    if COUNT_EXPERTS and S.rank < 0:
        from sglang.srt.distributed import get_tp_group
        S.rank = get_tp_group().rank_in_group
        _count_prepare(model)
    if (not any(MODE.values()) and not (DEC_T and any(DEC_LIB.values()))) or S.zeros is not None:
        return
    try:
        from sglang.srt.distributed import get_tp_group
        tp = get_tp_group()
        S.rank = tp.rank_in_group
        dev = torch.device("cuda", torch.cuda.current_device())
        S.zeros = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
        if (MODE["kdamoe"] or MODE["mlamoe"]) and MODE["moe"] != "run":
            raise RuntimeError("MPK_K3_KDAMOE / _MLAMOE need MPK_K3_MOE=run (the MoE of the other calls)")
        for kind in KINDS:   # (each a collective: the same order on every rank)
            if MODE[kind] == "run":
                S.kern[kind] = RankKernel(LIB[kind], tp.device_group, dev)
                _log(f"{kind}: library {LIB[kind]}, exchange region {S.kern[kind].exchange.ptrs[S.rank]:#x}")
        for part in ("kda", "mla", "moe") + DEC_FUSED + ("magg",):   # (plain decode: per batch size its own library; collectives in order)
            for t in DEC_T if DEC_LIB[part] else ():
                path = DEC_LIB[part].format(T=t)
                if not os.path.exists(path):   # (a size without this part's library: the next part / SGLang's at that size)
                    _log(f"decode {part}: no library for batch size {t} ({path})")
                    continue
                k = RankKernel(path, tp.device_group, dev)
                if k.tokens != t:
                    raise RuntimeError(f"{path}: built for {k.tokens} tokens, listed for batch size {t}")
                S.dkern[(part, t)] = k
                S.dzeros[t] = torch.zeros(t, H, dtype=torch.bfloat16, device=dev)
            if DEC_LIB[part] and DEC_T:
                _log(f"decode {part}: libraries for batch sizes {[t for t in DEC_T if (part, t) in S.dkern]} ({DEC_LIB[part]})")
        for part in ("fkda", "fmla"):   # (MPK_AMD_DEC_*_ALT: the second library set of a paired A / B; after the main ones, same order)
            for t in DEC_T if DEC_ALT[part] else ():
                path = DEC_ALT[part].format(T=t)
                if not os.path.exists(path):
                    continue
                k = RankKernel(path, tp.device_group, dev)
                if k.tokens != t:
                    raise RuntimeError(f"{path}: built for {k.tokens} tokens, listed for batch size {t}")
                S.dkern[(part + "_alt", t)] = k
            if DEC_ALT[part]:
                _log(f"decode {part} ALT: libraries for batch sizes {[t for t in DEC_T if (part + '_alt', t) in S.dkern]} ({DEC_ALT[part]})")
        if "moe" in S.kern or any(p == "moe" for p, _ in S.dkern):
            _moe_prepare(model)
        if any(k in S.kern for k in ("kda", "mla", "kdamoe", "mlamoe")) or any(p in ("kda", "mla") for p, _ in S.dkern):
            _attn_prepare(model)
        torch.cuda.synchronize()
        tp.barrier()   # (every rank's layers and exchange regions set up before any rank's first launch)
    except Exception as e:  # noqa
        import traceback
        _log(f"prepare FAILED: {e!r}\n{traceback.format_exc()}")
        raise
