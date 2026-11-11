"""Our GLM-5.3 layer kernels (one kernel per layer, per-rank libraries: amd/serving/glm53/build_glm53_libs.sh) inside SGLang ROCm
v0.5.22's deepseek_v2.py (GlmMoeDsaForCausalLM), plain decode, one TP rank per process (TP 8, no EP). The contract:
amd/notes/sglang_v0522_glm53_layer_contract.md; the graph and the weight loaders: glm53_layers.py.

  prepare(causal_lm)            after load_weights: this rank's weights from the checkpoint (glm53_layers' loaders), the
                                libraries (each a collective: the exchange regions over HIP IPC), per layer and batch size an
                                instance, the run buffers
  forward(model, ids, pos, fb)  DeepseekV2Model.forward on a decode step we take (active): embed (SGLang's, with its all-reduce),
                                x = input_layernorm[0](h), residual = h; per layer one kernel (attention + MLP; the last layer's
                                RMSNorm is model.norm); returns the final-normed hidden [T, H]
active(fb): MPK_G53=run, a decode step, T in MPK_G53_TOKENS, and the batch's longest sequence <= 2048 (SGLang's "dense" graph
variant: the row list is every position; the scored top-2048 is not ours yet) - in a graph capture: the captured variant.
Our kernels read / write SGLang's caches in place: per layer the fp8 MLA cache rows at out_cache_loc, the full layers' index-K
pages; the shared layers read the last full layer's row list (a buffer of ours, written by that layer's kernel).
Environment:
  MPK_G53          "run": ours (unset: SGLang's path everywhere)
  MPK_G53_TOKENS   the batch sizes with libraries (comma-separated; SGLang's captured decode sizes)
  MPK_G53_DLIB     the libraries' directory (<kind>_t<T>/lib.so, kinds g53full g53shared g53dense)
  MPK_G53_MODEL    the checkpoint directory (default /dev/shm/glm53)
  MPK_G53_DEBUG_SYNC=1 (with --disable-cuda-graph): after every kernel a device sync and a line (layer, finite outputs)
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for d in (HERE, os.path.join(ROOT, "amd", "serving")):
    if d not in sys.path:
        sys.path.insert(0, d)
import glm53_layers as gl  # noqa: E402

ON = os.environ.get("MPK_G53", "") == "run"
TOKENS = sorted({int(x) for x in os.environ.get("MPK_G53_TOKENS", "").split(",") if x})
DLIB = os.environ.get("MPK_G53_DLIB", "/mpk-amd/glm53/dlib")
MODEL = os.environ.get("MPK_G53_MODEL", "/dev/shm/glm53")
DEBUG_SYNC = os.environ.get("MPK_G53_DEBUG_SYNC", "") == "1"
_S = {}


def _log(msg):
    print(f"[mpk-g53 rank {_S.get('rank', '?')}] {msg}", flush=True)


def prepare(causal_lm):
    """the libraries, this rank's layer instances, the run buffers (once, after the weights are loaded)"""
    arch = (getattr(causal_lm.config, "architectures", None) or [""])[0]
    if not ON or not TOKENS or arch != "GlmMoeDsaForCausalLM":
        return
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group
    from rank_lib import RankKernel
    t0 = time.time()
    group = get_tp_group().device_group
    rank, n = dist.get_rank(group), dist.get_world_size(group)
    dev = torch.device("cuda", torch.cuda.current_device())
    _S.update(rank=rank, n=n, dev=dev)
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    L, nd = cfg["num_hidden_layers"], cfg["first_k_dense_replace"]
    kinds = ["g53dense" if i < nd else ("g53full" if cfg["indexer_types"][i] == "full" else "g53shared") for i in range(L)]
    assert all(cfg["indexer_types"][i] == "full" for i in range(nd)), "the dense layers have their own indexer (g53dense)"
    assert cfg["indexer_types"][0] == "full"
    libs = {(k, T): RankKernel(os.path.join(DLIB, f"{k}_t{T}", "lib.so"), group, dev) for k in sorted(set(kinds)) for T in TOKENS}
    _log(f"libraries {sorted(libs)} ({time.time() - t0:.0f} s)")
    ck = gl.Ckpt(MODEL)
    layers, keep = {}, []
    for i in range(L):
        w = gl.attn_weights(ck, cfg, i, rank, n, dev)
        w.update(gl.dense_weights(ck, cfg, i, rank, n, dev) if i < nd else gl.moe_weights(ck, cfg, i, rank, n, dev))
        keep.append(w)
        for T in TOKENS:
            layers[(i, T)] = libs[(kinds[i], T)].create_layer(w)
        if i % 8 == 0:
            _log(f"layer {i} weights ({time.time() - t0:.0f} s, {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB)")
    bufs = {}
    for T in TOKENS:
        z = lambda *sh, dt=torch.bfloat16: torch.zeros(*sh, dtype=dt, device=dev)
        bufs[T] = {"x": [z(T, gl.H), z(T, gl.H)], "res": [z(T, gl.H), z(T, gl.H)], "rows": z(T, gl.K_TOP, dt=torch.int32), "res_mid": z(T, gl.H),
                   "req_idx": torch.arange(T, dtype=torch.int64, device=dev)}
    _S.update(L=L, kinds=kinds, libs=libs, layers=layers, keep=keep, bufs=bufs, eps=cfg["rms_norm_eps"])
    _log(f"ready: {L} layers x {len(TOKENS)} sizes ({time.time() - t0:.0f} s, {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB)")


def active(forward_batch) -> bool:
    if not ON or "layers" not in _S:
        return False
    fb = forward_batch
    if not fb.forward_mode.is_decode() or fb.input_ids.shape[0] not in _S["bufs"]:
        return False
    from sglang.srt.model_executor.runner import get_is_capture_mode
    if get_is_capture_mode():   # (the graph replays this branch: the captured variant decides, as dsa_indexer.py does)
        from sglang.srt.layers.attention.graph_variants import DSA_DENSE
        from sglang.srt.model_executor.runner_utils.capture_mode import get_capture_attention_variant
        return get_capture_attention_variant() == DSA_DENSE
    if fb.seq_lens_cpu is not None and fb.seq_lens_cpu.numel() > 0:
        return int(fb.seq_lens_cpu.max().item()) <= gl.K_TOP
    return int(fb.seq_lens.max().item()) <= gl.K_TOP


def _rope_tables(model, dev):
    """the MLA / indexer rope's bf16 cos / sin [positions][32]: aiter RotaryEmbedding's cos_cache / sin_cache when the module has
    them, else the same formula (inv_freq = 1 / theta^(2i / 64), fp32, then bf16)"""
    if "rope" not in _S:
        rope = model.layers[0].self_attn.rotary_emb
        if hasattr(rope, "cos_cache") and hasattr(rope, "sin_cache"):
            cos, sin = rope.cos_cache.view(rope.cos_cache.shape[0], -1), rope.sin_cache.view(rope.sin_cache.shape[0], -1)
            src = "aiter cos_cache / sin_cache"
        else:
            theta = float(json.load(open(os.path.join(MODEL, "config.json")))["rope_parameters"]["rope_theta"])
            inv = 1.0 / (theta ** (torch.arange(0, gl.RP, 2, dtype=torch.float32, device=dev) / gl.RP))
            f = torch.einsum("i,j->ij", torch.arange(1 << 20, dtype=torch.float32, device=dev), inv)
            cos, sin = f.cos().to(torch.bfloat16), f.sin().to(torch.bfloat16)
            src = f"computed ({type(rope).__name__} has no cos_cache)"
        _S["rope"] = (cos.to(torch.bfloat16).contiguous(), sin.to(torch.bfloat16).contiguous())   # (the kernels read bf16)
        _log(f"rope tables {tuple(cos.shape)} {cos.dtype}: {src}")
    return _S["rope"]


def _check_once(T, i, io):
    key = ("chk", T, i)
    if key in _S:
        return
    _S[key] = True
    for k, v in io.items():
        if not v.is_contiguous():
            raise RuntimeError(f"[mpk-g53] T {T} layer {i}: {k} is not contiguous ({tuple(v.shape)} strides {tuple(v.stride())})")
    _log(f"T {T} layer {i} run tensors: " + "; ".join(f"{k} {str(v.dtype).replace('torch.', '')} {tuple(v.shape)}" for k, v in io.items()))


def forward(model, input_ids, positions, forward_batch, input_embeds=None):
    """one decode step with our kernels: DeepseekV2Model.forward's result (the final-normed hidden [T, H])"""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    h = input_embeds if input_embeds is not None else model.embed_tokens(input_ids)
    T = h.shape[0]
    b, layers = _S["bufs"][T], _S["layers"]
    x0 = model.layers[0].input_layernorm(h)   # (layer 0: rmsnorm of the embedding, the residual = the embedding)
    if isinstance(x0, tuple):
        x0 = x0[0]
    x0, res0 = x0.contiguous(), h.contiguous()
    be = get_attn_backend()
    be = getattr(be, "primary", be)
    seq32 = be.forward_metadata.cache_seqlens_int32[:T]
    pool = be.token_to_kv_pool   # (the DSA backend's: dsa_backend.py:552-555)
    # (the requests' req_to_token rows, the first 2048 positions: the libraries' table is [T][2048], row t = request t)
    r2t = be.req_to_token[forward_batch.req_pool_indices[:T], :gl.K_TOP].contiguous()
    cos, sin = _rope_tables(model, h.device)
    common = {"out_loc": forward_batch.out_cache_loc[:T], "positions": positions[:T], "seq_lens": seq32, "cos": cos, "sin": sin}
    cur_x, cur_r = x0, res0
    for i in range(_S["L"]):
        nx, nr = b["x"][i % 2], b["res"][i % 2]   # (the outputs alternate between the two persistent buffers)
        io = {"x_attn": cur_x, "res_in": cur_r, "res_mid": b["res_mid"], "res_out": nr, "x_next": nx,
              "cache": pool.get_key_buffer(i).view(torch.uint8).view(-1, gl.KVL + gl.RP), **common}
        if _S["kinds"][i] == "g53shared":
            io["rows"] = b["rows"]
        else:
            io.update(icache=pool.get_index_k_with_scale_buffer(i).view(torch.uint8).view(-1, 64 * (gl.ID + 4)), r2t=r2t,
                      req_idx=b["req_idx"], rows_g=b["rows"])
        if i < 3:
            _check_once(T, i, io)
        layers[(i, T)].run(io)
        if DEBUG_SYNC:
            torch.cuda.synchronize()
            _log(f"T {T} layer {i} ({_S['kinds'][i]}) done: x_next finite {bool(torch.isfinite(nx.float()).all())}")
        cur_x, cur_r = nx, nr
    return cur_x
