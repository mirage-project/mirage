"""The static-megakernel GLM-5.3-Flash DSA layer that dsa_static_hook.py puts into SGLang (no SGLang import here): one GPU's
attention with the sublayer boundary inside (as Kimi K3's fused layers, #811), built for the serving launch
(-DSTATIC_RESET_IN_KERNEL, launch_stream on the caller's stream).
  layer        dsa_layer.build_graph: the lightning indexer with its pooled keys selects the positions (dsa_kpool_update ->
               dsa_index_score -> dsa_index_topk), MLA attends to them, then o_proj's K parts -> allreduce_send ->
               hc_post_mix (the TP sum over the ranks in the kernel; the attention's mHC post with SGLang's h_post / h_res,
               and the FFN's mHC mixes) -> hc_pre (the rest of the FFN's mHC pre with post_attention_layernorm): run_fused
               returns x_ffn, the streams R' and the FFN's h_post / h_res (what SGLang's attn_boundary.finish +
               ffn_boundary.prepare hand the FFN)
  latent       SGLang's latent pool of the layer itself (kv_buffer [slots + page, 1, 512] bf16, by token slot): dsa_kv_write
               writes this step's row in it, sparse_mla reads it, no copy
  pooled keys  SGLang's index-K buffer of the layer ([pages, 64 x 132]: e4m3 keys, fp32 scales) and its tail buffers
               ([request slots, RING, 128] bf16, k and gate), no copy
  per step     seq_lens (int32), the block table (page 1: row t = req_to_token of the request, position -> token slot), the
               pooled page table (entry j = the page of token 256 j: SGLang's DSA KV pages hold 256 tokens = 64 pools),
               the request slots (int64): bound to SGLang's own tensors of the step (its DSA metadata cache_seqlens_int32,
               page_table_1, pooled_real_page_table; req_pool_indices) when their dtype and row stride are the layer's (the
               CUDA graph buffers: no copy), else copied into the layer's tensors (an eager step's freshly made metadata)
  padding      a CUDA graph of T rows replays with fewer requests: SGLang pads req_pool_indices with 0 and seq_lens with 1;
               request slot 0 is SGLang's dummy (its req_to_token row: the dummy token slot 0), so the padded rows read and
               write only dummy rows
  launch       launch_stream on the current stream (a CUDA graph captures it); pdl: a programmatic launch, the kernel lets
               the next one launch at its start (-DSTATIC_PDL_TRIGGER=1)
  ranks        built for the tp ranks, each process its own rank alone (compile_static rank / exchange), the exchange region
               in torch symmetric memory over the TP group (multicast: SGLang's custom all-reduce allocates its push buffers
               so); every rank's layers must be built before any rank's first launch (the hook's barrier)
  plan         a plan file of the fused graph of this T (dsa_fused_demo --search, one GPU's lists) or "rr": the round-robin
               plan of build_graph's grids; the verify layer: round robin
"""
import os
import sys

import torch

ROOT = os.environ.get("DSA_STATIC_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))   # default: this checkout
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "demo", "glm53_flash"))

TOKENS = int(os.environ.get("STATIC_TOKENS", "8"))   # the default of StaticDSA(tokens=): a process may build several T
# sparse_mla's splits per token by T (of dsa_layer.MLA_TILES): T = 8: 18 (two tiles each, 144 tasks); T = 16: 9 (four tiles
# each, 144 tasks: one wave of the 148 SMs); T = 32: 4 (nine tiles each, 128 tasks)
SPLITS = {8: 18, 16: 9, 32: 4}
KV_PAGE = 256                   # SGLang's DSA KV page (tokens): one page of the index-K buffer (64 pools)
PHYS_PAGE = 64                  # SGLang's DSA real_page_table pages (tokens): pooled_real_page_table takes every 4th


def _write_plan(compiler, plan: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "plan.json")
    compiler.write_plan_file(path, plan)
    return path


def _backed(v: torch.Tensor, rows: int) -> bool:
    """v (contiguous) is the first rows of a buffer of at least `rows` rows of its row stride (sglang_kda's backed): a view
    of a larger buffer, which a kernel bound to it may read past v's rows."""
    if not v.is_contiguous():
        return False
    rowb = (v.stride(0) if v.dim() > 1 else 1) * v.element_size()
    return v.untyped_storage().nbytes() - v.storage_offset() * v.element_size() >= rows * rowb


_VERIFY_META = {}   # T -> {"key": a forward's key, "owner": the StaticDSA whose own per-token metadata that forward's other
                    # verify layers of this T read, "n": its rows per request} (run_verify step_key)


def _exchange_region(nbytes: int, dev, group):
    """nbytes of torch symmetric memory over the process group, zero-filled (as SGLang's custom_all_reduce_v2 allocates its
    push buffers): (the tensor, its rendezvous handle, the multicast address)."""
    from torch._C._distributed_c10d import _SymmetricMemory
    buf = _SymmetricMemory.empty_strided_p2p((nbytes,), [1], torch.uint8, dev, group.group_name)
    handle = _SymmetricMemory.rendezvous(buf)
    mc = int(handle.multicast_ptr)
    if mc == 0:
        raise RuntimeError("the TP group's symmetric memory has no multicast address (NVLS)")
    buf.zero_()
    return buf, handle, mc


class StaticDSA:
    """One chosen DSA layer of this rank: the built static layer and its graph tensors."""

    def __init__(self, layer: int, model_path: str, rank: int, tp: int, kv_buffer: torch.Tensor, req_to_token: torch.Tensor,
                 out_root: str, indexer: dict, fused: dict, pdl: bool = False, eps: float = 1e-5, verify: int = 0,
                 fused_plan: str = "rr", tokens: int = None, requests: int = 1, weights: dict = None):
        """indexer: SGLang's {"pcache": index-K buffer, "tail_k", "tail_score": tail buffers} of the layer. fused (the module
        docstring): {"group": the TP process group, "hc_fn" [24, 4 H], "hc_base" [24], "hc_scale" [3] fp32 (the layer's
        hc_ffn_*), "norm" [H] bf16 (post_attention_layernorm), "norm_eps", "rms_eps", "hc_eps", "iters"}. fused_plan: a plan
        file of the fused graph of this T or "rr". verify: the layer of a speculative verify step: `verify` draft tokens of a
        request as rows t at positions seq_lens - 1 + t (the other rows idle), the pooled indexer keys and tail updated token
        by token (dsa_kpool_update tokens_per_request): run_verify; requests: R requests of T / R rows each (request r's
        rows from r T / R). tokens: the rows T of this layer (default TOKENS; a process may hold several). weights: another
        StaticDSA's checkpoint tensors of this layer (dsa_mla.load_layer's), shared."""
        import dsa_layer
        import dsa_mla as demo
        from mirage.mpk import compiler
        from mirage.mpk.static_megakernel import StaticMegakernel

        T = self.T = tokens if tokens is not None else TOKENS
        self.requests = int(requests)
        assert T % self.requests == 0 and (not verify or 0 < T // self.requests <= 8), \
            f"StaticDSA: {self.requests} requests of {T // self.requests} rows in {T}"
        dev = kv_buffer.device
        assert kv_buffer.dtype == torch.bfloat16 and kv_buffer.shape[-1] == demo.KVL and kv_buffer.is_contiguous(), \
            f"layer {layer}: latent pool {tuple(kv_buffer.shape)} {kv_buffer.dtype}: expected [rows, 1, 512] bf16 " \
            f"(run SGLang with kv_cache_dtype bfloat16: on SM100 its DSA default is FP8)"
        w = weights if weights is not None else demo.load_layer(model_path, layer, tp, rank, dev)
        width = req_to_token.shape[1]
        t = demo.gpu_tensors(w, {
            "x": torch.zeros(T, demo.H, dtype=torch.bfloat16, device=dev),
            "seq_lens": torch.zeros(T, dtype=torch.int32, device=dev),
            "block_table": torch.zeros(T, width, dtype=torch.int32, device=dev),
            "cache": kv_buffer.view(-1, demo.KVL),
            "indices": torch.full((T, demo.POOLS), -1, dtype=torch.int32, device=dev)})
        t["o"] = torch.zeros(T, demo.H, dtype=torch.bfloat16, device=dev)
        self.t = t
        pc, tk, ts = indexer["pcache"], indexer["tail_k"], indexer["tail_score"]
        assert pc.dtype == torch.uint8 and pc.shape[1] == dsa_layer.PPAGE * 132 and pc.is_contiguous(), \
            f"layer {layer}: index-K buffer {tuple(pc.shape)} {pc.dtype}: expected [pages, 64 x 132] uint8"
        assert tk.dtype == ts.dtype == torch.bfloat16 and tk.shape == ts.shape and tk.shape[2] == dsa_layer.ID
        assert tk.is_contiguous() and ts.is_contiguous(), "SGLang's tail buffers, bound as they are"
        # the pooled page table's row stride: SGLang's pooled_real_page_table (ceil(width / 64) real pages, every 4th);
        # the score node scores whole 1024-pool chunks below it
        self.maxpp = ((width + PHYS_PAGE - 1) // PHYS_PAGE + 3) // 4
        maxpools = width // 4096 * 4096 // 4
        wi = dsa_layer.load_indexer(model_path, layer, dev)
        self.t["pool_idx"] = self.t.pop("indices")
        self.t.update({k: wi[k] for k in ("w_kgw", "w_iqb", "k_norm_w", "k_norm_b", "ape")})
        self.t.update({"req_slots": torch.zeros(T, dtype=torch.int64, device=dev),
                       "pooled_bt": torch.zeros(T, self.maxpp, dtype=torch.int32, device=dev),
                       "pcache": pc, "tail_k": tk, "tail_score": ts,
                       "idx_scores": torch.zeros(T, maxpools, dtype=torch.float32, device=dev),
                       "q_c": torch.zeros(T, demo.QL, dtype=torch.bfloat16, device=dev)})
        hc_small = torch.zeros(32, dtype=torch.float32)
        hc_small[:24], hc_small[24:27] = fused["hc_base"].float().cpu(), fused["hc_scale"].float().cpu()
        HCH = fused["hc_fn"].shape[1]
        self.t.update({"o": torch.zeros(T, demo.H, dtype=torch.int64, device=dev),
                       "R_in": torch.zeros(T, HCH, dtype=torch.bfloat16, device=dev),
                       "h_post_in": torch.zeros(T, 4, dtype=torch.float32, device=dev),
                       "h_res_in": torch.zeros(T, 16, dtype=torch.float32, device=dev),
                       "R_out": torch.zeros(T, HCH, dtype=torch.bfloat16, device=dev),
                       "hc_ffn_fn": fused["hc_fn"].float().contiguous().to(dev), "hc_ffn_small": hc_small.to(dev),
                       "ffn_norm": fused["norm"].to(torch.bfloat16).contiguous().to(dev),
                       "x_ffn": torch.zeros(T, demo.H, dtype=torch.bfloat16, device=dev),
                       "h_post_out": torch.zeros(T, 4, dtype=torch.float32, device=dev),
                       "h_res_out": torch.zeros(T, 16, dtype=torch.float32, device=dev)})
        mpk = StaticMegakernel(num_gpus=tp)
        grids = dsa_layer.build_graph(mpk, self.t, 1, SPLITS.get(T, 18), eps, 1e-6, {
            "rms_eps": fused["rms_eps"], "hc_eps": fused["hc_eps"], "norm_eps": fused["norm_eps"], "iters": fused["iters"],
            "verify": bool(verify), "requests": self.requests})
        if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the layer's GEMM weights into L2 at its start (core.cuh kernel_begin)
            mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))
        # the searched plans are of the decode graphs (kpool_update per request): a verify layer runs round robin
        self.plan = "rr" if verify or not fused_plan else fused_plan
        plan = demo.round_robin_plan(mpk, grids) if self.plan == "rr" else compiler.plan_from_file(self.plan)
        out_dir = os.path.join(out_root, f"rank{rank}", f"layer{layer}{f'_t{T}' if T != 8 else ''}{'_verify' if verify else ''}")
        flags = ["-DSTATIC_RESET_IN_KERNEL", f"-DSTATIC_PDL_TRIGGER={int(pdl)}"]
        self.verify = int(verify)
        self.group = fused["group"]
        self.fused_args = fused
        # every rank: the same lists; this rank alone, the exchange region in symmetric memory
        plan["lists"] = plan["lists"] * tp
        plan = compiler.plan_from_file(_write_plan(compiler, plan, out_dir))
        paths, self.info = compiler.compile_plan(mpk, plan, os.path.join(out_dir, "schedules"))
        self.xbuf, self.xhandle, mc = _exchange_region(mpk.exchange_region_bytes(), dev, fused["group"])
        self.sk = mpk.compile_static(paths, gpu_tensors=[self.t], out_dir=os.path.join(out_dir, "build"),
                                     extra_flags=flags, device=dev.index, rank=rank, exchange=(mc, self.xbuf.data_ptr()))
        self.layer, self.pdl, self.req_to_token, self.eps = layer, pdl, req_to_token, eps
        self.w = w   # the checkpoint's tensors of this rank (dsa_mla.load_layer)
        self._bound = {}    # graph tensor name -> the address bound now
        self.binds = None   # the last run's: per per-step input, bound (True) or copied (False)
        self.gather = {}    # run_verify with R > 1 requests: the outputs' rows gathered, per output

    def _bind(self, name: str, tensor: torch.Tensor):
        if self._bound.get(name) != tensor.data_ptr():
            self.sk.rebind(name, tensor.data_ptr())
            self._bound[name] = tensor.data_ptr()

    def _take(self, name: str, src, fill) -> bool:
        """Graph tensor `name` reads src itself if src has its dtype and row stride (and at least T rows), else the layer's own
        tensor, filled by fill(own). Returns True when bound."""
        own = self.t[name]
        ok = src is not None and src.dtype == own.dtype and src.dim() == own.dim() and src.shape[0] >= self.T and \
            src.stride(-1) == 1 and (own.dim() == 1 or src.stride(0) == own.shape[1])
        self._bind(name, src if ok else own)
        if not ok:
            fill(own)
        return ok

    def _launch(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor, req_pool_indices: torch.Tensor, meta=None):
        """x and the per-step inputs bound (or copied), then one launch on the current stream. hidden_states [T, H] bf16
        (bound when contiguous, else copied), seq_lens [T] (the token of this step is at seq_lens - 1), req_pool_indices [T]
        int64 (rows of req_to_token and of the tails), meta: SGLang's DSA metadata of the step (DSAMetadata:
        cache_seqlens_int32, page_table_1, pooled_real_page_table) or None. Only tensor ops, host-side binds and the launch:
        a CUDA graph captures it."""
        x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        self._bind("x", x)
        T = self.T
        rpi = req_pool_indices[:T]
        get = lambda k: getattr(meta, k, None)
        self.binds = {
            "seq_lens": self._take("seq_lens", get("cache_seqlens_int32"), lambda own: own.copy_(seq_lens[:T])),
            "block_table": self._take("block_table", get("page_table_1"),
                                      lambda own: own.copy_(self.req_to_token[rpi.long()])),
            "req_slots": self._take("req_slots", req_pool_indices, lambda own: own.copy_(rpi)),
            "pooled_bt": self._take(
                "pooled_bt", get("pooled_real_page_table"),
                lambda own: own.copy_(self.req_to_token[rpi.long()][:, ::KV_PAGE][:, :self.maxpp] // KV_PAGE))}
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)

    def run_verify(self, hidden_states, residual, h_post, h_res, seq_lens, req_pool_indices, step_key=None):
        """The verify layer (StaticDSA verify): R = self.requests requests' n <= verify draft tokens each as rows (request
        r's at rows r n .. r n + n - 1 of the inputs, as run_fused's; rows r NT .. of the layer, NT = T / R, the rest
        idle), the requests' context lengths seq_lens (int32 [>= R]: the tokens are at positions seq_len + t) and request
        slots req_pool_indices (int64 [>= R]); returns run_fused's outputs' R n rows (contiguous). With R = 1 an input is
        bound (no copy) when it is the first rows of a buffer of at least T rows (_backed: a CUDA graph's static buffer, a
        static layer's output; the kernel reads its idle rows too), else copied into the layer's own (self.binds). The
        per-token metadata (seq_lens, the page tables, the slot; the idle rows' seq_lens 0) is the same for every verify
        layer of a forward: with step_key (the forward's key, the hook's), the first layer run in a forward fills its own
        tensors and the others bind theirs to them (_VERIFY_META; a CUDA graph captures the first layer's fills once)."""
        assert self.verify
        R, NT = self.requests, self.T // self.requests
        rows = hidden_states.shape[0]
        assert rows % R == 0 and 0 < rows // R <= self.verify, \
            f"run_verify: {rows} rows of {R} requests, the layer's draft tokens {self.verify}"
        n = rows // R
        own = self.t
        binds = {}
        for name, src in (("x", hidden_states), ("R_in", residual), ("h_post_in", h_post), ("h_res_in", h_res)):
            src = src.reshape(rows, -1)
            ok = R == 1 and src.dtype == own[name].dtype and src.shape[1] == own[name].shape[1] and _backed(src, self.T)
            if ok:
                self._bind(name, src)
            else:
                self._bind(name, own[name])
                if R == 1:
                    own[name][:n].copy_(src)
                else:
                    own[name].view(R, NT, -1)[:, :n].copy_(src.reshape(R, n, -1))
            binds[name] = ok
        META = ("seq_lens", "block_table", "req_slots", "pooled_bt")
        shared = _VERIFY_META.get(self.T)
        if step_key is not None and shared is not None and shared["key"] == step_key and shared["n"] == n and \
                shared["owner"] is not self:
            for name in META:   # this forward's metadata, in another layer's tensors
                self._bind(name, shared["owner"].t[name])
            binds["meta"] = True
        else:
            for name in META:
                self._bind(name, own[name])
            own["seq_lens"].zero_()
            ar = torch.arange(1, n + 1, dtype=torch.int32, device=own["seq_lens"].device)
            own["seq_lens"].view(R, NT)[:, :n].copy_(seq_lens[:R].to(torch.int32)[:, None] + ar[None, :])
            rpi = req_pool_indices[:R].long()
            row = self.req_to_token[rpi]                       # [R, width]
            own["block_table"].view(R, NT, -1)[:, :n].copy_(row[:, None, :].expand(R, n, -1))
            own["req_slots"].view(R, NT)[:, :n].copy_(rpi[:, None].expand(R, n))
            own["pooled_bt"].view(R, NT, -1)[:, :n].copy_(
                (row[:, ::KV_PAGE][:, :self.maxpp] // KV_PAGE)[:, None, :].expand(R, n, -1))
            if step_key is not None:
                _VERIFY_META[self.T] = {"key": step_key, "owner": self, "n": n}
            binds["meta"] = False
        self.binds = binds
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        if R == 1:
            return own["x_ffn"][:n], own["R_out"][:n], own["h_post_out"][:n], own["h_res_out"][:n]
        outs = []
        for name in ("x_ffn", "R_out", "h_post_out", "h_res_out"):   # the requests' rows gathered (contiguous)
            g = self.gather.get(name)
            if g is None:
                g = self.gather[name] = torch.empty_like(own[name])
            g[:rows].view(R, n, -1).copy_(own[name].view(R, NT, -1)[:, :n])
            outs.append(g[:rows])
        return tuple(outs)

    def run_fused(self, hidden_states, residual, h_post, h_res, seq_lens, req_pool_indices, meta=None):
        """The decode layer (the module docstring): one launch on the current stream from the attention's input
        hidden_states [T, H] bf16, the streams residual [T, 4 H] bf16 and the attention's h_post [T, 4] / h_res [T, 16] fp32
        (bound when contiguous, else copied) to (x_ffn [T, H] bf16, R' [T, 4 H] bf16, the FFN's h_post [T, 4], h_res
        [T, 16] fp32): the layer's own buffers. The other inputs as _launch."""
        for name, src in (("R_in", residual), ("h_post_in", h_post), ("h_res_in", h_res)):
            src = src.reshape(src.shape[0], -1)
            if src.is_contiguous() and src.dtype == self.t[name].dtype and src.shape[1] == self.t[name].shape[1]:
                self._bind(name, src)
            else:
                self._bind(name, self.t[name])
                self.t[name].copy_(src[:self.T])
        self._launch(hidden_states, seq_lens, req_pool_indices, meta)
        return self.t["x_ffn"], self.t["R_out"], self.t["h_post_out"], self.t["h_res_out"]
