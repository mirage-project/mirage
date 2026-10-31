"""The static-megakernel KDA layer that kda_static_hook.py puts into SGLang (no SGLang import here, so a test can build it
alone): one GPU's attention with the sublayer boundary inside (as the DSA layer, sglang_dsa/dsa_static_layer.py; Kimi K3's
#811), on states laid out as SGLang's KDA pool (conv [slots][3][conv dim], temporal [slots][H][V][K] fp32), built for the
serving launch (-DSTATIC_RESET_IN_KERNEL, launch_stream on the caller's stream).
  layer      x -> gemm_tile (q|k|v|b|f_a|g_a) -> kda_decode -> o_proj (its K parts added, int64 fixed point: this rank's
             partial sum) -> allreduce_send (each 128-column block once its parts are added, bf16 to every rank) ->
             hc_post_mix (the sum over the ranks; the attention's mHC post with SGLang's h_post / h_res; the FFN's mHC mixes)
             -> hc_pre (the FFN's mHC pre with post_attention_layernorm): run_fused returns x_ffn, the streams R' and the
             FFN's h_post / h_res
  verify     the layer of a speculative verify step (SGLang's KDA target-verify contract with ReplaySSM): run_verify
  bind       x and the slot list bound to the caller's tensors (StaticKernel.rebind) when their dtype and shape are the
             layer's, else copied into the layer's own; a CUDA graph keeps the addresses of its capture
  plan       the grids and SM lists (compiler.compile_plan's plan, or a plan file): default plan_of with build_graph's grids
  ranks      built for the tp ranks, each process its own rank alone, the exchange region in torch symmetric memory over the
             TP group; every rank's layers must be built before any rank's first launch (the hook's barrier)
"""
import itertools
import os
import sys

import torch

ROOT = os.environ.get("KDA_STATIC_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))   # default: this checkout
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "demo", "glm53_flash"))

T, HC = int(os.environ.get("STATIC_TOKENS", "8")), 4   # tokens (rows) per step (the kernel's T), mHC streams
TOKENS = T   # the default of StaticKDA(tokens=) / layer_tensors(tokens=): a process may build layers of several T
O_PARTS = 4         # o_proj's K parts (added in int64 fixed point)


def backed(v: torch.Tensor, rows: int) -> bool:
    """v (contiguous) is the first rows of a buffer of at least `rows` rows of its row stride: a view of a larger buffer (a
    CUDA graph's static buffer, a static layer's own output), which a kernel bound to it may read past v's rows."""
    if not v.is_contiguous():
        return False
    rowb = (v.stride(0) if v.dim() > 1 else 1) * v.element_size()
    return v.untyped_storage().nbytes() - v.storage_offset() * v.element_size() >= rows * rowb


def exchange_region(nbytes: int, dev, group):
    """nbytes of torch symmetric memory over the process group, zero-filled (as dsa_static_layer._exchange_region): (the
    tensor, its rendezvous handle, the multicast address)."""
    from torch._C._distributed_c10d import _SymmetricMemory
    buf = _SymmetricMemory.empty_strided_p2p((nbytes,), [1], torch.uint8, dev, group.group_name)
    handle = _SymmetricMemory.rendezvous(buf)
    mc = int(handle.multicast_ptr)
    if mc == 0:
        raise RuntimeError("the TP group's symmetric memory has no multicast address (NVLS)")
    buf.zero_()
    return buf, handle, mc


# kda_decode's requests per task by T: T = 16 / 32 2 / 4 per task (128 tasks: one wave of the 148 SMs; 256 / 512 tasks of one
# took two / four); T = 8 one each (128 tasks already)
PAIRS = {16: 2, 32: 4}


def plan_of(mpk, grids: dict) -> dict:
    """grids {graph_idx: grid} for the nodes they name (the others: their layer's grid); the tasks dealt to the SMs in turn, in
    graph order (x slowest)."""
    from mirage.mpk import compiler
    num_sms, lists = mpk._num_sms(), []
    per_sm, cursor = [[] for _ in range(num_sms)], 0
    for n in compiler.graph_nodes(mpk):
        grid = grids.get(n.graph_idx, n.grid)
        for pos in itertools.product(*(range(d) for d in grid)):
            per_sm[cursor % num_sms].append([n.graph_idx, list(pos)])
            cursor += 1
    lists.append(per_sm)
    return {"grids": {str(k): list(v) for k, v in grids.items()}, "lists": lists}


def layer_tensors(layer: int, model_path: str, rank: int, tp: int, conv: torch.Tensor, temporal: torch.Tensor,
                  dev: torch.device, fused: dict, verify: dict = None, weights: dict = None, tokens: int = None) -> dict:
    """The graph tensors of this rank's layer on dev (weights read from the checkpoint, or `weights`: another build's
    tensors of this layer, shared), the boundary's (o_proj's partial sum o_fx, the streams and coefficients in and out, the
    FFN's mHC parameters: fused, StaticKDA's); verify (StaticKDA's): the framework's verify buffers (the rings, the
    per-token conv windows) and the per-launch nvalid / icw_row. tokens: the rows (default TOKENS)."""
    import kda_reference as ref
    T = tokens if tokens is not None else TOKENS
    if weights is None:
        s = ref.shard_kda(ref.load_kda_layer(ref.Checkpoint(model_path), layer), rank, tp, dev)
    else:
        s = {k: weights[v] for k, v in (("w_in", "w_in"), ("conv_w", "conv_w"), ("f_b", "f_b"), ("g_b", "g_b"),
                                        ("small", "small"), ("w_o", "w_o"))}
    H, HLD = s["w_o"].shape[0], s["f_b"].shape[0]
    assert conv.is_contiguous() and conv.dtype == torch.bfloat16 and tuple(conv.shape[1:]) == (3, 3 * HLD), \
        f"layer {layer}: conv state {tuple(conv.shape)} {conv.dtype}, expected [slots, 3, {3 * HLD}] bf16"
    assert temporal.is_contiguous() and temporal.dtype == torch.float32 and \
        tuple(temporal.shape[1:]) == (HLD // 128, 128, 128), \
        f"layer {layer}: temporal state {tuple(temporal.shape)} {temporal.dtype}, expected [slots, {HLD // 128}, 128, 128] fp32"
    z = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=dev)
    t = {"x": z((T, H), torch.bfloat16), "w_in": s["w_in"], "conv_w": s["conv_w"], "f_b": s["f_b"], "g_b": s["g_b"],
         "small": s["small"], "conv": conv.view(conv.shape[0], -1), "rec": temporal.view(temporal.shape[0], -1),
         "rows": z((T,), torch.int32), "w_o": s["w_o"], "o": z((T, HLD), torch.bfloat16)}
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = \
        fused["hc_base"].float().cpu(), fused["hc_scale"].float().cpu()
    t.update({"o_fx": z((T, H), torch.int64), "R_in": z((T, HC * H), torch.bfloat16), "h_post_in": z((T, HC), torch.float32),
              "h_res_in": z((T, HC * HC), torch.float32), "R_out": z((T, HC * H), torch.bfloat16),
              "hc_ffn_fn": fused["hc_fn"].float().contiguous().to(dev), "hc_ffn_small": hc_small.to(dev),
              "ffn_norm": fused["norm"].to(torch.bfloat16).contiguous().to(dev), "x_ffn": z((T, H), torch.bfloat16),
              "h_post_out": z((T, HC), torch.float32), "h_res_out": z((T, HC * HC), torch.float32)})
    if verify is not None:   # R = verify["requests"] (default 1) requests' draft tokens per launch (grid x = R; request r's
        # n = draft rows start at row r n, its slot in rows[r]; nvalid[0] = n)
        R = int(verify.get("requests", 1))
        assert T % R == 0 and 1 <= T // R <= 8, f"verify: {R} requests of {T // R} rows (at most 8) in {T}"
        for k in ("ring_v", "ring_k", "ring_g", "ring_beta", "icw"):
            assert verify[k].is_contiguous(), f"verify: {k} contiguous"
        t.update({k: verify[k] for k in ("ring_v", "ring_k", "ring_g", "ring_beta", "icw")})
        t.update({"nvalid": torch.full((R,), int(verify["draft"]), dtype=torch.int32, device=dev),
                  "icw_row": z((R,), torch.int32), "accepted": torch.ones(R, dtype=torch.int32, device=dev)})
    return t


def build_graph(mpk, t: dict, fused: dict, verify: dict = None) -> dict:
    """The layer's nodes on t's tensors (layer_tensors'); returns the grids plan_of gives by default {graph_idx: grid}. fused:
    {"rms_eps", "hc_eps", "norm_eps", "iters", "o_parts"}: the boundary after o_proj (the module docstring). verify
    (layer_tensors' verify tensors in t): {"draft"[, "requests"]}: the framework's speculative verify step of R = requests
    (default 1) requests' draft tokens (kda_decode_layer serve: T / R tokens per request, request r's rows from r T / R)."""
    from mirage.core import bfloat16, float32
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import HC_MIX_STRIDE
    H, T = t["w_o"].shape[0], t["x"].shape[0]
    attached = {}   # name -> its graph tensor, attached once (a second attach would be another tensor: no dependency)
    a = lambda k: attached.get(k) or attached.setdefault(k, mpk.attach_input(torch_tensor=t[k], name=k))
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    x, w_in, conv_w, f_b, g_b, small = a("x"), a("w_in"), a("conv_w"), a("f_b"), a("g_b"), a("small")
    conv_t, rec_t, rows, w_o = a("conv"), a("rec"), a("rows"), a("w_o")
    o = a("o")
    proj = nt((T, w_in.dim(0)), float32, "proj")
    mpk.gemm_tile_layer(input=x, weight=w_in, output=proj, combine="slots", reader_rearms=False)
    if verify is not None:
        mpk.kda_decode_layer(proj=proj, conv_weight=conv_w, f_b=f_b, g_b=g_b, small=small, conv_state=conv_t,
                             rec_state=rec_t, rows=rows, output=o, accepted=a("accepted"),
                             tokens_per_request=T // int(verify.get("requests", 1)),
                             serve={"nvalid": a("nvalid"), "icw_row": a("icw_row"), "icw": a("icw"), "ring_v": a("ring_v"),
                                    "ring_k": a("ring_k"), "ring_g": a("ring_g"), "ring_beta": a("ring_beta"),
                                    "draft": int(verify["draft"])})
    else:
        mpk.kda_decode_layer(proj=proj, conv_weight=conv_w, f_b=f_b, g_b=g_b, small=small, conv_state=conv_t,
                             rec_state=rec_t, rows=rows, output=o, requests_per_task=PAIRS.get(T, 1))
    # the boundary: all-reduce, the attention's mHC post, the FFN's mHC pre
    o_fx = a("o_fx")
    mpk.gemm_tile_layer(input=o, weight=w_o, output=o_fx, combine="add")
    sent = nt((T, H), bfloat16, "o_sent")
    mpk.allreduce_send_layer(input=o_fx, output=sent)
    mix = nt((T, HC_MIX_STRIDE), float32, "ffn_hc_mix")
    mpk.hc_post_mix_layer(sent=sent, residual=a("R_in"), coef=(a("h_post_in"), a("h_res_in")), fn=a("hc_ffn_fn"),
                          output=a("R_out"), mix=mix)
    mpk.hc_pre_layer(mix=mix, residual=a("R_out"), hc_small=a("hc_ffn_small"), gamma=a("ffn_norm"), output=a("x_ffn"),
                     rms_eps=fused["rms_eps"], hc_eps=fused["hc_eps"], norm_eps=fused["norm_eps"],
                     sinkhorn_iters=fused["iters"], export=(a("h_post_out"), a("h_res_out")))
    gemms = [n for n in compiler.graph_nodes(mpk) if n.name == "gemm_tile"]
    return {gemms[0].graph_idx: (gemms[0].grid[0], 2, 1), gemms[1].graph_idx: (gemms[1].grid[0], fused["o_parts"], 1)}


class StaticKDA:
    """One chosen layer of this rank: the built static layer and its graph tensors."""

    def __init__(self, layer: int, model_path: str, rank: int, tp: int, conv: torch.Tensor, temporal: torch.Tensor,
                 out_root: str, fused: dict, pdl: bool = True, plan=None, verify: dict = None, weights: dict = None,
                 tokens: int = None):
        """fused (the module docstring): {"group": the TP process group, "hc_fn" [24, 4 H], "hc_base" [24], "hc_scale" [3]
        fp32 (the decoder layer's hc_ffn_*), "norm" [H] bf16 (post_attention_layernorm), "norm_eps", "rms_eps", "hc_eps",
        "iters"} (as StaticDSA's). plan: a plan or a plan file (None: plan_of). verify: the layer of a speculative verify
        step (its draft tokens as rows, the rest idle): {"draft": the framework's draft tokens, "ring_v", "ring_k", "ring_g",
        "ring_beta": this layer's ReplaySSM rings, "icw": its per-token conv windows [rows, draft, 3, 3 HL D] bf16 (SGLang's
        intermediate_conv_window: after token t, (t, j) = x_{t - 2 + j})}; run_verify. verify["requests"] (default 1): R
        requests of T / R rows each (request r's rows from r T / R; run_verify). weights: another StaticKDA's tensors of
        this layer (its weights shared). tokens: the rows T of this layer (default TOKENS; a process may hold several)."""
        from mirage.mpk import compiler
        from mirage.mpk.static_megakernel import StaticMegakernel

        dev = torch.device("cuda", torch.cuda.current_device())
        self.T = tokens if tokens is not None else TOKENS
        self.requests = int(verify.get("requests", 1)) if verify is not None else 1
        self.t = layer_tensors(layer, model_path, rank, tp, conv, temporal, dev, fused, verify, weights, tokens=self.T)
        mpk = StaticMegakernel(num_gpus=tp)
        grids = build_graph(mpk, self.t, {"rms_eps": fused["rms_eps"], "hc_eps": fused["hc_eps"],
                                          "norm_eps": fused["norm_eps"], "iters": fused["iters"], "o_parts": O_PARTS}, verify)
        if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the layer's GEMM weights into L2 at its start (core.cuh kernel_begin)
            mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))
        if plan is None:
            plan = plan_of(mpk, grids)
        elif isinstance(plan, str):
            plan = compiler.plan_from_file(plan)
        plan = dict(plan, lists=plan["lists"] * tp)   # every rank: the same lists (no node splits its rows over the GPUs)
        out_dir = os.path.join(out_root, f"rank{rank}",
                               f"layer{layer}{f'_t{self.T}' if self.T != 8 else ''}{'_verify' if verify is not None else ''}")
        self.paths, self.info = compiler.compile_plan(mpk, plan, os.path.join(out_dir, "schedules"))
        self.mpk = mpk   # the graph (cost_records.records_of_run)
        flags = ["-DSTATIC_RESET_IN_KERNEL", "-DSTATIC_PDL_TRIGGER=1"]
        self.fused_args = fused
        self.verify, self.draft = verify is not None, (int(verify["draft"]) if verify is not None else 0)
        self.group = fused["group"]
        # this rank alone, the exchange region in symmetric memory
        self.xbuf, self.xhandle, mc = exchange_region(mpk.exchange_region_bytes(), dev, fused["group"])
        self.sk = mpk.compile_static(self.paths, gpu_tensors=[self.t], out_dir=os.path.join(out_dir, "build"),
                                     extra_flags=flags, device=dev.index, rank=rank, exchange=(mc, self.xbuf.data_ptr()))
        self.layer, self.pdl = layer, pdl
        self.inputs = None   # the last run's: "bound" or how they were copied
        self._bound = {}   # graph tensor name -> the address bound now
        self.binds = {}   # run_verify: per input, bound (True) or copied (False)
        self._nvalid = None   # run_verify: the draft tokens the layer's nvalid holds

    def _bind(self, name: str, ptr: int):
        if self._bound.get(name) != ptr:
            self.sk.rebind(name, ptr)
            self._bound[name] = ptr

    def run_fused(self, hidden_states, residual, h_post, h_res, indices):
        """The fused layer (the module docstring): one launch on the current stream from the attention's input hidden_states
        [8, H] bf16, the streams residual [8, 4 H] bf16 and the attention's h_post [8, 4] / h_res [8, 16] fp32 (bound when
        contiguous, else copied) to (x_ffn [8, H] bf16, R' [8, 4 H] bf16, the FFN's h_post [8, 4], h_res [8, 16] fp32): the
        layer's own buffers (fewer rows than T: those rows of them). indices: the T state slots (-1: idle)."""
        assert not self.verify
        n = self._mhc_inputs(hidden_states, residual, h_post, h_res)
        self._launch(hidden_states, indices)
        return self.t["x_ffn"][:n], self.t["R_out"][:n], self.t["h_post_out"][:n], self.t["h_res_out"][:n]

    def run_verify(self, hidden_states, residual, h_post, h_res, slot: torch.Tensor, icw_row: torch.Tensor):
        """The verify layer (StaticKDA verify): R = self.requests requests' n <= draft tokens each as the rows (request r's
        at rows r n .. r n + n - 1 of the inputs and of the layer: the framework's verify batch as it is; the layer's rows
        R n .. T - 1 idle) from the attention's inputs as run_fused's, the requests' mamba slots `slot` (int32 [>= R]: the
        pool's state and conv window, the ring) and their rows of the per-token conv windows `icw_row` (int32 [>= R]);
        returns run_fused's outputs' first R n rows. An input is bound (no copy) when it is the first rows of a buffer of
        at least T rows (backed: a CUDA graph's static buffer, a static layer's output; the kernel reads its idle rows
        too), else copied into the layer's own; slot / icw_row bound when int32 (self.binds). The pools are not written:
        the framework's commit folds the ring."""
        assert self.verify
        R, T = self.requests, self.T
        rows = hidden_states.shape[0]
        assert rows % R == 0 and 0 < rows // R <= self.draft, \
            f"run_verify: {rows} rows of {R} requests, the layer's draft tokens {self.draft}"
        n = rows // R
        binds = {}
        for name, src in (("x", hidden_states), ("R_in", residual), ("h_post_in", h_post), ("h_res_in", h_res)):
            own = self.t[name]
            src = src.reshape(rows, -1)
            ok = src.dtype == own.dtype and src.shape[1] == own.shape[1] and backed(src, T)
            if ok:
                self._bind(name, src.data_ptr())
            else:
                self._bind(name, own.data_ptr())
                own[:rows].copy_(src)
            binds[name] = ok
        for name, src in (("rows", slot), ("icw_row", icw_row)):
            own = self.t[name]
            ok = src.dtype == torch.int32 and backed(src, R)   # the kernel reads R entries
            if ok:
                self._bind(name, src.data_ptr())
            else:
                self._bind(name, own.data_ptr())
                own[:R].copy_(src[:R].to(torch.int32))
            binds[name] = ok
        if self._nvalid != n:   # constant per verify graph (the framework's draft tokens): set once
            self.t["nvalid"].fill_(n)
            self._nvalid = n
        self.binds = binds
        self.inputs = (f"{rows} rows of {R} requests; bound: {', '.join(k for k, v in binds.items() if v) or 'none'}; "
                       f"copied: {', '.join(k for k, v in binds.items() if not v) or 'none'}")
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        return self.t["x_ffn"][:rows], self.t["R_out"][:rows], self.t["h_post_out"][:rows], self.t["h_res_out"][:rows]

    def _mhc_inputs(self, hidden_states, residual, h_post, h_res) -> int:
        """The boundary's inputs bound or copied (their first rows when fewer than T); returns the rows."""
        n, T = hidden_states.shape[0], self.T
        assert 0 < n <= T
        for name, src in (("R_in", residual), ("h_post_in", h_post), ("h_res_in", h_res)):
            src = src.reshape(src.shape[0], -1)
            if n == T and src.is_contiguous() and src.dtype == self.t[name].dtype and src.shape[1] == self.t[name].shape[1]:
                self._bind(name, src.data_ptr())
            else:
                self._bind(name, self.t[name].data_ptr())
                self.t[name][:n].copy_(src[:n])
        return n

    def _launch(self, hidden_states: torch.Tensor, indices: torch.Tensor):
        n, T = hidden_states.shape[0], self.T
        if n == T and hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous() and \
                tuple(hidden_states.shape) == tuple(self.t["x"].shape) and indices.dtype == torch.int32 and \
                indices.is_contiguous():
            self._bind("x", hidden_states.data_ptr())
            self._bind("rows", indices.data_ptr())
            self.inputs = "bound"
        else:
            self._bind("x", self.t["x"].data_ptr())
            self._bind("rows", self.t["rows"].data_ptr())
            self.t["x"][:n].copy_(hidden_states)
            self.t["rows"][:n].copy_(indices[:n])
            if n < T:
                self.t["rows"][n:].fill_(-1)   # idle rows
            self.inputs = f"copied (x {hidden_states.dtype} {tuple(hidden_states.shape)}, slots {indices.dtype})"
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
