"""The static-megakernel GLM-5.3-Flash MoE layer that moe_static_hook.py puts into SGLang (no SGLang import here): one GPU of
the TP layer with the sublayer's end in the kernel (Kimi K3's scope, PR #811; moe_demo.build_graph): x -> the experts (the
shared one in their queue) -> allreduce_send -> hc_post MODE 1 with the coefficients SGLang's FFN mHC pre left, or (attn_next)
hc_post_mix + hc_pre: also the next layer's attention mHC pre. Built for the serving launch (-DSTATIC_RESET_IN_KERNEL,
launch_stream on the caller's stream).
  weights   its own copy, as #799 does: this rank's shard of the layer from the checkpoint (moe_demo.load_layer), packed
            into the kernel's bank (moe_demo.bank: 578 sub-experts of 256 features, the router padded to 384 rows), about
            1.8 GB per layer and GPU; SGLang keeps its own (its prefill and other batch sizes use them)
  ranks     this process GPU `rank` of the group's layer (compile_static rank / exchange: the exchange region in torch
            symmetric memory over the group, as SGLang's custom all-reduce allocates its push buffers; ExchangePool: the
            layers' regions in one allocation); every rank builds its layers before any rank launches one (the caller's
            barrier)
  run_mhc   (x, the FFN's residual streams, h_post, h_res) -> the new streams [T, 4 H] bf16 (attn_next: and the next
            layer's x_attn, h_post, h_res), the layer's own buffers; bound when contiguous (else copied)
  launch    launch_stream on the current stream (a CUDA graph captures it); pdl: a programmatic launch, the kernel lets the
            next one launch at its start (-DSTATIC_PDL_TRIGGER=1)
  plan      moe_demo.round_robin_plan, or a plan file of moe_demo's graph
"""
import os
import sys

import torch

ROOT = os.environ.get("MOE_STATIC_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))   # default: this checkout
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "demo", "glm53_flash"))

T = int(os.environ.get("STATIC_TOKENS", "8"))   # tokens (rows) per step: the kernel's T
TOKENS = T   # the default of StaticMoE(tokens=): a process may build layers of several T


def _backed(v: torch.Tensor, rows: int) -> bool:
    """v (contiguous) is the first rows of a buffer of at least `rows` rows of its row stride (sglang_kda's backed)."""
    if not v.is_contiguous():
        return False
    rowb = (v.stride(0) if v.dim() > 1 else 1) * v.element_size()
    return v.untyped_storage().nbytes() - v.storage_offset() * v.element_size() >= rows * rowb


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


class ExchangePool:
    """The exchange regions of several layers in one allocation of symmetric memory over the process group (one rendezvous,
    one multicast object): `slots` regions of the first one's size (rounded up to 64 KB), taken in order. Every rank takes
    them in the same order (its layers in the same order); the first take is collective. The pool owns the memory: it must
    stay referenced as long as the layers run (a pool collected after the builds frees the regions under the kernels: the
    layers' second launch then faults, found 2026-10-28)."""

    def __init__(self, group, dev, slots: int):
        self.group, self.dev, self.slots, self.taken, self.buf = group, dev, slots, 0, None

    def take(self, nbytes: int) -> tuple:
        """(the multicast address, this GPU's address) of the next region of nbytes."""
        if self.buf is None:
            self.slot = -(-nbytes // 65536) * 65536
            self.buf, self.handle, self.mc = _exchange_region(self.slot * self.slots, self.dev, self.group)
        assert nbytes <= self.slot and self.taken < self.slots, "ExchangePool: more or larger regions than its slots"
        off = self.taken * self.slot
        self.taken += 1
        return self.mc + off, self.buf.data_ptr() + off


class StaticMoE:
    """One chosen MoE layer of this rank: the built static layer and its graph tensors."""

    def __init__(self, layer: int, model_path: str, rank: int, tp: int, dev, out_root: str, group, pdl: bool = False,
                 keep_weights: bool = False, plan: str = None, pool: ExchangePool = None, attn_next: dict = None,
                 tokens: int = None, bank: dict = None, weights: dict = None, rows: int = None):
        """keep_weights: keep the checkpoint's tensors of this rank (self.w, for the hook's references); else only the bank.
        group: the TP process group (the module docstring); the exchange region from `pool` (ExchangePool over the group),
        else its own. attn_next: the next decoder layer's attention mHC pre at the end too {"hc_fn" [24, 4 H], "hc_base" [24],
        "hc_scale" [3] fp32 (its hc_attn_*), "norm" [H] bf16 (its input_layernorm), "norm_eps", "rms_eps", "hc_eps",
        "iters"}: run_mhc also returns its x_attn, h_post, h_res (moe_demo.build_graph attn_next). tokens: the rows T of
        this layer (default TOKENS; a process may hold several). bank: another StaticMoE's expert bank of this layer
        (self.bank: the weights shared; then its checkpoint tensors as `weights`, for keep_weights). rows: the live rows
        (default T): the layer of a speculative verify step of `rows` < T rows (bs x the draft tokens): the rows past them
        are idle: not routed (moe_demo.build_graph valid_rows: no expert of their own), their outputs unused."""
        import moe_demo
        from mirage.mpk import compiler
        from mirage.mpk.static_megakernel import StaticMegakernel

        T = self.T = tokens if tokens is not None else TOKENS
        self.rows = rows if rows is not None else T
        assert 0 < self.rows <= T, f"StaticMoE: {rows} live rows of {T}"
        w = (weights if keep_weights else None) if bank is not None else moe_demo.load_layer(model_path, layer, tp, rank, dev)
        self.bank = bank if bank is not None else moe_demo.bank(w)
        self.t = moe_demo.gpu_tensors(self.bank, torch.zeros(T, moe_demo.H, dtype=torch.bfloat16, device=dev),
                                      attn_next=attn_next)
        self.group = group
        self.attn_next = attn_next is not None
        self.w = w if keep_weights else None
        del w
        mpk = StaticMegakernel(num_gpus=tp)
        grids = moe_demo.build_graph(mpk, self.t, attn_next=None if attn_next is None else {
            k: attn_next[k] for k in ("rms_eps", "hc_eps", "norm_eps", "iters")},
            valid_rows=self.rows if self.rows < T else None)
        if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the layer's GEMM weights into L2 at its start (core.cuh kernel_begin)
            mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))
        plan = moe_demo.round_robin_plan(mpk, grids) if plan is None else compiler.plan_from_file(plan)
        if len(plan["lists"]) == 1:   # a one-GPU plan file (moe_demo --search): every rank the same lists
            plan["lists"] = plan["lists"] * mpk.num_gpus
        out_dir = os.path.join(out_root, f"rank{rank}",
                               f"layer{layer}{f'_t{T}' if T != 8 else ''}{'_pre' if attn_next is not None else ''}"
                               f"{f'_v{self.rows}' if self.rows < T else ''}")
        paths, self.info = compiler.compile_plan(mpk, plan, os.path.join(out_dir, "schedules"))
        self.pool = pool   # the exchange region's owner (its symmetric memory lives as long as the pool: keep it referenced)
        if pool is not None:
            exchange = pool.take(mpk.exchange_region_bytes())
        else:
            self.xbuf, self.xhandle, mc = _exchange_region(mpk.exchange_region_bytes(), dev, group)
            exchange = (mc, self.xbuf.data_ptr())
        self.sk = mpk.compile_static(paths, gpu_tensors=[self.t], out_dir=os.path.join(out_dir, "build"),
                                     extra_flags=["-DSTATIC_RESET_IN_KERNEL", f"-DSTATIC_PDL_TRIGGER={int(pdl)}"],
                                     device=dev.index, rank=rank, exchange=exchange)
        self.layer, self.pdl = layer, pdl
        self._at = {}   # graph tensor -> the address it is bound to now
        self.binds = {}   # run_mhc: per input, bound (True) or copied (False)

    def _bind(self, name: str, v: torch.Tensor) -> bool:
        """Graph tensor `name` read at v's address from the next launch on (v contiguous, of its dtype and row width, and
        either of T rows or the first rows of a buffer of at least T rows: a CUDA graph's static buffer or a static layer's
        output, e.g. a verify step's 6 rows of a fused attention layer's x_ffn; the kernel reads the rows beyond v's as idle
        rows), else v copied into the layer's own tensor (bound there; fewer rows than T: into its first rows, the others
        keep what they hold). Returns whether v is bound."""
        own = self.t[name]
        v = v.reshape(v.shape[0], -1)
        ok = v.is_contiguous() and v.dtype == own.dtype and v.shape[1] == own.shape[1] and \
            (v.shape[0] == own.shape[0] or _backed(v, own.shape[0]))
        if not ok:
            rows = v.shape[0]
            own[:rows].copy_(v)
            v = own
        if self._at.get(name) != v.data_ptr():
            self.sk.rebind(name, v.data_ptr())
            self._at[name] = v.data_ptr()
        return ok

    def run_mhc(self, x: torch.Tensor, streams: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
        """The layer with its end (group): one launch on the current stream; returns the new residual streams [8, 4 H] bf16
        (the layer's own buffer) = h_res's mix of `streams` + h_post x the MoE output summed over the group's GPUs. x [8, H]
        bf16 (the FFN's input), streams [8, 4 H] bf16 (its residual), h_post [8, 4], h_res [8, 16] fp32 (MHCState's; any
        shape of those elements): bound when contiguous and of the dtype, else copied (self.binds). attn_next: (the new
        streams, the next layer's x_attn [8, H] bf16, its h_post [8, 4], h_res [8, 16] fp32)."""
        rows, T = x.shape[0], self.T   # a verify step of fewer tokens: its rows of the layer's buffers (the others idle)
        assert 0 < rows <= T, f"run_mhc: {rows} rows, the layer has {T}"
        self.binds = {k: self._bind(k, v[:T]) for k, v in (("x", x), ("streams", streams), ("h_post", h_post),
                                                             ("h_res", h_res))}
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        if self.attn_next:   # and the next layer's attention input and coefficients
            return (self.t["streams_out"][:rows], self.t["x_attn"][:rows], self.t["h_post_next"][:rows],
                    self.t["h_res_next"][:rows])
        return self.t["streams_out"][:rows]
