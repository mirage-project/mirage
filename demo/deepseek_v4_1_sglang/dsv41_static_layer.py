"""The DeepSeek-V4.1-Flash static layers that dsv41_static_hook.py puts into SGLang (no SGLang import here): one TP rank's
attention (demo/deepseek_v4_1_csa2) and MoE (demo/deepseek_v4_1_moe), T = 8 rows. Without an exchange pool each is built for one
GPU: y is this rank's partial sum and SGLang's own all-reduce stays outside. With one (ExchangePool, torch symmetric memory over
the TP group) each is this rank of a layer on tp GPUs: the all-reduce is in the kernel (sum_store) and y is the sum; every rank's
layers must be built before any rank's first launch. SGLang's mHC stays outside. Serving build (-DSTATIC_RESET_IN_KERNEL),
launch_stream on the caller's stream.

  weights  this rank's shard from the checkpoint (the demos' layer_tensors); the MoE's expert bank is its own copy
  caches   bind_pools: the graph tensors rebound to SGLang's own pools (SGLANG_DSV4_KV_LAYOUT=v41_new: the kernels' row bytes):
           the window pool of the layer, the compressed KV and index K pools of its source layer, the compressor's ring.
           tables(): this step's locations out of SGLang's metadata. So SGLang's prefill fills what the kernels read, and a
           step with another row count runs SGLang's layer on the same state. Without bind_pools (the fallback for an SGLang on
           another layout) the layer keeps its own caches: replay() launches it over the prompt on SGLang's prefill x, the rows
           of every step must be those of the prompt, and every step after the prefill must run here
  sources  a Reuse or Reindex layer reads its source layers' tensors (Source): the same memory
"""
import os
import sys

import torch

ROOT = os.environ.get("DSV41_STATIC_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "demo"))   # the demos as packages: both have a demo.py and a weights.py

from deepseek_v4_1_csa2 import demo as attn_demo          # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights    # noqa: E402
from deepseek_v4_1_moe import demo as moe_demo            # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights      # noqa: E402

T, HID = attn_demo.T, attn_demo.HID
_readers = {}


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
    """The layers' exchange regions in one allocation over the process group (demo/glm53_flash/sglang_moe/moe_static_layer.py):
    `slots` regions of the first one's size, taken in the same order on every rank. It must stay referenced while the layers run."""

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


def reader(ckpt: str):
    return _readers.setdefault(ckpt, attn_weights.checkpoint_reader(ckpt))


def _bind_x(layer, x: torch.Tensor):
    """Graph tensor x reads SGLang's x when it is contiguous bf16 [T, H] (a CUDA graph captures the address), else a copy."""
    ok = x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape == (T, HID)
    ptr = x.data_ptr() if ok else layer.w["x"].data_ptr()
    if getattr(layer, "x_ptr", None) != ptr:
        layer.sk.rebind("x", ptr)
        layer.x_ptr = ptr
    if not ok:
        layer.w["x"].copy_(x)


class Sources:
    """The state one layer's readers share, by the checkpoint's source lists (model.py L1172): compress_cache and index_k from
    the last kv_source_layers entry, compress_idx from the last index_source_layers entry, candidates from
    candidate_source_layer."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kv, self.index, self.cand = None, None, None

    def shared(self, layer: int, variant: str) -> dict:
        out = {}
        if variant in ("reuse", "reindex"):
            if self.kv is None:
                raise RuntimeError(f"layer {layer}: its KV source layer is not static (replace the group together)")
            out["compress_cache"] = self.kv["compress_cache"]
        if variant == "reuse":
            if self.index is None:
                raise RuntimeError(f"layer {layer}: its top-k source layer is not static")
            out["compress_idx"] = self.index["compress_idx"]
        if variant == "reindex":
            out["index_k"] = self.kv["index_k"]
            if self.cand is None:
                raise RuntimeError(f"layer {layer}: the candidate source layer is not static")
            out["candidates"] = self.cand["candidates"]
        return out

    def note(self, layer: int, w: dict):
        if layer in self.cfg["kv_source_layers"]:
            self.kv = w
        if layer in self.cfg["index_source_layers"]:
            self.index = w
        if layer == self.cfg["candidate_source_layer"]:
            self.cand = w


class StepTables:
    """One step's locations, one set for all the attention layers (their graph tensors are these): the first layer of a step fills
    them (tables), so no copy kernel sits between two static layers."""

    def __init__(self, dev, caps: dict):
        z = lambda *shape, dt=torch.int32: torch.zeros(*shape, dtype=dt, device=dev)
        self.t = {"positions": z(T), "window_rows": z(T, attn_demo.W), "window_loc": z(T), "req_slots": z(T, dt=torch.int64)}
        self.pages = {r: z(T, cap // 128) for r, cap in caps.items()}   # by compress ratio, entries of 128 rows

    def fill(self, core, positions, req_pool_indices):
        """SGLang's metadata core (DSV4AttnMetadata) of this step."""
        t = self.t
        t["positions"].copy_(positions.view(-1))
        t["window_rows"].copy_(core.swa_page_indices[:T])
        t["window_loc"].copy_(core.swa_out_cache_loc[:T])
        t["req_slots"].copy_(req_pool_indices[:T])
        pt = core.page_table[:T].to(torch.int32)
        for r, pages in self.pages.items():
            # SGLang's pages: 256 / r rows; ratio 1 expands to entries of 128 (expand_index_page_table)
            e = (pt[:, :, None] * 2 + torch.arange(2, device=pt.device, dtype=torch.int32)).flatten(1) if r == 1 else pt
            m = min(e.shape[1], pages.shape[1])
            pages[:, :m].copy_(e[:, :m])


class StaticAttn:
    def __init__(self, layer: int, ckpt: str, rank: int, tp: int, dev, out_root: str, max_pos: int, sources: Sources, pdl: bool,
                 exchange_pool: ExchangePool = None, tables: StepTables = None):
        cfg = sources.cfg
        self.layer, self.pdl, self.steps, self.filled, self.pool = layer, pdl, 0, 0, False
        self.variant = attn_weights.variant_of(cfg, layer)
        ratio = cfg["compress_ratios"][layer]
        self.ratio = ratio
        csrc = layer == cfg["candidate_source_layer"]
        rows = (max_pos + 2 + 255) // 256 * 256
        cap = max(((max_pos + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
        w = {**attn_weights.layer_tensors(reader(ckpt), cfg, layer, tp, rank, rows, dev),
             **attn_demo.state_tensors(self.variant, ratio, csrc, cap, dev), **sources.shared(layer, self.variant)}
        if tables is not None:
            shared = {k: v for k, v in tables.t.items() if k in w}
            w.update(shared)
            if "compress_pages" in w:
                w["compress_pages"] = tables.pages[ratio]
        out = os.path.join(out_root, f"rank{rank}", f"attn{layer}")
        flags = ["-DSTATIC_RESET_IN_KERNEL", f"-DSTATIC_PDL_TRIGGER={int(pdl)}"] + os.environ.get("DSV41_STATIC_FLAGS", "").split()
        # the node buffers are torch tensors the kernel writes to: they must stay referenced as long as the layer runs
        rk = dict(num_gpus=tp, rank=rank, exchange_pool=exchange_pool) if exchange_pool is not None else {}
        self.sk, self.bufs, self.info = attn_demo.build([w], self.variant, ratio, csrc, 64 // tp, out, extra_flags=flags,
                                                        device=dev.index, **rk)
        self.w = w
        sources.note(layer, w)

    def _launch(self):
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)

    def bind_pools(self, pool):
        """The caches are SGLang's: its pools' rows have the kernels' bytes and stride (the module comment)."""
        L, v = self.layer, self.variant
        assert pool.kv_layout.value == "v41_new", f"{pool.kv_layout}: run SGLang with SGLANG_DSV4_KV_LAYOUT=v41_new"
        self.sk.rebind("window_cache", pool.get_swa_raw_buffer(L).data_ptr())
        if v != "swa":
            assert pool.get_extra_key_layout(L).value == "v41_fp4_new" and pool.get_extra_key_bytes_per_token(L) == 288
            self.sk.rebind("compress_cache", pool.get_extra_key_buffer(L).data_ptr())
        if v in ("full_r2", "full_r1", "reindex"):
            assert pool.get_index_k_page_size(self.ratio) == 128 and not pool.low_ratio_index_k_is_split(L)
            self.sk.rebind("index_k", pool.get_index_k_with_scale_buffer(L).data_ptr())
        if "cmp_state" in self.w:
            state = pool.get_attention_compress_states(L)
            assert state.ring_size == self.ratio == 2 and state.kv_score_buffer.kv_score.shape[1] == 1024
            self.sk.rebind("cmp_state", state.kv_score_buffer.kv_score.data_ptr())
        self.pool = True

    def replay(self, x: torch.Tensor, positions: torch.Tensor):
        """The caches from a prefill of T prompts of the same length: x [T L, H], positions [T L], one launch per position."""
        L = x.shape[0] // T
        xs, ps = x.view(T, L, -1), positions.view(T, L)
        for i in range(L):
            self.w["x"].copy_(xs[:, i])
            self.w["positions"].copy_(ps[:, i])
            attn_demo.window_tables(self.w)
            self._launch()
        self.filled += L

    def run(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """One decode step of T rows: this rank's y [T, H] bf16 (the layer's own buffer), or with the exchange the sum. With the
        pools bound the step's tables are filled (StepTables) and x is read where SGLang left it: no copy kernel."""
        if self.pool:
            _bind_x(self, x)
        else:
            self.w["x"].copy_(x)
            self.w["positions"].copy_(positions.view(-1))
            attn_demo.window_tables(self.w)
        self._launch()
        self.steps += 1
        return self.w["y"]


class StaticMoE:
    def __init__(self, layer: int, ckpt: str, rank: int, tp: int, dev, out_root: str, pdl: bool,
                 exchange_pool: ExchangePool = None):
        self.layer, self.pdl, self.steps = layer, pdl, 0
        read = reader(ckpt)
        w = moe_weights.layer_tensors(read, layer, tp, rank, dev)
        w.update(moe_weights.routed_tensors(read, layer, tp, [rank], dev)[0])
        w["x"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
        w["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
        out = os.path.join(out_root, f"rank{rank}", f"moe{layer}")
        flags = ["-DSTATIC_RESET_IN_KERNEL", f"-DSTATIC_PDL_TRIGGER={int(pdl)}"] + os.environ.get("DSV41_STATIC_FLAGS", "").split()
        rk = dict(num_gpus=tp, rank=rank, exchange_pool=exchange_pool) if exchange_pool is not None else {}
        self.sk, self.bufs, self.info = moe_demo.build([w], out, extra_flags=flags, device=dev.index, **rk)   # (bufs: as StaticAttn)
        self.w = w

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """One decode step of T rows: this rank's part of y [T, H] bf16 (the layer's own buffer), or with the exchange the sum."""
        _bind_x(self, x)
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        self.steps += 1
        return self.w["y"]
