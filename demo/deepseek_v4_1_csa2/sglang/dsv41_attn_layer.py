"""The DeepSeek-V4.1-Flash attention layer that dsv41_attn_hook.py puts into SGLang (no SGLang import here): one TP rank's layer
(demo/deepseek_v4_1_csa2), T = 8 rows, built for one GPU: y is this rank's partial sum, and SGLang's fused all-reduce and mHC
follow. A serving build on the caller's stream: PDL, no start barrier (STATIC_RESET_PARITY).

  weights  this rank's shard from the checkpoint (the demo's layer_tensors)
  caches   bind_pools: the graph tensors rebound to SGLang's own pools (SGLANG_DSV4_KV_LAYOUT=v41_new: the kernels' row bytes):
           the window pool of the layer, the compressed KV and index K pools of its source layer, the compressor's ring.
           StepTables: this step's locations out of SGLang's metadata. So SGLang's prefill fills what the kernels read, and a
           step with another row count runs SGLang's layer on the same state
  sources  a Reuse or Reindex layer reads its source layers' tensors (Sources): the same memory
  plan     plans/<variant>_r<ratio>_p<max_pos>_sms<SMs>.json when it exists, else the demo's hand plan for these SMs
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DSV41_STATIC_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "demo"))   # the demos as packages: both have a demo.py and a weights.py

from deepseek_v4_1_csa2 import demo as attn_demo          # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights    # noqa: E402

T, HID = attn_demo.T, attn_demo.HID
FREE_SMS = 24   # SMs left out of the launch: SGLang's mHC side stream runs on them (without them the seam starts about 27 us late)
FLAGS = ["-DSTATIC_RESET_IN_KERNEL", "-DSTATIC_NO_START_BARRIER", "-DSTATIC_RESET_PARITY"]
_readers = {}


def reader(ckpt: str):
    return _readers.setdefault(ckpt, attn_weights.checkpoint_reader(ckpt))


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
    them, so no copy kernel sits between two static layers."""

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
                 tables: StepTables):
        cfg = sources.cfg
        self.layer, self.pdl, self.steps = layer, pdl, 0
        self.variant = attn_weights.variant_of(cfg, layer)
        ratio = cfg["compress_ratios"][layer]
        self.ratio = ratio
        csrc = layer == cfg["candidate_source_layer"]
        rows = (max_pos + 2 + 255) // 256 * 256
        cap = max(((max_pos + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
        w = {**attn_weights.layer_tensors(reader(ckpt), cfg, layer, tp, rank, rows, dev),
             **attn_demo.state_tensors(self.variant, ratio, csrc, cap, dev), **sources.shared(layer, self.variant)}
        w.update({k: v for k, v in tables.t.items() if k in w})
        if "compress_pages" in w:
            w["compress_pages"] = tables.pages[ratio]
        sms = torch.cuda.get_device_properties(dev).multi_processor_count - FREE_SMS
        plan = os.path.join(HERE, "plans", f"{self.variant}_r{ratio}_p{max_pos}_sms{sms}.json")
        # the node buffers are torch tensors the kernel writes to: they must stay referenced as long as the layer runs
        self.sk, self.bufs, self.info = attn_demo.build([w], self.variant, ratio, csrc, 64 // tp,
                                                        os.path.join(out_root, f"rank{rank}", f"attn{layer}"),
                                                        plan=plan if os.path.exists(plan) else "hand",
                                                        extra_flags=FLAGS + [f"-DSTATIC_PDL_TRIGGER={int(pdl)}"], device=dev.index,
                                                        free_sms=FREE_SMS)
        self.w = w
        sources.note(layer, w)

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

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """One decode step of T rows: this rank's y [T, H] bf16 (the layer's own buffer). The step's tables are filled (StepTables)
        and x is read where SGLang left it when it is contiguous bf16 [T, H] (a CUDA graph captures the address), else a copy."""
        ok = x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape == (T, HID)
        ptr = x.data_ptr() if ok else self.w["x"].data_ptr()
        if getattr(self, "x_ptr", None) != ptr:
            self.sk.rebind("x", ptr)
            self.x_ptr = ptr
        if not ok:
            self.w["x"].copy_(x)
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        self.steps += 1
        return self.w["y"]
