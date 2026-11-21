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
import hashlib
import json
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
# DSV41_STATIC_TIMING=1 (measurement, 2026-11-17): the timing build (every task stamps its start and end; dsv41_static_prof
# then samples per node). Not a serving number: the stamps cost a little in every task.
TIMING = os.environ.get("DSV41_STATIC_TIMING", "0") != "0"
I1C_VARIANTS = tuple(os.environ.get("DSV41_I1C_VARIANTS", "full_r2,full_r1,reindex").split(","))   # (DSV41_I1C)
# (11-20, default off) DSV41_SHARE_MODULES=1: same-kind layers share one kernel module. Each layer is otherwise a module of its
# own (compile_static loads its own copy of the .so even from the build cache), so in situ every kernel's code comes from HBM.
# With it, the first layer of a kind builds and loads its module as before; a later layer whose generated layer.cu, flags,
# device and rank are the same loads nothing and takes that module and its node buffers; before a launch of the module after
# another layer's, the layer rebinds all its graph tensors (static_host_rebind: the buffer slots read by pointer and the tensor
# maps over them; a CUDA graph captures the arguments of its moment). The code is then fetched from HBM once a step and found in
# L2 by the next layers of the kind (the weights stream with evict-first loads; docs/draft.md "Code fetch, measured"). The
# per-SM launch numbers, both buffer sets and the exchange region follow the module's own launches, which the layers make in
# turn. Not with the timing build or the in-situ samples (each reads its module's stamps).
SHARE = os.environ.get("DSV41_SHARE_MODULES", "0") != "0"
_SHARED = {}   # md5 of layer.cu, the flags, device, rank and GPU count -> [sk, bufs, info, w] of the first layer built with it
# a graph tensor the module cannot rebind is one of its own buffers: a sharing layer reads the first layer's (aliased in its w);
# only these may be such (anything else, e.g. a state tensor of the layer's own, is an error). y: with the in-kernel
# all-reduce the sum's own buffer (written at its build address); with mHC SGLang takes the sublayer's end from hc_state, and y
# only marks the output (_OUTS in the hook), so every layer of a group returning the first layer's y is the same. x_out: a
# post that computes no next input (the last layer) reads and writes none
SHARE_ALIAS = ("oa_q", "y", "x_out")
# DSV41_STATIC_PROF with sharing (11-20): each layer its own per-SM stamps and start-barrier word, bound like a graph tensor
# (host.cuh STATIC_STAMP_REBIND: "@stamps" / "@start_barrier"); dsv41_static_prof reads the layer's own (prof_stamps)
PROF_SHARE = SHARE and bool(os.environ.get("DSV41_STATIC_PROF"))


def _head_demo():
    """head_demo.py beside this file, whatever sys.path the process has (a bench script elsewhere, the forced boot)"""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import head_demo
    return head_demo


class _ShareHit(Exception):
    def __init__(self, entry):
        super().__init__("module shared")
        self.entry = entry


def _build_shared(build, *args, **kw):
    """build(*args, **kw) (attn_demo.build / moe_demo.build): (sk, bufs, info, shared). With DSV41_SHARE_MODULES, compile_static
    first generates the layer's code; when a module of the same code, flags, device and rank is loaded, the build stops there
    and returns that module and its node buffers (shared: the first layer's w, for _init_binds; else None)."""
    if not SHARE:
        return (*build(*args, **kw), None)
    if TIMING:
        raise RuntimeError("DSV41_SHARE_MODULES: not with DSV41_STATIC_TIMING (each reads its module's task stamps)")
    from mirage.mpk import static_megakernel, static_schedule
    orig = static_megakernel.StaticMegakernel.compile_static
    key = []

    def compile_static(self, schedule_paths, gpu_tensors=None, out_dir=None, extra_flags=None, profile=False, device=0,
                       rank=None, exchange=None):
        schedules = []
        for path in schedule_paths:
            with open(path) as f:
                schedules.append(json.load(f))
        h = hashlib.md5(static_schedule.generate_code(self, schedules, profile).encode())
        h.update(repr((list(extra_flags or []), int(device), rank, len(schedule_paths))).encode())
        key.append(h.hexdigest())
        if key[-1] in _SHARED:
            raise _ShareHit(_SHARED[key[-1]])
        return orig(self, schedule_paths, gpu_tensors, out_dir, extra_flags=extra_flags, profile=profile, device=device,
                    rank=rank, exchange=exchange)

    static_megakernel.StaticMegakernel.compile_static = compile_static
    try:
        sk, bufs, info = build(*args, **kw)
    except _ShareHit as e:
        return (*e.entry[:3], e.entry[3])
    finally:
        static_megakernel.StaticMegakernel.compile_static = orig
    _SHARED[key[-1]] = [sk, bufs, info, None]   # (w: set by the layer, _init_binds)
    return sk, bufs, info, None


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


HC = 4   # mHC streams


def mhc_tensors(read, layer: int, kind: str, gamma_name: str, dev, stats: bool = False) -> dict:
    """The tensors of StaticMegakernel.dsv41_mhc_nodes for one sublayer (kind "attn" or "ffn"): the streams in and out, the mixes'
    weight (fn as bf16 hi | lo rows, padded to 128 rows), base and scale, pre, and with gamma_name the next sublayer's input."""
    p = f"layers.{layer}."
    fn, base, scale = (read(p + f"hc_{kind}_{n}").float() for n in ("fn", "base", "scale"))
    hi = fn.bfloat16()
    fn_w = torch.cat([hi, (fn - hi.float()).bfloat16(), torch.zeros(128 - 2 * fn.shape[0], fn.shape[1], dtype=torch.bfloat16)])
    z = lambda *shape, dt=torch.bfloat16: torch.zeros(*shape, dtype=dt, device=dev)
    w = dict(streams=z(T, HC * HID), streams_out=z(T, HC * HID), hc_fn_w=fn_w.to(dev).contiguous(),
             hc_small=torch.cat([base, scale]).to(dev).contiguous(), h_pre=z(T, HC, dt=torch.float32), x_out=z(T, HID))
    if gamma_name is not None:
        w["gamma_next"] = read(gamma_name).bfloat16().to(dev).contiguous()
    if stats:   # only the coefficients: post and comb for SGLang's own post
        w.update(h_post=z(T, HC, dt=torch.float32), h_comb=z(T, HC * HC, dt=torch.float32))
    return w


class MhcMixin:
    """A static layer whose kernel also runs mHC: the streams it reads, and the state SGLang's next sublayer takes. Also the
    layer's bindings (every rebind goes through _rb; DSV41_SHARE_MODULES: _own before a launch)."""

    def _rb(self, name: str, ptr: int):
        """sk.rebind, remembered as this layer's binding; with a shared module only while this layer holds it (else _own)."""
        self.binds[name] = ptr
        if not SHARE or getattr(self.sk, "_owner", None) is self:
            self.sk.rebind(name, ptr)

    def _init_binds(self, w: dict, first: dict):
        """(DSV41_SHARE_MODULES) this layer's bindings: every graph tensor of w that the module reads by pointer or through a
        tensor map, rebound now (this layer holds the module from here on). first: the w of the layer that built the module
        (None: this one built it, and whatever it cannot rebind is its own). In a sharing layer a tensor the module cannot
        rebind is the first layer's (a node's output, kept at its build address, or unused): allowed for SHARE_ALIAS only,
        and its w takes the first layer's tensor of that name (the one the module writes)."""
        fixed = []
        for name, t in w.items():
            if not torch.is_tensor(t):
                continue
            try:
                self.sk.rebind(name, t.data_ptr())
                self.binds[name] = t.data_ptr()
            except KeyError:
                fixed.append(name)
        bad = [n for n in fixed if n not in SHARE_ALIAS] if first is not None else []
        if bad:
            raise RuntimeError(f"DSV41_SHARE_MODULES: {type(self).__name__} L{self.layer}: graph tensors the module cannot "
                               f"rebind (a layer's own state or weights would be another layer's): {bad}")
        if first is not None:
            for name in fixed:
                w[name] = first[name]
        else:
            next(v for v in _SHARED.values() if v[0] is self.sk)[3] = w
        if PROF_SHARE:   # this layer's own stamps (the module's G points at the last launch's layer's)
            from mirage.mpk.static_harness import Harness
            lay = Harness(self.sk).layout
            dev = next(v.device for v in w.values() if torch.is_tensor(v))
            self.prof_stamps = torch.zeros(lay["num_sms"] * lay["num_stamps"], dtype=torch.int64, device=dev)
            self.prof_sb = torch.zeros(8, dtype=torch.int64, device=dev)
            for name, buf in (("@stamps", self.prof_stamps), ("@start_barrier", self.prof_sb)):
                self.sk.rebind(name, buf.data_ptr())
                self.binds[name] = buf.data_ptr()
        self.sk._owner = self
        if os.environ.get("DSV41_SHARE_LOG"):
            print(f"[dsv41_static] {type(self).__name__} L{self.layer}: {'shares' if first is not None else 'built'} its module "
                  f"({len(self.binds)} tensors bound, not rebindable: {fixed})", flush=True)

    def _own(self):
        """(DSV41_SHARE_MODULES) before a launch: when another layer holds the module, rebind every tensor of this layer."""
        if SHARE and getattr(self.sk, "_owner", None) is not self:
            for name, ptr in self.binds.items():
                self.sk.rebind(name, ptr)
            self.sk._owner = self

    def bind_streams(self, R: torch.Tensor):
        assert R.is_contiguous() and R.dtype == torch.bfloat16 and R.shape == (T, HC, HID), f"streams {R.shape} {R.dtype}"
        ptr = R.data_ptr()
        if self.binds.get("streams") != ptr:
            self._rb("streams", ptr)

    def triplet(self):
        """(pre [T, HC], post [T, HC], comb [T, HC, HC]) fp32 of this launch, what deepseek_v4_mhc's stats kernels return."""
        w = self.w
        return w["h_pre"], w["h_post"], w["h_comb"].view(T, HC, HC)

    def hc_state(self, next, mhc):
        """deepseek_v4_mhc.HcState of the sublayer's end: R' (with the next layer's Engram in the kernel: R''), pre and, when the
        kernel made it and the next sublayer takes it, its input."""
        w = self.w
        return mhc.HcState(w["streams_eg" if "streams_eg" in w else "streams_out"].view(T, HC, HID), w["h_pre"],
                           mhc.HcNormed(w["x_out"]) if ("gamma_next" in w and next is not None) or "eg_gamma" in w else None)
        # (eg_gamma, DSV41_EG_X: the Engram layer's attention input from R'', whatever next is: SGLang's post before an
        # Engram layer passes next=None, as it computes no input there)


# H1 stage A (DSV41_X_FLAG=1, docs/h1_design.md; serving builds with -DSTATIC_NO_START_BARRIER, mHC in the kernels): every
# kernel is built with -DSTATIC_X_FLAG; an attention layer's post stores its launch number into its own 32 words after x
# (G::xflag_out), and a MoE layer whose x is that attention's x_out polls them instead of waiting for the attention's grid
# (G::xflag_in). DSV41_X_FLAG_BIND=0: the same builds, the flags left unbound (the classic path: the same-binary arm).
# DSV41_X_FLAG=1: the MoE's whole front waits for the flag (pure H1); 2: only x's readers do (+ the front's weights early)
XFLAG = os.environ.get("DSV41_X_FLAG", "0") != "0"
XFLAG_PREFILL = os.environ.get("DSV41_X_FLAG", "0") == "2"
# stage B (DSV41_X_FLAG_ATTN=1, with DSV41_X_FLAG): also the attention after a MoE layer: the MoE's post (when it computes the
# next attention's input) stores its words, the attention whose x is that x_out polls them
XFLAG_ATTN = XFLAG and os.environ.get("DSV41_X_FLAG_ATTN", "0") != "0"
# DSV41_X_FLAG_ATTN_BIND=0: stage B's builds (the attention's modes in its kernel), the attention left unbound (it waits
# for the MoE's grid as before): stage A alone in the binary of stage A + B, for a same-binary arm
XFLAG_ATTN_BIND = os.environ.get("DSV41_X_FLAG_ATTN_BIND", "1") != "0"
XFLAG_BIND = os.environ.get("DSV41_X_FLAG_BIND", "1") != "0"
_XFLAG = {}   # an attention layer's x_out address -> its 32 flag words' address


def _bind_xflag_out(layer):
    """H1: this attention layer's post words (zeroed, owned by the layer) and its x_out's entry in _XFLAG."""
    w = layer.w
    w["xflag"] = torch.zeros(32, dtype=torch.int32, device=w["x_out"].device)
    if XFLAG_BIND:
        layer._rb("@xflag_out", w["xflag"].data_ptr())
        _XFLAG[w["x_out"].data_ptr()] = w["xflag"].data_ptr()


def _bind_xflag_in(layer):
    """H1: a MoE layer reading an attention layer's x_out in place polls that layer's words; else the classic wait (0)."""
    p = _XFLAG.get(layer.binds.get("x"), 0)
    if layer.binds.get("@xflag_in") != p:
        layer._rb("@xflag_in", p)
        if os.environ.get("DSV41_X_FLAG_LOG"):   # (a check: which layers poll a flag)
            print(f"[dsv41_static] H1: {type(layer).__name__} L{layer.layer}: "
                  f"{'polls the flag of the kernel before' if p else 'no flag (the wait for the grid)'}", flush=True)


def _bind_x(layer, x: torch.Tensor):
    """Graph tensor x reads SGLang's x when it is contiguous bf16 [T, H] (a CUDA graph captures the address), else a copy."""
    ok = x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape == (T, HID)
    ptr = x.data_ptr() if ok else layer.w["x"].data_ptr()
    if layer.binds.get("x") != ptr:
        layer._rb("x", ptr)
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
        """SGLang's metadata core (DSV4AttnMetadata) of this step. DSV41_FILL_FUSED=1 (11-20 night): in one kernel
        (fill_tables.py), else as torch ops (6 or more small kernels)."""
        if os.environ.get("DSV41_FILL_FUSED", "0") != "0":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import fill_tables
            fill_tables.fill(self, core, positions, req_pool_indices, T)
            if os.environ.get("DSV41_FILL_CHECK", "0") != "0" and not torch.cuda.is_current_stream_capturing():
                got = {k: v.clone() for k, v in self.t.items()}
                gp = {r: p.clone() for r, p in self.pages.items()}
                self._fill_torch(core, positions, req_pool_indices)
                ok = all(torch.equal(got[k], self.t[k]) for k in got) and all(torch.equal(gp[r], self.pages[r]) for r in gp)
                print(f"[dsv41_static] DSV41_FILL_CHECK: the fused fill equals the torch fill: {ok}", flush=True)
                assert ok, "DSV41_FILL_FUSED: the fused fill differs"
            return
        self._fill_torch(core, positions, req_pool_indices)

    def _fill_torch(self, core, positions, req_pool_indices):
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


class StaticAttn(MhcMixin):
    def __init__(self, layer: int, ckpt: str, rank: int, tp: int, dev, out_root: str, max_pos: int, sources: Sources, pdl: bool,
                 exchange_pool: ExchangePool = None, tables: StepTables = None, mhc: bool = False, stats: bool = False,
                 engram_kv: tuple = None):
        """engram_kv (11-20, DSV41_ENGRAM=2): (kv view, multicast address[, Engram layer, default the next[, rows]]) of
        engram_kv_buffer: this kernel also computes that layer's Engram kv into it (moe_demo.engram_kv_nodes), for the gate in
        the MoE kernel before that layer. rows (DSV41_EG_ROWS=1): {emb: engram_emb_buffer's view, emb_mc: its multicast
        address, and engram_rows_tables' tensors}: the n-gram rows looked up in this kernel too (run's ids each step)."""
        cfg = sources.cfg
        self.layer, self.pdl, self.steps, self.filled, self.pool, self.binds = layer, pdl, 0, 0, False, {}
        self.variant = attn_weights.variant_of(cfg, layer)
        ratio = cfg["compress_ratios"][layer]
        self.ratio = ratio
        csrc = layer == cfg["candidate_source_layer"]
        rows = (max_pos + 2 + 255) // 256 * 256
        cap = max(((max_pos + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
        w = {**attn_weights.layer_tensors(reader(ckpt), cfg, layer, tp, rank, rows, dev),
             **attn_demo.state_tensors(self.variant, ratio, csrc, cap, dev), **sources.shared(layer, self.variant)}
        self.engram = engram_kv is not None
        if self.engram:
            w.update(engram_kv_tensors(reader(ckpt), engram_kv[2] if len(engram_kv) > 2 else layer + 1, tp, rank, dev,
                                       engram_kv[0]))
            eg_rows = engram_kv[3] if len(engram_kv) > 3 else None
            if eg_rows is not None:   # the lookup's tensors (eg_ids, eg_emb_mc: stand-ins, rebound below and in run)
                w.update(eg_emb=eg_rows["emb"], eg_emb_mc=torch.zeros_like(eg_rows["emb"]),
                         eg_ids=torch.zeros(T, 24, dtype=torch.int64, device=dev),
                         **{k: eg_rows[k] for k in ("eg_tw", "eg_ts", "eg_starts")})
        if tables is not None:
            shared = {k: v for k, v in tables.t.items() if k in w}
            w.update(shared)
            if "compress_pages" in w:
                w["compress_pages"] = tables.pages[ratio]
        if mhc or stats:
            w.update(mhc_tensors(reader(ckpt), layer, "attn", f"layers.{layer}.ffn_norm.weight" if mhc else None, dev, stats))
            os.environ["HC_DRY"] = os.environ.get("HC_DRY_ATTN", "0")
        out = os.path.join(out_root, f"rank{rank}", f"attn{layer}")
        flags = ["-DSTATIC_RESET_IN_KERNEL", f"-DSTATIC_PDL_TRIGGER={int(pdl)}"] + os.environ.get("DSV41_STATIC_FLAGS", "").split()
        if PROF_SHARE:
            flags.append("-DSTATIC_STAMP_REBIND")
        if XFLAG:
            assert mhc and "-DSTATIC_NO_START_BARRIER" in flags, "DSV41_X_FLAG: mHC in the kernels, a serving build"
            flags.append("-DSTATIC_X_FLAG")
        # the index layers' scores by tcgen05 MMAs (tasks/indexer_score.cuh IS_TCGEN05, from mirage-opt: the same y bits, 1.4 to
        # 4.6 us less a layer); DSV41_STATIC_IS_TCGEN05=1 turns it on here (default off: this tree's measured builds)
        if os.environ.get("DSV41_STATIC_IS_TCGEN05", "0") != "0" and self.variant not in ("swa", "reuse"):
            flags.append("-DIS_TCGEN05")
        # I1c (11-19; default off): DSV41_I1C=1 builds the index layers' attention kernels (DSV41_I1C_VARIANTS) with
        # -DI1C_HCP: the post's R and gamma prefetched into L1 instead of held in registers across the all-reduce wait
        # (tasks/hc_post_combine.cuh; the same values). DSV41_I1C=2 adds -DI1C_SA for full_r2 and full_r1: sparse_attn's
        # split combine staged in shared memory one split at a time (fewer spills; its polled loads one after the other).
        # Spill stores / loads in B, TP4 full_r2 timing build 668 / 1136 -> HCP 640 / 1092, both 368 / 604; reindex 20 / 24 ->
        # HCP 4 / 4 (SA would make it 44 / 48); a reuse layer's kernel has none (SA would add 24): not there
        i1c = os.environ.get("DSV41_I1C", "0")
        if i1c != "0" and self.variant in I1C_VARIANTS:
            flags.append("-DI1C_HCP")
            if i1c == "2" and self.variant in ("full_r2", "full_r1"):
                flags.append("-DI1C_SA")
        # the node buffers are torch tensors the kernel writes to: they must stay referenced as long as the layer runs
        rk = dict(num_gpus=tp, rank=rank, exchange_pool=exchange_pool) if exchange_pool is not None else {}
        # C16 (11-18; default off): DSV41_SA_DRY_IDX=1 turns sparse_attn's dry pass on for the index layers (full_r2,
        # full_r1, reindex), whatever SA_DRY says. Their inputs (the indexer's top-k) come late, so the dry pass runs before
        # them and the real pass runs warm. TP4 shards on 2 GPUs with the mHC nodes (local/sa_mhc.sh), the layer's span
        # without -> with: full_r2 69.3 -> 64.8, full_r1 68.6 -> 67.6, reindex 64.6 -> 61.0 us. A reuse layer's inputs come
        # early and there the dry pass sits on the path (40.7 -> 44.2); swa gains nothing (39.0 -> 39.2): the arms' SA_DRY=0
        # stays for those
        sa_dry = os.environ.get("SA_DRY")
        if os.environ.get("DSV41_SA_DRY_IDX", "0") != "0" and self.variant in ("full_r2", "full_r1", "reindex"):
            os.environ["SA_DRY"] = "1"
        try:
            xk = dict(x_flag=True, x_flag_prefill=XFLAG_PREFILL) if XFLAG_ATTN else {}   # (stage B: the attention reads)
            self.sk, self.bufs, self.info, self.shared = _build_shared(attn_demo.build, [w], self.variant, ratio, csrc, 64 // tp,
                                                                       out, extra_flags=flags, device=dev.index, profile=TIMING,
                                                                       **rk, **xk)
        finally:
            if sa_dry is None:
                os.environ.pop("SA_DRY", None)
            else:
                os.environ["SA_DRY"] = sa_dry
        self.w = w
        if SHARE:
            self._init_binds(w, self.shared)
        if self.engram:   # the gathered kv's multicast address (the stand-in's slot)
            self._rb("eg_kv_mc", engram_kv[1])
            if "eg_tw" in w:
                self._rb("eg_emb_mc", engram_kv[3]["emb_mc"])
        sources.note(layer, w)
        if XFLAG:
            _bind_xflag_out(self)

    def _launch(self):
        self._own()
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)

    def bind_pools(self, pool):
        """The caches are SGLang's: its pools' rows have the kernels' bytes and stride (the module comment)."""
        L, v = self.layer, self.variant
        assert pool.kv_layout.value == "v41_new", f"{pool.kv_layout}: run SGLang with SGLANG_DSV4_KV_LAYOUT=v41_new"
        self._rb("window_cache", pool.get_swa_raw_buffer(L).data_ptr())
        if v != "swa":
            assert pool.get_extra_key_layout(L).value == "v41_fp4_new" and pool.get_extra_key_bytes_per_token(L) == 288
            self._rb("compress_cache", pool.get_extra_key_buffer(L).data_ptr())
        if v in ("full_r2", "full_r1", "reindex"):
            assert pool.get_index_k_page_size(self.ratio) == 128 and not pool.low_ratio_index_k_is_split(L)
            self._rb("index_k", pool.get_index_k_with_scale_buffer(L).data_ptr())
        if "cmp_state" in self.w:
            state = pool.get_attention_compress_states(L)
            assert state.ring_size == self.ratio == 2 and state.kv_score_buffer.kv_score.shape[1] == 1024
            self._rb("cmp_state", state.kv_score_buffer.kv_score.data_ptr())
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

    def run(self, x: torch.Tensor, positions: torch.Tensor, R: torch.Tensor = None, emb: torch.Tensor = None,
            ids: torch.Tensor = None) -> torch.Tensor:
        """One decode step of T rows: this rank's y [T, H] bf16 (the layer's own buffer), or with the exchange the sum. With the
        pools bound the step's tables are filled (StepTables) and x is read where SGLang left it: no copy kernel."""
        if self.pool:
            _bind_x(self, x)
            if XFLAG_ATTN and XFLAG_ATTN_BIND:
                _bind_xflag_in(self)
        else:
            self.w["x"].copy_(x)
            self.w["positions"].copy_(positions.view(-1))
            attn_demo.window_tables(self.w)
        if R is not None:
            self.bind_streams(R)
        if emb is not None:   # (DSV41_ENGRAM=2) the next layer's n-gram rows, [T, 6144] bf16 (SGLang's gather)
            assert emb.is_contiguous() and emb.dtype == torch.bfloat16 and tuple(emb.shape) == tuple(self.w["eg_emb"].shape)
            self.eg_emb_ref = emb
            if self.binds.get("eg_emb") != emb.data_ptr():
                self._rb("eg_emb", emb.data_ptr())
        if ids is not None:   # (DSV41_EG_ROWS=1) the Engram layer's hash ids [T, 24] int64, rows moe_demo.EG_ROWS_STRIDE apart
            from deepseek_v4_1_moe import demo as moe_demo
            assert ids.dtype == torch.int64 and tuple(ids.shape) == (T, 24) and ids.stride() == (moe_demo.EG_ROWS_STRIDE, 1)
            self.eg_ids_ref = ids
            if self.binds.get("eg_ids") != ids.data_ptr():
                self._rb("eg_ids", ids.data_ptr())
        self._launch()
        self.steps += 1
        return self.w["y"]


def engram_kv_buffer(tp: int, dev, group):
    """(11-20, DSV41_ENGRAM=2) one Engram layer's gathered kv: torch symmetric memory over the TP group, [tp T, W] bf16 (GPU
    r's W = 25600 / tp columns at rows [r T, r T + T)), 0xFF-filled (the gate polls it and re-arms it): (its local view, its
    rendezvous handle and buffer (keep them referenced), its multicast address). Collective: every rank in the same order."""
    W = (HC + 1) * HID // tp
    buf, handle, mc = _exchange_region(tp * T * W * 2, dev, group)
    buf.fill_(0xFF)
    return buf.view(torch.bfloat16).view(tp * T, W), (buf, handle), mc


def engram_emb_buffer(tp: int, dev, group):
    """(11-20, DSV41_EG_ROWS=1) one Engram layer's n-gram rows looked up in the kernel: torch symmetric memory over the TP
    group, [T, 6144] bf16, 0xFF-filled (engram_rows writes it on every GPU, the act_quant polls and re-arms it): (its local
    view, its keep-alive, its multicast address). Collective: every rank in the same order."""
    n = T * 6144 * 2
    buf, handle, mc = _exchange_region(n, dev, group)
    buf.fill_(0xFF)
    return buf.view(torch.bfloat16).view(T, 6144), (buf, handle), mc


def engram_rows_tables(module, tp: int, dev, group) -> dict:
    """(DSV41_EG_ROWS=1) SGLang's table of an Engram layer as engram_rows reads it. SGLang's PEER_READ layout (its default on
    one NVLink domain): its own mappings of every rank's shard (w_bias, s_bias, starts [P]: engram_gather_peer's), peer=True:
    every rank reads every row in place. The plain sharded layout: this rank's fp8 rows and e8m0 scales (its
    EngramEmbedding's parameters) and every rank's first row (all-gathered over the TP group: collective)."""
    import torch.distributed as dist
    e = module.embed
    if e.peer is not None:
        assert e.peer.ready and e.peer.w_bias.dtype == torch.int64 and e.peer.starts.numel() == e.peer.w_bias.numel()
        return {"eg_tw": e.peer.w_bias, "eg_ts": e.peer.s_bias, "eg_starts": e.peer.starts, "peer": True}
    assert e.host_table is None and not getattr(e, "_shared", False), \
        "DSV41_EG_ROWS: the Engram table in device memory (PEER_READ or sharded)"
    assert e.weight.dim() == 2 and e.scale.dim() == 2 and e.weight.shape[1] == e.dim and e.scale.shape[1] * 32 == e.dim
    lohi = torch.tensor([e.row_start, e.row_start + e.rows], dtype=torch.int64, device=dev)
    parts = [torch.zeros_like(lohi) for _ in range(tp)]
    dist.all_gather(parts, lohi, group=group)
    parts = [p.tolist() for p in parts]
    assert all(parts[r][1] == parts[r + 1][0] for r in range(tp - 1)), f"Engram table shards {parts}"
    starts = [p[0] for p in parts] + [parts[-1][1]]
    return {"eg_tw": e.weight.data.view(torch.uint8), "eg_ts": e.scale.data.view(torch.uint8),
            "eg_starts": torch.tensor(starts, dtype=torch.int64, device=dev), "peer": False}


def engram_kv_tensors(read, layer: int, tp: int, rank: int, dev, kv: torch.Tensor) -> dict:
    """(DSV41_ENGRAM=2) moe_demo.engram_kv_nodes' tensors in the attention kernel before layer `layer`'s Engram: the wkv
    shard (as engram_tensors), the n-gram rows' buffer, kv (engram_kv_buffer's view) and eg_kv_mc (a stand-in of kv's shape:
    the layer rebinds it to kv's multicast address)."""
    w = {k: v for k, v in engram_tensors(read, layer, tp, rank, dev).items() if k in ("eg_wkv", "eg_wkv_scales", "eg_emb")}
    w.update(eg_kv=kv, eg_kv_mc=torch.zeros_like(kv))
    return w


def engram_tensors(read, layer: int, tp: int, rank: int, dev) -> dict:
    """(11-20) The tensors of moe_demo.engram_nodes for the Engram of layer `layer` (inference/model.py Engram), GPU rank of tp:
    the wkv weight's rows [r W, r W + W) of 25600 (W = 25600 / tp; e4m3 bytes, its 32 x 32 block scales as gemm_fp8_tile's
    chunks), q_weight x k_weight in fp32 ([HC, dim], as the official multiplies them), the gathered n-gram rows' buffer
    (eg_emb, bound to SGLang's gather output each step) and R'' (streams_eg)."""
    from mirage.mpk.static_megakernel import w8_scale_chunks
    p = f"layers.{layer}.engram."
    wkv, sc = read(p + "wkv.weight"), read(p + "wkv.scale")
    n = wkv.shape[0] // tp
    assert wkv.shape[0] == (HC + 1) * HID and n % 128 == 0, f"Engram wkv {tuple(wkv.shape)}"
    rows = slice(rank * n, rank * n + n)
    w = {"eg_wkv": wkv[rows].contiguous().view(torch.uint8),
         "eg_wkv_scales": w8_scale_chunks(sc[rank * n // 32:(rank * n + n) // 32].contiguous()),
         "eg_qk": read(p + "q_weight").float() * read(p + "k_weight").float(),
         "eg_emb": torch.zeros(T, wkv.shape[1], dtype=torch.bfloat16),
         "streams_eg": torch.zeros(T, HC * HID, dtype=torch.bfloat16)}
    return {k: v.to(dev).contiguous() for k, v in w.items()}


class StaticMoE(MhcMixin):
    def __init__(self, layer: int, ckpt: str, rank: int, tp: int, dev, out_root: str, pdl: bool,
                 exchange_pool: ExchangePool = None, mhc: bool = False, norm_next: bool = False, stats: bool = False,
                 engram: bool = False, gamma_name: str = None, engram_kv: torch.Tensor = None, head: bool = False):
        """engram (11-20, DSV41_ENGRAM=1, with mHC and no next input): the next layer's Engram in this kernel after the post
        (moe_demo.engram_nodes); SGLang then takes R'' (hc_state) and skips its own (dsv41_static_hook)."""
        self.layer, self.pdl, self.steps, self.binds = layer, pdl, 0, {}
        read = reader(ckpt)
        w = moe_weights.layer_tensors(read, layer, tp, rank, dev)
        w.update(moe_weights.routed_tensors(read, layer, tp, [rank], dev)[0])
        w["x"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
        w["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
        if mhc or stats:
            gn = gamma_name or f"layers.{layer + 1}.attn_norm.weight"   # (the last layer with the head: norm.weight)
            w.update(mhc_tensors(read, layer, "ffn", gn if norm_next and mhc else None, dev, stats))
            os.environ["HC_DRY"] = os.environ.get("HC_DRY_MOE", "1")
        self.engram = engram
        if engram:
            assert mhc and not norm_next and not stats, "the Engram nodes: mHC's post in the kernel, no next input"
            eg = engram_tensors(read, layer + 1, tp, rank, dev)
            if engram_kv is not None:   # (DSV41_ENGRAM=2) kv from the attention kernel before: the gate only
                eg = {"eg_qk": eg["eg_qk"], "streams_eg": eg["streams_eg"], "eg_kv": engram_kv}
                if os.environ.get("DSV41_EG_X", "0") != "0":   # (11-20) and the Engram layer's attention input from R''
                    eg["eg_gamma"] = read(f"layers.{layer + 1}.attn_norm.weight").bfloat16().to(dev).contiguous()
            w.update(eg)
        self.head = head
        if head:   # (11-20, DSV41_HEAD=2) the LM head in this kernel (moe_demo.head_nodes), after the post's next input
            head_demo = _head_demo()
            assert mhc and norm_next, "the head in the MoE kernel: mHC's post computes its input (norm.weight)"
            hw = head_demo.head_tensors(read, tp, rank, dev)
            w.update({k: hw[k] for k in ("head_w", "logits", "head_ids") if k in hw})
        out = os.path.join(out_root, f"rank{rank}", f"moe{layer}")
        flags = ["-DSTATIC_RESET_IN_KERNEL", f"-DSTATIC_PDL_TRIGGER={int(pdl)}"] + os.environ.get("DSV41_STATIC_FLAGS", "").split()
        if engram or head:   # the gate / the head's GEMM waits on the post's counter: a release (tasks/engram_gate.cuh)
            fast = "-DHCP_FAST_END" in flags
            flags = [f for f in flags if f != "-DHCP_FAST_END"]
            if fast and os.environ.get("DSV41_FAST_REARM", "0") != "0":   # (11-20) its early re-arm only
                flags.append("-DHCP_FAST_REARM")
        if head:   # its GEMM tasks behind the experts prefill their weights before the wait; the post's x (generic stores)
            # before the GEMM's TMA reads (tasks/gemm_tile.cuh)
            flags += ["-DGEMM_RELAID_PREFILL", "-DGEMM_ACT_PROXY_FENCE"]
        if PROF_SHARE:
            flags.append("-DSTATIC_STAMP_REBIND")
        if XFLAG:
            assert mhc and "-DSTATIC_NO_START_BARRIER" in flags, "DSV41_X_FLAG: mHC in the kernels, a serving build"
            flags.append("-DSTATIC_X_FLAG")
        rk = dict(num_gpus=tp, rank=rank, exchange_pool=exchange_pool) if exchange_pool is not None else {}
        self.sk, self.bufs, self.info, self.shared = _build_shared(moe_demo.build, [w], out, extra_flags=flags, device=dev.index,
                                                                   profile=TIMING, x_flag=XFLAG, x_flag_prefill=XFLAG_PREFILL,
                                                                   **rk)   # (bufs: as StaticAttn)
        self.w = w
        if SHARE:
            self._init_binds(w, self.shared)
        if XFLAG_ATTN and "gamma_next" in w:   # stage B: this post computes the next attention's input
            _bind_xflag_out(self)

    def run(self, x: torch.Tensor, R: torch.Tensor = None, emb: torch.Tensor = None) -> torch.Tensor:
        """One decode step of T rows: this rank's part of y [T, H] bf16 (the layer's own buffer), or with the exchange the sum."""
        _bind_x(self, x)
        if XFLAG:
            _bind_xflag_in(self)
        if R is not None:
            self.bind_streams(R)
        if emb is not None:   # (Engram) this step's n-gram rows of the next layer, [T, 6144] bf16 (SGLang's gather)
            assert emb.is_contiguous() and emb.dtype == torch.bfloat16 and tuple(emb.shape) == tuple(self.w["eg_emb"].shape)
            self.eg_emb_ref = emb   # (kept alive until the next step's)
            if self.binds.get("eg_emb") != emb.data_ptr():
                self._rb("eg_emb", emb.data_ptr())
        self._own()
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        self.steps += 1
        return self.w["y"]


class StaticHead:
    """(11-20, DSV41_HEAD=1) the LM head of a decode step as one static kernel (head_demo.py): x (the last MoE kernel's next
    input: the final norm of the streams' collapse) -> this GPU's vocab rows of head.weight (bf16 GEMM) -> every GPU's row
    gathered -> the logits [T, 129280] fp32 (SGLang's next_token_logits_buffer when it has one)."""

    def __init__(self, ckpt: str, rank: int, tp: int, dev, out_root: str, pdl: bool, exchange_pool: ExchangePool):
        head_demo = _head_demo()
        self.pdl, self.steps, self.binds = pdl, 0, {}
        self.w = head_demo.head_tensors(reader(ckpt), tp, rank, dev)
        flags = ["-DSTATIC_RESET_IN_KERNEL", f"-DSTATIC_PDL_TRIGGER={int(pdl)}"] + \
            [f for f in os.environ.get("DSV41_STATIC_FLAGS", "").split() if f.startswith("-DSTATIC_")]
        if PROF_SHARE:
            flags.append("-DSTATIC_STAMP_REBIND")
        out = os.path.join(out_root, f"rank{rank}", "head")
        self.sk, self.bufs, self.info = head_demo.build([self.w], out, extra_flags=flags, device=dev.index, num_gpus=tp,
                                                        rank=rank, exchange_pool=exchange_pool)
        for name in ("x", "logits", "head_w"):
            self.sk.rebind(name, self.w[name].data_ptr())
            self.binds[name] = self.w[name].data_ptr()

    def _bind(self, name: str, t: torch.Tensor):
        if self.binds.get(name) != t.data_ptr():
            self.sk.rebind(name, t.data_ptr())
            self.binds[name] = t.data_ptr()

    def run(self, x: torch.Tensor, logits: torch.Tensor = None) -> torch.Tensor:
        """x [T, 5120] bf16 (read where it is); logits: [T, vocab] fp32 to write (else the head's own buffer)."""
        assert x.is_contiguous() and x.dtype == torch.bfloat16 and tuple(x.shape) == tuple(self.w["x"].shape)
        out = self.w["logits"] if logits is None else logits
        assert out.is_contiguous() and out.dtype == torch.float32 and tuple(out.shape) == tuple(self.w["logits"].shape)
        self._bind("x", x)
        self._bind("logits", out)
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        self.steps += 1
        return out
