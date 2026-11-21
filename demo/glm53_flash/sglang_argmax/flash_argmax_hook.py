"""Greedy decode without gathering the logits (lm_argmax.cu next to this file): the GLM-5.3-Flash copy of Kimi K3's
mpk_kernels integrations/sglang_k3_argmax.py, self-contained (its own nvcc build cache and its own exchange region). Hooked
into SGLang 0.5.22 by sglang_patch_argmax.py (next to this file):
    r = maybe_defer(self, logits, logits_metadata); if r is not None: return r
        in LogitsProcessor._get_logits right after the lm_head GEMM (before logits.mul_(logit_scale) and the all-gather)
    ids = before_sample(self, logits_output, forward_batch, observer); if ids is not None: return ids
        in ModelRunner.sample, before the logits bias and the sampler

Decode step, bs 64, TP 4, vocab 154880 (38720 columns per rank). SGLang: all_gather [64][154880] bf16 -> copy to fp32 ->
(after the graph) torch.argmax. Here, in the graph: one kernel per step reduces this rank's [64][38720] shard, exchanges
64 x 4 x 2 keys and writes int64 ids [64] (the same ids as torch.argmax, tie -> lowest index, NaN -> first NaN); the logits
are not gathered. What the logits processor returns is a placeholder of the logits' shape (a broadcast view of one float)
whose address names the step's buffers. After the replay, ModelRunner.sample:
  every request greedy, no logprobs, no penalties / logit bias / grammar / custom processor / sampling masks / observer /
  watermark, NaN checks / sanitising and async asserts off, no token sync, the stock Sampler -> the ids (a copy) are the
  next tokens;
  otherwise -> the shard goes through SGLang's own mul_ / gatherer / buffer copy now (the logits SGLang would have had) and
  sampling proceeds unchanged.
Only plain decode (forward_mode DECODE) on TP > 1 without DP attention / attn-TP lm_head / lm_head all-to-all / softcapping /
fp32 lm_head.

Exchange: torch symmetric memory over the TP group (as SGLang's multimem all-gather allocates its buffer); the kernel stores
into every rank's copy through the peer pointers (no multicast address needed).
Build: nvcc (FLASH_ARGMAX_NVCC, else nvcc on PATH, else $CUDA_HOME/bin/nvcc, else /usr/local/cuda/bin/nvcc) into
FLASH_ARGMAX_CACHE (default ~/.cache/flash_argmax)/<md5 of lm_argmax.cu + flags + arch>/lib.so; a file lock makes the TP
ranks' first calls build once. `python flash_argmax_hook.py --build` builds it ahead (prints the library path).

Environment: FLASH_ARGMAX=1 turns it on (unset: the patched SGLang lines do not import this module).
Logs (rank 0, stderr -> the run's log): "[flash argmax] ..." init, every captured graph, the first step on the ids, the first
fallback per reason, and every FLASH_ARGMAX_LOG_EVERY (default 2000) sampled steps the fast / fallback counts.
"""
import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "lm_argmax.cu")
FLAGS = ["-O3", "-std=c++17"]
ENABLED = os.environ.get("FLASH_ARGMAX", "") == "1"
LOG_EVERY = int(os.environ.get("FLASH_ARGMAX_LOG_EVERY", "2000"))
_vp = ctypes.c_void_p


# ---- build / load / exchange (no SGLang import: argmax_check.py uses these on its own) ----

def _nvcc() -> str:
    for cand in (os.environ.get("FLASH_ARGMAX_NVCC"), shutil.which("nvcc"),
                 os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc") if os.environ.get("CUDA_HOME") else None,
                 "/usr/local/cuda/bin/nvcc"):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("flash argmax: nvcc not found (set FLASH_ARGMAX_NVCC or CUDA_HOME)")


def _arch(dev=None) -> str:
    major, minor = torch.cuda.get_device_capability(dev)
    return f"{major}{minor}{'a' if major >= 9 else ''}"   # B300: 103a


def build(dev=None, verbose: bool = False) -> str:
    """The path of lm_argmax.cu's library for dev's arch, built (nvcc) if not cached."""
    arch = _arch(dev)
    with open(SRC, "rb") as f:
        src = f.read()
    key = hashlib.md5(src + " ".join(FLAGS).encode() + arch.encode()).hexdigest()[:16]
    out_dir = os.path.join(os.path.expanduser(os.environ.get("FLASH_ARGMAX_CACHE", "~/.cache/flash_argmax")), key)
    lib = os.path.join(out_dir, "lib.so")
    if os.path.exists(lib):
        return lib
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if os.path.exists(lib):   # another rank built it while this one waited
                return lib
            tmp = lib + f".tmp{os.getpid()}"
            cmd = [_nvcc(), *FLAGS, "-gencode", f"arch=compute_{arch},code=sm_{arch}", "-shared", "-Xcompiler=-fPIC", SRC,
                   "-o", tmp]
            if verbose:
                print("flash argmax: building", " ".join(cmd), flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                with open(os.path.join(out_dir, "build.log"), "w") as f:
                    f.write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
                raise RuntimeError(f"flash argmax: nvcc failed (log {out_dir}/build.log):\n{p.stderr[-2000:]}")
            os.replace(tmp, lib)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return lib


def load(dev=None) -> ctypes.CDLL:
    """The built library with its entry points typed."""
    lib = ctypes.CDLL(build(dev))
    lib.flash_lm_argmax_slot_bytes.restype = ctypes.c_ulonglong
    lib.flash_lm_argmax_max_rows.restype = ctypes.c_int
    lib.flash_lm_argmax_max_ranks.restype = ctypes.c_int
    lib.flash_lm_argmax.restype = ctypes.c_int
    lib.flash_lm_argmax.argtypes = [_vp, _vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_float, ctypes.c_int, _vp, ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int,
                                    ctypes.c_int, _vp, _vp]
    return lib


class Exchange:
    """This rank's copy of the symmetric exchange region (buffer, kept alive with its handle) and every rank's copy mapped
    in this process (peers: a ctypes array for the launch)."""

    def __init__(self, buffer, handle, peer_ptrs):
        self.buffer, self.handle, self.peer_ptrs = buffer, handle, peer_ptrs
        self.peers = (ctypes.c_ulonglong * len(peer_ptrs))(*peer_ptrs)


def make_exchange(nbytes: int, group, dev) -> Exchange:
    """Collective over `group` (a NCCL process group; every rank the same nbytes): nbytes of torch symmetric memory,
    zero-filled, rendezvoused, every rank's copy mapped; returns after a barrier (every copy zero before any rank launches)."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    with torch.inference_mode(False), torch.no_grad():   # (SGLang's forward runs under inference mode)
        buf = symm_mem.empty(nbytes, dtype=torch.uint8, device=dev)
        buf.zero_()
    torch.cuda.synchronize(dev)
    h = symm_mem.rendezvous(buf, group=group)
    if h.rank != rank or h.world_size != world:
        raise RuntimeError(f"flash argmax: symmetric memory rank {h.rank} / {h.world_size}, group rank {rank} / {world}")
    peers = [int(p) for p in h.buffer_ptrs][:world]
    if len(peers) != world or not all(peers):
        raise RuntimeError(f"flash argmax: peer pointers {peers}")
    dist.barrier(group=group)
    return Exchange(buf, h, peers)


def launch(lib, x: torch.Tensor, rows: int, col0: int, vocab: int, scale, ex: Exchange, rank: int, world: int,
           cnt: torch.Tensor, ids: torch.Tensor) -> None:
    """One launch on the current stream: ids[:rows] = argmax over every rank's [rows][W] shard (x: this rank's, bf16,
    contiguous rows of W = x.shape[1] columns, global columns col0 ..); scale: None / 1.0 (none) or the logit scale."""
    use_scale = scale is not None and float(scale) != 1.0   # x * 1.0 rounded to bf16 is x (a NaN stays a NaN)
    w = x.shape[1]
    st = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.flash_lm_argmax(_vp(st), _vp(x.data_ptr()), rows, x.stride(0), w, col0, vocab,
                             float(scale) if use_scale else 1.0, int(use_scale), _vp(ex.buffer.data_ptr()), ex.peers, rank,
                             world, _vp(cnt.data_ptr()), _vp(ids.data_ptr()))
    if rc != 0:
        raise RuntimeError(f"flash argmax: launch failed (cudaError {rc}) rows {rows} width {w}")


# ---- the SGLang hook ----

class _Entry:
    __slots__ = ("ph", "ids", "local", "buf", "proc", "rows", "vocab", "captured")


class _State:
    lib = None; failed = False; rank = -1; world = 0; ex = None; cnt = None; maxr = 0
    by_ptr = {}; eager = {}; logged = set(); n_fast = 0; n_slow = 0; n_cap = 0; env_block = None; sampler_cls = None


S = _State()


def _log(msg):
    if S.rank <= 0:
        print(f"[flash argmax] {msg}", file=sys.stderr, flush=True)


def _log_once(key, msg):
    if key not in S.logged:
        S.logged.add(key)
        _log(msg)


def _init(dev):
    """Collective over the TP group (the decode graph warm-up: every rank, same order). False: SGLang's path for good."""
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        S.rank, S.world = tp.rank_in_group, tp.world_size
        lib = load(dev)
        if S.world > lib.flash_lm_argmax_max_ranks():
            raise RuntimeError(f"TP {S.world} > {lib.flash_lm_argmax_max_ranks()} ranks")
        S.maxr = lib.flash_lm_argmax_max_rows()
        S.ex = make_exchange(int(lib.flash_lm_argmax_slot_bytes()), tp.device_group, dev)
        S.cnt = torch.zeros(S.maxr, dtype=torch.int32, device=dev)
        torch.cuda.synchronize(dev)
        S.env_block = _env_block()
        from sglang.srt.layers import sampler as _smp
        S.sampler_cls = _smp.Sampler
        S.lib = lib
        _log(f"ready: rank {S.rank} / {S.world}, exchange {S.ex.buffer.numel()} B, rows <= {S.maxr}, lib {build(dev)}"
             f"{'' if S.env_block is None else ', every step falls back: ' + S.env_block}")
        return True
    except Exception as e:  # noqa
        S.failed = True
        _log(f"init FAILED ({type(e).__name__}: {e}): SGLang's path")
        return False


def _env_block():
    """The environment switches whose sampler behaviour the ids do not reproduce (NaN checks / sanitising, async asserts,
    token sync, aiter sampler)."""
    try:
        from sglang.srt.environ import envs
        for name in ("SGLANG_SANITIZE_NAN_LOGITS", "SGLANG_ENABLE_ASYNC_ASSERT", "SGLANG_ENABLE_NAN_LOGITS_CHECK"):
            if getattr(envs, name).get():
                return name
    except Exception as e:  # noqa
        return f"environ unreadable ({e})"
    from sglang.srt.layers import sampler as _smp
    if getattr(_smp, "SYNC_TOKEN_IDS_ACROSS_TP", False):
        return "SYNC_TOKEN_IDS_ACROSS_TP"
    if getattr(_smp, "_use_aiter", False):
        return "aiter greedy sampler"
    return None


def _why_not_proc(proc, logits, md):
    if not getattr(proc, "do_tensor_parallel_all_gather", False):
        return "no TP all-gather"
    if getattr(proc, "use_attn_tp_group", False) or getattr(proc, "do_tensor_parallel_all_gather_dp_attn", False):
        return "DP attention / attn-TP lm_head"
    if getattr(proc, "use_tp_lm_head_all_to_all", False):
        return "lm_head all-to-all"
    if getattr(proc, "final_logit_softcapping", None):
        return "softcapping"
    if getattr(md, "draft_extend_select_index", None) is not None:
        return "draft extend"
    if logits.dtype != torch.bfloat16 or logits.dim() != 2 or not logits.is_contiguous() or logits.data_ptr() % 16:
        return f"logits {logits.dtype} {tuple(logits.shape)} contiguous {logits.is_contiguous()}"
    return None


def maybe_defer(proc, logits, md):
    """LogitsProcessor._get_logits after the lm_head GEMM: None (SGLang's path) or the placeholder logits [rows][vocab]."""
    if not ENABLED or S.failed:
        return None
    fm = getattr(md, "forward_mode", None)
    if fm is None or not fm.is_decode():
        return None
    why = _why_not_proc(proc, logits, md)
    if why is not None:
        _log_once(("proc", why), f"SGLang's path: {why}")
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if S.lib is None:
        if capturing:
            _log_once("cap-uninit", "not initialised when the first graph was captured: SGLang's path")
            return None
        if not _init(logits.device):
            return None
    rows, w = logits.shape
    vocab = int(proc.vocab_size)
    if rows > S.maxr or w * S.world < vocab or vocab >= (1 << 18) or w % 16:
        _log_once(("shape", rows, w), f"SGLang's path: rows {rows} width {w} x {S.world} vocab {vocab}")
        return None
    dev = logits.device
    if capturing:   # every captured graph its own buffers (from the graph's pool, kept alive by S.by_ptr; empty: no fill
        # kernel in the graph; the kernel writes every row of ids)
        e = _Entry()
        e.ph = torch.empty(1, dtype=torch.float32, device=dev)
        e.ids = torch.empty(rows, dtype=torch.int64, device=dev)
        e.captured = True
        S.by_ptr[e.ph.data_ptr()] = e
        S.n_cap += 1
        _log(f"graph captured with the per-rank argmax: rows {rows} (graphs so far {S.n_cap})")
    else:           # eager decode steps of this row count share one entry (its tensors replaced every call)
        e = S.eager.get(rows)
        if e is None:
            e = _Entry()
            e.ph = torch.zeros(1, dtype=torch.float32, device=dev)
            e.ids = torch.zeros(rows, dtype=torch.int64, device=dev)
            e.captured = False
            S.eager[rows] = e
            S.by_ptr[e.ph.data_ptr()] = e
    e.local, e.buf, e.proc, e.rows, e.vocab = logits, getattr(md, "next_token_logits_buffer", None), proc, rows, vocab
    launch(S.lib, logits, rows, S.rank * w, vocab, proc.logit_scale, S.ex, S.rank, S.world, S.cnt, e.ids)
    return e.ph.expand(rows, vocab)


def _why_not_sample(mr, lo, fb, observer):
    if S.env_block is not None:
        return S.env_block
    if type(getattr(mr, "sampler", None)) is not S.sampler_cls:
        return f"sampler {type(getattr(mr, 'sampler', None)).__name__}"
    if not fb.forward_mode.is_decode():
        return "not decode"
    si = fb.sampling_info
    if si is None:
        return "no sampling_info"
    if not si.is_all_greedy:
        return "not all greedy"
    if fb.return_logprob:
        return "logprobs"
    if any(si.return_sampling_masks or []) or getattr(si, "sampling_mask_batch_indices", None) is not None:
        return "sampling masks"
    if si.has_custom_logit_processor:
        return "custom logit processor"
    if si.grammars:
        return "grammar"
    if getattr(si, "acc_additive_penalties", None) is not None or getattr(si, "acc_scaling_penalties", None) is not None:
        return "penalties"
    po = getattr(si, "penalizer_orchestrator", None)
    if po is not None and po.is_required:
        return "penalties"
    if getattr(si, "logit_bias", None) is not None:
        return "logit bias"
    if getattr(mr, "watermark_state", None) is not None and getattr(si, "has_watermark_candidates", False):
        return "watermark"
    if observer is not None and observer.is_active(si):
        return "sampling observer"
    return None


def _materialize(e, lo, n):
    """SGLang's _get_logits tail on this step's shard: the logits it would have returned (rows 0 .. n-1)."""
    proc = e.proc
    x = e.local[:n]
    if proc.logit_scale is not None:
        x = x * proc.logit_scale             # logits.mul_(logit_scale): the same bf16 functor, out of place
    full = proc._logits_gatherer(x.contiguous())
    buf = e.buf[:n] if e.buf is not None else None
    lo.next_token_logits = proc._copy_logits_to_buffer(full, types.SimpleNamespace(next_token_logits_buffer=buf),
                                                       use_buffer=buf is not None)


def before_sample(mr, lo, fb, observer):
    """ModelRunner.sample before the logits bias: int64 next-token ids [n], or None (SGLang samples; full logits restored)."""
    if not S.by_ptr:
        return None
    t = lo.next_token_logits
    if t is None:
        return None
    e = S.by_ptr.get(t.data_ptr())
    if e is None or t.stride() != (0, 0):
        if t.dim() == 2 and t.stride() == (0, 0) and t.numel() > 1:   # a placeholder this module did not make (a copy /
            # view of one): sampling it would give token 0 for every request
            raise RuntimeError(f"flash argmax: broadcast logits {tuple(t.shape)} of unknown address (unset FLASH_ARGMAX)")
        return None
    if getattr(lo, "beam", None) is not None:
        raise RuntimeError("flash argmax: beam search logits capture is not supported (unset FLASH_ARGMAX)")
    n = t.shape[0]
    why = _why_not_sample(mr, lo, fb, observer)
    if why is None:
        if S.n_fast == 0:
            _log(f"first step on the per-rank argmax ids: rows {n} of {e.rows} ({'graph' if e.captured else 'eager'})")
        S.n_fast += 1
        out = e.ids[:n].clone()   # the graph's ids buffer is rewritten by the next replay
    else:
        S.n_slow += 1
        _log_once(("sample", why), f"fallback to SGLang's sampling ({why}): logits gathered after the graph")
        _materialize(e, lo, n)
        out = None
    if (S.n_fast + S.n_slow) % LOG_EVERY == 0:
        _log(f"sampled steps: per-rank argmax {S.n_fast}, fallback {S.n_slow}")
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["--build"]:
        sys.exit("usage: python flash_argmax_hook.py --build   (builds lm_argmax.cu for GPU 0's arch, prints the library)")
    print("FLASH-ARGMAX-LIB", build(0, verbose=True), flush=True)
