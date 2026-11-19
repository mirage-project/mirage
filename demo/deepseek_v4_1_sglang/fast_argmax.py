"""fast_argmax.py (S2-1, 11-18): torch.argmax(logits, -1) of the sampler's greedy path, the same result in two short kernels.

SGLang's greedy sampler (sglang/srt/layers/sampler.py) takes torch.argmax(logits, -1) over the gathered, softcapped logits
[batch, vocab] fp32; in the B300 TP4 node trace that is a reduce_kernel of 29.5 us a step at batch 8 (129280 columns a row).
Here: kernel 1 one program per (row, CHUNK columns) -> each chunk's maximum and its first index (and its first NaN), kernel
2 one program per row over the chunks. The result is torch.argmax's: the index of the first maximal value, a NaN counting
as the maximum (the first NaN's index), -0.0 == 0.0 (the first of them), a row of -inf -> 0. Anything else (not 2-D fp32
CUDA with unit column stride, keepdim, another dim) goes to torch.argmax itself.

install(module): the module's global `torch` becomes a proxy whose argmax is this one (every other attribute the real
torch's), so only that module's torch.argmax(x, -1) calls change."""
import torch
import triton
import triton.language as tl

CHUNK = 4096
_BIG = 2147483647


@triton.jit
def _chunk_argmax(x_ptr, val_ptr, idx_ptr, nan_ptr, ncols, row_stride, nchunks, CHUNK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    cols = ch * CHUNK + tl.arange(0, CHUNK)
    mask = cols < ncols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    isn = (x != x) & mask
    nan_first = tl.min(tl.where(isn, cols, 2147483647), axis=0)
    xv = tl.where(isn | (cols >= ncols), -float("inf"), x)
    v, i = tl.max(xv, axis=0, return_indices=True, return_indices_tie_break_left=True)
    o = row * nchunks + ch
    tl.store(val_ptr + o, v)
    tl.store(idx_ptr + o, ch * CHUNK + i)
    tl.store(nan_ptr + o, nan_first)


@triton.jit
def _rows_argmax(val_ptr, idx_ptr, nan_ptr, out_ptr, nchunks, NCH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, NCH)
    m = c < nchunks
    v = tl.load(val_ptr + row * nchunks + c, mask=m, other=-float("inf"))
    ix = tl.load(idx_ptr + row * nchunks + c, mask=m, other=0)
    nf = tl.load(nan_ptr + row * nchunks + c, mask=m, other=2147483647)
    nan_min = tl.min(nf, axis=0)
    _, bc = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
    best = tl.sum(tl.where(c == bc, ix, 0), axis=0)
    res = tl.where(nan_min < 2147483647, nan_min, best)
    tl.store(out_ptr + row, res.to(tl.int64))


@triton.jit
def _row_argmax(x_ptr, out_ptr, ncols, row_stride, CHUNK: tl.constexpr):
    """One program per row: the chunks in order, (max, first index) kept with a strict > across chunks, the first NaN."""
    row = tl.program_id(0).to(tl.int64)
    best_val = -float("inf")
    best_idx = 0
    nan_idx = 2147483647
    for start in range(0, ncols, CHUNK):
        cols = start + tl.arange(0, CHUNK)
        mask = cols < ncols
        x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
        isn = (x != x) & mask
        nan_idx = tl.minimum(nan_idx, tl.min(tl.where(isn, cols, 2147483647), axis=0))
        xv = tl.where(isn, -float("inf"), x)
        v, i = tl.max(xv, axis=0, return_indices=True, return_indices_tie_break_left=True)
        take = v > best_val
        best_idx = tl.where(take, start + i, best_idx)
        best_val = tl.where(take, v, best_val)
    res = tl.where(nan_idx < 2147483647, nan_idx, best_idx)
    tl.store(out_ptr + row, res.to(tl.int64))


_scratch = {}   # (device, rows, chunks): the two-stage path's partials (used inside one call only, stream-ordered)
_seen, _fell = [], []
MODE = ["two"]   # "two": the chunk kernel + the rows kernel (GPU 4.3 us at [8, 129280] on a B200, torch.argmax 25.0); "row": one
# kernel, a program per row (27.8 us: 8 programs)


def fast_argmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """torch.argmax(x, dim) for 2-D fp32 CUDA x with unit column stride and dim -1 / 1; else torch.argmax itself. The
    output is a new tensor every call (the caller may still be copying the last one)."""
    if not (x.is_cuda and x.dim() == 2 and dim in (-1, 1) and x.dtype == torch.float32 and x.stride(1) == 1
            and x.shape[0] > 0 and 0 < x.shape[1] < _BIG):
        if not _fell:   # (once a process: a call that went to torch.argmax)
            _fell.append(1)
            print(f"[fast_argmax] torch.argmax for {tuple(x.shape)} {x.dtype} cuda {x.is_cuda} stride {x.stride()}", flush=True)
        return torch.argmax(x, dim)
    rows, ncols = x.shape
    if not _seen:   # (once a process: the fast path is the one taken)
        _seen.append(1)
        print(f"[fast_argmax] first call: {tuple(x.shape)} {x.dtype} mode {MODE[0]}", flush=True)
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    if MODE[0] == "row":
        _row_argmax[(rows,)](x, out, ncols, x.stride(0), CHUNK=CHUNK, num_warps=8)
        return out
    nch = triton.cdiv(ncols, CHUNK)
    key = (x.device, rows, nch)
    if key not in _scratch:
        _scratch[key] = (torch.empty(rows * nch, dtype=torch.float32, device=x.device),
                         torch.empty(rows * nch, dtype=torch.int32, device=x.device),
                         torch.empty(rows * nch, dtype=torch.int32, device=x.device))
    val, idx, nan = _scratch[key]
    _chunk_argmax[(rows, nch)](x, val, idx, nan, ncols, x.stride(0), nch, CHUNK=CHUNK, num_warps=8)
    _rows_argmax[(rows,)](val, idx, nan, out, nch, NCH=max(16, triton.next_power_of_2(nch)), num_warps=1)
    return out


class _TorchProxy:
    """The real torch, but argmax(x, -1) / argmax(x, dim=1) (no keepdim) is fast_argmax."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def argmax(self, input, dim=None, keepdim=False, **kw):
        if dim in (-1, 1) and not keepdim and not kw:
            return fast_argmax(input, dim)
        return self._real.argmax(input, dim=dim, keepdim=keepdim, **kw)


def install(module) -> None:
    """module.torch -> the proxy (idempotent)."""
    if not isinstance(getattr(module, "torch", None), _TorchProxy):
        module.torch = _TorchProxy(module.torch)
