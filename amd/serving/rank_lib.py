"""A generated layer kernel's per-rank library (amd/tools/gen_rank_lib.py, include/mirage/static_megakernel_hip/rank_host.cuh)
in a process of a tensor-parallel group, one GPU per process (as SGLang runs it); the AMD counterpart of B300's
mpk_kernels.layer.LayerKernel.

    k = RankKernel(lib_path, group)            # once per process: the library, the exchange region (HIP IPC over `group`)
    layer = k.create_layer({name: tensor})     # per model layer: this rank's weights (k.weight_names)
    layer.run({name: tensor}, stream)          # one launch; tensors by k.run_names (a HIP graph can capture it)
"""
import ctypes
from typing import Dict, List, Optional, Tuple

import torch

_vp = ctypes.c_void_p


class Exchange:
    """This rank's exchange region (allocated by the library) and every rank's, mapped in this process (HIP IPC handles
    exchanged over the group)."""

    def __init__(self, lib, nbytes: int, group):
        import torch.distributed as dist
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        p = _vp()
        r = lib.mpk_region_alloc(ctypes.c_ulonglong(nbytes), ctypes.byref(p))
        if r != 0:
            raise RuntimeError(f"mpk_region_alloc({nbytes}): HIP error {r}")
        h = (ctypes.c_ubyte * 64)()
        r = lib.mpk_region_handle(p, h)
        if r != 0:
            raise RuntimeError(f"mpk_region_handle: HIP error {r}")
        handles = [None] * self.world
        dist.all_gather_object(handles, bytes(h), group=group)
        self.ptrs: List[int] = []
        for q, hq in enumerate(handles):
            if q == self.rank:
                self.ptrs.append(p.value)
                continue
            pq = _vp()
            r = lib.mpk_region_open((ctypes.c_ubyte * 64).from_buffer_copy(hq), ctypes.byref(pq))
            if r != 0:
                raise RuntimeError(f"mpk_region_open (rank {q}'s region): HIP error {r}")
            self.ptrs.append(pq.value)
        dist.barrier(group=group)


class RankKernel:
    def __init__(self, lib_path: str, group, device: Optional[torch.device] = None):
        import torch.distributed as dist
        self.lib_path = lib_path
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        lib = ctypes.CDLL(lib_path)
        lib.mpk_sizes.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2 + [ctypes.POINTER(ctypes.c_ulonglong)]
        lib.mpk_region_alloc.argtypes = [ctypes.c_ulonglong, ctypes.POINTER(_vp)]
        lib.mpk_region_handle.argtypes = [_vp, ctypes.POINTER(ctypes.c_ubyte)]
        lib.mpk_region_open.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(_vp)]
        lib.mpk_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_vp)]
        lib.mpk_weight_name.restype = ctypes.c_char_p
        lib.mpk_weight_name.argtypes = [ctypes.c_int]
        lib.mpk_run_name.restype = ctypes.c_char_p
        lib.mpk_run_name.argtypes = [ctypes.c_int]
        lib.mpk_layer_create.restype = _vp
        lib.mpk_layer_create.argtypes = [ctypes.POINTER(_vp)]
        lib.mpk_run.argtypes = [_vp, _vp, ctypes.POINTER(_vp)]
        lib.mpk_span.argtypes = [_vp, ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong)]
        lib.mpk_stamps.argtypes = [_vp, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
        lib.mpk_tstamp.argtypes = [_vp, _vp]
        lib.mpk_test_align.argtypes = [_vp]
        if hasattr(lib, "mpk_stream_wait_ge"):   # (libraries built before 10-16 lack it)
            lib.mpk_stream_wait_ge.argtypes = [_vp, _vp, ctypes.c_uint, ctypes.c_int]
        self.weight_names: List[str] = [lib.mpk_weight_name(i).decode() for i in range(lib.mpk_num_weights())]
        self.run_names: List[str] = [lib.mpk_run_name(i).decode() for i in range(lib.mpk_num_run())]
        t_, h_, rg = ctypes.c_int(), ctypes.c_int(), ctypes.c_ulonglong()
        lib.mpk_sizes(t_, h_, rg)
        self.tokens, self.hidden = t_.value, h_.value
        with torch.cuda.device(self.device):
            self.exchange = Exchange(lib, rg.value, group)
            peers = (_vp * self.world)(*self.exchange.ptrs)
            if lib.mpk_init(self.rank, self.world, peers) != 0:
                raise RuntimeError(f"{lib_path}: mpk_init failed (rank {self.rank} of {self.world})")
        self.lib = lib

    def create_layer(self, weights: Dict[str, torch.Tensor]) -> "RankLayer":
        missing = [n for n in self.weight_names if n not in weights]
        if missing:
            raise ValueError(f"{self.lib_path}: weights missing: {missing}")
        ws = [weights[n] for n in self.weight_names]
        for n, w in zip(self.weight_names, ws):
            if not (w.is_cuda and w.is_contiguous() and w.device == self.device):
                raise ValueError(f"weight {n} must be contiguous on {self.device}")
        ctx = self.lib.mpk_layer_create((_vp * len(ws))(*[w.data_ptr() for w in ws]))
        if not ctx:
            raise RuntimeError("mpk_layer_create failed")
        return RankLayer(self, ctx, ws)

    def tstamp(self, out: torch.Tensor) -> None:
        """one wall-clock stamp (ns) into out (int64 [1] on this GPU), on the current stream (capturable)"""
        r = self.lib.mpk_tstamp(_vp(torch.cuda.current_stream().cuda_stream), _vp(out.data_ptr()))
        if r != 0:
            raise RuntimeError(f"mpk_tstamp failed ({r})")

    def stream_wait_ge(self, stream, addr: int, value: int, reset: bool = True) -> None:
        """`stream` waits until the 32-bit word at device address `addr` >= value (hipStreamWaitValue32; capturable), then
        (reset) sets it to 0 (a stream memset)"""
        r = self.lib.mpk_stream_wait_ge(_vp(stream.cuda_stream), _vp(addr), ctypes.c_uint(value), ctypes.c_int(int(reset)))
        if r != 0:
            raise RuntimeError(f"mpk_stream_wait_ge failed ({r})")

    def test_align(self) -> None:
        """tests only: the ranks line up on the GPU, on the current stream (capturable)"""
        r = self.lib.mpk_test_align(_vp(torch.cuda.current_stream().cuda_stream))
        if r != 0:
            raise RuntimeError(f"mpk_test_align failed ({r})")


class RankLayer:
    def __init__(self, kernel: RankKernel, ctx: int, weights):
        self.k, self.ctx, self.weights = kernel, ctx, weights   # the weights stay referenced (read by pointer)
        self._keep = None

    def run(self, tensors: Dict[str, torch.Tensor], stream: Optional[torch.cuda.Stream] = None) -> None:
        missing = [n for n in self.k.run_names if n not in tensors]
        if missing:
            raise ValueError(f"run tensors missing: {missing}")
        io = [tensors[n] for n in self.k.run_names]
        st = stream or torch.cuda.current_stream()
        r = self.k.lib.mpk_run(_vp(self.ctx), _vp(st.cuda_stream), (_vp * len(io))(*[t.data_ptr() for t in io]))
        if r != 0:
            raise RuntimeError(f"mpk_run failed ({r})")
        self._keep = io   # alive until the kernel has used them

    def stamps(self) -> torch.Tensor:
        """the last run's per-CU stamps (int64 [CUs, NSTAMP], wall clock ns; config.cuh STAMP_*: 0 start (the go), 3 end), synchronous"""
        buf = (ctypes.c_longlong * (256 * 8))()
        n = self.k.lib.mpk_stamps(_vp(self.ctx), buf, 256 * 8)
        if n <= 0:
            raise RuntimeError("mpk_stamps failed")
        return torch.tensor(list(buf)[:256 * n], dtype=torch.int64).view(256, n)

    def span_us(self) -> float:
        """the last run's span on this GPU (last CU past its start -> latest CU end), synchronous"""
        b, e = ctypes.c_longlong(), ctypes.c_longlong()
        self.k.lib.mpk_span(_vp(self.ctx), b, e)
        return (e.value - b.value) / 1e3
