"""GPT-OSS's routed experts as two kernels after a layer's static megakernel (which then ends at the gate):
include/mirage/static_megakernel/sm90_moe_split.cu, built per (T, H, I, E, TOPK) with nvcc and called through ctypes."""
import ctypes, hashlib, os, shutil, subprocess
import torch

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "include", "mirage", "static_megakernel", "sm90_moe_split.cu")
_LUT = None


def sm90_moe_split_weights(blocks: torch.Tensor, rows: int) -> torch.Tensor:
    """sm90_moe_pack's blocks [..., rows x 68] for sm90_moe_split.cu's decode: each value byte's bits permuted (low-slot
    nibble to bits {5: sign, 2, 1: exponent, 0: mantissa}, high-slot to {7, 4, 3, 6}); a zero scale (an all-zero
    block: the pack's K padding) becomes 120 (the decode takes every scale in [108, 138])."""
    global _LUT
    if _LUT is None or _LUT.device != blocks.device:
        b = torch.arange(256, device=blocks.device)
        lo, hi = b & 15, b >> 4
        _LUT = (((lo >> 3) & 1) << 5 | ((lo >> 2) & 1) << 2 | ((lo >> 1) & 1) << 1 | (lo & 1) |
                ((hi >> 3) & 1) << 7 | ((hi >> 2) & 1) << 4 | ((hi >> 1) & 1) << 3 | (hi & 1) << 6).to(torch.uint8)
    v, sc = blocks[..., :rows * 64], blocks[..., rows * 64:]
    zero = sc == 0
    assert bool(((sc >= 108) & (sc <= 138) | zero).all()), "sm90_moe_split: E8M0 scales in [108, 138]"
    v.copy_(_LUT[v.long()])
    sc[zero] = 120
    return blocks


class Sm90MoeSplit:
    """The two kernels for T token rows; launch() after a layer's kernel, on its stream: programmatic dependent launches
    (W13 waits for the layer kernel, W2 for W13, the next layer's kernel for W2)."""

    def __init__(self, T, H, I, E, TOPK, h2: torch.Tensor, act: torch.Tensor, out_dir: str = None):
        """h2 [T, H] f16: the gate's MoE input; act [E, T, I] f16: W13's output (both every layer's). out_dir: where the
        build goes (by the source's and flags' hash; default MIRAGE_STATIC_BUILD_CACHE, else ~/.cache/mirage)."""
        out_dir = out_dir or os.environ.get("MIRAGE_STATIC_BUILD_CACHE") or os.path.expanduser("~/.cache/mirage")
        flags = [f"-DTT={T}", f"-DHH={H}", f"-DII={I}", f"-DEE={E}", f"-DTOPKK={TOPK}"]
        src = open(_SRC).read()
        key = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()[:16]
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, f"sm90_moe_split_{key}.so")
        if not os.path.exists(so):
            major, minor = torch.cuda.get_device_capability(0)
            subprocess.check_call([shutil.which("nvcc"), "-O3", "-std=c++17", "-gencode",
                                   f"arch=compute_{major}{minor}a,code=sm_{major}{minor}a", "-shared", "-Xcompiler=-fPIC",
                                   "-lcuda", _SRC, "-o", so + ".tmp"] + flags)
            os.replace(so + ".tmp", so)
        self.lib = ctypes.CDLL(so)
        self.lib.moe_ctx.restype = ctypes.c_void_p
        self.keep = (h2, act)
        self.ctx = ctypes.c_void_p(self.lib.moe_ctx(ctypes.c_void_p(h2.data_ptr()), ctypes.c_void_p(act.data_ptr())))

    def launch(self, routing, w13, b13, act, w2, b2, eo, topk_w, x1, x_out, done, limit, alpha, stream=None):
        """routing [E, T] i32 (token t's top-k slot + 1 for expert e, else 0), w13 / w2 (sm90_moe_split_weights), b13 [E,
        2 I] / b2 [E, H] bf16, eo [T, TOPK, H] bf16 (scratch), topk_w [T, TOPK] fp32, x1 [T, H] bf16 (the residual), x_out
        [T, H] bf16 = x1 + sum_k topk_w[t, k] eo[t, k] (the routed tokens' rows), done [T, H / 64] i32 zeros (left zero)."""
        stream = stream or torch.cuda.current_stream()
        P = lambda t: ctypes.c_void_p(t.data_ptr())
        rc = self.lib.moe_launch(self.ctx, P(routing), P(w13), P(b13), P(act), P(w2), P(b2), P(eo), P(topk_w), P(x1),
                                 P(x_out), P(done), ctypes.c_float(limit), ctypes.c_float(alpha),
                                 ctypes.c_void_p(stream.cuda_stream))
        assert rc == 0, "sm90_moe_split: launch failed"
