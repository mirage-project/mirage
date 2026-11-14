"""GPT-OSS's routed experts as two kernels after a layer's static megakernel (which then ends at the gate):
include/mirage/static_megakernel/sm90_moe_split.cu, built per (T, H, I, E, TOPK) with nvcc and called through ctypes."""
import ctypes, hashlib, os, shutil, subprocess
import torch

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "include", "mirage", "static_megakernel", "sm90_moe_split.cu")


def sm90_moe_split_weights(blocks: torch.Tensor, rows: int) -> torch.Tensor:
    """sm90_moe_pack's blocks [..., rows x 68] in place for sm90_moe_split.cu's decode: each value byte's bits permuted
    (low-slot nibble to bits {5: sign, 2, 1: exponent, 0: mantissa}, high-slot to {7, 4, 3, 6}); a zero scale (an all-zero
    block: the pack's K padding) becomes 120 (the decode takes every scale in [108, 138])."""
    b = torch.arange(256, device=blocks.device)
    lo, hi = b & 15, b >> 4
    lut = (((lo >> 3) & 1) << 5 | ((lo >> 2) & 1) << 2 | ((lo >> 1) & 1) << 1 | (lo & 1) |
           ((hi >> 3) & 1) << 7 | ((hi >> 2) & 1) << 4 | ((hi >> 1) & 1) << 3 | (hi & 1) << 6).to(torch.uint8)
    v, sc = blocks[..., :rows * 64], blocks[..., rows * 64:]
    zero = sc == 0
    assert bool(((sc >= 108) & (sc <= 138) | zero).all()), "sm90_moe_split: E8M0 scales in [108, 138]"
    v.copy_(lut[v.long()])
    sc[zero] = 120
    return blocks


class Sm90MoeSplit:
    """The two kernels for T token rows; launch() after a layer's kernel, on its stream: programmatic dependent launches
    (W13 waits for the layer kernel, W2 for W13, the next layer's kernel for W2)."""

    def __init__(self, T, H, I, E, TOPK, h2: torch.Tensor, act: torch.Tensor):
        """h2 [T, H] f16: the gate's MoE input; act [E, T, I] f16: W13's output (both every layer's). The build goes to
        MIRAGE_STATIC_BUILD_CACHE (else ~/.cache/mirage), by the source's and the flags' hash."""
        flags = [f"-DSTATIC_TOKENS={T}", f"-DMOE_H={H}", f"-DMOE_I={I}", f"-DMOE_E={E}", f"-DMOE_TOPK={TOPK}"]
        out_dir = os.environ.get("MIRAGE_STATIC_BUILD_CACHE") or os.path.expanduser("~/.cache/mirage")
        os.makedirs(out_dir, exist_ok=True)
        key = hashlib.sha1((open(_SRC).read() + " ".join(flags)).encode()).hexdigest()[:16]
        so = os.path.join(out_dir, f"sm90_moe_split_{key}.so")
        if not os.path.exists(so):
            subprocess.check_call([shutil.which("nvcc"), "-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a",
                                   "-shared", "-Xcompiler=-fPIC", "-lcuda", _SRC, "-o", so + ".tmp"] + flags)
            os.replace(so + ".tmp", so)
        self.lib = ctypes.CDLL(so)
        self.lib.moe_ctx.restype = ctypes.c_void_p
        self.ctx = ctypes.c_void_p(self.lib.moe_ctx(ctypes.c_void_p(h2.data_ptr()), ctypes.c_void_p(act.data_ptr())))

    def launch(self, routing, w13, b13, act, w2, b2, eo, topk_w, x1, x_out, done, limit, alpha):
        """routing [E, T] i32 (token t's top-k slot + 1 for expert e, else 0), w13 / w2 (sm90_moe_split_weights), b13 [E,
        2 I] / b2 [E, H] bf16, eo [T, TOPK, H] bf16 (scratch), topk_w [T, TOPK] fp32, x1 [T, H] bf16 (the residual), x_out
        [T, H] bf16 = x1 + sum_k topk_w[t, k] eo[t, k] (the routed tokens' rows), done [T, H / 64] i32 zeros (left zero)."""
        P = lambda t: ctypes.c_void_p(t.data_ptr())
        rc = self.lib.moe_launch(self.ctx, P(routing), P(w13), P(b13), P(act), P(w2), P(b2), P(eo), P(topk_w), P(x1),
                                 P(x_out), P(done), ctypes.c_float(limit), ctypes.c_float(alpha),
                                 ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert rc == 0, "sm90_moe_split: launch failed"
