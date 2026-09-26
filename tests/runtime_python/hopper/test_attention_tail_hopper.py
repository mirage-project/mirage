"""Compare Hopper attention with a reference after poisoning shared memory.

Run on a free Hopper GPU. Builds a standalone wrapper in a temporary directory;
no model weights are required. Covers prefill/decode and full/partial KV tiles.
"""

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch


def check_case(lib, capacity, active, context):
    qkv = torch.randn(capacity, 1280, device="cuda", dtype=torch.bfloat16) * 0.2
    kc = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    vc = torch.randn_like(kc)
    out = torch.full((capacity, 1024), float("nan"), device="cuda",
                     dtype=torch.bfloat16)
    k = torch.cat((kc[:context], qkv[:active, 1024:1152])).float()
    v = torch.cat((vc[:context], qkv[:active, 1152:])).float()
    q = qkv[:active, :1024].reshape(active, 8, 128).float()
    scores = torch.einsum("thd,sd->ths", q, k) / 128**0.5
    mask = (torch.arange(context + active, device="cuda")[None, :]
            <= (context + torch.arange(active, device="cuda"))[:, None])
    prob = scores.masked_fill(~mask[:, None, :], -float("inf")).softmax(-1)
    ref = torch.einsum("ths,sd->thd", prob, v).reshape(active, 1024)
    meta = [torch.tensor(a, device="cuda", dtype=torch.int32)
            for a in ([0, active], [0, 1], [0], [context + active])]
    torch.cuda.synchronize()
    rc = lib.launch(*[t.data_ptr() for t in [qkv, kc, vc, out, *meta]], capacity)
    assert rc == 0, f"CUDA error {rc}"
    finite = torch.isfinite(out[:active]).all().item()
    error = (out[:active].float() - ref).abs().max().item()
    ok = finite and error < 0.005
    # A decode task must leave the inactive output rows untouched.
    ok = ok and torch.isnan(out[active:]).all().item()
    print(f"capacity={capacity} active={active} context={context} "
          f"finite={finite} max_error={error:.6f} {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


def main():
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This regression test requires Hopper (sm90).")
    root = Path(__file__).resolve().parents[3]
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc is required to build the test wrapper.")
    with tempfile.TemporaryDirectory(prefix="mirage-attention-tail-") as build:
        library = Path(build) / "attention_tail.so"
        subprocess.run([
            nvcc, str(Path(__file__).with_name("attention_tail_wrapper.cu")),
            "-o", str(library), "-shared", "-Xcompiler", "-fPIC", "-std=c++20",
            "-O3", "-gencode", "arch=compute_90a,code=sm_90a",
            "-DMIRAGE_BACKEND_USE_CUDA", "-DMIRAGE_GRACE_HOPPER",
            "-DMPK_TARGET_CC=90", f"-I{root / 'include'}",
            f"-I{root / 'include/mirage/persistent_kernel'}",
            f"-I{root / 'deps/cutlass/include'}", "--expt-relaxed-constexpr",
            "-lcuda",
        ], check=True)
        lib = ctypes.CDLL(str(library))
        lib.launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int]
        lib.launch.restype = ctypes.c_int
        torch.manual_seed(740)
        results = [check_case(lib, capacity, active, context)
                   for capacity, active in [(1, 1), (8, 1), (8, 8)]
                   for context in [0, 39, 56, 63, 64, 65, 120, 128]]
    assert all(results), f"{results.count(False)}/{len(results)} cases failed"


if __name__ == "__main__":
    main()
