"""Regressions for Hopper attention's empty state, finite scores and KV splits."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[3]
POINTERS = ("qkv", "kc", "vc", "output", "partial", "lse", "q_scale",
            "k_scale", "cos", "sin", "qo", "pi", "pages", "last")


def build(root, library, finite_mask=False):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required")
    command = [
        nvcc, str(Path(__file__).with_name("attention_softmax_wrapper.cu")),
        "-o", str(library), "-shared", "-Xcompiler", "-fPIC", "-std=c++20",
        "-O3", "-gencode", "arch=compute_90a,code=sm_90a",
        "-DMIRAGE_BACKEND_USE_CUDA", "-DMIRAGE_GRACE_HOPPER",
        "-DMPK_TARGET_CC=90", f"-I{root / 'include'}",
        f"-I{root / 'include/mirage/persistent_kernel'}",
        f"-I{ROOT / 'deps/cutlass/include'}", "--expt-relaxed-constexpr", "-lcuda",
    ]
    if finite_mask:
        command.append("-DMIRAGE_FINITE_REFERENCE_MASK=1")
    subprocess.run(command, check=True)
    return command


def load(library):
    lib = ctypes.CDLL(str(library))
    lib.enqueue.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    lib.enqueue.restype = ctypes.c_int
    lib.resources.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.resources.restype = ctypes.c_int
    lib.merge_only.argtypes = [ctypes.c_void_p] * 7
    lib.merge_only.restype = ctypes.c_int
    assert lib.configure() == 0
    return lib


def norm_rotate(x, weight, cos, sin):
    x = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                + 1e-6)).to(x.dtype) * weight
    half = torch.cat((-x[..., 64:], x[..., :64]), dim=-1)
    return x * cos + half * sin


def make_case(active, context, mode, split):
    dtype, device = torch.bfloat16, "cuda"
    qkv = torch.randn((8, 8, 4, 128), device=device, dtype=dtype) * 0.2
    kc = torch.randn((512, 8, 128), device=device, dtype=dtype) * 0.2
    vc = torch.randn_like(kc)
    q_scale = torch.ones(128, device=device, dtype=dtype)
    k_scale = torch.ones_like(q_scale)
    normalized = mode in ("normalized", "normalized-below")
    if mode in ("sentinel", "below", "deep-negative"):
        query, key = {"sentinel": (200, -250), "below": (200, -252),
                      "deep-negative": (512, -1024)}[mode]
        qkv.zero_()
        qkv[:active, :, :2, 0] = query
        qkv[:active, :, 2, 0] = key
        qkv[:active, :, 3, :] = 1
        kc.zero_()
        kc[:, :, 0] = key
        vc.fill_(1)
    elif mode == "normalized-below":
        qkv.fill_(1)
        q_scale.fill_(20)
        k_scale.fill_(-20)
        vc.fill_(1)
        assert context == 0
    elif mode == "split-boundary":
        qkv.zero_()
        kc.zero_()
        vc.zero_()
        qkv[0, :, 3, :] = 128
    elif mode in ("nan", "positive-infinity", "negative-infinity"):
        qkv.fill_(1)
        qkv[:active, :, 2, :] = {
            "nan": float("nan"), "positive-infinity": float("inf"),
            "negative-infinity": -float("inf"),
        }[mode]
        assert context == 0
    frequency = 1.0 / (1e6 ** (torch.arange(0, 128, 2, device=device).float() / 128))
    angle = torch.arange(512, device=device).float()[:, None] * frequency[None, :]
    angle = torch.cat((angle, angle), dim=-1)
    cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
    q = qkv[:active, :, :2].reshape(active, 16, 128).transpose(0, 1)
    new_k = qkv[:active, :, 2].transpose(0, 1)
    if normalized:
        q = norm_rotate(q, q_scale, cos[context:context + active],
                        sin[context:context + active])
        new_k = norm_rotate(new_k, k_scale, cos[context:context + active],
                            sin[context:context + active])
    k = torch.cat((kc[:context].transpose(0, 1), new_k), dim=1)
    v = torch.cat((vc[:context].transpose(0, 1),
                   qkv[:active, :, 3].transpose(0, 1)), dim=1)
    scores = q.float() @ k.repeat_interleave(2, 0).float().transpose(-1, -2)
    scores *= 128 ** -0.5
    future = (torch.arange(context + active, device=device)[None, :]
              > context + torch.arange(active, device=device)[:, None])
    probabilities = scores.masked_fill(future, -float("inf")).softmax(-1)
    reference = (probabilities @ v.repeat_interleave(2, 0).float())
    reference = reference.transpose(0, 1).reshape(active, 2048)
    if mode == "negative-infinity":
        # Preserve the helper's existing all-negative-infinity convention.
        reference.zero_()
    result = {
        "qkv": qkv, "kc": kc, "vc": vc,
        "output": torch.full((8, 2048), float("nan"), device=device, dtype=dtype),
        "partial": torch.full((8, 8, 2, 2, 128), float("nan"), device=device, dtype=dtype),
        "lse": torch.full((8, 8, 2, 2), float("nan"), device=device),
        "q_scale": q_scale, "k_scale": k_scale, "cos": cos, "sin": sin,
        "reference": reference, "active": active, "context": context,
        "mode": mode, "split": split, "normalized": normalized,
    }
    for name, values in zip(("qo", "pi", "pages", "last"),
                            ([0, active], [0, 1], [0], [context + active])):
        result[name] = torch.tensor(values, device=device, dtype=torch.int32)
    if mode.startswith("merge-"):
        result["lse"].fill_(-float("inf"))
        if mode == "merge-deep":
            result["lse"][0].fill_(-1e6)
            result["partial"][0, :, 0].fill_(3)
            result["partial"][0, :, 1].fill_(5)
            expected = 4
        elif mode == "merge-empty":
            result["lse"][0, :, 1].fill_(0)
            result["partial"][0, :, 1].fill_(7)
            expected = 7
        else:
            expected = 0
        result["reference"] = torch.full((active, 2048), expected, device=device)
    return result


def enqueue(lib, case, poison=True):
    result = lib.enqueue(*[case[name].data_ptr() for name in POINTERS],
                         int(case["split"]), int(case["normalized"]), int(poison),
                         torch.cuda.current_stream().cuda_stream)
    assert result == 0, f"CUDA error {result}"


def check_case(lib, active, context, mode, split):
    case = make_case(active, context, mode, split)
    readonly = ("qkv", "q_scale", "k_scale", "cos", "sin", "qo", "pi", "pages", "last")
    snapshots = {name: case[name].view(torch.uint8).clone() for name in readonly}
    old_k, old_v = case["kc"].clone(), case["vc"].clone()
    if mode.startswith("merge-"):
        rc = lib.merge_only(*[case[name].data_ptr() for name in
                              ("lse", "partial", "qo", "pi", "last", "output")],
                            torch.cuda.current_stream().cuda_stream)
        assert rc == 0
    else:
        enqueue(lib, case)
    torch.cuda.synchronize()
    output = case["output"][:active].float()
    reference = case["reference"]
    if mode in ("nan", "positive-infinity"):
        correct = bool(torch.isnan(output).all())
        error = None
    else:
        error = (output - reference).abs().max().item() if active else 0.0
        correct = bool(torch.isfinite(output).all()) and error < 0.02
    preserved = all(torch.equal(case[name].view(torch.uint8), snapshots[name])
                    for name in readonly)
    for name, old in (("kc", old_k), ("vc", old_v)):
        preserved &= torch.equal(case[name][:context], old[:context])
        preserved &= torch.equal(case[name][context + active:], old[context + active:])
    preserved &= bool(torch.isnan(case["output"][active:]).all())
    empty_ok = True
    if mode == "split-boundary":
        empty_ok = bool((case["partial"][:6, :, 1] == 0).all())
        empty_ok &= bool(torch.isneginf(case["lse"][:6, :, 1]).all())
    return {"active": active, "context": context, "mode": mode, "split": split,
            "max_abs_error": error, "preserved": bool(preserved),
            "empty_chunks_correct": empty_ok, "passed": correct and preserved and empty_ok}


def configurations():
    cases = [(active, context, "random", split)
             for split in (False, True) for active in (1, 7, 8)
             for context in (0, 63, 64, 128, 250, 256, 300)]
    cases += [(active, context, mode, split)
              for split in (False, True) for active in (1, 7)
              for context in (0, 256) for mode in ("sentinel", "below", "deep-negative")]
    cases += [(active, 0, "normalized-below", split)
              for split in (False, True) for active in (1, 7)]
    cases += [(1, 0, mode, split) for split in (False, True)
              for mode in ("nan", "positive-infinity", "negative-infinity")]
    cases += [(0, 0, "random", False), (0, 0, "random", True),
              (7, 250, "split-boundary", True)]
    cases += [(1, 300, mode, True)
              for mode in ("merge-deep", "merge-empty", "merge-all-empty")]
    return cases


def main():
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This regression requires Hopper (sm90).")
    torch.set_num_threads(4)
    torch.manual_seed(793)
    with tempfile.TemporaryDirectory(prefix="mirage-attention-softmax-") as directory:
        library = Path(directory) / "attention.so"
        build(ROOT, library)
        lib = load(library)
        results = [check_case(lib, *config) for config in configurations()]
    for result in results:
        if not result["passed"]:
            print(result)
    assert all(result["passed"] for result in results)
    print(f"Passed {len(results)} Hopper attention softmax cases.")


if __name__ == "__main__":
    main()
