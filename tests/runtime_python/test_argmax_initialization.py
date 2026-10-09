"""Exercise both production argmax helpers without building the Python extension.

Run with ``python tests/runtime_python/test_argmax_initialization.py`` on a
CUDA GPU with BF16 support. Requires nvcc and the checked-out CUTLASS submodule;
no model weights are needed.
"""

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

BATCH, PADDED, VOCAB, PARTIALS = 8, 153600, 151936, 96
IMPLEMENTATIONS = ("ampere", "sm100-partial/ampere-reduce", "sm100")
WINNERS = [0, 31, 127, 1599, 1600, 150400, 151935, 77777]


def compile_wrapper(directory):
    root = Path(__file__).resolve().parents[2]
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required to build the test wrapper.")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("This test requires a CUDA GPU with BF16 support.")
    cc = f"{major}{minor}"
    arch = cc + ("a" if major >= 9 else "")
    library = Path(directory) / "argmax_initialization.so"
    command = [
        nvcc, str(Path(__file__).with_name("argmax_initialization_wrapper.cu")),
        "-o", str(library), "-shared", "-Xcompiler", "-fPIC", "-std=c++20",
        "-O3", "--use_fast_math", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-DMIRAGE_BACKEND_USE_CUDA", f"-DMPK_TARGET_CC={cc}",
        f"-I{root / 'include'}", f"-I{root / 'include/mirage/persistent_kernel'}",
        f"-I{root / 'deps/cutlass/include'}", "--expt-relaxed-constexpr",
    ]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(library))
    lib.launch.argtypes = ([ctypes.c_void_p] * 4
                           + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.launch.restype = ctypes.c_int
    return lib


def cases():
    for name, background, peak in (
        ("below-old-sentinel", -65536, None),
        ("equal-old-bf16-sentinel", -49920, None),
        ("minimum-finite-bf16", torch.finfo(torch.bfloat16).min, None),
        ("unique-below-sentinel", -131072, -65536),
        ("unique-at-sentinel", -65536, -49920),
        ("unique-above-sentinel", -65536, -49664),
        ("zero-padding-control", -2, None),
        ("positive-control", -2, 3),
    ):
        scores = torch.full((BATCH, PADDED), background,
                            device="cuda", dtype=torch.bfloat16)
        scores[:, VOCAB:] = 0
        if peak is not None:
            for row, index in enumerate(WINNERS):
                scores[row, index] = peak
        yield name, scores
    scores = torch.zeros((BATCH, PADDED), device="cuda", dtype=torch.bfloat16)
    scores[:, 1] = scores[:, 4] = 3
    scores[:, VOCAB:] = float("inf")
    yield "tied-maxima-with-padding", scores
    generator = torch.Generator(device="cuda").manual_seed(758)
    scores = torch.randn((BATCH, PADDED), generator=generator,
                         device="cuda", dtype=torch.bfloat16)
    scores[:, VOCAB:] = float("inf")
    yield "random-with-padding", scores


def check_case(lib, scores, active, implementation):
    values = torch.full((BATCH, PARTIALS), 42,
                        device="cuda", dtype=torch.bfloat16)
    indices = torch.full((BATCH, PARTIALS), -17, device="cuda", dtype=torch.int64)
    output = torch.full((BATCH,), -17, device="cuda", dtype=torch.int64)
    saved = scores.view(torch.int16).clone()
    stream = torch.cuda.current_stream()
    error = lib.launch(
        *[t.data_ptr() for t in (scores, values, indices, output)],
        active, implementation, stream.cuda_stream)
    if error:
        raise RuntimeError(f"Argmax launch failed with CUDA error {error}.")
    stream.synchronize()
    logical = scores[:active, :VOCAB].float().cpu()
    maxima = logical.max(dim=1).values.tolist()
    winners = output[:active].cpu().tolist()
    ok = all(0 <= index < VOCAB and logical[row, index].item() == maxima[row]
             for row, index in enumerate(winners))
    ok = ok and torch.equal(scores.view(torch.int16), saved)
    ok = ok and bool((output[active:] == -17).all())
    ok = ok and bool((values[active:] == 42).all())
    ok = ok and bool((indices[active:] == -17).all())
    return ok, winners


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="mirage-argmax-init-") as directory:
        lib = compile_wrapper(directory)
        for name, scores in cases():
            for implementation, label in enumerate(IMPLEMENTATIONS):
                for active in (0, 1, 7, 8):
                    ok, winners = check_case(lib, scores, active, implementation)
                    results.append(ok)
                    print(f"{name} {label} active={active}: "
                          f"{'PASS' if ok else 'FAIL'} {winners}", flush=True)
    assert all(results), f"{results.count(False)}/{len(results)} cases failed"
    print(f"PASSED: {len(results)} cases")


if __name__ == "__main__":
    main()
