"""Build the kernel-wrapper extension for the MPK FP8 swap-AB Linear kernel.

Compiled with `-lineinfo` plus host-side `-g` so cuda-gdb / compute-sanitizer
can attribute to source lines. We avoid full device debug (`-G`) by default
because it disables optimization and slows nvcc/runtime substantially. Set
SM100_LINEAR_FP8_MPK_DEBUG=1 in the environment to add `-G`.
"""
import os
import sys
import shutil
from glob import glob
from setuptools import setup


def _import_torch_cpp_extension():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        return BuildExtension, CUDAExtension
    except ModuleNotFoundError as e:
        prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
        if prefix:
            pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            for sp in (
                os.path.join(prefix, "lib", pyver, "site-packages"),
                os.path.join(prefix, "lib", "site-packages"),
            ):
                if os.path.isdir(os.path.join(sp, "torch")):
                    sys.path.insert(0, sp)
                    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
                    return BuildExtension, CUDAExtension
        raise e


BuildExtension, CUDAExtension = _import_torch_cpp_extension()
import torch
import torch.utils.cpp_extension as torch_cpp_extension

this_dir = os.path.dirname(os.path.abspath(__file__))


def _resolve_cuda_home():
    env_cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_cuda_home and os.path.exists(os.path.join(env_cuda_home, "bin", "nvcc")):
        return env_cuda_home
    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda:
        candidate = os.path.join("/usr/local", f"cuda-{torch_cuda}")
        if os.path.exists(os.path.join(candidate, "bin", "nvcc")):
            return candidate
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        return os.path.dirname(os.path.dirname(nvcc_path))
    return "/usr/local/cuda"


cuda_home = _resolve_cuda_home()
os.environ["CUDA_HOME"] = cuda_home
os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
torch_cpp_extension.CUDA_HOME = cuda_home

cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

blackwell_task_dir = os.path.join(
    this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell',
)
blackwell_depends = sorted(
    glob(os.path.join(blackwell_task_dir, '**', '*.cuh'), recursive=True)
)

debug_extra = []
if os.environ.get("SM100_LINEAR_FP8_MPK_DEBUG", "0") == "1":
    # Full device debug: disables optimization, enables cuda-gdb stepping.
    debug_extra = ["-G", "-Xcompiler=-g", "-Xcompiler=-O0"]
    nvcc_opt = "-O0"
else:
    debug_extra = ["-Xcompiler=-g"]
    nvcc_opt = "-O3"

setup(
    name='runtime_kernel_blackwell_linear_fp8_mpk',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_blackwell_linear_fp8_mpk',
            sources=[os.path.join(this_dir, 'runtime_kernel_wrapper.cu')],
            depends=blackwell_depends,
            define_macros=[
                ("MIRAGE_BACKEND_USE_CUDA", None),
                ("MIRAGE_FINGERPRINT_USE_CUDA", None),
                ("MIRAGE_GRACE_BLACKWELL", None),
                ("MPK_ENABLE_TMA", None),
            ],
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks'),
                os.path.join(this_dir, '../../../../include'),
                os.path.join(this_dir, '../../../../deps/cutlass/include'),
                os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_BLACKWELL', '-g', '-O3'],
                'nvcc': [
                    nvcc_opt,
                    '-lineinfo',
                    '-gencode=arch=compute_100a,code=sm_100a',
                    '-DMIRAGE_GRACE_BLACKWELL',
                    '-DMPK_ENABLE_TMA',
                    '-std=c++20',
                    '--expt-relaxed-constexpr',
                    '-use_fast_math',
                ] + debug_extra,
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
