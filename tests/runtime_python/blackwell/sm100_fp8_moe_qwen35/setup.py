from setuptools import setup
import os, sys
import shutil
from glob import glob

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
    this_dir,
    '../../../../include/mirage/persistent_kernel/tasks/blackwell',
)
blackwell_depends = sorted(
    glob(os.path.join(blackwell_task_dir, '**', '*.cuh'), recursive=True)
)

setup(
    name='runtime_kernel_blackwell_fp8_moe_qwen35',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_blackwell_fp8_moe_qwen35',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_qwen35.cu'),
            ],
            depends=blackwell_depends,
            # MPK_TARGET_CC and MODE_OFFLINE are NOT optional. The MoE
            # grouped-GEMM guards its shared-memory arena with
            #   static_assert(smem_bytes_k(...) <= MAX_DYNAMIC_SHARED_MEMORY_SIZE)
            # and that constant is selected by these macros (runtime_header.h:
            # 35-64). An UNDEFINED MPK_TARGET_CC preprocesses as 0, so the TU
            # picked the 163 KiB fallback instead of B200's 207 KiB budget -- the
            # budget the megakernel actually launches with (persistent_kernel.py
            # passes exactly these). That was invisible while the only path here
            # needed 41 KiB; M4-I7's PATH 2 needs 166,580 B, which is over the
            # fallback and comfortably under the real budget, so the wrong
            # constant made an admissible path fail to BUILD.
            # M4-I2 fixed sm100_linear_fp8_blockscale/setup.py for the same
            # reason; 12 of the 14 blackwell test setups still lack these.
            define_macros=[("MIRAGE_BACKEND_USE_CUDA", None),
                           ("MIRAGE_FINGERPRINT_USE_CUDA", None),
                           ("MPK_TARGET_CC", "100"),
                           ("MODE_OFFLINE", None)],
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
                'cxx': ['-DMIRAGE_GRACE_BLACKWELL'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_100a,code=sm_100a',
                    '-DMIRAGE_GRACE_BLACKWELL',
                    # The grouped FP8 MoE kernel is TMA + tcgen05 block-scaled
                    # UMMA; these are the same flags the in-tree DSV3 MoE test
                    # extension builds with (sm100_fp8_moe/setup.py:33-40).
                    '-DMPK_ENABLE_TMA',
                    '-DCUTE_ARCH_TCGEN05_TMEM_ENABLED',
                    '-DCUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED',
                    '--expt-relaxed-constexpr',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
