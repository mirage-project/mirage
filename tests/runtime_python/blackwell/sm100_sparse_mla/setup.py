"""Build with the user's CUDA_HOME; no hard-coded toolkit path."""

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent
repo_root = this_dir.parents[3]

setup(
    name="runtime_kernel_sparse_mla",
    ext_modules=[CUDAExtension(
        name="runtime_kernel_sparse_mla",
        sources=[str(this_dir / "runtime_kernel_wrapper_sparse_mla.cu")],
        include_dirs=[str(repo_root / "include")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "-arch=sm_100a", "-lineinfo",
                     "--expt-relaxed-constexpr"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
