from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, "../../../.."))

nvcc_path = "/usr/local/cuda-12.8/bin/nvcc"
cuda_home = "/usr/local/cuda-12.8"
os.environ["CUDA_HOME"] = cuda_home
os.environ["PATH"] = os.path.dirname(nvcc_path) + ":" + os.environ.get("PATH", "")

cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

setup(
    name='runtime_kernel_mla_decode',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_mla_decode',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_mla_decode.cu'),
            ],
            include_dirs=[
                os.path.join(repo_root, 'include'),
                os.path.join(repo_root, 'include/mirage/persistent_kernel'),
                os.path.join(cuda_home, 'include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_100a',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
