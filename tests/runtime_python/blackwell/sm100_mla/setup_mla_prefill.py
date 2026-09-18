from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, "../../../.."))

cuda_home = "/usr/local/cuda-12.8"
os.environ["CUDA_HOME"] = cuda_home
os.environ["PATH"] = os.path.join(cuda_home, "bin") + ":" + os.environ.get("PATH", "")

setup(
    name='runtime_kernel_mla_prefill',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_mla_prefill',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_mla_prefill.cu'),
            ],
            include_dirs=[
                os.path.join(repo_root, 'include'),
                os.path.join(repo_root, 'include/mirage/persistent_kernel'),
                os.path.join(cuda_home, 'include'),
            ],
            libraries=["cuda"],
            library_dirs=[
                os.path.join(cuda_home, "lib64"),
                os.path.join(cuda_home, "lib64", "stubs"),
            ],
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
