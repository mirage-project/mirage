from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil
from os import path

this_dir = os.path.dirname(os.path.abspath(__file__))

nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
else:
    cuda_home = "/usr/local/cuda"

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

setup(
    name='runtime_kernel_cute_hopper',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_cute_hopper',
            sources=[
                os.path.join(this_dir, 'gemm.cu'),
            ],
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/cute/hopper'),
                os.path.join(this_dir, '../../../../include'),
                os.path.join(this_dir, '../../../../deps/cutlass/include'),
                os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_HOPPER',
                '-DMIRAGE_BACKEND_USE_CUDA'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_90a,code=sm_90a',
                    '-DMIRAGE_GRACE_HOPPER',
                    '-DMIRAGE_PROFILE_HOPPER',
                    '-DMIRAGE_BACKEND_USE_CUDA',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
