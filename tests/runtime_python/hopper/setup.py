from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil
from os import path
from glob import glob

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

hopper_task_dir = os.path.join(
    this_dir,
    '../../../../include/mirage/persistent_kernel/tasks/hopper',
)
hopper_depends = sorted(
    glob(os.path.join(hopper_task_dir, '**', '*.cuh'), recursive=True)
)

setup(
    name='runtime_kernel_hopper',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_hopper',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_hopper.cu'),
            ],
            depends=hopper_depends,
            include_dirs=[
                os.path.join(this_dir, '../../../include/mirage/persistent_kernel'),
                os.path.join(this_dir, '../../../include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_HOPPER',
                '-DMIRAGE_BACKEND_USE_CUDA',
                '-DMPK_TARGET_CC=90'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_90a,code=sm_90a',
                    '-DMIRAGE_BACKEND_USE_CUDA',
                    '-DMIRAGE_GRACE_HOPPER',
                    '-DMIRAGE_BACKEND_USE_CUDA',
                    # '-DMIRAGE_PROFILE_HOPPER',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
