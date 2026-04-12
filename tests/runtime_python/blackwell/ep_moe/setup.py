from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil
import glob

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

macros = [("MIRAGE_BACKEND_USE_CUDA", None), ("MIRAGE_FINGERPRINT_USE_CUDA", None)]

setup(
    name='runtime_kernel_ep_moe',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_ep_moe',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_ep_moe.cu'),
            ],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/common/moe_routing_distributed.cuh'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/common/all_to_all_combine_task.cuh'),
            ],
            define_macros=macros,
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/'),
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
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
