from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='runtime_kernel',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper.cu'),
            ],
            include_dirs=[
                os.path.join(this_dir, '../../src/runtime'),
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_80,code=sm_80',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)