from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='norm_linear_cuda',
    ext_modules=[
        CUDAExtension(
            name='norm_linear_cuda',
            sources=[
                os.path.join(this_dir, 'norm_linear_wrapper.cu'),
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