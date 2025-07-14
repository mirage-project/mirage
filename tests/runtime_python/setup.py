import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(this_dir, '../../include/mirage/persistent_kernel/tasks')
header_root_dir = os.path.join(this_dir, '../../include')
spec_decode_include_dir = os.path.join(include_dir, 'speculative_decoding')

# Collect header files for the 'depends' argument. This tells the build system
# to recompile if any of these headers change, without trying to compile them directly.
header_files = glob.glob(os.path.join(include_dir, '*.cuh'))
header_files += glob.glob(os.path.join(spec_decode_include_dir, '*.cuh'))
header_files += glob.glob(os.path.join(header_root_dir, '*.h'))

print(header_files)

setup(
    name='runtime_kernel',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel',
            sources=[os.path.join(this_dir, 'runtime_kernel_wrapper.cu')],
            depends=header_files,
            include_dirs=[
                include_dir,
                spec_decode_include_dir,
                header_root_dir,
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