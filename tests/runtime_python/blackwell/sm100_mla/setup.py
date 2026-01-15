from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

nvcc_path = "/usr/local/cuda-12.8/bin/nvcc"
cuda_home = "/usr/local/cuda-12.8"

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

macros = [("MIRAGE_BACKEND_USE_CUDA", None), ("MIRAGE_FINGERPRINT_USE_CUDA", None)]

setup(
    name='runtime_kernel_blackwell_mla',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_blackwell_mla',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_sm100_mla.cu'),
            ],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/attention_mla_sm100.cuh'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/flashinfer_mla/device/sm100_mla.hpp'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/flashinfer_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp'),
            ],
            define_macros=macros,
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell'),
                os.path.join(this_dir, '../../../../include'),
                os.path.join(this_dir, '../../../../deps/cutlass/include'),
                os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_BLACKWELL', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_100a,code=sm_100a',
                    '-DMIRAGE_GRACE_BLACKWELL',
                    '-DMPK_ENABLE_TMA',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
