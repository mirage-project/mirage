from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil

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

common_include_dirs = [
    os.path.join(this_dir, '../../../../include/mirage/'),
    os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
    os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/'),
    os.path.join(this_dir, '../../../../include'),
    os.path.join(this_dir, '../../../../deps/cutlass/include'),
    os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
    os.path.join(this_dir, '../../../../deps/DeepGEMM/deep_gemm/include'),
]

common_nvcc_flags = [
    '-O3',
    '-gencode=arch=compute_100a,code=sm_100a',
    '-DMIRAGE_GRACE_BLACKWELL',
    '-DMPK_ENABLE_TMA',
    '-DCUTE_ARCH_TCGEN05_TMEM_ENABLED',
    '-DCUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED',
]

setup(
    name='runtime_kernels_fp8_moe',
    ext_modules=[
        # FP8 MoE kernel
        CUDAExtension(
            name='runtime_kernel_fp8_moe',
            sources=[os.path.join(this_dir, 'runtime_kernel_wrapper.cu')],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/fp8_group_gemm_sm100.cuh'),
            ],
            define_macros=macros,
            include_dirs=common_include_dirs,
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_BLACKWELL'],
                'nvcc': common_nvcc_flags,
            }
        ),
        # BF16 MoE kernel
        CUDAExtension(
            name='runtime_kernel_bf16_moe',
            sources=[os.path.join(this_dir, 'runtime_kernel_wrapper_bf16.cu')],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/blackwell/moe_linear_sm100.cuh'),
            ],
            define_macros=macros,
            include_dirs=common_include_dirs,
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_BLACKWELL'],
                'nvcc': common_nvcc_flags,
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
