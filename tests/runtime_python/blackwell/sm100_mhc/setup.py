from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

cuda_home = "/usr/local/cuda-12.8"
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

macros = [
    ("MIRAGE_BACKEND_USE_CUDA", None),
    ("MIRAGE_FINGERPRINT_USE_CUDA", None),
]

setup(
    name="runtime_kernel_blackwell_mhc",
    ext_modules=[
        CUDAExtension(
            name="runtime_kernel_blackwell_mhc",
            sources=[os.path.join(this_dir, "runtime_kernel_wrapper_sm100.cu")],
            depends=[
                os.path.join(
                    this_dir,
                    "../../../../include/mirage/persistent_kernel/tasks/blackwell/mHC_post.cuh",
                ),
                os.path.join(
                    this_dir,
                    "../../../../include/mirage/persistent_kernel/tasks/blackwell/mHC_post_pre.cuh",
                ),
                os.path.join(
                    this_dir,
                    "../../../../include/mirage/persistent_kernel/tasks/blackwell/mHC_pre.cuh",
                ),
                os.path.join(
                    this_dir,
                    "../../../../include/mirage/persistent_kernel/tasks/blackwell/mul_sum_add_sm100.cuh",
                ),
                os.path.join(
                    this_dir,
                    "../../../../include/mirage/persistent_kernel/tasks/blackwell/sinkhorn.cuh",
                ),
            ],
            define_macros=macros,
            include_dirs=[
                os.path.join(this_dir, "../../../../include/mirage/persistent_kernel"),
                os.path.join(this_dir, "../../../../include/mirage/persistent_kernel/tasks"),
                os.path.join(this_dir, "../../../../include"),
                os.path.join(this_dir, "../../../../deps/cutlass/include"),
                os.path.join(this_dir, "../../../../deps/cutlass/tools/util/include"),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-DMIRAGE_GRACE_BLACKWELL"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_100a,code=sm_100a",
                    "-DMIRAGE_GRACE_BLACKWELL",
                    "-DMPK_ENABLE_TMA",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
