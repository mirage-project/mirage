import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
include_root = os.path.join(this_dir, "../../../include")
# Target compute capability; selects MAX_DYNAMIC_SHARED_MEMORY_SIZE in
# runtime_header.h the same way the megakernel build does.
target_cc = os.environ.get("MPK_TARGET_CC", "86")

setup(
    name="linear_smem_budget",
    ext_modules=[
        CUDAExtension(
            name="linear_smem_budget",
            sources=[os.path.join(this_dir, "runtime_kernel_wrapper.cu")],
            define_macros=[("MIRAGE_BACKEND_USE_CUDA", None)],
            include_dirs=[
                include_root,
                os.path.join(include_root, "mirage/persistent_kernel"),
                os.path.join(include_root, "mirage/persistent_kernel/tasks/ampere"),
                os.path.join(this_dir, "../../../deps/cutlass/include"),
            ],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-O3",
                    f"-gencode=arch=compute_{target_cc},code=sm_{target_cc}",
                    f"-DMPK_TARGET_CC={target_cc}",
                    "-DMIRAGE_UNIT_TEST",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
