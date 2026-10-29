import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
include_root = os.path.join(this_dir, "../../../include")

setup(
    name="paged_attention_page_offset",
    ext_modules=[
        CUDAExtension(
            name="paged_attention_page_offset",
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
                "nvcc": ["-O3", "-gencode=arch=compute_80,code=sm_80", "-DMIRAGE_UNIT_TEST"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
