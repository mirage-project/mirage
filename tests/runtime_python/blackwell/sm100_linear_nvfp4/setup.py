from setuptools import setup
import os, sys
import shutil
from glob import glob

def _import_torch_cpp_extension():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        return BuildExtension, CUDAExtension
    except ModuleNotFoundError as e:
        prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
        if prefix:
            pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            for sp in (
                os.path.join(prefix, "lib", pyver, "site-packages"),
                os.path.join(prefix, "lib", "site-packages"),
            ):
                if os.path.isdir(os.path.join(sp, "torch")):
                    sys.path.insert(0, sp)
                    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
                    return BuildExtension, CUDAExtension
        raise e

BuildExtension, CUDAExtension = _import_torch_cpp_extension()

this_dir = os.path.dirname(os.path.abspath(__file__))

nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
else:
    cuda_home = "/usr/local/cuda"

cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

blackwell_task_dir = os.path.join(
    this_dir,
    '../../../../include/mirage/persistent_kernel/tasks/blackwell',
)
blackwell_depends = sorted(
    glob(os.path.join(blackwell_task_dir, '**', '*.cuh'), recursive=True)
)

setup(
    name='runtime_kernel_blackwell_linear_nvfp4',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_blackwell_linear_nvfp4',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_sm100.cu'),
            ],
            depends=blackwell_depends,
            define_macros=[("MIRAGE_BACKEND_USE_CUDA", None), ("MIRAGE_FINGERPRINT_USE_CUDA", None)],
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks'),
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
                    '-DMPK_ENABLE_TMA',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)