import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(this_dir, '../../include/mirage/persistent_kernel/')
cutlass_dir = os.path.join(this_dir, "../../deps/cutlass/include")
header_root_dir = os.path.join(this_dir, '../../include')
spec_decode_include_dir = os.path.join(include_dir, 'speculative_decoding')

# NVSHMEM paths — prefer conda-env installation, fall back to system.
import sys
_conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
_nvshmem_conda = os.path.join(
    _conda_prefix,
    'lib/python3.12/site-packages/nvidia/nvshmem')
_nvshmem_sys   = '/usr'

if os.path.isdir(os.path.join(_nvshmem_conda, 'include')):
    nvshmem_include_dir = os.path.join(_nvshmem_conda, 'include')
    nvshmem_lib_dir     = os.path.join(_nvshmem_conda, 'lib')
else:
    nvshmem_include_dir = os.path.join(_nvshmem_sys, 'include/nvshmem_12')
    nvshmem_lib_dir     = os.path.join(_nvshmem_sys, 'lib/x86_64-linux-gnu')

print(f"NVSHMEM include: {nvshmem_include_dir}")
print(f"NVSHMEM lib:     {nvshmem_lib_dir}")

# Collect header files for the 'depends' argument. This tells the build system
# to recompile if any of these headers change, without trying to compile them directly.
header_files = glob.glob(os.path.join(include_dir, '*.cuh'))
header_files += glob.glob(os.path.join(include_dir, 'tasks/common/*.cuh'))
header_files += glob.glob(os.path.join(cutlass_dir, '*.cuh'))
header_files += glob.glob(os.path.join(spec_decode_include_dir, '*.cuh'))
header_files += glob.glob(os.path.join(header_root_dir, '*.h'))

print(header_files)

macros = [
    ("MIRAGE_BACKEND_USE_CUDA", None),
    ("MIRAGE_FINGERPRINT_USE_CUDA", None),
    ("USE_NVSHMEM", "1"),
]

setup(
    name='runtime_kernel',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel',
            sources=[os.path.join(this_dir, 'runtime_kernel_wrapper.cu')],
            depends=header_files,
            define_macros=macros,
            include_dirs=[
                include_dir,
                cutlass_dir,
                spec_decode_include_dir,
                header_root_dir,
                nvshmem_include_dir,
            ],
            library_dirs=[nvshmem_lib_dir],
            # nvshmem_device: resolves extern __device__ symbols during nvlink
            # (the device link step triggered by --relocatable-device-code=true).
            # nvshmem_host: host-side init/malloc/barrier/finalize APIs.
            libraries=['nvshmem_device', 'nvshmem_host'],
            extra_link_args=[
                f'-Wl,-rpath,{nvshmem_lib_dir}',
            ],
            extra_compile_args={
                'cxx': [
                    '-DMIRAGE_GRACE_BLACKWELL',
                    '-DUSE_NVSHMEM=1',
                ],
                'nvcc': [
                    '-O3',
                    # Relocatable device code: allows extern __device__ symbols
                    # (nvshmemi_transfer_rma_nbi etc.) to be resolved at device
                    # link time from libnvshmem_device.a / .bc.
                    '--relocatable-device-code=true',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_90a,code=sm_90a',
                    '-gencode=arch=compute_100,code=sm_100',
                    '-gencode=arch=compute_100a,code=sm_100a',
                    '-DMIRAGE_GRACE_BLACKWELL',
                    '-DUSE_NVSHMEM=1',
                ],
                # Device link step (triggered by --relocatable-device-code=true).
                # PyTorch's cuda_devlink ninja rule runs nvcc without -dlink, so
                # we must add it here.  This step resolves NVSHMEM extern
                # __device__ symbols from libnvshmem_device.a.
                'nvcc_dlink': [
                    '-dlink',
                    f'-L{nvshmem_lib_dir}',
                    '-lnvshmem_device',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_90a,code=sm_90a',
                    '-gencode=arch=compute_100,code=sm_100',
                    '-gencode=arch=compute_100a,code=sm_100a',
                ],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
