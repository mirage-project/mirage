from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil
from os import path
import subprocess

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

# NVSHMEM paths
nvshmem_home = os.environ.get("NVSHMEM_HOME")  # PLACEHOLDER
nvshmem_include_dir = os.path.join(nvshmem_home, "include")
nvshmem_library_dirs = [os.path.join(nvshmem_home, "lib")]

# MPI paths
mpi_home = os.environ.get("MPI_HOME")  # PLACEHOLDER
mpi_include_dir = os.path.join(mpi_home, "include")
mpi_library_dirs = [os.path.join(mpi_home, "lib")]

# Check if NVSHMEM should be used
use_nvshmem = True

macros=[("MIRAGE_BACKEND_USE_CUDA", None), ("MIRAGE_FINGERPRINT_USE_CUDA", None),
        ("MIRAGE_GRACE_HOPPER", None)]

include_dirs = [
    os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
    os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks'),
    os.path.join(this_dir, '../../../../include'),
    os.path.join(this_dir, '../../../../deps/cutlass/include'),
    os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
]

library_dirs = list(cuda_library_dirs)
libraries = ["cuda", "cudadevrt"]

extra_compile_args_nvcc = [
    '-O3',
    '-std=c++20',
    '-ccbin=mpic++',
    '-arch=native',
    # '-gencode=arch=compute_90,code=sm_90',
    # '-gencode=arch=compute_100a,code=sm_100a',
    # Torch sets __CUDA_NO_HALF* by default; NVSHMEM reduce needs half/bfloat16 operators
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
]

if use_nvshmem:
    macros.append(("USE_NVSHMEM", None))
    include_dirs.extend([nvshmem_include_dir, mpi_include_dir])
    library_dirs.extend(nvshmem_library_dirs + mpi_library_dirs)
    libraries.extend(["nvshmem_host", "nvshmem_device", "mpi"])
    extra_compile_args_nvcc.extend([
        '-rdc=true',  # Required for NVSHMEM
        '-DUSE_NVSHMEM',
    ])
else:
    extra_compile_args_nvcc.append('-rdc=false')

class BuildExtensionWithDeviceLink(BuildExtension):
    """
    Torch's BuildExtension doesn't insert a device-link step when -rdc=true is used.
    NVSHMEM device calls need that step to emit __cudaRegisterLinkedBinary_* symbols,
    otherwise the .so fails to load. We rerun a lightweight nvcc --dlink and relink
    the shared object with the produced stub.
    """

    def build_extension(self, ext):
        super().build_extension(ext)
        if not any(source.endswith(".cu") for source in ext.sources):
            return

        objects = self.compiler.object_filenames(ext.sources, output_dir=self.build_temp)
        cuda_home_local = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        nvcc_bin = os.path.join(cuda_home_local, "bin", "nvcc")
        dlink_obj = os.path.join(self.build_temp, f"{ext.name}_dlink.o")

        cmd = [nvcc_bin, '-dlink', '-Xcompiler=-fPIC', '-o', dlink_obj]
        for ld in ext.library_dirs:
            cmd += ['-L', ld]
        cmd += objects
        cmd += ['-arch=native', '-std=c++20', '-lcuda', "-lcudart", '--expt-relaxed-constexpr', '-lnvshmem_host', '-lnvshmem_device', '-lmpi', '-lcudadevrt']
        cmd += extra_compile_args_nvcc
        print(f"Running device link step: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        fullpath = self.get_ext_fullpath(ext.name)
        extra_postargs = ext.extra_link_args or []
        self.compiler.link_shared_object(
            objects + [dlink_obj],
            fullpath,
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_postargs,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
        )

setup(
    name='runtime_kernel_hopper_allreduce',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_hopper_allreduce',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_hopper.cu'),
            ],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/hopper/allreduce.cuh'),
            ],
            define_macros=macros,
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args={
                'cxx': ["-std=c++20"],
                'nvcc': extra_compile_args_nvcc
            }
        )
    ],
    # cmdclass={'build_ext': BuildExtension}
    cmdclass={'build_ext': BuildExtensionWithDeviceLink}
)
