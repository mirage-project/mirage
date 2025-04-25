import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig

from .kernel import get_key_paths

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *input_list, *output_list, *py_buffer, *py_stream, *py_profiler_buffer;
  void *buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> output_tensors;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOOOO", &input_list, &output_list, &py_buffer, &py_stream, &py_profiler_buffer)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(input_list) || !PyList_Check(output_list)) {
    PyErr_SetString(PyExc_TypeError, "Both arg1 and arg2 must be lists.");
    return NULL;
  }

  Py_ssize_t input_size = PyList_Size(input_list);
  Py_ssize_t output_size = PyList_Size(output_list);

  for(Py_ssize_t i = 0; i < input_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (input) to void pointer", i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < output_size; i++) {
    PyObject *item = PyList_GetItem(output_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (output) to void pointer", i);
      return NULL;
    }
    output_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  buffer = PyLong_AsVoidPtr(py_buffer);
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);
  cudaStream_t stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  execute_mugraph(input_tensors, output_tensors, buffer, stream, profiler_buffer);

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""

def get_compile_command(target_cc=target,
                        cc=cc,
                        file_name=file_name,
                        py_include_dir=py_include_dir,
                        mirage_inc_path=mirage_inc_path,
                        mirage_deps_path=mirage_deps_path,
                        nvshmem_inc_path=nvshmem_inc_path,
                        nvshmem_lib_path=nvshmem_lib_path,
                        mpi_inc_path=mpi_inc_path,
                        mpi_lib_path=mpi_lib_path,
                        py_so_path=py_so_path,
                        profiling=profiling,
                        use_nvshmem=use_nvshmem):
    common_cmd = [
        cc,
        file_name,
        "-O3",
        f"-I{py_include_dir}",
        f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",
    ]

    flags = [
        "-ccbin=mpic++",
        "-shared",
        "-std=c++17",
        "-rdc=true",
        "-use_fast_math",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        so_path
    ]

    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
        nvshmem_flags = [
            "-lnvshmem_host",
            "-lnvshmem_device",
            "-lmpi"
        ]
        common_cmd = common_cmd + nvshmem_cmd
        flags = flags + nvshmem_flags

    if target == 90:
        specific_cmd = [
            "-arch=sm_90a",
            "-gencode=arch=compute_90a,code=sm_90a",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    else:
        specific_cmd = [
            "-arch=native",
        ]

    return common_cmd + specific_cmd + flags

class PersistentKernel:
    def __init__(self, filepath : str, use_nvshmem : bool):
        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        tempdir_obj = tempfile.TemporaryDirectory()
        tempdir = tempdir_obj.name
        so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")
        cc = shutil.which("nvcc")
        if cc is None:
            raise RuntimeError(
                "nvcc not found. Please make sure you have installed CUDA."
            )
        # This function was renamed and made public in Python 3.10
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
        # path changes to include 'local'. This change is required to use triton with system-wide python.
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

        # find nvshmem include folder and library foldera
        if "NVSHMEM_INC_PATH" in os.environ:
            NVSHMEM_INC_PATH = os.environ.get("NVSHMEM_INC_PATH")
            header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
            if not os.path.exists(header_file_path):
                raise RuntimeError(
                    "Environment variable NVSHMEM_INC_PATH is set but cannot find nvshmem.h at {header_file_path}")
        else:
            NVSHMEM_INC_PATH = "/usr/include/nvshmem_12/"
            header_file_path = os.path.join(NVSHMEM_ROOT, "nvshmem.h")
            if not os.path.exists(header_file_path):
                raise RuntimeError(
                    "Cannot find nvshmem.h, please set environment variable NVSHMEM_INC_PATH")
        #find nvshmem shared library
        if "NVSHMEM_LIB_PATH" in os.environ:
            NVSHMEM_LIB_PATH = os.environ.get("NVSHMEM_LIB_PATH")
            lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
            if not os.path.exists(lib_file_path):
                raise RuntimeError(
                    "Environment variable NVSHMEM_LIB_PATH is set but cannot find libnvshmem.a at {lib_file_path}")
        else:
            NVSHMEM_LIB_PATH = "/usr/lib/x86_64-linux-gnu/"
            lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
            if not os.path.exists(lib_file_path):
                raise RuntimeError(
                    "Cannot find libnvshmem.a, please set environment variable NVSHMEM_LIB_PATH")
        # find mpi include foler
        if "MPI_INC_PATH" in os.environ:
            MPI_INC_PATH = os.environ.get("MPI_INC_PATH")
            header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
            if not os.path.exists(header_file_path):
                raise RuntimeError(
                    "Environment variable MPI_INC_PATH is set but cannot find mpi.h at {header_file_path}")
        else:
            MPI_INC_PATH = "/usr/include/"
            header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
            if not os.path.exists(header_file_path):
                raise RuntimeError(
                    "Cannot find nvshmem.h, please set environment variable MPI_INC_PATH")
        #find mpi shared library
        if "MPI_LIB_PATH" in os.environ:
            MPI_LIB_PATH = os.environ.get("MPI_LIB_PATH")
            lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libmpi.so")
            if not os.path.exists(lib_file_path):
                raise RuntimeError(
                    "Environment variable MPI_LIB_PATH is set but cannot find libmpi.so at {lib_file_path}")
        else:
            NVSHMEM_LIB_PATH = "/usr/lib/"
            lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libmpi.so")
            if not os.path.exists(lib_file_path):
                raise RuntimeError(
                    "Cannot find libnvshmem.a, please set environment variable MPI_LIB_PATH")
        
        cc_cmd = get_compile_command(targe_cc=target_cc,
                                     cc=cc,
                                     file_name=file_path,
                                     py_include_dir=py_include_dir,
                                     mirage_inc_path=INCLUDE_PATH,
                                     mirage_deps_path=DEPS_PATH,
                                     nvshmem_inc_path=NVSHMEM_INC_PATH,
                                     nvshmem_lib_path=NVSHMEM_LIB_PATH,
                                     mpi_inc_path=MPI_INC_PATH,
                                     mpi_lib_path=MPI_LIB_PATH,
                                     py_so_path=so_path,
                                     profiling=profiling,
                                     use_nvshmem=use_nvshmem)
        subprocess.check_call(cc_cmd)

        import importlib.util
        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")

    def __call__(self, **kwargs):
        input_tensors = kwargs.get("inputs", [])
        stream = kwargs.get("stream", None)
        if stream is None:
            stream = torch.cuda.default_stream()
