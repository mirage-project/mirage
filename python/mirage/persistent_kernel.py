import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig

from .core import *
from .kernel import get_key_paths, KNGraph, TBGraph

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *input_list, *meta_list, *py_profiler_buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> meta_tensors;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOOiiii", &input_list, &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(input_list) || !PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "Both arg1 and arg2 must be lists.");
    return NULL;
  }

  Py_ssize_t input_size = PyList_Size(input_list);
  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < input_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (input) to void pointer", i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(input_tensors, meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers);

  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  launch_persistent_kernel();

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
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

def get_compile_command(target_cc,
                        cc,
                        file_name,
                        py_include_dir,
                        mirage_home_path,
                        mirage_inc_path,
                        mirage_deps_path,
                        nvshmem_inc_path,
                        nvshmem_lib_path,
                        mpi_inc_path,
                        mpi_lib_path,
                        py_so_path,
                        profiling,
                        use_nvshmem):
    common_cmd = [
        cc,
        file_name,
        "-O3",
        f"-I{py_include_dir}",
        f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",
        f"-I{os.path.join(mirage_home_path, 'deps/json/include')}",
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
        py_so_path
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

    if target_cc == 90:
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
    def __init__(self, mpi_rank : int, num_workers : int, num_local_schedulers: int, num_remote_schedulers : int):
        self.__finalized__ = False
        self.mpi_rank = mpi_rank
        self.num_workers = num_workers
        self.num_local_schedulers = num_local_schedulers
        self.num_remote_schedulers = num_remote_schedulers
        self.kn_graph = KNGraph(core.CyKNGraph())
        self.task_configs = []

    def new_input(self, torch_tensor: torch.Tensor, name: str = None) -> DTensor:
        # Currently only support bfloat16
        assert torch_tensor.dtype == torch.bfloat16
        dims = [d for d in torch_tensor.shape]
        strides = [s for s in torch_tensor.stride()]
        # Assert a row-major layout
        total_elements = 1
        default_strides = []
        for d in reversed(dims):
            default_strides.append(total_elements)
            total_elements *= d
        assert default_strides == strides
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=core.bfloat16)
        # FIXME: currently assert that name is not None
        assert name is not None
        self.kn_graph.attach_torch_tensor(t, torch_tensor, name)
        return t

    def new_tensor(self, dims: tuple, strides: tuple = None, dtype: dtype = bfloat16, name: str = None, io_category: str = "cuda_tensor") -> DTensor:
        # Currently only support bfloat16
        assert dtype == bfloat16
        # Assert a row-major layout
        total_elements = 1
        default_strides = []
        for d in reversed(dims):
            default_strides.append(total_elements)
            total_elements *= d
        assert default_strides == strides
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        if io_category == "cuda_tensor":
            self.kn_graph.attach_cuda_tensor(t, name)
        elif io_category == "nvshmem_tensor":
            self.kn_graph.attach_nvshmem_tensor(t, name)
        else:
            raise RuntimeError(f"Invalid io_category: {io_category}")
        return t

    def fuse_tensors(self, inputs: List[DTensor], fused_dim: int, groups: int) -> DTensor:
        # Currently only support fusing the 0-th dimension
        assert fused_dim == 0
        dims = None
        dtype = None
        for input in inputs:
            if dims is None:
                dims = [input.dim(i) for i in range(input.num_dims)]
                dtype = input.dtype
            else:
                dims[0] += input.dim(0)
                assert dtype == input.dtype
        t = self.kn_graph.new_input(dims=dims, dtype=dtype)
        self.kn_graph.fuse_tensors(inputs, t)
        return t

    def embed(self, input: DTensor, weight: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple):
        tb_graph = TBGraph(core.CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "embedding")

    def rmsnorm_linear(self, input: DTensor, weight_norm: DTensor, weight_linear: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple):
        # Currently assume that the input/weight_linear/output are 2D tensors
        assert input.num_dims == 2
        assert weight_linear.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(core.CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight_norm, (-1, -1, -1), 0, True)
        tb_graph.new_input(weight_linear, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)
        self.task_configs.append("rmsnorm_linear")

    def attention(self, input: DTensor, q_norm: DTensor, k_norm: DTensor, k_cache: DTensor, v_cache: DTensor, cos_pos_embed: DTensor, sin_pos_embed: DTensor, output: DTensor):
        # Currently assume that input/output
        assert input.num_dims == 2 # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2 # (batch_size, hidden_size / world_size)
        assert q_norm.num_dims == 1 # (head_dim)
        assert k_norm.num_dims == 1 # (head_dim)
        assert k_cache.num_dims == 4 # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4 # (batch_size, seq_len, kv_heads, head_dim)
        assert cos_pos_embed.num_dims == 2 # (seq_len, head_dim)
        tb_graph = TBGraph(core.CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2 -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2 -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)

    def compile(self, file_path : str, mpi_rank : int, num_workers : int, num_local_schedulers : int, num_remote_schedulers : int, **kwargs):
        self.__finalized__ = False
        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        tempdir_obj = tempfile.TemporaryDirectory()
        tempdir = tempdir_obj.name
        full_src_file = os.path.join(tempdir, "test.cu")
        so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")
        # check json file
        json_file_path = os.path.join(os.path.dirname(file_path), "task_graph.json")
        new_json_path = os.path.join(tempdir, "task_graph.json")
        if not os.path.exists(json_file_path):
            raise RuntimeError(f"Cannot find json file in directory {json_file_path}")
        with open(json_file_path, "r") as f:
            task_graph_json = f.read()
        with open(new_json_path, "w") as f:
            f.write(task_graph_json)
        with open(file_path, "r") as f:
            task_graph_impl = f.read()
        with open(full_src_file, "w") as f:
            f.write(task_graph_impl + HARD_CODE)

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

        #find mirage home
        if "MIRAGE_HOME" in os.environ:
            MIRAGE_HOME_PATH = os.environ.get("MIRAGE_HOME")
        else:
            raise RuntimeError("MIRAGE_HOME unspecified")

        # find nvshmem include folder and library foldera
        if "NVSHMEM_INC_PATH" in os.environ:
            NVSHMEM_INC_PATH = os.environ.get("NVSHMEM_INC_PATH")
            header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
            if not os.path.exists(header_file_path):
                raise RuntimeError(
                    "Environment variable NVSHMEM_INC_PATH is set but cannot find nvshmem.h at {header_file_path}")
        else:
            NVSHMEM_INC_PATH = "/usr/include/nvshmem_12/"
            header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
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
                    "Cannot find mpi.h, please set environment variable MPI_INC_PATH")
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
                    "Cannot find libmpi.so, please set environment variable MPI_LIB_PATH")
        target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        profiling = kwargs.get("profiling", False)
        use_nvshmem = kwargs.get("use_nvshmem", True)

        cc_cmd = get_compile_command(target_cc=target_cc,
                                     cc=cc,
                                     file_name=full_src_file,
                                     py_include_dir=py_include_dir,
                                     mirage_home_path=MIRAGE_HOME_PATH,
                                     mirage_inc_path=INCLUDE_PATH,
                                     mirage_deps_path=DEPS_PATH,
                                     nvshmem_inc_path=NVSHMEM_INC_PATH,
                                     nvshmem_lib_path=NVSHMEM_LIB_PATH,
                                     mpi_inc_path=MPI_INC_PATH,
                                     mpi_lib_path=MPI_LIB_PATH,
                                     py_so_path=so_path,
                                     profiling=profiling,
                                     use_nvshmem=use_nvshmem)
        print(cc_cmd)
        subprocess.check_call(cc_cmd)

        import importlib.util
        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")
        self.launch_func = getattr(mod, "launch_func")
        self.finalize_func = getattr(mod, "finalize_func")

        # initialize persistent kernel
        input_tensors = kwargs.get("input_tensors", [])
        meta_tensors = kwargs.get("meta_tensors", [])
        self.profiler_tensor=kwargs.get("profiler_tensor")

        input_tensors_ptr = [tensor.data_ptr() for tensor in input_tensors]
        meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
        profiler_buffer_ptr = self.profiler_tensor.data_ptr() if self.profiler_tensor is not None else 0
        self.init_func(input_tensors_ptr, meta_tensors_ptr, profiler_buffer_ptr, mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers)

        #self.call_func = getattr(mod, "call_func")

    def __call__(self, **kwargs):
        #stream = kwargs.get("stream", None)
        #if stream is None:
        #    stream = torch.cuda.default_stream()
        self.launch_func()
        if self.profiler_tensor is not None:
            from .profiler_persistent import export_to_perfetto_trace
            export_to_perfetto_trace(self.profiler_tensor, f'mirage_{self.mpi_rank}.perfetto-trace')

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        self.finalize_func()
        self.__finalized__ = True
