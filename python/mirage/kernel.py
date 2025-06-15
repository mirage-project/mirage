import torch

import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig
from typing import *

from .core import *
from .threadblock import *
from .visualizer import *
from .utils import *
from .global_config import global_config
from .graph_dataset import graph_dataset

from collections import deque

MAX_THREADS = os.cpu_count()

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>

static PyObject *launch(PyObject *self, PyObject *args) {
  PyObject *input_list, *output_list, *comm_list, *py_buffer, *py_stream, *py_profiler_buffer;
  void *buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> output_tensors;
  std::vector<void*> comm_buffers;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOOOOO", &input_list, &output_list, &comm_list, &py_buffer, &py_stream, &py_profiler_buffer)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(input_list) || !PyList_Check(output_list) || !PyList_Check(comm_list)) {
    PyErr_SetString(PyExc_TypeError, "Both arg1 and arg2 must be lists.");
    return NULL;
  }

  Py_ssize_t input_size = PyList_Size(input_list);
  Py_ssize_t output_size = PyList_Size(output_list);
  Py_ssize_t comm_size = PyList_Size(comm_list);

  for(Py_ssize_t i = 0; i < input_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (input) to void pointer", (int)i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < output_size; i++) {
    PyObject *item = PyList_GetItem(output_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (output) to void pointer", (int)i);
      return NULL;
    }
    output_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < comm_size; i++) {
    PyObject *item = PyList_GetItem(comm_list, i);
    void* comm_buffer = PyLong_AsVoidPtr(item);
    if(!comm_buffer) {
      PyErr_Format(PyExc_TypeError, "Invalid comm buffer at index %d", (int)i);
      return NULL;
    }
    comm_buffers.push_back(comm_buffer);
  }

  buffer = PyLong_AsVoidPtr(py_buffer);
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);
  cudaStream_t stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  execute_mugraph(input_tensors, output_tensors, comm_buffers, buffer, stream, profiler_buffer);

  Py_RETURN_NONE;
}

#if USE_NVSHMEM
static PyObject* allocate_comm_buffers(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    const std::vector<size_t> sizes = get_comm_sizes();
    
    PyObject* buffer_list = PyList_New(sizes.size());
    if (!buffer_list) return NULL;
    
    for (size_t i = 0; i < sizes.size(); i++) {
        void* ptr;
        if (sizes[i] <= 1024) {
            assert(sizes[i] % sizeof(uint64_t) == 0);
            ptr = nvshmem_calloc(sizes[i] / sizeof(uint64_t), sizeof(uint64_t));
        } else {
            ptr = nvshmem_malloc(sizes[i]);
        }
        if (!ptr) {
            // Cleanup previously allocated buffers
            for (size_t j = 0; j < i; j++) {
                void* prev_ptr = PyLong_AsVoidPtr(PyList_GET_ITEM(buffer_list, j));
                nvshmem_free(prev_ptr);
            }
            Py_DECREF(buffer_list);
            PyErr_SetString(PyExc_RuntimeError, "nvshmem_malloc failed");
            return NULL;
        }
        
        PyObject* py_ptr = PyLong_FromVoidPtr(ptr);
        if (!py_ptr) {
            nvshmem_free(ptr);
            Py_DECREF(buffer_list);
            return NULL;
        }
        PyList_SET_ITEM(buffer_list, i, py_ptr);
    }
    
    return buffer_list;
}

static PyObject* free_comm_buffers(PyObject* self, PyObject* args) {
    PyObject* buffer_list;
    
    // Parse the input argument as a list
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &buffer_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of buffer pointers");
        return NULL;
    }

    Py_ssize_t num_buffers = PyList_Size(buffer_list);
    
    // Free each buffer in the list
    for (Py_ssize_t i = 0; i < num_buffers; i++) {
        PyObject* item = PyList_GetItem(buffer_list, i);
        void* ptr = PyLong_AsVoidPtr(item);
        
        if (!ptr) {
            PyErr_Format(PyExc_ValueError, 
                        "Invalid pointer at index %zd (might already be freed)", i);
            return NULL;
        }
        
        nvshmem_free(ptr);
    }

    Py_RETURN_NONE;
}

static PyObject* initialize_mpi_nvshmem(PyObject* self, PyObject* args) {
    int rank;
    if (!PyArg_ParseTuple(args, "i", &rank)) {
        PyErr_SetString(PyExc_TypeError, "Invalid parameter - expected integer rank");
        return NULL;
    }
    initialize_mpi_nvshmem(rank);
    Py_RETURN_NONE;
}

static PyObject* finalize_mpi_nvshmem(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    finalize_mpi_nvshmem();
    Py_RETURN_NONE;
}
#endif

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  #if USE_NVSHMEM
  {"allocate_comm_buffers", allocate_comm_buffers, METH_NOARGS, 
     "Allocate nvshmem buffers"},
  {"free_comm_buffers", free_comm_buffers, METH_VARARGS, "Free nvshmem buffers"},
  {"initialize_mpi_nvshmem", initialize_mpi_nvshmem, METH_VARARGS,
     "Initialize MPI and NVSHMEM with given rank"},
  {"finalize_mpi_nvshmem", finalize_mpi_nvshmem, METH_NOARGS,
     "Finalize MPI and NVSHMEM"},
  #endif
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

dtype_map = {
    'int8':    torch.int8,
    'int16':   torch.int16,
    'int32':   torch.int32,
    'int64':   torch.int64,
    'uint8':   torch.uint8,
    'fp16':    torch.float16,
    'bf16':    torch.bfloat16,
    'fp32':    torch.float32,
    'fp64':    torch.float64
}

# Because pip install -e . and pip install . have different directory structure,
# we need to check the directory structure to find the correct MIRAGE_ROOT.
def get_key_paths():
    root_dir = os.path.join(
        os.path.dirname(__file__), "../.."
    )  # Using pip install -e .
    if not os.path.exists(os.path.join(root_dir, "deps")):  # Using pip install .
        root_dir = os.path.dirname(__file__)

    # If MIRAGE_ROOT is not set, use the root_dir as MIRAGE_ROOT
    MIRAGE_ROOT = os.environ.get("MIRAGE_ROOT", root_dir)

    INCLUDE_PATH = ""
    DEPS_PATH = ""
    if os.path.exists(os.path.join(MIRAGE_ROOT, "deps")):
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "deps")
    else:
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "include/deps")

    assert os.path.exists(
        MIRAGE_ROOT
    ), "No MIRAGE_ROOT directory found. Likely using the wrong MIRAGE_ROOT."
    assert os.path.exists(
        INCLUDE_PATH
    ), "No /include directory found. Likely using the wrong MIRAGE_ROOT."
    assert os.path.exists(
        DEPS_PATH
    ), "No /deps directory found. Likely using the wrong MIRAGE_ROOT."

    return MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH

def get_cc_cmd(
    target, cc, FILE_NAME, py_include_dir, INCLUDE_PATH, DEPS_PATH, so_path, profiling
):
    common_cmd = [
        cc,
        FILE_NAME,
        "-O3",
        f"-I{py_include_dir}",
        f"-I{os.path.join(INCLUDE_PATH, 'mirage/transpiler/runtime')}",
        f"-I{os.path.join(DEPS_PATH, 'cutlass/include')}",
        "-ccbin=mpic++",
        # f"-I{NCCL_ROOT}/include/",
        # f"-L{NCCL_ROOT}/lib/",
        #f"-I{MPI_ROOT}/include",
        #f"-L{MPI_ROOT}/lib",
        #f"-I{NVSHMEM_ROOT}/include",
        #f"-L{NVSHMEM_ROOT}/lib",
        f"-I/usr/include/nvshmem_12/",
        f"-L/usr/lib/x86_64-linux-gnu/",
        "-shared",
        "-std=c++17",
        "-rdc=true",
        "-use_fast_math",
        #"-lnvshmem",
        "-lnvshmem_host",
        "-lnvshmem_device", # Two include more than only nvshmem
        "-lcublas",
        "-lnccl",
        "-lmpi",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        so_path,
    ]

    if target == 90:
        specific_cmd = [
            "-arch=sm_90a",
            "-gencode=arch=compute_90a,code=sm_90a",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    else:
        specific_cmd = [
            "-arch=native",
        ]+ (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])

    return common_cmd[:6] + specific_cmd + common_cmd[6:]


def check_stride(dims, strides, layout="row-major"):
    curr_stride = 1
    if layout == "row-major":
        for i in range(len(dims) - 1, -1, -1):
            if strides[i] != curr_stride:
                return False
            curr_stride *= dims[i]
    elif layout == "column-major":
        for i in range(len(dims)):
            if strides[i] != curr_stride:
                return False
            curr_stride *= dims[i]
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return True


def gen_empty_tensor(alloc_size, shape, stride, device, dtype=torch.float16):
    return torch.empty(alloc_size, dtype=dtype, device=device).as_strided(shape, stride)


class Handle:
    def __init__(self, handles=[], remain_op=None) -> None:
        self.handles = handles
        self.remain_op = remain_op

    def wait(self):
        for handle in self.handles:
            handle.wait()
        if self.remain_op:
            self.remain_op()


class KNGraph:
    def __init__(self, graph, gpu_dim: tuple = (1, 1, 1)):
        self.cygraph = graph

        self._is_compiled = False
        self.run = None
        self.initialize_mpi_nvshmem = None
        self.finalize_mpi_nvshmem = None
        self.allocate_comm_buffers = None
        self.free_comm_buffers = None
        self._valid_cuda_kernels = False
        self._cached_results = None
        self.visualizer = None
        self.use_nvshmem = False
        self.nvshmem = None
        self.gpu_dim = gpu_dim
        self.input_maps = []

        self.backend = "cuda"

    def new_input(
        self, dims: tuple, strides: tuple = None, gpu_input_map: tuple = (-1, -1, -1), dtype: dtype = float16
    ) -> DTensor:
        # use the default strided layout if strides = None
        if strides is None:
            total_elements = 1
            strides = []
            for d in reversed(dims):
                strides.append(total_elements)
                total_elements *= d
            strides = reversed(strides)
        else:
            assert len(dims) == len(strides)
            assert check_stride(dims, strides, "row-major") | check_stride(
                dims, strides, "column-major"
            )
        self.input_maps.append(gpu_input_map)
        return self.cygraph.new_input(dims, tuple(strides), gpu_input_map, dtype)

    def mark_output(self, A: DTensor, strides: tuple = None):
        return self.cygraph.mark_output(A, strides)

    def matmul(self, A: DTensor, B: DTensor) -> DTensor:
        return self.cygraph.matmul(A, B)

    def reduction(self, A: DTensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def exp(self, A: DTensor):
        return self.cygraph.exp(A)

    def silu(self, A: DTensor):
        return self.cygraph.silu(A)

    def gelu(self, A: DTensor):
        return self.cygraph.gelu(A)

    def relu(self, A: DTensor):
        return self.cygraph.relu(A)

    def clamp(self, A: DTensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)

    def sqrt(self, A: DTensor):
        return self.cygraph.sqrt(A)

    def square(self, A: DTensor):
        return self.cygraph.square(A)

    def add(self, A: DTensor, B: DTensor):
        return self.cygraph.add(A, B)

    def mul(self, A: DTensor, B: DTensor):
        return self.cygraph.mul(A, B)

    def div(self, A: DTensor, B: DTensor):
        return self.cygraph.div(A, B)

    def pow(self, A: DTensor, B: DTensor):
        return self.cygraph.pow(A, B)

    def rms_norm(self, A: DTensor, normalized_shape: tuple):
        return self.cygraph.rms_norm(A, normalized_shape)

    def customized(self, inputs: list[DTensor], bgraph: TBGraph) -> list[DTensor]:
        self.use_nvshmem = bgraph.use_nvshmem
        return self.cygraph.customized(inputs, bgraph.cygraph)

    # TODO (linsj20)
    def allreduce(self, A: DTensor, reduce_op="sum", inplace=False):
        return self.cygraph.all_reduce(A, reduce_op, inplace)
    
    def get_owner_independent_hash(self):
        return self.cygraph.get_owner_independent_hash()

    def valid_kernels(self):
        assert self._is_compiled, "Should check kernel validness after compilation"
        return self._valid_cuda_kernels
    
    def get_tensor_slice(self, tensor, input_map, rank):
        if rank == 2:
            print("[", rank, "] input_map: ", input_map)
        remaining_rank = rank
        gpu_indices = [0, 0, 0]
        for i in range(len(self.gpu_dim) - 1, -1, -1):
            if self.gpu_dim[i] > 1:
                gpu_indices[i] = remaining_rank % self.gpu_dim[i]
                remaining_rank //= self.gpu_dim[i]
        
        slices = [slice(None) for _ in range(len(tensor.shape))]
        
        for gpu_dim, tensor_dim in enumerate(input_map):
            if rank == 2:
                print("[", rank, "] tensor_dim: ", tensor_dim, "gpu_dim: ", gpu_dim)
            if self.gpu_dim[gpu_dim] > 1:
                dim_size = tensor.shape[tensor_dim]
                chunk_size = (dim_size + self.gpu_dim[gpu_dim] - 1) // self.gpu_dim[gpu_dim]
                
                start_idx = gpu_indices[gpu_dim] * chunk_size
                end_idx = min(start_idx + chunk_size, dim_size)
                
                slices[tensor_dim] = slice(start_idx, end_idx)

        if rank == 2:
            print("[", rank, "] slices: ", slices)
        
        result_slice = tensor[tuple(slices)]
        result = result_slice.contiguous().to(dtype=tensor.dtype, device=tensor.device)
        
        return result

    def get_divided_inputs(self, inputs):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        results = []
        for input_tensor, input_map in zip(inputs, self.input_maps):
            results.append(self.get_tensor_slice(input_tensor, input_map, rank))
        return results

    def get_error_message(self):
        assert self._is_compiled, "Should check error message after compilation"
        return self._error_message

    def __call__(self, **kwargs):
        if self.backend == "cuda":
            return self.cuda_call(**kwargs)
        elif self.backend == "nki":
            raise NotImplementedError("NKI backend is not implemented yet")
        elif self.backend == "triton":
            return self.triton_call(**kwargs)

    def triton_call(self, **kwargs):
        assert self.run is not None, "The graph is not compiled to triton yet."
        input_tensors = kwargs.get("inputs", [])
        verbose = kwargs.get("verbose", False)

        output_shapes = self._cached_results["output_shapes"]
        output_tensors = [
            torch.zeros(
                shape, dtype=input_tensors[0].dtype, device=input_tensors[0].device
            )
            for shape in output_shapes
        ]
        if verbose:
            print("Input tensors:")
            for t in input_tensors:
                print(f"Shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")
            print("Output tensors:")
            for t in output_tensors:
                print(f"Shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")

        self.run(*input_tensors, *output_tensors)
        return output_tensors

    def cuda_call(self, **kwargs):
        results = self.compile(**kwargs)
        if kwargs.get("save_codes", False):
            print("Saving generated codes to generated_codes/generated_kernel.cu")
            os.makedirs("generated_codes", exist_ok=True)
            with open("generated_codes/generated_kernel.cu", "w") as f:
                f.write(results["code"])

        # directly return if the Transpiler cannot generate valid CUDA kernels
        if not self._valid_cuda_kernels:
            return None

        assert self.run is not None, "The graph is not compiled yet."

        _input_tensors = kwargs.get("inputs", [])
        input_tensors = []
        rank = kwargs.get("rank", 0)

        if len(self.input_maps) > 0:
            # Multi GPU using
            input_tensors = self.get_divided_inputs(_input_tensors)
            # if rank == 2:
                # print("[", rank, "] input_tensors: ", input_tensors[0].shape, input_tensors[0], "\n", input_tensors[1].shape, input_tensors[1])
        else:
            input_tensors = _input_tensors

        
        # print input_tensors info
        for i, tensor in enumerate(input_tensors):
            print(f"input_tensors[{i}].shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}, strides: {tensor.stride()}")

        stream = kwargs.get("stream", None)
        if stream is None:
            stream = torch.cuda.default_stream()

        assert self.cygraph.get_num_inputs() == len(
            input_tensors
        ), "Expected {} input tensors, got {}".format(
            self.cygraph.get_num_inputs(), len(input_tensors)
        )

        # TODO: dtype and device
        buffer_tensor = torch.empty(
            results["buf_size"], dtype=torch.uint8, device=input_tensors[0].device
        ).contiguous()

        output_tensors = [
            gen_empty_tensor(
                meta["alloc_size"],
                meta["shape"],
                meta["strides"],
                device=input_tensors[0].device,
                dtype=input_tensors[0].dtype,
            )
            for meta in results["output_directives"]
        ]

        prodiler_buffer_tensor = torch.empty(
            results["profiler_buf_size"],
            dtype=torch.uint64,
            device=input_tensors[0].device,
        ).contiguous()

        buffer_tensor_ptr = buffer_tensor.data_ptr()
        input_tensors_ptr = [tensor.data_ptr() for tensor in input_tensors]
        output_tensors_ptr = [tensor.data_ptr() for tensor in output_tensors]
        prodiler_buffer_tensor_ptr = prodiler_buffer_tensor.data_ptr()
        
        comm_buffers_ptr = []
        if self.use_nvshmem:
            self.initialize_mpi_nvshmem(rank)
            try:
                comm_buffers_ptr = self.allocate_comm_buffers()
            except Exception as e:
                print(f"Error when allocating comm buffers: {e}")
                sys.exit(1)
        self.run(
            input_tensors_ptr,
            output_tensors_ptr,
            comm_buffers_ptr,
            buffer_tensor_ptr,
            stream.cuda_stream,
            prodiler_buffer_tensor_ptr,
        )
        if self.use_nvshmem:
            self.free_comm_buffers(comm_buffers_ptr)
            self.finalize_mpi_nvshmem()

        if results["profiler_buf_size"] > 0:
            from .profiler import export_to_perfetto_trace
            profiler_result_dir = "./profiling_results"
            profiler_result_file = os.path.join(profiler_result_dir, 'mirage.perfetto-trace')
            os.makedirs(profiler_result_dir, exist_ok=True)
            export_to_perfetto_trace(prodiler_buffer_tensor, profiler_result_file)
            print(f"Exported profiling results to {profiler_result_file}, please view it with perfetto: https://ui.perfetto.dev/")
        return output_tensors

    def compile(self, async_=False, **kwargs):
        if self._is_compiled:
            return self._cached_results

        input_tensors = kwargs.get("inputs", [])
        input_strides = []
        # Check that the input_strides match uGraph's specification
        dtensors = self.cygraph.get_input_dtensors()
        assert len(dtensors) == len(
            input_tensors
        ), "Given number of inputs do not match the uGraph's inputs"
        for i in range(len(dtensors)):
            dims, strides = self.cygraph.get_input_dtensor_shape_and_stride(dtensors[i])
            # These are noted because the input tensors may need to be divided when multi-GPU is used
            # assert (
            #     dims == input_tensors[i].shape
            # ), "Expected input dims {}, got input dims {}".format(
            #     dims, input_tensors[i].shape
            # )
            # assert (
            #     strides == input_tensors[i].stride()
            # ), "Expected input strides {}, got input strides {}".format(
            #     strides, input_tensors[i].stride()
            # )
            input_strides.append(strides)
        target_cc = kwargs.get(
            "target_cc",
            torch.cuda.get_device_properties(0).major * 10
            + torch.cuda.get_device_properties(0).minor,
        )
        num_warp_groups = kwargs.get("num_warp_groups", 2)
        pipeline_stages = kwargs.get("pipeline_stages", 2)
        # TODO, add profling for Ampere later to show gpu wave
        profiling = kwargs.get("profiling", False)
        enable_online_softmax = kwargs.get("enable_online_softmax", False)

        result = generate_cuda_program(
            self.cygraph,
            target_cc=target_cc,
            input_strides=input_strides,
            num_warp_groups=num_warp_groups,
            pipeline_stages=pipeline_stages,
            profiling=profiling,
            enable_online_softmax=enable_online_softmax,
        )

        # print(result['code'])
        if result["max_smem_size"] > get_shared_memory_capacity(target_cc):
            # the transpiled kernel exceeds shared memory limit
            print(
                f"required shared memory size {result['max_smem_size']} exceed max shared memory size of current gpu arch {get_shared_memory_capacity(target_cc)}"
            )
            self._is_compiled = True
            self._valid_cuda_kernels = False
            self._error_message = "shared memory usage exceed limit"

            if async_:
                return Handle([], None)
            else:
                return None

        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        # if True:
        #     tempdir = './test/'

        tempdir_obj = tempfile.TemporaryDirectory()
        tempdir = tempdir_obj.name
        saved_addr = ""
        file_id = kwargs.get("file_id", -1)
        if file_id != -1:
            print(f"file_id: {file_id}")
            saved_addr = f"./generated_codes/{file_id}/"
        FILE_NAME = os.path.join(tempdir, "test.cu")
        so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")

        FILE_NAME = "./test.cu"

        with open(FILE_NAME, "w") as f:
            f.write(result["code"] + HARD_CODE)
            if saved_addr != "":
                print(f"saved_addr: {saved_addr}")
                os.makedirs(saved_addr, exist_ok=True)
                with open(saved_addr + "test" + str(file_id) + ".cu", "w") as f:
                    f.write(result["code"] + HARD_CODE)

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
        cc_cmd = get_cc_cmd(
            target_cc,
            cc,
            FILE_NAME,
            py_include_dir,
            INCLUDE_PATH,
            DEPS_PATH,
            so_path,
            profiling,
        )

        def remain_op():
            import importlib.util

            spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.run = getattr(mod, "launch")
            if self.use_nvshmem:
                self.initialize_mpi_nvshmem = getattr(mod, "initialize_mpi_nvshmem")
                self.finalize_mpi_nvshmem = getattr(mod, "finalize_mpi_nvshmem")
                self.allocate_comm_buffers = getattr(mod, "allocate_comm_buffers")
                self.free_comm_buffers = getattr(mod, "free_comm_buffers")
            try:
                spec = importlib.util.spec_from_file_location(
                    "__mirage_launcher", so_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.run = getattr(mod, "launch")

                self._is_compiled = True
                self._valid_cuda_kernels = True
                self._cached_results = result
                self._error_message = "No error"
                tempdir_obj.cleanup()
                return self._cached_results
            except ImportError:
                # cannot import the built shared library likely due to
                # compilation errors
                self._is_compiled = True
                self._valid_cuda_kernels = False
                self._cached_results = None
                self._error_message = "CUDA compilation error"
                return None

        if async_:
            if global_config.bypass_compile_errors:
                ret = subprocess.Popen(
                    cc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
            else:
                ret = subprocess.Popen(cc_cmd)
            return Handle([ret], remain_op)
        else:
            ret = subprocess.check_call(cc_cmd)
            return remain_op()

        # so_path = './test.cpython-38-x86_64-linux-gnu.so'

    def superoptimize(
        self,
        imaps: list = None,
        omaps: list = None,
        griddims: list = None,
        blockdims: list = None,
        fmaps: list = None,
        franges: list = None,
        verbose: bool = False,
        config: str = None,
        backend: str = "cuda",
        warmup_iters: int = 16,
        profile_iters: int = 1000,
        use_graph_dataset: bool = True,
        use_cached_graphs: bool = True,
        save_codes: bool = False,
    ):
        if use_graph_dataset:
            cached_graph = graph_dataset.find(
                self.cygraph,
                imaps=imaps,
                omaps=omaps,
                griddims=griddims,
                blockdims=blockdims,
                fmaps=fmaps,
                franges=franges,
                backend=backend,
            )
            if cached_graph is not None:
                return cached_graph
        if use_cached_graphs:
            previous_checkpoint = "mirage_cached_mugraphs_{:x}.json".format(
                self.cygraph.get_owner_independent_hash()
            )
        else:
            previous_checkpoint = None
        cygraphs = search(
            self.cygraph,
            imaps=imaps,
            omaps=omaps,
            griddims=griddims,
            blockdims=blockdims,
            fmaps=fmaps,
            franges=franges,
            previous_checkpoint=previous_checkpoint,
            verbose=verbose,
            default_config=config,
        )
        all_graphs = [KNGraph(g) for g in cygraphs]
        print("Finished search, discovering {} mugraphs ...".format(len(all_graphs)))
        if backend == "cuda":
            # profile and use the best graph
            best_graph, best_perf = None, float("inf")
            print("Transpiling {} muGraphs ...".format(len(all_graphs)))
            handles = deque()

            target_cc = (
                torch.cuda.get_device_properties(0).major * 10
                + torch.cuda.get_device_properties(0).minor
            )
            if target_cc >= 90:
                pipeline_stages_list = [2, 3, 4]
                num_warp_groups_list = [2, 3, 4]
                for idx, g in enumerate(all_graphs):
                    for pipeline_stages in pipeline_stages_list:
                        for num_warp_groups in num_warp_groups_list:
                            dtensors = g.cygraph.get_input_dtensors()
                            input_tensors = list()
                            for t in dtensors:
                                dims, strides = (
                                    g.cygraph.get_input_dtensor_shape_and_stride(t)
                                )
                                dtype = convert_dtype_to_torch_type(t.dtype)
                                x = torch.randn(
                                    dims,
                                    dtype=dtype,
                                    device="cuda:{}".format(
                                        global_config.gpu_device_id
                                    ),
                                )
                                x = torch.as_strided(x, size=dims, stride=strides)
                                input_tensors.append(x)
                            starter = torch.cuda.Event(enable_timing=True)
                            ender = torch.cuda.Event(enable_timing=True)
                            new_g = g
                            if len(handles) == MAX_THREADS:
                                handles.popleft().wait()
                            handle = new_g.compile(
                                async_=True,
                                inputs=input_tensors,
                                pipeline_stages=pipeline_stages,
                                num_warp_groups=num_warp_groups,
                            )
                            handles.append(handle)
            else:
                for idx, g in enumerate(all_graphs):
                    dtensors = g.cygraph.get_input_dtensors()
                    input_tensors = list()
                    for t in dtensors:
                        dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                        # dims = [t.dim(i) for i in range(t.num_dims)]
                        dtype = convert_dtype_to_torch_type(t.dtype)
                        x = torch.randn(
                            dims,
                            dtype=dtype,
                            device="cuda:{}".format(global_config.gpu_device_id),
                        )
                        x = torch.as_strided(x, size=dims, stride=strides)
                        input_tensors.append(x)
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    if len(handles) == MAX_THREADS:
                        handles.popleft().wait()
                    handle = g.compile(async_=True, inputs=input_tensors)
                    handles.append(handle)
            while handles:
                handles.popleft().wait()
            for idx, g in enumerate(all_graphs):
                dtensors = g.cygraph.get_input_dtensors()
                input_tensors = list()
                for t in dtensors:
                    dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                    dtype = convert_dtype_to_torch_type(t.dtype)
                    x = torch.randn(
                        dims,
                        dtype=dtype,
                        device="cuda:{}".format(global_config.gpu_device_id),
                    )
                    x = torch.as_strided(x, size=dims, stride=strides)
                    input_tensors.append(x)
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                if not g.valid_kernels():
                    print("muGraph {}: {}".format(idx, g.get_error_message()))
                    continue
                # Warmup runs
                for _ in range(warmup_iters):
                    g(inputs=input_tensors)
                torch.cuda.synchronize()
                starter.record()
                for _ in range(profile_iters):
                    g(inputs=input_tensors)
                ender.record()
                torch.cuda.synchronize()
                perf = starter.elapsed_time(ender) / profile_iters
                print("muGraph {}: profiled performance (ms) = {}".format(idx, perf))
                if perf < best_perf:
                    best_graph, best_perf = g, perf
            best_graph.backend = "cuda"
            if use_graph_dataset:
                graph_dataset.store(
                    input_graph=self.cygraph,
                    optimized_graph=best_graph,
                    imaps=imaps,
                    omaps=omaps,
                    griddims=griddims,
                    blockdims=blockdims,
                    fmaps=fmaps,
                    franges=franges,
                    backend=backend,
                )
            return best_graph
        elif backend == "nki":
            return all_graphs
        elif backend == "triton":
            from .triton_profiler import profile_and_select_best_graph

            MIRAGE_ROOT, INCLUDE_PATH, _ = get_key_paths()
            os.environ["KERNELS_PATH"] = os.path.join(
                INCLUDE_PATH, "mirage/triton_transpiler/runtime"
            )  # for triton
            best_graph, best_file_path, best_output_shapes = (
                profile_and_select_best_graph(
                    all_graphs,
                    target_cc=torch.cuda.get_device_properties(0).major * 10
                    + torch.cuda.get_device_properties(0).minor,
                    warmup_iters=warmup_iters,
                    profile_iters=profile_iters,
                    debug_mode=verbose,
                    save_codes=save_codes,
                )
            )
            # load execute_mugraph func from the generated file
            print(f"Loading the best muGraph from {best_file_path}")
            if not os.path.exists(best_file_path):
                raise FileNotFoundError(f"File not found: {best_file_path}")
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "__mirage_launcher", best_file_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "execute_mugraph"):
                best_graph.run = getattr(mod, "execute_mugraph")
            else:
                raise AttributeError(
                    "The module does not contain an 'execute_mugraph' function."
                )
            best_graph._cached_results = {"output_shapes": best_output_shapes}
            best_graph.backend = "triton"
            if use_graph_dataset:
                graph_dataset.store(
                    input_graph=self.cygraph,
                    optimized_graph=best_graph,
                    imaps=imaps,
                    omaps=omaps,
                    griddims=griddims,
                    blockdims=blockdims,
                    fmaps=fmaps,
                    franges=franges,
                    backend=backend,
                )

            return best_graph
        else:
            assert False, "Unsupported backend"
            return None

    def visualize(self, file_name):
        operators = self.cygraph.get_graph_structure()
        self.visualizer = visualizer(file_name)
        self.visualizer.draw_graphs(operators)
    
    def to_json(self, filename):
        cy_to_json(self.cygraph, filename)
    
    def from_json(self, filename):
        self.cygraph = cy_from_json(filename)