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

HARD_CODE = """
#include <Python.h>

static PyObject *launch(PyObject *self, PyObject *args) {
  PyObject *input_list, *output_list, *py_buffer;
  void *buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> output_tensors;

  if (!PyArg_ParseTuple(args, "OOO", &input_list, &output_list, &py_buffer)) {
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
  execute_mugraph(input_tensors, output_tensors, buffer);

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


class KNGraph:
    def __init__(self, graph):
        self.cygraph = graph

        self._is_compiled = False
        self.run = None
        self._cached_results = None

    def new_input(
        self, dims: tuple, strides: tuple = None, dtype: dtype = float16
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
        return self.cygraph.new_input(dims, tuple(strides), dtype)

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

    def add(self, A: DTensor, B: DTensor):
        return self.cygraph.add(A, B)

    def mul(self, A: DTensor, B: DTensor):
        return self.cygraph.mul(A, B)

    def div(self, A: DTensor, B: DTensor):
        return self.cygraph.div(A, B)

    def rms_norm(self, A: DTensor, normalized_shape: tuple):
        return self.cygraph.rms_norm(A, normalized_shape)

    def customized(self, inputs: list[DTensor], bgraph: TBGraph) -> list[DTensor]:
        return self.cygraph.customized(inputs, bgraph.cygraph)

    def __call__(self, **kwargs):
        results = self.compile(**kwargs)

        assert self.run is not None, "The graph is not compiled yet."

        input_tensors = kwargs.get("inputs", [])

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
                dtype=torch.float16,
            )
            for meta in results["output_directives"]
        ]

        buffer_tensor_ptr = buffer_tensor.data_ptr()
        input_tensors_ptr = [tensor.data_ptr() for tensor in input_tensors]
        output_tensors_ptr = [tensor.data_ptr() for tensor in output_tensors]

        self.run(input_tensors_ptr, output_tensors_ptr, buffer_tensor_ptr)

        return output_tensors

    def compile(self, **kwargs):
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
            input_strides.append(self.cygraph.get_input_dtensor_layout(dtensors[i]))
        target_cc = kwargs.get(
            "target_cc",
            torch.cuda.get_device_properties(0).major * 10
            + torch.cuda.get_device_properties(0).minor,
        )

        result = generate_cuda_program(
            self.cygraph, target_cc=target_cc, input_strides=input_strides
        )
        # print(result)

        MIRAGE_ROOT = os.environ.get(
            "MIRAGE_ROOT", os.path.join(os.path.dirname(__file__), "include")
        )

        # if True:
        #     tempdir = './test/'
        with tempfile.TemporaryDirectory() as tempdir:
            FILE_NAME = os.path.join(tempdir, "test.cu")
            so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")

            with open(FILE_NAME, "w") as f:
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

            if not os.path.exists(MIRAGE_ROOT):
                print(
                    f"Error: MIRAGE_ROOT ({MIRAGE_ROOT}) not found. Please set the MIRAGE_ROOT env variable correctly"
                )
                sys.exit(1)
            cc_cmd = [
                cc,
                FILE_NAME,
                "-O3",
                f"-I{py_include_dir}",
                f"-I{MIRAGE_ROOT}/include/mirage/transpiler/runtime/",
                f"-I{MIRAGE_ROOT}/deps/cutlass/include",
                "-shared",
                "-std=c++17",
                "-arch=native",
                "-use_fast_math",
                "-lcublas",
                "-Xcompiler=-fPIC",
                "--expt-relaxed-constexpr",
                "-o",
                so_path,
            ]

            ret = subprocess.check_call(cc_cmd)
            # so_path = './test.cpython-38-x86_64-linux-gnu.so'

            import importlib.util

            spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.run = getattr(mod, "launch")

        self._is_compiled = True
        self._cached_results = result
        return self._cached_results

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
    ):
        cygraphs = search(
            self.cygraph,
            imaps=imaps,
            omaps=omaps,
            griddims=griddims,
            blockdims=blockdims,
            fmaps=fmaps,
            franges=franges,
            verbose=verbose,
            default_config=config,
        )
        all_graphs = [KNGraph(g) for g in cygraphs]

        # profile and use the best graph
        best_graph, best_perf = None, float("inf")
        for idx, g in enumerate(all_graphs):
            dtensors = g.cygraph.get_input_dtensors()
            input_tensors = list()
            for t in dtensors:
                dims = [t.dim(i) for i in range(t.num_dims)]
                input_tensors.append(
                    torch.randn(dims, dtype=torch.float16, device="cuda:0")
                )
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            print("Transpiling muGraph {}...".format(idx))
            for _ in range(16):
                g(inputs=input_tensors)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(1000):
                g(inputs=input_tensors)
            ender.record()
            torch.cuda.synchronize()
            perf = starter.elapsed_time(ender) / 1000
            print("Profiling muGraph {} performance (ms) = {}".format(idx, perf))
            if perf < best_perf:
                best_graph, best_perf = g, perf

        return best_graph
