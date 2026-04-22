import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig

from ..core import *
from ..kernel import get_key_paths, KNGraph, TBGraph
from .speculative import (
    SpecDecodeConfig,
    PromptLookupConfig,
)
from .multigpu import (
  auto_select_allreduce_implementation
)
from typing import Optional

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>

// Stubs for host symbols from libnvshmem_device.a that collective_launch.cpp.o
// references. We don't link the full device archive (it forces -rdc=true), so
// Host-side stubs for symbols normally in libnvshmem_device.a.
// We don't link the .a (it forces rdc=true → 255 regs on SM100a).
// Init is done via nvshmemid_hostlib_init_attr + our callback.
#ifdef NVSHMEM_NO_DEVICE_LIB
// Stubs for host-side symbols from libnvshmem_device.a needed by collective_launch.cpp.o
struct nvshmemi_device_only_state_stub { char data[1024]; };
nvshmemi_device_only_state_stub nvshmemi_device_only_state;
extern "C" {
  void nvshmemi_finalize() {}
  void _Z31nvshmemi_check_state_and_init_dv() {}
  void* nvshmemi_get_device_state_ptrs() { return nullptr; }
}
#endif

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer;
  std::vector<void*> meta_tensors;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers;
  int max_seq_length, total_num_requests;
  long long eos_token_id;
  int allocate_nvshmem_teams, is_test_mode;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, 
      "OOiiiiiiLii", 
      &meta_list, &py_profiler_buffer, 
      &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, 
      &eos_token_id, 
      &allocate_nvshmem_teams, &is_test_mode)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor && !is_test_mode) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id, allocate_nvshmem_teams, is_test_mode);

  Py_RETURN_NONE;
}

static PyObject *init_request_func(PyObject *self, PyObject *args) {
  Py_BEGIN_ALLOW_THREADS
  init_request_resources();
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  PyObject *py_stream;
  cudaStream_t stream;
  if (!PyArg_ParseTuple(args, "O", &py_stream)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  launch_persistent_kernel(stream);

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"init_request_func", init_request_func, METH_VARARGS, "initialize request resources"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
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

valid_persistent_kernel_modes = {"offline", "online", "online_notoken", "onepass", "online_multi_turn"}

def _detect_cxx_standard():
    """Use c++20 if the host compiler supports it, otherwise fall back to c++17."""
    try:
        result = subprocess.run(
            ["g++", "-std=c++20", "-x", "c++", "-E", "-"],
            input="", capture_output=True, text=True,
        )
        if result.returncode == 0:
            return "-std=c++20"
    except FileNotFoundError:
        pass
    return "-std=c++17"

def get_compile_command(
    mpk,
    target_cc,
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
    use_nvshmem,
    num_workers=None,
    num_local_schedulers=None,
    num_remote_schedulers=None,
    use_cutlass_kernel=True,
    test_mode=False,
):
    max_worker_per_scheduler = 128
    if num_workers != None and num_local_schedulers != None and num_remote_schedulers != None:
        min_schedulers = 0
        if num_remote_schedulers == 0:
            min_schedulers = num_local_schedulers
        else:
            min_schedulers = min(num_local_schedulers, num_remote_schedulers)
        # advance by 1 for the scheduler who are handling the not divisiable num_worker.
        max_worker_per_scheduler = (num_workers // min_schedulers) + 1

    common_cmd = [
        cc,
        # "--default-stream per-thread" is used to create new stream for 
        # each host thread as default stream instead of using the same 
        # legacy stream for all host threads
        # This is important in multi-threaded environment.
        # "--default-stream",
        # "per-thread",
        file_name,
        "-O3",
        # Use following flags when debugging
        # "-O0",
        # "-g",
        # "-G",
        # ptxas verbose: use MPK_PTXAS_VERBOSE=1 to enable (heavy perf impact on large kernels)
        *(["--ptxas-options=-v"] if os.environ.get("MPK_PTXAS_VERBOSE") == "1" else []),
        "-lineinfo",
        f"-I{py_include_dir}",
        f"-I{mirage_inc_path}",
        f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/tools/util/include')}",
        f"-I{os.path.join(mirage_deps_path, 'json/include')}",
        f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
        f"-DMIRAGE_USE_CUTLASS_KERNEL={'1' if use_cutlass_kernel else '0'}",
    ]

    flags = [
        "-shared",
        _detect_cxx_standard(),
        # Blackwell: rdc=false (self-contained allreduce, avoids 255-reg inflation)
        # Hopper/Ampere: rdc=true when NVSHMEM (needed for libnvshmem_device.a)
        "-rdc=false" if (not use_nvshmem or target_cc >= 100) else "-rdc=true",
        "-use_fast_math",
        "-lcuda",
        "-lcudart",
        "-lstdc++fs",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        py_so_path,
    ]
    flags = flags + [f"-DMPK_TARGET_CC={target_cc}", "-DMIRAGE_BACKEND_USE_CUDA"]

    if test_mode:
        flags = flags + ["-DMPK_TEST_MODE"]
    if mpk.mode == "offline":
        flags = flags + ["-DMODE_OFFLINE"]
    elif mpk.mode == "online":
        flags = flags + ["-DMODE_ONLINE"]
    elif mpk.mode == "online_notoken":
        flags = flags + ["-DMODE_ONLINE_NOTOKEN"]
    elif mpk.mode == "onepass":
        flags = flags + ["-DMODE_ONEPASS"]
    elif mpk.mode == "online_multi_turn":
        flags = flags + ["-DMODE_MULTI_TURN"]
    else:
        raise ValueError(f"Invalid persistent kernel mode: {mpk.mode}")

    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}"]
    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}"]
    flags = flags + [f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}"]
    flags = flags + [f"-DMPK_PAGE_SIZE={mpk.page_size}"]
    flags = flags + [f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}"]
    # Use when debugging
    if os.environ.get("MPK_ENABLE_VERBOSE", "0") == "1":
        flags = flags + [f"-DMPK_ENABLE_VERBOSE"]
    if os.environ.get("MPK_AR_LOCAL_COPY", "0") == "1":
        flags = flags + ["-DMPK_AR_LOCAL_COPY"]

    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
        if target_cc >= 100:
            # Blackwell: self-contained allreduce, no libnvshmem_device.a
            _dev_a = os.path.join(nvshmem_lib_path, "libnvshmem_device.a")
            _host_obj_dir = os.path.join(os.path.dirname(py_so_path), "nvshmem_host_objs")
            os.makedirs(_host_obj_dir, exist_ok=True)
            _coll_obj = os.path.join(_host_obj_dir, "collective_launch.cpp.o")
            if not os.path.exists(_coll_obj):
                import subprocess as _sp
                _sp.check_call(["ar", "x", _dev_a, "collective_launch.cpp.o"], cwd=_host_obj_dir)
            nvshmem_flags = ["-DUSE_NVSHMEM", "-DNVSHMEM_NO_DEVICE_LIB",
                             "-ccbin=mpic++", "-lnvshmem_host", "-lmpi",
                             _coll_obj,
                             "-Xlinker", "--disable-new-dtags",
                             "-Xlinker", f"-rpath", "-Xlinker", nvshmem_lib_path,
                             "-Xlinker", f"-rpath", "-Xlinker", mpi_lib_path]
        else:
            # Hopper/Ampere: standard NVSHMEM with device library + rdc=true
            nvshmem_flags = ["-DUSE_NVSHMEM",
                             "-ccbin=mpic++", "-lnvshmem_host", "-lnvshmem_device", "-lmpi",
                             "-Xlinker", "--disable-new-dtags",
                             "-Xlinker", f"-rpath", "-Xlinker", nvshmem_lib_path,
                             "-Xlinker", f"-rpath", "-Xlinker", mpi_lib_path]
        common_cmd = common_cmd + nvshmem_cmd
        flags = flags + nvshmem_flags

    if target_cc == 90:
        specific_cmd = [
            "-gencode=arch=compute_90a,code=sm_90a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_HOPPER",
            "-DNDEBUG",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target_cc == 100:
        specific_cmd = [
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_BLACKWELL",
        ]
    else:
        specific_cmd = [
            "-arch=native",
        ]
    
    if profiling:
        flags = flags + ["-DMPK_ENABLE_PROFILING"]

    return common_cmd + specific_cmd + flags


class PersistentKernel:
    def __init__(
        self,
        mode: str,
        world_size: int,
        mpi_rank: int,
        num_workers: int,
        num_local_schedulers: int,
        num_remote_schedulers: int,
        max_seq_length: int,
        max_num_batched_requests: int,
        max_num_batched_tokens: int,
        max_num_pages: int,
        page_size: int,
        meta_tensors: dict,
        profiler_tensor: torch.Tensor,
        trace_name: str,
        spec_decode_config: SpecDecodeConfig,
        use_cutlass_kernel: bool,
        eos_token_id: int64 = -1,
        test_mode: bool = False,
    ):
        self.__finalized__ = False
        self._is_compiled = False
        self.test_mode = test_mode

        if mode not in valid_persistent_kernel_modes:
            raise ValueError(f"Invalid persistent kernel mode: {mode}")
        self.mode = mode
        self.world_size = world_size
        self.mpi_rank = mpi_rank
        self.num_workers = num_workers
        self.num_local_schedulers = num_local_schedulers
        self.num_remote_schedulers = num_remote_schedulers
        self.max_seq_length = max_seq_length
        self.max_num_batched_requests = max_num_batched_requests
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.eos_token_id = eos_token_id
        self.kn_graph = KNGraph(CyKNGraph(disable_fingerprint=True))
        # Prevent GC of PyTorch tensors whose GPU pointers are baked into the
        # generated persistent-kernel code (attach_input stores raw pointers).
        self._torch_tensor_refs = []
        self.meta_tensors = meta_tensors
        # Auto-allocate scheduler snapshot buffer for in-place compaction
        if "paged_kv_indices_snapshot" not in self.meta_tensors:
            self.meta_tensors["paged_kv_indices_snapshot"] = torch.empty(
                max_num_pages, dtype=torch.int32, device="cuda")
        self.profiler_tensor = profiler_tensor
        self.trace_name = trace_name
        self.use_nvshmem = True if (world_size > 1 and os.environ.get("MPK_NO_NVSHMEM", "0") != "1") else False
        self.spec_decode_config = spec_decode_config
        self.use_cutlass_kernel = use_cutlass_kernel
        self._spec_decode_handlers = {
            "promptlookup": self.prompt_lookup_spec_handler,
        }
        self._spec_verify_handlers = {
            "promptlookup": self.prompt_lookup_verify_handler,
        }
        self.allocate_nvshmem_teams = 0
        # determine total number of requests for offline serving
        self.target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor

        if test_mode:
            # Skip all following checks
            self.total_num_requests = 1
            return

        self.total_num_requests = meta_tensors["tokens"].shape[0]

        # Checks
        assert self.max_seq_length == meta_tensors["tokens"].shape[1]
        qo_indptr_buffer = self.meta_tensors["qo_indptr_buffer"]
        # Asserts "==" below is not guaranteed by vllm, because the shape is changed depending on real situation. But the mem space won't change.
        assert qo_indptr_buffer.shape[0] <= self.max_num_batched_requests+1, f"qo_indptr_buffer.shape: {qo_indptr_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        paged_kv_indptr_buffer = self.meta_tensors["paged_kv_indptr_buffer"]
        assert paged_kv_indptr_buffer.shape[0] <= self.max_num_batched_requests+1, f"paged_kv_indptr_buffer.shape: {paged_kv_indptr_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        paged_kv_indices_buffer = self.meta_tensors["paged_kv_indices_buffer"]
        # assert paged_kv_indices_buffer.shape == (self.max_num_pages,), f"paged_kv_indices_buffer.shape: {paged_kv_indices_buffer.shape}, max_num_pages: {self.max_num_pages}"
        # TODO: This is because the paged_kv_indices_buffer can be limited by max len on vllm side
        assert paged_kv_indices_buffer.shape[0] <= self.max_num_pages, f"paged_kv_indices_buffer.shape: {paged_kv_indices_buffer.shape}, max_num_pages: {self.max_num_pages}"
        paged_kv_last_page_len_buffer = self.meta_tensors["paged_kv_last_page_len_buffer"]
        assert paged_kv_last_page_len_buffer.shape[0] <= self.max_num_batched_requests, f"paged_kv_last_page_len_buffer.shape: {paged_kv_last_page_len_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"

        # check type of meta_tensors
        assert self.meta_tensors["tokens"].dtype == torch.int64, f"tokens.dtype: {self.meta_tensors['tokens'].dtype}"
        assert self.meta_tensors["input_tokens"].dtype == torch.int64, f"input_tokens.dtype: {self.meta_tensors['input_tokens'].dtype}"
        assert self.meta_tensors["output_tokens"].dtype == torch.int64, f"output_tokens.dtype: {self.meta_tensors['output_tokens'].dtype}"
        assert self.meta_tensors["num_new_tokens"].dtype == torch.int32, f"num_new_tokens.dtype: {self.meta_tensors['num_new_tokens'].dtype}"
        assert self.meta_tensors["prompt_lengths"].dtype == torch.int32, f"prompt_lengths.dtype: {self.meta_tensors['prompt_lengths'].dtype}"
        assert qo_indptr_buffer.dtype == torch.int32, f"qo_indptr_buffer.dtype: {qo_indptr_buffer.dtype}"
        assert paged_kv_indptr_buffer.dtype == torch.int32, f"paged_kv_indptr_buffer.dtype: {paged_kv_indptr_buffer.dtype}"
        assert paged_kv_indices_buffer.dtype == torch.int32, f"paged_kv_indices_buffer.dtype: {paged_kv_indices_buffer.dtype}"
        assert paged_kv_last_page_len_buffer.dtype == torch.int32, f"paged_kv_last_page_len_buffer.dtype: {paged_kv_last_page_len_buffer.dtype}"
    
    @classmethod
    def get_default_init_parameters(cls):
        return {
            "mode": "offline",
            "world_size": 1,
            "mpi_rank": 0,
            "num_workers": 1,
            "num_local_schedulers": 4,
            "num_remote_schedulers": 0,
            "max_seq_length": 1,
            "max_num_batched_requests": 1,
            "max_num_batched_tokens": 1,
            "max_num_pages": 1,
            "page_size": 1,
            "meta_tensors": dict(),
            "profiler_tensor": None,
            "trace_name": "test_trace",
            "spec_decode_config": None,
            "use_cutlass_kernel": False,
            "eos_token_id": -1,
        }

    def attach_input(self, torch_tensor: torch.Tensor, name: str = None) -> DTensor:
        dims = tuple([d for d in torch_tensor.shape])
        strides = tuple([s for s in torch_tensor.stride()])
        # Check layout: row-major or column-major (for FP8 scale tensors)
        is_row_major = all(strides[d] == strides[d + 1] * dims[d + 1] for d in range(len(dims) - 1))
        is_col_major = len(dims) == 2 and strides[0] == 1 and strides[1] >= dims[0]
        assert is_row_major or is_col_major, \
            f"Tensor must be row-major or column-major, got dims={dims} strides={strides}"
        dtype = convert_torch_type_to_dtype(torch_tensor.dtype)
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        # Sanitize name for C++ codegen (dots are illegal in identifiers)
        safe_name = name.replace('.', '_')
        self.kn_graph.attach_torch_tensor(t, torch_tensor, safe_name)
        # Keep a reference to the PyTorch tensor so it is not garbage-collected.
        # The generated persistent kernel code stores the raw GPU data pointer;
        # if the tensor is freed, the pointer becomes dangling.
        self._torch_tensor_refs.append(torch_tensor)
        return t

    def new_tensor(
        self,
        dims: tuple,
        strides: tuple = None,
        dtype: dtype = bfloat16,
        name: str = None,
        io_category: str = "cuda_tensor",
    ) -> DTensor:
        # Assert a row-major layout
        # if strides is not None:
        #     for d in range(len(dims) - 1):
        #         assert strides[d] == strides[d + 1] * dims[d + 1]
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        safe_name = name.replace('.', '_') if name else name
        if io_category == "cuda_tensor":
            self.kn_graph.attach_cuda_tensor(t, safe_name)
        elif io_category == "nvshmem_tensor":
            self.kn_graph.attach_nvshmem_tensor(t, name)
        else:
            raise RuntimeError(f"Invalid io_category: {io_category}")
        return t

    def fuse_tensors(
        self, inputs: list[DTensor], fused_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support fusing the 0-th dimension
        assert fused_dim == 0
        t = self.kn_graph.fuse_tensors(inputs, fused_dim, num_groups, name)
        return t

    def shuffle_tensors(
        self, inputs: list[DTensor], shuffled_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support shuffling the 0-th dimension
        assert shuffled_dim == 0
        t = self.kn_graph.shuffle_tensors(inputs, shuffled_dim, num_groups, name)
        return t

    def embed_layer(
        self,
        input: DTensor, # [batch_size, num_spec_tokens]
        weight: DTensor, # [vocab_size, hidden_size]
        output: DTensor, # [batch_size, hidden_size]
        grid_dim: tuple,
        block_dim: tuple,
        input_source: int = 0, # 0: all_tokens, 1: input_token
    ):
        # TODO: Support batch size > 1
        # tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # tb_graph.new_input(input, (-1, -1, -1), -1, True)
        # tb_graph.new_input(weight, (-1, -1, -1), -1, True)
        # tb_graph.new_input(output, (-1, -1, -1), -1, True)
        # self.kn_graph.customized([input, weight, output], tb_graph)
        # self.kn_graph.register_task(tb_graph, "embedding")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(weight, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "embedding" if self.target_cc == 90 else "embedding", [input_source])

    def rmsnorm_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that the input/output are 2D tensors
        assert input.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(output, (0, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "rmsnorm_hopper" if self.target_cc >= 90 else "rmsnorm")

    def rmsnorm_linear_layer(
        self,
        input: DTensor,
        weight_norm: DTensor,
        weight_linear: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that the input/weight_linear/output are 2D tensors
        assert input.num_dims == 2
        assert weight_linear.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight_norm, (-1, -1, -1), 0, True)
        tb_graph.new_input(weight_linear, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "rmsnorm_linear")

    def attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "attention", params)
        
    def single_batch_extend_attention_layer(
        self,
        input: DTensor, # [6, 6144]
        k_cache: DTensor, 
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple, # (6, 8, 1)
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim # 32
        rotary_embed = 0
        output_stride = output.dim(1)

        extend_num = input.dim(0) - 1
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: extend_num
        # params[5]: output_stride
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, extend_num, output_stride]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "single_batch_extend_attention", params)

    def paged_attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_hopper", params)
        elif self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_sm100", params)
        else:
            self.kn_graph.register_task(tb_graph, "paged_attention", params)

    
    def paged_attention_split_kv_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        lse: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        assert output.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv * head_dim / world_size, num_kv_heads)
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        # params[6]: num_kv_chunks
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size, num_kv_chunks]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse, (-1, 2, 1), -1, True)
        tb_graph.new_input(output, (-1, 2, 1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                lse,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_sm100", params)
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_hopper", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")

    def paged_attention_split_kv_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)
        assert output_tmp.num_dims == 3  # (num_tokens, num_chunks, hidden_size / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_qo_heads_per_kv = num_q_heads / grid_dim[1]
        num_kv_heads = grid_dim[1]
        # params[0]: num_qo_heads_per_kv
        # params[1]: head_dim
        # params[2]: max_seq_len
        # params[3]: page_size
        # params[4]: num_kv_heads
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length, self.page_size, num_kv_heads]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(lse, (-1, 2, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, 2, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                lse,
                output_tmp,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_merge_sm100", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")
            
    # MLA (Multi-head Latent Attention) Layers
    def mla_kv_gather_layer(
        self,
        c_latent_new: DTensor,
        k_pe_new: DTensor,
        paged_cache: DTensor,
        contiguous_kv: DTensor,
        mla_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        d_k, d_v, page_size = mla_params
        params = [d_k, d_v, page_size]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(c_latent_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_pe_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(paged_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(contiguous_kv, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [c_latent_new, k_pe_new, paged_cache, contiguous_kv], tb_graph)
        self.kn_graph.register_task(tb_graph, "mla_kv_gather_sm100", params)

    def mla_kv_gather_split_layer(
        self,
        c_latent_new: DTensor,
        k_pe_new: DTensor,
        paged_cache: DTensor,
        ckv_sep: DTensor,     # [max_seq_len, D_V=512] output
        kpe_sep: DTensor,     # [max_seq_len, D_K-D_V=64] output
        mla_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Gather paged KV into SEPARATE CKV / KPE contiguous buffers.

        Variant of ``mla_kv_gather_layer`` that writes the gathered sequence
        to two dense tensors instead of a single concatenated [S, D_K] buffer.
        This is the layout ``mla_prefill_sm100`` expects.
        """
        d_k, d_v, page_size = mla_params
        params = [d_k, d_v, page_size]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(c_latent_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_pe_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(paged_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(ckv_sep, (-1, -1, -1), -1, True)
        tb_graph.new_input(kpe_sep, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [c_latent_new, k_pe_new, paged_cache, ckv_sep, kpe_sep], tb_graph)
        self.kn_graph.register_task(tb_graph, "mla_kv_gather_split_sm100", params)

    def mla_decode_layer(
        self,
        q_input: DTensor,         # Q tensor (attached with TMA desc)
        kv_input: DTensor,        # KV cache tensor (attached with TMA desc)
        output_partial: DTensor,  # partial O: [B*Q_LEN*sk, D_V*NUM_HEADS] float32 (or bf16)
        output_lse: DTensor,      # partial LSE: [B*Q_LEN*sk, NUM_HEADS] float32
        mla_params: tuple,        # (num_heads, d_k, d_v, num_splits, kv_len) or (..., q_len)
        grid_dim: tuple,
        block_dim: tuple,
        q_len: int = 1,
    ):
        # Allow q_len passed via mla_params 6-tuple as well as separate arg.
        if len(mla_params) == 6:
            num_heads, d_k, d_v, num_splits, kv_len, q_len = mla_params
        else:
            num_heads, d_k, d_v, num_splits, kv_len = mla_params
        params = [num_heads, d_k, d_v, num_splits, kv_len, q_len]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_input, (0, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (0, -1, -1), -1, True)
        # When q_len > 1, grid is (num_splits, num_head_groups, 1). grid.y
        # blocks read the full output buffer and offset internally via
        # block_linear (bi*num_head_groups*sk + gi*sk + si). Don't partition.
        partial_map = (-1, -1, -1) if q_len > 1 else (0, -1, -1)
        tb_graph.new_input(output_partial, partial_map, -1, True)
        tb_graph.new_input(output_lse, partial_map, -1, True)
        self.kn_graph.customized(
            [q_input, kv_input, output_partial, output_lse], tb_graph
        )
        self.kn_graph.register_task(tb_graph, "mla_decode_sm100", params)

    def mla_reduce_layer(
        self,
        input_partial: DTensor,   # partial O from decode tasks
        input_lse: DTensor,       # partial LSE from decode tasks
        output: DTensor,          # final O: [B*Q_LEN, NUM_HEADS, D_V] bf16
        mla_params: tuple,        # (num_heads, d_v, num_splits, d_start, d_count) or (..., q_len)
        grid_dim: tuple,
        block_dim: tuple,
        q_len: int = 1,
    ):
        if len(mla_params) == 6:
            num_heads, d_v, num_splits, d_start, d_count, q_len = mla_params
        else:
            num_heads, d_v, num_splits, d_start, d_count = mla_params
        params = [num_heads, d_v, num_splits, d_start, d_count, q_len]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # When q_len > 1, grid.x maps to head_group (not batch). The kernel
        # uses block_linear = bi * num_head_groups * sk + gi * sk to
        # offset into the same shared input buffer, so we must NOT partition
        # input/output along grid.x — every block needs the full base pointer.
        partial_map = (-1, -1, -1) if q_len > 1 else (0, -1, -1)
        tb_graph.new_input(input_partial, partial_map, -1, True)
        tb_graph.new_input(input_lse, partial_map, -1, True)
        tb_graph.new_input(output, partial_map, -1, True)
        self.kn_graph.customized(
            [input_partial, input_lse, output], tb_graph
        )
        self.kn_graph.register_task(tb_graph, "mla_reduce_sm100", params)

    def mla_prefill_layer(
        self,
        q_nope: DTensor,   # [S, H, D_CKV]
        q_pe: DTensor,     # [S, H, D_KPE]
        ckv: DTensor,      # [S, D_CKV]
        kpe: DTensor,      # [S, D_KPE]
        output: DTensor,   # [S, H, D_V]
        mla_params: tuple, # (num_heads, seq_len, d_ckv, d_kpe, d_v)
        grid_dim: tuple,   # (H, num_q_blocks, B)
        block_dim: tuple,  # (256, 1, 1)
    ):
        num_heads, seq_len, d_ckv, d_kpe, d_v = mla_params
        params = [num_heads, seq_len, d_ckv, d_kpe, d_v]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Kernel reads based on task_metadata.{request_id=head, kv_idx=q_block}
        # and computes its own (S, H, D) offsets, so MPK must NOT try to
        # auto-partition dim 0 by grid.x (grid.x is H, not S). Use -1 on all
        # dims → full barrier event semantics.
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(ckv, (-1, -1, -1), -1, True)
        tb_graph.new_input(kpe, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_nope, q_pe, ckv, kpe, output], tb_graph
        )
        self.kn_graph.register_task(tb_graph, "mla_prefill_sm100", params)

    def mla_mtp_decode_layer(
        self,
        q_input: DTensor,          # Q tensor [B*Q_LEN*H, D_K] (with TMA desc)
        kv_input: DTensor,         # KV tensor [B*KL, D_K] (with TMA desc)
        output_partial: DTensor,   # Oa: partial output buffer
        output_lse: DTensor,       # La: partial LSE buffer
        q_len: int,
        kv_len: int,
    ):
        # Derive internal params (DeepSeek V3: 128 heads, TILE_S=128)
        hpb = 128 // q_len
        while 128 % hpb != 0:
            hpb -= 1
        num_head_groups = 128 // hpb
        num_splits = (kv_len + 128 - 1) // 128

        params = [num_head_groups, q_len, kv_len, num_splits]
        grid_dim = (num_splits, num_head_groups, 1)  # (sk, groups, B=1)
        block_dim = (128, 1, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_input, (0, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (0, -1, -1), -1, True)
        tb_graph.new_input(output_partial, (0, -1, -1), -1, True)
        tb_graph.new_input(output_lse, (0, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_input, kv_input, output_partial, output_lse], tb_graph
        )
        self.kn_graph.register_task(tb_graph, "mla_mtp_decode_sm100", params)

    def mla_mtp_reduce_layer(
        self,
        input_partial: DTensor,    # Oa from decode tasks
        input_lse: DTensor,        # La from decode tasks
        output: DTensor,           # final O [B, Q_LEN, H, D_V]
        q_len: int,
        kv_len: int,
    ):
        hpb = 128 // q_len
        while 128 % hpb != 0:
            hpb -= 1
        num_head_groups = 128 // hpb
        num_splits = (kv_len + 128 - 1) // 128
        d_v = 512
        # TODO: rd_dv=2 gives 256-1024 reduce blocks (many small tasks in MPK).
        # Consider rd_dv=4 with loop to halve block count, but benchmarked slower.
        # Revisit after MPK runtime refactor when task dispatch overhead is known.
        rd_dv = 2

        params = [num_head_groups, q_len, num_splits, rd_dv]
        grid_dim = ((d_v + rd_dv - 1) // rd_dv, num_head_groups, 1)
        block_dim = (256, 1, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_partial, (0, -1, -1), -1, True)
        tb_graph.new_input(input_lse, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized(
            [input_partial, input_lse, output], tb_graph
        )
        self.kn_graph.register_task(tb_graph, "mla_mtp_reduce_sm100", params)

    # ─────────── MLA-MTP TP variants (ferret-derived, no PDL) ───────────
    # Shape: NUM_HEADS = 128/TP per rank, D_K=576, D_V=512
    # Three variants (TP=2/4/8) — each is a (decode + reduce) pair.

    def _mla_mtp_decode_tp_layer(
        self,
        q_input, kv_input, output_partial, output_lse,
        q_len, kv_len, num_heads,
        task_name, has_v_split=False, q_len_real=None,
    ):
        """Internal helper for TP=2/4/8 decode dispatch.
          q_len: padded Q_LEN passed to the kernel
          q_len_real: TP=8 only — actual unpadded Q_LEN
          num_heads: 64/32/16 per TP variant
          has_v_split: TP=4 only — block_x doubled to encode v_half in low bit
        """
        if num_heads == 64:
            qpg = min(2, q_len)
        elif num_heads == 32:
            qpg = min(4, q_len)
        else:  # TP=8
            qpg = 2
        num_groups = (q_len + qpg - 1) // qpg
        num_splits = (kv_len + 128 - 1) // 128  # TILE_S=128
        # TP=4 packs v_half into block_x low bit → 2× tasks. Kernel unpacks.
        x_mul = 2 if has_v_split else 1
        grid_dim = (num_groups * num_splits * x_mul, 1, 1)
        block_dim = (128, 1, 1)

        if num_heads == 16:  # TP=8
            params = [num_groups, q_len, kv_len, num_splits,
                      q_len_real if q_len_real is not None else q_len]
        else:  # TP=2 and TP=4
            params = [num_groups, q_len, kv_len, num_splits]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_lse, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_input, kv_input, output_partial, output_lse], tb_graph
        )
        self.kn_graph.register_task(tb_graph, task_name, params)

    def _mla_mtp_reduce_tp_layer(
        self,
        input_partial, input_lse, output,
        q_len, kv_len, num_heads, task_name,
    ):
        if num_heads == 64:
            qpg = min(2, q_len)
        elif num_heads == 32:
            qpg = min(4, q_len)
        else:
            qpg = 2
        num_groups = (q_len + qpg - 1) // qpg
        num_splits = (kv_len + 128 - 1) // 128
        d_v = 512
        rd_dv = 2

        params = [num_groups, q_len, num_splits, rd_dv]
        grid_dim = ((d_v + rd_dv - 1) // rd_dv, num_groups, 1)
        block_dim = (256, 1, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input_partial, input_lse, output], tb_graph
        )
        self.kn_graph.register_task(tb_graph, task_name, params)

    def mla_mtp_decode_tp2_layer(
        self, q_input, kv_input, output_partial, output_lse, q_len, kv_len,
    ):
        self._mla_mtp_decode_tp_layer(
            q_input, kv_input, output_partial, output_lse,
            q_len, kv_len, num_heads=64,
            task_name="mla_mtp_decode_tp2_sm100",
        )

    def mla_mtp_decode_tp2_reduce_layer(
        self, input_partial, input_lse, output, q_len, kv_len,
    ):
        self._mla_mtp_reduce_tp_layer(
            input_partial, input_lse, output, q_len, kv_len, num_heads=64,
            task_name="mla_mtp_decode_tp2_reduce_sm100",
        )

    def mla_mtp_decode_tp4_layer(
        self, q_input, kv_input, output_partial, output_lse, q_len, kv_len,
    ):
        # TP=4 V-split: 2× tasks (v_half=0,1). Each writes to a disjoint TMEM
        # column range; output_partial is a single buffer covering both.
        self._mla_mtp_decode_tp_layer(
            q_input, kv_input, output_partial, output_lse,
            q_len, kv_len, num_heads=32,
            task_name="mla_mtp_decode_tp4_sm100", has_v_split=True,
        )

    def mla_mtp_decode_tp4_reduce_layer(
        self, input_partial, input_lse, output, q_len, kv_len,
    ):
        self._mla_mtp_reduce_tp_layer(
            input_partial, input_lse, output, q_len, kv_len, num_heads=32,
            task_name="mla_mtp_decode_tp4_reduce_sm100",
        )

    def mla_mtp_decode_tp8_layer(
        self, q_input, kv_input, output_partial, output_lse,
        q_len_real, kv_len,
    ):
        # TP=8 pads Q_LEN to even
        q_len = (q_len_real + 1) & ~1
        self._mla_mtp_decode_tp_layer(
            q_input, kv_input, output_partial, output_lse,
            q_len, kv_len, num_heads=16,
            task_name="mla_mtp_decode_tp8_sm100", q_len_real=q_len_real,
        )

    def mla_mtp_decode_tp8_reduce_layer(
        self, input_partial, input_lse, output, q_len_real, kv_len,
    ):
        q_len = (q_len_real + 1) & ~1
        self._mla_mtp_reduce_tp_layer(
            input_partial, input_lse, output, q_len, kv_len, num_heads=16,
            task_name="mla_mtp_decode_tp8_reduce_sm100",
        )

    # MoE Layers
    def tensor_init_layer(
        self,
        input: DTensor,
        dummy_input: DTensor,
        dummy_output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that output
        assert input.num_dims == 2  # (batch_size, output_size)
        assert dummy_input.num_dims == 2 # (batch_size, hidden_size)
        assert dummy_output.num_dims == 2 # (batch_size, output_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_output, (0, -1, -1), -1, True)
        self.kn_graph.customized([input, dummy_input, dummy_output], tb_graph)

        self.kn_graph.register_task(tb_graph, "tensor_init")
    
    def moe_topk_softmax_routing_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, num_experts)
        assert len(output) == 3
        moe_topk_weight, moe_routing_indices, moe_masks = output
        assert moe_topk_weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
        assert moe_masks.num_dims == 1  # (num_experts + 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_masks, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, moe_topk_weight, moe_routing_indices, moe_masks], tb_graph)

        self.kn_graph.register_task(tb_graph, "moe_topk_softmax_sm100")

    def moe_topk_sigmoid_routing_layer(
        self,
        input: DTensor,
        bias: DTensor,
        output: tuple[DTensor, DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
        num_groups: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
    ):
        import struct

        assert input.num_dims == 2  # (batch_size, num_experts)
        assert bias.num_dims == 1  # (num_experts,)
        assert len(output) == 3
        moe_topk_weight, moe_routing_indices, moe_masks = output
        assert moe_topk_weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
        assert moe_masks.num_dims == 1  # (num_experts + 1)

        scaling_bits = struct.unpack("i", struct.pack("f", routed_scaling_factor))[0]
        params = [num_groups, topk_group, scaling_bits]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_masks, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, bias, moe_topk_weight, moe_routing_indices, moe_masks],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "moe_topk_sigmoid_sm100", params)

    def moe_w13_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 3  # (num_experts, 2*intermediate_size, hidden_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, 2*intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm90")
        else:
            assert False
            
    def moe_w13_fp8_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # input_fp8:           (batch_size, hidden_size)          FP8 E4M3
        # input_scale:         (batch_size, hidden_size//128)     float32
        # weight_fp8:          (num_experts, 2*intermediate_size, hidden_size)  FP8 E4M3
        # weight_scale:        (num_experts, 2*intermediate_size, hidden_size//128)  float32
        # moe_routing_indices: (num_experts, batch_size)  int32, expert-major
        # moe_mask:            (num_experts + 1,)         int32  1-index, not 0-index!
        # output:              (batch_size, num_experts_per_tok, 2*intermediate_size)  BF16
        # The scale factor is fixed to 128.
        assert input_fp8.num_dims == 2
        assert input_scale.num_dims == 2
        assert weight_fp8.num_dims == 3
        assert weight_scale.num_dims == 3
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Note: store_in_dmem=True for all inputs to work around a TBGraph
        # segfault with 3D tensors when store_in_dmem=False.
        tb_graph.new_input(input_fp8,           (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale,         (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8,          (-1, 1, -1),  -1, True)
        tb_graph.new_input(weight_scale,        (-1, 1, -1),  -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask,            (-1, -1, -1), -1, True)
        tb_graph.new_input(output,              (-1, 2, -1),  -1, True)
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale,
             moe_routing_indices, moe_mask, output], tb_graph)
        assert self.target_cc == 100, "FP8 group GEMM requires SM100 (Blackwell)"
        self.kn_graph.register_task(tb_graph, "moe_w13_fp8_sm100")

    def moe_w2_fp8_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # input_fp8:           (batch_size, num_experts_per_tok, intermediate_size)  FP8 E4M3
        # input_scale:         (batch_size, num_experts_per_tok, intermediate_size//128)  float32
        # weight_fp8:          (num_experts, hidden_size, intermediate_size)  FP8 E4M3
        # weight_scale:        (num_experts, hidden_size, intermediate_size//128)  float32
        # moe_routing_indices: (num_experts, batch_size)  int32, expert-major
        # moe_mask:            (num_experts + 1,)         int32
        # output:              (batch_size, num_experts_per_tok, hidden_size)  BF16
        assert input_fp8.num_dims == 3
        assert input_scale.num_dims == 3
        assert weight_fp8.num_dims == 3
        assert weight_scale.num_dims == 3
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Note: store_in_dmem=True for all inputs to work around a TBGraph
        # segfault with 3D tensors when store_in_dmem=False.
        tb_graph.new_input(input_fp8,           (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale,         (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8,          (-1, 1, -1),  -1, True)
        tb_graph.new_input(weight_scale,        (-1, 1, -1),  -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask,            (-1, -1, -1), -1, True)
        tb_graph.new_input(output,              (-1, 2, -1),  -1, True)
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale,
             moe_routing_indices, moe_mask, output], tb_graph)
        assert self.target_cc == 100, "FP8 group GEMM requires SM100 (Blackwell)"
        self.kn_graph.register_task(tb_graph, "moe_w2_fp8_sm100")

    # === FP8 Dense Layers ===
    def quantize_fp8_layer(
        self,
        input: DTensor,
        output_fp8: DTensor,
        output_scale: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        scale_ue8m0: bool = True,
    ):
        """Quantize BF16 input to FP8 with block-wise scale.

        scale_ue8m0=True: output scale is packed UE8M0 uint32 (for FP8 linear GEMM)
        scale_ue8m0=False: output scale is float32 (for MoE group GEMM)
        """
        params = []
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_scale, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, output_fp8, output_scale], tb_graph)
        task_name = "quantize_fp8_sm100" if scale_ue8m0 else "quantize_fp8_f32scale_sm100"
        self.kn_graph.register_task(tb_graph, task_name, params)

    def linear_fp8_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        params = []
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Grid partitions output along dim0 (output_size): each block handles 128 rows
        # input_fp8 and input_scale: not partitioned (all blocks read same input)
        # weight_fp8: partitioned along dim0 by grid.x (output dim)
        # weight_scale: partitioned along dim1 by grid.x (output dim, stored as [pk, aligned_M])
        # output: partitioned along dim1 by grid.x
        # Grid partitions: weight dim0=output, scale dim0=M (column-major), output dim1
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8, (0, -1, -1), -1, True)    # grid.x splits dim0 (output)
        tb_graph.new_input(weight_scale, (0, -1, -1), -1, True)  # grid.x splits dim0 (M=output, col-major)
        tb_graph.new_input(output, (1, -1, -1), -1, True)        # grid.x splits dim1 (output)
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear_fp8_sm100", params)

    def linear_fp8_with_residual_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        params = [1]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8, (0, -1, -1), -1, True)    # grid.x splits dim0
        tb_graph.new_input(weight_scale, (0, -1, -1), -1, True)  # grid.x splits dim0 (col-major M)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)      # grid.x splits dim1
        tb_graph.new_input(output, (1, -1, -1), -1, True)        # grid.x splits dim1
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale, residual, output],
            tb_graph)
        self.kn_graph.register_task(
            tb_graph, "linear_fp8_with_residual_sm100", params)

    def moe_silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3 # (batch_size, num_expert_per_tok, 2 * intermediate_size)
        assert output.num_dims == 3 # (batch_size, num_expert_per_tok, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_silu_mul")
            
    def moe_w2_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor, 
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_expert_per_tok, intermediate_size)
        assert weight.num_dims == 3  # (num_experts, hidden_size, intermediate_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm90")
        else:
            assert False
        
    def moe_mul_sum_add_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_experts_per_tok, hidden_size)
        assert weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 2, -1), -1, True)
        tb_graph.new_input(weight, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)

        self.kn_graph.register_task(tb_graph, "moe_mul_sum_add_sm100")

    def splitk_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), 1, True)
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "splitk_linear_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "splitk_linear_swapAB_hopper")
        else:
            assert False

    def linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc >= 100 and self.target_cc < 120:
            self.kn_graph.register_task(tb_graph, "linear_sm100")
        elif self.target_cc >= 90 and self.target_cc < 100:
            if weight.dim(0) // grid_dim[0] <= 64:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_hopper")
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
        elif self.target_cc >= 80 and self.target_cc < 90:
            self.kn_graph.register_task(tb_graph, "linear")
        else:
            assert False, f"Unsupported compute capability: {self.target_cc}"

    def linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        
        params = []
        enable_residual = 1
        if self.world_size > 1 and self.mpi_rank != 0:
            enable_residual = 0
        params.append(enable_residual)
        if self.target_cc >= 100 and self.target_cc < 120:
            self.kn_graph.register_task(tb_graph, "linear_with_residual_sm100", params)
        elif self.target_cc >= 90 and self.target_cc < 100:
            if weight.dim(0) // grid_dim[0] <= 64:
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_with_residual_hopper")
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper", params)
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper", params)
        elif self.target_cc >= 80 and self.target_cc < 90:
            self.kn_graph.register_task(tb_graph, "linear_with_residual")
        else:
            assert False, f"Unsupported compute capability: {self.target_cc}"

    def allreduce_layer(
        self,
        input: DTensor,
        buffer: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # MPK_SKIP_ALLREDUCE=1: debug mode — skip allreduce to isolate crashes
        if os.environ.get("MPK_SKIP_ALLREDUCE", "0") == "1":
            return
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size)
        assert buffer.num_dims == 3  # (world_size, batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        # params[0]: num_gpus
        # params[1]: my_gpu_id
        best_implementation = auto_select_allreduce_implementation(self.world_size, self.mpi_rank)
        tensors = {
            "input": input,
            "buffer": buffer,
            "output": output,
        }
        params = [self.world_size, self.mpi_rank]
        best_implementation.register_tasks(self, tensors=tensors, grid_dim=grid_dim,
                                           block_dim=block_dim, params=params)


    def silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2 # (batch_size, 2 * intermediate_size)
        assert output.num_dims == 2 # (batch_size, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul" if self.target_cc == 90 else "silu_mul")

    def identity_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        dependent_tensor: DTensor = None,
    ):
        # TODO: Add support from kn_graph
        last_dim = 0
        assert input.num_dims == output.num_dims
        for i in range(input.num_dims):
            assert input.dim(i) == output.dim(i)
            last_dim = i
        assert last_dim == 1 or last_dim == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (last_dim, -1, -1), 1, True)
        tb_graph.new_input(output, (last_dim, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "identity")

    def elementwise_add_layer(
        self,
        input_a: DTensor,
        input_b: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Element-wise add: output = input_a + input_b.
        Used for residual connections when fused with_residual kernels are broken."""
        assert input_a.num_dims == 2
        assert input_b.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_a, (0, -1, -1), -1, True)
        tb_graph.new_input(input_b, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized([input_a, input_b, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "elementwise_add_sm100")

    def silu_mul_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, 2*intermediate_size)
        assert weight.num_dims == 2  # (hidden_size, intermediate_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul_linear_with_residual")

    def argmax_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "argmax")

    def argmax_partial_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert len(output) == 2
        output_value, output_index = output
        assert output_value.num_dims == 2  # (batch_size, num_tasks)
        assert output_index.num_dims == 2  # (batch_size, num_tasks)
        num_tasks = grid_dim[0]
        self.argmax_partial_output_size = input.dim(1) // num_tasks
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, 0, -1), -1, True)
        tb_graph.new_input(output_value, (1, 0, -1), -1, True)
        tb_graph.new_input(output_index, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, output_value, output_index], tb_graph)
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "argmax_partial_sm100", [num_tasks])
        else:
            self.kn_graph.register_task(tb_graph, "argmax_partial", [num_tasks])

    def argmax_reduce_layer(
        self,
        input: tuple[DTensor, DTensor],
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert len(input) == 2
        input_value, input_index = input
        assert input_value.num_dims == 2  # (batch_size, num_tasks)
        assert input_index.num_dims == 2  # (batch_size, num_tasks)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_value, (1, 0, -1), -1, True)
        tb_graph.new_input(input_index, (1, 0, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True) #TODO: Make sure the output map is expected
        self.kn_graph.customized([input_value, input_index, output], tb_graph)
        if self.target_cc == 100:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce_sm100", [self.argmax_partial_output_size]
            )
        else:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce", [self.argmax_partial_output_size]
            )

    def sampling_sm100_layer(
        self,
        logits: DTensor,      # [batch_size, vocab_size]
        output: DTensor,      # [batch_size, 1]
        grid_dim: tuple,
        block_dim: tuple,
        seed: int = 42,
    ):
        """Sampling from logits using Gumbel-Max trick for stochastic token generation."""
        assert logits.num_dims == 2      # (batch_size, vocab_size)
        assert output.num_dims == 2      # (batch_size, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized([logits, output], tb_graph)

        # Register task with seed parameter
        self.kn_graph.register_task(tb_graph, "sampling_sm100", [seed])

    def find_ngram_partial_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, seq_len)
        assert output.num_dims == 2  # (batch_size, num_tasks)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_partial", [ngram_size])
        
    def find_ngram_global_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3, spec_length: int = 5):
        # Currently assume that input/output
        assert len(input) == 2
        partial_results, tokens = input
        assert partial_results.num_dims == 2  # (batch_size, num_tasks)
        assert tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial_results, (-1, -1, -1), -1, True)
        tb_graph.new_input(tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([partial_results, tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_global", [ngram_size, spec_length])

    def prompt_lookup_spec_handler(
        self, 
        spec_decode_config: PromptLookupConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        partial_ngram_output = self.new_tensor(
            dims=(tokens.dim(0), 96),
            dtype=int64,
            name="partial_ngram_output",
            io_category="cuda_tensor",
        )
        self.find_ngram_partial_layer(
            input=tokens, 
            output=partial_ngram_output, 
            grid_dim=grid_dim, 
            block_dim=block_dim, 
            ngram_size=spec_decode_config.ngram_size
        )
        spec_tokens = self.new_tensor(
            dims=(tokens.dim(0), spec_decode_config.spec_length + 1),
            dtype=int64,
            name="spec_tokens",
            io_category="cuda_tensor",
        )   
        self.find_ngram_global_layer(
            input=(partial_ngram_output, tokens), 
            output=spec_tokens, 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1), 
            ngram_size=spec_decode_config.ngram_size,
            spec_length=spec_decode_config.spec_length
        )
        return spec_tokens
    
    def draft_forward_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        method = spec_decode_config.method
        handler = self._spec_decode_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, tokens, grid_dim, block_dim)
    
    def target_verify_greedy_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple):
        # Currently assume that input/output
        # This tensor is not realy used
        assert len(input) == 2
        spec_tokens, target_tokens = input
        assert spec_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert target_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(spec_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([spec_tokens, target_tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "target_verify_greedy")
        
    def prompt_lookup_verify_handler(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        # This tensor is not realy used
        verify_out = self.new_tensor(
            dims=(1, 1),
            dtype=int64,
            name="verify_out",
            io_category="cuda_tensor",
        )
        self.target_verify_greedy_layer(
            input=(spec_tokens, target_output), output=verify_out, grid_dim=grid_dim, block_dim=block_dim
        )
        return verify_out
    
    def verify_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int] = (1, 1, 1),
        block_dim: tuple[int, int, int] = (128, 1, 1),
    ):
        method = spec_decode_config.method
        handler = self._spec_verify_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, spec_tokens, target_output, grid_dim, block_dim)

    # === MTP (Multi-Token Prediction) Layers ===
    def mtp_token_scatter_layer(
        self,
        src: DTensor,
        dst: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        batch_size: int,
        num_slots: int,
        slot_idx: int,
    ):
        params = [batch_size, num_slots, slot_idx]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(src, (-1, -1, -1), -1, True)
        tb_graph.new_input(dst, (-1, -1, -1), -1, True)
        self.kn_graph.customized([src, dst], tb_graph)
        self.kn_graph.register_task(tb_graph, "mtp_token_scatter", params)

    def mtp_prepare_verify_layer(
        self,
        main_token: DTensor,
        draft_tokens: DTensor,
        tokens_buffer: DTensor,
        step: DTensor,
        num_new_tokens: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        num_draft_tokens: int,
        max_seq_len: int,
    ):
        params = [num_draft_tokens, max_seq_len]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(main_token, (-1, -1, -1), -1, True)
        tb_graph.new_input(draft_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(tokens_buffer, (-1, -1, -1), -1, True)
        tb_graph.new_input(step, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_new_tokens, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [main_token, draft_tokens, tokens_buffer, step, num_new_tokens], tb_graph)
        self.kn_graph.register_task(tb_graph, "mtp_prepare_verify", params)

    def mtp_build_embed_input_layer(
        self,
        output_tokens: DTensor,       # [mbt, 1] int64 — main model's argmax
        mtp_input_tokens: DTensor,    # [mbt, 1] int64 — MTP embed input (written)
        grid_dim: tuple,
        block_dim: tuple,
        batch_size: int,
        max_seq_len: int,
    ):
        """Build MTP's per-iteration embedding input token buffer.
        vLLM-aligned (eagle.py L666-669): positions [0..mbt-2] read from shifted
        ground-truth prompt tokens (`runtime_config.tokens[step[0] + i + 1]`),
        position mbt-1 reads from `output_tokens[mbt-1]` (current iter's argmax).
        `tokens` buffer and `step` are read via runtime_config, not attached.
        """
        params = [batch_size, max_seq_len]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(mtp_input_tokens, (-1, -1, -1), -1, True)
        self.kn_graph.customized([output_tokens, mtp_input_tokens], tb_graph)
        self.kn_graph.register_task(tb_graph, "mtp_build_embed_input", params)

    def softmax_gather_layer(
        self,
        logits: DTensor,          # [batch, vocab_size] BF16
        token_ids: DTensor,       # [batch, 1] int64
        output_probs: DTensor,    # [batch, 1] float32
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Fused softmax + gather: output[b] = softmax(logits[b])[token_id[b]]."""
        assert logits.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (-1, -1, -1), -1, True)
        tb_graph.new_input(token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_probs, (-1, -1, -1), -1, True)
        self.kn_graph.customized([logits, token_ids, output_probs], tb_graph)
        self.kn_graph.register_task(tb_graph, "softmax_gather_sm100")

    def mtp_float_scatter_layer(
        self,
        src: DTensor,       # [batch, 1] float32
        dst: DTensor,       # [batch, num_slots] float32
        grid_dim: tuple,
        block_dim: tuple,
        batch_size: int,
        num_slots: int,
        slot_idx: int,
    ):
        """Copy single float value to specific slot in buffer (compile-time index)."""
        params = [batch_size, num_slots, slot_idx]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(src, (-1, -1, -1), -1, True)
        tb_graph.new_input(dst, (-1, -1, -1), -1, True)
        self.kn_graph.customized([src, dst], tb_graph)
        self.kn_graph.register_task(tb_graph, "mtp_float_scatter", params)

    def prob_scatter_layer(
        self,
        prob: DTensor,           # [batch, 1] float32
        step_counter: DTensor,   # [batch] int32 (runtime step position)
        buffer: DTensor,         # [batch, max_positions] float32
        grid_dim: tuple,
        block_dim: tuple,
        max_positions: int,
    ):
        """Scatter current prob into per-position buffer at runtime step position."""
        params = [max_positions]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(prob, (-1, -1, -1), -1, True)
        tb_graph.new_input(step_counter, (-1, -1, -1), -1, True)
        tb_graph.new_input(buffer, (-1, -1, -1), -1, True)
        self.kn_graph.customized([prob, step_counter, buffer], tb_graph)
        self.kn_graph.register_task(tb_graph, "prob_scatter_sm100", params)

    def prob_extract_layer(
        self,
        buffer: DTensor,         # [batch, max_positions] float32
        offset: DTensor,         # [batch] int32 (runtime offset)
        output: DTensor,         # [batch, num_extract] float32
        grid_dim: tuple,
        block_dim: tuple,
        max_positions: int,
        num_extract: int,
    ):
        """Extract buffer[batch, offset+1..offset+num_extract] into contiguous output."""
        params = [max_positions, num_extract]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(buffer, (-1, -1, -1), -1, True)
        tb_graph.new_input(offset, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([buffer, offset, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "prob_extract_sm100", params)

    def mtp_verify_probabilistic_layer(
        self,
        draft_token_ids: DTensor,
        target_token_ids: DTensor,
        target_probs: DTensor,
        draft_probs: DTensor,
        seed: DTensor,
        accepted_count: DTensor,
        output_tokens: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        num_draft_tokens: int,
    ):
        """Probabilistic verification: accept if P_target > u * P_draft."""
        params = [num_draft_tokens]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(draft_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_probs, (-1, -1, -1), -1, True)
        tb_graph.new_input(draft_probs, (-1, -1, -1), -1, True)
        tb_graph.new_input(seed, (-1, -1, -1), -1, True)
        tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [draft_token_ids, target_token_ids, target_probs, draft_probs,
             seed, accepted_count, output_tokens], tb_graph)
        self.kn_graph.register_task(tb_graph, "mtp_verify_probabilistic", params)

    def mtp_verify_strict_layer(
        self,
        draft_token_ids: DTensor,
        target_token_ids: DTensor,
        accepted_count: DTensor,
        output_tokens: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        num_draft_tokens: int,
    ):
        params = [num_draft_tokens]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(draft_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [draft_token_ids, target_token_ids, accepted_count, output_tokens], tb_graph)
        self.kn_graph.register_task(tb_graph, "mtp_verify_strict", params)

    def mtp_accept_commit_layer(
        self,
        accepted_count: DTensor,
        output_tokens: DTensor,
        current_position: DTensor,
        new_position: DTensor,
        final_output: DTensor,
        num_new_tokens: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        num_draft_tokens: int,
    ):
        params = [num_draft_tokens]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(current_position, (-1, -1, -1), -1, True)
        tb_graph.new_input(new_position, (-1, -1, -1), -1, True)
        tb_graph.new_input(final_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_new_tokens, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [accepted_count, output_tokens, current_position,
             new_position, final_output, num_new_tokens], tb_graph)
        self.kn_graph.register_task(tb_graph, "mtp_accept_commit", params)

    def compile(
        self,
        **kwargs,
    ):
        assert not self._is_compiled
        
        output_dir = kwargs.get("output_dir", None)

        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        if self.mode == "online_notoken" or self.mode == "online" or self.mode == "multi_turn":
            # We will init for multiple times so the output directory should be permanent
            tempdir = "./permanent_output_dir/"
        else:
            tempdir_obj = tempfile.TemporaryDirectory()
            tempdir = tempdir_obj.name
        os.makedirs(tempdir, exist_ok=True)
        results = self.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.mpi_rank)

        cuda_code_path = os.path.join(tempdir, "test.cu")
        so_path = os.path.join(tempdir, "test" + sysconfig.get_config_var("EXT_SUFFIX"))
        # check json file
        json_file_path = os.path.join(tempdir, "task_graph.json")
        # build if files are not exist
            
        with open(json_file_path, "w") as f:
            f.write(results["json_file"])
        hard_code = HARD_CODE
        with open(cuda_code_path, "w") as f:
            f.write(results["cuda_code"] + hard_code)
            
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(cuda_code_path, os.path.join(output_dir, f"test_rank{self.mpi_rank}.cu"))
            shutil.copy(json_file_path, os.path.join(output_dir, f"task_graph_rank{self.mpi_rank}.json"))

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

        # find mirage home (fall back to MIRAGE_ROOT from get_key_paths)
        MIRAGE_HOME_PATH = os.environ.get("MIRAGE_HOME", MIRAGE_ROOT)

        NVSHMEM_INC_PATH = None
        NVSHMEM_LIB_PATH = None
        MPI_INC_PATH = None
        MPI_LIB_PATH = None
        if self.use_nvshmem:
            # find nvshmem include folder and library folder
            if "NVSHMEM_INC_PATH" in os.environ:
                NVSHMEM_INC_PATH = os.environ.get("NVSHMEM_INC_PATH")
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_INC_PATH is set but cannot find nvshmem.h at {header_file_path}"
                    )
            else:
                NVSHMEM_INC_PATH = "/usr/include/nvshmem_12/"
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Cannot find nvshmem.h, please set environment variable NVSHMEM_INC_PATH"
                    )
            # find nvshmem shared library
            if "NVSHMEM_LIB_PATH" in os.environ:
                NVSHMEM_LIB_PATH = os.environ.get("NVSHMEM_LIB_PATH")
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem_device.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_LIB_PATH is set but cannot find libnvshmem_device.a at {lib_file_path}"
                        " MPK requires NVSHMEM >= 3.5.19"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/x86_64-linux-gnu/"
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem_device.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Cannot find libnvshmem_device.a, please set environment variable NVSHMEM_LIB_PATH"
                        " MPK requires NVSHMEM >= 3.5.19"
                    )
            # find mpi include foler
            if "MPI_INC_PATH" in os.environ:
                MPI_INC_PATH = os.environ.get("MPI_INC_PATH")
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_INC_PATH is set but cannot find mpi.h at {header_file_path}"
                    )
            else:
                MPI_INC_PATH = "/usr/include/"
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Cannot find mpi.h, please set environment variable MPI_INC_PATH"
                    )
            # find mpi shared library
            if "MPI_LIB_PATH" in os.environ:
                MPI_LIB_PATH = os.environ.get("MPI_LIB_PATH")
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_LIB_PATH is set but cannot find libmpi.so at {lib_file_path}"
                    )
            else:
                MPI_LIB_PATH = "/usr/lib/"
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Cannot find libmpi.so, please set environment variable MPI_LIB_PATH"
                    )

        cc_cmd = get_compile_command(
            mpk=self,
            target_cc=self.target_cc,
            cc=cc,
            file_name=cuda_code_path,
            py_include_dir=py_include_dir,
            mirage_home_path=MIRAGE_HOME_PATH,
            mirage_inc_path=INCLUDE_PATH,
            mirage_deps_path=DEPS_PATH,
            nvshmem_inc_path=NVSHMEM_INC_PATH,
            nvshmem_lib_path=NVSHMEM_LIB_PATH,
            mpi_inc_path=MPI_INC_PATH,
            mpi_lib_path=MPI_LIB_PATH,
            py_so_path=so_path,
            profiling=True if self.profiler_tensor is not None else False,
            use_nvshmem=self.use_nvshmem,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_local_schedulers,
            num_remote_schedulers=self.num_remote_schedulers,
            use_cutlass_kernel=self.use_cutlass_kernel,
            test_mode=self.test_mode,
        )
        precompiled_so = os.environ.get("MPK_PRECOMPILED_SO")
        if precompiled_so and os.path.exists(precompiled_so):
            shutil.copy(precompiled_so, so_path)
            # Also copy task_graph.json to the directory where __FILE__ points
            # (the .so reads json from __FILE__'s parent directory)
            precompiled_dir = os.path.dirname(precompiled_so)
            shutil.copy(json_file_path, os.path.join(precompiled_dir, "task_graph.json"))
            print(f"Using precompiled .so: {precompiled_so}")
        else:
            print("Compiling megakernel using the following command line:")
            print(cc_cmd)
            subprocess.check_call(cc_cmd)

        import importlib.util

        # Set MPK_SO_PATH so init_persistent_kernel() can load the module via
        # cuLibraryLoadFromFile for nvshmemx_culibrary_init (NVSHMEM_NO_DEVICE_LIB mode).
        os.environ["MPK_SO_PATH"] = so_path

        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")
        self.launch_func = getattr(mod, "launch_func")
        self.finalize_func = getattr(mod, "finalize_func")
        print("Finished megakernel compilation...")

        expected_order = [
            "step",
            "tokens",
            "input_tokens",
            "output_tokens",
            "num_new_tokens",
            "prompt_lengths",
            "qo_indptr_buffer",
            "paged_kv_indptr_buffer",
            "paged_kv_indices_buffer",
            "paged_kv_last_page_len_buffer",
            "paged_kv_indices_snapshot",
        ]
        meta_tensors_ptr = []
        for key in expected_order:
            if key not in self.meta_tensors:
                if self.test_mode:
                    # In test mode, we can allow missing meta tensors and pass null pointer
                    meta_tensors_ptr.append(0)  
                else:
                  raise ValueError(f"Missing meta tensor: {key}")
            else:
              meta_tensors_ptr.append(self.meta_tensors[key].data_ptr())
        profiler_buffer_ptr = (
            self.profiler_tensor.data_ptr() if self.profiler_tensor is not None else 0
        )
        self.eos_token_id = kwargs.get("eos_token_id", self.eos_token_id)
        self.init_func(
            meta_tensors_ptr,
            profiler_buffer_ptr,
            self.mpi_rank,
            self.num_workers,
            self.num_local_schedulers,
            self.num_remote_schedulers,
            self.max_seq_length,
            self.total_num_requests,
            self.eos_token_id,
            self.allocate_nvshmem_teams,
            self.test_mode,
        )

        self._is_compiled = True

    def __call__(self, **kwargs):
        stream = kwargs.get("default_stream", None)
        if stream is None:
           stream = torch.cuda.current_stream()
        # Convert torch.cuda.Stream to raw pointer (integer) for the C launcher
        stream_ptr = 0
        if hasattr(stream, "cuda_stream"):
            try:
                stream_ptr = int(stream.cuda_stream)
            except Exception:
                try:
                    stream_ptr = int(stream.cuda_stream.value)
                except Exception as e:
                    raise ValueError(f"Invalid stream object: {stream} is of type {type(stream)}: {e}")
        elif isinstance(stream, int):
            stream_ptr = stream
        else:
            raise ValueError("Invalid stream object")
        self.launch_func(stream_ptr)
        if self.profiler_tensor is not None:
            from .profiler_persistent import export_to_perfetto_trace
            
            if self.trace_name:
                trace_name = self.trace_name + ".perfetto-trace"
            else:
                trace_name = f"mirage_{self.mpi_rank}.perfetto-trace"

            export_to_perfetto_trace(
                self.profiler_tensor, trace_name
            )

    def run_test_mode(self):
        """Test-mode execution: launch the task graph once.

        Input/output tensors must be pre-attached via attach_input() before
        compile(). After run_test_mode() returns, the output tensors contain the results.
        """
        assert self.test_mode, "run_test_mode() is only available in test mode"
        assert self._is_compiled, "Must call compile() before run_test_mode()"

        stream = torch.cuda.current_stream()
        # Convert torch.cuda.Stream to raw pointer (integer) for the C launcher
        stream_ptr = 0
        if hasattr(stream, "cuda_stream"):
            try:
                stream_ptr = int(stream.cuda_stream)
            except Exception:
                try:
                    stream_ptr = int(stream.cuda_stream.value)
                except Exception as e:
                    raise ValueError(f"Invalid stream object: {stream} is of type {type(stream)}: {e}")
        elif isinstance(stream, int):
            stream_ptr = stream
        else:
            raise ValueError("Invalid stream object")
        self.launch_func(stream_ptr)

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        if self._is_compiled:
            self.finalize_func()
        self.__finalized__ = True
