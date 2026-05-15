import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig
import json

from ..core import *
from ..kernel import get_key_paths, KNGraph, TBGraph
from .speculative import (
    SpecDecodeConfig,
    PromptLookupConfig,
)
from .multigpu import (
  allocate_nvshmem_teams,
  auto_select_allreduce_implementation,
)
from typing import Optional

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

extern std::string g_task_graph_json_path;

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
  PyObject *meta_list, *py_profiler_buffer, *tensor_names_list, *tensor_ptrs_list, *py_json_path;
  std::vector<void*> meta_tensors;
  std::vector<std::string> model_tensor_names;
  std::vector<void*> model_tensor_ptrs;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests;
  long long eos_token_id;
  int allocate_nvshmem_teams;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOiiiiiiLiOOO", &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, &eos_token_id, &allocate_nvshmem_teams, &tensor_names_list, &tensor_ptrs_list, &py_json_path)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }
  if(!PyList_Check(tensor_names_list)) {
    PyErr_SetString(PyExc_TypeError, "tensor_names must be a list.");
    return NULL;
  }
  if(!PyList_Check(tensor_ptrs_list)) {
    PyErr_SetString(PyExc_TypeError, "tensor_ptrs must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);
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

  Py_ssize_t num_tensors = PyList_Size(tensor_names_list);
  for(Py_ssize_t i = 0; i < num_tensors; i++) {
    PyObject *name_item = PyList_GetItem(tensor_names_list, i);
    PyObject *ptr_item = PyList_GetItem(tensor_ptrs_list, i);
    
    const char *name_str = PyUnicode_AsUTF8(name_item);
    if (!name_str) {
      PyErr_Format(PyExc_TypeError, "Failed to convert tensor name %d to string", i);
      return NULL;
    }
    model_tensor_names.push_back(std::string(name_str));
    
    void *ptr = PyLong_AsVoidPtr(ptr_item);
    model_tensor_ptrs.push_back(ptr);
  }

  if (PyUnicode_Check(py_json_path)) {
    const char *json_path = PyUnicode_AsUTF8(py_json_path);
    if (json_path && strlen(json_path) > 0) {
      g_task_graph_json_path = std::string(json_path);
    }
  }

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id, allocate_nvshmem_teams, model_tensor_names, model_tensor_ptrs);

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

def _mla_tp4_v_splits(max_seq_length=None):
    env_value = os.environ.get("MPK_MLA_TP4_V_SPLITS")
    if env_value is not None:
        value = int(env_value)
    else:
        # Standalone ablation on B200:
        #   KV <= 2048: 8 V splits is fastest for the current MPK single-split
        #   write-final path.
        #   KV >= 3072: 2 V splits matches the original TP4 MLA PR and avoids
        #   redoing the QK/softmax work eight times.
        value = 2 if max_seq_length is not None and max_seq_length >= 3072 else 8
    if value not in (1, 2, 4, 8):
        raise ValueError("MPK_MLA_TP4_V_SPLITS must be one of 1, 2, 4, 8")
    return value

def _mla_tp4_head_groups():
    value = int(os.environ.get("MPK_MLA_TP4_HEAD_GROUPS", "1"))
    if value not in (1, 2, 4, 8):
        raise ValueError("MPK_MLA_TP4_HEAD_GROUPS must be one of 1, 2, 4, 8")
    return value

def _mla_tp4_rd_dv():
    value = int(os.environ.get("MPK_MLA_TP4_RD_DV", "2"))
    if value not in (2, 4, 8):
        raise ValueError("MPK_MLA_TP4_RD_DV must be one of 2, 4, 8")
    return value

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
        "-lineinfo",
        f"-I{py_include_dir}",
        f"-I{mirage_inc_path}",
        f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/tools/util/include')}",
        f"-I{os.path.join(mirage_deps_path, 'json/include')}",
        f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
        f"-DMIRAGE_USE_CUTLASS_KERNEL={'1' if use_cutlass_kernel else '0'}",
        f"-DMIRAGE_MLA_TP4_V_SPLITS={_mla_tp4_v_splits(mpk.max_seq_length)}",
        f"-DMIRAGE_MLA_TP4_HEAD_GROUPS={_mla_tp4_head_groups()}",
        f"-DMIRAGE_MLA_TP4_RD_DV={_mla_tp4_rd_dv()}",
    ]
    # rdc=true is the default on every NVSHMEM build. The old Blackwell
    # rdc=false + self-contained-allreduce workaround (hand-rolled
    # nvshmemi_device_state_d + nvshmemid_hostlib_init_attr callback, needed
    # because rdc=true previously inflated registers 166→255 on sm_100a) is
    # kept behind MPK_RDC_FALSE=1 as a safety escape hatch — on CUDA 13.2 +
    # NVSHMEM 3.6.5 the register-spill issue is gone (verified 2026-04-22 at
    # TP=2 and TP=4 across mbt∈{1,64} and MTP spec∈{0,1,3}).
    _rdc_false = os.environ.get("MPK_RDC_FALSE", "0") == "1" and target_cc >= 100
    flags = [
        "-shared",
        _detect_cxx_standard(),
        "-rdc=false" if (not use_nvshmem or _rdc_false) else "-rdc=true",
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

    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
        if _rdc_false:
            # Blackwell MPK_RDC_FALSE=1 escape hatch: self-contained allreduce,
            # no libnvshmem_device.a (kept for regression isolation; see the
            # block above for when this path is needed).
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
            # Default path: standard NVSHMEM link with device library +
            # rdc=true. Used everywhere unless MPK_RDC_FALSE=1 on Blackwell.
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
        kv_cache: Optional[dict] = None,
    ):
        self.__finalized__ = False
        self._is_compiled = False
        self.test_mode = test_mode
        # KV cache pool registered at top level (Option-III in the
        # PyTorch-module refactor). New-API attention layers retrieve
        # their per-layer slice via ``self.get_kv_cache(layer_idx)``.
        # Expected shape: dict with keys "k_cache" and "v_cache", each
        # mapping to a tensor whose dim-0 is num_layers. None when the
        # legacy demo wires KV cache through model state instead.
        self._kv_cache = kv_cache

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
        self.use_nvshmem = world_size > 1
        self.spec_decode_config = spec_decode_config
        self.use_cutlass_kernel = use_cutlass_kernel
        # Dictionary to track attached model tensors for kernel reuse
        self._model_tensors = {}
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
            # Auto-allocate any meta tensors the test author didn't provide so
            # the kernel sees valid GPU pointers. Shapes are derived from the
            # kernel-level params already on `self`. Defaults model "single
            # prefill of max_num_batched_tokens tokens, content all zero" — the
            # test author overrides any subset by setting them in
            # `params["meta_tensors"]` before constructing PersistentKernel.
            self._apply_test_mode_meta_defaults()

        self.total_num_requests = self.meta_tensors["tokens"].shape[0]

        # Checks
        assert self.max_seq_length == self.meta_tensors["tokens"].shape[1]
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

    def compile_scope(self):
        # Returning the free-function contextmanager (rather than implementing
        # it inline) keeps the ContextVar binding logic in one place — see
        # mirage.mpk.context.compile_scope. Imported lazily to avoid a hard
        # import-time cycle (context.py only references PersistentKernel under
        # TYPE_CHECKING).
        from .context import compile_scope as _compile_scope
        return _compile_scope(self)

    def get_kv_cache(self, layer_idx: int):
        # Returns the (k_slice, v_slice) torch tensors for one decoder
        # layer's paged KV cache. The pool was registered into the PK
        # at construction via ``kv_cache={"k_cache": ..., "v_cache": ...}``.
        # New-API ``PagedAttention.compile()`` calls this then
        # ``attach_input`` to convert to DTensors.
        if self._kv_cache is None:
            raise RuntimeError(
                "PersistentKernel was constructed without kv_cache=; "
                "pass kv_cache={'k_cache': k_pool, 'v_cache': v_pool} "
                "where each pool has dim-0 == num_layers."
            )
        return self._kv_cache["k_cache"][layer_idx], self._kv_cache["v_cache"][layer_idx]

    def _apply_test_mode_meta_defaults(self):
        # Allocate any missing meta tensors with shapes derived from the
        # kernel-level params. Mirrors the production wiring in
        # demo/qwen3/demo.py (qo/paged_kv buffers sized to max_num_*).
        # `total_num_requests` is taken from `tokens.shape[0]` after this
        # function runs, so default `tokens` to a single-request buffer.
        device = "cuda"
        if "tokens" not in self.meta_tensors:
            self.meta_tensors["tokens"] = torch.zeros(
                1, self.max_seq_length, dtype=torch.int64, device=device)
        n_req = self.meta_tensors["tokens"].shape[0]
        if "step" not in self.meta_tensors:
            self.meta_tensors["step"] = torch.zeros(
                n_req, dtype=torch.int32, device=device)
        if "prompt_lengths" not in self.meta_tensors:
            # Default to a single prefill that fills one iter's batched-token
            # budget. Test authors override for decode/multi-request scenarios.
            self.meta_tensors["prompt_lengths"] = torch.full(
                (n_req,), self.max_num_batched_tokens,
                dtype=torch.int32, device=device)
        if "input_tokens" not in self.meta_tensors:
            self.meta_tensors["input_tokens"] = torch.zeros(
                self.max_num_batched_tokens, dtype=torch.int64, device=device)
        if "output_tokens" not in self.meta_tensors:
            self.meta_tensors["output_tokens"] = torch.zeros(
                self.max_num_batched_tokens, dtype=torch.int64, device=device)
        if "num_new_tokens" not in self.meta_tensors:
            self.meta_tensors["num_new_tokens"] = torch.zeros(
                1, dtype=torch.int32, device=device)
        if "qo_indptr_buffer" not in self.meta_tensors:
            self.meta_tensors["qo_indptr_buffer"] = torch.zeros(
                self.max_num_batched_requests + 1,
                dtype=torch.int32, device=device)
        if "paged_kv_indptr_buffer" not in self.meta_tensors:
            self.meta_tensors["paged_kv_indptr_buffer"] = torch.zeros(
                self.max_num_batched_requests + 1,
                dtype=torch.int32, device=device)
        if "paged_kv_indices_buffer" not in self.meta_tensors:
            self.meta_tensors["paged_kv_indices_buffer"] = torch.zeros(
                self.max_num_pages, dtype=torch.int32, device=device)
        if "paged_kv_last_page_len_buffer" not in self.meta_tensors:
            self.meta_tensors["paged_kv_last_page_len_buffer"] = torch.zeros(
                self.max_num_batched_requests,
                dtype=torch.int32, device=device)

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

    def _save_kernel_metadata(self, path: str) -> None:
        """Save kernel config for validation when loading."""
        metadata = {
            "mode": self.mode,
            "max_seq_length": self.max_seq_length,
            "max_num_batched_requests": self.max_num_batched_requests,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_pages": self.max_num_pages,
            "page_size": self.page_size,
            "world_size": self.world_size,
            "rank": self.mpi_rank,
            "cuda_cc": self.target_cc,
            "tensor_names": sorted(self._model_tensors.keys()),
        }
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _validate_kernel_compatibility(self, metadata_path: str) -> None:
        """Validate saved kernel is compatible with current config."""
        with open(metadata_path, "r") as f:
            saved = json.load(f)
        
        errors = []
        checks = [
            ("mode", self.mode),
            ("max_seq_length", self.max_seq_length),
            ("max_num_batched_requests", self.max_num_batched_requests),
            ("max_num_batched_tokens", self.max_num_batched_tokens),
            ("max_num_pages", self.max_num_pages),
            ("page_size", self.page_size),
            ("world_size", self.world_size),
            ("rank", self.mpi_rank),
            ("cuda_cc", self.target_cc),
        ]
        for key, current in checks:
            if saved.get(key) != current:
                errors.append(f"{key}: saved={saved.get(key)}, current={current}")
        
        # Check tensor names
        saved_tensors = set(saved.get("tensor_names", []))
        current_tensors = set(self._model_tensors.keys())
        if saved_tensors != current_tensors:
            missing = saved_tensors - current_tensors
            extra = current_tensors - saved_tensors
            if missing:
                errors.append(f"missing tensors: {sorted(missing)}")
            if extra:
                errors.append(f"extra tensors: {sorted(extra)}")
        
        if errors:
            raise ValueError(
                f"Kernel incompatible with current config:\n  " + "\n  ".join(errors)
            )

    def attach_input(self, torch_tensor: torch.Tensor, name: str = None) -> DTensor:
        dims = tuple([d for d in torch_tensor.shape])
        strides = tuple([s for s in torch_tensor.stride()])
        # Check layout: row-major (possibly with padded outer strides — supports
        # slice views like q_nope_pe[:, :, :512] of a (mbt, H, 576) parent for
        # the MPK_DSV3_BMM TMA-stride fuse) or column-major (FP8 scale tensors).
        # Padded row-major: each outer stride covers AT LEAST a contiguous
        # row at the next level (== for contig, > for slice view of a wider
        # parent). Innermost stride must still be 1.
        is_row_major = (
            all(strides[d] >= strides[d + 1] * dims[d + 1]
                for d in range(len(dims) - 1))
            and strides[-1] == 1
        )
        is_col_major = len(dims) == 2 and strides[0] == 1 and strides[1] >= dims[0]
        assert is_row_major or is_col_major, \
            f"Tensor must be row-major or column-major, got dims={dims} strides={strides}"
        dtype = convert_torch_type_to_dtype(torch_tensor.dtype)
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        self.kn_graph.attach_torch_tensor(t, torch_tensor, name)
        # Track tensor for kernel reuse - tensor pointer can be passed at runtime
        self._model_tensors[name] = torch_tensor
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

    # ---- Virtual tensor (view) APIs ----------------------------------------
    # These operators return DTensors that share memory with `input`. The
    # returned view has its own GUID plus base_guid + view_offset metadata.
    # The dependency analyzer treats any edge involving a view as a coarse
    # barrier edge (one event per layer instead of GCD-based per-tile
    # events); see docs / annotated_graph.cc for details.

    def view(self, input: DTensor, new_shape: list) -> DTensor:
        """Reshape into a new shape that has the same total element count.
        Returns a single virtual DTensor."""
        return self.kn_graph.view(input, list(new_shape))

    def narrow(self, input: DTensor, dim: int, start: int, length: int) -> DTensor:
        """Take a contiguous-window virtual DTensor of `input` along `dim`."""
        return self.kn_graph.narrow(input, dim, start, length)

    def split(self, input: DTensor, sizes_or_chunks, dim: int) -> list:
        """Split `input` into multiple virtual DTensors along `dim`.

        sizes_or_chunks may be an int (equal-size chunk count) or a list of
        explicit sizes summing to `input.dim[dim]`."""
        return self.kn_graph.split(input, sizes_or_chunks, dim)

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
        process_dim: int = None,
        in_offset_elems: int = 0,
        out_offset_elems: int = 0,
    ):
        # process_dim / in_offset_elems / out_offset_elems support
        # row-slice RMSnorm where input/output are slices of a wider buffer
        # (e.g., QKV-a fused qkv_a_out where q_a_layernorm reads
        # cols [0:1536) and kv_a_layernorm reads cols [1536:2048)). Defaults
        # match legacy contiguous behaviour.
        assert input.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(output, (0, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        task_name = "rmsnorm_hopper" if self.target_cc >= 90 else "rmsnorm"
        if (process_dim is None and in_offset_elems == 0
                and out_offset_elems == 0):
            self.kn_graph.register_task(tb_graph, task_name)
        else:
            if process_dim is None:
                process_dim = output.dim(1)
            self.kn_graph.register_task(
                tb_graph,
                task_name,
                [process_dim, in_offset_elems, out_offset_elems],
            )

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
        c_latent_row_stride: int = None,
        c_latent_offset_elems: int = 0,
        k_pe_row_stride: int = None,
        k_pe_offset_elems: int = 0,
    ):
        # Stride/offset overrides let the kernel read c_latent / k_pe from
        # a wider parent buffer (QKV-a fused: qkv_a_out (mbt, 2176) with
        # c_latent at [1536:2048) and k_pe at [2048:2112)). Defaults
        # preserve legacy contiguous inputs.
        d_k, d_v, page_size = mla_params
        slice_override = (
            c_latent_row_stride is not None or
            c_latent_offset_elems != 0 or
            k_pe_row_stride is not None or
            k_pe_offset_elems != 0)
        if slice_override:
            params = [
                d_k, d_v, page_size,
                c_latent_row_stride if c_latent_row_stride is not None else d_v,
                c_latent_offset_elems,
                k_pe_row_stride if k_pe_row_stride is not None else 128,
                k_pe_offset_elems,
            ]
        else:
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

    def mla_kv_gather_unified_layer(
        self,
        c_latent_new: DTensor,
        k_pe_new: DTensor,
        paged_cache: DTensor,
        contiguous_kv: DTensor,
        ckv_sep: DTensor,
        kpe_sep: DTensor,
        mla_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
        c_latent_row_stride: int = None,
        c_latent_offset_elems: int = 0,
        k_pe_row_stride: int = None,
        k_pe_offset_elems: int = 0,
    ):
        """Append paged KV once, then materialize decode or prefill views.

        Stride/offset overrides let the kernel read c_latent / k_pe from a
        wider parent buffer (QKV-a fused path). Defaults preserve legacy.
        """
        d_k, d_v, page_size = mla_params
        slice_override = (
            c_latent_row_stride is not None or
            c_latent_offset_elems != 0 or
            k_pe_row_stride is not None or
            k_pe_offset_elems != 0)
        if slice_override:
            params = [
                d_k, d_v, page_size,
                c_latent_row_stride if c_latent_row_stride is not None else d_v,
                c_latent_offset_elems,
                k_pe_row_stride if k_pe_row_stride is not None else 128,
                k_pe_offset_elems,
            ]
        else:
            params = [d_k, d_v, page_size]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(c_latent_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_pe_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(paged_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(contiguous_kv, (-1, -1, -1), -1, True)
        tb_graph.new_input(ckv_sep, (-1, -1, -1), -1, True)
        tb_graph.new_input(kpe_sep, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [
                c_latent_new,
                k_pe_new,
                paged_cache,
                contiguous_kv,
                ckv_sep,
                kpe_sep,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "mla_kv_gather_unified_sm100", params)

    def deepseek_mla_rope_q_layer(
        self,
        q_nope_pe: DTensor,
        q_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        num_heads: int,
        has_split_q: bool,
        grid_dim: tuple,
        block_dim: tuple = (128, 1, 1),
        q_tile_size: int = 16,
    ):
        params = [num_heads, q_tile_size, 1 if has_split_q else 0]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Duplicate Q tensors are used as task outputs. This gives downstream
        # MLA tasks a real dependency on the in-place RoPE write without
        # joining the independent K-RoPE dependency chain.
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [
                q_nope_pe,
                q_pe,
                cos_pos_embed,
                sin_pos_embed,
                q_nope_pe,
                q_pe,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "deepseek_mla_rope_q_sm100", params)

    def deepseek_mla_rope_q_fused_layer(
        self,
        q_nope_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        num_heads: int,
        grid_dim: tuple,
        block_dim: tuple = (128, 1, 1),
        q_tile_size: int = 16,
    ):
        params = [num_heads, q_tile_size]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_nope_pe, cos_pos_embed, sin_pos_embed, q_nope_pe],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "deepseek_mla_rope_q_fused_sm100", params)

    def deepseek_mla_rope_q_split_layer(
        self,
        q_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        num_heads: int,
        grid_dim: tuple,
        block_dim: tuple = (128, 1, 1),
        q_tile_size: int = 16,
        qfused_mode: int = 0,
    ):
        # qfused_mode = 0: q_pe is a standalone (mbt, num_heads*64) tensor.
        # qfused_mode = 1: q_pe is the same DTensor as the fused q_b_prefill
        # buffer (mbt, num_heads*192) with row-swap layout. Kernel uses
        # row_stride = num_heads*192 and pe_base_in_row = num_heads*128.
        params = [num_heads, q_tile_size]
        if qfused_mode != 0:
            params.append(qfused_mode)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_pe, cos_pos_embed, sin_pos_embed, q_pe],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "deepseek_mla_rope_q_split_sm100", params)

    def deepseek_mla_rope_k_layer(
        self,
        k_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        grid_dim: tuple,
        block_dim: tuple = (128, 1, 1),
        q_tile_size: int = 16,
        k_pe_row_stride: int = None,
        k_pe_offset: int = 0,
    ):
        # k_pe_row_stride / k_pe_offset support running the K_PE rotation
        # in-place on a slice of a wider buffer (e.g., qkv_a_out (mbt, 2176)
        # where k_pe lives at cols [2048:2112)). Defaults preserve legacy.
        params = [q_tile_size]
        if k_pe_row_stride is not None or k_pe_offset != 0:
            if k_pe_row_stride is None:
                k_pe_row_stride = 128
            params = [q_tile_size, k_pe_row_stride, k_pe_offset]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(k_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_pe, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [
                k_pe,
                cos_pos_embed,
                sin_pos_embed,
                k_pe,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "deepseek_mla_rope_k_sm100", params)

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

    def mla_prefill_absorbed_layer(
        self,
        q_nope_pe: DTensor,  # [S, H, D_CKV + D_KPE] flattened
        kv: DTensor,         # [B * max_seq_len, D_CKV + D_KPE]
        output: DTensor,     # [S, H, D_V]
        mla_params: tuple,   # (num_heads, seq_len, d_ckv, d_kpe, d_v)
        grid_dim: tuple,     # (H, num_q_blocks, B)
        block_dim: tuple,    # (256, 1, 1)
    ):
        num_heads, seq_len, d_ckv, d_kpe, d_v = mla_params
        params = [num_heads, seq_len, d_ckv, d_kpe, d_v]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([q_nope_pe, kv, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "mla_prefill_absorbed_sm100", params)

    def mla_prefill_tp8_layer(
        self,
        q_nope: DTensor,   # [B, S, H, D_QK_NOPE=128]
        q_pe: DTensor,     # [B, S, H, D_QK_ROPE=64]
        k: DTensor,        # [B, S, D_QK=192] (nope+rope concat along last dim)
        v: DTensor,        # [B, S, D_V=128]
        output: DTensor,   # [B, S, H, D_V=128]
        mla_params: tuple, # (num_heads, seq_len)
        grid_dim: tuple,   # (H, num_q_blocks, B)
        block_dim: tuple,  # (128, 1, 1)
    ):
        # MLA Prefill TP=8 (unabsorbed, TMA K/V). NUM_HEADS per rank = 16.
        # Grid: (H, ceil(S/BM), B) where BM=64.
        num_heads, seq_len = mla_params
        params = [num_heads, seq_len]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Kernel does its own per-block slicing (head, q_block, batch come via
        # task metadata). Each input is presented as the full tensor.
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(k, (-1, -1, -1), -1, True)
        tb_graph.new_input(v, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_nope, q_pe, k, v, output], tb_graph
        )
        self.kn_graph.register_task(tb_graph, "mla_prefill_tp8_sm100", params)

    def mla_prefill_tp8_chunked_layer(
        self,
        q_nope: DTensor,    # [B, q_len, H, 128] OR fused-Q [B, q_len, H, 192]
        q_pe: DTensor,      # [B, q_len, H, 64]  OR same as q_nope if fused
        k_nope: DTensor,    # [B, kv_len, H, 128]
        k_rope: DTensor,    # [B, kv_len, 1, 64]
        v: DTensor,         # [B, kv_len, H, 128]
        output: DTensor,    # [B, q_len, H, 128]
        mla_params: tuple,  # (num_heads, q_len, kv_len, q_start)
        grid_dim: tuple,    # (H, ceil(q_len/64), B)
        block_dim: tuple,   # (128, 1, 1)
        qfused_mode: int = 0,  # 0 = legacy split q_nope/q_pe; 1 = fused
    ):
        num_heads, q_len, kv_len, q_start = mla_params
        params = [num_heads, q_len, kv_len, q_start, qfused_mode]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_rope, (-1, -1, -1), -1, True)
        tb_graph.new_input(v, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_nope, q_pe, k_nope, k_rope, v, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "mla_prefill_tp8_chunked_sm100", params
        )

    def mla_prefill_tp8_chunked_splitk_layer(
        self,
        q_nope: DTensor,    # [B, q_len, H, 128]
        q_pe: DTensor,      # [B, q_len, H, 64]
        k_nope: DTensor,    # [B, kv_len, H, 128]
        k_rope: DTensor,    # [B, kv_len, 1, 64]
        v: DTensor,         # [B, kv_len, H, 128]
        partial: DTensor,   # float, num_splits*B*nqb*H*BM*(D_V+4)
        mla_params: tuple,  # (num_heads, q_len, kv_len, q_start, num_splits)
        grid_dim: tuple,    # (H, nqb*num_splits, B)
        block_dim: tuple,   # (128, 1, 1)
    ):
        num_heads, q_len, kv_len, q_start, num_splits = mla_params
        nqb = (q_len + 63) // 64
        params = [num_heads, q_len, kv_len, q_start, num_splits, nqb]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_rope, (-1, -1, -1), -1, True)
        tb_graph.new_input(v, (-1, -1, -1), -1, True)
        tb_graph.new_input(partial, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [q_nope, q_pe, k_nope, k_rope, v, partial], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "mla_prefill_tp8_chunked_splitk_sm100", params
        )

    def mla_prefill_tp8_chunked_reduce_layer(
        self,
        partial: DTensor,
        output: DTensor,    # [B, q_len, H, 128]
        mla_params: tuple,  # (num_heads, q_len, num_splits)
        grid_dim: tuple,    # (H, nqb, B)
        block_dim: tuple,   # (256, 1, 1)
    ):
        num_heads, q_len, num_splits = mla_params
        nqb = (q_len + 63) // 64
        params = [num_heads, q_len, num_splits, nqb]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([partial, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "mla_prefill_tp8_chunked_reduce_sm100", params
        )

    def mla_unified_layer(
        self,
        q_nope: DTensor,          # [S, H, D_CKV] flattened
        q_pe: DTensor,            # [S, H, D_KPE] flattened
        ckv: DTensor,             # [B*S, D_CKV]
        kpe: DTensor,             # [B*S, D_KPE]
        output: DTensor,          # final O [S, H, D_V], prefill branch writes
        q_input: DTensor,         # fused Q [S, H*(D_CKV+D_KPE)] with TMA desc
        kv_input: DTensor,        # contiguous KV [B*S, D_K] with TMA desc
        output_partial: DTensor,  # decode partial O
        output_lse: DTensor,      # decode partial LSE
        q_len: int,
        kv_len: int,
        num_heads: int,
        tp_size: int,
        d_ckv: int = 512,
        d_kpe: int = 64,
        d_v: int = 512,
        decode_q_len: int | None = None,
    ):
        num_splits = (kv_len + 128 - 1) // 128
        # q_len is the prompt-prefill chunk budget. Decode width is controlled
        # by generation semantics (one token, or MTP verify width), not MBT.
        decode_q_len = min(decode_q_len if decode_q_len is not None else q_len, 8)
        if tp_size == 1:
            hpb = num_heads // decode_q_len
            if hpb < 1:
                hpb = 1
            while num_heads % hpb != 0:
                hpb -= 1
            num_groups = num_heads // hpb
            x_mul = 1
        elif tp_size == 2:
            qpg = min(2, decode_q_len)
            num_groups = (decode_q_len + qpg - 1) // qpg
            x_mul = 1
        elif tp_size == 4:
            qpg = min(4, decode_q_len)
            num_groups = (decode_q_len + qpg - 1) // qpg
            x_mul = 2
        elif tp_size == 8:
            q_len_padded = (decode_q_len + 1) & ~1
            qpg = 2
            num_groups = (q_len_padded + qpg - 1) // qpg
            x_mul = 1
        else:
            raise ValueError(f"Unsupported MLA unified tp_size={tp_size}")

        num_q_blocks = (q_len + 64 - 1) // 64
        decode_blocks_x = num_groups * num_splits * x_mul
        grid_dim = (
            max(num_heads, decode_blocks_x),
            max(num_q_blocks, self.max_num_batched_requests),
            self.max_num_batched_requests,
        )
        block_dim = (256, 1, 1)
        params = [
            num_heads,
            decode_q_len,
            kv_len,
            num_splits,
            tp_size,
            d_ckv,
            d_kpe,
            d_v,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(ckv, (-1, -1, -1), -1, True)
        tb_graph.new_input(kpe, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_lse, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [
                q_nope,
                q_pe,
                ckv,
                kpe,
                output,
                q_input,
                kv_input,
                output_partial,
                output_lse,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "mla_unified_sm100", params)

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
        grid_dim = (num_splits, num_head_groups, self.max_num_batched_requests)
        block_dim = (128, 1, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_lse, (-1, -1, -1), -1, True)
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
        grid_dim = ((d_v + rd_dv - 1) // rd_dv,
                    num_head_groups,
                    self.max_num_batched_requests)
        block_dim = (256, 1, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
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
        task_name, has_v_split=False, q_len_real=None, head_groups=1,
        v_splits=2, num_splits_override=None,
    ):
        """Internal helper for TP=2/4/8 decode dispatch.
          q_len: padded Q_LEN passed to the kernel
          q_len_real: TP=8 only — actual unpadded Q_LEN
          num_heads: 64/32/16 per TP variant
          has_v_split: TP=4 only — block_x multiplied to encode V split id
          head_groups: additional head split packed into block_x
        """
        if num_heads == 64:
            qpg = min(2, q_len)
        elif num_heads == 32:
            qpg = min(4, q_len)
        else:  # TP=8
            qpg = 2
        num_groups = (q_len + qpg - 1) // qpg
        num_splits = (
            num_splits_override
            if num_splits_override is not None
            else (kv_len + 128 - 1) // 128
        )  # TILE_S=128
        # TP=4 packs the V split id into block_x → multiple tasks per split.
        x_mul = v_splits if has_v_split else 1
        grid_dim = (num_groups * num_splits * x_mul * head_groups,
                    self.max_num_batched_requests,
                    1)
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
        # TP4 can be compiled with a different RD_DV for ablation. Keep the
        # grid matched to the compiled coverage; standalone tests currently
        # show RD_DV=2 is fastest for KV=4096.
        rd_dv = _mla_tp4_rd_dv() if task_name == "mla_mtp_decode_tp4_reduce_sm100" else 2

        params = [num_groups, q_len, num_splits, rd_dv]
        grid_dim = ((d_v + rd_dv - 1) // rd_dv,
                    num_groups,
                    self.max_num_batched_requests)
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
        num_splits_override=None,
    ):
        self._mla_mtp_decode_tp_layer(
            q_input, kv_input, output_partial, output_lse,
            q_len, kv_len, num_heads=64,
            task_name="mla_mtp_decode_tp2_sm100",
            head_groups=2,
            num_splits_override=num_splits_override,
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
        num_splits_override=None,
    ):
        # TP=4 V-split: each split writes a disjoint D_V chunk. Keep this
        # configurable for ablation because each split repeats QK/softmax.
        self._mla_mtp_decode_tp_layer(
            q_input, kv_input, output_partial, output_lse,
            q_len, kv_len, num_heads=32,
            task_name="mla_mtp_decode_tp4_sm100", has_v_split=True,
            head_groups=_mla_tp4_head_groups(),
            v_splits=_mla_tp4_v_splits(self.max_seq_length),
            num_splits_override=num_splits_override,
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
        q_len_real, kv_len, num_splits_override=None,
    ):
        # TP=8 pads Q_LEN to even
        q_len = (q_len_real + 1) & ~1
        self._mla_mtp_decode_tp_layer(
            q_input, kv_input, output_partial, output_lse,
            q_len, kv_len, num_heads=16,
            task_name="mla_mtp_decode_tp8_sm100", q_len_real=q_len_real,
            num_splits_override=num_splits_override,
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
        target: DTensor,
        dummy: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        dummy_input_map: tuple,
        target_input_map: tuple,
    ):
        """Zero-fill `target` using a custom kernel.

        `dummy` only carries a dependency edge: it appears as both an input and
        an output of the task so the MPK dep-tracker chains tensor_init between
        the producer of `dummy` and any downstream consumer of `dummy`. The
        kernel never reads or writes `dummy`'s data.

        Optimization: when `target_input_map[i]` == -1 for grid axis `i`, that
        grid axis does NOT partition `target`, so all CTAs on that axis would
        zero the same target tile redundantly (1 logical wave with K-fold
        replication). Collapse such redundant axes to 1 — the dep edge stays
        task-level (no CTA-level partner-tracking in the runtime), so reducing
        the CTA count does not affect downstream consumers. Saves the splitk
        prepend-tensor_init from launching grid_y replicates of the same zero
        tile.
        """
        gx, gy, gz = grid_dim
        if target_input_map[1] == -1 and gy > 1:
            gy = 1
        if target_input_map[2] == -1 and gz > 1:
            gz = 1
        grid_dim = (gx, gy, gz)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # bgraph order = [dummy, target, dummy] -> arity (1, 2):
        #   input_ops[0]  = dummy   (read dep)
        #   output_ops[0] = target  (the buffer the kernel zeroes)
        #   output_ops[1] = dummy   (dep-only write)
        tb_graph.new_input(dummy, dummy_input_map, -1, True)
        tb_graph.new_input(target, target_input_map, -1, True)
        tb_graph.new_input(dummy, dummy_input_map, -1, True)
        self.kn_graph.customized([dummy, target, dummy], tb_graph)

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
        local_expert_start: int = 0,
    ):
        import struct

        assert input.num_dims == 2  # (batch_size, num_experts)
        total_num_experts = input.dim(1)
        assert bias.num_dims == 1  # (num_experts,)
        assert bias.dim(0) == total_num_experts
        assert len(output) == 3
        moe_topk_weight, moe_routing_indices, moe_masks = output
        assert moe_topk_weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (local_num_experts, batch_size)
        assert moe_masks.num_dims == 1  # (local_num_experts + 1)
        local_num_experts = moe_routing_indices.dim(0)
        assert moe_masks.dim(0) == local_num_experts + 1
        assert 0 <= local_expert_start
        assert local_expert_start + local_num_experts <= total_num_experts

        scaling_bits = struct.unpack("i", struct.pack("f", routed_scaling_factor))[0]
        params = [
            num_groups,
            topk_group,
            scaling_bits,
            local_expert_start,
            local_expert_start + local_num_experts,
        ]

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
    def _fp8_quantize_group_tiles(
        self, hidden_size: int, scale_ue8m0: bool, max_tiles: int = 16
    ) -> int:
        """Pick the per-row group-tile count for quantize_fp8.

        ``max_tiles`` is the caller's upper bound — passed by
        ``quantize_fp8_layer`` as ``num_workers // grid_y`` so the total
        launched CTAs (group_tiles * grid_y) stays ≤ num_workers and the
        task fits in a single dispatch wave. For prefill where row_count
        >= num_workers this collapses to ``group_tiles = 1`` (each CTA
        owns all groups for its row(s)); for decode-style row_count=1 it
        scales up to use idle workers, capped by num_groups and the UE8M0
        4-group alignment.
        """
        num_groups = max(1, hidden_size // 128)
        if scale_ue8m0:
            # Packed UE8M0 stores four group scales per uint32. Split only at
            # four-group boundaries so each CTA owns whole packed scale words.
            group_tiles = 1
            for candidate in range(min(max_tiles, num_groups), 1, -1):
                if num_groups % candidate == 0:
                    groups_per_tile = num_groups // candidate
                    if groups_per_tile % 4 == 0:
                        group_tiles = candidate
                        break
            return group_tiles
        # Float-scale MoE quantization has no packing hazard.
        return min(min(max_tiles, 4), max(1, num_groups // 8))

    def quantize_fp8_layer(
        self,
        input: DTensor,
        output_fp8: DTensor,
        output_scale: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        scale_ue8m0: bool = True,
        active_mode: int = 0,
        hidden_size_override: int = None,
        input_stride_override: int = None,
        in_offset_elems: int = 0,
    ):
        """Quantize BF16 input to FP8 with block-wise scale.

        scale_ue8m0=True: output scale is packed UE8M0 uint32 (for FP8 linear GEMM)
        scale_ue8m0=False: output scale is float32 (for MoE group GEMM)

        hidden_size_override / input_stride_override / in_offset_elems support
        quantizing a column slice of a wider input buffer (QKV-a fused path).
        Defaults preserve legacy whole-row quantize. The OUTPUT buffer should
        be sized for the slice (hidden_size_override columns).
        """
        legacy_hidden_size = input.dim(input.num_dims - 1)
        row_count = 1
        for axis in range(input.num_dims - 1):
            row_count *= input.dim(axis)
        slice_override = (hidden_size_override is not None or
                          input_stride_override is not None or
                          in_offset_elems != 0)
        hidden_size = hidden_size_override or legacy_hidden_size
        if input_stride_override is None:
            input_stride_override = legacy_hidden_size
        # Pick grid so total CTAs (group_tiles * grid_y) stays ≤ num_workers
        # and the task fits in one persistent-runtime wave (~1 μs per-task
        # gap × multi-wave is the main scheduler overhead). For prefill with
        # row_count >= num_workers the kernel internally loops ROWS_PER_TASK
        # = ceil(row_count / grid_y) rows per CTA (see register_quantize_fp8);
        # for row_count < num_workers we let group_tiles soak up the spare
        # workers (each CTA then owns a smaller per-row group slice).
        total_workers = max(self.num_workers, 1)
        grid_y = min(row_count, total_workers)
        workers_per_row = max(1, total_workers // grid_y)
        group_tiles = self._fp8_quantize_group_tiles(
            hidden_size, scale_ue8m0, max_tiles=workers_per_row
        )
        grid_dim = (group_tiles, grid_y, 1)
        if slice_override:
            params = [active_mode, hidden_size,
                      input_stride_override, in_offset_elems]
        else:
            params = [] if active_mode == 0 else [active_mode]
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
        gate_mode: int = 0,
    ):
        params = [] if gate_mode == 0 else [gate_mode]
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

    def _fp8_group_gemm_layer_impl(
        self,
        task_name: str,
        a_fp8: DTensor,
        b_fp8: DTensor,
        sfa_packed: DTensor,
        sfb_packed: DTensor,
        m_indices: DTensor,
        output: DTensor,
        num_workers: int,
        meta: DTensor = None,
    ):
        """Shared registration helper for the SM100 grouped FP8 block-scaled
        GEMM tasks (`fp8_group_gemm_smallm_sm100` / `fp8_group_gemm_largem_sm100`).

        Computes  D[r, :] = (A[r, :] * scale_a[r]) @ (B[m_indices[r]].T * scale_b)
        with hardware UE8M0 dequant via `tcgen05.mma.kind::mxf8f6f4.block_scale`.
        Rows in each BM=128 block must share the same expert id.

        Shape symbols
        -------------
            M_total : total number of rows across all experts (must be a
                      multiple of BM=128; pad-rows can carry a dummy expert).
            K       : reduction dim (must be a multiple of BK=128).
            N       : per-expert output dim.
            E       : number of experts.
            nk       = ceil(K / 128)              UE8M0 scales per row.
            num_sf_k = ceil(nk / 4)               uint32-packed scale columns
                                                   (4 UE8M0 per uint32 along K).

        DTensor inputs / output
        -----------------------
        a_fp8       (M_total, K)            fp8_e4m3 (attached as uint8)
                    row-major, K innermost. Activations (already permuted so
                    that contiguous BM=128 row-blocks share one expert).

        b_fp8       (E, N, K)               fp8_e4m3 (attached as uint8)
                    row-major per expert (K innermost). The kernel flattens
                    the buffer to (E*N, K) for its TMA descriptor; same memory.

        sfa_packed  (num_sf_k, M_total)     uint32, UE8M0-packed
                    Row-major with M_total innermost (PyTorch shape order;
                    same memory the kernel's TMA descriptor describes with
                    g=(M_total, num_sf_k) in its innermost-first convention).
                    Each uint32 packs 4 consecutive UE8M0 scales along the
                    K-block axis (one scale per 128-K-element block per row).

        sfb_packed  (num_sf_k, E*N)         uint32, UE8M0-packed
                    Same packing convention as SFA. Built by expanding the
                    per-expert per-block scale [E, N/128, K/128] →
                    [E*N, K/128] (repeat_interleave along N) → pack to
                    [num_sf_k, E*N] uint32. One scale per output element per
                    128-K-element block (after expansion).

        m_indices   (M_total,)              int32
                    Expert id per A row. Rows in [bm*BM, (bm+1)*BM) for any
                    bm must share the same expert (only m_indices[bm*BM] is
                    read per block). For static permuted layouts this is
                    typically `arange(M_total) // BM_PADDING`.

        output      (M_total, N)            bf16
                    Row-major, N innermost. Written via TMA store.

        Other params
        ------------
        task_name   : "fp8_group_gemm_smallm_sm100" (BN=64, NS=8) or
                      "fp8_group_gemm_largem_sm100" (BN=128, NS=6); picks the
                      tile/stage variant. Dispatch policy lives in
                      `fp8_group_gemm_layer`.
        num_workers : grid_dim.x. Each task instance handles a stride of
                      (bm, bn) tiles `task_desc.task_metadata.request_id ::
                      num_workers`; pick `self.num_workers` so every worker
                      gets a slice.

        Partitioning
        ------------
        All six tensors are registered with input_map (-1,-1,-1): every task
        gets the full base pointer. Tile selection is internal to the kernel
        (driven by worker_idx + num_workers), not by MPK's TBGraph slicer.
        block_dim is fixed at (256, 1, 1) — 8 warps with hard-coded roles
        (TMA-load / UTCCP-transpose / MMA-issue / epilogue+TMA-store).
        """
        assert a_fp8.num_dims == 2
        assert b_fp8.num_dims == 3
        assert output.num_dims == 2
        M_total = a_fp8.dim(0)
        K = a_fp8.dim(1)
        E = b_fp8.dim(0)
        N = b_fp8.dim(1)
        assert b_fp8.dim(2) == K
        assert m_indices.dim(0) == M_total
        if meta is None:
            active_mask_offset = -1
        else:
            assert meta.num_dims == 2
            # meta layout: row 0 = out_weights+tok_to_perm (length M_total+MBT*TOPK).
            # Row 1's first E entries hold active_expert_mask (int32).
            # Flat offset of row 1: meta.dim(1) (since row 0 occupies that).
            active_mask_offset = meta.dim(1)
        params = [M_total, N, K, E, num_workers, active_mask_offset]
        grid_dim = (num_workers, 1, 1)
        block_dim = (256, 1, 1)  # 8 warps fixed by kernel role layout
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(a_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(b_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(sfa_packed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sfb_packed, (-1, -1, -1), -1, True)
        tb_graph.new_input(m_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        operators = [a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output]
        if meta is not None:
            tb_graph.new_input(meta, (-1, -1, -1), -1, True)
            operators.append(meta)
        self.kn_graph.customized(operators, tb_graph)
        self.kn_graph.register_task(tb_graph, task_name, params)

    def fp8_group_gemm_smallm_layer(
        self, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
        num_workers, meta=None,
    ):
        # Smallm variant: BN=64, NS=8. Best for K>4096 && MPE<=8 (gate_up
        # M{1,4,8} on DSv3). MoE decode niche.
        self._fp8_group_gemm_layer_impl(
            "fp8_group_gemm_smallm_sm100",
            a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
            num_workers, meta=meta)

    def fp8_group_gemm_largem_layer(
        self, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
        num_workers, meta=None,
    ):
        # Largem variant: BN=128, NS=6. Default for everything outside the
        # smallm niche (most MoE configs incl. all prefill MPE >= 16 and any
        # K <= 4096 layer like down_proj).
        self._fp8_group_gemm_layer_impl(
            "fp8_group_gemm_largem_sm100",
            a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
            num_workers, meta=meta)

    def fp8_group_gemm_layer(
        self, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
        num_workers, meta=None,
    ):
        # Auto-dispatcher: pick smallm/largem by (K, M_per_expert).
        K = a_fp8.dim(1)
        M_total = a_fp8.dim(0)
        E = b_fp8.dim(0)
        MPE = M_total // E
        if K > 4096 and MPE <= 8:
            self.fp8_group_gemm_smallm_layer(
                a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
                num_workers, meta=meta)
        else:
            self.fp8_group_gemm_largem_layer(
                a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
                num_workers, meta=meta)

    def moe_permute_sm100_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        topk_weights: DTensor,
        routing_indices: DTensor,
        permuted_fp8: DTensor,
        permuted_scale: DTensor,
        meta: DTensor,
        bm_padding: int = 128,
    ):
        """MoE expand-permute-sort task — peripheral glue for the PR-674
        grouped FP8 GEMM. See moe_permute_sm100.cuh for the exact contract.

        One CTA per local expert (grid_dim = (E_local, 1, 1)). Scans
        routing_indices[my_expert, :], gathers matched tokens, and copies
        FP8 row + UE8M0-packed scale into the permuted layout. Small
        per-row metadata (permuted_weights + token_to_permuted) is packed
        into one int32 `meta` buffer so the task stays within MPK's
        3-outputs-per-task limit:

          meta[0       : M_TOTAL]            = permuted_weights (f32 bits)
          meta[M_TOTAL : M_TOTAL + MBT*TOPK] = token_to_permuted (row + 1;
                                                  0 = not routed locally;
                                                  caller must tensor_init
                                                  zero this region each
                                                  iter).

        `m_indices` is a STATIC constant the builder sets up once via
        attach_input (pattern: m_indices[r] = r / BM_PADDING). It is fed
        directly to the grouped FP8 GEMM and is NOT a per-iter output.

        IMPORTANT: input_scale must be UE8M0-PACKED uint32 (produced by
        quantize_fp8_layer with scale_ue8m0=True).
        """
        assert input_fp8.num_dims == 2
        assert input_scale.num_dims == 2
        assert topk_weights.num_dims == 2
        assert routing_indices.num_dims == 2
        assert permuted_fp8.num_dims == 2
        assert permuted_scale.num_dims == 2
        # meta is shaped (2, M_TOTAL + MBT*TOPK) int32 — see builder.py for
        # the BATCH_SIZE=2 rationale (full-byte tensor_init).
        assert meta.num_dims == 2
        assert meta.dim(0) == 2

        K = input_fp8.dim(1)
        K_PACKED = input_scale.dim(1)
        MBT = input_fp8.dim(0)
        TOPK = topk_weights.dim(1)
        E_LOCAL = routing_indices.dim(0)
        M_TOTAL = E_LOCAL * bm_padding
        assert routing_indices.dim(1) == MBT
        assert topk_weights.dim(0) == MBT
        assert permuted_fp8.dim(0) == M_TOTAL
        assert permuted_fp8.dim(1) == K
        assert permuted_scale.dim(0) == K_PACKED
        assert permuted_scale.dim(1) == M_TOTAL
        assert meta.dim(1) == M_TOTAL + MBT * TOPK, (
            f"meta length must be {M_TOTAL + MBT * TOPK}, got {meta.dim(1)}")

        params = [K, K_PACKED, MBT, TOPK, E_LOCAL, bm_padding]
        grid_dim = (E_LOCAL, 1, 1)
        block_dim = (128, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weights, (-1, -1, -1), -1, True)
        # routing_indices: (-1, -1, -1) so the kernel sees the FULL (E_LOCAL, MBT)
        # buffer and computes its expert row from task_metadata.expert_offset.
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(permuted_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(permuted_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(meta, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input_fp8, input_scale, topk_weights, routing_indices,
             permuted_fp8, permuted_scale, meta], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_permute_sm100", params)

    def moe_unpermute_sm100_layer(
        self,
        permuted_output: DTensor,
        meta: DTensor,
        residual: DTensor,
        output: DTensor,
    ):
        """MoE combine-unpermute task — inverse of moe_permute_sm100. See
        moe_unpermute_sm100.cuh for the contract. One CTA per token
        (grid_dim = (MBT, 1, 1)). Decodes `meta` into permuted_weights +
        token_to_permuted, then writes
        `output[t] = residual[t] +
                     sum_k(permuted_output[token_to_permuted[t,k]-1]
                            * permuted_weights[same row])`.
        """
        assert permuted_output.num_dims == 2
        # meta is shaped (2, M_TOTAL + MBT*TOPK) int32 — see
        # moe_permute_sm100_layer for the layout contract.
        assert meta.num_dims == 2
        assert meta.dim(0) == 2
        assert residual.num_dims == 2
        assert output.num_dims == 2

        MBT = residual.dim(0)
        HIDDEN = permuted_output.dim(1)
        M_TOTAL = permuted_output.dim(0)
        # meta = M_TOTAL (weights) + MBT*TOPK (token_to_permuted) entries.
        meta_len = meta.dim(1)
        TOPK = (meta_len - M_TOTAL) // MBT
        assert M_TOTAL + MBT * TOPK == meta_len
        assert residual.dim(1) == HIDDEN
        assert output.dim(0) == MBT
        assert output.dim(1) == HIDDEN

        params = [MBT, TOPK, HIDDEN, M_TOTAL]
        grid_dim = (MBT, 1, 1)
        block_dim = (128, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # All inputs/outputs are (-1, -1, -1) so the kernel sees the FULL
        # tensors and indexes them with task_metadata.request_id (= my_token).
        tb_graph.new_input(permuted_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(meta, (-1, -1, -1), -1, True)
        tb_graph.new_input(residual, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [permuted_output, meta, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_unpermute_sm100", params)

    def transpose_scale_sm100_layer(
        self, scale_in: DTensor, scale_out: DTensor,
    ):
        """(M, K_PACKED) uint32 → (K_PACKED, M) uint32 via single-CTA copy.

        Bridges quantize_fp8_layer's M-outermost packed scale output to the
        K-outermost layout the PR-674 fp8_group_gemm SFA/SFB TMA descriptors
        require. See transpose_scale_sm100.cuh.
        """
        assert scale_in.num_dims == 2
        assert scale_out.num_dims == 2
        M = scale_in.dim(0)
        K_PACKED = scale_in.dim(1)
        assert scale_out.dim(0) == K_PACKED
        assert scale_out.dim(1) == M
        params = [M, K_PACKED]
        grid_dim = (1, 1, 1)
        block_dim = (128, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(scale_in, (-1, -1, -1), -1, True)
        tb_graph.new_input(scale_out, (-1, -1, -1), -1, True)
        self.kn_graph.customized([scale_in, scale_out], tb_graph)
        self.kn_graph.register_task(tb_graph, "transpose_scale_sm100", params)

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
        gate_mode: int = 0,
    ):
        params = [1] if gate_mode == 0 else [1, gate_mode]
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

    def linear_fp8_swapAB_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        gate_mode: int = 0,
    ):
        # MPK-native FP8 linear (swapAB inside the kernel). Same Python-layer
        # API as linear_fp8_layer; the kernel maps weight->A and input->B.
        # Constraints (asserted at registration time):
        #   per-task output size (output.dim[1] / grid_dim.x) must be a
        #   multiple of 128, and batch_size must be <= 16 (decode-only).
        params = [] if gate_mode == 0 else [gate_mode]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8, (0, -1, -1), -1, True)
        tb_graph.new_input(weight_scale, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear_fp8_swapAB_sm100", params)

    def linear_fp8_swapAB_with_residual_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        gate_mode: int = 0,
    ):
        params = [1] if gate_mode == 0 else [1, gate_mode]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8, (0, -1, -1), -1, True)
        tb_graph.new_input(weight_scale, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale, residual, output],
            tb_graph)
        self.kn_graph.register_task(
            tb_graph, "linear_fp8_swapAB_with_residual_sm100", params)

    def assemble_q_decode_sm100_layer(
        self,
        q_nope_abs: DTensor,
        q_pe: DTensor,
        q_nope_pe: DTensor,
        grid_dim: tuple,
        block_dim: tuple = (128, 1, 1),
        pe_only: bool = False,
    ):
        """Interleave the BMM-absorbed q_nope (N, H, 512) with q_pe (N, H, 64)
        into per-head [nope|pe] layout (N, H, 576) for MLA decode.

        Used by the MPK_DSV3_BMM=1 decode Q path:
          rmsnorm_linear(q_a, q_b_nope) → q_nope (N, H, 128)
          quantize_fp8(q_nope)         → q_nope_fp8 (N, H, 128)
          linear_fp8_bmm_sm100(q_nope_fp8, kv_b_k_bmm) → q_nope_abs (N, H, 512)
          rmsnorm_linear(q_a, q_b_pe)  → q_pe (N, H, 64)
          assemble_q_decode_sm100(q_nope_abs, q_pe) → q_nope_pe (N, H, 576)

        Replaces the load-time absorbed q_b_proj GEMM. The BMM is per-head and
        loads smaller weights ((H, 512, 128) per head) vs the absorbed (H*576, q_lora)
        monolith, which is the perf win — smaller TMA traffic per task.

        grid_dim = (N, 1, 1); each CTA processes 1 token (all H heads).
        block_dim = (128, 1, 1) is plenty: at TP=4 (H=32) each CTA writes
        32*576 = 18432 bf16 elements = 144 elements/thread.
        """
        assert q_nope_abs.num_dims == 3
        assert q_pe.num_dims == 3
        # q_nope_pe may be 3D (N, H, D_TOTAL) or 2D (N, H*D_TOTAL) — same
        # byte layout, the register code handles both. 2D is convenient so
        # the existing q_nope_pe buffer doesn't need to be reshaped.
        assert q_nope_pe.num_dims in (2, 3)
        assert q_nope_abs.dim(0) == q_pe.dim(0) == q_nope_pe.dim(0)
        H = q_nope_abs.dim(1)
        assert q_pe.dim(1) == H
        D_TOTAL = q_nope_abs.dim(2) + q_pe.dim(2)
        if q_nope_pe.num_dims == 3:
            assert q_nope_pe.dim(1) == H
            assert q_nope_pe.dim(2) == D_TOTAL
        else:
            assert q_nope_pe.dim(1) == H * D_TOTAL
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope_abs, (0, -1, -1), -1, True)
        tb_graph.new_input(q_pe,       (0, -1, -1), -1, True)
        tb_graph.new_input(q_nope_pe,  (0, -1, -1), -1, True)
        self.kn_graph.customized([q_nope_abs, q_pe, q_nope_pe], tb_graph)
        params = [1] if pe_only else []
        self.kn_graph.register_task(tb_graph, "assemble_q_decode_sm100", params)

    def linear_fp8_bmm_sm100_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        output: DTensor,
        grid_dim: tuple,    # (m_shards_per_head, h_shards, 1)
        block_dim: tuple,   # (256, 1, 1) on SM100
    ):
        # Per-head FP8 batched matmul on SM100. Computes
        #     output[n, h, :] = input[n, h, :] @ weight[h, :, :]^T  (per head)
        # decode-only, batch_size <= 16. The H dimension is exposed as an
        # explicit workload split (grid.y) on top of the existing swapAB
        # M-tile split (grid.x). First cut requires grid.y == H — one head
        # per CTA — so the kernel stays a thin forward to the swapAB GEMM.
        #
        # Tensor layouts (all 3D; dim 1 is the head axis):
        #   input_fp8     [N, H, D_in]
        #   input_scale   [N, H, packed_K]   uint32 UE8M0 (4 logical scales / uint32)
        #   weight_fp8    [H, D_out, D_in]
        #   weight_scale  [H, D_out, packed_K]
        #   output        [N, H, D_out]
        #
        # Constraints (asserted at registration time):
        #   - D_out / grid.x must be a multiple of MMA_M=128
        #   - D_in must be a multiple of BLOCK_K=128
        #   - batch_size N <= MMA_N=16 (decode-only)
        #   - H % grid.y == 0; first cut requires H_PER_TASK == 1
        # Weight stays 3D (H, D_out, D_in) — the per-head TMA stride depends
        # on the explicit H dim. Input/output may be 2D (N, H*D_*) or 3D
        # (N, H, D_*); same byte layout, partition map adjusts the dim index.
        assert weight_fp8.num_dims == 3
        assert weight_scale.num_dims == 3
        assert input_fp8.num_dims in (2, 3)
        assert input_scale.num_dims in (2, 3)
        assert output.num_dims in (2, 3)
        params = []
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        in_h_axis = 1 if input_fp8.num_dims == 3 else 1
        in_sc_h_axis = 1 if input_scale.num_dims == 3 else 1
        out_h_axis = 1
        out_m_axis = 2 if output.num_dims == 3 else 1
        # input_fp8 / input_scale: grid.y splits the head axis. For 3D
        # (N, H, D_in), head axis is dim 1; for 2D (N, H*D_in), head
        # axis is also dim 1 because the partition map's axis index
        # refers to the DTensor's dim sequence; dim 1 still partitions
        # into H equal slices of D_in each (per-CTA STensor.dim[1] = D_in).
        tb_graph.new_input(input_fp8,    (-1, in_h_axis, -1), -1, True)
        tb_graph.new_input(input_scale,  (-1, in_sc_h_axis, -1), -1, True)
        # weight_fp8 / weight_scale [H, D_out, D_in or packed_K]:
        # grid.x splits dim 1 (D_out), grid.y splits dim 0 (H).
        tb_graph.new_input(weight_fp8,   (1, 0, -1), -1, True)
        tb_graph.new_input(weight_scale, (1, 0, -1), -1, True)
        # output: dim 1 (H) split by grid.y. For 3D, dim 2 (D_out) split
        # by grid.x; for 2D, dim 1 (H*D_out) is also split — but the
        # partition needs the SAME dim for both H and D_out splits, which
        # only works in 3D form. For 2D output, grid.x must be 1.
        if output.num_dims == 3:
            tb_graph.new_input(output, (out_m_axis, out_h_axis, -1), -1, True)
        else:
            assert grid_dim[0] == 1, (
                "linear_fp8_bmm with 2D output requires grid.x=1 "
                "(D_out cannot be sharded across CTAs when packed flat)")
            tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear_fp8_bmm_sm100", params)

    def _fp8_gemm_dense_layer_impl(
        self,
        task_name: str,
        input_fp8: DTensor,
        weight_fp8: DTensor,
        input_scale: DTensor,
        weight_scale: DTensor,
        output: DTensor,
        num_workers: int,
        runtime_m_mode: int = 0,
    ):
        # A: [M,K], B: [N,K], C: [M,N]. The kernel distributes output tiles
        # across `num_workers` persistent tasks. Inputs/output may also be
        # 3D (M, H_split, K/H_split or D_out/H_split) when the caller wants
        # to keep the head dimension explicit downstream (e.g. for BMM); the
        # GEMM kernel itself sees the buffer as flat M*K / M*N bytes via TMA.
        assert input_fp8.num_dims in (2, 3)
        assert weight_fp8.num_dims == 2
        assert input_scale.num_dims == 2
        assert weight_scale.num_dims == 2
        assert output.num_dims in (2, 3)
        M = input_fp8.dim(0)
        K = (input_fp8.dim(1) if input_fp8.num_dims == 2
             else input_fp8.dim(1) * input_fp8.dim(2))
        N = weight_fp8.dim(0)
        assert weight_fp8.dim(1) == K
        assert output.dim(0) == M
        out_flat_n = (output.dim(1) if output.num_dims == 2
                      else output.dim(1) * output.dim(2))
        assert out_flat_n == N
        params = [M, N, K, num_workers]
        if runtime_m_mode:
            params.append(runtime_m_mode)
        tb_graph = TBGraph(CyTBGraph((num_workers, 1, 1), (256, 1, 1), 1, 64))
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input_fp8, weight_fp8, input_scale, weight_scale, output],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, task_name, params)

    def fp8_gemm_dense_smallm_layer(self, input_fp8, weight_fp8, input_scale,
                                    weight_scale, output, num_workers,
                                    runtime_m_mode: int = 0):
        self._fp8_gemm_dense_layer_impl(
            "fp8_gemm_dense_smallm_sm100",
            input_fp8, weight_fp8, input_scale, weight_scale, output,
            num_workers, runtime_m_mode=runtime_m_mode)

    def fp8_gemm_dense_mediumm_layer(self, input_fp8, weight_fp8, input_scale,
                                     weight_scale, output, num_workers,
                                     runtime_m_mode: int = 0):
        self._fp8_gemm_dense_layer_impl(
            "fp8_gemm_dense_mediumm_sm100",
            input_fp8, weight_fp8, input_scale, weight_scale, output,
            num_workers, runtime_m_mode=runtime_m_mode)

    def linear_splitk_swapAB_fp8_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        output: DTensor,
        grid_dim: tuple,    # (num_M_shards, split_k_factor, 1)
        block_dim: tuple,   # (256, 1, 1) on SM100
        *,
        accumulate: bool,
    ):
        # Split-K variant of linear_fp8_swapAB_layer. grid.y CTAs each compute
        # a K-slice partial and TMA reduce-add into the shared output tile.
        #
        # The kernel uses tma_reduce_add_async and unconditionally adds onto
        # whatever `output` already contains. The `accumulate` flag selects:
        #   accumulate=True  -> caller owns `output` (e.g. residual). The
        #                       matmul is added on top; no tensor_init.
        #   accumulate=False -> layer prepends a tensor_init that zeroes
        #                       `output` first, so the result is a pure sum.
        # tensor_init shares the linear's grid_dim and per-tensor input_maps,
        # so grid.y CTAs zero the same tile redundantly (kept for dep-edge
        # alignment with the linear).
        #
        # Constraints (asserted at registration time):
        #   - output.dim[1] / grid.x must be a multiple of 128 (per-task N)
        #   - input.dim[1]  / grid.y must be a multiple of 128 (per-task K)
        #   - batch_size <= 16 (decode-only)
        if not accumulate:
            self.tensor_init_layer(
                target=output,
                dummy=input_fp8,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dummy_input_map=(-1, 1, -1),
                target_input_map=(1, -1, -1),
            )
        params = []
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # input_fp8 [batch, K]: grid.y splits K (dim 1).
        tb_graph.new_input(input_fp8, (-1, 1, -1), 1, True)
        # input_scale [batch, packed_K]: same K-split.
        tb_graph.new_input(input_scale, (-1, 1, -1), 1, True)
        # weight_fp8 [output, K]: grid.x splits output (dim 0), grid.y splits K (dim 1).
        tb_graph.new_input(weight_fp8, (0, 1, -1), 1, True)
        # weight_scale [output, packed_K]: same partition as weight.
        tb_graph.new_input(weight_scale, (0, 1, -1), 1, True)
        # output [batch, output]: grid.x splits dim 1; grid.y does NOT
        # partition (all grid.y CTAs reduce-add into the same M-shard).
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "splitk_linear_fp8_swapAB_sm100", params)

    def moe_silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        meta: DTensor = None,
        bm_padding: int = 128,
    ):
        # Accepts both:
        #   3D (batch, num_experts_per_tok, intermediate) — OLD MoE path.
        #   2D (M_total, intermediate) — NEW MoE path. Treated as (M_total, 1, N)
        #   inside the task_register codegen (num_experts_per_tok = 1).
        #
        # When ``meta`` is supplied (NEW MoE only), the kernel early-returns
        # if active_expert_mask[my_expert] == 0 so we skip the entire
        # silu+mul work for inactive-expert blocks. The mask lives at
        # meta[1, 0:E_LOCAL] (= flat offset meta.dim(1)). The CTA→expert
        # mapping depends on how grid.x partitions the M dim:
        #   ctas_per_expert = bm_padding // rows_per_cta
        #   my_expert       = bid.x / ctas_per_expert
        # For the standard NEW MoE config (grid.x == num_local_experts,
        # bm_padding == rows_per_cta == 128), ctas_per_expert == 1 and
        # my_expert == bid.x.
        assert input.num_dims in (2, 3)
        assert output.num_dims == input.num_dims
        if meta is None:
            active_mask_offset = -1
            ctas_per_expert = 0
        else:
            assert meta.num_dims == 2
            assert grid_dim[1] == 1 and grid_dim[2] == 1, (
                "active-mask skip requires grid.y == grid.z == 1")
            rows_per_cta = max(1, input.dim(0) // grid_dim[0])
            ctas_per_expert = max(1, bm_padding // rows_per_cta)
            active_mask_offset = meta.dim(1)
        params = [active_mask_offset, ctas_per_expert]
        if input.num_dims == 3:
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(input, (0, 1, -1), -1, True)
            tb_graph.new_input(output, (0, 1, -1), -1, True)
        else:
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(input, (0, -1, -1), -1, True)
            tb_graph.new_input(output, (0, -1, -1), -1, True)
        operators = [input, output]
        if meta is not None:
            tb_graph.new_input(meta, (-1, -1, -1), -1, True)
            operators.append(meta)
        self.kn_graph.customized(operators, tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_silu_mul", params)
            
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
        *,
        accumulate: bool,
    ):
        # The BF16 splitk kernel uses tma_reduce_add_async and unconditionally
        # adds onto whatever `output` already contains. The `accumulate` flag
        # selects:
        #   accumulate=True  -> caller owns `output` (e.g. residual). The
        #                       matmul is added on top; no tensor_init.
        #   accumulate=False -> layer prepends a tensor_init that zeroes
        #                       `output` first, so the result is a pure sum.
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        if not accumulate:
            self.tensor_init_layer(
                target=output,
                dummy=input,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dummy_input_map=(-1, 1, -1),
                target_input_map=(1, -1, -1),
            )
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
        residual: DTensor = None,
        gate_mode: int = 0,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size)
        assert buffer.num_dims == 3  # (world_size, batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        if residual is not None:
            assert residual.num_dims == 2  # (batch_size, hidden_size)
            assert residual.dim(0) == output.dim(0)
            assert residual.dim(1) == output.dim(1)
        # params[0]: num_gpus
        # params[1]: my_gpu_id
        best_implementation = auto_select_allreduce_implementation(self.world_size, self.mpi_rank)
        tensors = {
            "input": input,
            "buffer": buffer,
            "output": output,
        }
        if residual is not None:
            tensors["residual"] = residual
        params = [self.world_size, self.mpi_rank]
        if gate_mode:
            if getattr(best_implementation, "name", "") != "nvshmem_tile_allreduce":
                raise RuntimeError(
                    "Gated allreduce is currently implemented only for "
                    "nvshmem_tile_allreduce.")
            params.append(gate_mode)
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
        noop: bool = False,
    ):
        # When ``noop`` is True we still register the task graph node (so
        # downstream task-graph constraints — case-3 fork+join — are
        # preserved) but the codegen emits an empty kernel body. Use for
        # phantom-bridge identities where the output buffer is only
        # plumbed for the dependency edge, not actually consumed.
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
        params = [1 if noop else 0]
        self.kn_graph.register_task(tb_graph, "identity", params)

    def elementwise_add_layer(
        self,
        input_a: DTensor,
        input_b: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        in_a_row_stride: int = None,
        in_a_col_offset: int = 0,
    ):
        """Element-wise add: output = input_a + input_b.

        in_a_row_stride / in_a_col_offset let input_a read a column slice
        of a wider buffer (e.g. dump qkv_a_out's q_a slice into a (mbt, 1536)
        tensor). Defaults keep the legacy (matching shapes) behaviour.
        """
        assert input_a.num_dims == 2
        assert input_b.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_a, (0, -1, -1), -1, True)
        tb_graph.new_input(input_b, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized([input_a, input_b, output], tb_graph)
        slice_override = (in_a_row_stride is not None or
                          in_a_col_offset != 0)
        if slice_override:
            if in_a_row_stride is None:
                in_a_row_stride = input_a.dim(1)
            self.kn_graph.register_task(
                tb_graph, "elementwise_add_sm100",
                [in_a_row_stride, in_a_col_offset])
        else:
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

    def nvshmem_global_argmax_layer(
        self,
        partial_value: DTensor,
        partial_index: DTensor,
        scratch_value: DTensor,
        scratch_index: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        vocab_offset: int,
        valid_vocab_size: int,
        partial_chunk_size: int,
    ):
        assert self.world_size > 1
        assert self.use_nvshmem
        assert partial_value.num_dims == 2  # (batch_size, num_partial_tasks)
        assert partial_index.num_dims == 2  # (batch_size, num_partial_tasks)
        assert scratch_value.num_dims == 2  # (world_size, batch_size)
        assert scratch_index.num_dims == 2  # (world_size, batch_size)
        assert output.num_dims == 2  # (batch_size, 1)
        assert partial_value.dim(0) == partial_index.dim(0)
        assert partial_value.dim(1) == partial_index.dim(1)
        assert scratch_value.dim(0) == self.world_size
        assert scratch_index.dim(0) == self.world_size
        assert scratch_value.dim(1) == partial_value.dim(0)
        assert scratch_index.dim(1) == partial_value.dim(0)
        assert partial_chunk_size > 0
        assert 0 <= valid_vocab_size <= partial_value.dim(1) * partial_chunk_size

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial_value, (1, 0, -1), -1, True)
        tb_graph.new_input(partial_index, (1, 0, -1), -1, True)
        tb_graph.new_input(scratch_value, (-1, -1, -1), -1, True)
        tb_graph.new_input(scratch_index, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [partial_value, partial_index, scratch_value, scratch_index, output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph,
            "nvshmem_global_argmax",
            [
                self.world_size,
                self.mpi_rank,
                vocab_offset,
                valid_vocab_size,
                partial_chunk_size,
            ],
        )
        allocate_nvshmem_teams(self, grid_dim[0] * grid_dim[1] * grid_dim[2])

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
            so_output_path = os.path.join(output_dir, f"mpk_launcher_rank{self.mpi_rank}.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so")

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
        print("Compiling megakernel using the following command line:")
        print(cc_cmd)
        if os.environ.get("MPK_SERIALIZE_NVCC", "0") == "1":
            import fcntl

            lock_path = os.environ.get(
                "MPK_NVCC_LOCK_PATH",
                os.path.join(tempfile.gettempdir(), "mirage_mpk_nvcc.lock"),
            )
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                subprocess.check_call(cc_cmd)
        else:
            subprocess.check_call(cc_cmd)
        
        # Copy .so to output_dir if specified
        if output_dir is not None:
            so_output_path = os.path.join(output_dir, f"mpk_launcher_rank{self.mpi_rank}.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so")
            shutil.copy(so_path, so_output_path)
            print(f"Saved compiled kernel to: {so_output_path}")
            
            # Save kernel metadata for compatibility validation during load
            metadata_path = os.path.join(output_dir, f"kernel_metadata_rank{self.mpi_rank}.json")
            self._save_kernel_metadata(metadata_path)

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
        
        # Build model tensor name/pointer lists for runtime tensor lookup
        model_tensor_names = list(self._model_tensors.keys())
        model_tensor_ptrs = [t.data_ptr() for t in self._model_tensors.values()]
        
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
            model_tensor_names,
            model_tensor_ptrs,
            "",  # Empty JSON path = use __FILE__ based path during initial compile
        )

        self._is_compiled = True

        # self.call_func = getattr(mod, "call_func")

    def load_mpk_kernel(
        self,
        output_dir: str,
        **kwargs,
    ):
        """
        Load a pre-compiled MPK kernel from output_dir instead of recompiling.
        
        Args:
            output_dir: Directory containing the pre-compiled kernel files:
                       - mpk_launcher_rank{N}.cpython-*.so
                       - task_graph_rank{N}.json
        """
        assert not self._is_compiled, "Kernel is already compiled"
        
        # Find the compiled .so file
        so_pattern = f"mpk_launcher_rank{self.mpi_rank}.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
        so_path = os.path.join(output_dir, so_pattern)
        
        if not os.path.exists(so_path):
            raise FileNotFoundError(
                f"Pre-compiled kernel not found at {so_path}. "
                f"Run compile(output_dir='{output_dir}') first to generate it."
            )
        
        json_path = os.path.join(output_dir, f"task_graph_rank{self.mpi_rank}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Task graph not found at {json_path}. "
                f"Run compile(output_dir='{output_dir}') first to generate it."
            )
        
        # Validate kernel compatibility if metadata exists
        metadata_path = os.path.join(output_dir, f"kernel_metadata_rank{self.mpi_rank}.json")
        skip_validation = kwargs.get("skip_validation", False)
        if os.path.exists(metadata_path) and not skip_validation:
            self._validate_kernel_compatibility(metadata_path)
            print(f"[load_mpk_kernel] Kernel compatibility check passed!")
        elif not skip_validation:
            print(f"[load_mpk_kernel] Warning: No kernel metadata found. Skipping validation.")
        
        print(f"[load_mpk_kernel] Loading launcher from: {so_path}")
        print(f"[load_mpk_kernel] Using task graph JSON: {json_path}")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")
        self.launch_func = getattr(mod, "launch_func")
        self.init_request_func = getattr(mod, "init_request_func")
        self.finalize_func = getattr(mod, "finalize_func")
        
        # Prepare meta tensors
        meta_tensors = list()
        meta_tensors.append(self.meta_tensors["step"])
        meta_tensors.append(self.meta_tensors["tokens"])
        meta_tensors.append(self.meta_tensors["input_tokens"])
        meta_tensors.append(self.meta_tensors["output_tokens"])
        meta_tensors.append(self.meta_tensors["num_new_tokens"])
        meta_tensors.append(self.meta_tensors["prompt_lengths"])
        meta_tensors.append(self.meta_tensors["qo_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indices_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_last_page_len_buffer"])
        meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
        profiler_buffer_ptr = (
            self.profiler_tensor.data_ptr() if self.profiler_tensor is not None else 0
        )
        
        self.eos_token_id = kwargs.get("eos_token_id", self.eos_token_id)
        
        # Build model tensor name/pointer lists for runtime tensor lookup
        model_tensor_names = list(self._model_tensors.keys())
        model_tensor_ptrs = [t.data_ptr() for t in self._model_tensors.values()]
        
        print(f"[load_mpk_kernel] Passing {len(model_tensor_names)} model tensors to kernel")
        
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
            model_tensor_names,
            model_tensor_ptrs,
            json_path,  # Pass the JSON path for kernel reuse
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
            from .profiler_persistent import export_to_csv, export_to_perfetto_trace

            if self.trace_name:
                stem = self.trace_name
            else:
                stem = f"mirage_{self.mpi_rank}"

            export_to_perfetto_trace(
                self.profiler_tensor, stem + ".perfetto-trace"
            )
            export_to_csv(self.profiler_tensor, stem + ".csv")

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        if self._is_compiled:
            self.finalize_func()
        self.__finalized__ = True
