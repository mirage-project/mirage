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
  auto_select_allreduce_implementation
)
from typing import Optional

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <vector>

extern std::string g_task_graph_json_path;

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

  // MPK's workers and schedulers spin-wait on each other and never yield an
  // SM, so the whole grid must be co-resident; if it is not, the megakernel
  // deadlocks forever and holds the GPU through SIGTERM. Probe first and turn
  // that into an immediate, actionable error. MPK_SKIP_RESIDENCY_CHECK=1
  // opts out.
  if (getenv("MPK_SKIP_RESIDENCY_CHECK") != NULL) {
    // The ONLY fail-open path, and it is logged so it can never be a silent
    // default.
    fprintf(stderr,
            "[MPK] WARNING: MPK_SKIP_RESIDENCY_CHECK=1 -- launching WITHOUT "
            "verifying that the megakernel's grid can be co-resident. If it "
            "cannot, this launch deadlocks the GPU instead of raising.\\n");
  } else {
    int missing = 0;
    char probe_err[256];
    probe_err[0] = '\\0';
    Py_BEGIN_ALLOW_THREADS
    // Retry a couple of times: a short-lived co-tenant should cost a retry,
    // not a failed run. Sustained contention still fails. A probe-
    // infrastructure error is TERMINAL -- never retried into a success,
    // because a probe that could not run has not shown anything is resident.
    for (int attempt = 0; attempt < 3; attempt++) {
      missing = check_persistent_kernel_residency(
          0.25, probe_err, (int)sizeof(probe_err));
      if (missing == MPK_RESIDENCY_PROBE_ERROR || missing == 0) {
        break;
      }
    }
    Py_END_ALLOW_THREADS
    if (missing == MPK_RESIDENCY_PROBE_ERROR) {
      PyErr_Format(PyExc_RuntimeError,
                   "MPK residency check could not run: %s. Refusing to launch "
                   "rather than assume the megakernel's grid is co-resident -- "
                   "a wrong assumption deadlocks the GPU with a kernel that "
                   "does not die on SIGTERM. Fix the CUDA error, or set "
                   "MPK_SKIP_RESIDENCY_CHECK=1 to launch without the check.",
                   probe_err);
      return NULL;
    }
    if (missing > 0) {
      PyErr_Format(PyExc_RuntimeError,
                   "MPK residency check failed: %d of the megakernel's blocks "
                   "could not become co-resident on this GPU. The persistent "
                   "kernel claims every SM and its blocks spin-wait on each "
                   "other, so launching now would deadlock the GPU instead of "
                   "running. Give MPK an EXCLUSIVE GPU (check "
                   "`nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory "
                   "--format=csv` and pin CUDA_VISIBLE_DEVICES), or set "
                   "MPK_SKIP_RESIDENCY_CHECK=1 to launch anyway.",
                   missing);
      return NULL;
    }
  }

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

valid_persistent_kernel_modes = {"offline", "online", "online_notoken", "onepass", "online_multi_turn", "online_pinned"}

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
        "-rdc=false" if not use_nvshmem else "-rdc=true",
        # M2-I5 flagged that the megakernel JIT carries -use_fast_math while the
        # unit-test builds do not, so an exact-token gate validated on a unit
        # build would not be validating what actually ships. MPK_NO_FAST_MATH=1
        # builds the same graph without it, which is how M2-I9 measured whether
        # AC-3 exactness or decode latency depends on the flag. Default is
        # unchanged, so no existing caller is affected.
        *([] if os.environ.get("MPK_NO_FAST_MATH") == "1" else ["-use_fast_math"]),
        # M4-I2's A/B arm. With MPK_FP8_DENSE_BASELINE=1 the Qwen3.5 builder
        # dispatches the dense fp8 GEMM at the old slice-128 grid AND this -D
        # makes linear_fp8_blockscale_sm100.cuh compile the golden path, so the
        # generated task-279 code is the pre-M4-I2 code. That lets both arms of
        # the A/B come from one tree and interleave inside one GPU claim; the
        # knob must be part of the TU, hence a -D and not a runtime check.
        *(["-DMPK_FP8_DENSE_BASELINE=1"]
          if os.environ.get("MPK_FP8_DENSE_BASELINE") == "1" else []),
        # M4-I7's A/B arm, same shape as M4-I2's above. With
        # MPK_MOE_BLOCKSCALE_BASELINE=1 the grouped MoE GEMM (tasks 241/242)
        # compiles its frozen golden body, so the generated code is the
        # pre-M4-I7 code and both arms come from one tree.
        # MPK_MOE_PATH_POLICY=<0|1|2> pins one fetch path instead of the shipped
        # expert_stride rule -- the sweep knob that made the "PATH 1 dominates
        # PATH 0 in MPK" claim falsifiable.
        *(["-DMPK_MOE_BLOCKSCALE_BASELINE=1"]
          if os.environ.get("MPK_MOE_BLOCKSCALE_BASELINE") == "1" else []),
        *([f"-DMPK_MOE_PATH_POLICY={os.environ['MPK_MOE_PATH_POLICY']}"]
          if os.environ.get("MPK_MOE_PATH_POLICY") in ("0", "1", "2") else []),
        # M4-I8's two SCHEDULER arms, both default-off and both compile-time
        # because they live inside the worker loop of the one megakernel TU.
        # MPK_EVENT_WAIT_GPU_SCOPE=1  -- poll a local event's counter with
        #   ld.acquire.gpu instead of ld.acquire.sys (arm S).  The counter is
        #   written by atom.add.release.gpu, so .sys is a scope mismatch that
        #   every task pays on the kernel's hottest spin.
        # MPK_WORKER_OOO_POP=1 -- let a worker run the first READY task in its
        #   already-loaded task-desc buffer instead of blocking on the head
        #   (arm O).  The measured step spends 27-52% of its time on the
        #   critical chain waiting behind unrelated work in its own queue.
        *(["-DMPK_EVENT_WAIT_GPU_SCOPE=1"]
          if os.environ.get("MPK_EVENT_WAIT_GPU_SCOPE") == "1" else []),
        *(["-DMPK_WORKER_OOO_POP=1"]
          if os.environ.get("MPK_WORKER_OOO_POP") == "1" else []),
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
    elif mpk.mode == "online_pinned":
        flags = flags + ["-DMODE_ONLINE_PINNED",
                         f"-DMPK_PINNED_RING_CAPACITY={mpk.pinned_ring_capacity}"]
    else:
        raise ValueError(f"Invalid persistent kernel mode: {mpk.mode}")

    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}"]

    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}"]
    # M3-I9: emitted ONLY when asked for, so an unset knob leaves the compile
    # command byte-identical to the pre-M3-I9 one.
    if getattr(mpk, "max_tokens_per_request", None) is not None:
        flags = flags + [
            f"-DMPK_MAX_TOKENS_PER_REQUEST={mpk.max_tokens_per_request}"]
    flags = flags + [f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}"]
    flags = flags + [f"-DMPK_PAGE_SIZE={mpk.page_size}"]
    flags = flags + [f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}"]

    spec_cfg = getattr(mpk, 'spec_decode_config', None)
    if spec_cfg is not None and getattr(spec_cfg, 'method', None) == 'eagle3':
        flags = flags + ["-DMPK_SPEC_DECODE"]

    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
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
        # Tell the device-side profiler how many uint64 slots the caller's
        # buffer actually has, so PROFILER_EVENT_* can stop at the end instead
        # of walking past it. Profiling is already a compile-time decision, so
        # this needs no runtime plumbing. Without the flag the macros keep
        # their historical unbounded behaviour.
        if getattr(mpk, "profiler_tensor", None) is not None:
            flags = flags + [
                f"-DMPK_PROFILER_BUFFER_ENTRIES={int(mpk.profiler_tensor.numel())}"
            ]

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
        pinned_ring_capacity: int = 0,
        test_mode: bool = False,
        max_tokens_per_request: int = None,
    ):
        self.__finalized__ = False
        self._is_compiled = False
        self.test_mode = test_mode

        if mode not in valid_persistent_kernel_modes:
            raise ValueError(f"Invalid persistent kernel mode: {mode}")
        self.mode = mode
        self.pinned_ring_capacity = pinned_ring_capacity
        self.world_size = world_size
        self.mpi_rank = mpi_rank
        self.num_workers = num_workers
        self.num_local_schedulers = num_local_schedulers
        self.num_remote_schedulers = num_remote_schedulers
        self.max_seq_length = max_seq_length
        self.max_num_batched_requests = max_num_batched_requests
        self.max_num_batched_tokens = max_num_batched_tokens
        # M3-I9: cap on what ONE request may take from ONE iteration's token
        # budget (MODE_OFFLINE prefill only). None emits no define, so the
        # header's default MPK_MAX_NUM_BATCHED_TOKENS applies and the clamp is
        # the identity -- the generated graph is unchanged. See
        # include/mirage/persistent_kernel/admission_policy.h.
        if max_tokens_per_request is not None:
            if not (1 <= max_tokens_per_request <= max_num_batched_tokens):
                raise ValueError(
                    f"max_tokens_per_request={max_tokens_per_request} must be in "
                    f"[1, max_num_batched_tokens={max_num_batched_tokens}]")
            if mode != "offline":
                raise ValueError(
                    "max_tokens_per_request is a MODE_OFFLINE admission knob; "
                    f"mode={mode!r} does not use that scheduler")
        self.max_tokens_per_request = max_tokens_per_request
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.eos_token_id = eos_token_id
        self.kn_graph = KNGraph(CyKNGraph(disable_fingerprint=True))
        # Prevent GC of PyTorch tensors whose GPU pointers are baked into the
        # generated persistent-kernel code (attach_input stores raw pointers).
        self._torch_tensor_refs = []
        self.meta_tensors = meta_tensors
        # Auto-allocate scheduler snapshot buffer for in-place compaction
        if "paged_kv_indices_snapshot" not in self.meta_tensors and self.mode != "online_pinned":
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

    def dflash_attention_layer(
        self,
        q: DTensor,        # [B, q_size]            (q_norm + RoPE applied)
        ctx_k: DTensor,    # [ctx_len, kv_size]     (k_norm + RoPE; context cache)
        ctx_v: DTensor,    # [ctx_len, kv_size]     (raw v; context cache)
        blk_k: DTensor,    # [B, kv_size]           (k_norm + RoPE; this block)
        blk_v: DTensor,    # [B, kv_size]           (raw v; this block)
        output: DTensor,   # [B, q_size]
        grid_dim: tuple,   # (num_requests, 1, 1)
        block_dim: tuple,
        sliding_window: int = 0,
        head_dim: int = 128,
    ):
        # DFlash non-causal block attention (split ctx/block KV; one task/request).
        for t in (q, ctx_k, ctx_v, blk_k, blk_v, output):
            assert t.num_dims == 2
        params = [sliding_window, head_dim]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q, (-1, -1, -1), -1, True)
        tb_graph.new_input(ctx_k, (-1, -1, -1), -1, True)
        tb_graph.new_input(ctx_v, (-1, -1, -1), -1, True)
        tb_graph.new_input(blk_k, (-1, -1, -1), -1, True)
        tb_graph.new_input(blk_v, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([q, ctx_k, ctx_v, blk_k, blk_v, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "dflash_attention", params)

    def dflash_norm_rope_layer(
        self,
        x: DTensor,        # [N, num_heads*head_dim]
        weight: DTensor,   # [head_dim]  (q_norm or k_norm)
        cos: DTensor,      # [N, head_dim]
        sin: DTensor,      # [N, head_dim]
        output: DTensor,   # [N, num_heads*head_dim]
        grid_dim: tuple,
        block_dim: tuple,
        head_dim: int = 128,
    ):
        # DFlash per-head RMSNorm (eps 1e-5) + NeoX RoPE.
        assert x.num_dims == 2 and output.num_dims == 2
        params = [head_dim]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([x, weight, cos, sin, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "dflash_norm_rope", params)

    def dflash_kv_store_layer(
        self,
        kv_in: DTensor,         # [num_tokens, num_kv_heads*head_dim] bf16
        slot_mapping: DTensor,  # [num_tokens] int32 (absolute slot per token)
        cache: DTensor,         # [num_pages, page_size, num_kv_heads, head_dim]
        grid_dim: tuple,
        block_dim: tuple,
        head_dim: int = 128,
    ):
        # DFlash standalone paged KV-cache store (L4 materialize write/overwrite).
        assert kv_in.num_dims == 2
        assert cache.num_dims == 4
        params = [head_dim]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(kv_in, (-1, -1, -1), -1, True)
        tb_graph.new_input(slot_mapping, (-1, -1, -1), -1, True)
        tb_graph.new_input(cache, (-1, -1, -1), -1, True)
        self.kn_graph.customized([kv_in, slot_mapping, cache], tb_graph)
        self.kn_graph.register_task(tb_graph, "dflash_kv_store", params)

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
        enable_qk_norm: bool = True,
        q_len_override: int = 0,
        tail_offset: int = 0,
        attn_output_gate: bool = False,
        max_tokens_per_pass: int = 0,
    ):
        # attn_output_gate=True selects the fused QKVG input layout: each Q head
        # occupies 2*head_dim in the packed row as [q | gate], and the task
        # applies out *= sigmoid(gate) in its epilogue (Qwen3.5 full attention;
        # docs/qwen35/vllm-graph.md §2.2.2/§2.2.4, v1-architecture.md §4.2).
        #
        # max_tokens_per_pass>0 sizes the task's smem arena by that value
        # instead of by the activation's leading dim (= mbt) and loops
        # ceil(Q_LEN / max_tokens_per_pass) passes over the request's queries
        # (v1-architecture.md §4.3), so a large mbt no longer forces a smem
        # instantiation that does not fit.
        #
        # Both default to off; when both are off the generated code string is
        # byte-identical to what this layer produced before they existed.
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
        assert q_norm is not None and k_norm is not None, (
            "q_norm/k_norm must be valid DTensors; pass a dummy + "
            "enable_qk_norm=False when norm is disabled")
        assert q_norm.num_dims == 1  # (head_dim)
        assert k_norm.num_dims == 1  # (head_dim)
        assert q_norm.dim(0) == head_dim
        assert k_norm.dim(0) == head_dim
        qk_norm = 1 if enable_qk_norm else 0

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        # params[6]: q_len_override (only included if non-zero; for Eagle3 K>1 chain)
        # params[7]: tail_offset    (only included if non-zero; for Eagle3 K>1 chain)
        # params[8]: attn_output_gate    (only included if either 8/9 is set)
        # params[9]: max_tokens_per_pass (only included if either 8/9 is set)
        gate_flag = 1 if attn_output_gate else 0
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed,
                  self.max_seq_length, self.page_size]
        if gate_flag != 0 or max_tokens_per_pass != 0:
            # params are positional: 8/9 require 6/7 to be present, so emit the
            # (possibly zero) Eagle3 pair first.
            params.extend([q_len_override, tail_offset,
                           gate_flag, max_tokens_per_pass])
        elif q_len_override != 0 or tail_offset != 0:
            params.extend([q_len_override, tail_offset])

        if gate_flag:
            # QKVG row: per kv group, num_q_heads/num_kv_heads heads of
            # [q|gate] plus one k and one v head.
            expected_in = (2 * (num_q_heads // num_kv_heads) + 2) * head_dim * num_kv_heads
            assert input.dim(1) == expected_in, (
                f"attn_output_gate expects a fused QKVG row of {expected_in} "
                f"elements, got {input.dim(1)}")
        if max_tokens_per_pass != 0:
            assert self.target_cc == 100, (
                "max_tokens_per_pass is implemented for the sm100 task only")
            assert max_tokens_per_pass <= input.dim(0), (
                "max_tokens_per_pass must not exceed the activation's leading dim")

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
    def gdn_conv1d_layer(
        self,
        input: DTensor,        # mixed_qkv [max_num_batched_tokens, conv_dim] bf16
        weight: DTensor,       # conv1d weight [conv_dim, kernel_size] bf16
        conv_state: DTensor,   # [num_slots, kernel_size-1, conv_dim] bf16, in/out
        output: DTensor,       # [max_num_batched_tokens, conv_dim] bf16
        grid_dim: tuple,       # (max_num_batched_requests, channel blocks, 1)
        block_dim: tuple,      # (256, 1, 1) on SM100
    ):
        """Gated-DeltaNet causal depthwise conv1d with a persistent state pool.

        One task per (request SLOT, channel block). The kernel reads its own
        token window from ``qo_indptr_buffer`` (chunk length varies per
        iteration) and its own conv-state slice from
        ``task_metadata.request_id``/``kv_idx``, so nothing is partitioned by
        the grid — every tensor is presented whole, exactly like
        ``mla_prefill_layer``.

        ``grid_dim[1]`` is the channel-block count and must divide ``conv_dim``.
        It is the prefill scaling knob: the FIR has no dependency between output
        tokens, so a long chunk parallelises across channel blocks. With
        ``grid_dim[1] == 1`` a 256-token chunk runs on a single SM (measured
        1.84 ms per layer at conv_dim 8192); vLLM's Triton kernel splits the
        same way, 32 blocks of 256 channels.

        State lifecycle is kernel-side: a slot whose request is at ``step == 0``
        (its first prefill chunk) treats the stored state as zero instead of
        loading it, and the updated state is written back unconditionally. Slot
        reuse by a later request therefore re-zeros implicitly — no
        ``prepare_next_batch`` change is needed (v1-architecture.md 3.3).
        """
        assert input.num_dims == 2
        assert output.num_dims == 2
        assert weight.num_dims == 2
        assert conv_state.num_dims == 3
        conv_dim = weight.dim(0)
        kernel_size = weight.dim(1)
        assert kernel_size >= 2
        assert input.dim(1) == conv_dim
        assert output.dim(1) == conv_dim
        assert conv_state.dim(1) == kernel_size - 1
        assert conv_state.dim(2) == conv_dim
        assert conv_state.dim(0) >= grid_dim[0], (
            "conv-state pool needs one slot per request "
            f"({conv_state.dim(0)} slots < grid_dim.x {grid_dim[0]})"
        )
        num_channel_blocks = grid_dim[1]
        assert num_channel_blocks >= 1 and conv_dim % num_channel_blocks == 0, (
            f"grid_dim.y ({num_channel_blocks}) must divide conv_dim "
            f"({conv_dim})"
        )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight, (-1, -1, -1), -1, True)
        tb_graph.new_input(conv_state, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, conv_state, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "gdn_conv1d_sm100", [])

    def gdn_recurrent_layer(
        self,
        qkv: DTensor,        # conv output [max_num_batched_tokens, qkv_stride] bf16
        ba: DTensor,         # [max_num_batched_tokens, 2*num_v_heads] bf16
        alog_dtbias: DTensor,  # [2, num_v_heads] fp32 (row 0 A_log, row 1 dt_bias)
        state: DTensor,      # [slots, num_v_heads, head_v_dim, head_k_dim] fp32
        z: DTensor,          # [max_num_batched_tokens, num_v_heads*head_v_dim] bf16
        norm_w: DTensor,     # [head_v_dim] fp32
        output: DTensor,     # [max_num_batched_tokens, num_v_heads*head_v_dim] bf16
        split_scratch: DTensor,  # [slots, num_v_heads, head_v_dim + 8] fp32
        num_k_heads: int,
        grid_dim: tuple,     # (num_v_heads, max_num_batched_requests, split)
        block_dim: tuple,    # (256, 1, 1) on SM100
        depth: int = 2,      # decode cp.async ring depth (2..4)
        decode_fastpath: bool = True,
    ):
        """Gated-DeltaNet recurrence with a fused gated RMSNorm/SiLU epilogue.

        One task per (v-head, request SLOT). The task owns the WHOLE per-layer
        chain between the conv output and the ``out_proj`` input: q/k L2 norm,
        the gating scalars, the delta-rule update of the fp32 state, the readout
        ``o = S q``, and the per-head gated norm - so no separate RMSNormGated
        task is needed (v1-architecture.md 3.2).

        The kernel reads its token window from ``qo_indptr_buffer`` and its
        state slice from ``task_metadata.request_id``/``kv_idx``, so nothing is
        partitioned by the grid - every tensor is presented whole, exactly like
        ``gdn_conv1d_layer`` and ``mla_prefill_layer``.

        Unlike the conv FIR, the recurrence is SEQUENTIAL in the token index, so
        a chunk's tokens cannot be spread across tasks. The only parallel axes
        are the v-head (grid.x) and the request slot (grid.y); a long prefill
        chunk is therefore walked in-task.

        ``grid.z`` is the DECODE v-row split. Within one token the v-rows of the
        recurrent state are mutually independent, so a decode step for one
        (head, slot) can be fanned out over ``grid.z`` cooperating tasks, each
        owning ``head_v_dim / grid.z`` rows; the last of them to arrive runs the
        shared gated-norm epilogue over the fp32 ``o`` partials the others left
        in ``split_scratch``. That matters at small batch: at bs1 the whole op
        is only ``num_v_heads`` == 32 tasks, against 128 workers.

        ``split_scratch`` is ``[slots, num_v_heads, head_v_dim + 8]`` fp32,
        ZERO-INITIALISED and shared by every GDN layer (layer L+1 is
        transitively downstream of layer L, so no two uses are ever in flight).
        Column ``head_v_dim`` is the arrival counter, read as ``unsigned int``
        and self-resetting; the remaining pad keeps each row 16 B aligned. With
        ``grid.z == 1`` the buffer is never touched.

        Prefill chunks and a slot's first chunk keep the unsplit path (their
        epilogue is inside the token loop): split 0 runs the whole chunk and the
        other splits return immediately.

        ``decode_fastpath=False`` routes decode back through the original
        unsplit implementation. It exists so an A/B can compare both arms from
        ONE build in ONE window instead of swapping git trees; with it off and
        ``grid.z == 1`` the emitted task body is the pre-split one.

        ``num_k_heads`` declares the GVA ratio: two v-heads share one q/k head
        on Qwen3.5 (32 v-heads, 16 k-heads). It cannot be inferred, because q
        and k live inside the packed ``qkv`` row.

        State lifecycle is kernel-side: a slot whose request is at ``step == 0``
        treats the stored state as zero instead of loading it, and the updated
        state is written back unconditionally, so slot reuse re-zeros implicitly
        (v1-architecture.md 3.3).
        """
        assert qkv.num_dims == 2
        assert ba.num_dims == 2
        assert alog_dtbias.num_dims == 2
        assert state.num_dims == 4
        assert z.num_dims == 2
        assert norm_w.num_dims == 1
        assert output.num_dims == 2
        num_v_heads = state.dim(1)
        head_v_dim = state.dim(2)
        head_k_dim = state.dim(3)
        assert num_k_heads >= 1 and num_v_heads % num_k_heads == 0, (
            f"num_v_heads ({num_v_heads}) must be a multiple of num_k_heads "
            f"({num_k_heads})"
        )
        assert head_k_dim % 32 == 0, "head_k_dim must be a multiple of 32"
        assert alog_dtbias.dim(0) == 2 and alog_dtbias.dim(1) == num_v_heads
        assert norm_w.dim(0) == head_v_dim
        assert ba.dim(1) >= 2 * num_v_heads, "ba packs [b | a]"
        assert qkv.dim(1) >= 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim, (
            "qkv row is too narrow for [q | k | v]"
        )
        assert z.dim(1) >= num_v_heads * head_v_dim
        assert output.dim(1) >= num_v_heads * head_v_dim
        assert grid_dim[0] == num_v_heads, (
            f"grid_dim.x ({grid_dim[0]}) must be one task per v-head "
            f"({num_v_heads})"
        )
        assert state.dim(0) >= grid_dim[1], (
            "recurrent-state pool needs one slot per request "
            f"({state.dim(0)} slots < grid_dim.y {grid_dim[1]})"
        )
        split = grid_dim[2]
        assert split >= 1 and head_v_dim % split == 0, (
            f"grid_dim.z ({split}), the decode v-row split, must divide "
            f"head_v_dim ({head_v_dim})"
        )
        assert 2 <= depth <= 4, f"decode cp.async ring depth must be 2..4, got {depth}"
        assert decode_fastpath or split == 1, (
            "decode_fastpath=False needs grid_dim.z == 1: with the fast path "
            f"off the extra splits would have nothing to do (got {split})"
        )
        assert split_scratch.num_dims == 3
        assert split_scratch.dim(0) >= grid_dim[1], (
            "split scratch needs one row per request slot "
            f"({split_scratch.dim(0)} < grid_dim.y {grid_dim[1]})"
        )
        assert split_scratch.dim(1) == num_v_heads
        assert split_scratch.dim(2) == head_v_dim + 8, (
            "split scratch row is [head_v_dim o partials | counter | padding], "
            f"expected {head_v_dim + 8}, got {split_scratch.dim(2)}"
        )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(qkv, (-1, -1, -1), -1, True)
        tb_graph.new_input(ba, (-1, -1, -1), -1, True)
        tb_graph.new_input(alog_dtbias, (-1, -1, -1), -1, True)
        tb_graph.new_input(state, (-1, -1, -1), -1, True)
        tb_graph.new_input(z, (-1, -1, -1), -1, True)
        tb_graph.new_input(norm_w, (-1, -1, -1), -1, True)
        tb_graph.new_input(split_scratch, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [qkv, ba, alog_dtbias, state, z, norm_w, split_scratch, output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gdn_recurrent_sm100",
            [num_k_heads, depth, 1 if decode_fastpath else 0]
        )

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
        grid_dim = (num_groups * num_splits * x_mul,
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
        rd_dv = 2

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
        round_weights_to_input_dtype: bool = False,
        gate_padding_rows: bool = False,
    ):
        """MoE router: fp32 softmax over ALL experts -> top-k (lower expert
        index wins ties) -> renormalize. Probe P5 verified each clause against
        HF's empirical behaviour
        (demo/qwen3_5/accept/probes/moe/p5_router_semantics.json).

        `round_weights_to_input_dtype` reproduces HF's
        `router_top_value.to(router_logits.dtype)` -- the Qwen3.5 router hands
        the combine BF16 weights. DeepSeek-V3's reference keeps fp32, so this
        defaults off.

        `gate_padding_rows` (M3-I8) restricts EXPERT ACTIVATION to the rows
        that carry a live token this iteration. The batch dimension is the
        compile-time `max_num_batched_tokens`, but a decode step usually fills
        only `qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` of those rows;
        the rest hold the previous iteration's residue and still route,
        activating expert groups no live token uses (M3-I1 measured 56.4
        activated groups per layer at bs1, where top-8 on one token needs 8).
        With the flag on the kernel reads that runtime scalar and marks
        `moe_routing_indices` / `moe_masks` for live rows only; the row read,
        the input-buffer zeroing and the top-k weight write are unchanged, so
        only the grouped-GEMM consumers see a difference. Defaults OFF, and
        when off the generated code is byte-identical to the pre-M3-I8 build.

        NOTE: one PASS of the task covers `WARP_SIZE * VPT / num_experts * 8`
        token rows (8 at num_experts=256 and VPT=8, 16 at VPT=16). The
        registration picks the widest legal VPT, and since M3-I5b the kernel
        loops over row tiles, so `batch_size` is unbounded: a task makes
        `ceil(batch_size / rows_per_pass)` passes. Before M3-I5b rows past the
        first pass were silently left unrouted (M2-I9).
        NOTE: the kernel ZEROES `input` as it reads it, which is what lets a
        split-k gate linear accumulate into the same buffer next step.
        """
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

        # Keep the params list EMPTY (or [1]) whenever the M3-I8 gate is off:
        # register_task_variant dedups on the emitted code string, so an
        # unchanged params tail is what keeps every existing caller's kernel
        # byte-identical.
        if gate_padding_rows:
            params = [1 if round_weights_to_input_dtype else 0, 1]
        else:
            params = [1] if round_weights_to_input_dtype else []
        self.kn_graph.register_task(tb_graph, "moe_topk_softmax_sm100", params)

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

    def moe_fp8_blockscale_layer(
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
        w13_linear: bool,
    ):
        """Grouped FP8 MoE GEMM on the checkpoint's PRESERVED float32 block
        scales -- the fail-closed alternative to moe_w13/w2_fp8_layer.

        Those two hand the kernel a per-ROW scale (the builder's
        `repeat_interleave(128)`), and the kernel then truncates every scale to
        a power of two because its block-scaled UMMA hardware-types the
        operand. Probe P2 measured that as a ~2x per-expert multiplicative
        shrink on this checkpoint (probes/fp8/p2_verdict.json), so this variant
        consumes `weight_scale_inv` in its SHIPPED shape --
        (num_experts, N//128, K//128) float32 -- and applies it in fp32 at the
        128-K-tile boundary instead.

        input_fp8/input_scale must come from
        quantize_fp8_layer(..., scale_ue8m0=False).
        """
        # input_fp8:    w13 (batch, hidden) / w2 (batch, topk, intermediate)
        # input_scale:  the same, last dim // 128, float32
        # weight_fp8:   (num_experts, N, K)                        FP8 E4M3
        # weight_scale: (num_experts, N//128, K//128)              float32
        # routing:      (num_experts, batch) int32, expert-major
        # mask:         (num_experts + 1,)   int32
        # output:       (batch, topk, N)     BF16
        assert input_fp8.num_dims == (2 if w13_linear else 3)
        assert input_scale.num_dims == input_fp8.num_dims
        assert weight_fp8.num_dims == 3
        assert weight_scale.num_dims == 3
        assert weight_scale.dim(1) * 128 == weight_fp8.dim(1)
        assert weight_scale.dim(2) * 128 == weight_fp8.dim(2)
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Same partitioning as the UE8M0 grouped layers: grid.y splits the
        # weight's N. weight_scale carries one row per 128 weight rows, so its
        # dim1 splits by the same factor.
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
        task_name = (
            "moe_w13_fp8_blockscale_sm100"
            if w13_linear
            else "moe_w2_fp8_blockscale_sm100"
        )
        self.kn_graph.register_task(tb_graph, task_name)

    # === FP8 Dense Layers ===
    def quantize_fp8_layer(
        self,
        input: DTensor,
        output_fp8: DTensor,
        output_scale: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        scale_ue8m0: bool = True,
        row_partition: tuple = (-1, -1, -1),
    ):
        """Quantize BF16 input to FP8 with block-wise scale.

        scale_ue8m0=True: output scale is packed UE8M0 uint32 (for FP8 linear GEMM)
        scale_ue8m0=False: output scale is float32 (for MoE group GEMM)

        `row_partition` splits the ROW (token) axes across the grid.

        The kernel (`per_token_group_quantize_fp8.cuh`) loops over all
        `BATCH_SIZE` rows of the tile it is handed, because under the persistent
        runtime `blockIdx.x` is the physical worker id and cannot be used as a
        row index. With the default `(-1,-1,-1)` every task is handed the WHOLE
        tensor, so a `grid_dim=(mbt,1,1)` launch runs the same full-tensor
        quantize `mbt` times and only the graph-width benefit of the extra tasks
        is real -- the work is `mbt`-fold redundant (M3-I1 measured 84 ms of
        worker time per decode step for 5.3 ms of useful work).

        Passing e.g. `(0,-1,-1)` makes grid.x split tensor dim 0 instead, so
        `BATCH_SIZE` becomes `dim0/grid.x` and each task quantizes only its own
        rows. This is BIT-EXACT: a group's fp8 bytes and its fp32 scale depend
        only on that group's own 128 elements, and the row loop carries no state
        across rows, so redistributing rows over CTAs cannot move a bit.

        Only the row axes may be split -- the group (last) axis must stay whole,
        because the fp32 scale row stride is `HIDDEN_SIZE/GROUP_SIZE` of the
        TILE, not of the full tensor. UE8M0 scales are stored column-major
        `[packed_k, aligned_batch]`, so their dim 0 is NOT the row axis and
        partitioning is refused for that path.
        """
        params = []
        if row_partition != (-1, -1, -1):
            assert not scale_ue8m0, (
                "row_partition is only valid for scale_ue8m0=False: the UE8M0 "
                "scale is column-major [packed_k, aligned_batch], so its dim 0 "
                "is the group axis, not the row axis")
            nd = input.num_dims
            assert output_fp8.num_dims == nd and all(
                output_fp8.dim(d) == input.dim(d) for d in range(nd)), (
                "row_partition: output_fp8 must have the input's exact shape "
                "(the kernel indexes both with one linear index)")
            assert output_scale.num_dims == nd, (
                f"row_partition needs the scale ({output_scale.num_dims}-D) to "
                f"carry the same leading row axes as the input ({nd}-D)")
            for d in range(nd - 1):
                assert output_scale.dim(d) == input.dim(d), (
                    f"row_partition: scale dim {d} ({output_scale.dim(d)}) must "
                    f"match input dim {d} ({input.dim(d)})")
            assert max(row_partition) <= nd - 2, (
                f"row_partition {row_partition} would split the group axis of a "
                f"{nd}-D tensor; only axes 0..{nd - 2} are row axes")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, row_partition, -1, True)
        tb_graph.new_input(output_fp8, row_partition, -1, True)
        tb_graph.new_input(output_scale, row_partition, -1, True)
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

    def linear_fp8_blockscale_layer(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        weight_fp8: DTensor,
        weight_scale: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        residual: DTensor = None,
    ):
        """Dense FP8 GEMM on the checkpoint's PRESERVED float32 block scales.

        Unlike linear_fp8_layer, which needs weights re-quantized under
        power-of-two UE8M0 scales, this consumes `weight_scale_inv` as shipped:
        float32, one value per 128x128 weight block, [N/128, K/128]. The
        activation must be quantized with the fp32-scale variant
        (quantize_fp8_layer(..., scale_ue8m0=False)), which produces a
        [batch, K/128] float32 scale.
        """
        params = [1] if residual is not None else []
        # FAIL CLOSED on the scale split. grid.x splits weight_scale's dim0 by
        # INTEGER DIVISION (runtime.cc: block_size = dim[input_map.x] /
        # grid_dim.x, offset = block_size * bid.x * stride). A grid finer than
        # dim0 therefore gives block_size == 0 and EVERY task silently reads
        # scale row 0 -- wrong numbers, no error. Since M4-I2 the kernel accepts
        # a per-task N slice finer than the checkpoint's 128-row scale block, and
        # the caller supplies the extra rows by ROW-REPLICATING weight_scale to
        # one row per task (bit-identical data; see
        # Qwen35Builder._fp8_block_scale). Require that here so a mis-wired
        # caller cannot compute with the wrong scales.
        n_tasks = grid_dim[0]
        assert weight_scale.dim(0) == n_tasks, (
            f"weight_scale dim0 must be exactly one row per task: got "
            f"{weight_scale.dim(0)} for grid.x={n_tasks}. grid.x splits this "
            f"tensor by integer division, so a coarser dim0 makes every task "
            f"read scale row 0. Row-replicate the checkpoint's [N/128, K/128] "
            f"weight_scale to [n_tasks, K/128] before attaching it.")
        assert weight_fp8.dim(0) % n_tasks == 0, (
            f"weight rows {weight_fp8.dim(0)} must divide evenly over "
            f"grid.x={n_tasks}")
        n_slice = weight_fp8.dim(0) // n_tasks
        assert n_slice % 128 == 0 or (128 % n_slice == 0 and n_slice >= 16), (
            f"per-task N slice {n_slice} must be a multiple of the 128-row "
            f"scale block or a sub-multiple >= 16 of one "
            f"(linear_fp8_blockscale_sm100.cuh: fast_path_ok)")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Same partitioning as linear_fp8_layer: grid.x splits the weight's
        # output rows, so weight_scale (whose dim0 is one row per TASK) splits on
        # dim0 too and the output splits on dim1.
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(weight_fp8, (0, -1, -1), -1, True)
        tb_graph.new_input(weight_scale, (0, -1, -1), -1, True)
        inputs = [input_fp8, input_scale, weight_fp8, weight_scale]
        if residual is not None:
            tb_graph.new_input(residual, (1, -1, -1), -1, True)
            inputs.append(residual)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(inputs + [output], tb_graph)
        task_name = (
            "linear_fp8_blockscale_with_residual_sm100"
            if residual is not None
            else "linear_fp8_blockscale_sm100"
        )
        self.kn_graph.register_task(tb_graph, task_name, params)

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

        # Under tensor parallelism the MoE output is row-parallel and followed by
        # an allreduce. The residual must be added on exactly one rank, otherwise
        # the allreduce sums it world_size times (double-counted residual). Mirror
        # the rank-0-only guard used by linear_with_residual_layer; the SM100
        # kernel skips the residual add when its pointer is null (params[0]==0).
        params = []
        enable_residual = 1
        if self.world_size > 1 and self.mpi_rank != 0:
            enable_residual = 0
        params.append(enable_residual)
        self.kn_graph.register_task(tb_graph, "moe_mul_sum_add_sm100", params)

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

    def sigmoid_gate_mul_add_layer(
        self,
        input: DTensor,
        gate_weight: DTensor,
        shared: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Qwen3.5 shared-expert gate + residual fold:

            output = residual + sigmoid(input @ gate_weight.T) * shared

        `input` is the PRE-MLP hidden state (the same tensor the router reads),
        `gate_weight` is the `[1, hidden]` unquantized `shared_expert_gate`, and
        `shared` is the shared expert's post-`down_proj` output. The gate scalar
        is applied AFTER the down projection, per
        `Qwen3_5MoeSparseMoeBlock.forward` (vllm-graph.md 2.3.3).

        Folding `residual` in here is what lets the result be passed straight to
        `moe_mul_sum_add_layer(residual=...)`, giving the block's final
        `sum_j w_j * y_j + residual + sigmoid(...) * shared` in two tasks
        (DeepSeek-V3's builder does the ungated version of this, mpk-gaps Gap 8).

        A `linear_layer` at N=1 is degenerate, so the gate GEMV is computed
        inline; the whole hidden row must live in one task, hence only the batch
        dimension is split across the grid.
        """
        assert input.num_dims == 2  # (batch_size, hidden_size)
        assert gate_weight.num_dims == 2  # (1, hidden_size)
        assert gate_weight.dim(0) == 1
        assert gate_weight.dim(1) == input.dim(1)
        assert shared.num_dims == 2  # (batch_size, output_size)
        assert residual.num_dims == 2  # (batch_size, output_size)
        assert output.num_dims == 2  # (batch_size, output_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(gate_weight, (-1, -1, -1), -1, True)
        tb_graph.new_input(shared, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, gate_weight, shared, residual, output], tb_graph
        )
        assert self.target_cc == 100, (
            "sigmoid_gate_mul_add_sm100 is registered for Blackwell only"
        )
        self.kn_graph.register_task(tb_graph, "sigmoid_gate_mul_add_sm100")

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

    # === Eagle3 layers ===
    def copy_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Generic memcpy: dst[i,j] = src[i,j] for a 2D bf16 tensor.

        Used by Eagle3 to capture target's intermediate hidden states into
        dedicated aux buffers.
        """
        assert input.num_dims == 2
        assert output.num_dims == 2
        assert input.dim(0) == output.dim(0)
        assert input.dim(1) == output.dim(1)
        batch_size = input.dim(0)
        hidden_dim = input.dim(1)
        params = [batch_size, hidden_dim]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "copy", params)

    def concat_layer(
        self,
        inputs: list,      # list of N (batch, hidden_dim) DTensors
        output: DTensor,   # (batch, N * hidden_dim)
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Concatenate N (B,H) tensors along dim 1 into (B, N*H)."""
        n = len(inputs)
        assert n >= 1
        assert all(t.num_dims == 2 for t in inputs)
        assert output.num_dims == 2
        batch_size = inputs[0].dim(0)
        hidden_dim = inputs[0].dim(1)
        assert all(t.dim(0) == batch_size and t.dim(1) == hidden_dim
                   for t in inputs)
        assert output.dim(0) == batch_size
        assert output.dim(1) == n * hidden_dim
        params = [batch_size, hidden_dim, n]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        for t in inputs:
            tb_graph.new_input(t, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([*inputs, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "concat", params)

    def eagle3_commit_layer(
        self,
        target_argmax: DTensor,     # (batch, 1) int64 — from argmax_reduce (= output_token DTensor)
        draft_tokens_new: DTensor,  # (batch, K) int64 — this iter's drafts (scatter output)
        accepted_count: DTensor,    # (batch, 1) int32 — from verify_strict (1st output)
        tokens_buffer: DTensor,     # (max_requests, max_seq_len) int64 — written in-place
        num_new_tokens: DTensor,    # (max_requests,) int32 — OUTPUT (= accept_count)
        drafts_prev: DTensor,       # (max_requests, K) int64 — attach_input cross-iter snapshot dst
        accept_hist: DTensor,       # (K+2,) int32 — debug: atomicAdd histogram of ac values
        grid_dim: tuple,
        block_dim: tuple,
        num_draft_tokens: int,      # K
        batch_size: int,            # mbt
        max_seq_len: int,
    ):
        params = [num_draft_tokens, batch_size, max_seq_len]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(target_argmax, (-1, -1, -1), -1, True)
        tb_graph.new_input(draft_tokens_new, (-1, -1, -1), -1, True)
        tb_graph.new_input(accepted_count, (-1, -1, -1), -1, True)
        tb_graph.new_input(tokens_buffer, (-1, -1, -1), -1, True)
        tb_graph.new_input(accept_hist, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_new_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(drafts_prev, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [target_argmax, draft_tokens_new, accepted_count, tokens_buffer,
             accept_hist, num_new_tokens, drafts_prev],
            tb_graph)
        self.kn_graph.register_task(tb_graph, "eagle3_commit", params)

    def eagle3_d2t_remap_layer(
        self,
        hot_token: DTensor,      # (batch, 1) int64 — argmax over draft logits
        d2t_table: DTensor,      # (draft_vocab_real,) int64
        target_token: DTensor,   # (batch, 1) int64 — output
        grid_dim: tuple,
        block_dim: tuple,
        draft_vocab_real: int,   # = d2t_table.dim(0); argmax indices ≥ this come from
                                 # lm_head's padded rows and must be sentinel-mapped to 0
    ):
        assert hot_token.num_dims == 2
        assert d2t_table.num_dims == 1
        assert target_token.num_dims == 2
        assert hot_token.dim(0) == target_token.dim(0)
        batch_size = hot_token.dim(0)
        params = [batch_size, draft_vocab_real]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(hot_token, (-1, -1, -1), -1, True)
        tb_graph.new_input(d2t_table, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_token, (-1, -1, -1), -1, True)
        self.kn_graph.customized([hot_token, d2t_table, target_token], tb_graph)
        self.kn_graph.register_task(tb_graph, "eagle3_d2t_remap", params)

    def compile(
        self,
        **kwargs,
    ):
        assert not self._is_compiled
        
        output_dir = kwargs.get("output_dir", None)

        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        if self.mode == "online_notoken" or self.mode == "online" or self.mode == "multi_turn" or self.mode=="online_pinned":
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
        pinned_extra_order=[
            "pinned_req_ready",
            "pinned_req_request_id",
            "pinned_req_prompt_len",
            "pinned_req_initial_step",
            "pinned_comp_ready",
            "pinned_comp_request_id",
            "pinned_comp_buffer_row",
            "pinned_comp_final_step",
            "pinned_shutdown",
            "pinned_step",
            "pinned_inbox_tokens",
            "pinned_rid_at_row",
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
        if self.mode=="online_pinned":
            for key in pinned_extra_order:
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
        meta_tensors.append(self.meta_tensors["paged_kv_indices_snapshot"])
        if self.mode == "online_pinned":
            meta_tensors.append(self.meta_tensors["pinned_req_ready"])
            meta_tensors.append(self.meta_tensors["pinned_req_request_id"])
            meta_tensors.append(self.meta_tensors["pinned_req_prompt_len"])
            meta_tensors.append(self.meta_tensors["pinned_req_initial_step"])
            meta_tensors.append(self.meta_tensors["pinned_comp_ready"])
            meta_tensors.append(self.meta_tensors["pinned_comp_request_id"])
            meta_tensors.append(self.meta_tensors["pinned_comp_buffer_row"])
            meta_tensors.append(self.meta_tensors["pinned_comp_final_step"])
            meta_tensors.append(self.meta_tensors["pinned_shutdown"])
            meta_tensors.append(self.meta_tensors["pinned_step"])
            meta_tensors.append(self.meta_tensors["pinned_inbox_tokens"])
            meta_tensors.append(self.meta_tensors["pinned_rid_at_row"])
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

            # The two exporters are independent views of the same buffer; run
            # them independently so a failure in one still leaves the other's
            # artifact behind (the Perfetto exporter used to raise and take the
            # otherwise-fine CSV with it -- see
            # demo/qwen3_5/accept/probes/runtime/p9_methodology.md step 2).
            first_error = None
            for fn, path in (
                (export_to_perfetto_trace, stem + ".perfetto-trace"),
                (export_to_csv, stem + ".csv"),
            ):
                try:
                    fn(self.profiler_tensor, path)
                except Exception as e:  # noqa: BLE001 - report, don't mask
                    print(f"[mpk] profiler export to {path} failed: {e!r}")
                    if first_error is None:
                        first_error = e
            if first_error is not None:
                raise first_error

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        if self._is_compiled:
            self.finalize_func()
        self.__finalized__ = True
