# Copyright 2026 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gpt_oss_moe_norm (tasks/gpt_oss_moe_norm.cuh) bit for bit against the unfused path it replaces: FlashInfer's
finalize writing the experts' sum, then sm90_rmsnorm's add (the residual stream and the norm), both in one static
kernel. The inputs hit finalize's edges: -1 slots, rows past R, R = 0, products below fp32's normal range (where flush
to zero acts), an inf. The task's code is plain CUDA (no tcgen05), so this runs on any GPU the static megakernel builds
for.

Two finalize backends, both built as FlashInfer JIT-builds its MoE module (-O3 -use_fast_math):
  replica     a verbatim copy of finalizeKernel's loop (the kernel FlashInfer's run() picks below 99 tokens);
  flashinfer  FlashInfer's own csrc (trtllm_fused_moe_dev_kernel.cu) called through moe::dev::finalize::run with the
              Data trtllm_fused_moe_runner.cu's setOpsData fills, so its dispatch (finalizeKernel / VecLoad /
              SmallBatch) is FlashInfer's. Needs FLASHINFER_DATA (the wheel's flashinfer/data: csrc, include, cutlass,
              cccl; default: the installed flashinfer's) and FLASHINFER_BMM_INC (a dir holding
              flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export: the trtllm-gen export headers FlashInfer's JIT
              downloads); skipped without them."""
import ctypes
import os
import shutil
import subprocess

import pytest
import torch

H, HP, TOPK = 2880, 3072, 4
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEC_FROM = 99   # FlashInfer's run(): finalizeKernelVecLoad once ceil(H / 256) x tokens >= 1184 (148 SMs x 8)

# FlashInfer 0.7.2.post1 csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu finalizeKernel (Type, TypeExpW:
# cutlass::bfloat16_t; no second input; no PDL wait), the loop verbatim; run() launches it for this shape at fewer than
# 99 tokens (12 x T blocks < 1184; top-k 4 is not past SMALL_BATCH_FINALIZE_MAX_LOADS) with 256 threads
REPLICA = r"""
#include <cuda_runtime.h>
#include "cutlass/bfloat16.h"
struct Params {
  using Type = cutlass::bfloat16_t;
  using TypeExpW = cutlass::bfloat16_t;
  Type const *inPtr;
  Type const *inPtr2;
  TypeExpW const *expertWeightsPtr;
  int const *expandedIdxToPermutedIdx;
  Type *outPtr;
  int hiddenDim, hiddenDimPadded, numTokens, topK;
};
__global__ void finalizeKernel(Params params) {
  using TypeExpW = typename Params::TypeExpW;
  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      float data = 0.0F;
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        if (permutedIdx == -1) {
          continue;
        }
        if (params.expertWeightsPtr != nullptr) {
          TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
          data += float{scale} *
                  float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
        } else {
          data += float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
        }
      }
      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = static_cast<typename Params::Type>(data);
    }
  }
}
extern "C" int finalize(void const *in, int const *idx, void const *w, void *out, int numTokens, int hiddenDim,
                        int hiddenDimPadded, int topK) {
  if (numTokens == 0) return 0;
  Params p{(Params::Type const *)in, nullptr, (Params::TypeExpW const *)w, idx, (Params::Type *)out, hiddenDim,
           hiddenDimPadded, numTokens, topK};
  int const threads = 256, bx = (hiddenDim - 1 + threads) / threads, by = numTokens < 8192 ? numTokens : 8192;
  finalizeKernel<<<dim3(bx, by), threads>>>(p);
  return (int)cudaDeviceSynchronize();
}
"""


# FlashInfer's own finalize, entered as trtllm_fused_moe_runner.cu enters it (setOpsData's finalizeData for a bf16
# output, bf16 routing weights, PDL on, no DeepSeek FP8, no second input)
SHIM = r"""
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
namespace tg = batchedGemm::trtllm::gen;
extern "C" int finalize(void const *in, int const *idx, void const *w, void *out, int numTokens, int hiddenDim,
                        int hiddenDimPadded, int topK) {
  if (numTokens == 0) return 0;   // a launch with no blocks: the model never calls the MoE with no tokens
  moe::dev::finalize::Data d;
  d.mDtypeElt = tg::Dtype::Bfloat16;
  d.mDtypeExpW = tg::Dtype::Bfloat16;
  d.mUsePdl = true;
  d.mUseDeepSeekFp8 = false;
  d.inPtr = const_cast<void *>(in);
  d.outPtr = out;
  d.expertWeightsPtr = const_cast<void *>(w);
  d.expandedIdxToPermutedIdx = const_cast<int *>(idx);
  d.numTokens = numTokens;
  d.numExperts = 128;
  d.topK = topK;
  d.hiddenDim = hiddenDim;
  d.hiddenDimPadded = hiddenDimPadded;
  d.totalNumPaddedTokens = nullptr;
  try {
    moe::dev::finalize::run(d, nullptr);
  } catch (...) {
    return -1;
  }
  return (int)cudaDeviceSynchronize();
}
"""
FI_DEFS = ["-DTLLM_GEN_EXPORT_INTERFACE", "-DTLLM_GEN_EXPORT_FLASHINFER", "-DTLLM_ENABLE_CUDA", "-DENABLE_BF16",
           "-DENABLE_FP8", "-DENABLE_FP4", "-DNDEBUG"]


def _nvcc_arch():
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found")
    cc = "".join(map(str, torch.cuda.get_device_capability()))
    cc = cc + "a" if cc in ("90", "100") else cc
    return nvcc, [f"-gencode=arch=compute_{cc},code=sm_{cc}"]


def _load(lib):
    sass = subprocess.run(["cuobjdump", "-sass", str(lib)], capture_output=True, text=True).stdout
    assert "FFMA.FTZ" in sass, "finalize's accumulate is not FFMA.FTZ (-use_fast_math)"
    f = ctypes.CDLL(str(lib)).finalize
    f.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
    return f


def _replica(d):
    nvcc, arch = _nvcc_arch()
    src, lib = d / "finalize.cu", d / "libfinalize.so"
    src.write_text(REPLICA)
    inc = os.environ.get("FLASHINFER_CUTLASS_INCLUDE", os.path.join(ROOT, "deps/cutlass/include"))
    subprocess.run([nvcc, "-O3", "-use_fast_math", "-std=c++17", *arch, "-shared", "-Xcompiler", "-fPIC",
                    f"-I{inc}", "-o", str(lib), str(src)], check=True)
    return _load(lib)


def _flashinfer(d):
    data = os.environ.get("FLASHINFER_DATA")
    if data is None:
        try:
            import flashinfer
            data = os.path.join(os.path.dirname(flashinfer.__file__), "data")
        except ImportError:
            pytest.skip("FLASHINFER_DATA unset and flashinfer not installed")
    bmm = os.environ.get("FLASHINFER_BMM_INC")
    if bmm is None:
        pytest.skip("FLASHINFER_BMM_INC unset (the trtllm-gen export headers)")
    dev = os.path.join(data, "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu")
    if not os.path.exists(dev):
        pytest.skip(f"{dev} not found")
    nvcc, arch = _nvcc_arch()
    inc = [f"-I{p}" for p in (os.path.join(data, "include"), os.path.join(data, "csrc"),
                              os.path.join(data, "csrc/nv_internal"), os.path.join(data, "csrc/nv_internal/include"),
                              os.path.join(data, "cutlass/include"), os.path.join(data, "cutlass/tools/util/include"),
                              bmm, os.path.join(data, "cccl/cub"), os.path.join(data, "cccl/libcudacxx/include"),
                              os.path.join(data, "cccl/thrust"))]
    common = [os.path.join(data, "csrc/nv_internal/cpp/common", f) for f in
              ("logger.cpp", "stringUtils.cpp", "tllmException.cpp", "envUtils.cpp")]
    src, lib = d / "shim.cu", d / "libfinalize_fi.so"
    src.write_text(SHIM)
    subprocess.run([nvcc, "-O3", "-use_fast_math", "-std=c++17", "--expt-relaxed-constexpr", *arch, "-shared",
                    "-Xcompiler", "-fPIC", *FI_DEFS, *inc, "-o", str(lib), str(src), dev, *common, "-lcuda"],
                   check=True)
    return _load(lib)


@pytest.fixture(scope="module")
def finalize_libs(tmp_path_factory):
    cache = {}

    def get(backend):
        if backend not in cache:
            d = tmp_path_factory.mktemp(backend)
            cache[backend] = _replica(d) if backend == "replica" else _flashinfer(d)
        return cache[backend]

    return get


def _bf16(x):
    return x.to(torch.bfloat16)


def _case(gen, rows, R, minus_one, tiny, inf):
    """gemm2 [rows, HP], idx [R TOPK] (distinct rows; minus_one: some slots -1), weights [R, TOPK] bf16."""
    scale = 10.0 ** torch.randint(-3, 2, (rows, 1), generator=gen).float()
    g2 = _bf16(torch.randn(rows, HP, generator=gen) * scale)
    g2[:, H:] = _bf16(torch.full((rows, HP - H), 1e4))   # the padding columns must not leak in
    w = _bf16(torch.rand(max(R, 1), TOPK, generator=gen) + 0.05)[:R]
    idx = torch.randperm(rows, generator=gen)[:R * TOPK].int()
    if minus_one and R:
        idx[torch.randperm(R * TOPK, generator=gen)[:max(1, R * TOPK // 4)]] = -1
    if tiny and R:   # products and sums below fp32's normal range: flush to zero decides them
        w[0, 0] = _bf16(torch.tensor(1e-30))
        p = int(idx[0]) if int(idx[0]) != -1 else 0
        idx[0] = p
        g2[p, :64] = _bf16(torch.randn(64, generator=gen) * 1e-9)
        g2[p, 64:72] = _bf16(torch.tensor([1e-39, -1e-39, 3e-39, 5e-40, -2e-38, 1e-38, 9e-39, -7e-39]))
        if R > 1:   # a row whose only expert is that row, weight 1: subnormal inputs (columns 64..71) flush to 0
            idx[TOPK:2 * TOPK] = -1
            idx[TOPK] = p
            w[1, 0] = 1.0
    if inf and R > 2:
        q = next(int(i) for i in idx[2 * TOPK:3 * TOPK] if int(i) != -1)
        g2[q, 100] = float("inf")
    return g2.cuda(), idx.cuda(), w.cuda()


def _cases(T):
    if T <= 8:
        return [dict(R=8, minus_one=False, tiny=False, inf=False), dict(R=8, minus_one=True, tiny=True, inf=False),
                dict(R=5, minus_one=True, tiny=True, inf=True), dict(R=1, minus_one=False, tiny=True, inf=False),
                dict(R=0, minus_one=False, tiny=False, inf=False)]
    # the static megakernel's largest bucket (config.cuh: T <= 64 < VEC_FROM, so FlashInfer always runs finalizeKernel)
    return [dict(R=T, minus_one=False, tiny=False, inf=False), dict(R=T, minus_one=True, tiny=True, inf=True),
            dict(R=T - 3, minus_one=True, tiny=True, inf=True), dict(R=1, minus_one=False, tiny=True, inf=False)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("backend,T", [("replica", 8), ("flashinfer", 8), ("flashinfer", 64)])
def test_gpt_oss_moe_norm_matches_flashinfer_finalize_then_rmsnorm(tmp_path, finalize_libs, backend, T):
    from mirage.mpk import compiler, plan_gen
    from mirage.mpk.static_megakernel import StaticMegakernel, sm90_flags, sm100_flags

    finalize = finalize_libs(backend)
    rows = max(64, T * TOPK)
    gen = torch.Generator().manual_seed(0)
    dev = "cuda"
    x = _bf16(torch.randn(T, H, generator=gen)).to(dev)
    x[1, 64:72] = 0   # where row 1's experts' sum is subnormal (the tiny cases): sum = y itself, flushed or not
    ln = _bf16(torch.randn(H, generator=gen) * 0.1 + 1).to(dev)
    y_ref = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    table = torch.zeros(4, dtype=torch.int64, device=dev)
    bufs = {"x": x, "y_ref": y_ref, "table": table, "ln": ln}
    for k in ("out_a", "sum_a", "out_b", "sum_b"):
        bufs[k] = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    eps = 1e-5
    mpk.sm90_rmsnorm_layer(A["x"], A["ln"], A["out_a"], eps, add=A["y_ref"], sum_out=A["sum_a"])
    mpk.gpt_oss_moe_norm_layer(A["x"], A["table"], A["ln"], A["out_b"], A["sum_b"], eps, top_k=TOPK, gemm2_width=HP)
    paths, _ = compiler.compile_plan(mpk, plan_gen.round_robin_plan(mpk), str(tmp_path / "schedules"))
    fl = sm100_flags() if torch.cuda.get_device_capability()[0] == 10 else sm90_flags()
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=str(tmp_path / "build"), extra_flags=fl)
    from mirage.mpk.static_harness import Harness
    h = Harness(sk)
    ftz_seen = 0
    try:
        for c in _cases(T):
            g2, idx, w = _case(gen, rows, **c)
            R = c["R"]
            y_ref.zero_()
            assert finalize(g2.data_ptr(), idx.data_ptr(), w.data_ptr(), y_ref.data_ptr(), R, H, HP, TOPK) == 0
            table.copy_(torch.tensor([g2.data_ptr(), idx.data_ptr(), w.data_ptr(), R], dtype=torch.int64))
            for k in ("out_a", "sum_a", "out_b", "sum_b"):
                bufs[k].fill_(float("nan"))
            torch.cuda.synchronize()
            h.launch(timeout_s=30)
            torch.cuda.synchronize()
            for a, b in (("sum_a", "sum_b"), ("out_a", "out_b")):
                d = (bufs[a].view(torch.int16) != bufs[b].view(torch.int16)).sum().item()
                assert d == 0, f"{c}: {b} differs from the unfused path's {a} in {d} elements"
            # the test's power: finalize's flush to zero is visible against an exact (no flush) emulation
            acc = torch.zeros(R, H, dtype=torch.float64, device=dev)
            for k in range(TOPK):
                p = idx.view(R, TOPK)[:, k].long() if R else idx.new_zeros(0).long()
                acc += torch.where((p != -1)[:, None], w[:, k:k + 1].double() * g2[p.clamp(min=0), :H].double(),
                                   torch.zeros((), dtype=torch.float64, device=dev))
            exact = acc.float().to(torch.bfloat16)
            ftz_seen += ((exact != y_ref[:R]) & (exact != 0) & (y_ref[:R] == 0)).sum().item()
            assert (y_ref[R:] == 0).all()
    finally:
        sk.finalize()
    assert ftz_seen > 0, "no element where flush to zero decided finalize's output: the tiny cases lost their power"
