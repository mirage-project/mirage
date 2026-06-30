"""
Tests for MPK constrained decoding (xgrammar).

Two independent checks, both runnable on a Blackwell (sm_100) box with xgrammar
installed — neither needs the full megakernel build:

1. test_structured_manager_host
   Drives StructuredGenerationManager with mock pinned buffers (plain CPU
   tensors) and a synthetic xgrammar grammar, validating the per-step bitmask
   fill, mask_seq publishing, accept_token sequencing, and the global flag.
   This exercises the host step-sync contract in structured.py.

2. test_apply_token_bitmask_kernel_cuda
   Compiles apply_token_bitmask_sm100.cuh standalone and runs it on the GPU,
   checking the masked output against a CPU reference for both the flag-off
   (plain copy) and flag-on (masked) paths, including the idle-row (-1) case.

Run directly:   python test_apply_token_bitmask.py
Run via pytest: pytest test_apply_token_bitmask.py
"""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
STRUCTURED_PY = REPO_ROOT / "python" / "mirage" / "mpk" / "structured.py"
KERNEL_DIR = REPO_ROOT / "include" / "mirage" / "persistent_kernel" / "tasks" / "blackwell"


def _load_structured():
    """Import structured.py directly (no heavy mirage package import)."""
    spec = importlib.util.spec_from_file_location("mpk_structured", STRUCTURED_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StructuredGenerationManager


def _toy_tokenizer_info(vocab=("a", "b", "c", "d", "e", "f", "g", "h")):
    import xgrammar as xgr

    ti = xgr.TokenizerInfo(list(vocab), vocab_type=xgr.VocabType.RAW,
                           vocab_size=len(vocab))
    return ti, vocab


# ── 1. Host logic ─────────────────────────────────────────────────────────

def test_structured_manager_host():
    xgr = pytest.importorskip("xgrammar")
    StructuredGenerationManager = _load_structured()

    ti, vocab = _toy_tokenizer_info()
    V = ti.vocab_size
    words = (V + 31) // 32
    n_req, max_seq = 4, 16

    # Mock pinned buffers (CPU tensors stand in for page-locked memory).
    token_bitmask = torch.full((n_req, words), -1, dtype=torch.int32)
    mask_seq = torch.zeros(n_req, dtype=torch.int32)
    flag = torch.zeros(1, dtype=torch.int32)
    tokens = torch.zeros(n_req, max_seq, dtype=torch.int64)
    pinned_step = torch.zeros(n_req, dtype=torch.int32)
    prompt_lengths = torch.zeros(n_req, dtype=torch.int32)

    row_map = {}  # rid -> buffer row (simulates the GPU's assignment)

    def find_row_for_rid(rid):
        return row_map.get(rid, -1)

    mgr = StructuredGenerationManager(
        token_bitmask=token_bitmask, mask_seq=mask_seq, constrained_flag=flag,
        tokens=tokens, pinned_step=pinned_step,
        find_row_for_rid=find_row_for_rid, prompt_lengths=prompt_lengths,
    )
    mgr.init_xgrammar(tokenizer_info=ti, vocab_size=V)

    EBNF = 'root ::= ("a" | "b")+'
    mgr.set_request_grammar(0, ebnf=EBNF)
    assert flag[0].item() == 1, "set_request_grammar must turn on the flag"

    # GPU assigns request 0 to row 2; prompt occupies tokens[0..2] (len 3), so
    # generation begins at position 3.
    row = 2
    row_map[0] = row
    prompt_lengths[row] = 3

    # PREFILL tick (pinned_step 2 < prompt_len): the model is consuming the
    # prompt. The manager publishes the *fresh* grammar mask and must NOT accept
    # any token (prefill argmax outputs are not real generations).
    pinned_step[row] = 2
    mgr.tick()
    assert mask_seq[row].item() == 2
    allowed = [i for i in range(V)
               if (token_bitmask[row, i // 32].item() >> (i % 32)) & 1]
    assert allowed == [0, 1], f"prefill fresh mask should allow a/b, got {allowed}"

    # DECODE tick: the first generated token 'a' (id 0) lands at tokens[row, 3];
    # the manager accepts it and masks the next position.
    tokens[row, 3] = 0
    pinned_step[row] = 3
    mgr.tick()
    assert mask_seq[row].item() == 3

    # Cross-check the published mask against a fresh matcher that accepted the
    # same token — the manager's accept_token sequencing must match.
    cg = mgr._compiler.compile_grammar(EBNF)
    ref = xgr.GrammarMatcher(cg)
    assert ref.accept_token(0)
    ref_bm = xgr.allocate_token_bitmask(1, V)
    ref.fill_next_token_bitmask(ref_bm, index=0)
    assert token_bitmask[row, 0].item() == ref_bm[0, 0].item(), \
        "manager mask diverged from reference after accept_token"

    # Idempotent within a step: a second tick at the same step does nothing.
    snap = token_bitmask[row, 0].item()
    mgr.tick()
    assert token_bitmask[row, 0].item() == snap

    # Releasing the only grammar turns the flag back off.
    mgr.release(0)
    assert flag[0].item() == 0, "flag must clear when no grammars remain"


# ── 2. CUDA kernel correctness ────────────────────────────────────────────

_CU_SRC = r"""
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>
typedef __nv_bfloat16 bfloat16;
#include "apply_token_bitmask_sm100.cuh"

__global__ void probe(void const* in, void* out, int32_t* bm, int32_t* seq,
                      int32_t* flag, int* rid, int* step,
                      int v, int w, int n) {
  kernel::apply_token_bitmask_sm100_kernel<bfloat16, 8>(
      in, out, bm, seq, flag, rid, step, v, w, n);
}

#define CK(x) do{ cudaError_t e=(x); if(e){ \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
  return 2; } }while(0)

int main() {
  const int batch = 3, vocab = 40, words = (vocab + 31) / 32, n_req = 4;
  std::vector<int> rid = {0, 1, -1};       // row0, row1, idle
  std::vector<int> step(n_req, 5);
  std::vector<int32_t> seq(n_req, 5);      // mask already published for step 5
  std::vector<int32_t> bm(n_req * words, 0);
  // row 0: even tokens allowed; row 1: tokens < 10 allowed
  for (int t = 0; t < vocab; t++) {
    if (t % 2 == 0) bm[0 * words + t / 32] |= (1 << (t % 32));
    if (t < 10)     bm[1 * words + t / 32] |= (1 << (t % 32));
  }
  std::vector<float> hin(batch * vocab, 1.0f);
  std::vector<bfloat16> hin_bf(batch * vocab);
  for (size_t i = 0; i < hin.size(); i++) hin_bf[i] = __float2bfloat16(hin[i]);

  bfloat16 *d_in, *d_out; int32_t *d_bm, *d_seq, *d_flag; int *d_rid, *d_step;
  CK(cudaMalloc(&d_in,  batch * vocab * sizeof(bfloat16)));
  CK(cudaMalloc(&d_out, batch * vocab * sizeof(bfloat16)));
  CK(cudaMalloc(&d_bm,  bm.size() * sizeof(int32_t)));
  CK(cudaMalloc(&d_seq, n_req * sizeof(int32_t)));
  CK(cudaMalloc(&d_flag, sizeof(int32_t)));
  CK(cudaMalloc(&d_rid, batch * sizeof(int)));
  CK(cudaMalloc(&d_step, n_req * sizeof(int)));
  CK(cudaMemcpy(d_in, hin_bf.data(), hin_bf.size()*sizeof(bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bm, bm.data(), bm.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_seq, seq.data(), n_req*sizeof(int32_t), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rid, rid.data(), batch*sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_step, step.data(), n_req*sizeof(int), cudaMemcpyHostToDevice));

  std::vector<bfloat16> hout(batch * vocab);
  for (int fl = 0; fl < 2; fl++) {
    int32_t flag = fl;
    CK(cudaMemcpy(d_flag, &flag, sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0, batch * vocab * sizeof(bfloat16)));
    probe<<<1, 256>>>(d_in, d_out, d_bm, d_seq, d_flag, d_rid, d_step,
                      vocab, words, batch);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hout.data(), d_out, hout.size()*sizeof(bfloat16), cudaMemcpyDeviceToHost));

    for (int b = 0; b < batch; b++) {
      int row = rid[b];
      for (int t = 0; t < vocab; t++) {
        float got = __bfloat162float(hout[b * vocab + t]);
        float want = 1.0f;
        bool do_mask = (fl == 1) && (row >= 0);
        if (do_mask) {
          bool allowed = (bm[row * words + t / 32] >> (t % 32)) & 1;
          if (!allowed) want = -INFINITY;
        }
        bool ok = (std::isinf(want) && std::isinf(got) && got < 0)
                  || (want == got);
        if (!ok) {
          printf("MISMATCH flag=%d b=%d t=%d row=%d got=%f want=%f\n",
                 fl, b, t, row, got, want);
          return 1;
        }
      }
    }
  }
  printf("KERNEL OK\n");
  return 0;
}
"""


def _nvcc():
    return shutil.which("nvcc")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.skipif(_nvcc() is None, reason="needs nvcc")
def test_apply_token_bitmask_kernel_cuda():
    cc = torch.cuda.get_device_capability(0)
    arch = f"sm_{cc[0]}{cc[1]}a" if cc[0] >= 9 else f"sm_{cc[0]}{cc[1]}"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "check.cu")
        binp = os.path.join(d, "check")
        with open(src, "w") as f:
            f.write(_CU_SRC)
        compile_cmd = [
            _nvcc(), "-std=c++17", f"-arch={arch}", src, "-o", binp,
            "-I", str(KERNEL_DIR),
        ]
        cp = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert cp.returncode == 0, f"nvcc failed:\n{cp.stderr}"
        rp = subprocess.run([binp], capture_output=True, text=True)
        assert rp.returncode == 0, f"kernel test failed:\n{rp.stdout}\n{rp.stderr}"
        assert "KERNEL OK" in rp.stdout


if __name__ == "__main__":
    print("[host] StructuredGenerationManager ...", flush=True)
    test_structured_manager_host()
    print("  PASS")
    if torch.cuda.is_available() and _nvcc() is not None:
        print("[cuda] apply_token_bitmask kernel ...", flush=True)
        test_apply_token_bitmask_kernel_cuda()
        print("  PASS")
    else:
        print("[cuda] skipped (no GPU/nvcc)")
    print("ALL PASS")
