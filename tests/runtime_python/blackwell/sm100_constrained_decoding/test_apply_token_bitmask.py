"""
Tests for MPK constrained decoding (xgrammar). Run:
    pytest test_apply_token_bitmask.py        # or: python test_apply_token_bitmask.py

Host tests drive StructuredGenerationManager with mock pinned buffers (CPU
tensors) + a synthetic grammar, checking the per-step bitmask fill, accept
sequencing, prefill/decode handling, and the flag. The CUDA test compiles
apply_token_bitmask_sm100.cuh and runs it on the GPU vs a CPU reference. None
need the full megakernel build.
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
STRUCTURED_PY = REPO_ROOT / "python/mirage/mpk/structured.py"
KERNEL_DIR = REPO_ROOT / "include/mirage/persistent_kernel/tasks/blackwell"


def _setup(vocab, n_req=4, max_seq=48):
    """Build a StructuredGenerationManager over a toy char vocab + mock pinned
    buffers. Returns (mgr, buffers, rows, vocab, V, allowed(row))."""
    import xgrammar as xgr
    spec = importlib.util.spec_from_file_location("mpk_structured", STRUCTURED_PY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    vocab = list(vocab)
    V = len(vocab); words = (V + 31) // 32
    ti = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=V)
    B = dict(token_bitmask=torch.full((n_req, words), -1, dtype=torch.int32),
             mask_seq=torch.zeros(n_req, dtype=torch.int32),
             constrained_flag=torch.zeros(1, dtype=torch.int32),
             tokens=torch.zeros(n_req, max_seq, dtype=torch.int64),
             pinned_step=torch.zeros(n_req, dtype=torch.int32),
             prompt_lengths=torch.zeros(n_req, dtype=torch.int32))
    rows = {}
    mgr = mod.StructuredGenerationManager(
        find_row_for_rid=lambda r: rows.get(r, -1), **B)
    mgr.init_xgrammar(tokenizer_info=ti, vocab_size=V)

    def allowed(row):
        bm = B["token_bitmask"]
        return [i for i in range(V) if (bm[row, i // 32].item() >> (i % 32)) & 1]
    return mgr, B, rows, vocab, V, allowed


def test_structured_manager_host():
    """Base grammar: prefill (fresh mask, no accept) → decode (accept + mask),
    cross-checked against a reference matcher, plus flag + idempotency."""
    import xgrammar as xgr
    pytest.importorskip("xgrammar")
    EBNF = 'root ::= ("a" | "b")+'
    mgr, B, rows, vocab, V, allowed = _setup("abcdefgh")
    mgr.set_request_grammar(0, ebnf=EBNF)
    assert int(B["constrained_flag"][0]) == 1

    row = 2; rows[0] = row; B["prompt_lengths"][row] = 3
    # Prefill (step < prompt_len): fresh mask, no token accepted.
    B["pinned_step"][row] = 2; mgr.tick()
    assert int(B["mask_seq"][row]) == 2 and allowed(row) == [0, 1]
    # Decode: first token 'a' (id 0) lands at tokens[row, 3]; accept + mask next.
    B["tokens"][row, 3] = 0; B["pinned_step"][row] = 3; mgr.tick()
    assert int(B["mask_seq"][row]) == 3

    ref = xgr.GrammarMatcher(mgr._compiler.compile_grammar(EBNF))
    assert ref.accept_token(0)
    rb = xgr.allocate_token_bitmask(1, V); ref.fill_next_token_bitmask(rb, index=0)
    assert B["token_bitmask"][row, 0].item() == rb[0, 0].item()

    snap = B["token_bitmask"][row, 0].item(); mgr.tick()  # idempotent within a step
    assert B["token_bitmask"][row, 0].item() == snap
    mgr.release(0); assert int(B["constrained_flag"][0]) == 0


def test_structured_manager_structural_tag():
    """Structural tag (tool-calling): free text until the trigger, then the
    schema. Driven through the manager's decode path; after the trigger+begin
    only '{' is allowed."""
    import json
    import xgrammar as xgr
    pytest.importorskip("xgrammar")
    mgr, B, rows, vocab, V, allowed = _setup(' \t\n{}[]":,.0123456789'
                                             'abcdefghijklmnopqrstuvwxyz<>=/_')
    import xgrammar.structural_tag as st
    idx = {c: i for i, c in enumerate(vocab)}
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    stag = st.StructuralTag(format=st.TriggeredTagsFormat(
        triggers=["<function="],
        tags=[st.TagFormat(begin="<function=get_weather>",
                           content=st.JSONSchemaFormat(json_schema=schema),
                           end="</function>")]))
    mgr.set_request_grammar(0, structural_tag=stag)
    assert int(B["constrained_flag"][0]) == 1

    row = 1; rows[0] = row; P = 2; B["prompt_lengths"][row] = P
    B["pinned_step"][row] = P - 1; mgr.tick()           # prefill → free text
    assert len(allowed(row)) > V // 2

    ref = xgr.GrammarMatcher(mgr._compiler.compile_structural_tag(stag))
    rb = xgr.allocate_token_bitmask(1, V); ref.fill_next_token_bitmask(rb, index=0)
    assert B["token_bitmask"][row, 0].item() == rb[0, 0].item()

    # Generate the trigger+begin; token k lands at tokens[row, P+k].
    begin = "<function=get_weather>"
    for k, ch in enumerate(begin):
        B["tokens"][row, P + k] = idx[ch]
    for s in range(P, P + len(begin)):
        B["pinned_step"][row] = s; mgr.tick()
    assert [vocab[i] for i in allowed(row)] == ['{']
    mgr.release(0); assert int(B["constrained_flag"][0]) == 0


# ── CUDA kernel correctness ────────────────────────────────────────────────

_CU_SRC = r"""
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>
typedef __nv_bfloat16 bfloat16;
#include "apply_token_bitmask_sm100.cuh"

__global__ void probe(void const* in, void* out, int32_t* bm, int32_t* seq,
                      int32_t* flag, int* rid, int* step, int* qo,
                      int v, int w, int nreq) {
  kernel::apply_token_bitmask_sm100_kernel<bfloat16, 4>(
      in, out, bm, seq, flag, rid, step, qo, v, w, nreq);
}

#define CK(x) do{ cudaError_t e=(x); if(e){ \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
  return 2; } }while(0)

int main() {
  // BATCH=4 logits rows; 2 decode requests (1 token each) on rows 5,6;
  // qo_indptr=[0,1,2] -> req b's next-token logits are row b. Rows 2,3 unused.
  const int BATCH = 4, vocab = 40, words = (vocab + 31) / 32, n_req = 8;
  const int num_requests = 2;
  std::vector<int> rid = {5, 6}, qo = {0, 1, 2}, step(n_req, 5);
  std::vector<int32_t> seq(n_req, 5), bm(n_req * words, 0);
  for (int t = 0; t < vocab; t++) {
    if (t % 2 == 0) bm[5 * words + t / 32] |= (1 << (t % 32)); // row5: even
    if (t < 10)     bm[6 * words + t / 32] |= (1 << (t % 32)); // row6: t<10
  }
  std::vector<bfloat16> hin(BATCH * vocab), hout(BATCH * vocab);
  for (auto &x : hin) x = __float2bfloat16(1.0f);

  bfloat16 *d_in, *d_out; int32_t *d_bm, *d_seq, *d_flag; int *d_rid, *d_step, *d_qo;
  CK(cudaMalloc(&d_in, BATCH*vocab*2)); CK(cudaMalloc(&d_out, BATCH*vocab*2));
  CK(cudaMalloc(&d_bm, bm.size()*4)); CK(cudaMalloc(&d_seq, n_req*4));
  CK(cudaMalloc(&d_flag, 4)); CK(cudaMalloc(&d_rid, num_requests*4));
  CK(cudaMalloc(&d_step, n_req*4)); CK(cudaMalloc(&d_qo, (num_requests+1)*4));
  CK(cudaMemcpy(d_in, hin.data(), BATCH*vocab*2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bm, bm.data(), bm.size()*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_seq, seq.data(), n_req*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rid, rid.data(), num_requests*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_step, step.data(), n_req*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_qo, qo.data(), (num_requests+1)*4, cudaMemcpyHostToDevice));

  for (int fl = 0; fl < 2; fl++) {
    CK(cudaMemcpy(d_flag, &fl, 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0, BATCH*vocab*2));
    probe<<<1, 256>>>(d_in, d_out, d_bm, d_seq, d_flag, d_rid, d_step, d_qo,
                      vocab, words, num_requests);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hout.data(), d_out, BATCH*vocab*2, cudaMemcpyDeviceToHost));
    for (int pos = 0; pos < BATCH; pos++) {
      int row = (pos == 0) ? 5 : (pos == 1) ? 6 : -1;  // pos 2,3 -> plain copy
      for (int t = 0; t < vocab; t++) {
        float got = __bfloat162float(hout[pos * vocab + t]), want = 1.0f;
        if (fl == 1 && row >= 0 && !((bm[row*words + t/32] >> (t%32)) & 1))
          want = -INFINITY;
        if (!((std::isinf(want) && std::isinf(got) && got < 0) || want == got)) {
          printf("MISMATCH fl=%d pos=%d t=%d got=%f want=%f\n", fl, pos, t, got, want);
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
        src, binp = os.path.join(d, "check.cu"), os.path.join(d, "check")
        open(src, "w").write(_CU_SRC)
        cp = subprocess.run([_nvcc(), "-std=c++17", f"-arch={arch}", src, "-o",
                             binp, "-I", str(KERNEL_DIR)],
                            capture_output=True, text=True)
        assert cp.returncode == 0, f"nvcc failed:\n{cp.stderr}"
        rp = subprocess.run([binp], capture_output=True, text=True)
        assert rp.returncode == 0 and "KERNEL OK" in rp.stdout, rp.stdout + rp.stderr


if __name__ == "__main__":
    test_structured_manager_host(); print("[host] base          PASS")
    test_structured_manager_structural_tag(); print("[host] structural    PASS")
    if torch.cuda.is_available() and _nvcc():
        test_apply_token_bitmask_kernel_cuda(); print("[cuda] kernel        PASS")
    else:
        print("[cuda] skipped (no GPU/nvcc)")
    print("ALL PASS")
