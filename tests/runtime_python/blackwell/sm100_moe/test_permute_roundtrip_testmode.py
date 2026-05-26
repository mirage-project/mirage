"""MoE permute <-> unpermute round-trip test (skeleton).

Intent: build a tiny graph that runs moe_permute_sm100_layer with a known
random routing, copies the permuted FP8 buffer through identity (or feeds
it directly back via an in-graph view) into moe_unpermute_sm100_layer,
and asserts the unpermuted output matches a reference computation
  output[t] = residual[t] + sum_k(weights[t,k] * dequant(permuted[t2p[t,k]]))
using the same FP8 quantization MPK does internally.

Currently SKIPPED because writing a faithful FP8 round-trip requires
mirroring MPK's per-token-group quantize + UE8M0-packed scale layout
exactly, and the in-graph identity copy needs a TASK_IDENTITY shim that
doesn't yet exist on SM100. Tracked as a follow-up; the kernel itself
is exercised end-to-end by demo/deepseek_v3/demo.py.
"""
import pytest


@pytest.mark.skip(
    reason="needs FP8 ref + TASK_IDENTITY shim; covered indirectly by DSv3 demo")
def test_permute_unpermute_roundtrip():
    raise AssertionError("placeholder — see module docstring")


if __name__ == "__main__":
    print(__doc__)
