# Regression test for #794: with a 128-wide output tile, three pipeline
# stages need 110,592 bytes of shared memory, more than the 98,304-byte
# worker budget on sm86. Run on an sm86 GPU (e.g. RTX 3090, A40):
#
#   cd tests/runtime_python/test_linear_smem_budget
#   pip install --no-build-isolation . && python test_linear_smem_budget.py
import torch
import linear_smem_budget

torch.manual_seed(0)
x = torch.randn((1, 1024), device="cuda", dtype=torch.bfloat16)
w = torch.randn((128, 1024), device="cuda", dtype=torch.bfloat16)
output = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)

linear_smem_budget.linear(x, w, output)
expected = (x.float() @ w.float().t()).to(torch.bfloat16)
torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-1)
print("PASS")
