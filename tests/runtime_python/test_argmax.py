import torch
import runtime_kernel

batch_size = 1
vocab_size = 32 * 1024
logits = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16, device='cuda')

output = torch.empty(1, device='cuda', dtype=torch.int32)
runtime_kernel.argmax(logits, output)
print(output)


torch_output = torch.argmax(logits, dim=1)
print(torch_output)
