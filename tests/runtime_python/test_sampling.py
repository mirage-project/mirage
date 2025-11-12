import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False)

vocab_size = 50257  # GPT-2 vocab size
batch_sizes = [1, 4, 8]


def torch_sampling_from_logits(logits, seed=42):
    torch.manual_seed(seed)
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    # Add to logits and find argmax
    noisy_logits = logits + gumbel_noise
    sampled_indices = torch.argmax(noisy_logits, dim=-1)
    return sampled_indices


for batch_size in batch_sizes:
    print(f"\n=== Testing batch_size = {batch_size}, vocab_size = {vocab_size} ===")

    logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.float32)
    output = torch.empty(batch_size, device="cuda", dtype=torch.int32)
    seed = 42

    runtime_kernel.sampling_from_logits(logits, output, seed)
    torch_out = torch_sampling_from_logits(logits, seed=seed)

    print("Kernel output:", output.cpu().numpy())
    print("Torch output: ", torch_out.cpu().numpy())

    continue

    # Warm-up
    for _ in range(16):
        runtime_kernel.sampling_from_logits(logits, output, seed)

    torch.cuda.synchronize()
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        runtime_kernel.sampling_from_logits(logits, output, seed)
    ender.record()
    torch.cuda.synchronize()

    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions

    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
