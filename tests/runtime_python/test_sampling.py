import mirage as mi
import torch
import numpy as np

torch.set_printoptions(sci_mode=False)

vocab_size = 50257  # GPT-2 vocab size
batch_sizes = [1, 4, 8]

print("Testing Sampling from Logits with Gumbel-Max Trick\n")

for batch_size in batch_sizes:
    print(f"\n=== Testing batch_size = {batch_size}, vocab_size = {vocab_size} ===")

    # Create test data
    logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.float32)

    # Set random seed for reproducibility
    seed = 42

    # Reference implementation using torch.multinomial with Gumbel-Max trick
    def torch_sampling_from_logits(logits, seed=42):
        torch.manual_seed(seed)
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        # Add to logits and find argmax
        noisy_logits = logits + gumbel_noise
        sampled_indices = torch.argmax(noisy_logits, dim=-1)
        return sampled_indices

    # Test with torch reference
    torch_output = torch_sampling_from_logits(logits, seed=seed)
    print(f"Torch sampled tokens: {torch_output.cpu().numpy()}")

    # Verify outputs are in valid range
    assert torch.all(torch_output >= 0) and torch.all(torch_output < vocab_size), \
        f"Sampled tokens out of range: {torch_output}"

    # Warm-up
    for _ in range(16):
        _ = torch_sampling_from_logits(logits, seed=seed)

    torch.cuda.synchronize()
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        _ = torch_sampling_from_logits(logits, seed=seed)
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions
    print(f"Torch average time over {repetitions} runs: {avg_time:.6f} ms")

    # Test different temperature values
    temperatures = [0.7, 1.0, 1.5]
    for temp in temperatures:
        scaled_logits = logits / temp
        sampled = torch_sampling_from_logits(scaled_logits, seed=seed)
        print(f"Temperature {temp}: sampled tokens = {sampled.cpu().numpy()}")

    print(f"✓ Test passed for batch_size = {batch_size}")

print("\n=== All tests passed ===")
