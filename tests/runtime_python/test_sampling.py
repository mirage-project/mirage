import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False)

vocab_size = 50257  # GPT-2 vocab size
batch_size = 1  # Fixed batch size for testing


def normal_distribution(shape, device, std=1.0):
    """Generate logits from normal distribution."""
    return torch.randn(shape, device=device) * std


def gumbel_distribution(shape, device, beta=1.0):
    """Generate logits from Gumbel distribution."""
    U = torch.rand(shape, device=device)
    eps = 1e-20
    return torch.log(-torch.log(U + eps) + eps) / beta


# Test 1: Verify kernel correctness with extreme logits
print("=== Test 1: Kernel Correctness (Extreme Logits) ===")
logits = torch.full((batch_size, vocab_size), -1000.0, device="cuda", dtype=torch.float32)
expected_idx = 100
logits[0, expected_idx] = 1000.0

output = torch.empty(batch_size, device="cuda", dtype=torch.int32)
runtime_kernel.sampling_from_logits(logits, output, 42)

if output[0] == expected_idx:
    print(f"  PASS - Sampled token {output[0].item()} (expected {expected_idx})")
else:
    print(f"  FAIL - Sampled token {output[0].item()} (expected {expected_idx})")


# Test 2: Verify determinism (same seed, same output)
print("\n=== Test 2: Determinism Test ===")
logits_test = torch.randn((1, vocab_size), device="cuda", dtype=torch.float32)
out1 = torch.empty(1, device="cuda", dtype=torch.int32)
out2 = torch.empty(1, device="cuda", dtype=torch.int32)
runtime_kernel.sampling_from_logits(logits_test, out1, 42)
runtime_kernel.sampling_from_logits(logits_test, out2, 42)
print(f"  Same seed (42): out1={out1[0].item()}, out2={out2[0].item()} - {'PASS' if out1[0]==out2[0] else 'FAIL'}")

out3 = torch.empty(1, device="cuda", dtype=torch.int32)
runtime_kernel.sampling_from_logits(logits_test, out3, 123)
print(f"  Different seed (123): out3={out3[0].item()} - {'PASS' if out3[0]!=out1[0] else 'WARN: Same output'}")


# Test 3: Bounds checking (samples must be in [0, vocab_size))
print("\n=== Test 3: Bounds Checking ===")
logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.float32)
output = torch.empty(batch_size, device="cuda", dtype=torch.int32)

num_trials = 5000
all_valid = True
for trial in range(num_trials):
    seed = 42 + trial
    runtime_kernel.sampling_from_logits(logits, output, seed)
    if torch.any(output < 0) or torch.any(output >= vocab_size):
        all_valid = False
        print(f"  FAIL - out of bounds at trial {trial}")
        break

if all_valid:
    print(f"  PASS - all {num_trials} samples in valid range [0, {vocab_size})")


# Test 4: Statistical frequency test
print("\n=== Test 4: Statistical Frequency Validation ===")
print("This validates that sampling distribution matches expected probabilities")
print("Using cosine similarity metric (should be > 0.99 for correct implementation)\n")

test_configs = [
    ("GPT-2 vocab, Normal (std=1)", 50257, normal_distribution, {"std": 1.0}),
    ("GPT-2 vocab, Normal (std=5)", 50257, normal_distribution, {"std": 5.0}),
    ("GPT-2 vocab, Gumbel (beta=0.1)", 50257, gumbel_distribution, {"beta": 0.1}),
]

for config_name, test_vocab_size, distribution_fn, dist_kwargs in test_configs:
    print(f"Config: {config_name} (vocab_size={test_vocab_size})")

    torch.manual_seed(42)
    num_trials = 5000000  # 5M samples for statistical significance

    # Generate logits from specified distribution
    logits = distribution_fn((1, test_vocab_size), "cuda", **dist_kwargs)

    # Compute expected probability distribution using torch.softmax
    expected_probs = torch.softmax(logits, dim=-1).squeeze(0)

    # Count samples
    counter = torch.zeros(test_vocab_size, dtype=torch.int32, device="cuda")
    output = torch.empty(1, device="cuda", dtype=torch.int32)

    print(f"  Running {num_trials} sampling trials...")
    for trial in range(num_trials):
        seed = trial
        runtime_kernel.sampling_from_logits(logits, output, seed)
        counter[output[0]] += 1

    # Compute empirical frequency distribution
    empirical_freq = counter.float() / num_trials

    # Validate distribution matches expected probabilities
    similarity = torch.cosine_similarity(empirical_freq.unsqueeze(0), expected_probs.unsqueeze(0))

    print(f"  Cosine similarity: {similarity.item():.6f}")
    print(f"  Top-5 expected probs: {torch.topk(expected_probs, 5).values.cpu().numpy()}")
    print(f"  Top-5 empirical freqs: {torch.topk(empirical_freq, 5).values.cpu().numpy()}")

    if similarity > 0.99:
        print(f"  Result: PASS (similarity {similarity.item():.6f} > 0.99)")
    else:
        print(f"  Result: FAIL (similarity {similarity.item():.6f} <= 0.99)")
    print()


# Test 5: Edge case with -inf logits (masked tokens)
print("\n=== Test 5: Masked Tokens Test (-inf logits) ===")
for zero_ratio in [0.0, 0.5, 0.9]:
    print(f"Zero ratio: {zero_ratio}")
    torch.manual_seed(42)

    test_vocab_size = 50257  # Must use supported vocab size
    num_trials = 1000000

    logits = torch.randn(1, test_vocab_size, device="cuda", dtype=torch.float32)

    # Randomly set some logits to -inf
    zero_indices = torch.randperm(test_vocab_size, device="cuda")[:int(test_vocab_size * zero_ratio)]
    logits[:, zero_indices] = float("-inf")

    output = torch.empty(1, device="cuda", dtype=torch.int32)

    # Verify we never sample from -inf positions
    invalid_samples = 0
    for trial in range(num_trials):
        runtime_kernel.sampling_from_logits(logits, output, trial)
        sampled_idx = output[0].item()
        if sampled_idx in zero_indices:
            invalid_samples += 1

    if invalid_samples == 0:
        print(f"  PASS - No samples from masked positions ({num_trials} trials)")
    else:
        print(f"  FAIL - {invalid_samples} samples from masked positions")


# Test 6: Uniform distribution test
print("\n=== Test 6: Uniform Distribution Test ===")
test_vocab_size = 50257  # Must use supported vocab size
num_trials = 100000

logits = torch.zeros((1, test_vocab_size), device="cuda", dtype=torch.float32)
output = torch.empty(1, device="cuda", dtype=torch.int32)

counter = torch.zeros(test_vocab_size, dtype=torch.int32, device="cuda")
for trial in range(num_trials):
    runtime_kernel.sampling_from_logits(logits, output, trial)
    counter[output[0]] += 1

empirical_freq = counter.float() / num_trials
expected_prob = 1.0 / test_vocab_size

mean_freq = empirical_freq.mean().item()
std_freq = empirical_freq.std().item()

print(f"  Expected probability: {expected_prob:.8f}")
print(f"  Mean frequency: {mean_freq:.8f}")
print(f"  Std deviation: {std_freq:.8f}")

# With large vocab, tolerance needs to scale with 1/sqrt(vocab_size)
tolerance = 5.0 / (test_vocab_size ** 0.5)  # ~0.00002 for vocab_size=50257
if abs(mean_freq - expected_prob) < tolerance:
    print(f"  Result: PASS (within tolerance {tolerance:.8f})")
else:
    print(f"  Result: FAIL (difference {abs(mean_freq - expected_prob):.8f} > {tolerance:.8f})")


# Test 7: Performance benchmarking
print("\n=== Test 7: Performance Benchmark ===")
print(f"Batch size = {batch_size}, vocab_size = {vocab_size}")

logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.float32)
output = torch.empty(batch_size, device="cuda", dtype=torch.int32)
seed = 42

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

print(f"  Average time over {repetitions} runs: {avg_time:.6f} ms")


print("\n=== All Tests Complete ===")
