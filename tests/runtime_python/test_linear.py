import mirage as mi
import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False)

reduction_size = 4096
output_sizes = [4096]

batch_size = 1

seed = 42
torch.manual_seed(seed)

for output_size in output_sizes:
    print(f"\n=== Testing output_size = {output_size} ===")
    x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((batch_size, output_size), device="cuda", dtype=torch.bfloat16)
    # x = torch.ones((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    # w = torch.ones((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    # residual = torch.ones((batch_size, output_size), device="cuda", dtype=torch.bfloat16) * 32
    # residual = torch.zeros((batch_size, output_size), device="cuda", dtype=torch.bfloat16) * 32
    output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

    runtime_kernel.linear(x, w, residual, output)
    print("kernel output 1:\n", output)
    # print(output[0][955 : 965])
    # print(output[0][4075 : 4080])
    # runtime_kernel.tail_linear(x, w, residual, output)
    print("kernel output 2:\n", output)
    print("residual:\n", residual)
    torch_out = torch.matmul(x, torch.transpose(w, 0, 1)) + residual
    print("torch output:\n", torch_out)

    print("Ratio (kernel / torch):")
    ratio = output / torch_out
    ratio = ratio[0]
    print("ratio:")
    print(ratio)
    print(ratio.shape)
    print(torch.allclose(ratio, torch.ones(ratio.shape, device="cuda", dtype=torch.bfloat16)))
    print(torch.max(ratio))
    print(torch.min(ratio))

    diff = output - torch_out
    diff = diff[0]
    print("diff:")
    print(diff)
    print(diff.shape)
    print(torch.max(diff))
    print(torch.min(diff))


    top_10_gaps, top_10_indices = torch.topk(diff, k=10)

    print("\nTop 10 gap scalars:")
    print(f"Values: {top_10_gaps}")
    print(f"Indices: {top_10_indices}")

    print("\nCorresponding values from original tensors:")
    for i in range(10):
        index = top_10_indices[i].item()
        gap = top_10_gaps[i].item()
        val1 = output[0][index].item()
        val2 = torch_out[0][index].item()
        print(f"Index {index}: Gap = {gap:.2f}, Tensor1 value = {val1:.2f}, Tensor2 value = {val2:.2f}")
    
    idx = 3422
    print("output[0][idx]: ", output[0][idx])
    print("torch_out[0][idx]: ", torch_out[0][idx])

    print(torch_out[0][4080 :])
    print(output[0][4080 :])
    # Warm-up
    # for _ in range(16):
    #     runtime_kernel.linear(x, w, residual, output)

    # torch.cuda.synchronize()
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    #     enable_timing=True
    # )
    # repetitions = 1000
    # starter.record()
    # for rep in range(repetitions):
    #     runtime_kernel.linear(x, w, residual, output)
    # ender.record()
    # torch.cuda.synchronize()
    # total_time = starter.elapsed_time(ender)
    # avg_time = total_time / repetitions
    # print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")

    # Compare with Mirage