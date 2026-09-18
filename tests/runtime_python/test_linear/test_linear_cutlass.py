import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)


reduction_sizes = [4096]
output_sizes = [64]
batch_size = 8

for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing output_size = {output_size} reduction_size = {reduction_size} ==="
        )

        x = torch.randn(
            (batch_size, reduction_size), device="cuda", dtype=torch.bfloat16
        )
        w = torch.randn(
            (output_size, reduction_size), device="cuda", dtype=torch.bfloat16
        )
        residual = torch.randn(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16
        )
        mirage_output = torch.empty(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16
        )

        runtime_kernel.linear(x, w, residual, mirage_output)  # without residual

        torch_out = torch.matmul(x, torch.transpose(w, 0, 1))

        print("Ratio (kernel / torch):")
        print(mirage_output / torch_out)


        diff = (mirage_output - torch_out).abs()

        # max diff value and flat index
        max_val, flat_idx = diff.view(-1).max(dim=0)

        # coordinates of that max
        coords = torch.unravel_index(flat_idx, diff.shape)

        print("max |diff|:", max_val.item())
        print("at index:", tuple(c.item() for c in coords))
        print("mirage_output:", mirage_output[coords].item())
        print("torch_out:    ", torch_out[coords].item())
        print("signed diff:  ", (mirage_output - torch_out)[coords].item())

        torch.testing.assert_close(mirage_output, torch_out, rtol=1e-1, atol=1e-1)
