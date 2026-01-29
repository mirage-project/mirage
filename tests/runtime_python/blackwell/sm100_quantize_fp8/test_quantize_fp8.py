import torch
import runtime_kernel_blackwell_quantize_fp8 as runtime_kernel_blackwell

torch.set_printoptions(sci_mode=False, profile="full")
g = torch.Generator(device="cuda").manual_seed(1234)

batch_sizes = [8]
hidden_sizes = [7168]
fp8_max = 448.0
group_size = 128

for batch_size in batch_sizes:
    for hidden_size in hidden_sizes:
        assert hidden_size % group_size == 0
        num_groups = hidden_size // group_size

        print(f"\n=== Testing batch_size = {batch_size} hidden_size = {hidden_size} group_size={group_size} ===")

        x = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.bfloat16)
        output = torch.empty((batch_size, hidden_size), device="cuda", dtype=torch.float8_e4m3fn)

        # per-group scales: [B, G]
        scales = torch.empty((batch_size, num_groups), device="cuda", dtype=torch.float32)

        runtime_kernel_blackwell.quantize_fp8_sm100(x, output, scales, group_size=group_size)

        x_f32 = x.float()

        xg = x_f32.view(batch_size, num_groups, group_size)
        max_abs_g = xg.abs().amax(dim=2)  # [B, G]

        scale_ref = max_abs_g / fp8_max
        scale_ref = torch.where(scale_ref > 0, scale_ref, torch.ones_like(scale_ref))  # [B, G]

        quant_ref = (xg / scale_ref[:, :, None]).to(torch.float8_e4m3fn)  # [B, G, GS]
        quant_ref = quant_ref.view(batch_size, hidden_size)

        dequant = output.float().view(batch_size, num_groups, group_size) * scales[:, :, None]
        dequant = dequant.view(batch_size, hidden_size)

        print("scales:    ", scales[4, :])
        print("scale_ref: ", scale_ref[4, :])

        print("output:    ", output[4, :128])
        print("quant_ref: ", quant_ref[4, :128])

        # print("dequant:   ", dequant[0, :128])
        # print("x_f32:     ", x_f32[0, :128])

        torch.testing.assert_close(scales, scale_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output.float(), quant_ref.float(), rtol=0, atol=0)

        # torch.testing.assert_close(dequant, x_f32, rtol=2e-1, atol=1e-1)

        print("Test passed!")
