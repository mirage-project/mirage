import torch
import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import make_sequential_nvfp4_tensors, nvfp4_block_scaled_matmul, nvfp4_scaled_mm, make_random_nvfp4_tensors, interleave_sf_tensor, make_unit_scale_factors

torch.set_printoptions(sci_mode=False, profile="full")

# BATCH_SIZE must be divisible by MMA_M = 128
# OUTPUT_SIZE must be divisible by MMA_N = 128
# REDUCTION_SIZE must be divisible by bK = 256
REDUCTION_SIZE = 1024*4
OUTPUT_SIZE = 1024*4
BATCH_SIZE = 1024*4

if __name__ == "__main__":
    # Test 2: Random nvfp4 tensors
    x, w, x_sf, w_sf = make_random_nvfp4_tensors(
        BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE
    )
    x_sf_interleaved = interleave_sf_tensor(x_sf)
    w_sf_interleaved = interleave_sf_tensor(w_sf)
    residual = None
    output = torch.empty(BATCH_SIZE, OUTPUT_SIZE, device="cuda", dtype=torch.float32)

    runtime_kernel_blackwell.linear_nvfp4_1d2d_sm100(x, x_sf_interleaved, w, w_sf_interleaved, residual, output)
