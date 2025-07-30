#!/usr/bin/env python
import torch
import numpy as np
from mirage import DTensor
from mirage.persistent_kernel import PersistentKernel
import tempfile
import subprocess
import sys

def test_minimal_multi_token_linear():
    # Simple configuration for testing
    num_tokens = 4
    reduction_size = 512  # Smaller for simplicity
    output_size = 384
    
    # Create input tensors
    input_np = np.random.randn(1, num_tokens * reduction_size).astype(np.float32)
    weight_np = np.random.randn(reduction_size, output_size).astype(np.float32)
    residual_np = np.random.randn(1, num_tokens * output_size).astype(np.float32)
    
    # Convert to torch tensors for reference computation
    input_torch = torch.from_numpy(input_np).bfloat16()
    weight_torch = torch.from_numpy(weight_np).bfloat16()
    residual_torch = torch.from_numpy(residual_np).bfloat16()
    
    # Reference implementation - reshape and compute per token
    input_reshaped = input_torch.view(num_tokens, reduction_size)
    output_ref = torch.matmul(input_reshaped, weight_torch) + residual_torch.view(num_tokens, output_size)
    output_ref = output_ref.view(1, -1)
    
    # Create DTensors
    input_dtensor = DTensor((1, num_tokens * reduction_size), dtype=DTensor.bfloat16)
    weight_dtensor = DTensor((reduction_size, output_size), dtype=DTensor.bfloat16)
    residual_dtensor = DTensor((1, num_tokens * output_size), dtype=DTensor.bfloat16)
    output_dtensor = DTensor((1, num_tokens * output_size), dtype=DTensor.bfloat16)
    
    # Initialize MPK  
    mpk = PersistentKernel()
    
    # First operation: dummy embedding to satisfy single-block constraint
    dummy_tokens = DTensor((1,), dtype=DTensor.int64)
    dummy_embedding = DTensor((10, reduction_size), dtype=DTensor.bfloat16)  # Small vocab size
    mpk.embed_layer(
        inputs=dummy_tokens,
        weight=dummy_embedding,
        output=input_dtensor,
        grid_dim=(1, 1, 1),
        block_dim=(64, 1, 1)
    )
    
    # Multi-token linear operation
    mpk.multi_token_linear_layer(
        input=input_dtensor,
        weight=weight_dtensor,
        residual=residual_dtensor,
        output=output_dtensor,
        grid_dim=(num_tokens, 1, 1),
        block_dim=(128, 1, 1),
        max_tokens=64
    )
    
    # Allocate and copy input data to GPU
    mpk.allocate()
    mpk.random()
    
    # Copy actual input data
    input_dtensor_ptr = input_dtensor.get_ptr()
    weight_dtensor_ptr = weight_dtensor.get_ptr()
    residual_dtensor_ptr = residual_dtensor.get_ptr()
    output_dtensor_ptr = output_dtensor.get_ptr()
    
    # Convert to device tensors
    input_device = torch.from_numpy(input_np).bfloat16().cuda()
    weight_device = torch.from_numpy(weight_np).bfloat16().cuda()
    residual_device = torch.from_numpy(residual_np).bfloat16().cuda()
    
    # Copy to MPK tensors
    torch.cuda.synchronize()
    input_device_ptr = input_device.data_ptr()
    weight_device_ptr = weight_device.data_ptr()
    residual_device_ptr = residual_device.data_ptr()
    
    import ctypes
    cuda = ctypes.CDLL('libcudart.so')
    cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    
    cuda.cudaMemcpy(input_dtensor_ptr, input_device_ptr, 
                    input_device.numel() * input_device.element_size(), 1)  # cudaMemcpyDeviceToDevice
    cuda.cudaMemcpy(weight_dtensor_ptr, weight_device_ptr,
                    weight_device.numel() * weight_device.element_size(), 1)
    cuda.cudaMemcpy(residual_dtensor_ptr, residual_device_ptr,
                    residual_device.numel() * residual_device.element_size(), 1)
    
    # Run kernel
    torch.cuda.synchronize()
    mpk.run(step=0, tokens=[])
    torch.cuda.synchronize()
    
    # Read output
    output_device = torch.zeros(1, num_tokens * output_size, dtype=torch.bfloat16).cuda()
    cuda.cudaMemcpy(output_device.data_ptr(), output_dtensor_ptr,
                    output_device.numel() * output_device.element_size(), 1)
    torch.cuda.synchronize()
    
    # Compare results
    output_cpu = output_device.cpu()
    diff = torch.abs(output_cpu - output_ref)
    max_diff = torch.max(diff).item()
    rel_diff = max_diff / torch.max(torch.abs(output_ref)).item()
    
    print(f"Test configuration:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  reduction_size: {reduction_size}")
    print(f"  output_size: {output_size}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Max relative difference: {rel_diff:.6f}")
    
    # Check if results match within tolerance
    tolerance = 1e-2  # BFloat16 precision
    if max_diff < tolerance:
        print("✓ Test PASSED: Multi-token linear results match!")
        return True
    else:
        print("✗ Test FAILED: Results do not match")
        print(f"Output shape: {output_cpu.shape}")
        print(f"Reference shape: {output_ref.shape}")
        # Print a few values for debugging
        print("\nFirst few output values:")
        print(f"MPK:       {output_cpu[0, :10]}")
        print(f"Reference: {output_ref[0, :10]}")
        return False

if __name__ == "__main__":
    success = test_minimal_multi_token_linear()
    sys.exit(0 if success else 1)