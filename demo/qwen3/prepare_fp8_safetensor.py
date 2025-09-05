"""
Run this script to rearrange the FP8 weights of Qwen3-8B-FP8 model
to the layout expected by our custom fp8 CUDA kernel.
This is for improving the memory access speed during the GEMM operation. It's only for sm80 a100 GPUs.
"""


import torch
from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os
from typing import List, Dict, Iterable, Optional
from safetensors import safe_open

def get_fp8_tensors() -> Dict[int, Dict[str, torch.Tensor]]:
    files = origin_files
    device = "cpu"  
    layer_weights = {}
    for path in files:
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                if "self_attn.o_proj" in key and ("weight" in key or "weight_scale_inv" in key):
                    parts = key.split(".")
                    try:
                        layer_id = int(parts[2])  
                    except ValueError:
                        continue
                    if layer_id not in layer_weights:
                        layer_weights[layer_id] = {}
                    tensor = f.get_tensor(key)
                    if "weight_scale_inv" in key:
                        layer_weights[layer_id]["weight_scale_inv"] = tensor
                    elif "weight" in key:
                        layer_weights[layer_id]["weight"] = tensor
    return layer_weights


def make_block_indices(device=torch.device("cuda")):
    rows_src = torch.empty((16, 16), dtype=torch.long, device=device)
    cols_src = torch.empty((16, 16), dtype=torch.long, device=device)

    for a in range(4):
        r0 = 2 * a
        r_seq = torch.tensor([r0, r0+1, r0+8, r0+9, r0, r0+1, r0+8, r0+9], device=device)
        for c0 in range(8):
            col_new = 2 * c0 + (a // 2)
            row_base = (a % 2) * 8
            c_seq = torch.tensor([c0, c0, c0, c0, c0+8, c0+8, c0+8, c0+8], device=device)

            rows_src[row_base:row_base+8, col_new] = r_seq
            cols_src[row_base:row_base+8, col_new] = c_seq

    return rows_src, cols_src



def relayout_fp8_4096x4096(w: torch.Tensor) -> torch.Tensor:
    """
    w: (4096, 4096), dtype=torch.float8_e4m3fn
    Returns the re-laid-out tensor with the same dtype and shape.
    """
    assert w.dtype == torch.float8_e4m3fn and w.shape == (4096, 4096)

    # View into 16x16 tiles laid out on a 256x256 grid
    # Shape: (Bi=256, R=16, Bj=256, C=16)
    x = w.view(256, 16, 256, 16)

    # Decompose rows: r = tb*8 + a*2 + r2
    #  -> (tb=2, a=4, r2=2)
    # Decompose cols: c = c8*8 + c0
    #  -> (c8=2, c0=8)
    x = x.view(256, 2, 4, 2, 256, 2, 8)  # (Bi, tb, a, r2, Bj, c8, c0)

    # Split a(4) -> (a_low2=2, a_high1=2), with a = a_low2 + 2*a_high1
    x = x.view(256, 2, 2, 2, 2, 256, 2, 8)  # (Bi, tb, a_low2, a_high1, r2, Bj, c8, c0)

    # We want:
    #   rows_new order: (a_low2, c8, tb, r2)  -> 2*2*2*2 = 16 rows
    #   cols_new order: (c0, a_high1)        -> 8*2     = 16 cols
    #   and tile grid: (Bi, Bj)
    y = x.permute(0, 5, 3, 6, 1, 4, 7, 2).contiguous()  # (Bi, Bj, a_low2, c8, tb, r2, c0, a_high1)

    # Collapse groups to (Bi, Bj, 16, 16), then to (4096, 4096)
    y = y.view(256, 256, 16, 16)
    y = y.permute(0, 2, 1, 3).contiguous().view(4096, 4096)

    # # Keep your original intent of forcing a contiguous row-major layout
    # y = y.t().contiguous().t()
    # temp
    y = y.t()
    return y.contiguous() 

origin_files = [
    "../../models/Qwen3-8B-FP8/model-00001-of-00002.safetensors",
    "../../models/Qwen3-8B-FP8/model-00002-of-00002.safetensors",
]

rearranged_fp8tensor_path = "../../models/Qwen3-8B-FP8v3-reorder"

if __name__ == "__main__":
    layers_fp8 = get_fp8_tensors()
    rows_src, cols_src = make_block_indices(device=torch.device("cuda"))
    new_layers_fp8 = {}
    for layer_id, layer_fp8 in layers_fp8.items():
        w = layer_fp8["weight"]  # (4096,4096), float8_e4m3fn
        w = w.t()
        new_w = relayout_fp8_4096x4096(w)
        key0 = "model.layers." + str(layer_id) + ".self_attn.o_proj.weight"
        key1 = "model.layers." + str(layer_id) + ".self_attn.o_proj.weight_scale_inv"
        new_layers_fp8[key0] = new_w.contiguous().cpu()
        new_layers_fp8[key1] = layer_fp8["weight_scale_inv"].contiguous().cpu()

    from safetensors.torch import save_file


    # save the new tensor to a file
    save_path = rearranged_fp8tensor_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file(new_layers_fp8, os.path.join(save_path, "qwen3-8b-fp8v3-reorder.pth"), metadata={"format": "fp8v3"})
