# Based on an implementation from DeepSeek-V3
# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/convert.py

import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file

mapping = {
    "embed_tokens": ("embed", None),
    "input_layernorm": ("attn_norm", None),
    "q_proj": ("wq", 0),
    "q_norm": ("wq", None),
    "k_proj": ("wk", 0),
    "k_norm": ("wk", None),
    "v_proj": ("wv", 0),
    "o_proj": ("wo", 1),
    "post_attention_layernorm": ("post_norm", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", None),
}


def main(hf_ckpt_path, save_path, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    torch.set_num_threads(8)
    state_dicts = [{} for _ in range(mp)]

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                assert name.startswith("model.") or name == "lm_head.weight"
                param: torch.Tensor = f.get_tensor(name)
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                for i in range(mp):
                    new_param = param
                    if dim is not None:
                        assert (
                            param.size(dim) % mp == 0
                        ), f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(
                            dim, i * shard_size, shard_size
                        ).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(
            state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors")
        )

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)

    for file_path in glob(os.path.join(hf_ckpt_path, "*config*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    main(args.hf_ckpt_path, args.save_path, args.model_parallel)
