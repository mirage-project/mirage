"""Load one GLM-5.3 MoE layer for shard rank of tp.

The checkpoint stores e4m3 expert weights and fp32 scales for 128 x 128 blocks.
The loader selects a TP shard and reorders its weights into pieces.
The e4m3 bytes and fp32 scale values do not change.
scaled_moe_experts and scaled_gemm_tile read these pieces.
mirage.mpk.fp8_fp32scale_weights defines the piece layout.
The tensors are:

  router [E, H] bf16, bias [E] fp32                           the same on every GPU
  w13 [E, 2 IR, H] uint8, w13_sf [E, IR / 128, H / 128, 2] fp32   the routed experts
  w2 [E, H, IR] uint8, w2_sf [E, H / 128, IR / 128] fp32          IR = the part of the expert intermediate on this GPU
                                                              (I / tp): gate / up rows [rank IR, rank IR + IR), down columns
                                                              the same
  the shared expert (its own scaled_gemm_tile nodes; the same pieces and scales as a routed expert):
    sgu [2 IR, H] uint8, sgu_sf [IR / 128, H / 128, 2] fp32       gate_up
    sd [H, IR] uint8, sd_sf [H / 128, IR / 128] fp32              down

download_layer locates the local snapshot or downloads the required shards.
check_config checks the fields required by the kernel implementation.
checkpoint_reader returns a function that reads a tensor by its checkpoint name.
"""
import json
import os

import torch

from mirage.mpk.fp8_fp32scale_weights import pack_w13, pack_w2


# These model fields must match topk_route, the SiLU experts, and demo.py.
EXPECTED_CONFIG = {"hidden_size": 6144, "moe_intermediate_size": 2048, "n_routed_experts": 256, "n_shared_experts": 1,
                   "num_experts_per_tok": 8, "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
                   "norm_topk_prob": True, "routed_scaling_factor": 2.5, "hidden_act": "silu"}


def download_layer(checkpoint: str, layer: int, cache_dir: str = None, revision: str = None) -> str:
    """Return a local snapshot path for the requested layer.

    If checkpoint is a local directory, return it without downloading files.
    Otherwise, download config.json, the tensor index, and the shards with this layer's MLP tensors.
    A shard can also contain tensors from other layers.
    Resolve revision (default: main) to one commit first, so that the index and the shards come from the same commit.
    """
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    sha = HfApi().model_info(checkpoint, revision=revision).sha
    print(f"{checkpoint}: revision {revision or 'main'} = commit {sha}")
    index = hf_hub_download(checkpoint, "model.safetensors.index.json", revision=sha, cache_dir=cache_dir)
    with open(index) as f:
        where = json.load(f)["weight_map"]
    shards = sorted({f for name, f in where.items() if name.startswith(f"model.layers.{layer}.mlp.")})
    if not shards:
        raise ValueError(f"{checkpoint}: no MLP tensors of layer {layer}")
    return snapshot_download(checkpoint, allow_patterns=["config.json", "model.safetensors.index.json"] + shards,
                             revision=sha, cache_dir=cache_dir)


def check_config(snapshot: str, layer: int) -> dict:
    """Check the snapshot config and return its fields.

    Require EXPECTED_CONFIG values, e4m3 weights, and 128 x 128 scale blocks.
    Require layer to follow the dense prefix and remain within the model.
    Raise ValueError when a checked field does not meet these requirements.
    """
    with open(os.path.join(snapshot, "config.json")) as f:
        cfg = json.load(f)
    bad = {k: (cfg.get(k), v) for k, v in EXPECTED_CONFIG.items() if cfg.get(k) != v}
    q = cfg.get("quantization_config", {})
    if q.get("fmt") != "e4m3" or q.get("weight_block_size") != [128, 128]:
        bad["quantization_config"] = (q.get("fmt"), q.get("weight_block_size"))
    if not cfg.get("first_k_dense_replace", 0) <= layer < cfg.get("num_hidden_layers", 0):
        bad["layer"] = (layer, f"MoE layers {cfg.get('first_k_dense_replace')}..{cfg.get('num_hidden_layers', 0) - 1}")
    if bad:
        raise ValueError(f"{snapshot}: config differs from what the kernels compute (got, expected): {bad}")
    return cfg


def checkpoint_reader(snapshot: str):
    """Return read(name), which opens the indexed shard and reads the tensor."""
    from safetensors import safe_open
    with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    def read(name):
        with safe_open(os.path.join(snapshot, where[name]), framework="pt") as f:
            return f.get_tensor(name)
    return read


def layer_tensors(read, layer: int, tp: int = 1, rank: int = 0, num_experts: int = 256, device="cpu") -> dict:
    """Load the router, correction bias, and expert weights for shard rank of tp.

    Select gate/up rows and down columns from the expert intermediate dimension.
    The module docstring defines the returned tensor names and layouts.
    """
    p = f"model.layers.{layer}.mlp."
    w = {"router": read(p + "gate.weight").to(device), "bias": read(p + "gate.e_score_correction_bias").float().to(device)}
    banks = {k: [] for k in ("w13", "w13_sf", "w2", "w2_sf")}
    for expert in [f"experts.{e}." for e in range(num_experts)] + ["shared_experts."]:
        get = lambda proj: (read(p + expert + proj + ".weight").view(torch.uint8).to(device),
                            read(p + expert + proj + ".weight_scale_inv").float().to(device))
        (gate, gs), (up, us), (down, ds) = get("gate_proj"), get("up_proj"), get("down_proj")
        assert gate.shape[0] % (128 * tp) == 0, f"intermediate {gate.shape[0]}: not whole 128-row blocks for each of {tp} GPUs"
        IR = gate.shape[0] // tp
        rows, blocks = slice(rank * IR, rank * IR + IR), slice(rank * IR // 128, (rank * IR + IR) // 128)
        for k, v in zip(("w13", "w13_sf"), pack_w13(gate[rows], gs[blocks], up[rows], us[blocks])):
            banks[k].append(v)
        for k, v in zip(("w2", "w2_sf"), pack_w2(down[:, rows].contiguous(), ds[:, blocks].contiguous())):
            banks[k].append(v)
    H, IR = w["router"].shape[1], banks["w2"][0].shape[1] * 128
    last = {k: v.pop() for k, v in banks.items()}   # the last entry is the shared expert
    w.update(sgu=last["w13"].view(2 * IR, H), sgu_sf=last["w13_sf"], sd=last["w2"].view(H, IR), sd_sf=last["w2_sf"])
    E = len(banks["w13"])
    shapes = {"w13": (E, 2 * IR, H), "w13_sf": (E, IR // 128, H // 128, 2), "w2": (E, H, IR),
              "w2_sf": (E, H // 128, IR // 128)}
    w.update({k: torch.stack(v).view(shapes[k]) for k, v in banks.items()})
    return w
