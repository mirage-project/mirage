"""The DeepSeek-V4.1-Flash attention layer's hooks for the loop (python/mirage/mpk/loop/part.py): demo.py's functions."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo     # noqa: E402
import weights  # noqa: E402
from mirage.mpk.static_megakernel import StaticMegakernel  # noqa: E402

TP = 4


def build(input_dir: str, gpus: int, tokens: int, options: dict):
    """input_dir: a checkpoint directory. gpus: 1 (shard 0), or 4 (shard g on GPU g, the all-reduce in the layer)."""
    if tokens != demo.T:
        raise ValueError(f"the attention layer takes {demo.T} tokens, asked for {tokens}")
    layer, pos = int(options["layer"]), int(options["pos"])
    cfg, read = weights.model_config(input_dir), weights.checkpoint_reader(input_dir)
    variant, ratio = weights.variant_of(cfg, layer), cfg["compress_ratios"][layer]
    csrc = layer == cfg["candidate_source_layer"]
    rows, cap = (pos + 2 + 255) // 256 * 256, max(((pos + 1) // max(ratio, 1) + 255) // 256 * 256, demo.NC)
    ws = []
    for g in range(gpus):
        d = torch.device("cuda", g)
        ws.append({**weights.layer_tensors(read, cfg, layer, TP, g, rows, d), **demo.state_tensors(variant, ratio, csrc, cap, d)})
        demo.random_inputs(ws[g], variant, ratio, pos)
    mpk = StaticMegakernel(num_gpus=gpus)
    demo.build_graph(mpk, ws[0], variant, ratio, csrc, 64 // TP)
    return ws, mpk


def check(h, ws, input_dir: str, out_dir: str) -> bool:
    """One launch: y has no unwritten value (0xFF bytes read as NaN), and it is the same bits on every GPU."""
    h.launch(timeout_s=60)
    y = ws[0]["y"].cpu()
    return bool(torch.isfinite(y.float()).all()) and all(torch.equal(w["y"].cpu().view(torch.int16), y.view(torch.int16)) for w in ws)


def watch(ws) -> dict:
    return {"y": ws[0]["y"]}
