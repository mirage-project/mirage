"""The DeepSeek-V4.1-Flash MoE layer's hooks for the loop (python/mirage/mpk/loop/part.py): demo.py's functions."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo     # noqa: E402
import weights  # noqa: E402
from mirage.mpk.static_megakernel import StaticMegakernel  # noqa: E402

TP = 4


def build(input_dir: str, gpus: int, tokens: int, options: dict):
    """input_dir: a checkpoint directory. gpus: 1 (rank 0's columns), or 4 (rank g on GPU g, the all-reduce in the layer)."""
    if tokens != demo.T:
        raise ValueError(f"the MoE layer takes {demo.T} tokens, asked for {tokens}")
    layer, read = int(options["layer"]), weights.checkpoint_reader(input_dir)
    devs = [torch.device("cuda", g) for g in range(gpus)]
    ws = [weights.layer_tensors(read, layer, TP, g, d) for g, d in enumerate(devs)]
    for w, rt in zip(ws, weights.routed_tensors(read, layer, TP, list(range(gpus)), devs)):
        w.update(rt)
    x = torch.randn(demo.T, demo.HID, generator=torch.Generator().manual_seed(1)).bfloat16()
    for w, d in zip(ws, devs):
        w["x"] = x.to(d).contiguous()
        w["y"] = torch.zeros(demo.T, demo.HID, dtype=torch.bfloat16, device=d)
    mpk = StaticMegakernel(num_gpus=gpus)
    demo.build_graph(mpk, ws[0])
    return ws, mpk


def check(h, ws, input_dir: str, out_dir: str) -> bool:
    """One launch: y has no unwritten value (0xFF bytes read as NaN), and it is the same bits on every GPU."""
    h.launch(timeout_s=60)
    y = ws[0]["y"].cpu()
    return bool(torch.isfinite(y.float()).all()) and all(torch.equal(w["y"].cpu().view(torch.int16), y.view(torch.int16)) for w in ws)


def watch(ws) -> dict:
    return {"y": ws[0]["y"]}
