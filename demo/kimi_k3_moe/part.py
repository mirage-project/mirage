"""The K3 MoE layer's hooks for the loop (python/mirage/mpk/loop/part.py): demo.py's functions."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo   # noqa: E402


def build(input_dir: str, gpus: int, tokens: int, options: dict):
    if tokens != demo.T:
        raise ValueError(f"the K3 MoE layer's input files hold {demo.T} tokens, asked for {tokens}")
    return demo.build(input_dir, gpus, bool(options.get("rmsnorm_recompute", False)))


def check(h, ws, input_dir: str, out_dir: str) -> bool:
    return demo.check_launch(h, ws, input_dir, out_dir)


def watch(ws) -> dict:
    return {"y": ws[0]["moe_out"], "R": ws[0]["routed_sum_partial"], "S": ws[0]["shared_down_partial"]}
