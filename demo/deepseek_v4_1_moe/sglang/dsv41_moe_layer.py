"""The DeepSeek-V4.1-Flash MoE layer that dsv41_moe_hook.py puts into SGLang (no SGLang import here): one TP rank's layer
(demo/deepseek_v4_1_moe), T = 8 rows, built for one GPU: y is this rank's part of the sum, and SGLang's fused all-reduce and mHC
follow. A serving build on the caller's stream: PDL, no start barrier (STATIC_RESET_PARITY).

  weights  this rank's shard from the checkpoint (the demo's layer_tensors); the expert bank is its own copy
  plan     plans/moe_sms<SMs>.json when it exists, else the demo's hand plan for these SMs
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DSV41_STATIC_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "demo"))   # the demos as packages: both have a demo.py and a weights.py

from deepseek_v4_1_moe import demo as moe_demo          # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights    # noqa: E402

T, HID = moe_demo.T, moe_demo.HID
FREE_SMS = 16   # SMs left out of the launch: SGLang's mHC side stream runs on them
FLAGS = ["-DSTATIC_RESET_IN_KERNEL", "-DSTATIC_NO_START_BARRIER", "-DSTATIC_RESET_PARITY"]
_readers = {}


def reader(ckpt: str):
    return _readers.setdefault(ckpt, moe_weights.checkpoint_reader(ckpt))


class StaticMoE:
    def __init__(self, layer: int, ckpt: str, rank: int, tp: int, dev, out_root: str, pdl: bool):
        self.layer, self.pdl, self.steps = layer, pdl, 0
        read = reader(ckpt)
        w = moe_weights.layer_tensors(read, layer, tp, rank, dev)
        w.update(moe_weights.routed_tensors(read, layer, tp, [rank], dev)[0])
        w["x"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
        w["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count - FREE_SMS
        plan = os.path.join(HERE, "plans", f"moe_sms{sms}.json")
        # the node buffers are torch tensors the kernel writes to: they must stay referenced as long as the layer runs
        self.sk, self.bufs, self.info = moe_demo.build([w], os.path.join(out_root, f"rank{rank}", f"moe{layer}"),
                                                       plan=plan if os.path.exists(plan) else "hand",
                                                       extra_flags=FLAGS + [f"-DSTATIC_PDL_TRIGGER={int(pdl)}"], device=dev.index,
                                                       free_sms=FREE_SMS)
        self.w = w

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """One decode step of T rows: this rank's part of y [T, H] bf16 (the layer's own buffer). x is read where SGLang left it when
        it is contiguous bf16 [T, H] (a CUDA graph captures the address), else a copy."""
        ok = x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape == (T, HID)
        ptr = x.data_ptr() if ok else self.w["x"].data_ptr()
        if getattr(self, "x_ptr", None) != ptr:
            self.sk.rebind("x", ptr)
            self.x_ptr = ptr
        if not ok:
            self.w["x"].copy_(x)
        self.sk.launch_stream(torch.cuda.current_stream().cuda_stream, self.pdl)
        self.steps += 1
        return self.w["y"]
