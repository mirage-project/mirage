"""The decode step's glue (step_glue.cu) built with nvcc as the layers are, called through ctypes."""
import ctypes
import os
import subprocess

import torch

from mirage.mpk.static_schedule import compile_command

_HERE = os.path.dirname(os.path.abspath(__file__))


class StepGlue:
    def __init__(self, logits_row, embed, x0_row, kv_indptr, last_page_len, prompt, gen, page, out_dir):
        assert kv_indptr.numel() == 2 and last_page_len.numel() == 1, "one request"
        assert logits_row.dtype == torch.float32 and embed.dtype == x0_row.dtype == torch.bfloat16
        assert prompt.dtype == gen.dtype == torch.long and kv_indptr.dtype == last_page_len.dtype == torch.int32
        so = os.path.join(out_dir, "step_glue.so")
        subprocess.check_call(compile_command(os.path.join(_HERE, "step_glue.cu"), so))
        self.fn = ctypes.CDLL(so).step_glue_launch
        self.fn.restype = ctypes.c_int
        p, i = ctypes.c_void_p, ctypes.c_int
        self.fn.argtypes = [p, i, p, p, p, i, i, i, p, p, p, i, p, p, i, p]
        self.key = torch.full((1,), -2**63, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.keep = (logits_row, embed, x0_row, kv_indptr, last_page_len, prompt, gen)
        self.args = (logits_row.data_ptr(), logits_row.numel(), self.key.data_ptr(), self.count.data_ptr(),
                     prompt.data_ptr(), prompt.numel())
        self.rest = (gen.data_ptr(), embed.data_ptr(), x0_row.data_ptr(), embed.shape[1], kv_indptr.data_ptr(),
                     last_page_len.data_ptr(), page)

    def __call__(self, pos: int, has_prev: bool):
        """Before step pos (has_prev: the last step's logits are there); pos == P + NEW - 1: the last token only."""
        err = self.fn(*self.args, pos, int(has_prev), *self.rest, torch.cuda.current_stream().cuda_stream)
        assert err == 0, f"step_glue launch: CUDA error {err}"
