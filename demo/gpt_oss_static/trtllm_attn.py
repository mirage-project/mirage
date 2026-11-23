# Copyright 2026 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Past 4K positions (gpt_oss_static.full_attention): the full-attention layers' decode attention as FlashInfer's
trtllm-gen kernel (SGLang's B200 GPT-OSS attention: flashinfer.decode.trtllm_batch_decode_with_kv_cache), between the layer's two static kernels. The
first leaves q (bias, RoPE) and seq_lens and appends the token's K / V (gpt_oss_rope_append); this reads the caches in
place ([pages, PAGE, KVH, D] bf16 = FlashInfer's NHD: a strided HND view, no copy) through a constant block table
(request r's page j is r MP + j) and writes the attention output the second kernel's o_proj reads. On Hopper FlashInfer
runs its xqa kernel for the same call (the local tests)."""
import torch


class TrtllmAttention:
    def __init__(self, R: int, QH: int, KVH: int, D: int, page: int, max_seq: int, workspace_mb: int = 128):
        self.R, self.QH, self.KVH, self.D, self.max_seq = R, QH, KVH, D, max_seq
        MP = max_seq // page
        dev = "cuda"
        self.block_table = (torch.arange(R, device=dev)[:, None] * MP + torch.arange(MP, device=dev)).to(torch.int32)
        # FlashInfer's multi-CTA KV counters live in the workspace (zeroed once; the kernel resets them)
        self.workspace = torch.zeros(workspace_mb << 20, dtype=torch.uint8, device=dev)

    @staticmethod
    def sinks_of(sinks: torch.Tensor) -> torch.Tensor:
        """A layer's sinks [KVH, Q / KVH] bf16 as FlashInfer's [Q] fp32 (query head h QPK + i)."""
        return sinks.float().reshape(-1).contiguous()

    def launch(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, seq_lens: torch.Tensor,
               out: torch.Tensor, sinks: torch.Tensor, backend: str = "auto"):
        """q [T, Q x D] bf16 (rows past R unused), the caches [pages, page, KVH, D] bf16, seq_lens [R] int32 (with the
        new token), out [T, Q x D] bf16, sinks [Q] fp32. max_seq_len: the caches' (the kernel and its KV splits are
        chosen for it at capture, as SGLang's max context; each request runs to its seq_lens)."""
        import flashinfer
        R, QH, D = self.R, self.QH, self.D
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q[:R].view(R, QH, D), (k_cache, v_cache), self.workspace, self.block_table, seq_lens, self.max_seq,
            bmm1_scale=D ** -0.5, bmm2_scale=1.0, window_left=-1, out=out[:R].view(R, QH, D), sinks=sinks,
            kv_layout="NHD", enable_pdl=True, backend=backend)
