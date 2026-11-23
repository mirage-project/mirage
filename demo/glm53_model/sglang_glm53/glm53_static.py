"""GLM-5.3 in SGLang with its decode on the mirage static megakernel (demo/glm53_model/decoder.py).

SGLang keeps the scheduler, the tokenizer, the sampling, the prefill and the KV pool. A decode step runs this model's
forward: the embedding and layer 0's input norm (SGLang's), then per layer the attention kernel and the FFN kernel, then
SGLang's logits. Each kernel ends its sublayer itself: the all-reduce over the TP group (symmetric memory, multicast), the
residual add and the next RMSNorm (the last kernel: the final norm). MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=0: the kernels end
with this rank's partial sum and SGLang's TP all-reduce and fused add + RMSNorm run between them. CUDA graphs capture the
step per batch size as SGLang's own decode. The prefill runs SGLang's own layers. The kernels have their own copy of this
rank's weights (read from the checkpoint before SGLang sizes its KV pool; the MoE layers about 1.2 GB each), so SGLang's
tensors stay as SGLang loaded them.

The class has the name of SGLang's (GlmMoeDsaForCausalLM), so SGLang picks it for the checkpoint's architecture:

  export PATH=/usr/local/cuda/bin:$PATH        # nvcc: the layer kernels are built at startup (cached in ~/.cache/mirage)
  export PYTHONPATH=<repo>:<repo>/demo/glm53_model SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_glm53
  python -m sglang.launch_server --model-path zai-org/GLM-5.3 --tp 8 --kv-cache-dtype bfloat16 \\
      --moe-runner-backend flashinfer_trtllm --max-running-requests 8 --cuda-graph-max-bs-decode 8 \\
      --context-length 16384 --disable-radix-cache

Requirements: TP 8 on GPUs of 148 SMs (the plans) with NVLink multicast (NVLS: the kernels' all-reduce; not with
MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=0), a bf16 latent cache in pages of 64, a context of at most 16384 (the plans), memory
for a second copy of the MoE weights (about 90 GB per GPU at TP 8: for example B300). The startup checks the TP size, the
context length, the NVLink multicast and the cache layout.
The kernels decode steps of at most T requests (T = STATIC_TOKENS, default 8: the rows of a kernel). A larger decode step
runs SGLang's own layers, as the prefill does (as demo/glm53_flash's hooks do for a batch size with no static layer): the
kernels and SGLang's layers read and write the same caches, so they take turns on them. Set --max-running-requests and
--cuda-graph-max-bs-decode to T to decode every step on the kernels.
Every rank prints "mirage static megakernel decode: on" once the kernels are ready. SGLang skips a model package that
fails to import (a warning) and then serves its own model: check that line.
"""
import os

import torch
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.parallel_state import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                                                   get_tp_group)
from sglang.srt.model_executor import model_runner as _model_runner
from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLM as _SGLangGlmMoeDsa

from demo.glm53_dsa import weights as dsa_weights
from demo.glm53_model.decoder import MAXCTX, TP, T, StaticDecoder


def _snapshot(model_path: str, revision: str = None) -> str:
    """The local snapshot directory of the checkpoint SGLang loaded."""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download
    patterns = ["*.json", "*.safetensors"]
    try:
        return snapshot_download(model_path, revision=revision, allow_patterns=patterns, local_files_only=True)
    except Exception:
        return snapshot_download(model_path, revision=revision, allow_patterns=patterns)


class GlmMoeDsaForCausalLM(_SGLangGlmMoeDsa):
    """SGLang's GLM-5.3 model with the static-megakernel decode."""

    def precompile_kernels_after_loading(self):
        """After SGLang's load and its post-load processing, before it sizes the KV pool: load the kernels' own tensors, so
        the pool leaves room for them."""
        parent = getattr(super(), "precompile_kernels_after_loading", None)
        if parent is not None:
            parent()
        from sglang.srt.runtime_context import get_server_args
        args = get_server_args()
        if get_tensor_model_parallel_world_size() != TP:
            raise ValueError(f"mirage static megakernel: the kernels are built for TP {TP}, the server runs TP "
                             f"{get_tensor_model_parallel_world_size()}")
        snapshot = _snapshot(args.model_path, getattr(args, "revision", None))
        cfg = dsa_weights.check_config(snapshot)
        ctx = getattr(args, "context_length", None) or cfg["max_position_embeddings"]
        if ctx > MAXCTX:
            raise ValueError(f"mirage static megakernel: context {ctx} > {MAXCTX} (the kernels' plans); pass "
                             f"--context-length {MAXCTX} or less")
        dev = torch.device("cuda", torch.cuda.current_device())
        # the device (NCCL) group of the TP group: the decoder allocates the symmetric memory of the kernels' exchange
        # regions over it (the kernels' all-reduce). Not used with MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=0
        self.mirage = StaticDecoder(cfg, dsa_weights.checkpoint_reader(snapshot), get_tensor_model_parallel_rank(), dev,
                                    group=get_tp_group().device_group)
        self.mirage.load_weights()

    def mirage_setup(self, runner):
        """Build the kernels and load the instances on the pool's tensors. Call it after the KV pool and before CUDA graph
        capture."""
        m = getattr(self, "mirage", None)
        if m is None or m.ready:
            return
        if runner.model_config.context_len > m.maxctx:
            raise ValueError(f"mirage static megakernel: context {runner.model_config.context_len} > {m.maxctx}")
        group = get_tp_group().cpu_group
        m.setup(runner.token_to_kv_pool, runner.req_to_token_pool.req_to_token,
                barrier=lambda: torch.distributed.barrier(group=group))
        torch.cuda.synchronize()
        # print, not the logger: the line must not depend on SGLang's log level (sgl_chain.sh counts it)
        print(f"mirage static megakernel decode: on (rank {m.rank}, {len(m.instances)} layer kernels, context {m.maxctx})",
              flush=True)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        m = getattr(self, "mirage", None)
        if m is not None and m.ready and input_embeds is None and forward_batch.forward_mode.is_decode() and \
                input_ids.shape[0] <= T:   # a larger decode step: SGLang's layers (module docstring)
            h = m.decode(self, input_ids, forward_batch, tensor_model_parallel_all_reduce)
            return self.logits_processor(input_ids, h, self.lm_head, forward_batch)
        return super().forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)


_init_cuda_graphs = _model_runner.ModelRunner.init_cuda_graphs


def _init_cuda_graphs_after_mirage(self, *args, **kwargs):
    setup = getattr(self.model, "mirage_setup", None)
    if setup is not None:
        setup(self)
    return _init_cuda_graphs(self, *args, **kwargs)


_model_runner.ModelRunner.init_cuda_graphs = _init_cuda_graphs_after_mirage

EntryClass = [GlmMoeDsaForCausalLM]
