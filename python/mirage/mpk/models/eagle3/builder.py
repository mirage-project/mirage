"""Eagle3 draft model graph builder for MPK.

Builds the Eagle3 speculative decoding draft loop on top of an existing target
model graph. The draft is a single Llama-style transformer block whose
attention QKV input takes ``2 * hidden_size`` (concat of token embedding and
projected hidden), with its own ``draft_vocab_size``-sized lm_head and a
``d2t`` table mapping draft ids back to target vocab.

See [/home/letianr/.claude/plans/mpk-eagle3-qwen3-sglang-claude-skills-s-merry-rocket.md]
for the design rationale and phase breakdown.

Currently targets the SM100 path (the only path where MTP + verify kernels are
wired). World-size > 1 not supported in v1.
"""

import json
import os
from typing import List, Optional

import torch
from safetensors.torch import load_file

from ..utils import (
    grid_for_rmsnorm_linear_layer,
    inplace_shuffle_tensors,
    shuffle_tensors,
)
from ...persistent_kernel import PersistentKernel
from ....core import bfloat16, int64


def _resolve_draft_path(model_path_or_repo: str) -> str:
    """Resolve a HF repo id or local path to a directory containing weights."""
    if os.path.isdir(model_path_or_repo):
        return model_path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(model_path_or_repo)


def _load_draft_state_dict(path: str) -> dict:
    """Load all *.safetensors shards from `path` into a single state_dict."""
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files under {path}")
    state_dict = {}
    for f in files:
        state_dict.update(load_file(os.path.join(path, f), device="cuda"))
    return state_dict


class Eagle3Builder:
    """Build the Eagle3 draft + verify portion of a target model's task graph.

    Usage from a target builder / demo:

        eagle3 = Eagle3Builder(
            mpk=mpk, draft_state_dict=sd, draft_config=cfg,
            target_hidden_size=2048,
            target_w_embed=shared_embed_dtensor,
            cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
            num_draft_steps=4,
        )
        eagle3.build_draft_loop(
            aux_h0=aux_h0_dtensor, aux_h1=..., aux_h2=...,
            target_argmax_token=argmax_out_dtensor,
        )
        # eagle3.all_draft_ids holds the per-step draft token IDs for verify

    After build_draft_loop the caller wires `eagle3.all_draft_ids` and
    `mpk.meta_tensors['output_tokens']` (target argmax) into the MTP verify
    pipeline (mtp_prepare_verify_layer → mtp_verify_strict_layer →
    mtp_accept_commit_layer).
    """

    def __init__(
        self,
        mpk: PersistentKernel,
        draft_state_dict: dict,
        draft_config: dict,
        target_hidden_size: int,
        target_w_embed,                    # shared target embedding DTensor
        cos_pos_embed,                     # DTensor (max_pos, head_dim)
        sin_pos_embed,                     # DTensor (max_pos, head_dim)
        num_draft_steps: int = 4,
        use_aux_norm: bool = False,
    ):
        assert mpk.world_size == 1, "Eagle3 builder v1 only supports world_size=1"
        if use_aux_norm:
            raise NotImplementedError("use_aux_norm=True not yet wired (no aux norms in default ckpt)")

        self.mpk = mpk
        self.sd = draft_state_dict
        self.cfg = draft_config
        self.mbt = mpk.max_num_batched_tokens
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size

        # Draft architecture parameters (from HF config.json)
        self.hidden_size = int(draft_config["hidden_size"])
        assert self.hidden_size == target_hidden_size, (
            f"Eagle3 draft hidden_size ({self.hidden_size}) must match target "
            f"({target_hidden_size})")
        self.intermediate_size = int(draft_config["intermediate_size"])
        self.num_q_heads = int(draft_config["num_attention_heads"])
        self.num_kv_heads = int(draft_config["num_key_value_heads"])
        self.head_dim = int(draft_config["head_dim"])
        self.draft_vocab_size = int(draft_config["draft_vocab_size"])
        # Fused QKV output dim (Qwen3-style packing: Q + K + V along dim 0).
        self.fused_outdim_qkv = (
            self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        # Fused gate+up dim
        self.fused_outdim_gateup = 2 * self.intermediate_size

        # Argmax over hot vocab: 32000 is divisible by num_workers (96) only
        # when num_workers divides 32000. 32000 = 2^7 * 5^3, divisible by 64.
        # Use the same num_workers grid the dense Qwen3 builder uses; the
        # partial kernel handles uneven splits internally.
        self.num_draft_steps = num_draft_steps
        self.target_w_embed = target_w_embed
        self.cos_pos_embed = cos_pos_embed
        self.sin_pos_embed = sin_pos_embed

        # Caches so we don't double-attach across draft steps.
        self._attach_cache = {}
        self._kept_tensors = []  # keep PyTorch tensors alive (mpk stores raw ptrs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _attach(self, tensor: torch.Tensor, name: str):
        if name in self._attach_cache:
            return self._attach_cache[name]
        # MPK stores raw GPU pointers; we must keep the torch tensor alive
        self._kept_tensors.append(tensor)
        dt = self.mpk.attach_input(torch_tensor=tensor, name=name)
        self._attach_cache[name] = dt
        return dt

    def _new(self, dims, dtype, name, io_category="cuda_tensor"):
        if name in self._attach_cache:
            return self._attach_cache[name]
        dt = self.mpk.new_tensor(dims=dims, dtype=dtype, name=name,
                                  io_category=io_category)
        self._attach_cache[name] = dt
        return dt

    # ------------------------------------------------------------------
    # Weight preparation (one-time)
    # ------------------------------------------------------------------
    def _prepare_weights(self):
        """Shuffle Q/K/V and gate/up into MPK's fused layouts; build d2t."""
        sd = self.sd

        # 1. Fused QKV. Eagle3 q/k/v weights take input dim = 2 * hidden_size.
        q = sd["midlayer.self_attn.q_proj.weight"]   # (num_q*head_dim, 2H)
        k = sd["midlayer.self_attn.k_proj.weight"]   # (num_kv*head_dim, 2H)
        v = sd["midlayer.self_attn.v_proj.weight"]   # (num_kv*head_dim, 2H)
        assert q.shape == (self.num_q_heads * self.head_dim, 2 * self.hidden_size)
        assert k.shape == (self.num_kv_heads * self.head_dim, 2 * self.hidden_size)
        assert v.shape == (self.num_kv_heads * self.head_dim, 2 * self.hidden_size)
        self._qkv_w = shuffle_tensors([q, k, v], self.num_kv_heads, 0)
        self._kept_tensors.append(self._qkv_w)

        # 2. Fused gate+up.
        gate = sd["midlayer.mlp.gate_proj.weight"]   # (interm, H)
        up = sd["midlayer.mlp.up_proj.weight"]       # (interm, H)
        rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(
            gate.shape[0] + up.shape[0])
        self._gateup_w = shuffle_tensors([gate, up], rmsnorm_num_tasks // 2, 0)
        self._kept_tensors.append(self._gateup_w)

        # 3. d2t table. State dict has it as int64 (sglang loads it as int64).
        # Convention: target_id = hot_id + d2t[hot_id].
        d2t = sd["d2t"]
        if d2t.dtype != torch.int64:
            d2t = d2t.to(torch.int64)
        self._d2t = d2t.contiguous()
        self._kept_tensors.append(self._d2t)

        # 4. fc / eh_proj (3H → H), only used at step 0.
        fc = sd["fc.weight"]
        assert fc.shape == (self.hidden_size, 3 * self.hidden_size)
        self._fc_w = fc.contiguous()
        self._kept_tensors.append(self._fc_w)

        # 5. lm_head. Eagle3 uses hot vocab head; size 32000.
        # Pad rows to make per-task output 16B-aligned so cuTensorMapEncodeTiled
        # accepts the OUTPUT TMA descriptor. 32000 → 32256 (= 96 × 336 elements);
        # per-task = 336 bf16 = 672 bytes ≡ 0 mod 16. Pad rows are zeros (same
        # strategy as target lm_head padding 151936 → 153600). Argmax over real
        # logits dominates zeros in practice; hot_id stays in [0, 32000).
        lm = sd["lm_head.weight"]
        assert lm.shape == (self.draft_vocab_size, self.hidden_size)
        self._padded_draft_vocab = 32256
        assert self._padded_draft_vocab % 96 == 0, (
            f"padded_draft_vocab {self._padded_draft_vocab} must be 96-aligned")
        pad_rows = self._padded_draft_vocab - self.draft_vocab_size
        if pad_rows > 0:
            self._lm_head_w = torch.cat(
                [lm,
                 torch.zeros((pad_rows, self.hidden_size),
                              dtype=lm.dtype, device=lm.device)],
                dim=0,
            ).contiguous()
        else:
            self._lm_head_w = lm.contiguous()
        assert self._lm_head_w.shape == (self._padded_draft_vocab, self.hidden_size)
        self._kept_tensors.append(self._lm_head_w)

        # 6. KV cache: own buffer, single layer.
        self._k_cache_buf = torch.zeros(
            (self.max_num_pages, self.page_size, self.num_kv_heads, self.head_dim),
            dtype=torch.bfloat16, device="cuda")
        self._v_cache_buf = torch.zeros_like(self._k_cache_buf)
        self._kept_tensors.append(self._k_cache_buf)
        self._kept_tensors.append(self._v_cache_buf)

        # 7. Dummy q_norm/k_norm (head_dim,) for paged_attention_layer slot.
        # qk_norm is disabled via enable_qk_norm=False but the slot needs a
        # valid DTensor.
        self._dummy_norm_buf = torch.zeros(
            (self.head_dim,), dtype=torch.bfloat16, device="cuda")
        self._kept_tensors.append(self._dummy_norm_buf)

    def _attach_weights(self):
        """Attach the prepared weights to the mpk graph."""
        # Eagle3 draft uses TWO RMSNorms before QKV:
        #   input_layernorm over the H-dim EMBED only
        #   hidden_norm     over the H-dim HIDDEN only
        # then concat → 2H → QKV. (sglang llama_eagle3.LlamaDecoderLayer.forward)
        self.w_input_ln = self._attach(
            self.sd["midlayer.input_layernorm.weight"].contiguous(),
            "eagle3_input_layernorm")
        self.w_hidden_norm = self._attach(
            self.sd["midlayer.hidden_norm.weight"].contiguous(),
            "eagle3_hidden_norm")
        self.w_post_ln = self._attach(
            self.sd["midlayer.post_attention_layernorm.weight"].contiguous(),
            "eagle3_post_attention_layernorm")
        self.w_qkv = self._attach(self._qkv_w, "eagle3_qkv_proj")
        self.w_o = self._attach(
            self.sd["midlayer.self_attn.o_proj.weight"].contiguous(),
            "eagle3_o_proj")
        self.w_gateup = self._attach(self._gateup_w, "eagle3_gateup_proj")
        self.w_down = self._attach(
            self.sd["midlayer.mlp.down_proj.weight"].contiguous(),
            "eagle3_down_proj")
        self.w_fc = self._attach(self._fc_w, "eagle3_fc")
        self.w_final_norm = self._attach(
            self.sd["norm.weight"].contiguous(), "eagle3_norm")
        self.w_lm_head = self._attach(self._lm_head_w, "eagle3_lm_head")
        self.d2t = self._attach(self._d2t, "eagle3_d2t")
        self.k_cache = self._attach(self._k_cache_buf, "eagle3_k_cache")
        self.v_cache = self._attach(self._v_cache_buf, "eagle3_v_cache")
        self.dummy_norm = self._attach(
            self._dummy_norm_buf, "eagle3_dummy_qk_norm")

    def _allocate_intermediates(self):
        """Allocate all per-iteration intermediate tensors (reused across steps)."""
        H = self.hidden_size
        mbt = self.mbt
        I = self.intermediate_size

        self.embed_out = self._new(
            (mbt, H), bfloat16, "eagle3_embed_out")
        self.embed_normed = self._new(
            (mbt, H), bfloat16, "eagle3_embed_normed")  # after input_layernorm
        self.hidden_normed = self._new(
            (mbt, H), bfloat16, "eagle3_hidden_normed")  # after hidden_norm
        self.aux_concat_out = self._new(
            (mbt, 3 * H), bfloat16, "eagle3_aux_concat_out")
        self.hidden_in = self._new(
            (mbt, H), bfloat16, "eagle3_hidden_in")    # eh_proj output (step 0)
        # Rolling hidden (also draft block's output for next step). Allocated
        # named so step>0 can read it as input.
        self.draft_hidden = self._new(
            (mbt, H), bfloat16, "eagle3_draft_hidden")
        self.qkv_in_2H = self._new(
            (mbt, 2 * H), bfloat16, "eagle3_qkv_in_2H")
        self.ln_out = self._new(
            (mbt, 2 * H), bfloat16, "eagle3_ln_out")
        self.attn_in = self._new(
            (mbt, self.fused_outdim_qkv), bfloat16, "eagle3_attn_in")
        self.attn_out = self._new(
            (mbt, self.num_q_heads * self.head_dim), bfloat16, "eagle3_attn_out")
        self.attn_proj_out = self._new(
            (mbt, H), bfloat16, "eagle3_attn_proj_out")
        self.post_ln_out = self._new(
            (mbt, H), bfloat16, "eagle3_post_ln_out")
        self.mlp_mid = self._new(
            (mbt, self.fused_outdim_gateup), bfloat16, "eagle3_mlp_mid")
        self.silu_mul_out = self._new(
            (mbt, I), bfloat16, "eagle3_silu_mul_out")
        self.norm_out = self._new(
            (mbt, H), bfloat16, "eagle3_norm_out")
        # Use padded draft vocab to satisfy 16B TMA alignment of OUTPUT.
        self.logits_hot = self._new(
            (mbt, self._padded_draft_vocab), bfloat16, "eagle3_logits_hot")
        self.argmax_part_value = self._new(
            (mbt, self.mpk.num_workers), bfloat16, "eagle3_argmax_part_value")
        self.argmax_part_index = self._new(
            (mbt, self.mpk.num_workers), int64, "eagle3_argmax_part_index")
        self.hot_token = self._new(
            (mbt, 1), int64, "eagle3_hot_token")
        self.target_token = self._new(
            (mbt, 1), int64, "eagle3_target_token")
        # Collection buffer for verify
        self.all_draft_ids = self._new(
            (mbt, self.num_draft_steps), int64, "eagle3_all_draft_ids")

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def build_draft_loop(
        self,
        aux_h0,      # DTensor, target hidden at capture layer 0
        aux_h1,
        aux_h2,
        target_argmax_token,  # DTensor (mbt, 1) int64 — main's argmax
    ):
        """Register the Eagle3 draft loop tasks on self.mpk's graph.

        Inputs are kernel-level DTensors produced earlier in the target graph.
        After this returns, self.all_draft_ids contains the K draft tokens per
        request (each in target-vocab space, after d2t remap), ready for the
        MTP verify pipeline.
        """
        self._prepare_weights()
        self._attach_weights()
        self._allocate_intermediates()

        K = self.num_draft_steps
        mbt = self.mbt
        H = self.hidden_size

        # The block_dim convention: 256 for SM>=90 / Blackwell, 128 for Ampere.
        # ALL kernels on Blackwell require 256 threads (NUM_THREADS hardcoded
        # in SM100 task templates); passing (128,1,1) causes illegal-instruction
        # crashes at runtime.
        bd_compute = (256, 1, 1) if self.mpk.target_cc >= 90 else (128, 1, 1)
        bd_small = bd_compute

        for step in range(K):
            # 1. Pick the draft input token for this step
            #    step 0 → target's argmax token (the bonus token from main fwd)
            #    step>0 → previous draft's d2t-remapped (target-vocab) token
            draft_in_token = target_argmax_token if step == 0 else self.target_token

            # 2. Embed lookup (shared with target). Embed kernel reads from
            #    runtime token buffer via input_source=1 (input_token).
            self.mpk.embed_layer(
                input=draft_in_token, weight=self.target_w_embed,
                output=self.embed_out,
                grid_dim=(1, 1, 1), block_dim=bd_small,
                input_source=1,
            )

            # 3. Compute hidden_in (the H-dim hidden state).
            if step == 0:
                # step 0: concat(h0, h1, h2) → 3H → eh_proj → H
                self.mpk.eagle3_aux_concat_layer(
                    h0=aux_h0, h1=aux_h1, h2=aux_h2,
                    output=self.aux_concat_out,
                    grid_dim=(1, 1, 1), block_dim=bd_compute,
                )
                # fc: 3H → H
                self.mpk.linear_layer(
                    input=self.aux_concat_out, weight=self.w_fc,
                    output=self.hidden_in,
                    grid_dim=(grid_for_rmsnorm_linear_layer(self.w_fc.dim(0)), 1, 1),
                    block_dim=bd_small,
                )
                step_hidden = self.hidden_in
            else:
                # step>0: reuse the draft block's output hidden from previous iter.
                step_hidden = self.draft_hidden

            # 4. Per-branch RMSNorm BEFORE concat (sglang Eagle3 convention):
            #   embed   → input_layernorm → embed_normed   (H)
            #   hidden  → hidden_norm     → hidden_normed  (H)
            # Then concat the two normed tensors into the 2H QKV input.
            self.mpk.rmsnorm_layer(
                input=self.embed_out, weight=self.w_input_ln,
                output=self.embed_normed,
                grid_dim=(mbt, 1, 1), block_dim=bd_small,
            )
            self.mpk.rmsnorm_layer(
                input=step_hidden, weight=self.w_hidden_norm,
                output=self.hidden_normed,
                grid_dim=(mbt, 1, 1), block_dim=bd_small,
            )
            self.mpk.eagle3_input_concat_layer(
                embed=self.embed_normed, hidden=self.hidden_normed,
                output=self.qkv_in_2H,
                grid_dim=(1, 1, 1), block_dim=bd_compute,
            )

            # 5. fused QKV projection: (fused_outdim_qkv, 2H)
            self.mpk.linear_layer(
                input=self.qkv_in_2H, weight=self.w_qkv, output=self.attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(self.w_qkv.dim(0)), 1, 1),
                block_dim=bd_small,
            )

            # 7. Paged attention (no q_norm/k_norm: Eagle3 Llama-style block).
            self.mpk.paged_attention_layer(
                input=self.attn_in,
                k_cache=self.k_cache, v_cache=self.v_cache,
                q_norm=self.dummy_norm, k_norm=self.dummy_norm,
                cos_pos_embed=self.cos_pos_embed,
                sin_pos_embed=self.sin_pos_embed,
                output=self.attn_out,
                grid_dim=(self.mpk.max_num_batched_requests,
                          self.num_kv_heads, 1),
                block_dim=bd_small,
                enable_qk_norm=False,
            )

            # 8. o_proj + residual. Residual is the H-dim step_hidden (not the
            #    2H pre-attn). This mirrors sglang's llama_eagle3 forward.
            self.mpk.linear_with_residual_layer(
                input=self.attn_out, weight=self.w_o,
                residual=step_hidden, output=self.attn_proj_out,
                grid_dim=(H // 64, 1, 1), block_dim=bd_small,
            )

            # 9. post_attention_layernorm
            self.mpk.rmsnorm_layer(
                input=self.attn_proj_out, weight=self.w_post_ln,
                output=self.post_ln_out,
                grid_dim=(mbt, 1, 1), block_dim=bd_small,
            )

            # 10. MLP: gate_up linear → silu_mul → down + residual
            gateup_num_tasks = grid_for_rmsnorm_linear_layer(self.w_gateup.dim(0))
            self.mpk.linear_layer(
                input=self.post_ln_out, weight=self.w_gateup,
                output=self.mlp_mid,
                grid_dim=(gateup_num_tasks, 1, 1), block_dim=bd_small,
            )
            self.mpk.silu_mul_layer(
                input=self.mlp_mid, output=self.silu_mul_out,
                grid_dim=(gateup_num_tasks // 2, 1, 1), block_dim=bd_small,
            )
            self.mpk.linear_with_residual_layer(
                input=self.silu_mul_out, weight=self.w_down,
                residual=self.attn_proj_out, output=self.draft_hidden,
                grid_dim=(H // 64, 1, 1), block_dim=bd_small,
            )

            # 11. final norm + lm_head (hot vocab)
            self.mpk.rmsnorm_layer(
                input=self.draft_hidden, weight=self.w_final_norm,
                output=self.norm_out,
                grid_dim=(mbt, 1, 1), block_dim=bd_small,
            )
            self.mpk.linear_layer(
                input=self.norm_out, weight=self.w_lm_head,
                output=self.logits_hot,
                grid_dim=(grid_for_rmsnorm_linear_layer(self.w_lm_head.dim(0)), 1, 1),
                block_dim=bd_small,
            )

            # 12. argmax (hot vocab → hot_token id)
            self.mpk.argmax_partial_layer(
                input=self.logits_hot,
                output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=(self.mpk.num_workers, 1, 1), block_dim=bd_small,
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=self.hot_token,
                grid_dim=(1, 1, 1), block_dim=bd_small,
            )

            # 13. d2t remap: hot_id → target vocab id (guarded against padded
            # argmax: hot_id ≥ draft_vocab_size → sentinel 0)
            self.mpk.eagle3_d2t_remap_layer(
                hot_token=self.hot_token, d2t_table=self.d2t,
                target_token=self.target_token,
                grid_dim=(1, 1, 1), block_dim=bd_small,
                draft_vocab_real=self.draft_vocab_size,
            )

            # 14. scatter target_token into all_draft_ids[:, step]
            self.mpk.mtp_token_scatter_layer(
                src=self.target_token, dst=self.all_draft_ids,
                grid_dim=(1, 1, 1), block_dim=bd_small,
                batch_size=mbt, num_slots=K, slot_idx=step,
            )

        return self.all_draft_ids


def load_eagle3_draft(draft_model_path_or_repo: str):
    """Resolve checkpoint path, load state_dict and config.

    Returns (state_dict, config_dict).
    """
    path = _resolve_draft_path(draft_model_path_or_repo)
    with open(os.path.join(path, "config.json")) as fp:
        config = json.load(fp)
    state_dict = _load_draft_state_dict(path)
    return state_dict, config
