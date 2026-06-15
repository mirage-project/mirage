"""DFlashBuilder — build the Kimi-K2.6 DFlash speculative-decoding DRAFT (L2->L6)
into a target model's MPK task graph.

Mirrors Eagle3Builder, but the DFlash draft is a SINGLE non-causal forward per
decode step (not Eagle3's K autoregressive steps), it owns a PAGED KV cache that
accumulates committed context K/V across decode steps, and it shares the target's
embedding + lm_head.

Pipeline per decode step (all in one megakernel):
  L2  ctx = hidden_norm(fc(target_hidden[S, K*H_t]))                  -> [S, H_d]
  L4  per draft layer: k = k_norm(k_proj(ctx)) + YaRN-RoPE; v = v_proj(ctx);
      dflash_kv_store writes (k,v) into that layer's paged cache at the committed
      slots (OVERWRITING the draft's temporary block-K/V there).
  L5  draft block [t0, MASK*(B-1)] runs a non-causal forward; each layer reads its
      paged cache (context) + this block's K/V; sliding_window=2048 on layers 0-4,
      full on the last layer.
  L6  final_hidden -> target lm_head -> argmax over the s=B-1 MASK slots -> draft
      tokens (chain = [t0, d1..d_{B-1}]).

Usage from a target builder / demo (once the K2.6 target main body is in MPK):

    dflash = DFlashBuilder(
        mpk=mpk,
        draft_state_dict=draft_sd, draft_config=draft_cfg,
        target_w_embed=shared_embed_2d,        # [vocab, H_t] torch.bf16 (target's)
        target_w_lm_head=shared_lm_head_2d,    # [vocab, H_t] torch.bf16 (target's)
        num_speculative_tokens=7,
        max_seq_len=4096,
    )
    # cos/sin cache built once from the rope config (YaRN, mscale=1.4159):
    #   self.cos_sin built in __init__; re-used by every step.

    # Per decode step, with the target's captured aux hidden + bonus token id(s):
    #   target_hidden : [S, K*H_t]  (committed context, this step's new commits)
    #   input_ids     : [B] = [t0, MASK*(B-1)]   (embedded via target_w_embed)
    draft_tokens = dflash.build_step(
        target_hidden_dt, input_ids_dt, ctx_len=S, slot_start=committed_len)
    # draft_tokens: DTensor [B,1] int64 (slot 0 = bonus echo; [1:B] = the s drafts)

The paged caches persist across steps so context accumulates; `slot_start` is the
committed-length offset where this step's new context K/V get written.

NOTE on cross-iter / runtime driving: this builder constructs ONE decode step's
draft graph. Two ways to drive it:
  (a) integrated (target in MPK): append after the target fwd in the same graph,
      like Eagle3Builder.build_draft_loop; the runtime's persistent paged cache +
      slot_mapping accumulate context across steps.
  (b) replay/per-step (target external): the demo holds the paged caches and calls
      build_step on a fresh PersistentKernel each step (ctx_len grows -> recompile);
      the caches (torch tensors held by this builder) persist and accumulate.
The validated reference for both is tests/runtime_python/blackwell/sm100_dflash/
{test_dflash_full_step_testmode.py (single step), test_dflash_paged_xiter.py
(cross-iter to EOS)}.
"""
from __future__ import annotations

import math

import torch

from ..utils import grid_for_rmsnorm_linear_layer
from ....core import bfloat16, int64


def _gpl(out_dim: int) -> int:
    """Grid for a linear/rmsnorm layer (96 if 96-aligned else 64)."""
    return 96 if out_dim % 96 == 0 else 64


class DFlashBuilder:
    def __init__(
        self,
        mpk,
        draft_state_dict: dict,
        draft_config: dict,
        target_w_embed: torch.Tensor,    # [vocab, H_t] shared target embedding
        target_w_lm_head: torch.Tensor,  # [vocab, H_t] shared target lm_head
        num_speculative_tokens: int = 7,
        max_seq_len: int = 4096,
        page_size: int = 8,
        mscale: float = 1.4159,          # vLLM/sglang native YaRN mscale for K2.6
        cos_sin_cache: torch.Tensor | None = None,  # [max_seq_len, head_dim], optional
        device: str = "cuda",
    ):
        self.mpk = mpk
        self.sd = draft_state_dict
        self.cfg = draft_config
        self.device = device
        self.dtype = torch.bfloat16

        self.hidden_size = int(draft_config["hidden_size"])            # H_d = 7168
        self.num_layers = int(draft_config["num_hidden_layers"])       # L = 6
        self.num_q_heads = int(draft_config["num_attention_heads"])    # 64
        self.num_kv_heads = int(draft_config["num_key_value_heads"])   # 8
        self.head_dim = int(draft_config.get("head_dim", 128))
        self.intermediate_size = int(draft_config["intermediate_size"])
        self.mask_token_id = int(draft_config["dflash_config"]["mask_token_id"])
        self.target_layer_ids = list(draft_config["dflash_config"]["target_layer_ids"])
        self.K = len(self.target_layer_ids)                            # captured layers = 6
        self.vocab_size = int(draft_config["vocab_size"])
        self.sliding_window = int(draft_config.get("sliding_window", 2048))
        # layer_types: "sliding_attention" -> windowed, else full. Default K2.6:
        # sliding on layers 0..L-2, full on the last layer.
        lt = draft_config.get("layer_types")
        if lt:
            self.layer_sliding = [self.sliding_window if t == "sliding_attention" else 0
                                  for t in lt[: self.num_layers]]
        else:
            self.layer_sliding = [self.sliding_window] * (self.num_layers - 1) + [0]

        self.B = num_speculative_tokens + 1            # block size (bonus + spec)
        self.num_speculative_tokens = num_speculative_tokens
        self.q_size = self.num_q_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = float(draft_config.get("rms_norm_eps", 1e-5))
        self.mscale = mscale
        self.page_size = page_size
        self.max_seq_len = max_seq_len
        self.max_num_pages = (max_seq_len + page_size - 1) // page_size

        self.target_w_embed = target_w_embed.contiguous()
        self.target_w_lm_head = target_w_lm_head.contiguous()
        assert target_w_lm_head.shape[1] == self.hidden_size

        self._kept = []          # keep torch tensors alive (MPK holds raw pointers)
        self._attach_cache = {}
        self.bd = (256, 1, 1) if mpk.target_cc >= 90 else (128, 1, 1)

        # Host-readable output buffers (the demo reads draft tokens off these; in an
        # integrated graph they're consumed on-device by verify instead).
        self.final_buf = torch.zeros((self.B, self.hidden_size), dtype=self.dtype, device=device)
        self.draft_tokens_buf = torch.zeros((self.B, 1), dtype=torch.int64, device=device)
        self._kept += [self.final_buf, self.draft_tokens_buf]

        # argmax over the full target vocab: split into arg_tasks chunks of
        # vocab/arg_tasks, which must be 8-aligned (vectorized) and divide vocab.
        self.arg_tasks = mpk.num_workers
        while self.vocab_size % self.arg_tasks != 0 or (self.vocab_size // self.arg_tasks) % 8 != 0:
            self.arg_tasks -= 1

        if cos_sin_cache is None:
            cos_sin_cache = self.build_yarn_cos_sin_cache(
                draft_config, max_seq_len, mscale, device)
        self.cos_full, self.sin_full = cos_sin_cache  # each [max_seq_len, head_dim]

        self._prepare_weights()

    # ---------------------------------------------------------------- helpers
    def _attach(self, tensor: torch.Tensor, name: str):
        if name in self._attach_cache:
            return self._attach_cache[name]
        self._kept.append(tensor)
        dt = self.mpk.attach_input(torch_tensor=tensor.contiguous(), name=name)
        self._attach_cache[name] = dt
        return dt

    def _new(self, dims, name, dtype=bfloat16):
        return self.mpk.new_tensor(dims=dims, dtype=dtype, name=name)

    # ------------------------------------------------------------ YaRN cos/sin
    @staticmethod
    def build_yarn_cos_sin_cache(draft_config, max_seq_len, mscale, device="cuda",
                                 dtype=torch.bfloat16):
        """Build [max_seq_len, head_dim] cos/sin for the draft's YaRN RoPE, scaled
        by `mscale` (vLLM/sglang native = 1.4159; HF ref = 1.0). Uses HF's
        Qwen3RotaryEmbedding for the YaRN frequency schedule, then rescales."""
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Config, Qwen3RotaryEmbedding)
        cfg = Qwen3Config(**{k: v for k, v in draft_config.items()
                             if k not in ("architectures",)})
        cfg.head_dim = int(draft_config.get("head_dim", 128))
        rope = Qwen3RotaryEmbedding(cfg).to(device)
        pos = torch.arange(max_seq_len, device=device).unsqueeze(0)
        dummy = torch.zeros(1, 1, device=device, dtype=dtype)
        cos, sin = rope(dummy, pos)   # [1, max_seq_len, head_dim], HF scaling=1.0
        return (cos[0] * mscale).to(dtype).contiguous(), (sin[0] * mscale).to(dtype).contiguous()

    # ------------------------------------------------------------ weight prep
    def _prepare_weights(self):
        sd = self.sd

        def w(name):
            return sd[name].contiguous()

        self.w_fc = w("fc.weight")                       # [H_d, K*H_t]
        self.w_hidden_norm = w("hidden_norm.weight")     # [H_d]
        self.w_final_norm = w("norm.weight")             # [H_d]
        assert self.w_fc.shape == (self.hidden_size, self.K * self.target_w_embed.shape[1])

        self.layers = []
        for i in range(self.num_layers):
            p = f"layers.{i}."
            self.layers.append(dict(
                iln=w(p + "input_layernorm.weight"),
                q=w(p + "self_attn.q_proj.weight"),
                k=w(p + "self_attn.k_proj.weight"),
                v=w(p + "self_attn.v_proj.weight"),
                o=w(p + "self_attn.o_proj.weight"),
                qn=w(p + "self_attn.q_norm.weight"),
                kn=w(p + "self_attn.k_norm.weight"),
                pln=w(p + "post_attention_layernorm.weight"),
                gate=w(p + "mlp.gate_proj.weight"),
                up=w(p + "mlp.up_proj.weight"),
                down=w(p + "mlp.down_proj.weight"),
            ))

        # Draft-owned paged KV caches, ONE per layer, separate from the target's.
        # Layout [max_num_pages, page_size, num_kv_heads, head_dim]; contiguous
        # pages -> slot s maps to flat row s.
        self.k_cache_bufs = []
        self.v_cache_bufs = []
        for i in range(self.num_layers):
            kb = torch.zeros((self.max_num_pages, self.page_size, self.num_kv_heads,
                              self.head_dim), dtype=self.dtype, device=self.device)
            vb = torch.zeros_like(kb)
            self.k_cache_bufs.append(kb)
            self.v_cache_bufs.append(vb)
        self._kept += self.k_cache_bufs + self.v_cache_bufs
        self._kept += [self.target_w_embed, self.target_w_lm_head,
                       self.cos_full, self.sin_full]

    # --------------------------------------------------------- L2+L4: materialize
    def materialize_context_kv(self, target_hidden, slot_start, num_new):
        """L2 + L4: project `target_hidden` [num_new, K*H_t] for the newly-committed
        tokens into ctx, then per layer compute k_norm+RoPE'd K and raw V and
        dflash_kv_store them into each layer's paged cache at slots
        [slot_start : slot_start+num_new] (overwriting any temp block-K/V there).

        `target_hidden` is a DTensor [num_new, K*H_t]. Appends graph layers.
        """
        mpk, bd = self.mpk, self.bd
        H = self.hidden_size
        S = num_new
        fc = self._attach(self.w_fc, "dflash_fc")
        hn = self._attach(self.w_hidden_norm, "dflash_hidden_norm")
        cos_c = mpk.narrow(self._attach(self.cos_full, "dflash_cos_full"), 0, slot_start, S)
        sin_c = mpk.narrow(self._attach(self.sin_full, "dflash_sin_full"), 0, slot_start, S)
        slot = self._attach(
            torch.arange(slot_start, slot_start + S, device=self.device, dtype=torch.int32),
            f"dflash_ctx_slot_{slot_start}_{S}")

        fc_out = self._new((S, H), "dflash_fc_out"); ctx = self._new((S, H), "dflash_ctx")
        mpk.linear_layer(input=target_hidden, weight=fc, output=fc_out,
                         grid_dim=(_gpl(H), 1, 1), block_dim=bd)
        mpk.rmsnorm_layer(input=fc_out, weight=hn, output=ctx,
                          grid_dim=(S, 1, 1), block_dim=bd)
        self._ctx = ctx

        for i in range(self.num_layers):
            w = self.layers[i]
            kw = self._attach(w["k"], f"dflash_L{i}_k")
            vw = self._attach(w["v"], f"dflash_L{i}_v")
            kn = self._attach(w["kn"], f"dflash_L{i}_kn")
            kc = self._attach(self.k_cache_bufs[i], f"dflash_kcache_{i}")
            vc = self._attach(self.v_cache_bufs[i], f"dflash_vcache_{i}")
            kraw = self._new((S, self.kv_size), f"dflash_L{i}_kraw")
            Kn = self._new((S, self.kv_size), f"dflash_L{i}_Kn")
            vraw = self._new((S, self.kv_size), f"dflash_L{i}_vraw")
            mpk.linear_layer(input=ctx, weight=kw, output=kraw,
                             grid_dim=(_gpl(self.kv_size), 1, 1), block_dim=bd)
            mpk.dflash_norm_rope_layer(x=kraw, weight=kn, cos=cos_c, sin=sin_c, output=Kn,
                                       grid_dim=(1, 1, 1), block_dim=bd, head_dim=self.head_dim)
            mpk.dflash_kv_store_layer(kv_in=Kn, slot_mapping=slot, cache=kc,
                                      grid_dim=(1, 1, 1), block_dim=bd, head_dim=self.head_dim)
            mpk.linear_layer(input=ctx, weight=vw, output=vraw,
                             grid_dim=(_gpl(self.kv_size), 1, 1), block_dim=bd)
            mpk.dflash_kv_store_layer(kv_in=vraw, slot_mapping=slot, cache=vc,
                                      grid_dim=(1, 1, 1), block_dim=bd, head_dim=self.head_dim)

    # ------------------------------------------------------ L5+L6: draft + sample
    def build_draft_forward(self, query_embed, ctx_len, query_pos_start):
        """L5 + L6: non-causal draft over the B-token block `query_embed` [B, H]
        (already embedded [t0, MASK*(B-1)]), reading each layer's paged cache for
        context [0:ctx_len] with the per-layer sliding window; then lm_head + argmax.

        Returns (final_hidden_dt [B,H], draft_tokens_dt [B,1] int64). The s drafts
        are draft_tokens[1:B] (slot 0 is the bonus echo)."""
        mpk, bd = self.mpk, self.bd
        H, B = self.hidden_size, self.B
        I = self.intermediate_size
        gut = grid_for_rmsnorm_linear_layer(2 * I)
        cap = self.max_num_pages * self.page_size
        cosB = mpk.narrow(self._attach(self.cos_full, "dflash_cos_full"), 0, query_pos_start, B)
        sinB = mpk.narrow(self._attach(self.sin_full, "dflash_sin_full"), 0, query_pos_start, B)

        hidden = query_embed
        for i in range(self.num_layers):
            w = self.layers[i]
            iln = self._attach(w["iln"], f"dflash_L{i}_iln")
            qw = self._attach(w["q"], f"dflash_L{i}_q")
            kw = self._attach(w["k"], f"dflash_L{i}_k")
            vw = self._attach(w["v"], f"dflash_L{i}_v")
            ow = self._attach(w["o"], f"dflash_L{i}_o")
            qn = self._attach(w["qn"], f"dflash_L{i}_qn")
            kn = self._attach(w["kn"], f"dflash_L{i}_kn")
            pln = self._attach(w["pln"], f"dflash_L{i}_pln")
            gw = self._attach(w["gate"], f"dflash_L{i}_gate")
            uw = self._attach(w["up"], f"dflash_L{i}_up")
            dw = self._attach(w["down"], f"dflash_L{i}_down")
            gu = mpk.shuffle_tensors(inputs=[gw, uw], shuffled_dim=0,
                                     num_groups=gut // 2, name=f"dflash_L{i}_gateup")
            kc = self._attach(self.k_cache_bufs[i], f"dflash_kcache_{i}")
            vc = self._attach(self.v_cache_bufs[i], f"dflash_vcache_{i}")
            ck = mpk.narrow(mpk.view(kc, [cap, self.kv_size]), 0, 0, ctx_len)
            cv = mpk.narrow(mpk.view(vc, [cap, self.kv_size]), 0, 0, ctx_len)

            h = self._new((B, H), f"dflash_L{i}_h")
            qr = self._new((B, self.q_size), f"dflash_L{i}_qr"); Q = self._new((B, self.q_size), f"dflash_L{i}_Q")
            bkr = self._new((B, self.kv_size), f"dflash_L{i}_bkr"); BK = self._new((B, self.kv_size), f"dflash_L{i}_BK")
            bv = self._new((B, self.kv_size), f"dflash_L{i}_bv")
            at = self._new((B, self.q_size), f"dflash_L{i}_at"); ao = self._new((B, H), f"dflash_L{i}_ao")
            h2 = self._new((B, H), f"dflash_L{i}_h2"); h3 = self._new((B, H), f"dflash_L{i}_h3")
            mid = self._new((B, 2 * I), f"dflash_L{i}_mid"); su = self._new((B, I), f"dflash_L{i}_su")
            mo = self._new((B, H), f"dflash_L{i}_mo"); nx = self._new((B, H), f"dflash_L{i}_nx")

            mpk.rmsnorm_layer(input=hidden, weight=iln, output=h, grid_dim=(B, 1, 1), block_dim=bd)
            mpk.linear_layer(input=h, weight=qw, output=qr, grid_dim=(_gpl(self.q_size), 1, 1), block_dim=bd)
            mpk.dflash_norm_rope_layer(x=qr, weight=qn, cos=cosB, sin=sinB, output=Q, grid_dim=(1, 1, 1), block_dim=bd, head_dim=self.head_dim)
            mpk.linear_layer(input=h, weight=kw, output=bkr, grid_dim=(_gpl(self.kv_size), 1, 1), block_dim=bd)
            mpk.dflash_norm_rope_layer(x=bkr, weight=kn, cos=cosB, sin=sinB, output=BK, grid_dim=(1, 1, 1), block_dim=bd, head_dim=self.head_dim)
            mpk.linear_layer(input=h, weight=vw, output=bv, grid_dim=(_gpl(self.kv_size), 1, 1), block_dim=bd)
            mpk.dflash_attention_layer(q=Q, ctx_k=ck, ctx_v=cv, blk_k=BK, blk_v=bv, output=at,
                                       grid_dim=(1, 1, 1), block_dim=bd,
                                       sliding_window=self.layer_sliding[i], head_dim=self.head_dim)
            mpk.linear_layer(input=at, weight=ow, output=ao, grid_dim=(_gpl(H), 1, 1), block_dim=bd)
            mpk.elementwise_add_layer(input_a=hidden, input_b=ao, output=h2, grid_dim=(B, 1, 1), block_dim=bd)
            mpk.rmsnorm_layer(input=h2, weight=pln, output=h3, grid_dim=(B, 1, 1), block_dim=bd)
            mpk.linear_layer(input=h3, weight=gu, output=mid, grid_dim=(gut, 1, 1), block_dim=bd)
            mpk.silu_mul_layer(input=mid, output=su, grid_dim=(gut // 2, 1, 1), block_dim=bd)
            mpk.linear_layer(input=su, weight=dw, output=mo, grid_dim=(_gpl(H), 1, 1), block_dim=bd)
            mpk.elementwise_add_layer(input_a=h2, input_b=mo, output=nx, grid_dim=(B, 1, 1), block_dim=bd)
            hidden = nx

        nw = self._attach(self.w_final_norm, "dflash_final_norm")
        final = self._attach(self.final_buf, "dflash_final")
        mpk.rmsnorm_layer(input=hidden, weight=nw, output=final, grid_dim=(B, 1, 1), block_dim=bd)

        # L6: lm_head + argmax over the full target vocab (int64 partial indices).
        lm = self._attach(self.target_w_lm_head, "dflash_lm_head")
        V = self.vocab_size
        logits = self._new((B, V), "dflash_logits")
        pv = self._new((B, self.arg_tasks), "dflash_argmax_pval")
        pi = self._new((B, self.arg_tasks), "dflash_argmax_pidx", dtype=int64)
        tok = self._attach(self.draft_tokens_buf, "dflash_draft_tokens")
        mpk.linear_layer(input=final, weight=lm, output=logits, grid_dim=(_gpl(V), 1, 1), block_dim=bd)
        mpk.argmax_partial_layer(input=logits, output=(pv, pi), grid_dim=(self.arg_tasks, 1, 1), block_dim=bd)
        mpk.argmax_reduce_layer(input=(pv, pi), output=tok, grid_dim=(1, 1, 1), block_dim=bd)
        self.final_hidden = final
        self.draft_tokens = tok
        return final, tok

    # ----------------------------------------------------------------- one step
    def build_step(self, target_hidden, query_embed, ctx_len, slot_start, num_new):
        """Convenience: full L2->L6 for one decode step.
          target_hidden : DTensor [num_new, K*H_t]   (this step's new committed ctx)
          query_embed   : DTensor [B, H_d]            (embed_tokens([t0, MASK*(B-1)]))
          ctx_len       : total committed length AFTER this step's commits
          slot_start    : where this step's new ctx K/V are written (= ctx_len-num_new)
          num_new       : number of newly-committed tokens this step
        Returns (final_hidden, draft_tokens). query positions start at ctx_len."""
        self.materialize_context_kv(target_hidden, slot_start, num_new)
        return self.build_draft_forward(query_embed, ctx_len, ctx_len)
