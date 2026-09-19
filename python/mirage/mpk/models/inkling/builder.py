"""Inkling (thinkingmachines/Inkling) model builder for Mirage MPK.

Architecture (from config.json text_config + modular_inkling.py):
- 66 decoder layers, hidden 6144, 64 Q heads, head_dim 128, no RoPE.
- Attention: q = q_norm(wq x) per-head; k = k_norm(k_sconv(wk x));
  v = v_sconv(wv x); relative-position bias from r = wr x (d_rel=16 per head)
  through rel_logits_proj [d_rel, extent]; scores scaled by 1/head_dim; global
  layers additionally scaled by tau = 1 + 0.1*ln(max((P+1)/128000, 1)).
- Local layers (55): 16 KV heads, sliding window 512, extent 512.
  Global layers (11, (i+1)%6==0): 8 KV heads, no window, extent 1024.
- Every layer output (attn and mlp) passes through a depthwise short conv
  (k=4, fp32, adds its own input) before the outer residual add.
- MLP: layers 0-1 dense (I=24576, output * global_scale, folded into w2);
  layers 2-65 MoE: 256 routed experts (I=3072, top-6) + 2 shared experts.
  Router: sigmoid+bias top-6; weights = softmax(logsigmoid(sel ++ shared))
  * route_scale(8) * global_scale. Shared experts are folded into the expert
  tensors as experts 256/257 (always selected at slots 6/7) so the standard
  moe_w13 / silu_mul / moe_w2 / mul_sum_add pipeline runs unchanged.
- Logits: unembed(h / 24), real vocab 200058 (folded: lm_head rows / 24).

v1 limitations: decode-only (max_num_batched_tokens == 1), world_size == 1,
BF16 weights (NVFP4 path to follow). Skips model.mtp.* / audio / visual.
"""

import json
import os
import re
from typing import Optional

import torch

from ..utils import grid_for_rmsnorm_linear_layer, shuffle_tensors
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ....core import bfloat16, float32, int32, int64

# ---- Inkling architecture constants (config.json text_config) --------------
HIDDEN_SIZE = 6144
NUM_Q_HEADS = 64
HEAD_DIM = 128
Q_DIM = NUM_Q_HEADS * HEAD_DIM          # 8192
D_REL = 16
D_REL_PAD = 64                          # bias-GEMM K padded for TMA alignment
GLOBAL_KV_HEADS = 8
LOCAL_KV_HEADS = 16
GLOBAL_EXTENT = 1024
LOCAL_EXTENT = 512
SLIDING_WINDOW = 512
LOG_SCALING_ALPHA = 0.1                 # global layers only
LOG_SCALING_N_FLOOR = 128000
NUM_LAYERS = 66
DENSE_MLP_LAYERS = 2                    # dense_mlp_idx: layers [0, 2) dense
DENSE_INTERMEDIATE = 24576
MOE_INTERMEDIATE = 3072
NUM_ROUTED_EXPERTS = 256
N_SHARED_EXPERTS = 2
NUM_TOTAL_EXPERTS = NUM_ROUTED_EXPERTS + N_SHARED_EXPERTS  # 258
TOPK = 6
K_OUT = TOPK + N_SHARED_EXPERTS         # 8 weight slots per token
ROUTE_SCALE = 8.0
GATE_PADDED = 384                       # 258 -> 384 rows (48 tasks x 8 rows)
SCONV_K = 4
VOCAB_SIZE = 201024
UNPADDED_VOCAB_SIZE = 200058
LOGITS_MUP_DIV = 24.0
EOS_TOKEN_ID = 200006


@register_model_builder("Inkling", "thinkingmachines/Inkling", "inkling")
class InklingBuilder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.rank = mpk.mpi_rank
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.tokenizer = None
        self.model_name: str = None
        self.model_path: str = None
        self.eos_token_id = EOS_TOKEN_ID
        # Keep references to transformed torch tensors (attach_input also keeps
        # refs, but transforms below build tensors before attaching).
        self.pinned_tensors = []
        self._bufs = {}

        # architecture defaults, overridable by config.json
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.num_q_heads = NUM_Q_HEADS
        self.head_dim = HEAD_DIM
        self.d_rel = D_REL
        self.global_extent = GLOBAL_EXTENT
        self.sliding_window = SLIDING_WINDOW
        self.log_scaling_alpha = LOG_SCALING_ALPHA
        self.log_scaling_n_floor = LOG_SCALING_N_FLOOR
        self.dense_mlp_layers = DENSE_MLP_LAYERS
        self.vocab_size = VOCAB_SIZE
        self.unpadded_vocab_size = UNPADDED_VOCAB_SIZE
        self.local_layer_ids = None      # set from config; else derived

        assert self.world_size == 1, "Inkling v1 supports world_size == 1 only"

    # ------------------------------------------------------------- helpers
    def _pin(self, t: torch.Tensor) -> torch.Tensor:
        self.pinned_tensors.append(t)
        return t

    def _attach(self, t: torch.Tensor, name: str):
        return self.mpk.attach_input(torch_tensor=t, name=name)

    def _buf(self, name: str, dims: tuple, dtype=bfloat16):
        """Get-or-create a shared intermediate cuda tensor."""
        if name not in self._bufs:
            self._bufs[name] = self.mpk.new_tensor(
                dims=dims, dtype=dtype, name=name, io_category="cuda_tensor"
            )
        return self._bufs[name]

    @staticmethod
    def _get(state_dict: dict, *names: str) -> torch.Tensor:
        for n in names:
            if n in state_dict:
                return state_dict[n]
        raise KeyError(f"None of {names} found in state_dict")

    @staticmethod
    def _conv_w(t: torch.Tensor) -> torch.Tensor:
        """Checkpoint conv weight [C, 1, K] (or [C, K]) -> [C, K] fp32."""
        if t.dim() == 3:
            t = t.squeeze(1)
        assert t.dim() == 2 and t.shape[1] == SCONV_K, f"conv weight {t.shape}"
        return t.float().contiguous().cuda()

    def _is_local(self, layer_idx: int) -> bool:
        if self.local_layer_ids is not None:
            return layer_idx in self.local_layer_ids
        return (layer_idx + 1) % 6 != 0

    @property
    def max_ctx(self) -> int:
        if self.max_num_pages and self.page_size:
            return self.max_num_pages * self.page_size
        return getattr(self.mpk, "max_seq_length", None) or 8192

    # ------------------------------------------------------------- loading
    def build_from_config(self, model_config: MirageModelConfig):
        if model_config.hidden_size:
            self.hidden_size = model_config.hidden_size
        if model_config.num_layers:
            self.num_layers = model_config.num_layers
        if model_config.vocab_size:
            self.vocab_size = model_config.vocab_size
        self.build_from_dict(model_config.state_dict, model_config.with_lm_head)

    def build_from_model(self, model_name: str, model_path: str | None = None):
        from transformers import AutoTokenizer

        path = model_path or model_name
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(model_name)
        self.model_name = model_name
        self.model_path = path

        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        self.hidden_size = tc.get("hidden_size", self.hidden_size)
        self.num_layers = tc.get("num_hidden_layers", self.num_layers)
        self.num_q_heads = tc.get("num_attention_heads", self.num_q_heads)
        self.head_dim = tc.get("head_dim", self.head_dim)
        self.d_rel = tc.get("d_rel", self.d_rel)
        self.global_extent = tc.get("rel_extent", self.global_extent)
        self.sliding_window = tc.get("sliding_window_size", self.sliding_window)
        self.log_scaling_alpha = tc.get("log_scaling_alpha", self.log_scaling_alpha)
        self.log_scaling_n_floor = tc.get("log_scaling_n_floor", self.log_scaling_n_floor)
        self.dense_mlp_layers = tc.get("dense_mlp_idx", self.dense_mlp_layers)
        self.vocab_size = tc.get("vocab_size", self.vocab_size)
        self.unpadded_vocab_size = tc.get("unpadded_vocab_size", self.unpadded_vocab_size)
        if "local_layer_ids" in tc:
            self.local_layer_ids = set(tc["local_layer_ids"])
        eos = cfg.get("eos_token_id")
        if isinstance(eos, int):
            self.eos_token_id = eos

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            print(f"[inkling] tokenizer load failed ({e}); continuing without")

        state_dict = self._load_llm_state_dict(path)
        self.build_from_dict(state_dict, True)

    def _load_llm_state_dict(self, path: str) -> dict:
        """Load only model.llm.* weights (skip mtp/audio/visual) onto cuda."""
        from safetensors import safe_open

        index_path = os.path.join(path, "model.safetensors.index.json")
        state_dict = {}
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            by_shard = {}
            for key, shard in weight_map.items():
                if key.startswith("model.llm."):
                    by_shard.setdefault(shard, []).append(key)
            for shard, keys in sorted(by_shard.items()):
                with safe_open(os.path.join(path, shard), framework="pt",
                               device="cuda") as f:
                    for key in keys:
                        state_dict[key] = f.get_tensor(key)
        else:
            # single-file checkpoint
            shard = os.path.join(path, "model.safetensors")
            with safe_open(shard, framework="pt", device="cuda") as f:
                for key in f.keys():
                    if key.startswith("model.llm."):
                        state_dict[key] = f.get_tensor(key)
        return state_dict

    # ------------------------------------------------- intermediate tensors
    def new_intermediate_tensors(self):
        mbt = self.mpk.max_num_batched_tokens
        assert mbt == 1, "Inkling v1 is decode-only (max_num_batched_tokens=1)"
        self.mbt = mbt
        H = self.hidden_size

        self.embed_out = self._buf("embed_out", (mbt, H))
        self.embed_normed = self._buf("embed_normed", (mbt, H))
        self.rmsnorm_out = self._buf("rmsnorm_out", (mbt, H))
        self.q_buf = self._buf("q_proj_out", (mbt, Q_DIM))
        self.q_normed = self._buf("q_normed", (mbt, Q_DIM))
        self.r_buf = self._buf("r_proj_out", (mbt, NUM_Q_HEADS * D_REL_PAD))
        self.attn_out = self._buf("attn_out", (mbt, Q_DIM))
        self.attn_proj_out = self._buf("attn_proj_out", (mbt, H))
        self.attn_sconv_out = self._buf("attn_sconv_out", (mbt, H))
        self.mlp_out = self._buf("mlp_out", (mbt, H))
        self.mlp_sconv_out = self._buf("mlp_sconv_out", (mbt, H))

        # zero residual for the MoE mul_sum_add (real residual is added after
        # mlp_sconv); attached, never written.
        self.zero_residual = self._attach(
            self._pin(torch.zeros(mbt, H, dtype=torch.bfloat16, device="cuda")),
            "inkling_zero_residual",
        )

        # MoE routing/pipeline buffers (shared across layers)
        self.moe_logits = self._buf("moe_gate_logits", (mbt, GATE_PADDED))
        self.moe_topk_weights = self._buf(
            "moe_topk_weights", (mbt, K_OUT), dtype=float32)
        self.moe_routing_indices = self._buf(
            "moe_routing_indices", (NUM_TOTAL_EXPERTS, mbt), dtype=int32)
        self.moe_active = self._buf(
            "moe_active_experts", (NUM_TOTAL_EXPERTS + 1,), dtype=int32)
        self.moe_mid = self._buf("moe_mid", (mbt, K_OUT, 2 * MOE_INTERMEDIATE))
        self.moe_silu = self._buf("moe_silu", (mbt, K_OUT, MOE_INTERMEDIATE))
        self.moe_down = self._buf("moe_down", (mbt, K_OUT, H))

        if self.dense_mlp_layers > 0:
            self.dense_mid = self._buf("dense_mid", (mbt, 2 * DENSE_INTERMEDIATE))
            self.dense_silu = self._buf("dense_silu", (mbt, DENSE_INTERMEDIATE))

        # lm head / argmax
        self.padded_vocab_size = ((self.vocab_size + 255) // 256) * 256
        self.argmax_in = self._buf("argmax_in", (mbt, self.padded_vocab_size))
        self.argmax_part_value = self._buf(
            "argmax_part_value", (mbt, self.mpk.num_workers))
        self.argmax_part_index = self._buf(
            "argmax_part_index", (mbt, self.mpk.num_workers), dtype=int64)

    def _kv_bufs(self, nkv: int):
        """Width-dependent K/V activation buffers (local vs global layers)."""
        w = nkv * self.head_dim
        return (
            self._buf(f"k_proj_out_{w}", (self.mbt, w)),
            self._buf(f"v_proj_out_{w}", (self.mbt, w)),
            self._buf(f"k_conv_out_{w}", (self.mbt, w)),
            self._buf(f"v_conv_out_{w}", (self.mbt, w)),
            self._buf(f"k_normed_{w}", (self.mbt, w)),
        )

    def _bias_buf(self, extent: int):
        return self._buf(f"rel_bias_{extent}", (NUM_Q_HEADS, extent))

    # ------------------------------------------------------------- layers
    def _build_attention(self, i: int, state_dict: dict):
        mpk = self.mpk
        H = self.hidden_size
        D = self.head_dim
        prefix = f"model.llm.layers.{i}."
        is_local = self._is_local(i)
        nkv = LOCAL_KV_HEADS if is_local else GLOBAL_KV_HEADS
        extent = LOCAL_EXTENT if is_local else self.global_extent
        sw = self.sliding_window if is_local else 0
        alpha = 0.0 if is_local else self.log_scaling_alpha
        kv_width = nkv * D
        k_buf, v_buf, k_conv, v_conv, k_normed = self._kv_bufs(nkv)
        bias_buf = self._bias_buf(extent)

        # attn_norm
        w_norm = self._attach(
            self._get(state_dict, f"{prefix}attn_norm.weight"),
            f"layer_{i}_attn_norm")
        mpk.rmsnorm_layer(
            input=self.x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        # q/k/v/r projections
        wq_t = self._get(state_dict, f"{prefix}attn.wq_du.weight")
        assert wq_t.shape == (Q_DIM, H), f"wq {wq_t.shape}"
        wq = self._attach(wq_t, f"layer_{i}_wq")
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=wq, output=self.q_buf,
            grid_dim=(grid_for_rmsnorm_linear_layer(Q_DIM), 1, 1),
            block_dim=(128, 1, 1))

        wk_t = self._get(state_dict, f"{prefix}attn.wk_dv.weight")
        assert wk_t.shape == (kv_width, H), \
            f"layer {i} wk {wk_t.shape}, expected ({kv_width}, {H})"
        wk = self._attach(wk_t, f"layer_{i}_wk")
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=wk, output=k_buf,
            grid_dim=(grid_for_rmsnorm_linear_layer(kv_width), 1, 1),
            block_dim=(128, 1, 1))

        wv_t = self._get(state_dict, f"{prefix}attn.wv_dv.weight")
        assert wv_t.shape == (kv_width, H)
        wv = self._attach(wv_t, f"layer_{i}_wv")
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=wv, output=v_buf,
            grid_dim=(grid_for_rmsnorm_linear_layer(kv_width), 1, 1),
            block_dim=(128, 1, 1))

        # r projection, d_rel padded 16 -> 64 per head (zero rows)
        wr_t = self._get(state_dict, f"{prefix}attn.wr_du.weight")
        assert wr_t.shape == (NUM_Q_HEADS * self.d_rel, H), f"wr {wr_t.shape}"
        wr_pad = torch.zeros(NUM_Q_HEADS * D_REL_PAD, H,
                             dtype=wr_t.dtype, device=wr_t.device)
        wr_pad.view(NUM_Q_HEADS, D_REL_PAD, H)[:, :self.d_rel, :] = \
            wr_t.view(NUM_Q_HEADS, self.d_rel, H)
        wr = self._attach(self._pin(wr_pad), f"layer_{i}_wr")
        state_dict.pop(f"{prefix}attn.wr_du.weight", None)
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=wr, output=self.r_buf,
            grid_dim=(grid_for_rmsnorm_linear_layer(NUM_Q_HEADS * D_REL_PAD), 1, 1),
            block_dim=(128, 1, 1))

        # k/v short convolutions (state updated in place)
        for name, in_buf, out_buf in (("k", k_buf, k_conv),
                                      ("v", v_buf, v_conv)):
            cw = self._attach(
                self._pin(self._conv_w(
                    self._get(state_dict, f"{prefix}attn.{name}_sconv.weight"))),
                f"layer_{i}_{name}_sconv_w")
            cs = self._attach(
                self._pin(torch.zeros(SCONV_K - 1, kv_width,
                                      dtype=torch.float32, device="cuda")),
                f"layer_{i}_{name}_conv_state")
            mpk.inkling_sconv_layer(
                x=in_buf, weight=cw, conv_state=cs, output=out_buf,
                grid_dim=(kv_width // 128, 1, 1), block_dim=(128, 1, 1))

        # per-head q_norm / k_norm (rmsnorm over views [heads, head_dim])
        w_qn = self._attach(
            self._get(state_dict, f"{prefix}attn.q_norm.weight"),
            f"layer_{i}_q_norm")
        mpk.rmsnorm_layer(
            input=mpk.view(self.q_buf, [NUM_Q_HEADS, D]),
            weight=w_qn,
            output=mpk.view(self.q_normed, [NUM_Q_HEADS, D]),
            grid_dim=(NUM_Q_HEADS, 1, 1), block_dim=(128, 1, 1))
        w_kn = self._attach(
            self._get(state_dict, f"{prefix}attn.k_norm.weight"),
            f"layer_{i}_k_norm")
        mpk.rmsnorm_layer(
            input=mpk.view(k_conv, [nkv, D]),
            weight=w_kn,
            output=mpk.view(k_normed, [nkv, D]),
            grid_dim=(nkv, 1, 1), block_dim=(128, 1, 1))

        # relative-position bias table: [heads, d_rel_pad] @ projT -> [heads, extent]
        proj_t = self._get(state_dict, f"{prefix}attn.rel_logits_proj.proj")
        assert proj_t.shape == (self.d_rel, extent), \
            f"layer {i} rel proj {proj_t.shape}, expected ({self.d_rel}, {extent})"
        proj_pad = torch.zeros(extent, D_REL_PAD,
                               dtype=torch.bfloat16, device=proj_t.device)
        proj_pad[:, :self.d_rel] = proj_t.t().to(torch.bfloat16)
        w_proj = self._attach(self._pin(proj_pad.contiguous()),
                              f"layer_{i}_rel_proj")
        state_dict.pop(f"{prefix}attn.rel_logits_proj.proj", None)
        mpk.linear_layer(
            input=mpk.view(self.r_buf, [NUM_Q_HEADS, D_REL_PAD]),
            weight=w_proj, output=bias_buf,
            grid_dim=(grid_for_rmsnorm_linear_layer(extent), 1, 1),
            block_dim=(128, 1, 1))

        # KV caches: 2D for attention reads, 4D view for the paged store
        k_cache = self._attach(
            self._pin(torch.zeros(self.max_ctx, kv_width,
                                  dtype=torch.bfloat16, device="cuda")),
            f"layer_{i}_k_cache")
        v_cache = self._attach(
            self._pin(torch.zeros(self.max_ctx, kv_width,
                                  dtype=torch.bfloat16, device="cuda")),
            f"layer_{i}_v_cache")
        mpk.dflash_kv_store_layer(
            kv_in=k_normed, slot_mapping=self.step_dt,
            cache=mpk.view(k_cache, [self.max_ctx, 1, nkv, D]),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1), head_dim=D)
        mpk.dflash_kv_store_layer(
            kv_in=v_conv, slot_mapping=self.step_dt,
            cache=mpk.view(v_cache, [self.max_ctx, 1, nkv, D]),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1), head_dim=D)

        mpk.inkling_attention_layer(
            q=self.q_normed, ctx_k=k_cache, ctx_v=v_cache,
            blk_k=k_normed, blk_v=v_conv, bias=bias_buf, step=self.step_dt,
            output=self.attn_out,
            grid_dim=(nkv, 1, 1), block_dim=(128, 1, 1),
            sliding_window=sw, extent=extent, head_dim=D,
            log_scaling_alpha=alpha,
            log_scaling_n_floor=self.log_scaling_n_floor)

        # o_proj -> attn short conv -> outer residual add
        wo_t = self._get(state_dict, f"{prefix}attn.wo_ud.weight")
        assert wo_t.shape == (H, Q_DIM)
        wo = self._attach(wo_t, f"layer_{i}_wo")
        mpk.linear_layer(
            input=self.attn_out, weight=wo, output=self.attn_proj_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(H), 1, 1),
            block_dim=(128, 1, 1))

        cw = self._attach(
            self._pin(self._conv_w(
                self._get(state_dict, f"{prefix}attn_sconv.weight"))),
            f"layer_{i}_attn_sconv_w")
        cs = self._attach(
            self._pin(torch.zeros(SCONV_K - 1, H,
                                  dtype=torch.float32, device="cuda")),
            f"layer_{i}_attn_conv_state")
        mpk.inkling_sconv_layer(
            x=self.attn_proj_out, weight=cw, conv_state=cs,
            output=self.attn_sconv_out,
            grid_dim=(H // 128, 1, 1), block_dim=(128, 1, 1))

        attn_resid = self.mpk.new_tensor(
            dims=(self.mbt, H), dtype=bfloat16,
            name=f"layer_{i}_attn_resid", io_category="cuda_tensor")
        mpk.elementwise_add_layer(
            input_a=self.x, input_b=self.attn_sconv_out, output=attn_resid,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))
        self.x = attn_resid

    def _build_dense_mlp(self, i: int, state_dict: dict):
        mpk = self.mpk
        H = self.hidden_size
        I = DENSE_INTERMEDIATE
        prefix = f"model.llm.layers.{i}.mlp."

        w13_t = self._get(state_dict, f"{prefix}w13_dn.weight")
        assert w13_t.shape == (2 * I, H), f"w13_dn {w13_t.shape}"
        linear_grid = grid_for_rmsnorm_linear_layer(2 * I)   # 192
        split = linear_grid // 2                              # 96
        w13_shuf = shuffle_tensors([w13_t[:I], w13_t[I:]], split, 0)
        w13 = self._attach(self._pin(w13_shuf), f"layer_{i}_w13_dn")
        state_dict.pop(f"{prefix}w13_dn.weight", None)
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=w13, output=self.dense_mid,
            grid_dim=(linear_grid, 1, 1), block_dim=(128, 1, 1))
        mpk.silu_mul_layer(
            input=self.dense_mid, output=self.dense_silu,
            grid_dim=(split, 1, 1), block_dim=(128, 1, 1))

        # fold the dense global_scale scalar into w2
        gscale = float(self._get(state_dict, f"{prefix}global_scale")
                       .float().item())
        w2_t = self._get(state_dict, f"{prefix}w2_md.weight")
        assert w2_t.shape == (H, I)
        w2_scaled = (w2_t.float() * gscale).to(torch.bfloat16).contiguous()
        w2 = self._attach(self._pin(w2_scaled), f"layer_{i}_w2_md")
        state_dict.pop(f"{prefix}w2_md.weight", None)
        mpk.linear_layer(
            input=self.dense_silu, weight=w2, output=self.mlp_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(H), 1, 1),
            block_dim=(128, 1, 1))

    def _build_moe_mlp(self, i: int, state_dict: dict):
        mpk = self.mpk
        H = self.hidden_size
        I = MOE_INTERMEDIATE
        prefix = f"model.llm.layers.{i}.mlp."

        # gate linear: [258, H] padded to [384, H] so grid 48 keeps >=8 bf16
        # rows (16B) per block; router reads stride 384, first 258 columns.
        gate_t = self._get(state_dict, f"{prefix}gate.weight")
        assert gate_t.shape == (NUM_TOTAL_EXPERTS, H), f"gate {gate_t.shape}"
        gate_pad = torch.zeros(GATE_PADDED, H,
                               dtype=gate_t.dtype, device=gate_t.device)
        gate_pad[:NUM_TOTAL_EXPERTS] = gate_t
        w_gate = self._attach(self._pin(gate_pad), f"layer_{i}_moe_gate")
        state_dict.pop(f"{prefix}gate.weight", None)
        gate_grid = min(grid_for_rmsnorm_linear_layer(GATE_PADDED),
                        GATE_PADDED // 8)                     # 48
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_gate, output=self.moe_logits,
            grid_dim=(gate_grid, 1, 1), block_dim=(128, 1, 1))

        # zero mlp_out before the pipeline (mirrors deepseek_v3)
        mpk.tensor_init_layer(
            input=self.mlp_out, dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        # router: e_score_correction_bias [256] fp32, global_scale [1] fp32
        bias_t = self._get(state_dict, f"{prefix}gate.bias")
        assert bias_t.shape == (NUM_ROUTED_EXPERTS,), f"gate.bias {bias_t.shape}"
        w_bias = self._attach(self._pin(bias_t.float().contiguous()),
                              f"layer_{i}_moe_gate_bias")
        gs_t = self._get(state_dict, f"{prefix}gate.global_scale")
        w_gs = self._attach(self._pin(gs_t.reshape(1).float().contiguous()),
                            f"layer_{i}_moe_global_scale")
        mpk.inkling_moe_router_layer(
            logits=self.moe_logits, bias=w_bias, global_scale=w_gs,
            output=(self.moe_topk_weights, self.moe_routing_indices,
                    self.moe_active),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            route_scale=ROUTE_SCALE, n_shared=N_SHARED_EXPERTS)

        # expert weights with shared experts folded in as experts 256/257
        w13_t = self._get(state_dict, f"{prefix}experts.w13_weight")
        shared_w13_t = self._get(
            state_dict, f"{prefix}shared_experts.shared_w13_weight")
        assert w13_t.shape == (NUM_ROUTED_EXPERTS, 2 * I, H), f"{w13_t.shape}"
        assert shared_w13_t.shape == (N_SHARED_EXPERTS, 2 * I, H), \
            f"{shared_w13_t.shape}"
        w13_cat = torch.cat([w13_t, shared_w13_t], dim=0).contiguous()
        state_dict.pop(f"{prefix}experts.w13_weight", None)
        state_dict.pop(f"{prefix}shared_experts.shared_w13_weight", None)
        w13 = self._attach(self._pin(w13_cat), f"layer_{i}_experts_w13")
        mpk.moe_w13_linear_layer(
            input=self.rmsnorm_out, weight=w13,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_active, output=self.moe_mid,
            grid_dim=(NUM_TOTAL_EXPERTS, 1, 1), block_dim=(128, 1, 1))

        mpk.moe_silu_mul_layer(
            input=self.moe_mid, output=self.moe_silu,
            grid_dim=(self.mbt, K_OUT, 1), block_dim=(128, 1, 1))

        w2_t = self._get(state_dict, f"{prefix}experts.w2_weight")
        shared_w2_t = self._get(
            state_dict, f"{prefix}shared_experts.shared_w2_weight")
        assert w2_t.shape == (NUM_ROUTED_EXPERTS, H, I), f"{w2_t.shape}"
        assert shared_w2_t.shape == (N_SHARED_EXPERTS, H, I), \
            f"{shared_w2_t.shape}"
        w2_cat = torch.cat([w2_t, shared_w2_t], dim=0).contiguous()
        state_dict.pop(f"{prefix}experts.w2_weight", None)
        state_dict.pop(f"{prefix}shared_experts.shared_w2_weight", None)
        w2 = self._attach(self._pin(w2_cat), f"layer_{i}_experts_w2")
        mpk.moe_w2_linear_layer(
            input=self.moe_silu, weight=w2,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_active, output=self.moe_down,
            grid_dim=(NUM_TOTAL_EXPERTS, 1, 1), block_dim=(128, 1, 1))

        # weighted sum over the 8 slots (6 routed + 2 shared gammas); the
        # residual here is zero -- the real residual is added after mlp_sconv.
        mpk.moe_mul_sum_add_layer(
            input=self.moe_down, weight=self.moe_topk_weights,
            residual=self.zero_residual, output=self.mlp_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

    def _build_mlp(self, i: int, state_dict: dict):
        mpk = self.mpk
        H = self.hidden_size
        prefix = f"model.llm.layers.{i}."

        w_norm = self._attach(
            self._get(state_dict, f"{prefix}mlp_norm.weight"),
            f"layer_{i}_mlp_norm")
        mpk.rmsnorm_layer(
            input=self.x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        if i < self.dense_mlp_layers:
            self._build_dense_mlp(i, state_dict)
        else:
            self._build_moe_mlp(i, state_dict)

        # mlp short conv -> outer residual add
        cw = self._attach(
            self._pin(self._conv_w(
                self._get(state_dict, f"{prefix}mlp_sconv.weight"))),
            f"layer_{i}_mlp_sconv_w")
        cs = self._attach(
            self._pin(torch.zeros(SCONV_K - 1, H,
                                  dtype=torch.float32, device="cuda")),
            f"layer_{i}_mlp_conv_state")
        mpk.inkling_sconv_layer(
            x=self.mlp_out, weight=cw, conv_state=cs,
            output=self.mlp_sconv_out,
            grid_dim=(H // 128, 1, 1), block_dim=(128, 1, 1))

        mlp_resid = self.mpk.new_tensor(
            dims=(self.mbt, H), dtype=bfloat16,
            name=f"layer_{i}_mlp_resid", io_category="cuda_tensor")
        mpk.elementwise_add_layer(
            input_a=self.x, input_b=self.mlp_sconv_out, output=mlp_resid,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))
        self.x = mlp_resid

    def build_layers(self, state_dict: dict):
        for i in range(self.num_layers):
            self._build_attention(i, state_dict)
            self._build_mlp(i, state_dict)

    # ------------------------------------------------------------- top level
    def build_from_dict(self, state_dict: dict, with_lm_head: bool):
        mpk = self.mpk

        # detect layer count from checkpoint if it disagrees with config
        layer_ids = set()
        pat = re.compile(r"model\.llm\.layers\.(\d+)\.attn_norm\.weight")
        for k in state_dict:
            m = pat.match(k)
            if m:
                layer_ids.add(int(m.group(1)))
        if layer_ids:
            detected = max(layer_ids) + 1
            if detected != self.num_layers:
                print(f"[inkling] num_layers {self.num_layers} -> {detected} "
                      f"(from checkpoint)")
                self.num_layers = detected

        self.x = self._attach(self.input_tokens, "input_token")
        # decode step counter ([:1] view of the runtime step tensor); doubles
        # as the kv-store slot mapping and the attention ctx_len input.
        # (fetched here, not in __init__: test mode allocates it lazily)
        step_tensor = mpk.meta_tensors["step"]
        assert step_tensor.dtype == torch.int32
        self.step_dt = self._attach(step_tensor[:1], "inkling_step")

        self.new_intermediate_tensors()

        argmax_out = self._attach(self.output_tokens, "output_token")

        # embed + embed_norm
        w_embed = self._attach(
            self._get(state_dict, "model.llm.embed.weight"), "embed_tokens")
        mpk.embed_layer(
            input=self.x, weight=w_embed, output=self.embed_out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)
        w_embed_norm = self._attach(
            self._get(state_dict, "model.llm.embed_norm.weight"), "embed_norm")
        mpk.rmsnorm_layer(
            input=self.embed_out, weight=w_embed_norm,
            output=self.embed_normed,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))
        self.x = self.embed_normed

        self.build_layers(state_dict)

        # final norm
        w_norm = self._attach(
            self._get(state_dict, "model.llm.norm.weight"), "model_norm")
        mpk.rmsnorm_layer(
            input=self.x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        if not with_lm_head:
            return

        # lm head: real vocab rows / mup_multiplier, padded with zeros
        unembed = self._get(state_dict, "model.llm.unembed.weight")
        assert unembed.shape[1] == self.hidden_size
        real = unembed[:self.unpadded_vocab_size].float() / LOGITS_MUP_DIV
        lm_head = torch.zeros(self.padded_vocab_size, self.hidden_size,
                              dtype=torch.bfloat16, device=unembed.device)
        lm_head[:self.unpadded_vocab_size] = real.to(torch.bfloat16)
        w_lm = self._attach(self._pin(lm_head), "lm_head")
        state_dict.pop("model.llm.unembed.weight", None)
        mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_lm, output=self.argmax_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(self.padded_vocab_size), 1, 1),
            block_dim=(128, 1, 1))

        mpk.argmax_partial_layer(
            input=self.argmax_in,
            output=(self.argmax_part_value, self.argmax_part_index),
            grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
        mpk.argmax_reduce_layer(
            input=(self.argmax_part_value, self.argmax_part_index),
            output=argmax_out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    # ------------------------------------------------------------- tokenizer
    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
