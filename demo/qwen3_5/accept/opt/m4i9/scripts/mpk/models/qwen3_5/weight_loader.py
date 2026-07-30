"""Qwen3.5 checkpoint -> MPK runtime weight loader.

Two halves, deliberately separable:

  * `plan_checkpoint()` — a PURE function from the safetensors index's key list
    to a destination for EVERY key: either a runtime tensor slot or an explicit
    skip with a reason. It touches no tensor data, so the round-trip inventory
    test can assert "each of the 64 196 keys is consumed exactly once or skipped
    with a reason" without a GPU or 37 GB of I/O.
  * `Qwen35WeightLoader.load()` — executes that plan, applying the
    `transforms.py` maps, and asserts at the end that it consumed exactly the
    planned key set, once each.

The checkpoint -> runtime map is `docs/qwen35/vllm-graph.md` §5.1-§5.4; the
transforms are `docs/qwen35/v1-architecture.md` §2.0.

Skips (both legitimate, both explicit):
  * `model.visual.*`   — the vision tower. This is a text-only decode path
    (goal AC-2); `vllm-graph.md` §5.4 marks these 333 tensors "not needed for
    text decode".
  * `mtp.*`            — the MTP draft layer, which the main model itself skips
    (`skip_prefixes=["mtp."]`, `qwen3_5.py:372`, `vllm-graph.md` §5.4).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from . import transforms as T

LANG = "model.language_model."
SKIP_PREFIXES: Dict[str, str] = {
    "model.visual.": "vision tower - text-only decode path (goal AC-2; "
                     "vllm-graph.md 5.4)",
    "mtp.": "MTP draft layer - skipped by the main model "
            "(skip_prefixes=['mtp.'], vllm-graph.md 5.4)",
}

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class Qwen35Config:
    hidden_size: int
    num_layers: int
    layer_types: List[str]
    vocab_size: int
    rms_norm_eps: float
    eos_token_id: int
    # full attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    # GDN
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    # MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int

    @classmethod
    def from_json(cls, path: str) -> "Qwen35Config":
        with open(path) as f:
            raw = json.load(f)
        tc = raw.get("text_config", raw)
        rp = tc.get("rope_parameters", {})
        head_dim = tc["head_dim"]
        return cls(
            hidden_size=tc["hidden_size"],
            num_layers=tc["num_hidden_layers"],
            layer_types=list(tc["layer_types"]),
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            eos_token_id=tc["eos_token_id"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=head_dim,
            rotary_dim=int(head_dim * rp.get("partial_rotary_factor", 1.0)),
            rope_theta=float(rp.get("rope_theta", 10000.0)),
            linear_num_key_heads=tc["linear_num_key_heads"],
            linear_num_value_heads=tc["linear_num_value_heads"],
            linear_key_head_dim=tc["linear_key_head_dim"],
            linear_value_head_dim=tc["linear_value_head_dim"],
            linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
            num_experts=tc["num_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            moe_intermediate_size=tc["moe_intermediate_size"],
            shared_expert_intermediate_size=tc["shared_expert_intermediate_size"],
        )

    # ---- derived shapes -------------------------------------------------
    @property
    def gdn_layers(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def attn_layers(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def conv_dim(self) -> int:
        """`2*key_dim + value_dim` — the width the depthwise conv sees."""
        return (2 * self.linear_num_key_heads * self.linear_key_head_dim
                + self.linear_num_value_heads * self.linear_value_head_dim)

    @property
    def gdn_z_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def qkvg_dim(self) -> int:
        """MPK's packed attention row width (`persistent_kernel.py:988`)."""
        qo_per_kv = self.num_attention_heads // self.num_key_value_heads
        return (2 * qo_per_kv + 2) * self.head_dim * self.num_key_value_heads

    @property
    def padded_vocab_size(self) -> int:
        """No padding needed: 248320 = 970*256 (`v1-architecture.md` §7).

        The Qwen3 builder hardcodes 153600 and must not be "fixed" in place
        (`mpk-gaps.md` Gap 6) — this builder computes its own and asserts the
        256-divisor grid constraint `grid_for_rmsnorm_linear_layer` imposes.
        """
        v = self.vocab_size
        if v % 256 != 0:
            v = ((v + 255) // 256) * 256
        return v


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------
@dataclass
class CheckpointPlan:
    consumed: Dict[str, str] = field(default_factory=dict)   # key -> destination
    skipped: Dict[str, str] = field(default_factory=dict)    # key -> reason

    def summary(self) -> Dict[str, int]:
        by_reason: Dict[str, int] = {}
        for reason in self.skipped.values():
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return {
            "total": len(self.consumed) + len(self.skipped),
            "mapped": len(self.consumed),
            "skipped": len(self.skipped),
            **{f"skipped::{r}": c for r, c in sorted(by_reason.items())},
        }


_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$")

# suffix -> runtime destination template, for the per-layer keys that map 1:1.
_GDN_SUFFIX = {
    "linear_attn.in_proj_qkv.weight": "layer_{i}_gdn_in_proj_qkv",
    "linear_attn.in_proj_qkv.weight_scale_inv": "layer_{i}_gdn_in_proj_qkv_scale",
    "linear_attn.in_proj_z.weight": "layer_{i}_gdn_in_proj_z",
    "linear_attn.in_proj_z.weight_scale_inv": "layer_{i}_gdn_in_proj_z_scale",
    "linear_attn.in_proj_b.weight": "layer_{i}_gdn_in_proj_ba[b]",
    "linear_attn.in_proj_a.weight": "layer_{i}_gdn_in_proj_ba[a]",
    "linear_attn.conv1d.weight": "layer_{i}_gdn_conv1d",
    "linear_attn.A_log": "layer_{i}_gdn_alog_dtbias[A_log]",
    "linear_attn.dt_bias": "layer_{i}_gdn_alog_dtbias[dt_bias]",
    "linear_attn.norm.weight": "layer_{i}_gdn_norm",
    "linear_attn.out_proj.weight": "layer_{i}_gdn_out_proj",
    "linear_attn.out_proj.weight_scale_inv": "layer_{i}_gdn_out_proj_scale",
}
_ATTN_SUFFIX = {
    "self_attn.q_proj.weight": "layer_{i}_qkvg_proj[q]",
    "self_attn.q_proj.weight_scale_inv": "layer_{i}_qkvg_proj_scale[q]",
    "self_attn.k_proj.weight": "layer_{i}_qkvg_proj[k]",
    "self_attn.k_proj.weight_scale_inv": "layer_{i}_qkvg_proj_scale[k]",
    "self_attn.v_proj.weight": "layer_{i}_qkvg_proj[v]",
    "self_attn.v_proj.weight_scale_inv": "layer_{i}_qkvg_proj_scale[v]",
    "self_attn.q_norm.weight": "layer_{i}_q_norm",
    "self_attn.k_norm.weight": "layer_{i}_k_norm",
    "self_attn.o_proj.weight": "layer_{i}_o_proj",
    "self_attn.o_proj.weight_scale_inv": "layer_{i}_o_proj_scale",
}
_COMMON_SUFFIX = {
    "input_layernorm.weight": "layer_{i}_input_layernorm",
    "post_attention_layernorm.weight": "layer_{i}_post_attention_layernorm",
    "mlp.gate.weight": "layer_{i}_router",
    "mlp.shared_expert_gate.weight": "layer_{i}_shared_expert_gate",
    "mlp.shared_expert.gate_proj.weight": "layer_{i}_shared_gate_up[gate]",
    "mlp.shared_expert.gate_proj.weight_scale_inv": "layer_{i}_shared_gate_up_scale[gate]",
    "mlp.shared_expert.up_proj.weight": "layer_{i}_shared_gate_up[up]",
    "mlp.shared_expert.up_proj.weight_scale_inv": "layer_{i}_shared_gate_up_scale[up]",
    "mlp.shared_expert.down_proj.weight": "layer_{i}_shared_down",
    "mlp.shared_expert.down_proj.weight_scale_inv": "layer_{i}_shared_down_scale",
}
_MODEL_LEVEL = {
    "embed_tokens.weight": "embed_tokens",
    "norm.weight": "model_norm",
}


def plan_checkpoint(keys, config: Qwen35Config) -> CheckpointPlan:
    """Map EVERY checkpoint key to a runtime destination or an explicit skip.

    Raises on any key this loader does not recognise — an unmapped tensor is a
    silently-dropped weight, which is exactly the failure mode the round-trip
    inventory test exists to catch.
    """
    plan = CheckpointPlan()
    for key in keys:
        skipped = False
        for prefix, reason in SKIP_PREFIXES.items():
            if key.startswith(prefix):
                plan.skipped[key] = reason
                skipped = True
                break
        if skipped:
            continue

        if key == "lm_head.weight":
            plan.consumed[key] = "lm_head"
            continue
        if not key.startswith(LANG):
            raise KeyError(f"unrecognised checkpoint key (no plan): {key}")

        rest = key[len(LANG):]
        if rest in _MODEL_LEVEL:
            plan.consumed[key] = _MODEL_LEVEL[rest]
            continue

        m = _LAYER_RE.match(rest)
        if not m:
            raise KeyError(f"unrecognised checkpoint key (no plan): {key}")
        i, suffix = int(m.group(1)), m.group(2)
        if not 0 <= i < config.num_layers:
            raise KeyError(f"layer index {i} out of range in {key}")
        is_gdn = config.layer_types[i] == "linear_attention"

        e = _EXPERT_RE.match(suffix)
        if e:
            expert, proj, kind = int(e.group(1)), e.group(2), e.group(3)
            if not 0 <= expert < config.num_experts:
                raise KeyError(f"expert index {expert} out of range in {key}")
            bank = "w2" if proj == "down_proj" else "w13"
            slot = "" if kind == "weight" else "_scale"
            plan.consumed[key] = f"layer_{i}_{bank}{slot}[e{expert}:{proj}]"
            continue

        table = _COMMON_SUFFIX
        if suffix in table:
            plan.consumed[key] = table[suffix].format(i=i)
            continue
        table = _GDN_SUFFIX if is_gdn else _ATTN_SUFFIX
        if suffix in table:
            plan.consumed[key] = table[suffix].format(i=i)
            continue
        wrong = _ATTN_SUFFIX if is_gdn else _GDN_SUFFIX
        if suffix in wrong:
            raise KeyError(
                f"{key} is a {'full-attention' if is_gdn else 'GDN'} tensor but "
                f"layer_types[{i}] == {config.layer_types[i]}")
        raise KeyError(f"unrecognised checkpoint key (no plan): {key}")
    return plan


def plan_from_index(index_path: str, config: Qwen35Config) -> CheckpointPlan:
    with open(index_path) as f:
        index = json.load(f)
    return plan_checkpoint(index["weight_map"].keys(), config)


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------
class Qwen35WeightLoader:
    """Materialise the runtime tensors the Qwen3.5 builder attaches.

    `qk_dense_path` selects how the RoPE-permuted attention projections are
    represented — see `transforms.permute_fp8_blockscale_rows` for why this is
    a choice at all:

      * `"fp8"`      (default): permute the fp8 rows, rescale only the rows
        whose 128-block scale changes. Keeps the whole dense path on the
        preserved-fp32-block-scale GEMM (`v1-architecture.md` §6.2).
      * `"bf16"`: exact block-dequant, permute, one bf16 rounding, plain bf16
        GEMM. `v1-architecture.md` §6.2 allows this only as a debugging
        scaffold.
    """

    def __init__(self, snapshot_dir: str, config: Optional[Qwen35Config] = None,
                 *, device: str = "cuda", qk_dense_path: str = "fp8",
                 layer_filter: Optional[List[int]] = None):
        if qk_dense_path not in ("fp8", "bf16"):
            raise ValueError(f"qk_dense_path must be 'fp8' or 'bf16', got {qk_dense_path}")
        self.snapshot_dir = snapshot_dir
        self.device = device
        self.qk_dense_path = qk_dense_path
        self.config = config or Qwen35Config.from_json(
            os.path.join(snapshot_dir, "config.json"))
        self.index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
        with open(self.index_path) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self.plan = plan_checkpoint(self.weight_map.keys(), self.config)
        # None -> every layer; used by the single-layer tests to load one GDN
        # layer and one full-attention layer instead of all 40.
        self.layer_filter = layer_filter
        self._handles: Dict[str, object] = {}
        self._reads: Dict[str, int] = {}
        self.transform_stats: Dict[str, dict] = {}

    # ---- checkpoint access ---------------------------------------------
    def _get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[key]
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.snapshot_dir, shard),
                          framework="pt", device=self.device)
            self._handles[shard] = h
        self._reads[key] = self._reads.get(key, 0) + 1
        return h.get_tensor(key)

    def _lang(self, suffix: str) -> torch.Tensor:
        return self._get(LANG + suffix)

    def _layer(self, i: int, suffix: str) -> torch.Tensor:
        return self._lang(f"layers.{i}.{suffix}")

    def _scale(self, i: int, suffix: str) -> torch.Tensor:
        return self._layer(i, suffix + ".weight_scale_inv").float()

    # ---- per-layer builders --------------------------------------------
    def _load_gdn(self, i: int, out: Dict[str, torch.Tensor]) -> None:
        c = self.config
        out[f"layer_{i}_gdn_in_proj_qkv"] = self._layer(i, "linear_attn.in_proj_qkv.weight").contiguous()
        out[f"layer_{i}_gdn_in_proj_qkv_scale"] = self._scale(i, "linear_attn.in_proj_qkv").contiguous()
        out[f"layer_{i}_gdn_in_proj_z"] = self._layer(i, "linear_attn.in_proj_z.weight").contiguous()
        out[f"layer_{i}_gdn_in_proj_z_scale"] = self._scale(i, "linear_attn.in_proj_z").contiguous()
        out[f"layer_{i}_gdn_in_proj_ba"] = T.concat_ba(
            self._layer(i, "linear_attn.in_proj_b.weight"),
            self._layer(i, "linear_attn.in_proj_a.weight"))
        out[f"layer_{i}_gdn_conv1d"] = T.conv1d_weight(
            self._layer(i, "linear_attn.conv1d.weight"))
        out[f"layer_{i}_gdn_alog_dtbias"] = T.pack_alog_dtbias(
            self._layer(i, "linear_attn.A_log"),
            self._layer(i, "linear_attn.dt_bias"))
        # the ONE non-Gemma norm: ones-init, fp32, no `+1`
        out[f"layer_{i}_gdn_norm"] = T.no_fold_norm(
            self._layer(i, "linear_attn.norm.weight"))
        out[f"layer_{i}_gdn_out_proj"] = self._layer(i, "linear_attn.out_proj.weight").contiguous()
        out[f"layer_{i}_gdn_out_proj_scale"] = self._scale(i, "linear_attn.out_proj").contiguous()
        assert out[f"layer_{i}_gdn_in_proj_qkv"].shape == (c.conv_dim, c.hidden_size)
        assert out[f"layer_{i}_gdn_norm"].shape == (c.linear_value_head_dim,)

    def _load_attn(self, i: int, out: Dict[str, torch.Tensor]) -> None:
        c = self.config
        q = self._layer(i, "self_attn.q_proj.weight")
        k = self._layer(i, "self_attn.k_proj.weight")
        v = self._layer(i, "self_attn.v_proj.weight")
        qs = self._scale(i, "self_attn.q_proj")
        ks = self._scale(i, "self_attn.k_proj")
        vs = self._scale(i, "self_attn.v_proj")
        q_map = T.q_proj_dest_from_src(c.num_attention_heads, c.head_dim,
                                       c.rotary_dim, q.device)
        k_map = T.k_proj_dest_from_src(c.num_key_value_heads, c.head_dim,
                                       c.rotary_dim, k.device)

        if self.qk_dense_path == "fp8":
            qp, qsp, q_stats = T.permute_fp8_blockscale_rows(q, qs, q_map)
            kp, ksp, k_stats = T.permute_fp8_blockscale_rows(k, ks, k_map)
            out[f"layer_{i}_qkvg_proj"] = T.fuse_qkvg(qp, kp, v, c.num_key_value_heads)
            out[f"layer_{i}_qkvg_proj_scale"] = T.fuse_qkvg_scale(
                qsp, ksp, vs, c.num_key_value_heads)
            self.transform_stats[f"layer_{i}_q_proj_permute"] = q_stats
            self.transform_stats[f"layer_{i}_k_proj_permute"] = k_stats
        else:
            out[f"layer_{i}_qkvg_proj"] = T.fuse_qkvg(
                T.permute_bf16_rows(q, qs, q_map),
                T.permute_bf16_rows(k, ks, k_map),
                T.dequant_blockscale(v, vs).to(BF16),
                c.num_key_value_heads)

        # Gemma fold FIRST (fp32 add), then the same head_dim permutation, then
        # the bf16 store MPK's kernel signature forces (`transforms.gemma_fold`).
        out[f"layer_{i}_q_norm"] = T.permute_head_dim_norm(
            T.gemma_fold(self._layer(i, "self_attn.q_norm.weight")),
            c.head_dim, c.rotary_dim)
        out[f"layer_{i}_k_norm"] = T.permute_head_dim_norm(
            T.gemma_fold(self._layer(i, "self_attn.k_norm.weight")),
            c.head_dim, c.rotary_dim)
        out[f"layer_{i}_o_proj"] = self._layer(i, "self_attn.o_proj.weight").contiguous()
        out[f"layer_{i}_o_proj_scale"] = self._scale(i, "self_attn.o_proj").contiguous()
        assert out[f"layer_{i}_qkvg_proj"].shape == (c.qkvg_dim, c.hidden_size)

    def _load_moe(self, i: int, out: Dict[str, torch.Tensor]) -> None:
        c = self.config
        out[f"layer_{i}_router"] = self._layer(i, "mlp.gate.weight").to(BF16).contiguous()
        out[f"layer_{i}_shared_expert_gate"] = (
            self._layer(i, "mlp.shared_expert_gate.weight").to(BF16).contiguous())

        split = c.shared_expert_intermediate_size // T.BLOCK
        out[f"layer_{i}_shared_gate_up"] = T.fuse_gate_up(
            self._layer(i, "mlp.shared_expert.gate_proj.weight"),
            self._layer(i, "mlp.shared_expert.up_proj.weight"), split)
        out[f"layer_{i}_shared_gate_up_scale"] = T.fuse_gate_up(
            self._scale(i, "mlp.shared_expert.gate_proj"),
            self._scale(i, "mlp.shared_expert.up_proj"), split).float().contiguous()
        out[f"layer_{i}_shared_down"] = (
            self._layer(i, "mlp.shared_expert.down_proj.weight").contiguous())
        out[f"layer_{i}_shared_down_scale"] = self._scale(i, "mlp.shared_expert.down_proj").contiguous()

        gates, ups, downs, gs, us, ds = [], [], [], [], [], []
        for e in range(c.num_experts):
            p = f"mlp.experts.{e}."
            gates.append(self._layer(i, p + "gate_proj.weight"))
            ups.append(self._layer(i, p + "up_proj.weight"))
            downs.append(self._layer(i, p + "down_proj.weight"))
            gs.append(self._scale(i, p + "gate_proj"))
            us.append(self._scale(i, p + "up_proj"))
            ds.append(self._scale(i, p + "down_proj"))
        out[f"layer_{i}_w13"] = T.stack_expert_w13(gates, ups)
        out[f"layer_{i}_w13_scale"] = T.stack_expert_w13_scale(gs, us)
        out[f"layer_{i}_w2"] = T.stack_expert_w2(downs)
        out[f"layer_{i}_w2_scale"] = T.stack_expert_w2_scale(ds)
        del gates, ups, downs, gs, us, ds
        assert out[f"layer_{i}_w13"].shape == (
            c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size)
        assert out[f"layer_{i}_w2"].shape == (
            c.num_experts, c.hidden_size, c.moe_intermediate_size)

    # ---- entry point ----------------------------------------------------
    def load(self, *, with_model_level: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """Materialise the runtime tensors.

        `with_model_level` defaults to "yes for a full load, no for a partial
        one": a `layer_filter` load is by definition a single-layer probe, which
        never touches the 0.95 GB embedding or lm_head.
        """
        c = self.config
        out: Dict[str, torch.Tensor] = {}
        layers = self.layer_filter if self.layer_filter is not None else range(c.num_layers)
        if with_model_level is None:
            with_model_level = self.layer_filter is None

        if with_model_level:
            out["embed_tokens"] = self._lang("embed_tokens.weight").to(BF16).contiguous()
            out["model_norm"] = T.gemma_fold(self._lang("norm.weight"))
            out["lm_head"] = self._get("lm_head.weight").to(BF16).contiguous()
            assert out["embed_tokens"].shape == (c.vocab_size, c.hidden_size)
            assert out["lm_head"].shape == (c.vocab_size, c.hidden_size)

        for i in layers:
            out[f"layer_{i}_input_layernorm"] = T.gemma_fold(
                self._layer(i, "input_layernorm.weight"))
            out[f"layer_{i}_post_attention_layernorm"] = T.gemma_fold(
                self._layer(i, "post_attention_layernorm.weight"))
            if c.layer_types[i] == "linear_attention":
                self._load_gdn(i, out)
            else:
                self._load_attn(i, out)
            self._load_moe(i, out)

        if self.layer_filter is None:
            self.assert_round_trip()
        return out

    def assert_round_trip(self) -> Dict[str, int]:
        """Every planned key read exactly once; nothing outside the plan read.

        This is the acceptance property of the loader: a weight that is never
        read is a silently-zero projection, and a weight read twice usually
        means two runtime slots disagree about who owns it.
        """
        planned = set(self.plan.consumed)
        read = set(self._reads)
        missing = sorted(planned - read)
        extra = sorted(read - planned)
        twice = sorted(k for k, n in self._reads.items() if n != 1)
        if missing or extra or twice:
            raise AssertionError(
                f"loader round-trip failed: {len(missing)} planned-but-unread "
                f"(e.g. {missing[:3]}), {len(extra)} read-but-unplanned "
                f"(e.g. {extra[:3]}), {len(twice)} read more than once "
                f"(e.g. {twice[:3]})")
        return {"read_once": len(read), **self.plan.summary()}

    def inventory_report(self) -> Dict[str, object]:
        rep: Dict[str, object] = dict(self.plan.summary())
        rep["shards"] = len(set(self.weight_map.values()))
        rep["reads"] = len(self._reads)
        rep["max_reads_per_key"] = max(self._reads.values()) if self._reads else 0
        rep["qk_dense_path"] = self.qk_dense_path
        return rep


def resolve_snapshot(model_name: Optional[str] = None,
                     model_path: Optional[str] = None) -> str:
    """Local snapshot directory for the checkpoint (never downloads weights
    when the HF cache already has them)."""
    if model_path:
        return model_path
    if not model_name:
        raise ValueError("model_name or model_path is required")
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name)
