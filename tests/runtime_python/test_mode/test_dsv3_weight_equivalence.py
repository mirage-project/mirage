"""CPU-only weight-equivalence test for the DeepSeek V3 streaming loader.

This proves ``DeepseekV3ForCausalLM.load_weights`` (in
``python/mirage/mpk/models/deepseek_v3/modeling.py``) loads REAL checkpoint
weights CORRECTLY by comparing the loaded ``named_parameters()`` against an
independent ground truth computed directly from the SAME safetensors using
the original, proven ``demo/deepseek_v3/models/convert.py`` math
(``dequantize_fp8`` / ``is_fp8`` / ``absorb_kv_into_q`` /
``find_scale_for_weight``).

The whole pipeline (fp8->bf16 dequant, EP expert stacking at ep_size=1, name
remaps, MLA absorption q_b<-kv_b + W_UV->o_proj fusion) is pure tensor ops:
no compile scope, no GPU. We build a reduced 4-layer model (dense layers 0-2 +
first MoE layer 3) so it exercises MLA + dense MLP + MoE experts + router +
shared experts + fp8 while keeping host RAM bounded.

Run CPU-only:

    export CUDA_VISIBLE_DEVICES=""
    python tests/runtime_python/test_mode/test_dsv3_weight_equivalence.py
"""

import os
import sys

import torch
from safetensors import safe_open

# --- import paths ---------------------------------------------------------
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_PKG_ROOT = os.path.join(_REPO_ROOT, "python")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
# convert.py lives in the demo tree; add it so we import the PROVEN helpers
# rather than re-deriving them.
_CONVERT_DIR = os.path.join(_REPO_ROOT, "demo", "deepseek_v3", "models")
if _CONVERT_DIR not in sys.path:
    sys.path.insert(0, _CONVERT_DIR)

import convert  # noqa: E402  (the proven ground-truth math)
from mirage.mpk.models.deepseek_v3.modeling import (  # noqa: E402
    DeepseekV3ForCausalLM,
)
from mirage.mpk.weight_loader import (  # noqa: E402
    find_safetensors_files,
    safetensors_weights_iterator,
)

MODEL_PATH = "/mnt/shared/models/DeepSeek-V3"
NUM_LAYERS = 4  # dense layers 0-2 + first MoE layer 3


# ---------------------------------------------------------------------------
# Ground-truth helpers built directly on top of convert.py.
# ---------------------------------------------------------------------------


class GroundTruth:
    """Reads raw tensors from the on-disk safetensors and dequantizes via the
    PROVEN ``convert.py`` helpers (the same math the loader transcribed)."""

    def __init__(self, files):
        # Map every (non-scale) weight name and every scale name to the file
        # that holds it, so we can lazily ``safe_open`` + ``get_tensor`` only
        # what each check needs (host RAM stays bounded).
        self._name_to_file = {}
        self._scale_names = set()
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as fh:
                for name in fh.keys():
                    self._name_to_file[name] = path
                    if name.endswith("_scale_inv") or name.endswith("_scale"):
                        self._scale_names.add(name)

    def _raw(self, name):
        path = self._name_to_file.get(name)
        if path is None:
            raise KeyError(f"ground-truth: {name!r} not in checkpoint")
        with safe_open(path, framework="pt", device="cpu") as fh:
            return fh.get_tensor(name)

    def _scales_dict_for(self, name):
        """convert.find_scale_for_weight wants a {scale_name: tensor} dict;
        build a tiny one holding just the candidate scale partners for ``name``."""
        d = {}
        for suffix in ("_scale_inv", "_scale"):
            cand = name + suffix
            if cand in self._scale_names:
                d[cand] = self._raw(cand)
            if name.endswith(".weight"):
                cand2 = name[: -len(".weight")] + suffix
                if cand2 in self._scale_names:
                    d[cand2] = self._raw(cand2)
        return d

    def dequant(self, name):
        """Ground-truth bf16 tensor for HF key ``name`` via convert.py math:
        if fp8, pair with its scale and dequantize; else cast bf16."""
        w = self._raw(name)
        if convert.is_fp8(w):
            scale = convert.find_scale_for_weight(name, self._scales_dict_for(name))
            return convert.dequantize_fp8(w, scale, target_dtype=torch.bfloat16)
        return w.to(torch.bfloat16)

    def raw_fp32(self, name):
        """Raw tensor cast to fp32 (for the router bias exactness check)."""
        return self._raw(name).to(torch.float32)


# ---------------------------------------------------------------------------
# Comparison harness.
# ---------------------------------------------------------------------------

_results = []  # (label, passed, max_abs_diff_or_msg)


def _check(label, got, expected, *, exact=True, atol=0.0, rtol=0.0):
    """Compare two tensors; record PASS/FAIL with max abs diff."""
    if got.shape != expected.shape:
        _results.append((label, False, f"SHAPE {tuple(got.shape)} != {tuple(expected.shape)}"))
        return False
    g = got.detach().to(torch.float32)
    e = expected.detach().to(torch.float32)
    max_abs = (g - e).abs().max().item() if g.numel() else 0.0
    if exact:
        ok = torch.equal(got.detach(), expected.detach())
    else:
        ok = torch.allclose(g, e, atol=atol, rtol=rtol)
    _results.append((label, ok, max_abs))
    return ok


def _check_dtype(label, got, expected_dtype):
    ok = got.dtype == expected_dtype
    _results.append((label, ok, f"dtype={got.dtype} (want {expected_dtype})"))
    return ok


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    # Hard CPU-only: GPUs may be busy with another user.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_grad_enabled(False)

    from transformers import AutoConfig

    print(f"[setup] loading HF config from {MODEL_PATH}")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config.num_hidden_layers = NUM_LAYERS
    print(
        f"[setup] config: hidden={config.hidden_size} heads={config.num_attention_heads} "
        f"q_lora={config.q_lora_rank} kv_lora={config.kv_lora_rank} "
        f"qk_nope={config.qk_nope_head_dim} qk_rope={config.qk_rope_head_dim} "
        f"v_head_dim={config.v_head_dim} n_experts={config.n_routed_experts} "
        f"moe_inter={config.moe_intermediate_size} first_moe={config.first_k_dense_replace} "
        f"layers={config.num_hidden_layers}"
    )

    # Params dict for convert.absorb_kv_into_q (HF-native dims).
    cvt_params = {
        "num_heads": config.num_attention_heads,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "kv_lora_rank": config.kv_lora_rank,
        "v_head_dim": config.v_head_dim,
        "q_lora_rank": config.q_lora_rank,
    }

    print("[setup] constructing DeepseekV3ForCausalLM on CPU")
    # Construct on CPU only; do NOT move to cuda and do NOT blanket-cast to
    # bf16. The catalog params are created with their intended dtypes (most
    # are fp32 containers that load_weights fills with bf16-valued data via
    # copy_; the router e_score_correction_bias is fp32 and MUST stay fp32 —
    # a blanket .to(bf16) would clobber that and defeat the no-round-trip
    # property the loader is designed to preserve).
    model = DeepseekV3ForCausalLM(config)

    print("[load] streaming real checkpoint through load_weights ...")
    files = find_safetensors_files(MODEL_PATH)
    print(f"[load] {len(files)} safetensors files")
    # If load_weights raises, that's a REAL loading bug — let it propagate
    # (the traceback IS the report).
    model.load_weights(safetensors_weights_iterator(files))
    print("[load] load_weights completed cleanly on real weights")

    params = dict(model.named_parameters())

    def P(name):
        if name not in params:
            avail = [k for k in params if name.split(".")[-1] in k][:8]
            raise KeyError(f"param {name!r} not found. similar: {avail}")
        return params[name].data

    print("[gt] indexing safetensors for ground truth ...")
    gt = GroundTruth(files)

    inter = config.moe_intermediate_size
    n_heads = config.num_attention_heads
    kv_lora = config.kv_lora_rank
    v_head_dim = config.v_head_dim

    # ------------------------------------------------------------------
    # Global tensors.
    # ------------------------------------------------------------------
    _check("embed_tokens.weight", P("model.embed_tokens.weight"),
           gt.dequant("model.embed_tokens.weight"))
    _check("model.norm.weight", P("model.norm.weight"),
           gt.dequant("model.norm.weight"))
    _check("lm_head.weight", P("lm_head.weight"),
           gt.dequant("lm_head.weight"))

    # ------------------------------------------------------------------
    # Layer 0 MLA: directly-written projections + layernorms.
    # ------------------------------------------------------------------
    _check("L0 mla q_a_proj_weight",
           P("model.layers.0.self_attn.q_a_proj_weight"),
           gt.dequant("model.layers.0.self_attn.q_a_proj.weight"))
    _check("L0 mla kv_a_proj_with_mqa_weight",
           P("model.layers.0.self_attn.kv_a_proj_with_mqa_weight"),
           gt.dequant("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"))
    _check("L0 mla q_a_layernorm",
           P("model.layers.0.self_attn.q_a_layernorm"),
           gt.dequant("model.layers.0.self_attn.q_a_layernorm.weight"))
    _check("L0 mla kv_a_layernorm",
           P("model.layers.0.self_attn.kv_a_layernorm"),
           gt.dequant("model.layers.0.self_attn.kv_a_layernorm.weight"))

    # ------------------------------------------------------------------
    # Layer 0 MLA ABSORPTION (the critical checks).
    # ------------------------------------------------------------------
    # q_b absorption: ground truth via convert.absorb_kv_into_q on the raw
    # dequant'd q_b / kv_b safetensors.
    raw_q_b = gt.dequant("model.layers.0.self_attn.q_b_proj.weight")
    raw_kv_b = gt.dequant("model.layers.0.self_attn.kv_b_proj.weight")
    gt_q_b = convert.absorb_kv_into_q(raw_q_b, raw_kv_b, cvt_params).to(torch.bfloat16)
    # Absorption is bmm in fp32 then bf16 cast in BOTH loader and convert; the
    # loader uses an inlined copy of the SAME math, so expect exact equality.
    _check("L0 mla q_b_proj_weight (ABSORBED kv_b->q_b)",
           P("model.layers.0.self_attn.q_b_proj_weight"), gt_q_b, exact=True)

    # o_proj W_UV fusion: replicate demo_new.py / modeling.process_weights math.
    raw_o = gt.dequant("model.layers.0.self_attn.o_proj.weight")  # (hidden, H*v_head_dim)
    hidden = raw_o.shape[0]
    W_UV = raw_kv_b.reshape(
        n_heads, config.qk_nope_head_dim + v_head_dim, kv_lora
    )[:, config.qk_nope_head_dim:, :]  # (H, v_head_dim, kv_lora)
    o_fused = torch.einsum(
        "dhn,hnk->dhk",
        raw_o.reshape(hidden, n_heads, v_head_dim).float(),
        W_UV.float(),
    )  # (hidden, H, kv_lora)
    gt_o = o_fused.reshape(hidden, n_heads * kv_lora).to(torch.bfloat16)
    _check("L0 mla o_proj_weight (W_UV FUSED)",
           P("model.layers.0.self_attn.o_proj_weight"), gt_o, exact=True)

    # ------------------------------------------------------------------
    # Dense MLP layers 0/1/2.
    # ------------------------------------------------------------------
    for li in (0, 1, 2):
        for proj in ("gate", "up", "down"):
            _check(f"L{li} dense mlp {proj}_proj_weight",
                   P(f"model.layers.{li}.mlp.{proj}_proj_weight"),
                   gt.dequant(f"model.layers.{li}.mlp.{proj}_proj.weight"))

    # ------------------------------------------------------------------
    # Layer 3 MoE router gate + bias.
    # ------------------------------------------------------------------
    _check("L3 moe router gate_weight",
           P("model.layers.3.mlp.gate_weight"),
           gt.dequant("model.layers.3.mlp.gate.weight"))

    # Router bias MUST be fp32 (no bf16 round-trip) and exactly the raw fp32.
    bias_param = P("model.layers.3.mlp.routing.bias")
    _check_dtype("L3 moe routing.bias dtype is fp32", bias_param, torch.float32)
    gt_bias = gt.raw_fp32("model.layers.3.mlp.gate.e_score_correction_bias")
    _check("L3 moe routing.bias (fp32 exact)", bias_param, gt_bias, exact=True)

    # ------------------------------------------------------------------
    # Layer 3 MoE routed experts (e=0, e=1): stacked into w13 / w2.
    # ------------------------------------------------------------------
    w13 = P("model.layers.3.mlp.w13.weight")  # (E_local, 2*inter, hidden)
    w2 = P("model.layers.3.mlp.w2.weight")    # (E_local, hidden, inter)
    for e in (0, 1):
        _check(f"L3 moe expert {e} w13[:inter]==gate",
               w13[e, :inter],
               gt.dequant(f"model.layers.3.mlp.experts.{e}.gate_proj.weight"))
        _check(f"L3 moe expert {e} w13[inter:]==up",
               w13[e, inter:],
               gt.dequant(f"model.layers.3.mlp.experts.{e}.up_proj.weight"))
        _check(f"L3 moe expert {e} w2==down",
               w2[e],
               gt.dequant(f"model.layers.3.mlp.experts.{e}.down_proj.weight"))

    # ------------------------------------------------------------------
    # Layer 3 shared experts.
    # ------------------------------------------------------------------
    for proj in ("gate", "up", "down"):
        _check(f"L3 moe shared {proj} proj",
               P(f"model.layers.3.mlp.shared_{proj}_proj_weight"),
               gt.dequant(f"model.layers.3.mlp.shared_experts.{proj}_proj.weight"))

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'CHECK':<48}{'RESULT':<8}{'max|diff| / note'}")
    print("-" * 78)
    n_fail = 0
    for label, ok, info in _results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        info_s = info if isinstance(info, str) else f"{info:.3e}"
        print(f"{label:<48}{status:<8}{info_s}")
    print("=" * 78)
    total = len(_results)
    print(f"{total - n_fail}/{total} checks passed")

    if n_fail:
        print(f"\nFAIL: {n_fail} weight-equivalence check(s) mismatched.")
        sys.exit(1)
    print("\nPASS: all weight-equivalence checks matched convert.py ground truth.")


if __name__ == "__main__":
    main()
