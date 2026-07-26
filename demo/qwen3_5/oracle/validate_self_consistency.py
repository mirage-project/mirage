#!/usr/bin/env python3
"""Self-consistency validator for the M2-I3 HF numerical oracle (probe P6's acceptance centerpiece).

For every dumped op, recompute it in torch FROM ITS OWN DUMPED INPUTS (via `pytorch_reference.py`)
and compare to the dumped OUTPUT. This catches hook misplacement: a hook capturing the wrong
tensor (e.g. pre-RoPE q mislabeled as post-RoPE, or a view before vs. after a reshape) produces a
recompute that is STRUCTURALLY wrong and fails even the loosest tolerance below; a correctly
placed hook passes at the tolerance appropriate to that op (see "Dtype and tolerance policy" in
README.md, summarized in TIGHT / LOOSE_FP8 below).

Usage:
    python validate_self_consistency.py --dump-dir ~/mpk-qwen35/oracle-work/dumps --mode both
Exit code 0 iff every non-skipped check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import pytorch_reference as ref

# TIGHT: ops with no fp8 quantization in the loop (norms, RoPE, softmax/topk, elementwise combine,
# reshape/split) -- the recompute is the literally-same formula on the literally-same dumped
# input, so agreement should be near float32 machine precision; a little slack covers GPU
# reduction-order nondeterminism.
TIGHT = dict(atol=5e-3, rtol=5e-3)

# LOOSE_FP8: dense/expert GEMMs whose real forward path is HF's fp8 x fp8 kernel (dynamic
# per-token-group activation quantization THEN a genuine fp8 MMA); the recompute here dequantizes
# the WEIGHT only and matmuls against the full-precision (unquantized) dumped activation, so it
# cannot reproduce the activation-side quantization noise. A quantization-scale-sized diff is
# EXPECTED here, not a bug -- see README.md.
LOOSE_FP8 = dict(atol=2.0, rtol=0.25)

# LOOSE_KERNEL: ops whose real backend is confirmed (empirically, via `attention_mask_used_is_none`
# in runtime_diagnostics.json) to differ in FORMULATION from the literal eager transcription --
# specifically `attn.core_attn_out`, where the real run used `attention_mask=None` (an implicit
# `is_causal=True`-style backend, e.g. SDPA/flash-attention) rather than materializing a mask for
# an eager softmax. Empirical evidence this is benign, not a bug: BEFORE reconstructing causal
# masking in the recompute, max_abs error was 2.56 (prefill) / 0.017 (decode) with a clear
# structural signature; AFTER adding the causal mask, prefill collapsed 100x to 0.023 while decode
# (whose single query needs no masking either way) stayed at 0.017 -- i.e. once the one real bug
# (missing causal treatment) was fixed, both modes converge to the SAME small residual, consistent
# with ordinary fused-kernel-vs-naive-matmul accumulation differences in bf16, not a second bug.
LOOSE_KERNEL = dict(atol=0.05, rtol=0.10)


class Loader:
    def __init__(self, dump_dir: Path, mode: str):
        self.dump_dir = dump_dir
        self.mode = mode
        manifest_path = dump_dir / mode / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def has(self, name: str) -> bool:
        return name in self.manifest["tensors"]

    def get(self, name: str) -> torch.Tensor:
        entry = self.manifest["tensors"].get(name)
        if entry is None:
            raise KeyError(f"[{self.mode}] tensor {name!r} not in manifest")
        return torch.load(self.dump_dir / entry["file"], map_location="cpu", weights_only=False)

    def meta(self, name: str, default=None):
        return self.manifest["meta"].get(name, default)


class Report:
    def __init__(self):
        self.rows: list[tuple[str, object, float, float, str]] = []

    def check(self, op_name: str, recomputed: torch.Tensor, dumped: torch.Tensor, tol: dict, note: str = ""):
        recomputed = recomputed.to(torch.float32)
        dumped = dumped.to(torch.float32)
        if recomputed.shape != dumped.shape:
            self.rows.append((op_name, False, float("nan"), float("nan"),
                               f"SHAPE MISMATCH recompute={tuple(recomputed.shape)} dumped={tuple(dumped.shape)} {note}"))
            return False
        diff = (recomputed - dumped).abs()
        max_abs = diff.max().item()
        denom = dumped.abs().clamp_min(1e-6)
        max_rel = (diff / denom).max().item()
        ok = bool(torch.allclose(recomputed, dumped, atol=tol["atol"], rtol=tol["rtol"]))
        self.rows.append((op_name, ok, max_abs, max_rel, note))
        return ok

    def check_exact_int(self, op_name: str, recomputed: torch.Tensor, dumped: torch.Tensor, note: str = ""):
        ok = bool(torch.equal(recomputed, dumped))
        self.rows.append((op_name, ok, 0.0 if ok else -1.0, 0.0 if ok else -1.0, note))
        return ok

    def skip(self, op_name: str, reason: str):
        self.rows.append((op_name, None, float("nan"), float("nan"), f"SKIPPED: {reason}"))

    def print_table(self, mode: str):
        print(f"\n=== self-consistency report: mode={mode} ===")
        print(f"{'op':45s} {'status':8s} {'max_abs':>12s} {'max_rel':>12s}  note")
        for name, ok, max_abs, max_rel, note in self.rows:
            status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            abs_s = "n/a" if max_abs != max_abs and ok is None else f"{max_abs:.4g}"
            rel_s = "n/a" if max_rel != max_rel and ok is None else f"{max_rel:.4g}"
            print(f"{name:45s} {status:8s} {abs_s:>12s} {rel_s:>12s}  {note}")
        n_pass = sum(1 for r in self.rows if r[1] is True)
        n_fail = sum(1 for r in self.rows if r[1] is False)
        n_skip = sum(1 for r in self.rows if r[1] is None)
        print(f"--- {mode}: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP (of {len(self.rows)}) ---")

    def all_passed(self) -> bool:
        return all(r[1] is not False for r in self.rows)


def check_gdn(ld: Loader, rep: Report, prefix: str, mode: str):
    if not ld.has(f"{prefix}.layer_input"):
        rep.skip(f"{prefix}.*", "gdn dumps absent in this manifest")
        return

    layer_input = ld.get(f"{prefix}.layer_input")
    w_ln = ld.get(f"{prefix}.input_layernorm.__weight.weight")
    ln_out = ld.get(f"{prefix}.input_layernorm.output")
    ln_in = ld.get(f"{prefix}.input_layernorm.input")
    rep.check(f"{prefix}.input_layernorm", ref.gemma_rmsnorm(ln_in, w_ln), ln_out, TIGHT)

    w_qkv = ld.get(f"{prefix}.__weight.in_proj_qkv")
    s_qkv = ld.get(f"{prefix}.__weight.in_proj_qkv_scale_inv") if ld.has(f"{prefix}.__weight.in_proj_qkv_scale_inv") else None
    qkv_out = ld.get(f"{prefix}.qkv_proj_out")
    if s_qkv is not None:
        rep.check(f"{prefix}.qkv_proj_out(fp8 dequant)", ref.fp8_linear_recompute(layer_input, w_qkv, s_qkv), qkv_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.qkv_proj_out(bf16)", F.linear(layer_input, w_qkv), qkv_out, TIGHT)

    w_z = ld.get(f"{prefix}.__weight.in_proj_z")
    s_z = ld.get(f"{prefix}.__weight.in_proj_z_scale_inv") if ld.has(f"{prefix}.__weight.in_proj_z_scale_inv") else None
    z_out = ld.get(f"{prefix}.z_proj_out")
    if s_z is not None:
        rep.check(f"{prefix}.z_proj_out(fp8 dequant)", ref.fp8_linear_recompute(layer_input, w_z, s_z), z_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.z_proj_out(bf16)", F.linear(layer_input, w_z), z_out, TIGHT)

    w_b = ld.get(f"{prefix}.__weight.in_proj_b")
    w_a = ld.get(f"{prefix}.__weight.in_proj_a")
    b_out = ld.get(f"{prefix}.b_proj_out")
    a_out = ld.get(f"{prefix}.a_proj_out")
    rep.check(f"{prefix}.b_proj_out(bf16)", F.linear(layer_input, w_b), b_out, TIGHT)
    rep.check(f"{prefix}.a_proj_out(bf16)", F.linear(layer_input, w_a), a_out, TIGHT)

    conv_in = ld.get(f"{prefix}.conv_in")
    conv_w = ld.get(f"{prefix}.__weight.conv1d_weight")
    conv_out = ld.get(f"{prefix}.conv_out")
    if mode == "decode" and ld.has(f"{prefix}.conv_state_before"):
        conv_state_before = ld.get(f"{prefix}.conv_state_before").clone()
        recomputed = ref.torch_causal_conv1d_update(conv_in.clone(), conv_state_before, conv_w.squeeze(1), None, "silu")
        rep.check(f"{prefix}.conv_out(decode, causal_conv1d_update)", recomputed, conv_out, TIGHT)
        if ld.has(f"{prefix}.conv_state_after"):
            rep.check(f"{prefix}.conv_state_after(decode)", conv_state_before, ld.get(f"{prefix}.conv_state_after"), TIGHT)
    else:
        seq_len = conv_in.shape[-1]
        recomputed = ref.torch_causal_conv1d_prefill(conv_in, conv_w, None, seq_len)
        rep.check(f"{prefix}.conv_out(prefill, plain-conv1d)", recomputed, conv_out, TIGHT)

    beta = ld.get(f"{prefix}.beta")
    b_recompute = ref.gdn_beta(b_out)
    rep.check(f"{prefix}.beta", b_recompute, beta, TIGHT)

    decay_g = ld.get(f"{prefix}.decay_g")
    a_log = ld.get(f"{prefix}.__weight.A_log")
    dt_bias = ld.get(f"{prefix}.__weight.dt_bias")
    rep.check(f"{prefix}.decay_g", ref.gdn_decay_g(a_out, a_log, dt_bias), decay_g, TIGHT)

    q = ld.get(f"{prefix}.q_split")
    k = ld.get(f"{prefix}.k_split")
    v = ld.get(f"{prefix}.v_split")
    num_v_heads, num_k_heads = v.shape[2], q.shape[2]
    rep_factor = num_v_heads // num_k_heads
    if rep_factor > 1:
        q_r = q.repeat_interleave(rep_factor, dim=2)
        k_r = k.repeat_interleave(rep_factor, dim=2)
    else:
        q_r, k_r = q, k
    core_attn_out = ld.get(f"{prefix}.core_attn_out")
    if mode == "decode" and ld.has(f"{prefix}.core_state_before"):
        s_before = ld.get(f"{prefix}.core_state_before")
        out_r, s_after_r = ref.torch_recurrent_gated_delta_rule(
            q_r, k_r, v, decay_g, beta, initial_state=s_before, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        rep.check(f"{prefix}.core_attn_out(decode, recurrent_gated_delta_rule)", out_r, core_attn_out, TIGHT)
        if ld.has(f"{prefix}.core_state_after"):
            rep.check(f"{prefix}.core_state_after(decode)", s_after_r, ld.get(f"{prefix}.core_state_after"), TIGHT)
    else:
        out_r, s_after_r = ref.torch_chunk_gated_delta_rule(
            q_r, k_r, v, decay_g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        rep.check(f"{prefix}.core_attn_out(prefill, chunk_gated_delta_rule)", out_r, core_attn_out, TIGHT)
        if ld.has(f"{prefix}.core_state_after"):
            rep.check(f"{prefix}.core_state_after(prefill)", s_after_r, ld.get(f"{prefix}.core_state_after"), TIGHT)

    z_flat = ld.get(f"{prefix}.z_proj_out").reshape(-1, v.shape[-1])
    core_attn_out_flat = core_attn_out.reshape(-1, v.shape[-1])
    norm_w = ld.get(f"{prefix}.__weight.norm_weight")
    gated = ld.get(f"{prefix}.gated_norm_out")
    rep.check(f"{prefix}.gated_norm_out", ref.gdn_gated_rmsnorm(core_attn_out_flat, z_flat, norm_w), gated, TIGHT)

    w_out = ld.get(f"{prefix}.__weight.out_proj")
    s_out = ld.get(f"{prefix}.__weight.out_proj_scale_inv") if ld.has(f"{prefix}.__weight.out_proj_scale_inv") else None
    out_proj_out = ld.get(f"{prefix}.out_proj_out")
    # `gated` (gdn.gated_norm_out) is dumped BEFORE the model's own `.reshape(batch,seq,-1)` --
    # its native shape is [B*T*num_v_heads, head_v_dim]; regroup heads back into [B*T, value_dim]
    # (out_proj's actual input shape) using out_proj_out's own leading dims as the B,T reference.
    out_dim = w_out.shape[0]
    n_tokens = out_proj_out.numel() // out_dim
    gated_flat = gated.reshape(n_tokens, -1)
    out_flat_dumped = out_proj_out.reshape(-1, out_dim)
    if s_out is not None:
        rep.check(f"{prefix}.out_proj_out(fp8 dequant)", ref.fp8_linear_recompute(gated_flat, w_out, s_out), out_flat_dumped, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.out_proj_out(bf16)", F.linear(gated_flat, w_out), out_flat_dumped, TIGHT)

    if ld.has(f"{prefix}.post_attention_layernorm.input"):
        pln_in = ld.get(f"{prefix}.post_attention_layernorm.input")
        pln_out = ld.get(f"{prefix}.post_attention_layernorm.output")
        pln_w = ld.get(f"{prefix}.post_attention_layernorm.__weight.weight")
        rep.check(f"{prefix}.post_attention_layernorm", ref.gemma_rmsnorm(pln_in, pln_w), pln_out, TIGHT)


def check_attn(ld: Loader, rep: Report, prefix: str, mode: str):
    if not ld.has(f"{prefix}.layer_input"):
        rep.skip(f"{prefix}.*", "attn dumps absent in this manifest")
        return

    ln_in = ld.get(f"{prefix}.input_layernorm.input")
    ln_out = ld.get(f"{prefix}.input_layernorm.output")
    ln_w = ld.get(f"{prefix}.input_layernorm.__weight.weight")
    rep.check(f"{prefix}.input_layernorm", ref.gemma_rmsnorm(ln_in, ln_w), ln_out, TIGHT)

    layer_input = ld.get(f"{prefix}.layer_input")
    w_q = ld.get(f"{prefix}.__weight.q_proj")
    s_q = ld.get(f"{prefix}.__weight.q_proj_scale_inv") if ld.has(f"{prefix}.__weight.q_proj_scale_inv") else None
    q_proj_out = ld.get(f"{prefix}.q_proj_out")
    if s_q is not None:
        rep.check(f"{prefix}.q_proj_out(fp8 dequant)", ref.fp8_linear_recompute(layer_input, w_q, s_q), q_proj_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.q_proj_out(bf16)", F.linear(layer_input, w_q), q_proj_out, TIGHT)

    head_dim = ld.get(f"{prefix}.q_split").shape[-1]
    input_shape = q_proj_out.shape[:-1]
    q_chunk, gate_chunk = torch.chunk(q_proj_out.view(*input_shape, -1, head_dim * 2), 2, dim=-1)
    rep.check(f"{prefix}.q_split", q_chunk, ld.get(f"{prefix}.q_split"), TIGHT)
    rep.check(f"{prefix}.gate_split", gate_chunk.reshape(*input_shape, -1), ld.get(f"{prefix}.gate_split"), TIGHT)

    q_norm_w = ld.get(f"{prefix}.__weight.q_norm_weight")
    q_norm_out = ld.get(f"{prefix}.q_norm_out")
    rep.check(f"{prefix}.q_norm_out", ref.gemma_rmsnorm(ld.get(f"{prefix}.q_split"), q_norm_w), q_norm_out, TIGHT)

    w_k = ld.get(f"{prefix}.__weight.k_proj")
    s_k = ld.get(f"{prefix}.__weight.k_proj_scale_inv") if ld.has(f"{prefix}.__weight.k_proj_scale_inv") else None
    k_proj_out = ld.get(f"{prefix}.k_proj_out")
    if s_k is not None:
        rep.check(f"{prefix}.k_proj_out(fp8 dequant)", ref.fp8_linear_recompute(layer_input, w_k, s_k), k_proj_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.k_proj_out(bf16)", F.linear(layer_input, w_k), k_proj_out, TIGHT)

    k_norm_w = ld.get(f"{prefix}.__weight.k_norm_weight")
    k_norm_out = ld.get(f"{prefix}.k_norm_out")
    k_hidden_shape = (*input_shape, -1, head_dim)
    rep.check(f"{prefix}.k_norm_out", ref.gemma_rmsnorm(k_proj_out.view(k_hidden_shape), k_norm_w), k_norm_out, TIGHT)

    w_v = ld.get(f"{prefix}.__weight.v_proj")
    s_v = ld.get(f"{prefix}.__weight.v_proj_scale_inv") if ld.has(f"{prefix}.__weight.v_proj_scale_inv") else None
    v_proj_out = ld.get(f"{prefix}.v_proj_out")
    if s_v is not None:
        rep.check(f"{prefix}.v_proj_out(fp8 dequant)", ref.fp8_linear_recompute(layer_input, w_v, s_v), v_proj_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.v_proj_out(bf16)", F.linear(layer_input, w_v), v_proj_out, TIGHT)

    q_for_rope = q_norm_out.transpose(1, 2)
    k_for_rope = k_norm_out.transpose(1, 2)
    cos = ld.get(f"{prefix}.rope_cos")
    sin = ld.get(f"{prefix}.rope_sin")
    q_rope_r, k_rope_r = ref.apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin)
    rep.check(f"{prefix}.q_rope", q_rope_r, ld.get(f"{prefix}.q_rope"), TIGHT)
    rep.check(f"{prefix}.k_rope", k_rope_r, ld.get(f"{prefix}.k_rope"), TIGHT)

    q_rope = ld.get(f"{prefix}.q_rope")
    if ld.has(f"{prefix}.kv_cache_k_after_write"):
        k_for_attn = ld.get(f"{prefix}.kv_cache_k_after_write")
        v_for_attn = ld.get(f"{prefix}.kv_cache_v_after_write")
    else:
        k_for_attn = ld.get(f"{prefix}.k_rope")
        v_for_attn = v_proj_out.view(k_hidden_shape).transpose(1, 2)

    attn_mask = ld.get(f"{prefix}.attention_mask_used") if ld.has(f"{prefix}.attention_mask_used") else None
    num_kv_groups = q_rope.shape[1] // k_for_attn.shape[1]
    scaling = head_dim ** -0.5
    core_out_r, _ = ref.eager_attention(q_rope, k_for_attn, v_for_attn, attn_mask, scaling, num_kv_groups)
    core_out_r = core_out_r.reshape(*input_shape, -1).contiguous()
    rep.check(f"{prefix}.core_attn_out", core_out_r, ld.get(f"{prefix}.core_attn_out"), LOOSE_KERNEL,
              note="naive eager-softmax recompute (with reconstructed causal mask) vs the real "
                   "backend (attention_mask=None -> implicit is_causal, e.g. SDPA); see LOOSE_KERNEL")

    gated = ref.attn_output_gate(ld.get(f"{prefix}.core_attn_out"), ld.get(f"{prefix}.gate_split"))
    rep.check(f"{prefix}.gate_sigmoid_mul_out", gated, ld.get(f"{prefix}.gate_sigmoid_mul_out"), TIGHT)

    w_o = ld.get(f"{prefix}.__weight.o_proj")
    s_o = ld.get(f"{prefix}.__weight.o_proj_scale_inv") if ld.has(f"{prefix}.__weight.o_proj_scale_inv") else None
    o_out = ld.get(f"{prefix}.o_proj_out")
    gated_dumped = ld.get(f"{prefix}.gate_sigmoid_mul_out")
    if s_o is not None:
        rep.check(f"{prefix}.o_proj_out(fp8 dequant)", ref.fp8_linear_recompute(gated_dumped, w_o, s_o), o_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.o_proj_out(bf16)", F.linear(gated_dumped, w_o), o_out, TIGHT)

    if ld.has(f"{prefix}.post_attention_layernorm.input"):
        pln_in = ld.get(f"{prefix}.post_attention_layernorm.input")
        pln_out = ld.get(f"{prefix}.post_attention_layernorm.output")
        pln_w = ld.get(f"{prefix}.post_attention_layernorm.__weight.weight")
        rep.check(f"{prefix}.post_attention_layernorm", ref.gemma_rmsnorm(pln_in, pln_w), pln_out, TIGHT)


def check_moe(ld: Loader, rep: Report, prefix: str):
    if not ld.has(f"{prefix}.layer_input"):
        rep.skip(f"{prefix}.*", "moe dumps absent in this manifest")
        return

    x = ld.get(f"{prefix}.layer_input")

    w_sg = ld.get(f"{prefix}.__weight.shared_expert.gate_proj")
    s_sg = ld.get(f"{prefix}.__weight.shared_expert.gate_proj_scale_inv") if ld.has(f"{prefix}.__weight.shared_expert.gate_proj_scale_inv") else None
    sg_out = ld.get(f"{prefix}.shared_gate_proj_out")
    if s_sg is not None:
        rep.check(f"{prefix}.shared_gate_proj_out(fp8 dequant)", ref.fp8_linear_recompute(x, w_sg, s_sg), sg_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.shared_gate_proj_out(bf16)", F.linear(x, w_sg), sg_out, TIGHT)

    w_su = ld.get(f"{prefix}.__weight.shared_expert.up_proj")
    s_su = ld.get(f"{prefix}.__weight.shared_expert.up_proj_scale_inv") if ld.has(f"{prefix}.__weight.shared_expert.up_proj_scale_inv") else None
    su_out = ld.get(f"{prefix}.shared_up_proj_out")
    if s_su is not None:
        rep.check(f"{prefix}.shared_up_proj_out(fp8 dequant)", ref.fp8_linear_recompute(x, w_su, s_su), su_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.shared_up_proj_out(bf16)", F.linear(x, w_su), su_out, TIGHT)

    silu_mul = F.silu(sg_out) * su_out
    rep.check(f"{prefix}.shared_silu_mul_out", silu_mul, ld.get(f"{prefix}.shared_silu_mul_out"), TIGHT)

    w_sd = ld.get(f"{prefix}.__weight.shared_expert.down_proj")
    s_sd = ld.get(f"{prefix}.__weight.shared_expert.down_proj_scale_inv") if ld.has(f"{prefix}.__weight.shared_expert.down_proj_scale_inv") else None
    sd_out = ld.get(f"{prefix}.shared_down_proj_out")
    silu_mul_dumped = ld.get(f"{prefix}.shared_silu_mul_out")
    if s_sd is not None:
        rep.check(f"{prefix}.shared_down_proj_out(fp8 dequant)", ref.fp8_linear_recompute(silu_mul_dumped, w_sd, s_sd), sd_out, LOOSE_FP8)
    else:
        rep.check(f"{prefix}.shared_down_proj_out(bf16)", F.linear(silu_mul_dumped, w_sd), sd_out, TIGHT)

    w_gate = ld.get(f"{prefix}.__weight.router_gate_weight")
    logits_r = F.linear(x, w_gate)
    rep.check(f"{prefix}.router_logits(bf16)", logits_r, ld.get(f"{prefix}.router_logits"), TIGHT)

    logits_dumped = ld.get(f"{prefix}.router_logits")
    probs_r = F.softmax(logits_dumped, dtype=torch.float32, dim=-1)
    rep.check(f"{prefix}.router_probs", probs_r, ld.get(f"{prefix}.router_probs"), TIGHT)

    probs_dumped = ld.get(f"{prefix}.router_probs")
    dumped_ids = ld.get(f"{prefix}.topk_ids")
    top_k = dumped_ids.shape[-1]
    # torch.topk's tie-breaking is backend/device-dependent (not literally specified); the
    # original ran on CUDA, so recompute on CUDA too when available to match it as closely as
    # possible -- this alone resolves most ties, since the same kernel on the same input is
    # deterministic.
    topk_device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_raw_r, ids_r = torch.topk(probs_dumped.to(topk_device), top_k, dim=-1)
    weights_raw_r, ids_r = weights_raw_r.cpu(), ids_r.cpu()
    rep.check(f"{prefix}.topk_weights_raw", weights_raw_r, ld.get(f"{prefix}.topk_weights_raw"), TIGHT)
    # Tie-tolerant id check: a differently-broken EXACT tie (same probability value, different
    # expert index) is not a hook-placement bug -- verify it by gathering router_probs at both
    # index sets and requiring them to match; only a genuine value mismatch fails.
    ids_match = torch.equal(ids_r, dumped_ids)
    if ids_match:
        rep.check_exact_int(f"{prefix}.topk_ids", ids_r, dumped_ids)
    else:
        vals_at_recompute = torch.gather(probs_dumped, -1, ids_r)
        vals_at_dumped = torch.gather(probs_dumped, -1, dumped_ids)
        tie_tolerant_ok = torch.allclose(vals_at_recompute, vals_at_dumped, atol=1e-6, rtol=1e-6)
        n_diff_rows = int((ids_r != dumped_ids).any(dim=-1).sum().item())
        rep.rows.append((
            f"{prefix}.topk_ids", tie_tolerant_ok, 0.0 if tie_tolerant_ok else -1.0, 0.0 if tie_tolerant_ok else -1.0,
            f"index order differs in {n_diff_rows} row(s) (torch.topk tie-break is backend-dependent);"
            f" values-at-selected-ids {'MATCH (benign tie)' if tie_tolerant_ok else 'DIFFER (real bug)'}"
        ))

    weights_raw_dumped = ld.get(f"{prefix}.topk_weights_raw")
    renorm_r = weights_raw_dumped / weights_raw_dumped.sum(dim=-1, keepdim=True)
    rep.check(f"{prefix}.topk_renorm_weights", renorm_r, ld.get(f"{prefix}.topk_renorm_weights"), TIGHT)

    # Routed experts: the real forward is a black-box FP8 backend (`FP8Experts`, confirmed via
    # runtime_diagnostics.json), not the plain eager loop -- so there is no per-expert dumped
    # "contribution" to check individually. Instead: recompute EVERY hit expert independently
    # via dequantized weights, index_add them together, and compare the FULL reconstructed sum
    # against the real module's dumped aggregate output. This still exercises routing-token
    # assignment + per-expert GEMM + weighting + combine end-to-end, at the documented fp8 loose
    # tolerance (a routing/hook-placement bug shows up as a gross, non-quantization-shaped error;
    # correct placement shows a small quantization-noise-shaped error).
    reconstructed = torch.zeros_like(x)
    any_expert_checked = False
    for key in list(ld.manifest["meta"].keys()):
        if not key.startswith(f"{prefix}.expert_") or not key.endswith(".token_idx"):
            continue
        ep = key[: -len(".token_idx")]
        token_idx = torch.tensor(ld.meta(key), dtype=torch.long)
        gate_up_w = ld.get(f"{ep}.__weight.gate_up_proj")
        down_w = ld.get(f"{ep}.__weight.down_proj")
        weights_for_tokens = ld.get(f"{ep}.weights_for_tokens")
        x_tokens = x[token_idx]
        s_gu = ld.get(f"{ep}.__weight.gate_up_proj_scale_inv") if ld.has(f"{ep}.__weight.gate_up_proj_scale_inv") else None
        s_dp = ld.get(f"{ep}.__weight.down_proj_scale_inv") if ld.has(f"{ep}.__weight.down_proj_scale_inv") else None
        if s_gu is not None and s_dp is not None:
            gate, up = ref.fp8_linear_recompute(x_tokens, gate_up_w, s_gu).chunk(2, dim=-1)
            h = F.silu(gate) * up
            expert_out = ref.fp8_linear_recompute(h, down_w, s_dp) * weights_for_tokens[:, None]
        else:
            expert_out = ref.moe_expert_weighted(x_tokens, gate_up_w, down_w, weights_for_tokens)
        reconstructed = reconstructed.index_add(0, token_idx, expert_out.to(reconstructed.dtype))
        any_expert_checked = True

    full_hit_count = ld.meta(f"{prefix}.num_distinct_experts_hit", 0)
    checked_count = sum(1 for k in ld.manifest["meta"] if k.startswith(f"{prefix}.expert_") and k.endswith(".token_idx"))
    if any_expert_checked and ld.has(f"{prefix}.routed_expert_output"):
        note = (f"reconstructed from {checked_count}/{full_hit_count} hit experts (dequantized-weight"
                 " recompute; FP8Experts internals are a black box, see README)")
        if checked_count < full_hit_count:
            note = f"PARTIAL: only {checked_count}/{full_hit_count} hit experts dumped (max_dumped_experts cap) -- " + note
        rep.check(f"{prefix}.routed_expert_output(reconstructed, {checked_count} experts)", reconstructed,
                  ld.get(f"{prefix}.routed_expert_output"), LOOSE_FP8, note=note)
    elif not any_expert_checked:
        rep.skip(f"{prefix}.routed_expert_output", "no expert dumps found (0 tokens routed?)")

    w_shared_gate = ld.get(f"{prefix}.__weight.shared_expert_gate_weight")
    shared_gate_logit_r = F.linear(x, w_shared_gate)
    rep.check(f"{prefix}.shared_gate_logit(bf16)", shared_gate_logit_r, ld.get(f"{prefix}.shared_gate_logit"), TIGHT)

    shared_gate_sigmoid_r = torch.sigmoid(ld.get(f"{prefix}.shared_gate_logit"))
    rep.check(f"{prefix}.shared_gate_sigmoid", shared_gate_sigmoid_r, ld.get(f"{prefix}.shared_gate_sigmoid"), TIGHT)

    shared_output_gated_r = ld.get(f"{prefix}.shared_gate_sigmoid") * sd_out
    rep.check(f"{prefix}.shared_output_gated", shared_output_gated_r, ld.get(f"{prefix}.shared_output_gated"), TIGHT)

    if ld.has(f"{prefix}.routed_expert_output"):
        combined_r = ld.get(f"{prefix}.routed_expert_output") + ld.get(f"{prefix}.shared_output_gated")
        rep.check(f"{prefix}.combined_output", combined_r, ld.get(f"{prefix}.combined_output"), TIGHT)


def validate_mode(dump_dir: Path, mode: str) -> Report:
    ld = Loader(dump_dir, mode)
    rep = Report()
    check_gdn(ld, rep, "gdn", mode)
    check_attn(ld, rep, "attn", mode)
    check_moe(ld, rep, "moe0")
    check_moe(ld, rep, "moe3")
    rep.print_table(mode)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    modes = ["prefill", "decode"] if args.mode == "both" else [args.mode]
    all_ok = True
    for mode in modes:
        rep = validate_mode(dump_dir, mode)
        all_ok = all_ok and rep.all_passed()

    print(f"\n=== OVERALL: {'PASS' if all_ok else 'FAIL'} ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
