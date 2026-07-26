#!/usr/bin/env python3
"""HF numerical oracle for Qwen/Qwen3.5-35B-A3B-FP8 (M2-I3 / probe P6).

Loads the HF FP8 checkpoint with the SAME recipe as
`workspace/demo/qwen3_5/accept/reference/generate_reference.py` (AutoModelForCausalLM first,
`TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`, pinned revision), then instruments ONE GDN layer, ONE
full-attention layer, and the MoE block belonging to each of those two layers (see README.md for
why two MoE dumps), dumping every intermediate op tensor named per `docs/qwen35/vllm-graph.md`'s
op tables.

Method: each target module's `forward` is temporarily replaced (bound as an instance method) with
an instrumented copy that is a line-for-line transcription of the real transformers 5.14.1 source
(verified against the actual installed venv-vllm copy on 2026-07-25) with `dump()` calls inserted
at every op boundary. The instrumented copy calls the module's OWN bound sub-callables
(self.in_proj_qkv, self.causal_conv1d_update, self.chunk_gated_delta_rule, self.gate, self.experts,
...) unchanged, so the dumped numbers are exactly what the real model computes -- this is tracing,
not reimplementation. (The independent reimplementation lives in `pytorch_reference.py`, used only
by the validator.)

Regeneration (see README.md for the full recipe + GPU-etiquette wrapper):
    TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1 python ref_dump.py \\
        --model-id Qwen/Qwen3.5-35B-A3B-FP8 \\
        --revision 9d1823d2dee688a6b25e77009dc727688c44936e \\
        --prompts-file <repo>/.pm/eval/prompts.jsonl --prompt-id p01-history \\
        --gdn-layer 0 --attn-layer 3 --prefill-tokens 8 --decode-steps 1 \\
        --out ~/mpk-qwen35/oracle-work/dumps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

import torch
import torch.nn.functional as F


def log(msg: str) -> None:
    print(f"[ref_dump] {msg}", flush=True)


# ============================================================================================
# Dump store
# ============================================================================================


class DumpStore:
    """Collects tensors under a namespaced name, writes each as an individual .pt file (native
    runtime dtype preserved -- see README "Dtype policy"), and accumulates a JSON manifest with
    shape/dtype/summary stats (not full data) so the manifest itself stays tiny and diffable.
    Non-tensor metadata (e.g. an expert's token index list) is stored directly in the manifest.
    """

    def __init__(self, out_dir: Path, mode: str):
        self.out_dir = out_dir
        self.mode = mode
        self.tensor_dir = out_dir / mode / "tensors"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: dict = {"mode": mode, "tensors": {}, "meta": {}}
        self._seen_weight_names: set[str] = set()

    def dump(self, name: str, tensor: torch.Tensor, is_weight: bool = False) -> None:
        if is_weight and name in self._seen_weight_names:
            return  # weights are static; only save once even if the forward runs multiple times
        if is_weight:
            self._seen_weight_names.add(name)
        t = tensor.detach()
        t_cpu = t.to("cpu")
        path = self.tensor_dir / f"{name.replace('/', '_')}.pt"
        torch.save(t_cpu, path)
        stats = self._stats(t_cpu)
        self.manifest["tensors"][name] = {
            "shape": list(t_cpu.shape),
            "dtype": str(t_cpu.dtype),
            "file": str(path.relative_to(self.out_dir)),
            "is_weight": is_weight,
            **stats,
        }

    def meta(self, name: str, value) -> None:
        self.manifest["meta"][name] = value

    @staticmethod
    def _stats(t: torch.Tensor) -> dict:
        try:
            tf = t.to(torch.float32)
            if tf.numel() == 0:
                return {"numel": 0}
            return {
                "numel": int(tf.numel()),
                "mean": float(tf.mean().item()),
                "std": float(tf.std().item()) if tf.numel() > 1 else 0.0,
                "min": float(tf.min().item()),
                "max": float(tf.max().item()),
                "any_nan": bool(torch.isnan(tf).any().item()),
                "any_inf": bool(torch.isinf(tf).any().item()),
            }
        except Exception as e:  # noqa: BLE001 - stats are best-effort, never fatal
            return {"stats_error": str(e)}

    def write_manifest(self) -> Path:
        path = self.out_dir / self.mode / "manifest.json"
        with open(path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        return path


# ============================================================================================
# Instrumented forwards -- verbatim transcriptions of transformers 5.14.1
# transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py, with dump() calls inserted.
# ============================================================================================


def make_gdn_forward(store: DumpStore, prefix: str):
    def instrumented_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_mask_to_padding_states

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        store.dump(f"{prefix}.layer_input", hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        conv_state = recurrent_state = None
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]
            store.dump(f"{prefix}.core_state_before", recurrent_state)
            store.dump(f"{prefix}.conv_state_before", conv_state)

        mixed_qkv = self.in_proj_qkv(hidden_states)
        store.dump(f"{prefix}.qkv_proj_out", mixed_qkv)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        store.dump(f"{prefix}.z_proj_out", z)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        store.dump(f"{prefix}.b_proj_out", b)
        store.dump(f"{prefix}.a_proj_out", a)
        store.dump(f"{prefix}.__weight.in_proj_b", self.in_proj_b.weight.detach(), is_weight=True)
        store.dump(f"{prefix}.__weight.in_proj_a", self.in_proj_a.weight.detach(), is_weight=True)

        store.dump(f"{prefix}.conv_in", mixed_qkv)  # pre-conv, fused [B,8192,T]
        store.dump(f"{prefix}.__weight.conv1d_weight", self.conv1d.weight.detach(), is_weight=True)

        if use_precomputed_states and seq_len == 1:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
            )
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                    activation=self.activation, seq_idx=kwargs.get("seq_idx"),
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        store.dump(f"{prefix}.conv_out", mixed_qkv)  # post-conv (+silu), fused, [B,8192,T]
        if cache_params is not None:
            try:
                store.dump(f"{prefix}.conv_state_after", cache_params.layers[self.layer_idx].conv_states[0])
            except Exception as e:  # noqa: BLE001
                log(f"  (non-fatal) could not dump conv_state_after: {e}")

        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv_t, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        store.dump(f"{prefix}.q_split", query)
        store.dump(f"{prefix}.k_split", key)
        store.dump(f"{prefix}.v_split", value)

        beta = b.sigmoid()
        store.dump(f"{prefix}.beta", beta)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        store.dump(f"{prefix}.decay_g", g)
        store.dump(f"{prefix}.__weight.A_log", self.A_log.detach(), is_weight=True)
        store.dump(f"{prefix}.__weight.dt_bias", self.dt_bias.detach(), is_weight=True)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )
        store.dump(f"{prefix}.core_attn_out", core_attn_out)
        if last_recurrent_state is not None:
            store.dump(f"{prefix}.core_state_after", last_recurrent_state)

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out_flat = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        gated = self.norm(core_attn_out_flat, z_flat)
        store.dump(f"{prefix}.gated_norm_out", gated)
        store.dump(f"{prefix}.__weight.norm_weight", self.norm.weight.detach(), is_weight=True)
        gated = gated.reshape(batch_size, seq_len, -1)

        output = self.out_proj(gated)
        store.dump(f"{prefix}.out_proj_out", output)

        for lin_name, lin in [("in_proj_qkv", self.in_proj_qkv), ("in_proj_z", self.in_proj_z),
                               ("out_proj", self.out_proj)]:
            store.dump(f"{prefix}.__weight.{lin_name}", lin.weight.detach(), is_weight=True)
            scale = find_scale_tensor(lin)
            if scale is not None:
                store.dump(f"{prefix}.__weight.{lin_name}_scale_inv", scale.detach(), is_weight=True)
        return output

    return instrumented_forward


def make_attn_forward(store: DumpStore, prefix: str):
    def instrumented_forward(self, hidden_states, position_embeddings, attention_mask,
                              past_key_values=None, **kwargs):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import eager_attention_forward, apply_rotary_pos_emb

        store.dump(f"{prefix}.layer_input", hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qproj_out = self.q_proj(hidden_states)
        store.dump(f"{prefix}.q_proj_out", qproj_out)
        query_states, gate = torch.chunk(qproj_out.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        store.dump(f"{prefix}.q_split", query_states)
        store.dump(f"{prefix}.gate_split", gate)

        query_states = self.q_norm(query_states.view(hidden_shape))
        store.dump(f"{prefix}.q_norm_out", query_states)
        query_states = query_states.transpose(1, 2)

        kproj_out = self.k_proj(hidden_states)
        store.dump(f"{prefix}.k_proj_out", kproj_out)
        key_states = self.k_norm(kproj_out.view(hidden_shape))
        store.dump(f"{prefix}.k_norm_out", key_states)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states)
        store.dump(f"{prefix}.v_proj_out", value_states)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        store.dump(f"{prefix}.rope_cos", cos)
        store.dump(f"{prefix}.rope_sin", sin)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        store.dump(f"{prefix}.q_rope", query_states)
        store.dump(f"{prefix}.k_rope", key_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            store.dump(f"{prefix}.kv_cache_k_after_write", key_states)
            store.dump(f"{prefix}.kv_cache_v_after_write", value_states)

        if isinstance(attention_mask, torch.Tensor):
            store.dump(f"{prefix}.attention_mask_used", attention_mask)
        else:
            store.meta(f"{prefix}.attention_mask_used_is_none", attention_mask is None)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        store.dump(f"{prefix}.core_attn_out", attn_output)

        attn_output = attn_output * torch.sigmoid(gate)
        store.dump(f"{prefix}.gate_sigmoid_mul_out", attn_output)

        output = self.o_proj(attn_output)
        store.dump(f"{prefix}.o_proj_out", output)

        for lin_name, lin in [("q_proj", self.q_proj), ("k_proj", self.k_proj), ("v_proj", self.v_proj),
                               ("o_proj", self.o_proj)]:
            store.dump(f"{prefix}.__weight.{lin_name}", lin.weight.detach(), is_weight=True)
            scale = find_scale_tensor(lin)
            if scale is not None:
                store.dump(f"{prefix}.__weight.{lin_name}_scale_inv", scale.detach(), is_weight=True)
        store.dump(f"{prefix}.__weight.q_norm_weight", self.q_norm.weight.detach(), is_weight=True)
        store.dump(f"{prefix}.__weight.k_norm_weight", self.k_norm.weight.detach(), is_weight=True)
        return output, attn_weights

    return instrumented_forward


def make_router_forward(store: DumpStore, prefix: str):
    """Instruments `Qwen3_5MoeTopKRouter.forward` itself (traced, not reimplemented in the
    parent block) -- closes a methodological gap: the MoE block's forward could otherwise inline
    the router math itself, making the self-consistency validator check that inlined copy against
    itself rather than against the real module's actual computation. `Qwen3_5MoeTopKRouter` has
    no `@use_..._from_hub`-style decorator (confirmed by source inspection), so this is expected
    to always be the plain class, but tracing it directly removes any doubt."""
    def instrumented_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        store.dump(f"{prefix}.router_logits", router_logits)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        store.dump(f"{prefix}.router_probs", router_probs)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        store.dump(f"{prefix}.topk_weights_raw", router_top_value)
        store.dump(f"{prefix}.topk_ids", router_indices)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        store.dump(f"{prefix}.topk_renorm_weights", router_top_value)
        store.dump(f"{prefix}.__weight.router_gate_weight", self.weight.detach(), is_weight=True)
        return router_logits, router_top_value, router_indices

    return instrumented_forward


def make_moe_forward(store: DumpStore, prefix: str, max_dumped_experts: int = 128):
    def instrumented_forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        store.dump(f"{prefix}.layer_input", hidden_states_reshaped)

        # --- shared expert MLP (Qwen3_5MoeMLP), instrumented inline ---
        se = self.shared_expert
        gate_proj_out = se.gate_proj(hidden_states_reshaped)
        up_proj_out = se.up_proj(hidden_states_reshaped)
        store.dump(f"{prefix}.shared_gate_proj_out", gate_proj_out)
        store.dump(f"{prefix}.shared_up_proj_out", up_proj_out)
        silu_mul = se.act_fn(gate_proj_out) * up_proj_out
        store.dump(f"{prefix}.shared_silu_mul_out", silu_mul)
        shared_expert_output = se.down_proj(silu_mul)
        store.dump(f"{prefix}.shared_down_proj_out", shared_expert_output)

        # --- router: call the REAL Qwen3_5MoeTopKRouter as a black box (its own forward is
        # separately instrumented via make_router_forward and bound onto self.gate below) ---
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        # --- routed experts: call the REAL module as a black box. Empirically (see
        # runtime_diagnostics.json "moe_experts_backend"), `self.experts` is NOT the plain
        # eager-loop `Qwen3_5MoeExperts` shown in the class body -- for this FP8 checkpoint it is
        # an `FP8Experts` backend swapped in by `@use_experts_implementation`, whose internal
        # per-expert accumulate/cast points are opaque (same situation as vLLM's FlashInfer
        # TRT-LLM MoE kernel, vllm-graph.md §2.3.4). So: dump only the module BOUNDARY
        # (input/output + which experts fired), and let the validator recompute per-expert via
        # dequantized weights at the documented LOOSE (fp8) tolerance -- do not reimplement
        # FP8Experts' internals here.
        experts = self.experts
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=experts.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        store.meta(f"{prefix}.num_distinct_experts_hit", int(expert_hit.numel()))

        final_hidden_states = experts(hidden_states_reshaped, selected_experts, routing_weights)
        store.dump(f"{prefix}.routed_expert_output", final_hidden_states)

        dumped = 0
        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0])
            if dumped >= max_dumped_experts:
                log(f"  ({prefix}) max_dumped_experts={max_dumped_experts} reached;"
                    f" {int(expert_hit.numel()) - dumped} more hit experts not dumped")
                break
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            weights_for_tokens = routing_weights[token_idx, top_k_pos]
            ep = f"{prefix}.expert_{expert_idx}"
            store.meta(f"{ep}.token_idx", token_idx.detach().cpu().tolist())
            store.dump(f"{ep}.weights_for_tokens", weights_for_tokens)
            store.dump(f"{ep}.__weight.gate_up_proj", experts.gate_up_proj[expert_idx].detach(), is_weight=True)
            store.dump(f"{ep}.__weight.down_proj", experts.down_proj[expert_idx].detach(), is_weight=True)
            scale_gu = find_scale_tensor(experts, param_name="gate_up_proj", expert_idx=expert_idx)
            scale_dp = find_scale_tensor(experts, param_name="down_proj", expert_idx=expert_idx)
            if scale_gu is not None:
                store.dump(f"{ep}.__weight.gate_up_proj_scale_inv", scale_gu.detach(), is_weight=True)
            if scale_dp is not None:
                store.dump(f"{ep}.__weight.down_proj_scale_inv", scale_dp.detach(), is_weight=True)
            dumped += 1

        shared_gate_logit = self.shared_expert_gate(hidden_states_reshaped)
        store.dump(f"{prefix}.shared_gate_logit", shared_gate_logit)
        shared_gate_sigmoid = F.sigmoid(shared_gate_logit)
        store.dump(f"{prefix}.shared_gate_sigmoid", shared_gate_sigmoid)
        shared_expert_output = shared_gate_sigmoid * shared_expert_output
        store.dump(f"{prefix}.shared_output_gated", shared_expert_output)
        store.dump(f"{prefix}.__weight.shared_expert_gate_weight", self.shared_expert_gate.weight.detach(),
                   is_weight=True)

        expert_output = final_hidden_states + shared_expert_output
        store.dump(f"{prefix}.combined_output", expert_output)
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)

        for lin_name, lin in [("shared_expert.gate_proj", se.gate_proj), ("shared_expert.up_proj", se.up_proj),
                               ("shared_expert.down_proj", se.down_proj)]:
            store.dump(f"{prefix}.__weight.{lin_name}", lin.weight.detach(), is_weight=True)
            scale = find_scale_tensor(lin)
            if scale is not None:
                store.dump(f"{prefix}.__weight.{lin_name}_scale_inv", scale.detach(), is_weight=True)
        return expert_output

    return instrumented_forward


def make_rmsnorm_hook(store: DumpStore, name: str):
    def hook(module, args, kwargs, output):
        x = args[0] if args else kwargs.get("x")
        store.dump(f"{name}.input", x)
        store.dump(f"{name}.output", output)
        store.dump(f"{name}.__weight.weight", module.weight.detach(), is_weight=True)

    return hook


def find_scale_tensor(module, param_name: str | None = None, expert_idx: int | None = None):
    """Best-effort lookup of an FP8 block-quant scale tensor on a module. HF's finegrained_fp8
    integration name is confirmed empirically at runtime (see runtime_diagnostics.json's
    `scale_attr_found` field) rather than hardcoded, defensively covering the plausible
    candidate names."""
    candidates = ["weight_scale_inv", "weight_scale", "input_scale", "scale"]
    if param_name is not None:
        candidates = [f"{param_name}_scale_inv", f"{param_name}_scale"] + candidates
        idx_candidates = [f"{param_name}_weight_scale_inv", f"{param_name}_scale_inv"]
        candidates = idx_candidates + candidates
    for c in candidates:
        if hasattr(module, c):
            val = getattr(module, c)
            if isinstance(val, torch.Tensor):
                if expert_idx is not None and val.dim() >= 1 and val.shape[0] >= expert_idx + 1 \
                        and val.dim() == 3:
                    return val[expert_idx]
                return val
    return None


# ============================================================================================
# Model loading (same recipe as accept/reference/generate_reference.py)
# ============================================================================================


def load_model_and_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    load_notes = {"attempts": []}
    model = None

    def try_causal_lm():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto", device_map="cuda:0")

    def try_image_text_to_text():
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype="auto", device_map="cuda:0"
        )

    def try_direct_class():
        from transformers import Qwen3_5MoeForConditionalGeneration
        return Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype="auto", device_map="cuda:0"
        )

    for name, fn in [
        ("AutoModelForCausalLM", try_causal_lm),
        ("AutoModelForImageTextToText", try_image_text_to_text),
        ("Qwen3_5MoeForConditionalGeneration (direct)", try_direct_class),
    ]:
        try:
            log(f"attempting load via {name} ...")
            t0 = time.time()
            model = fn()
            load_notes["attempts"].append({"class": name, "ok": True, "seconds": round(time.time() - t0, 1)})
            load_notes["loaded_via"] = name
            break
        except Exception as e:  # noqa: BLE001
            log(f"  failed via {name}: {type(e).__name__}: {e}")
            load_notes["attempts"].append({"class": name, "ok": False, "error": f"{type(e).__name__}: {e}"})

    if model is None:
        raise RuntimeError("Could not load the FP8 checkpoint via any known auto class.")
    model.eval()
    return model, tokenizer, load_notes


def _default_prompts_file() -> str:
    here = Path(__file__).resolve()
    try:
        return str(here.parents[4] / ".pm" / "eval" / "prompts.jsonl")
    except IndexError:
        return str(here.parent / "prompts.jsonl")


def load_prompts(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================================================
# Main
# ============================================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--revision", default="9d1823d2dee688a6b25e77009dc727688c44936e")
    ap.add_argument("--prompts-file", default=_default_prompts_file())
    ap.add_argument("--prompt-id", default="p01-history")
    ap.add_argument("--gdn-layer", type=int, default=0, help="layer index to hook as the GDN oracle layer")
    ap.add_argument("--attn-layer", type=int, default=3, help="layer index to hook as the full-attn oracle layer")
    ap.add_argument("--prefill-tokens", type=int, default=8, help="tokens in the chunked-prefill dump")
    ap.add_argument("--decode-steps", type=int, default=1, help="single-token decode steps to dump after prefill")
    ap.add_argument("--out", required=True, help="output directory (prefill/ and decode/ subdirs created)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers
    import accelerate

    log(f"torch {torch.__version__}, transformers {transformers.__version__}, accelerate {accelerate.__version__}")
    assert torch.cuda.is_available(), "This script requires a visible CUDA GPU."
    log(f"visible GPU: {torch.cuda.get_device_name(0)}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")

    from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

    fast_path_diag = {
        "is_causal_conv1d_available": is_causal_conv1d_available(),
        "is_flash_linear_attention_available": is_flash_linear_attention_available(),
    }
    log(f"GDN fast-path availability: {fast_path_diag}")

    prompts = load_prompts(Path(args.prompts_file))
    row = next((r for r in prompts if r["id"] == args.prompt_id), None)
    if row is None:
        raise SystemExit(f"prompt id {args.prompt_id!r} not found in {args.prompts_file}")

    model, tokenizer, load_notes = load_model_and_tokenizer(args.model_id, args.revision)
    log(f"model loaded via {load_notes.get('loaded_via')}")

    layer_types = list(model.config.layer_types)
    log(f"layer_types[:8] = {layer_types[:8]} ...")
    assert layer_types[args.gdn_layer] == "linear_attention", (
        f"--gdn-layer {args.gdn_layer} is {layer_types[args.gdn_layer]!r}, expected 'linear_attention'"
    )
    assert layer_types[args.attn_layer] == "full_attention", (
        f"--attn-layer {args.attn_layer} is {layer_types[args.attn_layer]!r}, expected 'full_attention'"
    )

    gdn_layer_module = model.model.layers[args.gdn_layer]
    attn_layer_module = model.model.layers[args.attn_layer]

    # empirically confirm which concrete forward is bound (hub-kernel override vs plain class)
    runtime_diag = {
        "fast_path": fast_path_diag,
        "gdn_forward_qualname_before_patch": type(gdn_layer_module.linear_attn).forward.__qualname__,
        "attn_forward_qualname_before_patch": type(attn_layer_module.self_attn).forward.__qualname__,
        "causal_conv1d_update_bound": repr(gdn_layer_module.linear_attn.causal_conv1d_update),
        "chunk_gated_delta_rule_bound": repr(gdn_layer_module.linear_attn.chunk_gated_delta_rule),
        "recurrent_gated_delta_rule_bound": repr(gdn_layer_module.linear_attn.recurrent_gated_delta_rule),
        "gdn_layer_idx": args.gdn_layer,
        "attn_layer_idx": args.attn_layer,
        "model_id": args.model_id,
        "revision": args.revision,
        "prompt_id": args.prompt_id,
        "prefill_tokens": args.prefill_tokens,
        "decode_steps": args.decode_steps,
        "load_notes": load_notes,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "accelerate": accelerate.__version__, "python": sys.version},
        "gpu": torch.cuda.get_device_name(0),
    }

    # scale-attribute empirical discovery, using in_proj_qkv as a representative fp8 linear
    scale = find_scale_tensor(gdn_layer_module.linear_attn.in_proj_qkv)
    runtime_diag["scale_attr_probe"] = {
        "module": "layers[gdn].linear_attn.in_proj_qkv",
        "weight_dtype": str(gdn_layer_module.linear_attn.in_proj_qkv.weight.dtype),
        "found_scale": scale is not None,
        "scale_shape": list(scale.shape) if scale is not None else None,
        "scale_dtype": str(scale.dtype) if scale is not None else None,
    }

    # MoE experts backend: does @use_experts_implementation swap in something other than the
    # plain-loop Qwen3_5MoeExperts.forward shown in the source? And what are the ACTUAL fp8
    # scale attribute names on the experts module (empirically, not guessed)?
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeExperts, Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeTopKRouter,
    )
    experts_mod = gdn_layer_module.mlp.experts
    runtime_diag_router = {
        "router_class": type(gdn_layer_module.mlp.gate).__name__,
        "forward_is_plain_class": type(gdn_layer_module.mlp.gate).forward is Qwen3_5MoeTopKRouter.forward,
    }
    scale_like_attrs = [n for n in vars(experts_mod).keys() if "scale" in n.lower()]
    scale_like_attrs += [n for n, _ in experts_mod.named_buffers(recurse=False) if "scale" in n.lower()]
    scale_like_attrs += [n for n, _ in experts_mod.named_parameters(recurse=False) if "scale" in n.lower()]
    runtime_diag["moe_experts_backend"] = {
        "experts_class": type(experts_mod).__name__,
        "forward_is_plain_loop_class": type(experts_mod).forward is Qwen3_5MoeExperts.forward,
        "forward_qualname": type(experts_mod).forward.__qualname__,
        "gate_up_proj_dtype": str(experts_mod.gate_up_proj.dtype),
        "gate_up_proj_shape": list(experts_mod.gate_up_proj.shape),
        "scale_like_attrs_found": sorted(set(scale_like_attrs)),
    }
    runtime_diag["gdn_backend"] = {
        "gdn_class": type(gdn_layer_module.linear_attn).__name__,
        "forward_is_plain_class": type(gdn_layer_module.linear_attn).forward is Qwen3_5MoeGatedDeltaNet.forward,
    }
    runtime_diag["router_backend"] = runtime_diag_router
    log(f"runtime diagnostics: {json.dumps(runtime_diag, indent=2, default=str)}")

    def build_and_patch(store: DumpStore):
        gdn_layer_module.linear_attn.forward = types.MethodType(
            make_gdn_forward(store, "gdn"), gdn_layer_module.linear_attn
        )
        attn_layer_module.self_attn.forward = types.MethodType(
            make_attn_forward(store, "attn"), attn_layer_module.self_attn
        )
        gdn_layer_module.mlp.forward = types.MethodType(make_moe_forward(store, "moe0"), gdn_layer_module.mlp)
        attn_layer_module.mlp.forward = types.MethodType(make_moe_forward(store, "moe3"), attn_layer_module.mlp)
        gdn_layer_module.mlp.gate.forward = types.MethodType(
            make_router_forward(store, "moe0"), gdn_layer_module.mlp.gate)
        attn_layer_module.mlp.gate.forward = types.MethodType(
            make_router_forward(store, "moe3"), attn_layer_module.mlp.gate)

        handles = []
        handles.append(gdn_layer_module.input_layernorm.register_forward_hook(
            make_rmsnorm_hook(store, "gdn.input_layernorm"), with_kwargs=True))
        handles.append(gdn_layer_module.post_attention_layernorm.register_forward_hook(
            make_rmsnorm_hook(store, "gdn.post_attention_layernorm"), with_kwargs=True))
        handles.append(attn_layer_module.input_layernorm.register_forward_hook(
            make_rmsnorm_hook(store, "attn.input_layernorm"), with_kwargs=True))
        handles.append(attn_layer_module.post_attention_layernorm.register_forward_hook(
            make_rmsnorm_hook(store, "attn.post_attention_layernorm"), with_kwargs=True))
        return handles

    # ---------------- Chunked-prefill dump (no cache; multi-token forward) ----------------
    messages = row["messages"]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids_full = encoded.to(model.device) if not hasattr(encoded, "input_ids") else encoded.input_ids.to(model.device)
    n_pf = min(args.prefill_tokens, input_ids_full.shape[1])
    prefill_ids = input_ids_full[:, :n_pf]
    log(f"chunked-prefill dump: prompt_id={args.prompt_id} prefill_tokens={n_pf} ids={prefill_ids.tolist()}")

    prefill_store = DumpStore(out_dir, "prefill")
    handles = build_and_patch(prefill_store)
    with torch.no_grad():
        out = model(input_ids=prefill_ids, attention_mask=torch.ones_like(prefill_ids), use_cache=True)
    for h in handles:
        h.remove()
    prefill_store.meta("input_ids", prefill_ids.detach().cpu().tolist())
    prefill_store.meta("gdn_layer_idx", args.gdn_layer)
    prefill_store.meta("attn_layer_idx", args.attn_layer)
    manifest_path = prefill_store.write_manifest()
    log(f"wrote {manifest_path} ({len(prefill_store.manifest['tensors'])} tensors)")

    # ---------------- Decode-step dump (continue from the primed cache) ----------------
    cache = out.past_key_values
    decode_store = DumpStore(out_dir, "decode")
    handles = build_and_patch(decode_store)
    next_tok = out.logits[:, -1:].float().argmax(dim=-1)
    log(f"decode-step dump: continuing from cache (seq_len so far={n_pf}), next_tok={next_tok.tolist()}")
    with torch.no_grad():
        for step in range(args.decode_steps):
            step_out = model(
                input_ids=next_tok,
                attention_mask=torch.ones((next_tok.shape[0], n_pf + step + 1), device=model.device, dtype=torch.long),
                use_cache=True,
                past_key_values=cache,
                cache_position=torch.tensor([n_pf + step], device=model.device),
            )
            cache = step_out.past_key_values
            next_tok = step_out.logits[:, -1:].float().argmax(dim=-1)
    for h in handles:
        h.remove()
    decode_store.meta("input_ids_context", prefill_ids.detach().cpu().tolist())
    decode_store.meta("gdn_layer_idx", args.gdn_layer)
    decode_store.meta("attn_layer_idx", args.attn_layer)
    manifest_path = decode_store.write_manifest()
    log(f"wrote {manifest_path} ({len(decode_store.manifest['tensors'])} tensors)")

    diag_path = out_dir / "runtime_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(runtime_diag, f, indent=2, default=str)
    log(f"wrote {diag_path}")
    log("done.")


if __name__ == "__main__":
    main()
