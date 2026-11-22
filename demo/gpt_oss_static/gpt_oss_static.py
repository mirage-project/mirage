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

"""Greedy GPT-OSS-20B/120B decode on Hopper (sm90) or Blackwell (sm100) with one static megakernel per layer."""
import argparse, json, os, statistics, subprocess, time
# 120B's weights fill most of an H100: the expert repacking's freed temporaries must be reusable for the next layer's
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from mirage.mpk import compiler, cost_records, plan_gen, search, static_schedule
from mirage.mpk.static_harness import Harness
from mirage.core import uint8
from mirage.mpk.static_megakernel import (StaticMegakernel, sm90_attention_parts_size, sm90_flags,
                                         sm90_moe_pack, sm100_flags)
from mirage.mpk.sm90_moe_split import Sm90MoeSplit, sm90_moe_split_weights
from mirage.mpk.models.utils import shuffle_tensors
from mxfp4 import Mxfp4Checkpoint, convert_expert_weight, prepare_gpt_oss_expert_bank

FI_GEMM2_WIDTH = 3072   # FlashInfer's unfinalized MoE rows (flashinfer_moe.H_PAD; Model checks): gpt_oss_moe_norm's row stride
T, PAGE = 8, 64                       # token rows of a step (main: the bucket of the batch), KV page size
BUCKETS = (8, 16, 32, 64)             # the kernels' token rows, as the other static models' (a step of R requests: the
                                      # smallest >= R; rows past R idle)
PLANS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plans")
SERVE = ["-DSTATIC_RESET_IN_KERNEL", "-DSTATIC_PDL_TRIGGER=1"]   # serving build: launches back to back
FLAGS = sm90_flags() + SERVE
# the kernels' architecture: the GPU's, 90 (Hopper) or 100 (Blackwell)
CC = 100 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10 else 90
NATIVE_PAD = 2944                     # Blackwell's native experts (gpt_oss_moe_experts): H and I padded to 23 K tiles
O_PARTS = 4   # o_proj's K parts a plan may choose (o32 holds them all; the gate adds them)
QKV_K = 3072   # Blackwell: the QKV weight's K zero-padded to 24 tiles of 128, so a plan may split it (QKV_PARTS parts in
QKV_PARTS = 4  # qkv32; the split attention adds them)


def qkv_parts() -> int:
    """The QKV GEMM's K parts a plan may choose: Blackwell's split attention adds them (MPK's sm100 attention: one)."""
    return QKV_PARTS if CC == 100 and os.environ.get("GPTOSS_ATTN") != "sm100" else 1


def pad_k(w: torch.Tensor) -> torch.Tensor:
    """The QKV weight [N, H] with K zero-padded to QKV_K when its K parts may split (qkv_parts)."""
    return torch.nn.functional.pad(w, (0, QKV_K - w.shape[1])).contiguous() if qkv_parts() > 1 else w


def min_kparts(gemm: str) -> int:
    """The QKV ("qkv") and o_proj ("o") GEMMs' fewest K parts on Blackwell, where the plan's fewer parts stream from too few
    SMs while the attention or the gate waits on the GEMM: 32+ token rows, 2 (one part a 128-row block from 40 / 45 SMs;
    64 requests at 8K: the sliding layer 57.9 -> 48.4 us); 8 rows, the QKV GEMM 3 (2 requests' plan, 80 tasks: the full
    layer's QKV GEMM ends at 15.1 us, the 8-request plan's 120 at 10.5). GPTOSS_MIN_KPARTS overrides both."""
    if os.environ.get("GPTOSS_MIN_KPARTS"):
        return int(os.environ["GPTOSS_MIN_KPARTS"])
    if CC != 100:
        return 1
    return 2 if T >= 32 else (3 if T == 8 and gemm == "qkv" else 1)


def pad_rows(w: torch.Tensor) -> torch.Tensor:
    """o_proj's weight [H, Q D] on Blackwell with its rows zero-padded to whole 128-row tiles (NATIVE_PAD): 23 tiles of
    128 rows instead of 45 of 64 (a 128-row task streams its weight about twice as fast; the rows past H not stored)."""
    if CC != 100 or w.shape[0] % 128 == 0 or os.environ.get("GPTOSS_O_PAD", "0") != "1":
        return w
    return torch.nn.functional.pad(w, (0, 0, 0, -w.shape[0] % 128)).contiguous()


def flags() -> list:
    """The layer kernels' build flags. Blackwell: the split attention's splits at least GPTOSS_ATTN_MIN_TILES tiles of 32
    positions (a 128-position window: one split a request, no one-tile units over all SMs)."""
    if CC != 100:
        return sm90_flags() + SERVE
    return sm100_flags() + SERVE + os.environ.get("GPTOSS_XFLAGS", "").split() + [f"-DSTATIC_ATTN_MIN_TILES={int(os.environ.get('GPTOSS_ATTN_MIN_TILES', 1))}"]


def moe_kind(requests: int) -> str:
    """The routed experts, after the layer kernel unless "native". Hopper: sm90_moe_split ("mpk") up to 16 requests,
    triton_kernels' matmul_ogs ("triton", triton_moe.py) from 32 (120B graphed decode step: 8 requests 7.34 vs 7.62 ms,
    16: 9.72 vs 9.55, 64: 30.7 -> 16.6; plans/h100/README.md). Blackwell: FlashInfer's trtllm-gen MXFP4 MoE
    ("flashinfer", flashinfer_moe.py: vLLM's B200 GPT-OSS kernel; 120B, 8 requests, real routing: 5.46 ms a step vs
    6.25 with the native task), or the native tcgen05 task in the layer kernel ("mpk": gpt_oss_moe_experts, 8 token
    rows). GPTOSS_MOE=mpk|triton|flashinfer picks one; a missing package falls back to the architecture's own kernel."""
    want = os.environ.get("GPTOSS_MOE", "")
    kind = want or ("triton" if CC == 90 and requests >= 32 else "flashinfer" if CC == 100 else "mpk")
    assert kind in ("mpk", "triton", "flashinfer") and not (kind == "flashinfer" and CC != 100), kind
    if kind != "mpk":
        try:
            __import__(kind + "_moe")
        except ImportError as e:
            if want:
                raise
            print(f"{kind}_moe not importable ({e}): the routed experts as the MPK kernel", flush=True)
            kind = "mpk"
    assert not (CC == 100 and kind == "mpk" and T != 8), "Blackwell's native experts (gpt_oss_moe_experts): 8 token rows"
    return kind


class Model:
    """The checkpoint's config and weights, converted for the static kernels (on the GPU), and the step's shared buffers."""

    def __init__(self, ckpt: str, max_seq: int, requests: int = 1):
        from transformers import AutoConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
        cfg = self.cfg = AutoConfig.from_pretrained(ckpt)
        self.H, self.I, self.E, self.TOPK = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts, cfg.num_experts_per_tok
        self.QH, self.KVH, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.L, self.types, self.window, self.V = cfg.num_hidden_layers, list(cfg.layer_types), cfg.sliding_window, cfg.vocab_size
        self.eps, self.limit = cfg.rms_norm_eps, getattr(cfg, "swiglu_limit", 7.0)
        self.max_seq, self.R = max_seq, requests
        R, MP = requests, max_seq // PAGE                            # requests (a token each), pages a request
        assert max_seq % PAGE == 0 and 1 <= R <= T
        sd = Mxfp4Checkpoint(ckpt)
        cos, sin = GptOssRotaryEmbedding(cfg)(torch.empty(0, dtype=torch.bfloat16), torch.arange(max_seq)[None])
        bf = lambda t: t.to(torch.bfloat16).contiguous().cuda()
        zeros = lambda *shape, dtype=torch.bfloat16: torch.zeros(*shape, dtype=dtype, device="cuda")
        H, I, E, TOPK, QH, KVH, D = self.H, self.I, self.E, self.TOPK, self.QH, self.KVH, self.D
        self.shared = {                                              # every layer's (the step's buffers)
            "cos": bf(torch.cat([cos[0], cos[0]], -1)), "sin": bf(torch.cat([sin[0], sin[0]], -1)),
            "qo_indptr": torch.arange(R + 1, dtype=torch.int32, device="cuda"),
            "kv_indptr": torch.arange(R + 1, dtype=torch.int32, device="cuda"),
            "kv_indices": torch.zeros(R * MP, dtype=torch.int32, device="cuda"),   # request r's pages r MP .., packed
            "last_page_len": torch.ones(R, dtype=torch.int32, device="cuda"),
            "num_tokens": torch.full((1,), R, dtype=torch.int32, device="cuda"),
            "attn_parts": zeros(sm90_attention_parts_size(R, QH, D, torch.cuda.get_device_properties(0)
                                                          .multi_processor_count), dtype=torch.float32),   # its splits
            "h": zeros(T, H), "qkv32": zeros(qkv_parts() * T, (QH + 2 * KVH) * D, dtype=torch.float32),
            "qkv": zeros(T, (QH + 2 * KVH) * D), "attn": zeros(T, QH * D), "o32": zeros(O_PARTS * T, H, dtype=torch.float32),
            "x1": zeros(T, H), "h2": zeros(T, H, dtype=torch.float16),   # h2: the MoE's f16 input
            "topk_w": zeros(T, TOPK, dtype=torch.float32), "routing": zeros(E, T, dtype=torch.int32),
            "eo": zeros(T, TOPK, H), "act": zeros(E, T, I, dtype=torch.float16),
        }
        self.x = [zeros(T, H), zeros(T, H)]                          # the residual stream, ping-pong between layers
        self.x_embed = zeros(T, H)                                   # layer 0's input: the head writes the next token's row
        self.layers = []
        self.done = zeros(T, H // 64, dtype=torch.int32)             # the experts' combine counters
        pack = lambda w, s, rows, glu: sm90_moe_split_weights(sm90_moe_pack(w.cuda(), s.cuda(), rows, glu=glu), rows)
        self.moe_kind = moe_kind(requests)
        self.native = CC == 100 and self.moe_kind == "mpk"            # the experts inside the layer kernel
        if self.native:
            self.shared["z_q"] = zeros(T, NATIVE_PAD, dtype=torch.uint8)   # h2 as MXFP8 (gpt_oss_mxfp8_quant)
        if self.moe_kind == "flashinfer":   # the gate writes FlashInfer's inputs: h2 in its padded rows (the columns past
            from flashinfer_moe import H_PAD    # H stay zero), the top-k ids [T][TOPK] in routing's rows past E
            # GPTOSS_FI_ACT=mxfp8: h2 as MXFP8 (SGLang's default activations on B200), else bf16 (vLLM's)
            self.fi_mxfp8 = os.environ.get("GPTOSS_FI_ACT", "mxfp8") == "mxfp8"
            # GPTOSS_FI_ROUTE=logits: the gate writes the router logits, FlashInfer's routing takes the top-k (SGLang's
            # path; no top-k in the gate); "ids": the gate's top-k ids and weights (FlashInfer's routed entry)
            self.fi_logits = os.environ.get("GPTOSS_FI_ROUTE", "logits") == "logits"
            self.shared["h2"] = (zeros(T, H_PAD * 33 // 32, dtype=torch.uint8) if self.fi_mxfp8 else zeros(T, H_PAD))
            self.shared["routing"] = zeros(E + TOPK, T, dtype=torch.int32)
        # FlashInfer's experts leave their sum in moe_y; the next layer's first norm (or the head's) adds it to the
        # x1 before (x_res: the residual stream, for the gate), no combine kernel between them. Layer 0 adds y0 (zeros).
        self.fused_residual = self.moe_kind == "flashinfer"
        if self.fused_residual:
            self.shared.update({"moe_y": zeros(T, H), "x_res": zeros(T, H)})
            self.y0 = zeros(T, H)
        # GPTOSS_MOE_FOLD=1 (Blackwell, FlashInfer from logits): FlashInfer runs unfinalized and the next layer's first
        # task (gpt_oss_moe_norm) computes its finalize, the residual add and the norm: one kernel less a layer. moe_tab
        # [L + 1, 4] int64: row j = the experts' outputs layer j's norm reads (row 0 none; FlashinferMoe.set_fold)
        self.moe_fold = (self.fused_residual and self.fi_logits and os.environ.get("GPTOSS_MOE_FOLD", "0") == "1")
        if self.moe_fold:
            self.moe_tab = zeros(self.L + 1, 4, dtype=torch.int64)
        ext = None if self.moe_kind == "mpk" else __import__(self.moe_kind + "_moe")
        self.texperts = []
        names = ("gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias", "down_proj_blocks",
                 "down_proj_scales", "down_proj_bias")
        for i in range(self.L):
            p = f"model.layers.{i}."
            at = lambda k: sd[p + k]
            if ext:
                self.texperts.append(ext.expert_weights(*(at(f"mlp.experts.{n}") for n in names)))
                # the layer kernels' graphs attach every layer tensor (plans key on its node ids): placeholders
                ex = {"w13": zeros(1, dtype=torch.uint8), "b13": zeros(1), "w2": zeros(1, dtype=torch.uint8),
                      "b2": zeros(1)}
            elif self.native:
                w13, s13, w2, s2, b13, b2 = (v.cuda() for v in prepare_gpt_oss_expert_bank(
                    *(at(f"mlp.experts.{n}") for n in names), padded_width=NATIVE_PAD))
                ex = {"w13": w13, "w13_scales": s13, "w2": w2, "w2_scales": s2, "b13": bf(b13), "b2": bf(b2)}
            else:
                w13, s13, b13 = convert_expert_weight(at("mlp.experts.gate_up_proj_blocks"),
                                                      at("mlp.experts.gate_up_proj_scales"),
                                                      at("mlp.experts.gate_up_proj_bias"), deinterleave=True)
                w2, s2, b2 = convert_expert_weight(at("mlp.experts.down_proj_blocks"), at("mlp.experts.down_proj_scales"),
                                                   at("mlp.experts.down_proj_bias"), deinterleave=False)
                ex = {"w13": pack(w13, s13, 128, True), "b13": bf(b13), "w2": pack(w2, s2, 64, False), "b2": bf(b2)}
            q = [at(f"self_attn.{n}_proj.weight") for n in "qkv"]
            qb = [at(f"self_attn.{n}_proj.bias") for n in "qkv"]
            self.layers.append({
                "ln1": bf(at("input_layernorm.weight")), "ln2": bf(at("post_attention_layernorm.weight")),
                "w_qkv": pad_k(bf(shuffle_tensors(q, KVH, 0))), "b_qkv": bf(shuffle_tensors(qb, KVH, 0)),
                "w_o": pad_rows(bf(at("self_attn.o_proj.weight"))), "b_o": bf(at("self_attn.o_proj.bias")),
                "sinks": bf(at("self_attn.sinks").view(KVH, QH // KVH)),
                "w_router": bf(at("mlp.router.weight")), "b_router": bf(at("mlp.router.bias")),
                **ex,
            })
        # The KV caches after all the experts' repacking (its temporaries beside long-context caches overflow an H100
        # with 120B). The sliding-window layers' caches are rings of RP pages a request (the window and the step's page).
        self.RP = self.window // PAGE + 2
        self.kv_indices_ring = torch.zeros(R * MP, dtype=torch.int32, device="cuda")   # theirs: r RP + j % RP, packed
        for i, layer in enumerate(self.layers):
            pages = R * (self.RP if self.types[i] == "sliding_attention" else MP)
            layer["k_cache"] = zeros(pages, PAGE, KVH, D)
            layer["v_cache"] = zeros(pages, PAGE, KVH, D)
        self.embed = bf(sd["model.embed_tokens.weight"])
        self.head = {"norm": bf(sd["model.norm.weight"]), "lm_head": bf(sd["lm_head.weight"]),
                     "h": zeros(T, H), "logits": zeros(T, self.V, dtype=torch.float32)}
        self.moe = (ext.TritonMoe(R, E, TOPK, self.limit) if self.moe_kind == "triton" else
                    ext.FlashinferMoe(R, E, TOPK, H, self.limit, self.shared["moe_y"], self.fi_logits)
                    if self.moe_kind == "flashinfer" else
                    None if self.native else Sm90MoeSplit(T, H, I, E, TOPK, self.shared["h2"], self.shared["act"], R))
        if self.moe_fold:
            assert ext.H_PAD == FI_GEMM2_WIDTH
            self.moe.set_fold(self.moe_tab)

    def layer_tensors(self, i: int) -> dict:
        """Layer i's tensors by graph name: its weights and cache, the shared buffers, its residual in and out."""
        ring = {"kv_indices": self.kv_indices_ring} if self.types[i] == "sliding_attention" else {}
        t = {**self.shared, **self.layers[i], **ring, "x_in": self.x_embed if i == 0 else self.x[i % 2],
             "x_out": self.x[(i + 1) % 2]}
        if self.fused_residual:   # x_in: the previous layer's x1 (its experts in y_in); the gate's x1 is x_out
            # fold: the experts' table row in y_in's place (the same graph inputs in the same order: the plans' node ids)
            y_in = ({"moe_tab": self.moe_tab[i]} if self.moe_fold else
                    {"y_in": self.y0 if i == 0 else self.shared["moe_y"]})
            t.update({**y_in, "x1": self.x[(i + 1) % 2]})
        return t

    def residual_in(self, i: int) -> torch.Tensor:
        """Layer i's input residual stream [T, H] (L: the head's), as the kernels compute it (a copy when fused)."""
        x = self.x_embed if i == 0 else self.x[i % 2]
        if not self.fused_residual or i == 0:
            return x.clone()
        return (x.float() + self.moe_y_of(i - 1).float()).to(x.dtype)

    def moe_y_of(self, i: int) -> torch.Tensor:
        """Layer i's experts' sum [T, H] bf16 after it ran (rows past R zero); fold: FlashInfer's finalize of its
        unfinalized outputs (FlashinferMoe.y_of), else moe_y (the last layer's)."""
        if not self.moe_fold:
            return self.shared["moe_y"]
        y = torch.zeros_like(self.shared["moe_y"])
        y[:self.R] = self.moe.y_of(self.moe.keep[i])
        return y

    def routed_experts(self, R: int) -> torch.Tensor:
        """The last layer's experts of the first R requests [R, TOPK], ascending, as the gate left them (ranks, or the
        logits' top-k when FlashInfer takes it)."""
        rt = self.shared["routing"]
        if getattr(self, "fi_logits", False):
            T = rt.shape[1]
            lg = rt.reshape(-1).view(torch.bfloat16)[:T * self.E].view(T, self.E)[:R].float()
            return lg.topk(self.TOPK, -1).indices.sort(-1).values
        return (rt[:self.E, :R] > 0).t().nonzero()[:, 1].view(R, -1)

    def x1_of(self, i: int) -> torch.Tensor:
        """Layer i's x1 (x_in + attention) after it ran."""
        return self.x[(i + 1) % 2] if self.fused_residual else self.shared["x1"]

    def head_tensors(self, prompt: torch.Tensor, gen: torch.Tensor) -> dict:
        """The final norm and lm_head, then the next step's tokens (prompt [R, P] i64; gen [R, NEW] i64: the generated
        tokens)."""
        y_in = {"moe_tab": self.moe_tab[self.L]} if getattr(self, "moe_fold", False) else {"y_in": self.shared.get("moe_y")}
        fused = {**y_in, "x_res": self.shared["x_res"]} if self.fused_residual else {}
        return {"x": self.x[self.L % 2], **self.head, **fused, "embed": self.embed, "prompt": prompt, "gen": gen,
                "x_embed": self.x_embed, "kv_indptr": self.shared["kv_indptr"],
                "last_page_len": self.shared["last_page_len"], "kv_indices": self.shared["kv_indices"],
                "kv_indices_ring": self.kv_indices_ring}

    def set_position(self, pos: int):
        """The page table for a step of every request at position pos (its K / V go to slot pos; request r's pages
        r MP .., in order)."""
        n, MP = pos // PAGE + 1, self.max_seq // PAGE
        self.shared["kv_indptr"].copy_(torch.arange(self.R + 1, dtype=torch.int32, device="cuda") * n)
        r, j = torch.arange(self.R, device="cuda")[:, None], torch.arange(n, device="cuda")
        self.shared["kv_indices"][:self.R * n].copy_((r * MP + j).flatten())
        self.kv_indices_ring[:self.R * n].copy_((r * self.RP + j % self.RP).flatten())
        self.shared["last_page_len"].fill_(pos + 1 - (n - 1) * PAGE)


def launch_layer(m: Model, kernel, inst: int, i: int):
    """Layer i: its kernel (instance inst), then its routed experts and combine (moe_kind): x_out = x1 + the experts'
    sum (Blackwell's native experts: inside the kernel)."""
    if i == 0 and getattr(m, "moe_fold", False):   # a capture's table restore, before the step's first kernel
        m.moe.begin_step()
    kernel.launch_instances(inst, 1)
    t, L = m.shared, m.layers[i]
    if m.native:
        return
    if m.moe_kind == "flashinfer":   # the experts' sum into moe_y (the next norm adds x1); fold: into moe_tab row i + 1
        m.moe.launch(t["routing"], t["topk_w"], t["h2"], m.texperts[i], layer=i)
        return
    if m.moe_kind != "mpk":
        m.moe.launch(t["routing"], t["topk_w"], t["h2"], t["x1"], m.x[(i + 1) % 2], m.texperts[i])
        return
    m.moe.launch(t["routing"], L["w13"], L["b13"], t["act"], L["w2"], L["b2"], t["eo"], t["topk_w"], t["x1"],
                 m.x[(i + 1) % 2], m.done, m.limit, 1.702)


def warm_moe(m: Model):
    """FlashInfer's MoE: its tactic search for this batch once before the steps (as vLLM's warm-up; the layers share
    their shapes)."""
    if m.moe_kind == "flashinfer":
        t = m.shared
        m.moe.autotune(t["routing"], t["topk_w"], t["h2"], m.texperts[0])
        if getattr(m, "moe_fold", False):   # the unfinalized outputs' contract on this GPU and FlashInfer build
            print(f"fold: FlashInfer finalize contract {m.moe.verify_fold(t['routing'], t['topk_w'], t['h2'], m.texperts[0])}",
                  flush=True)


def build_layer(m: Model, mpk: StaticMegakernel, t: dict, window: int):
    """The decoder layer up to the router: x1 = x_in + attn(norm(x_in)), h2 = norm(x1) and its routing (launch_layer
    then runs the experts)."""
    mpk.max_bufs = max(getattr(mpk, "max_bufs", 0), 64)   # the layer's tensors (Hopper's default; Blackwell's is 24)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    fused = "y_in" in A or "moe_tab" in A   # the residual stream x_in + y_in, into x_res (Model.fused_residual)
    first_norm(m, mpk, A, A["x_in"], A["ln1"])
    mpk.gemm_tile_layer(input=A["h"], weight=A["w_qkv"], output=A["qkv32"], combine="store",
                        store_parts=A["qkv32"].dim(0) // A["h"].dim(0), min_parts=min_kparts("qkv"))
    qkv_bias = dict(qkv_bias=A["b_qkv"], qkv_staged=A["qkv"])   # the attention tasks add the bias (no bias node)
    if mpk.target_cc == 100 and os.environ.get("GPTOSS_ATTN") == "sm100":   # A / B: MPK's Blackwell paged attention
        mpk.sm100_attention_layer(A["qkv32"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                  A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["attn"], num_q_heads=m.QH,
                                  window=window, sinks=A["sinks"], **qkv_bias)
    elif window and mpk.target_cc == 100:   # Blackwell: the window's tiles (the ring's pages) in one split a request,
        mpk.sm90_attention_split_layer(A["qkv32"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                       A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["attn_parts"],
                                       num_q_heads=m.QH, qkv_bias=A["b_qkv"], window=window,   # which writes the output
                                       output=A["attn"], sinks=A["sinks"])
    elif window:   # a sliding window: its two tiles in one task per KV head
        mpk.sm90_attention_layer(A["qkv32"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                 A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["attn"], num_q_heads=m.QH,
                                 window=window, sinks=A["sinks"], **qkv_bias)
    else:        # full attention: the KV split over the SMs (a few splits a request and KV head), then their merge (with the sinks)
        mpk.sm90_attention_split_layer(A["qkv32"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                       A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["attn_parts"],
                                       num_q_heads=m.QH, qkv_bias=A["b_qkv"])
        mpk.sm90_attention_merge_layer(A["attn_parts"], A["qo_indptr"], A["attn"], num_q_heads=m.QH, sinks=A["sinks"])
    mpk.gemm_tile_layer(input=A["attn"], weight=A["w_o"], output=A["o32"], combine="store",
                        store_parts=A["o32"].dim(0) // A["h"].dim(0), min_parts=min_kparts("o"))
    mpk.sm90_gate_layer(A["o32"], A["b_o"], A["x_res"] if fused else A["x_in"], A["ln2"], A["w_router"],
                        A["b_router"], A["num_tokens"], A["x1"], A["h2"], A["topk_w"], A["routing"], eps=m.eps,   # router splits: the plan's
                        topk_ids=getattr(m, "moe_kind", "") == "flashinfer" and not getattr(m, "fi_logits", False),
                        logits_only=getattr(m, "fi_logits", False),
                        h_mxfp8=A["h2"].dim(1) * 32 // 33 if A["h2"].dtype == uint8 else 0)
    if "z_q" in A:   # Blackwell's native experts in the layer kernel: h2 as MXFP8, then x_out = x1 + the experts
        mpk.gpt_oss_mxfp8_quant_layer(A["h2"], A["z_q"])
        mpk.gpt_oss_moe_experts_layer(A["z_q"], A["routing"], A["topk_w"], A["x1"], A["w13"], A["w13_scales"], A["w2"],
                                      A["w2_scales"], A["b13"], A["b2"], A["x_out"])
    if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the QKV weight's first MB into L2 at the start (as the GLM demos)
        mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))


def first_norm(m: Model, mpk: StaticMegakernel, A: dict, x, weight):
    """A kernel's first node: h = norm(x), with the previous layer's experts added first when fused (x_res = x + y):
    y read from y_in, or (fold, moe_tab) FlashInfer's finalize computed in the task (gpt_oss_moe_norm)."""
    if "moe_tab" in A:
        mpk.gpt_oss_moe_norm_layer(x, A["moe_tab"], weight, A["h"], A["x_res"], m.eps, top_k=m.TOPK,
                                   gemm2_width=FI_GEMM2_WIDTH)
    else:
        mpk.sm90_rmsnorm_layer(x, weight, A["h"], m.eps,
                               **(dict(add=A["y_in"], sum_out=A["x_res"]) if "y_in" in A else {}))


def build_head(m: Model, mpk: StaticMegakernel, t: dict):
    mpk.max_bufs = max(getattr(mpk, "max_bufs", 0), 64)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    first_norm(m, mpk, A, A["x"], A["norm"])
    mpk.gemm_tile_layer(input=A["h"], weight=A["lm_head"], output=A["logits"], combine="store")
    # the next step's token (the prompt's, else the argmax), its embedding row and page table: no launch between the steps
    mpk.sm90_next_token_layer(A["logits"], A["embed"], A["prompt"], A["x_embed"], A["gen"], A["kv_indptr"],
                              A["last_page_len"], A["kv_indices"], PAGE, kv_indices_ring=A["kv_indices_ring"],
                              ring_pages=m.RP)


def plan_path(m: Model, plans: str, kind: str) -> str:
    """The plan file of a kernel: one per graph (layer kind, expert count, requests, token rows past 8, the native
    experts' layers: their own graph)."""
    native = "_native" if getattr(m, "native", False) else ""
    return os.path.join(plans, f"{kind}_e{m.E}_r{m.R}" + (f"_t{T}" if T != 8 else "") + native + ".json")


def load_plan(mpk, path: str):
    """The plan file at path if it exists and has lists for this GPU's SMs, else None (round robin)."""
    if not path or not os.path.exists(path):
        return None
    p = compiler.plan_from_file(path)
    if len(p["lists"]) != 1 or len(p["lists"][0]) != mpk._num_sms():
        print(f"{path}: lists for another GPU ({len(p['lists'][0])} SMs, this one {mpk._num_sms()}): round robin")
        return None
    return p


def compile_kernel(name, build, t, out_dir, plan=None, profile=False):
    """Build one static kernel for a layer kind or the model head: plan a plan file (load_plan), else round robin."""
    mpk = StaticMegakernel(num_gpus=1, target_cc=CC)
    build(mpk, t)
    p = load_plan(mpk, plan)
    print(f"{name}: plan {plan if p else 'round robin'}", flush=True)
    paths, _ = compiler.compile_plan(mpk, p or plan_gen.round_robin_plan(mpk), os.path.join(out_dir, name, "schedules"))
    return mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out_dir, name, "build"), extra_flags=flags(),
                              profile=profile)


def tune(m: Model, kinds: dict, out: str, plans: str, costs: str, rounds: int, time_limit_s: float, tokens: torch.Tensor,
         reps: int = 10):
    """--tune: per kernel kind, the cost collection (plan_gen.collection_plans and the current plan, each a timing build run
    reps times with L2 emptied, the page table at mid-sequence, every layer input the embedding rows of tokens, one a request,
    so that the requests route as real ones: cost_records appended to costs), then search.search_plan from the current plan
    -> plan_path; last, the span of both plans (serving build). Builds run 8 at a time (the build cache)."""
    from concurrent.futures import ThreadPoolExecutor
    os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE", os.path.join(out, "build_cache"))
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
    os.makedirs(plans, exist_ok=True)
    rows = m.embed[tokens]
    for x in (m.x_embed, *m.x):
        x.zero_()
        x[:len(tokens)].copy_(rows)

    def graph(build, t):
        mpk = StaticMegakernel(num_gpus=1, target_cc=CC)
        build(mpk, t)
        return mpk

    def compiled(build, t, plan, tag, profile, build_only=False):
        mpk = graph(build, t)
        paths, _ = compiler.compile_plan(mpk, plan, os.path.join(out, tag, "schedules"))
        sk = mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out, tag, "build"), extra_flags=flags(),
                                profile=profile, build_only=build_only)
        return mpk, paths, sk

    def launch(h):
        m.set_position(m.max_seq // 2)
        h.launch(256 << 20, timeout_s=30)

    for kind, (build, t) in kinds.items():
        mpk = graph(build, t)
        cur = load_plan(mpk, plan_path(m, plans, kind)) or plan_gen.round_robin_plan(mpk)
        todo = [("cur", cur)] + plan_gen.collection_plans(mpk, rounds=rounds)
        print(f"tune {kind}: {len(todo)} plans", flush=True)
        with ThreadPoolExecutor(min(16, os.cpu_count() or 8)) as ex:   # nvcc in parallel; the runs below load from the cache
            list(ex.map(lambda np_: compiled(build, t, np_[1], f"{kind}/c_{np_[0]}", True, True), todo))
        for name, plan in todo:
            mpk, paths, sk = compiled(build, t, plan, f"{kind}/c_{name}", True)
            h, launches = Harness(sk), []
            for _ in range(reps):
                launch(h)
                launches.append((h.task_times(), h.timing()))
            cost_records.append_records(costs, cost_records.records_of_run(mpk, paths, launches, f"{kind}_{name}", machine))
            print(f"  {name}: span {h.span_us():.1f} us (the last launch)", flush=True)
            sk.finalize()
        mpk = graph(build, t)
        cost = search.Costs(costs, code=static_schedule.code_tag(), gpu=machine["gpu"])
        plan, summary = search.search_plan(mpk, cost, time_limit_s=time_limit_s, start_from=cur)
        print(f"tune {kind}: solver {({k: summary.get(k) for k in ('status', 'predicted_us', 'grids')})}", flush=True)
        spans = {"cur": [], "solved": []}
        for i, (tag, p) in enumerate((("cur", cur), ("solved", plan), ("cur", cur), ("solved", plan))):
            # a build directory each: a loaded layer's .so must not be overwritten
            _, _, sk = compiled(build, t, p, f"{kind}/span_{tag}{i}", False)
            h = Harness(sk)
            for _ in range(30):
                launch(h)
                spans[tag].append(h.span_us())
            print(f"tune {kind}: span {tag} median {statistics.median(spans[tag][-30:]):.1f} us", flush=True)
            sk.finalize()
        best = "solved" if statistics.median(spans["solved"]) < statistics.median(spans["cur"]) else "cur"
        compiler.write_plan_file(plan_path(m, plans, kind), plan if best == "solved" else cur)   # the faster, as measured
        print(f"tune {kind}: kept the {best} plan", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--ref", default=None)
    p.add_argument("--new", type=int, default=200)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--out", default="gpt_oss_static")
    p.add_argument("--json", default=None, help="write {prompt_ids, gen, text, timing} here")
    p.add_argument("--batch", type=int, default=1, help="requests a step (at most the largest bucket)")
    p.add_argument("--tokens", type=int, default=None, help="the kernels' token rows T (default: the smallest bucket >= "
                   "--batch)")
    p.add_argument("--prompts", default=None, help="a json list of user messages: request r's is r % len (all cut to the "
                   "shortest's tokens); default the one message")
    p.add_argument("--only", default=None, help="with --prompts: one request, message ONLY; ONLY a list (K,K,...: one "
                   "request, each message in turn in one process; --json: the references of all of them)")
    p.add_argument("--plans", default=None, help="the plan files' directory (plan_path; a missing file: round robin; "
                   "default plans/h100 or plans/b200 by the GPU)")
    p.add_argument("--tune", metavar="COSTS", default=None, help="collect cost records into COSTS, solve each kernel's "
                   "plan into --plans (tune), and exit")
    p.add_argument("--rounds", type=int, default=2, help="with --tune: random plans per candidate grid of each node")
    p.add_argument("--time", type=float, default=300, help="with --tune: the solver's time limit (s)")
    p.add_argument("--eager", action="store_true", help="launch every step from Python (default: a CUDA graph a step)")
    args = p.parse_args()
    args.plans = args.plans or os.path.join(PLANS, "b200" if CC == 100 else "h100")
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    R = args.batch
    global T
    T = args.tokens or next(b for b in BUCKETS if b >= R)
    assert T % 8 == 0 and R <= T, f"--batch {R}, --tokens {T}: R <= T, T a multiple of 8"
    ref = json.load(open(args.ref)) if args.ref else None
    only = None
    if args.prompts:
        msgs = json.load(open(args.prompts))
        ids = [tok(tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                           add_generation_prompt=True)).input_ids for c in msgs]
        if ref and len(ref["prompt_ids"]) == len(ids) and isinstance(ref["prompt_ids"][0], list):
            ids = ref["prompt_ids"]   # the reference's own (the chat template stamps the date into the system prompt)
        P = min(len(i) for i in ids)
        ids = [i[:P] for i in ids]
        only = [int(k) for k in args.only.split(",")] if args.only is not None else None
        assert only is None or R == 1, "--only: one request"
        prompts = [ids[only[0]]] if only else [ids[r % len(ids)] for r in range(R)]
    elif ref:
        prompts = [ref["prompt_ids"]] * R
    else:
        text = tok.apply_chat_template([{"role": "user", "content": "Give me a short introduction to large language models."}],
                                       tokenize=False, add_generation_prompt=True)
        prompts = [tok(text).input_ids] * R
    P, NEW = len(prompts[0]), args.new
    t0 = time.time()
    m = Model(args.ckpt, args.max_seq, R)
    assert P + NEW <= m.max_seq
    print(f"weights loaded and converted in {time.time() - t0:.0f} s", flush=True)
    # the kernels, and one instance per layer (layer i: instance inst[i] of kernel kind[i])
    sliding = [i for i in range(m.L) if m.types[i] == "sliding_attention"]
    full = [i for i in range(m.L) if m.types[i] != "sliding_attention"]
    gen = torch.zeros(R, NEW, dtype=torch.long, device="cuda")
    prompt_t = torch.tensor(prompts, dtype=torch.long, device="cuda")
    kinds = {"sliding": (lambda mpk, t: build_layer(m, mpk, t, m.window), m.layer_tensors(sliding[0])),
             "full": (lambda mpk, t: build_layer(m, mpk, t, 0), m.layer_tensors(full[0])),
             "head": (lambda mpk, t: build_head(m, mpk, t), m.head_tensors(prompt_t, gen))}
    if args.tune:
        tune(m, kinds, os.path.join(args.out, "tune"), args.plans, args.tune, args.rounds, args.time,
             prompt_t[torch.arange(R), (P // 2 + torch.arange(R)) % P])   # a distinct token a request
        return
    k_sl, k_fu, k_hd = (compile_kernel(k, b, t, args.out, plan_path(m, args.plans, k)) for k, (b, t) in kinds.items())
    kernel_of, inst = {}, {}
    for i in range(m.L):
        kernel_of[i] = k_sl if i in sliding else k_fu
        inst[i] = kernel_of[i].add_instance(m.layer_tensors(i))
    k_hd.add_instance(m.head_tensors(prompt_t, gen))
    warm_moe(m)

    def step():
        """One position through the model; the head leaves the next position's token, embedding row and page table."""
        for i in range(m.L):
            launch_layer(m, kernel_of[i], inst[i], i)
        k_hd.launch_instances(0, 1)

    step_graph = []

    def run_step(pos):
        """step(), from the fourth position as one CUDA graph (captured once, replayed: the step's tensors and page
        table stay put); launched eagerly from Python, many requests' steps are host-bound (triton_moe's launches)."""
        if args.eager or pos < 3:
            step()
            return
        if not step_graph:
            torch.cuda.synchronize()
            step_graph.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(step_graph[0]):
                step()
        step_graph[0].replay()

    def generate(runs):
        """The prompts then NEW tokens, `runs` times (the last timed); gen holds the tokens."""
        for run in range(runs):   # the first run warms up (and checks); the second is timed
            for layer in m.layers:
                layer["k_cache"].zero_()
                layer["v_cache"].zero_()
            m.set_position(0)   # position 0: the prompts' first tokens
            m.x_embed[:R].copy_(m.embed[prompt_t[:, 0]])
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(P + NEW)]
            s.record()
            for pos in range(P + NEW - 1):
                run_step(pos)
                marks[pos].record()
            e.record()
            torch.cuda.synchronize()
            total = s.elapsed_time(e)
            per = [marks[i - 1].elapsed_time(marks[i]) for i in range(1, P + NEW - 1)]
            # as the vLLM / SGLang sweep's (T(NEW tokens) - T(NEW / 2 tokens)) / (NEW / 2): the later half's steps
            dec = marks[P + NEW // 2 - 2].elapsed_time(marks[P + NEW - 2]) / (NEW - NEW // 2)
            print(f"run {run}: {P + NEW - 1} positions in {total:.1f} ms; per step median {statistics.median(per) * 1e3:.0f} "
                  f"us (min {min(per) * 1e3:.0f}, max {max(per) * 1e3:.0f}); decode {dec * 1e3:.0f} us/step, batch {R}: "
                  f"{R / dec * 1e3:.0f} tok/s", flush=True)
        return total, per, dec

    if only and len(only) > 1:   # one request a message in turn: their references
        gens = []
        for k in only:
            prompt_t[0].copy_(torch.tensor(ids[k], device="cuda"))
            generate(1)
            gens.append(gen[0].tolist())
            print(f"message {k}: {tok.decode(gens[-1])[:80]!r}", flush=True)
        if args.json:
            json.dump({"prompt_ids": [ids[k] for k in only], "gen": gens}, open(args.json, "w"))
        for k in (k_sl, k_fu, k_hd):
            k.finalize()
        return
    total, per, dec = generate(2)
    out = gen.tolist()
    text = tok.decode(out[0])
    print("text:", repr(text[:400]))
    if ref:
        refs = ref["gen"] if isinstance(ref["gen"][0], list) else [ref["gen"]]
        for r in range(R):
            want = refs[r % len(refs)]
            same = next((i for i, (a, b) in enumerate(zip(out[r], want)) if a != b), min(len(out[r]), len(want)))
            print(f"request {r}: tokens equal to the reference's for the first {same} of {min(len(out[r]), len(want))}")
    if args.json:
        json.dump({"prompt_ids": prompts if R > 1 else prompts[0], "prompt_len": P, "gen": out if R > 1 else out[0],
                   "text": text, "total_ms": total, "step_us_median": statistics.median(per) * 1e3,
                   "decode_us": dec * 1e3}, open(args.json, "w"))
    for k in (k_sl, k_fu, k_hd):
        k.finalize()


if __name__ == "__main__":
    main()
