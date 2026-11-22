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

"""GPT-OSS's routed experts on Blackwell with FlashInfer's trtllm-gen MXFP4 MoE (bf16 activations: the kernel vLLM runs
for GPT-OSS on B200, its FLASHINFER_TRTLLM_MXFP4_BF16 backend), fed the step's routing from the gate layer ([E, T] ranks
1 + slot, [T, TOPK] weights): x_out = x1 + the token's weighted experts. The weights are prepared as vLLM prepares them
for that backend (vllm/model_executor/layers/fused_moe/oracle/mxfp4.py): hidden padded to 3072 and the intermediate to
2944 (zeros after the checkpoint's rows), each gate / up row pair swapped (trtllm-gen's SwiGLU takes up first), then
trtllm-gen's row shuffle for its transposed MMA output and the scales' interleave. FlashInfer is not a Mirage dependency;
without it gpt_oss_static.py on Blackwell falls back to the native gpt_oss_moe_experts task."""
import os

import torch
from flashinfer import trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_routed_moe
from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
from flashinfer.fused_moe.core import RoutingMethodType, get_w2_permute_indices_with_cache

H_PAD, I_PAD, EPILOGUE_TILE_M = 3072, 2944, 128   # vLLM's round-ups for the TRTLLM backends (hidden 256, intermediate 128)
_PERMUTE_CACHE = {}


def _pad(t, shape):
    """t in the front of a zero tensor of shape (as vLLM's gpt_oss_mxfp4 weight loader: param[:, :d1, :d2].copy_)."""
    out = t.new_zeros(shape)
    out[tuple(slice(0, s) for s in t.shape)].copy_(t)
    return out


def _swap_pairs(x, axis):
    """Each pair of rows along axis swapped (vLLM: trtllm-gen's SwiGLU takes the up row before the gate row)."""
    shape = list(x.shape)
    axis %= len(shape)
    return x.reshape(*shape[:axis], shape[axis] // 2, 2, *shape[axis + 1:]).flip(axis + 1).reshape(shape)


def _shuffle(w, sf, bias):
    """One expert's [N, K/2] u8 weight, [N, K/32] u8 scales, [N] fp32 bias in trtllm-gen's order."""
    p = get_w2_permute_indices_with_cache(_PERMUTE_CACHE, w, EPILOGUE_TILE_M)
    w = w[p.to(w.device)].contiguous()
    p = get_w2_permute_indices_with_cache(_PERMUTE_CACHE, sf, EPILOGUE_TILE_M, num_elts_per_sf=16)
    sf = nvfp4_block_scale_interleave(sf[p.to(sf.device)].contiguous())
    b = bias.reshape(-1, 1)
    p = get_w2_permute_indices_with_cache(_PERMUTE_CACHE, b, EPILOGUE_TILE_M)
    return w, sf, b[p.to(b.device)].contiguous().reshape(-1)


def expert_weights(blocks13, scales13, bias13, blocks2, scales2, bias2):
    """A layer's experts from the checkpoint (blocks [E, N, K/32, 16] u8, scales [E, N, K/32] u8, bias [E, N]) on the
    GPU in the trtllm-gen MXFP4 layout."""
    E = blocks13.shape[0]
    w13 = _swap_pairs(_pad(blocks13.reshape(E, blocks13.shape[1], -1).cuda(), (E, 2 * I_PAD, H_PAD // 2)), -2)
    s13 = _swap_pairs(_pad(scales13.cuda(), (E, 2 * I_PAD, H_PAD // 32)), -2)
    b13 = _swap_pairs(_pad(bias13.float().cuda(), (E, 2 * I_PAD)), -1)
    w2 = _pad(blocks2.reshape(E, blocks2.shape[1], -1).cuda(), (E, H_PAD, I_PAD // 2))
    s2 = _pad(scales2.cuda(), (E, H_PAD, I_PAD // 32))
    b2 = _pad(bias2.float().cuda(), (E, H_PAD))
    g1, g2 = [_shuffle(w13[e], s13[e], b13[e]) for e in range(E)], [_shuffle(w2[e], s2[e], b2[e]) for e in range(E)]
    stack = lambda xs, i: torch.stack([x[i] for x in xs])
    return {"fw13": stack(g1, 0), "fs13": stack(g1, 1).reshape(E, 2 * I_PAD, H_PAD // 32).view(torch.float8_e4m3fn),
            "fb13": stack(g1, 2), "fw2": stack(g2, 0),
            "fs2": stack(g2, 1).reshape(E, H_PAD, I_PAD // 32).view(torch.float8_e4m3fn), "fb2": stack(g2, 2)}


class FlashinferMoe:
    def __init__(self, R, E, TOPK, H, limit, y, logits=False, alpha=1.702, beta=1.0):
        """logits: the gate leaves bf16 router logits [T][E] at routing's start and FlashInfer's routing takes the top-k
        (softmax over the top-k: Renormalize), else the gate's top-k ids and weights."""
        self.R, self.E, self.TOPK, self.H, self.logits = R, E, TOPK, H, logits
        dev, f32 = "cuda", torch.float32
        self.alpha = torch.full((E,), alpha, dtype=f32, device=dev)
        self.beta = torch.full((E,), beta, dtype=f32, device=dev)
        self.limit = torch.full((E,), limit, dtype=f32, device=dev)
        self.y = y[:R]   # the experts' sum [R, H] bf16 (vLLM's unpadded output; the next layer's norm adds it)
        self.table = None   # fold mode (set_fold): FlashInfer's finalize is the next layer's gpt_oss_moe_norm

    def set_fold(self, table):
        """Fold mode: FlashInfer runs unfinalized (do_finalize=False; the call passes output=y still, so that its
        autotuner key and tactic stay those of the finalized call) and each layer's gpt_oss_moe_norm task computes the
        finalize (tasks/gpt_oss_moe_norm.cuh). table int64 [L + 1, 4] on the GPU: row j is what layer j's norm reads (L:
        the head's), {gemm2_output, expanded_idx_to_permuted_idx, expert weights addresses, R}; row 0 stays {0, 0, 0,
        0} (no experts before layer 0). Layer i's call writes row i + 1: eagerly a copy from host memory (stream
        ordered, before layer i + 1's kernel); in a CUDA graph capture its addresses are the capture's own, written to
        that capture's pinned host table, which begin_step's copy at the step's start (captured: replayed) puts in the
        table, so that each replay restores its own addresses whatever ran before it."""
        assert self.logits, "fold mode: FlashInfer's routing from logits (GPTOSS_FI_ROUTE=logits)"
        assert table.dtype == torch.int64 and table.is_cuda and table.is_contiguous() and table.shape[1] == 4
        self.table = table
        self.keep = {}          # layer -> its last call's outputs (alive while the table points at them)
        self.capture_rows = None
        self.captures = []      # every capture's pinned rows (a replay reads them)

    def begin_step(self):
        """Fold mode, before layer 0's kernel of a step: in a capture, the copy that restores the capture's table."""
        if self.table is None:
            return
        if torch.cuda.is_current_stream_capturing():
            rows = torch.zeros(self.table.shape, dtype=torch.int64).pin_memory()
            self.captures.append(rows)
            self.capture_rows = rows
            self.table.copy_(rows, non_blocking=True)   # a memcpy node: reads rows at each replay
        else:
            self.capture_rows = None

    def _fold_out(self, out, i):
        """Fold mode: layer i's unfinalized outputs [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]
        checked against the task's contract, their addresses into table row i + 1."""
        R, TOPK = self.R, self.TOPK
        g2, w, idx = out
        assert g2.dtype == torch.bfloat16 and g2.dim() == 2 and g2.shape[1] == H_PAD and g2.is_contiguous(), \
            f"gemm2_output {g2.dtype} {tuple(g2.shape)}: the task reads rows of {H_PAD} bf16"
        assert w.dtype == torch.bfloat16 and tuple(w.shape) == (R, TOPK) and w.is_contiguous(), (w.dtype, w.shape)
        assert idx.dtype == torch.int32 and idx.numel() == R * TOPK and idx.is_contiguous(), (idx.dtype, idx.shape)
        assert g2.data_ptr() % 16 == 0
        # the task reproduces finalizeKernel, which FlashInfer 0.7.2.post1's finalize::run picks while
        # ceil(H / 256) x R < 1184 (R < 99; finalizeKernelVecLoad from there) and top-k <= 4 (SmallBatch past it)
        assert -(-self.H // 256) * R < 1184 and TOPK <= 4, (R, TOPK)
        self.keep[i] = (g2, w, idx)
        vals = [g2.data_ptr(), idx.data_ptr(), w.data_ptr(), R]
        if torch.cuda.is_current_stream_capturing():
            assert self.capture_rows is not None, "fold mode: a capture starts with begin_step (launch_layer's layer 0)"
            self.capture_rows[i + 1] = torch.tensor(vals, dtype=torch.int64)
        else:   # pageable: the copy has read vals when it returns
            self.table[i + 1].copy_(torch.tensor(vals, dtype=torch.int64))

    def launch(self, routing, topk_w, h2, L, layer=None, do_finalize=None, output=None):
        """routing [E + TOPK, T]: the gate's bf16 logits [T][E] at its start (self.logits), else its top-k ids [T][TOPK]
        in its rows past E; h2: its h, [T, H_PAD] bf16 or MXFP8
        (uint8 [T, H_PAD 33 / 32]: e4m3 [T][H_PAD], then UE8M0 scales [T][H_PAD / 32]) (sm90_gate_layer topk_ids,
        h_mxfp8). Fold mode (set_fold): unfinalized, layer's outputs into the table (do_finalize / output: verify_fold's
        override, no table write)."""
        R, E, TOPK, H = self.R, self.E, self.TOPK, self.H
        override = do_finalize is not None
        fold = (not do_finalize) if override else self.table is not None
        assert override or not fold or layer is not None, "fold mode: launch(..., layer=i)"
        T = routing.shape[1]
        assert routing.shape[0] == E + TOPK
        if h2.dtype == torch.uint8:
            assert h2.shape[1] == H_PAD * 33 // 32
            flat, e4m3 = h2.view(-1), torch.float8_e4m3fn
            x = flat[:T * H_PAD].view(T, H_PAD)[:R].view(e4m3)
            xs = flat[T * H_PAD:].view(T, H_PAD // 32)[:R].view(e4m3)
        else:
            assert h2.dtype == torch.bfloat16 and h2.shape[1] == H_PAD
            x, xs = h2[:R], None
        w = dict(hidden_states=x, hidden_states_scale=xs, gemm1_weights=L["fw13"], gemm1_weights_scale=L["fs13"],
                 gemm1_bias=L["fb13"], gemm1_alpha=self.alpha, gemm1_beta=self.beta, gemm1_clamp_limit=self.limit,
                 gemm2_weights=L["fw2"], gemm2_weights_scale=L["fs2"], gemm2_bias=L["fb2"], output1_scale_scalar=None,
                 output1_scale_gate_scalar=None, output2_scale_scalar=None, num_experts=E, top_k=TOPK, n_group=None,
                 topk_group=None, intermediate_size=I_PAD, local_expert_offset=0, local_num_experts=E,
                 routed_scaling_factor=None, routing_method_type=RoutingMethodType.Renormalize, do_finalize=not fold,
                 enable_pdl=True, output=self.y if output is None else output, tune_max_num_tokens=max(R, 8))
        if self.logits:   # bf16 logits [T][E] at routing's start; FlashInfer's routing: top-k, softmax over it
            logits = routing.reshape(-1).view(torch.bfloat16)[:T * E].view(T, E)[:R]
            out = trtllm_fp4_block_scale_moe(routing_logits=logits, routing_bias=None, **w)
        else:             # the gate's top-k ids [T][TOPK] in routing's rows past E, and its weights
            assert not fold
            ids = routing[E:].reshape(-1)[:T * TOPK].view(T, TOPK)[:R]
            out = trtllm_fp4_block_scale_routed_moe(topk_ids=(ids, topk_w[:R]), routing_bias=None, **w)
        if fold and not override:
            self._fold_out(out, layer)
        return out

    def y_of(self, out):
        """FlashInfer's finalizeKernel on unfinalized outputs [gemm2_output, expert_weights,
        expanded_idx_to_permuted_idx] in torch: [R, H] bf16 = bf16(sum over k ascending, idx != -1, of w gemm2[idx][:H])
        in fp32 from 0 (a product of two bf16 is exact in fp32, so each step rounds once as an fma; off only where the
        fma's flush to zero would act)."""
        g2, w, idx = out
        R, TOPK, H = self.R, self.TOPK, self.H
        idx = idx.view(R, TOPK).long()
        acc = torch.zeros(R, H, dtype=torch.float32, device=g2.device)
        for k in range(TOPK):
            p = idx[:, k]
            e = g2[p.clamp(min=0), :H].float()
            acc = torch.where((p != -1)[:, None], acc + w[:, k:k + 1].float() * e, acc)
        return acc.to(torch.bfloat16)

    def verify_fold(self, routing, topk_w, h2, L):
        """Fold mode's contract on this GPU and FlashInfer build: the finalized call's output equals y_of of the
        unfinalized call's (same inputs, bit for bit), and the unfinalized outputs match the task's layout. Raises."""
        ref = torch.zeros_like(self.y)
        self.launch(routing, topk_w, h2, L, do_finalize=True, output=ref)
        out = self.launch(routing, topk_w, h2, L, do_finalize=False)
        g2, w, idx = out
        assert g2.dtype == torch.bfloat16 and g2.shape[1] == H_PAD and g2.is_contiguous(), (g2.dtype, g2.shape)
        assert w.dtype == torch.bfloat16 and tuple(w.shape) == (self.R, self.TOPK), (w.dtype, w.shape)
        assert idx.dtype == torch.int32 and idx.numel() == self.R * self.TOPK, (idx.dtype, idx.shape)
        y = self.y_of(out)
        torch.cuda.synchronize()
        bad = (y.view(torch.int16) != ref.view(torch.int16)).sum().item()
        assert bad == 0, f"verify_fold: {bad} of {y.numel()} elements differ from FlashInfer's finalize"
        assert ref.abs().sum().item() > 0, "verify_fold: zero output (inputs not routed)"
        return {"elements": y.numel(), "minus_one_slots": int((idx == -1).sum().item())}

    def autotune(self, routing, topk_w, h2, L):
        """        FlashInfer's tactic search for this batch (as vLLM's warm-up): run once under its autotuner. The fold mode's
        unfinalized call shares the finalized call's tuning key (the same output tensor, shapes and runner), so both
        searches are this one call's. GPTOSS_FI_TUNE_CACHE: a tactic file loaded if it exists and saved after (one
        tactic for both arms of an A / B)."""
        from flashinfer.autotuner import autotune
        cache = os.environ.get("GPTOSS_FI_TUNE_CACHE") or None
        with autotune(True, cache=cache):
            self.launch(routing, topk_w, h2, L, do_finalize=True)
        torch.cuda.synchronize()
