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

"""GPT-OSS's routed experts at larger batches with triton_kernels' matmul_ogs (what SGLang / vLLM run on Hopper): the
checkpoint's MXFP4 experts in its layout, the step's routing (the gate layer's [E, T] ranks and [T, TOPK] weights) turned
into its ragged metadata, then W13 + SwiGLU and W2 with the routing weights and the sum over a token's experts;
x_out = x1 + that. Rows past the step's R requests are not computed. triton_kernels is not a Mirage dependency: SGLang's
and vLLM's installs ship it (or pip install it from the Triton repository's python/triton_kernels); without it
gpt_oss_static.py falls back to sm90_moe_split."""
import torch
import triton
import triton.language as tl
import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import FlexCtx, FnSpecs, FusedActivation, PrecisionConfig, matmul
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details.ragged_tensor import RaggedTensorMetadata
from triton_kernels.tensor_details import layout

# split_k 1 as SGLang's Hopper path; 4 warps (the scales swizzled to match): more CTAs an SM, so the W2 grid's last
# partial wave shrinks — 120B, 64 tokens on 57 experts, W13 + W2 341 -> 286 us (8 warps: SGLang's / vLLM's default)
NUM_WARPS = 4
opt_flags.update_opt_flags_constraints({"split_k": 1, "num_warps": NUM_WARPS})


def _swizzle(q, s, num_warps=NUM_WARPS):
    """[E, N, K/2] u8 and [E, N, K/32] u8 scales -> matmul_ogs' MXFP4 operand (K padded to 128 for the scales)."""
    want = (q.shape[-1] * 2 + 127) // 128 * 128 // 32
    if s.shape[-1] < want:
        s = torch.nn.functional.pad(s, (0, want - s.shape[-1]), value=127)
    q, s = q.transpose(-2, -1), s.transpose(-2, -1)
    w = convert_layout(wrap_torch_tensor(q, dtype=FP4), layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2))
    sc = convert_layout(wrap_torch_tensor(s), layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2,
                                                                                               num_warps=num_warps))
    return w, PrecisionConfig(b_mx_scale=sc, flex_ctx=FlexCtx(rhs_data=InFlexData()),
                              b_microblock_size=int(MXFP_BLOCK_SIZE))


def expert_weights(blocks13, scales13, bias13, blocks2, scales2, bias2):
    """A layer's experts from the checkpoint (gate / up rows interleaved, as matmul_ogs' SwiGLU wants)."""
    flat = lambda b: b.reshape(b.shape[0], b.shape[1], -1).cuda()
    w13, pc13 = _swizzle(flat(blocks13), scales13.cuda())
    w2, pc2 = _swizzle(flat(blocks2), scales2.cuda())
    return {"tw13": w13, "tpc13": pc13, "tb13": bias13.float().cuda(), "tw2": w2, "tpc2": pc2,
            "tb2": bias2.float().cuda()}


@triton.jit
def _route(routing, T, R, topk_w, h2, hb, sizes, offs, sched, n_sched, combine, gather, gammas, H,
           E: tl.constexpr, EP: tl.constexpr, RP: tl.constexpr, TOPK: tl.constexpr, NBS: tl.constexpr,
           BS0_LOG2: tl.constexpr, MAXB: tl.constexpr, SP: tl.constexpr, HP: tl.constexpr):
    """One program: matmul_ogs' routing inputs straight from the gate's [E, T] ranks (1 + slot, 0 unchosen) and [T, TOPK]
    weights, in place of topk + the bitmatrix and ragged metadata kernels: per expert its row count (sizes) and offset
    (offs row 0), its blocks of 16, 32, 64, 128 rows (offs rows 1.., the schedule: blk << 16 | expert, -1 padded); the
    rows grouped by expert, a request's in request order: combine = r TOPK + slot (W2's scatter), gather = r (W13's), the
    weight. Program 1..R: request r - 1's f16 MoE input as bf16 (matmul_ogs' Hopper MXFP4 path takes bf16)."""
    pid = tl.program_id(0)
    if pid > 0:
        r = pid - 1
        c = tl.arange(0, HP)
        tl.store(hb + r * H + c, tl.load(h2 + r * H + c, mask=c < H).to(tl.bfloat16), mask=c < H)
        return
    e = tl.arange(0, EP)
    rr = tl.arange(0, RP)
    rk = tl.load(routing + e[:, None] * T + rr[None, :], mask=(e[:, None] < E) & (rr[None, :] < R), other=0)
    on = (rk > 0).to(tl.int32)
    cnt = tl.sum(on, 1)                                  # [EP]
    start = tl.cumsum(cnt, 0) - cnt
    tl.store(sizes + e, cnt, mask=e < E)
    tl.store(offs + e, start, mask=e < E)
    tl.store(offs + E, tl.sum(cnt, 0))
    pos = start[:, None] + tl.cumsum(on, 1) - on        # [EP, RP]: the row's place
    slot = tl.maximum(rk - 1, 0)
    m = rk > 0
    tl.store(combine + pos, rr[None, :] * TOPK + slot, mask=m)
    tl.store(gather + pos, rr[None, :] + 0 * slot, mask=m)
    tl.store(gammas + pos, tl.load(topk_w + rr[None, :] * TOPK + slot, mask=m, other=0.), mask=m)
    j = tl.arange(0, MAXB)
    q = tl.arange(0, SP)
    for b in tl.static_range(NBS):
        nb = (cnt + (1 << (BS0_LOG2 + b)) - 1) >> (BS0_LOG2 + b)
        bstart = tl.cumsum(nb, 0) - nb
        tl.store(offs + (b + 1) * (E + 1) + e, bstart, mask=e < E)
        tl.store(offs + (b + 1) * (E + 1) + E, tl.sum(nb, 0))
        tl.store(sched + b * n_sched + q, -1, mask=q < n_sched)
        tl.debug_barrier()
        tl.store(sched + b * n_sched + bstart[:, None] + j[None, :], (j[None, :] << 16) + e[:, None],
                 mask=(j[None, :] < nb[:, None]) & (e[:, None] < E))


@triton.jit
def _combine(Y, x1, xo, H, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    """x_out = x1 + the token's TOPK weighted expert rows, in fp32, rounded once."""
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < H
    acc = tl.load(x1 + r * H + c, mask=m).to(tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(Y + (r * TOPK + k) * H + c, mask=m).to(tl.float32)
    tl.store(xo + r * H + c, acc.to(xo.dtype.element_ty), mask=m)


class TritonMoe:
    def __init__(self, R, E, TOPK, limit, alpha=1.702):
        self.R, self.E, self.TOPK = R, E, TOPK
        self.act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), (alpha, limit))
        dev, i32 = "cuda", torch.int32
        bs = RaggedTensorMetadata.block_sizes()
        self.n_sched = RaggedTensorMetadata.max_n_blocks(E, R * TOPK)
        self.sizes = torch.zeros(E, dtype=i32, device=dev)
        self.offs = torch.zeros(len(bs) + 1, E + 1, dtype=i32, device=dev)   # row 0: slice offsets; then block offsets
        self.sched = torch.full((len(bs), self.n_sched), -1, dtype=i32, device=dev)
        self.combine = torch.zeros(R * TOPK, dtype=i32, device=dev)
        self.gather = torch.zeros(R * TOPK, dtype=i32, device=dev)
        self.gammas = torch.zeros(R * TOPK, dtype=torch.float32, device=dev)
        self.ragged = RaggedTensorMetadata(self.sizes, self.offs[0], self.offs[1:], self.sched)
        self.bs = bs
        self.hb = None

    def launch(self, routing, topk_w, h2, x1, x_out, L):
        R, E, TOPK, H = self.R, self.E, self.TOPK, h2.shape[1]
        if self.hb is None:
            self.hb = torch.empty(R, H, dtype=torch.bfloat16, device=h2.device)
        _route[(R + 1,)](routing, routing.shape[1], R, topk_w, h2, self.hb, self.sizes, self.offs, self.sched,
                         self.n_sched, self.combine, self.gather, self.gammas, H, E=E, EP=triton.next_power_of_2(E),
                         RP=triton.next_power_of_2(R), TOPK=TOPK, NBS=len(self.bs),
                         BS0_LOG2=self.bs[0].bit_length() - 1, MAXB=triton.next_power_of_2(triton.cdiv(R, self.bs[0])),
                         SP=triton.next_power_of_2(self.n_sched), HP=triton.next_power_of_2(H),
                         num_warps=8)   # 64 requests: 13 -> 7.7 us (4 warps)
        inter = matmul(self.hb, L["tw13"], L["tb13"], a_ragged_metadata=self.ragged, gather_indx=self.gather,
                       precision_config=L["tpc13"], fused_activation=self.act)
        out = matmul(inter.view(R * TOPK, -1), L["tw2"], L["tb2"], a_ragged_metadata=self.ragged,
                     scatter_indx=self.combine, precision_config=L["tpc2"], gammas=self.gammas)   # a row per (token, slot)
        _combine[(R, triton.cdiv(H, 1024))](out, x1, x_out, H, TOPK=TOPK, BLOCK=1024)
