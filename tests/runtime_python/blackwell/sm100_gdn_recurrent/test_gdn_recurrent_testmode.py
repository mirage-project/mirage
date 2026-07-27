"""Test mode: `gdn_recurrent_layer` through the full MPK pipeline.

Exercises the parts the kernel-wrapper test cannot: the Python layer API, task
registration, C++ code generation (the emitted `qo_indptr` slicing, the
per-(slot, head) state offset, the `num_k_heads` param and the `step == 0`
predicate), nvcc compilation and runtime dispatch.

Two requests prefill in the same iteration, so one compile covers:

  * per-slot token windows taken from the runtime's own `qo_indptr_buffer`
    (the two prompts have DIFFERENT lengths, so a hardcoded stride would fail)
  * per-(slot, v-head) recurrent-state pool indexing - grid.x is the head axis
    and grid.y the slot axis, the TRANSPOSE of the conv task's grid, so a
    copy-pasted metadata mapping would show up here
  * both sides of the `step == 0` predicate: request 0 is seeded with step 0
    (fresh -> stored state ignored) and request 1 with a non-zero step
    (-> stored state consumed). Seeding `step` is the only way to reach the
    carried-state branch in test mode, which runs the task graph exactly once.

Run:
    python tests/runtime_python/blackwell/sm100_gdn_recurrent/\
test_gdn_recurrent_testmode.py
"""

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_V_HEADS = 8
NUM_K_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM          # 512
VAL_DIM = NUM_V_HEADS * HEAD_V_DIM          # 1024
QKV_STRIDE = 2 * KEY_DIM + VAL_DIM          # 2048
BA_STRIDE = 2 * NUM_V_HEADS                 # 16
PROMPT_LENS = [5, 3]                        # request 0, request 1
SEEDED_STEPS = [0, 7]                       # step==0 -> zero state
MAX_SEQ_LENGTH = 64
# grid.z, the decode v-row split. Both prompts here are PREFILL chunks, so this
# exercises the codegen's "split 0 runs the whole chunk, the rest are no-ops"
# branch - i.e. it proves a split-enabled graph still produces the unsplit
# prefill result. The decode branch's bit-exactness is gated in
# test_gdn_recurrent.py [8]; AC-3 gates it end to end.
GDN_SPLIT = 4
GDN_DEPTH = 2
BF16 = torch.bfloat16
EPS = 1e-6


def ref_chain(qkv, ba, z, A_log, dt_bias, norm_w, state, zero_state):
    """Inline reference: the HF rounding order, token-sequential.

    bf16 l2norm -> fp32 recurrence -> bf16 `o` -> gated RMSNorm with the
    normalized value rounded to bf16 before the weight multiply. See
    `test_gdn_recurrent_oracle.py`, which shows this order reproduces the real
    checkpoint's `gdn.core_attn_out` / `gdn.gated_norm_out` bit-exactly.

    state is MPK's [num_v_heads, head_v_dim, head_k_dim] layout.
    """
    T = qkv.shape[0]
    S = (torch.zeros_like(state) if zero_state else state.clone()).float()
    out = torch.zeros(T, VAL_DIM, dtype=BF16, device=qkv.device)
    beta = ba[:, :NUM_V_HEADS].sigmoid().float()
    g = -A_log.float().exp() * F.softplus(
        ba[:, NUM_V_HEADS:2 * NUM_V_HEADS].float() + dt_bias.float())
    scale = 1.0 / (HEAD_K_DIM ** 0.5)
    for t in range(T):
        o_t = torch.zeros(NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float32,
                          device=qkv.device)
        for hv in range(NUM_V_HEADS):
            ih = hv // (NUM_V_HEADS // NUM_K_HEADS)
            q = qkv[t, ih * HEAD_K_DIM:(ih + 1) * HEAD_K_DIM]
            k = qkv[t, KEY_DIM + ih * HEAD_K_DIM:
                       KEY_DIM + (ih + 1) * HEAD_K_DIM]
            v = qkv[t, 2 * KEY_DIM + hv * HEAD_V_DIM:
                       2 * KEY_DIM + (hv + 1) * HEAD_V_DIM].float()
            qn = (q * torch.rsqrt((q * q).sum(-1, keepdim=True)
                                  + EPS)).float() * scale
            kn = (k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)).float()
            Sh = S[hv] * g[t, hv].exp()
            delta = (v - (Sh * kn.unsqueeze(0)).sum(-1)) * beta[t, hv]
            Sh = Sh + delta.unsqueeze(-1) * kn.unsqueeze(0)
            o_t[hv] = (Sh * qn.unsqueeze(0)).sum(-1)
            S[hv] = Sh
        ob = o_t.to(BF16).float()
        xh = (ob * torch.rsqrt(ob.pow(2).mean(-1, keepdim=True)
                               + EPS)).to(BF16).float()
        y = norm_w.float() * xh
        out[t] = (y * F.silu(z[t].reshape(NUM_V_HEADS, HEAD_V_DIM).float())
                  ).to(BF16).reshape(-1)
    return out, S


def main():
    device = "cuda"
    torch.manual_seed(20260726)

    num_requests = len(PROMPT_LENS)
    total_tokens = sum(PROMPT_LENS)

    qkv = (torch.randn(total_tokens, QKV_STRIDE, dtype=torch.float32,
                       device=device) * 0.5).to(BF16)
    ba = (torch.randn(total_tokens, BA_STRIDE, dtype=torch.float32,
                      device=device) * 0.5).to(BF16)
    z = (torch.randn(total_tokens, VAL_DIM, dtype=torch.float32,
                     device=device) * 0.5).to(BF16)
    A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device)
    alog_dtbias = torch.stack([A_log, dt_bias]).contiguous()
    norm_w = torch.ones(HEAD_V_DIM, dtype=torch.float32, device=device)
    # Non-zero garbage in every slot: slot 0 must ignore it (step == 0), slot 1
    # must consume it (step != 0).
    state = torch.randn(num_requests, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM,
                        dtype=torch.float32, device=device) * 0.1
    state_in = state.clone()
    out = torch.zeros(total_tokens, VAL_DIM, dtype=BF16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=total_tokens,
        max_num_batched_requests=num_requests,
        max_num_pages=num_requests * 4,
        page_size=MAX_SEQ_LENGTH,
        max_seq_length=MAX_SEQ_LENGTH,
        meta_tensors={
            "tokens": torch.zeros(
                (num_requests, MAX_SEQ_LENGTH), dtype=torch.int64, device=device
            ),
            "prompt_lengths": torch.tensor(
                PROMPT_LENS, dtype=torch.int32, device=device
            ),
            "step": torch.tensor(SEEDED_STEPS, dtype=torch.int32, device=device),
        },
    )
    pk = PersistentKernel(**params)
    assert pk.target_cc >= 100, "gdn_recurrent_sm100 requires Blackwell"
    block_dim = (256, 1, 1)

    qkv_dt = pk.attach_input(qkv, name="gdn_qkv_c")
    ba_dt = pk.attach_input(ba, name="gdn_ba")
    ad_dt = pk.attach_input(alog_dtbias, name="gdn_alog_dtbias")
    state_dt = pk.attach_input(state, name="gdn_recurrent_state")
    z_dt = pk.attach_input(z, name="gdn_z")
    nw_dt = pk.attach_input(norm_w, name="gdn_norm_weight")
    out_dt = pk.attach_input(out, name="gdn_gated_out")
    split_scratch = torch.zeros(num_requests, NUM_V_HEADS, HEAD_V_DIM + 8,
                                dtype=torch.float32, device=device)
    ss_dt = pk.attach_input(split_scratch, name="gdn_split_scratch")

    pk.gdn_recurrent_layer(
        qkv=qkv_dt,
        ba=ba_dt,
        alog_dtbias=ad_dt,
        state=state_dt,
        z=z_dt,
        norm_w=nw_dt,
        output=out_dt,
        split_scratch=ss_dt,
        num_k_heads=NUM_K_HEADS,
        grid_dim=(NUM_V_HEADS, num_requests, GDN_SPLIT),
        block_dim=block_dim,
        depth=GDN_DEPTH,
    )

    pk.compile(output_dir="./test_output_gdn_recurrent")

    # `init_kernel` runs inside compile() and ZEROES `step`
    # (persistent_kernel.cuh:150-153), so the seeded value has to be written
    # back afterwards. New-prefill admission never reads `step`, so this only
    # changes which branch of `gdn_slot_is_first_chunk` each slot takes.
    pk.meta_tensors["step"].copy_(
        torch.tensor(SEEDED_STEPS, dtype=torch.int32, device=device)
    )
    pk()
    torch.cuda.synchronize()

    steps = pk.meta_tensors["step"].tolist()
    want_steps = [s + n for s, n in zip(SEEDED_STEPS, PROMPT_LENS)]
    print(f"  runtime step after finalize = {steps} (expected {want_steps})")
    assert steps == want_steps, (
        "prepare_next_batch did not schedule both prefills in one iteration"
    )

    worst = 0.0
    worst_state = 0.0
    off = 0
    for slot, n in enumerate(PROMPT_LENS):
        zero_state = SEEDED_STEPS[slot] == 0
        ref, ref_state = ref_chain(
            qkv[off:off + n], ba[off:off + n], z[off:off + n],
            A_log, dt_bias, norm_w, state_in[slot], zero_state
        )
        got = out[off:off + n]
        err = (got.float() - ref.float()).abs().max().item()
        serr = (state[slot] - ref_state).abs().max().item()
        worst = max(worst, err)
        worst_state = max(worst_state, serr)
        print(f"  slot {slot}: Q_LEN={n} step={SEEDED_STEPS[slot]} "
              f"zero_state={zero_state} out_max_abs_diff={err:.3e} "
              f"state_max_abs_diff={serr:.3e}")
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(state[slot], ref_state, atol=1e-4, rtol=1e-4)
        off += n

    # The predicate must actually be live: recompute slot 0 WITH its stale state
    # and confirm the kernel did not produce that.
    stale, _ = ref_chain(qkv[:PROMPT_LENS[0]], ba[:PROMPT_LENS[0]],
                         z[:PROMPT_LENS[0]], A_log, dt_bias, norm_w,
                         state_in[0], False)
    assert not torch.allclose(out[:PROMPT_LENS[0]], stale, atol=1e-3), (
        "slot 0 consumed its stale state - the step==0 predicate is dead"
    )

    pk.finalize()
    print(f"GDN_RECURRENT TEST-MODE PIPELINE PASSED (worst out diff "
          f"{worst:.3e} / budget 1e-2, worst state diff {worst_state:.3e})")


if __name__ == "__main__":
    main()
