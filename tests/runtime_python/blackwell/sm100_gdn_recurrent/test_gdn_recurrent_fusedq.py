#!/usr/bin/env python3
"""M4-I9 flag C — the GDN recurrence's fused fp8 quantize is BYTE-IDENTICAL.

WHAT IS BEING CLAIMED. `MPK_FUSE_RECUR_QUANT=1` makes the recurrence's
gated-RMSNorm epilogue emit the fp8 activation and its fp32 block scales
directly, replacing the downstream `quantize_fp8` task. That is legal because
`head_v_dim == 128` is exactly one fp8 scale group and this task owns the whole
of it — in the split configuration only the last-arriving task reaches the
epilogue, elected by the existing `__threadfence()` + `atomicAdd` counter.

Unlike the other two fusions this one does NOT call the shared quantize
function: the values live one-per-thread across four warps, so the amax needs a
block reduction rather than the warp shuffle the standalone impl uses. The
arithmetic is therefore hand-written and this test is what makes the
bit-exactness claim falsifiable. Three checks per case:

  1. the bf16 `out` from the fused kernel (WRITE_OUT=true) is byte-identical to
     the UNFUSED kernel's — the fusion does not perturb the recurrence;
  2. the updated `state` is byte-identical — nor its state write-back;
  3. the fp8 bytes and fp32 scales equal the STANDALONE quantize run over that
     same bf16 `out`, in the same TU with the same nvcc flags — the fused amax,
     scale and clamp are the shipped ones. `fmaxf` is exact and
     order-independent, which is why a different reduction SHAPE is allowed to
     be a different reduction shape;
  4. and the shipped form (WRITE_OUT=false, no bf16 store) produces the same fp8
     as the WRITE_OUT=true form — dropping the store cannot move a byte.

WHY BOTH nvcc LANES. `-use_fast_math` rewrites `rsqrtf`, `expf` and the
divisions in both the epilogue and the quantize, and the megakernel ships it
(`GDN_TEST_FAST_MATH=1` builds that arm).

Build + run (both lanes):
    cd tests/runtime_python/blackwell/sm100_gdn_recurrent
    python setup.py build_ext --inplace && python test_gdn_recurrent_fusedq.py
    GDN_TEST_FAST_MATH=1 python setup.py build_ext --inplace && \
        GDN_TEST_FAST_MATH=1 python test_gdn_recurrent_fusedq.py
"""
import os
import sys

import torch

import runtime_kernel_blackwell_gdn_recurrent as K

# The Qwen3.5 production shape: 32 v-heads / 16 k-heads, head dims 128,
# qkv_stride 8192, ba_stride 64, z_stride = out_stride = 4096.
HV, HK, DK, DV = 32, 16, 128, 128
QKV_S, BA_S, Z_S, OUT_S = 8192, 64, 4096, 4096

# (slots, split, depth). split 4 is the shipped decode value at mbr=1, split 1 at
# mbr>=4; split 2 covers mbr=2. depth 2 is the shipped ring.
CONFIGS = [(1, 1, 2), (1, 2, 2), (1, 4, 2), (2, 2, 2), (4, 1, 2)]
SCALES = [1.0, 1e-3, 30.0, 1e-8]


def mk(slots, seed, scale):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    f = lambda *sh: (torch.randn(sh, generator=g, device=dev,
                                 dtype=torch.float32) * scale)
    qkv = f(slots, QKV_S).to(torch.bfloat16)
    ba = f(slots, BA_S).to(torch.bfloat16)
    ad = torch.stack([f(HV).abs() * 0.1 - 2.0, f(HV) * 0.1]).contiguous()
    z = f(slots, Z_S).to(torch.bfloat16)
    nw = (f(DV) * 0.2 + 1.0).contiguous()
    st = f(slots, HV, DV, DK).contiguous()
    qo = torch.arange(slots + 1, dtype=torch.int32, device=dev)
    return qkv, ba, ad, z, nw, st, qo


def one(slots, split, depth, scale, seed):
    dev = "cuda"
    qkv, ba, ad, z, nw, st0, qo = mk(slots, seed, scale)
    scr = lambda: torch.zeros((slots, HV, DV + 8), dtype=torch.float32,
                              device=dev)

    # ---- reference: the UNFUSED split kernel, then the standalone quantize ---
    ref_out = torch.zeros((slots, OUT_S), dtype=torch.bfloat16, device=dev)
    ref_st = st0.clone()
    K.gdn_recurrent_decode_split_sm100(qkv, ba, ad, ref_st, z, nw, ref_out,
                                       scr(), qo, HK, split, depth)
    ref_q = torch.zeros((slots, OUT_S), dtype=torch.float8_e4m3fn, device=dev)
    ref_s = torch.zeros((slots, OUT_S // 128), dtype=torch.float32, device=dev)
    K.gdn_recurrent_decode_split_fusedq_sm100(
        qkv, ba, ad, st0.clone(), z, nw, ref_out, ref_q, ref_s, scr(), qo,
        HK, split, depth, 2)                       # mode 2 = reference quantize

    # ---- arm G, WRITE_OUT=true: bf16 + fp8 both emitted -------------------
    got_out = torch.zeros((slots, OUT_S), dtype=torch.bfloat16, device=dev)
    got_st = st0.clone()
    got_q = torch.zeros((slots, OUT_S), dtype=torch.float8_e4m3fn, device=dev)
    got_s = torch.zeros((slots, OUT_S // 128), dtype=torch.float32, device=dev)
    K.gdn_recurrent_decode_split_fusedq_sm100(
        qkv, ba, ad, got_st, z, nw, got_out, got_q, got_s, scr(), qo,
        HK, split, depth, 0)

    # ---- arm G as SHIPPED, WRITE_OUT=false: fp8 only ---------------------
    shp_q = torch.zeros((slots, OUT_S), dtype=torch.float8_e4m3fn, device=dev)
    shp_s = torch.zeros((slots, OUT_S // 128), dtype=torch.float32, device=dev)
    K.gdn_recurrent_decode_split_fusedq_sm100(
        qkv, ba, ad, st0.clone(), z, nw, got_out, shp_q, shp_s, scr(), qo,
        HK, split, depth, 1)

    return dict(
        out=int((ref_out.view(torch.int16) != got_out.view(torch.int16)).sum()),
        state=int((ref_st.view(torch.int32) != got_st.view(torch.int32)).sum()),
        q=int((ref_q.view(torch.uint8) != got_q.view(torch.uint8)).sum()),
        s=int((ref_s.view(torch.int32) != got_s.view(torch.int32)).sum()),
        shipped_q=int((got_q.view(torch.uint8) != shp_q.view(torch.uint8)).sum()),
        shipped_s=int((got_s.view(torch.int32) != shp_s.view(torch.int32)).sum()),
    )


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    fast = os.environ.get("GDN_TEST_FAST_MATH") == "1"
    print(f"lane: {'fastmath' if fast else 'nofastmath'}  "
          f"(GDN_TEST_FAST_MATH={os.environ.get('GDN_TEST_FAST_MATH', '0')})")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'slots':>6s}{'split':>6s}{'depth':>6s}{'scale':>9s}"
          f"{'bf16 out':>10s}{'state':>8s}{'fp8':>6s}{'scale':>7s}"
          f"{'ship fp8':>10s}{'ship scl':>10s}  verdict")
    total = 0
    ncase = 0
    seed = 20260732
    for slots, split, depth in CONFIGS:
        for sc in SCALES:
            seed += 1
            ncase += 1
            r = one(slots, split, depth, sc, seed)
            bad = sum(r.values())
            total += bad
            print(f"{slots:6d}{split:6d}{depth:6d}{sc:9.0e}"
                  f"{r['out']:10d}{r['state']:8d}{r['q']:6d}{r['s']:7d}"
                  f"{r['shipped_q']:10d}{r['shipped_s']:10d}  "
                  f"{'IDENTICAL' if bad == 0 else 'DIFFERS'}")
    print()
    if total == 0:
        print(f"PASS: fused == recurrence + standalone quantize, byte-identical "
              f"on all {ncase} cases (bf16 out, fp32 state, fp8 bytes, fp32 "
              f"scales), and the shipped no-store form matches")
        return 0
    print(f"FAIL: {total} differing elements over {ncase} cases -- the "
          f"bit-exactness-by-construction claim is refuted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
