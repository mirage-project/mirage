# M3-I3 correctness gate logs — GDN recurrent decode fast path

Raw run output for the two gates the perf evidence rests on: the **unit
integer-memcmp** gate (split path vs the frozen golden task impl) and the
**oracle** gate (both paths vs the real Qwen3.5 checkpoint's HF dumps), plus
the **test-mode** pipeline gate. AC-3 sweep and bisect logs live one level up.

Everything here was produced from `~/mpk-qwen35/mirage-rm` at upstream HEAD
`68f93b3a` with the I3 change applied, `gdn_recurrent_sm100.cuh`
md5 `355fee57457418dc85dadf82f0d1ab41` — byte-identical to the committed
version in `b0920b28`.

## Why there are two flag lanes

The **shipped** megakernel compiles with `-use_fast_math`
(`persistent_kernel.py`'s JIT default; the M2 ruling is that fast-math is both
more exact and faster for this model). The standalone unit harness
historically does not. Fast-math rewrites `expf`/`log1pf`/`rsqrtf` and the
reciprocal, so a bit-exactness claim has to hold in **both** lanes — the claim
is that those rewrites happen *identically* on the golden and split paths.
`setup.py` gained a `GDN_TEST_FAST_MATH=1` knob for this.

Flag lane is provable from the build logs (count of `use_fast_math` in the
emitted nvcc line):

| build log | `grep -c use_fast_math` | lane |
|---|---|---|
| `build_nofastmath.log` | 0 | no fast-math |
| `build_fastmath.log` | 1 | **shipped lane** |
| `build_baseline_fm.log` | 1 | shipped lane, HEAD control |

## Gate index

| gate | log | lane | pass line to grep |
|---|---|---|---|
| Unit integer-memcmp | `unit_nofastmath.log` | no fast-math | `ALL GDN_RECURRENT UNIT TESTS PASSED` |
| Unit integer-memcmp | `unit_fastmath.log` | **shipped** | `ALL GDN_RECURRENT UNIT TESTS PASSED` |
| Oracle vs HF dumps | `oracle_nofastmath.log` | no fast-math | `ALL BIT-EXACTNESS TARGETS MET` |
| Oracle vs HF dumps | `oracle_fastmath.log` | **shipped** | `^split=` arms — all `BIT-EXACT` (see caveat) |
| Oracle HEAD control | `oracle_baseline_fm.log` | shipped | `FAILED (6):` — identical list to the arm above |
| Test-mode pipeline | `testmode.log` | shipped (megakernel JIT) | `GDN_RECURRENT TEST-MODE PIPELINE PASSED` |

### Unit gate — what it actually compares

`test_gdn_recurrent.py` sections `[8]` and `[8b]`. Integer `memcmp` of BOTH
`out` (bf16, viewed int16) AND the updated recurrent `state` (fp32, viewed
int32) between the decode split path and the golden `..._task_impl`, on
identical inputs. This is the same gate the ferret loop that produced the
kernel ran on every iteration.

Coverage per lane: 7 shapes x splits {1,2,4}, plus the Qwen3.5 production
shape (32,16,128,128,8192,64,4096,4096) x splits {1,2,4,8,16,32} x depths
{2,3,4}; each at 1 and 3 request slots. Plus arrival-counter self-reset, and
`[8b]` 4 back-to-back launches reusing ONE scratch buffer.

    grep -c '    PASS' unit_fastmath.log   # 151
    grep -c '    FAIL' unit_fastmath.log   # 0

Both lanes: 151 PASS / 0 FAIL.

### Oracle gate — and the 6 pre-existing fast-math FAILs

`test_gdn_recurrent_oracle.py`. Section `[2a]` is the I3 addition: it runs the
decode fast path at splits {1,2,4,8,16,32} x depths {2,4} against the real
checkpoint dumps and checks four things per arm —

    grep '^split=' oracle_fastmath.log      # 31 lines, all BIT-EXACT

`o` vs `gdn.core_attn_out`, `y` vs `gdn.gated_norm_out`, and `y`/`state`
byte-identical to the golden path. (`split=1` has no `o` line by design: with
one task the readout never leaves shared memory, so the scratch buffer is
deliberately untouched.)

**`oracle_fastmath.log` ends in `FAILED (6)` and that is NOT a regression.**
`oracle_baseline_fm.log` is the control: the same oracle, same fast-math lane,
with `gdn_recurrent_sm100.cuh` reverted to HEAD (no I3 change at all,
md5 `ccd77f0cd4eded36fb81318bb31cd553`). It produces the **identical** six
failures:

    diff <(grep '^FAILED' oracle_fastmath.log) \
         <(grep '^FAILED' oracle_baseline_fm.log)   # no output

All six are on code paths I3 never touched — `gdn_gating_probe` (a separate
probe kernel) and the GOLDEN impl against the harness's torch EXACT-ORDER
references, which cannot reproduce fast-math intrinsics. In the no-fast-math
lane the same file reports `ALL BIT-EXACTNESS TARGETS MET`.

### Test-mode gate

Full MPK pipeline: python layer API -> task registration -> C++ codegen ->
nvcc -> runtime dispatch, run with the split ENABLED (`GDN_SPLIT=4`). Both
requests are prefill chunks, so this specifically covers the codegen branch
where prefill stays unsplit — split 0 runs the whole chunk, the other splits
are no-ops — and exercises both sides of the `step == 0` predicate.

    out_max_abs_diff = 0.000e+00 on both slots; TESTMODE_EXIT=0

## Reproduce

    bash i3_oracle_lanes.sh <gpu>   # builds + runs unit & oracle in both lanes
    bash i3_testmode_tee.sh         # test-mode, 3-sample GPU guard
