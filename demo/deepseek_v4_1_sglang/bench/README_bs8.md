# DeepSeek-V4.1-Flash at bs 8: the static layers against SGLang

This is a temporary branch for one measurement. It holds the static attention and MoE layers with mHC and the all-reduce
inside the kernels, and the optimizations of 11-17 to 11-19. Every switch is off by default. The command below turns on
the set that is being measured.

Only decode steps of exactly 8 rows run the static layers (T = 8). SGLang runs the other batch sizes and the prefill
with its own layers. A run at another batch size therefore compares SGLang with SGLang.

The method is PR #819's. `dsv41_spec_bench.py` is a copy of #819's `sgl_spec_bench.py`; only the model and the engine
arguments are changed. Each run starts one `sgl.Engine`. It profiles 64 decode steps of 256 tokens under nsys with the
CUDA graph trace. `period.py` takes the decode step as the period of the forward graph, from the start of one replay
to the start of the next.

## Setup

- 4 x B300. The static MoE keeps its own copy of the experts, so each GPU needs about 213 GiB with SGLang's weights.
  A B200 (179 GiB) is not enough.
- SGLang at commit `240f2a0a64` in its own venv. Build mirage from this branch with the same Python minor version.
  Put `nvcc` and `nsys` on `PATH`.
- `M`: the 40-layer DeepSeek-V4.1-Flash checkpoint (SGLang's model path).
- `CK`: a directory that holds the official repository's `inference/config.json` and links to the shards of `M`.
  `DSV41_STATIC_CKPT` reads the layers' weights from it.

```bash
export MIRAGE=/path/to/this/checkout M=/path/to/DeepSeek-V4.1-Flash CK=/path/to/ck
D=$MIRAGE/demo/deepseek_v4_1_sglang
ARGS="--bs 8 --spec off --prompt-tokens 4000 --max-total-tokens 81920 --reps 1 --profile --profile-steps 64 --profile-tokens 256"
NS="nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx \
    --cuda-graph-trace=graph --trace-fork-before-exec=true"
STATIC="PYTHONPATH=$D/boot DSV41_STATIC_ROOT=$MIRAGE DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_MAX_POS=4607 \
    SGLANG_DSV4_KV_LAYOUT=v41_new DSV41_STATIC_CKPT=$CK"
# the measured set: mHC + all-reduce in the kernels (MHC, PDL, flags), the tcgen05 index scores, S2-1, I1, I1b, I1c,
# C12 (MOE_ROWS_POLL), H1 stage A (X_FLAG with the attention unbound), S2-2 (S2_DEFER)
SET="DSV41_STATIC_PDL=1 DSV41_STATIC_MHC=1 SA_DRY=0 HC_DRY_MOE=1 HC_DRY_ATTN=0 \
    DSV41_STATIC_FLAGS='-DHCP_LOOP -DHCP_SSQ=2 -DHCP_VPOLL -DSTATIC_NO_START_BARRIER -DSTATIC_RESET_PARITY -DI1_BF16 -DI1B_FQ -DMOE_ROWS_POLL' \
    DSV41_STATIC_IS_TCGEN05=1 DSV41_S2_ARGMAX=1 DSV41_X_FLAG=1 DSV41_X_FLAG_ATTN=1 DSV41_X_FLAG_ATTN_BIND=0 DSV41_I1C=2 DSV41_S2_DEFER=1"
```

## The comparison

Run SGLang and the static layers in turn, on one machine, for at least 2 rounds. SGLang alone varies by about 2 % from
machine to machine, so compare only runs from the same machine.

```bash
for r in 1 2; do
  DSV41_MODEL=$M DSV41_TP=4 DSV41_PRINT_ALL=1 $NS -o sglang_r$r python $D/bench/dsv41_spec_bench.py $ARGS > sglang_r$r.log 2>&1
  eval env $STATIC $SET DSV41_MODEL=$M DSV41_TP=4 DSV41_PRINT_ALL=1 $NS -o static_r$r python $D/bench/dsv41_spec_bench.py $ARGS > static_r$r.log 2>&1
done
for n in sglang_r1 static_r1 sglang_r2 static_r2; do
  nsys export --type sqlite --force-overwrite true -o $n.sqlite $n.nsys-rep > /dev/null
  echo "$n: $(python $D/bench/period.py $n.sqlite | grep '^forward graph')"
done
```

Take the median `period median` of each side. The speedup is SGLang's period divided by the static layers' period,
minus 1.

- The SGLang runs use SGLang's default KV layout. The static layers need `SGLANG_DSV4_KV_LAYOUT=v41_new`, and in
  SGLang alone that layout is slower.
- The first static run builds the kernels, which takes some minutes. Later runs reuse them from
  `build/sgl_static_dsv41/cache`.
- `static_r*.log` must contain `first static decode step (L0): in a CUDA graph capture`. Without that line, the static
  layers did not run.
- `python $D/bench/pair_ab.py DIR sglang_r1` pairs the runs replay by replay. It needs the runs as `<arm>_r<k>` with
  their `.sqlite` and `.log` in DIR. It pairs only runs that made the same tokens; SGLang and the static layers make the
  same tokens only when they agree bit for bit, which they usually do not. So pair the static runs with each other, as a
  noise floor.

## The switches

| Switch | What it changes | Single-layer result (Modal B200, cold TLB) |
|---|---|---|
| `DSV41_STATIC_MHC=1` and the HCP flags | mHC and the all-reduce inside the kernels | (11-18 whole model: +3.6 % against SGLang) |
| `DSV41_S2_ARGMAX=1` | the greedy argmax as a 2-kernel Triton reduction | -23 us a step (whole model) |
| `-DI1_BF16`, `-DI1B_FQ` | fewer register spills on the reuse layers | -0.9 us each a reuse layer |
| `DSV41_I1C=2` | fewer spills on the index layers | -2.0 us a full_r2 layer |
| `-DMOE_ROWS_POLL` | sum_store polls the routed rows instead of a count | MoE y -0.5 us, span ~0 |
| `DSV41_X_FLAG=1 ..._ATTN_BIND=0` | H1 stage A: the MoE's front waits for the attention's x flag, not the grid | -2.2 us a MoE kernel (local TP2) |
| `DSV41_S2_DEFER=1` | greedy decode from the vocab shards, without the logits all-gather | correct, not timed |

All of them leave the outputs the same: the golden checks match line for line, and the greedy tokens match.
