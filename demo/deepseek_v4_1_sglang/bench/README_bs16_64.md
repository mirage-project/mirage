# DeepSeek-V4.1-Flash at bs 16 to 64: the static layers against SGLang

This branch adds to `dsv41-bs8-1119` the changes for decode batches of 16 to 64 rows. Every switch is off by default. The
`SET` below turns on the measured set.

Each run starts one `sgl.Engine` for one batch size with prompts of 256 tokens (`dsv41_spec_bench.py`: PR #819's
`sgl_spec_bench.py` for DSv4.1). Under nsys with the CUDA graph trace it traces the first 64 forward steps of one generate
call of 256 new tokens. `period.py` takes the decode step as the period of the forward graph, from the start of one replay
to the start of the next.

## Setup

- 4 x B300: the static MoE keeps its own copy of the experts.
- SGLang at commit `240f2a0a64` in its own venv. mirage built from this branch. `nvcc` and `nsys` on `PATH`.
- If mirage was built in another venv than SGLang's: that venv's `site-packages/z3/lib` on `LD_LIBRARY_PATH` (see
  README_bs8.md).
- `M`: the official 40-layer DeepSeek-V4.1-Flash checkout.

```bash
export MIRAGE=/path/to/this/checkout M=/path/to/DeepSeek-V4.1-Flash
D=$MIRAGE/demo/deepseek_v4_1_sglang
NS="nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx \
    --cuda-graph-trace=graph --trace-fork-before-exec=true"
STATIC="PYTHONPATH=$D/boot DSV41_STATIC_ROOT=$MIRAGE DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_MAX_POS=4607 \
    SGLANG_DSV4_KV_LAYOUT=v41_new"
# the bs 8 set of README_bs8.md, with -DMOE_CMP=16 in the flags; from DSV41_WOB2=1 on, the switches of this branch
SET="DSV41_STATIC_PDL=1 DSV41_STATIC_MHC=1 SA_DRY=0 HC_DRY_MOE=1 HC_DRY_ATTN=0 \
    DSV41_STATIC_FLAGS='-DHCP_LOOP -DHCP_SSQ=2 -DHCP_VPOLL -DSTATIC_NO_START_BARRIER -DSTATIC_RESET_PARITY -DI1_BF16 -DI1B_FQ -DHCP_FAST_END -DSTATIC_CNT_POLL_NS=0 -DSTATIC_POLL_NS=0 -DMOE_CMP=16' \
    DSV41_STATIC_IS_TCGEN05=1 DSV41_SHARE_MODULES=1 \
    DSV41_ENGRAM=2 DSV41_HEAD=2 DSV41_EG_KV14=1 DSV41_HEAD_ARGMAX=1 DSV41_EG_ROWS=1 DSV41_FILL_FUSED=1 DSV41_STATIC_ROWS=1,2,4 \
    DSV41_STATIC_T=16,32,64 DSV41_WOB2=1 \
    DSV41_MOE_MIX_SPREAD=1 DSV41_HC_FRONT=1 DSV41_SA_SKIP=1 DSV41_SA_ORDER=z DSV41_IDX_SKIP=1 \
    DSV41_SA_NOSPLIT_MIN_T=64 DSV41_HC_FRONT_WRAP=1 DSV41_TOPK_WITH_SA=1"
```

## The comparison

For each batch size, run the static layers and SGLang one after the other on the same machine. SGLang's own period differs
by about 30 us from machine to machine, so compare only runs from the same machine.

```bash
for bs in 16 32 64; do
  ARGS="--bs $bs --spec off --prompt-tokens 256 --max-total-tokens 81920 --reps 1 --profile --profile-steps 64 --profile-tokens 256"
  eval env $STATIC $SET DSV41_MODEL=$M DSV41_TP=4 DSV41_PRINT_ALL=1 $NS -o static_bs$bs python $D/bench/dsv41_spec_bench.py $ARGS > static_bs$bs.log 2>&1
  DSV41_MODEL=$M DSV41_TP=4 DSV41_PRINT_ALL=1 $NS -o sglang_bs$bs python $D/bench/dsv41_spec_bench.py $ARGS > sglang_bs$bs.log 2>&1
done
for bs in 16 32 64; do for n in static_bs$bs sglang_bs$bs; do
  nsys export --type sqlite --force-overwrite true -o $n.sqlite $n.nsys-rep > /dev/null
  echo "$n: $(python $D/bench/period.py $n.sqlite | grep '^forward graph')"
done; done
```

The speedup is SGLang's `period median` divided by the static layers' `period median`, minus 1.

- The first static run builds the kernels of every T, which takes some minutes. Later runs reuse them from
  `build/sgl_static_dsv41/cache`.
- `static_bs*.log` must contain `first static decode step (L0): in a CUDA graph capture`. Without that line, the static
  layers did not run.
- The runs are not forced to the same tokens. The two sides make different tokens: Engram and the head add in other fp32
  orders. One side can also make other tokens from one run to the next, with or without the switches of this branch. The
  step time depends a little on the text.

## Results

11-24, Modal, 4 x B300, 40 layers, prompt 256, 256 new tokens. Each line is one job on one machine: the static layers
first, then SGLang.

| bs | SGLang (us) | static layers (us) | speedup |
|---|---|---|---|
| 16 | 5402.5 | 4277.2 | +26.3 % |
| 32 | 6872.1 | 6032.7 | +13.9 % |
| 64 | 9074.2 | 8749.7 | +3.7 % |

bs 16 and 32 ran without the last two switches (`DSV41_HC_FRONT_WRAP`, `DSV41_TOPK_WITH_SA`). Those two change only the
layers of 64 rows. Other bs 64 runs on 11-24, each in its own job:

| run | SGLang (us) | static layers (us) | speedup |
|---|---|---|---|
| without the last two switches | 9085.2 | 9136.0 | -0.6 % |
| with `DSV41_HC_FRONT_WRAP` | 9058.2 | 8824.9 | +2.6 % |
| with both, `DSV41_TOPK_WITH_SA` on every index layer (before it was limited to the ratio-2 layers) | 9066.5 | 8973.3 | +1.0 % |

The switches of this branch change only steps of 16 or more rows. For bs 1 to 8 the static layers run the same kernels as
on `dsv41-bs8-1119`. Measured on 11-23 with the same commands, without the switches from `DSV41_MOE_MIX_SPREAD` on: bs 1
-2.6 %, bs 2 -3.0 %, bs 4 +12.6 %, bs 8 +20.7 %.

## The switches of this branch

| Switch | What it changes |
|---|---|
| `-DMOE_CMP=16` (in `DSV41_STATIC_FLAGS`) | At T >= 32 the routed experts compute only each expert's own tokens (16-token virtual slots), not all T of them. |
| `DSV41_WOB2=1` | T = 64's wo_b runs in 2 K parts: 80 tasks, one round of the 148 SMs, instead of 160 tasks in 2 rounds. |
| `DSV41_MOE_MIX_SPREAD=1` | T = 16 / 32 MoE: the 32 mHC mix tasks go one per SM, behind x_q's and the router's tasks. The hand plan had put all of them on one SM, where they ran one after another. In situ the MoE kernel went from 102 to 69 us at T = 16 and from 134 to 106 us at T = 32. |
| `DSV41_HC_FRONT=1` | T = 16 attention: the mHC tasks run first on sparse_attn's SMs, which wait about 20 us for q anyway. Before, they ran behind sparse_attn's tasks and delayed wo_b. |
| `DSV41_SA_SKIP=1`, `DSV41_SA_ORDER=z` | T = 32: an empty compressed pair's sparse_attn task returns at entry. The round robin's sparse_attn tasks are dealt pair-major, so the 12 tasks past the first round of the SMs are the last pairs, which are empty at short contexts. |
| `DSV41_IDX_SKIP=1` | A token with at most 512 compressed positions takes all of them: the top-k writes the list at once and the score tasks do nothing. Not layer 20, whose scores give the candidate blocks. |
| `DSV41_SA_NOSPLIT_MIN_T=64` | T = 64: sparse_attn is not split. One task per token gives 64 tasks in one round, instead of 320 in 3. Its loop runs over the live pairs only. |
| `DSV41_HC_FRONT_WRAP=1` | T = 64 attention, in the layers with compressed KV: the 96 mHC tasks are dealt over sparse_attn's 64 SMs in turn, ahead of sparse_attn. |
| `DSV41_TOPK_WITH_SA=1` | T = 64, the ratio-2 full layers (2, 8, 14): each token's top-k task runs right ahead of its sparse_attn task. The layer then gets the hand plan instead of round robin, which had put the top-k behind the indexer's q and k tasks. |

## Checks

- **Golden check of the attention** (the csa2 demo with the official cases, long contexts included; layers 2, 20, 21 and
  24):
  - At T = 16 and 32 the counts are the same as without the switches.
  - At T = 64 the unsplit sparse_attn is slightly closer to the official result, with the same largest relRMS. In layer 2,
    85312 of 86016 o_q heads are equal, against 85264. The two skips do not change these counts.
- **The plan switches** only move tasks between SMs. compile_plan checks every plan for a deadlock.
- **With every switch of this branch off**, the plans are those of `dsv41-bs8-1119`.
- **Two fixes are always on:**
  - Engram's kv nodes and the LM head at T > 8 are built with that T's MoE demo (c79f812f).
  - A layer that needs more than 32 tensor maps gets them (22a2dfa3; the T = 32 / 64 MoE kernels with the head).
