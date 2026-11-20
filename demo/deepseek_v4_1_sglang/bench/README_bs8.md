# DeepSeek-V4.1-Flash at bs 8: the static layers against SGLang

This is a temporary branch for one measurement. It holds the static attention and MoE layers with mHC and the all-reduce
inside the kernels, and the optimizations of 11-17 to 11-19. Every switch is off by default. The command below turns on
the set that is being measured.

Only decode steps of exactly 8 rows run the static layers (T = 8). SGLang runs the other batch sizes and the prefill
with its own layers. A run at another batch size therefore compares SGLang with SGLang.

The method is PR #819's. `dsv41_spec_bench.py` is #819's `sgl_spec_bench.py` (`demo/glm53_flash/sglang_dsa/` in upstream
mpk) with DSv4.1's model path and engine arguments; it also prints a hash of every row's output. Each run starts one
`sgl.Engine` and traces, under nsys with the CUDA graph trace, the first 64 forward steps of one generate call of 256 new
tokens. `period.py` takes the decode step as the period of the forward graph, from the start of one replay to the start
of the next.

## Setup

- 4 x B300. The static MoE keeps its own copy of the experts, so each GPU needs about 213 GiB with SGLang's weights.
  A B200 (179 GiB) is not enough.
- SGLang at commit `240f2a0a64` in its own venv. Build mirage from this branch (`git clone --recursive`, then
  `pip install -e .`) with the same Python minor version. Put `nvcc` and `nsys` on `PATH`.
- If mirage was built in another venv than SGLang's, add that venv's `site-packages/z3/lib` to `LD_LIBRARY_PATH` for the
  static runs, as `MIRAGE_Z3_LIB` does in #819's `sgl_chain.py`. In our setup SGLang's venv has z3 4.15 (tilelang's) and
  mirage's has 4.16; without that path `import mirage` fails with `libz3.so.4.16: cannot open shared object file`.
- `M`: the official 40-layer DeepSeek-V4.1-Flash checkout. The static layers read its `inference/config.json` and its
  shards. If your copy has no `inference/` folder, set `DSV41_STATIC_CKPT` to a folder that holds that file and links to
  the shards.

```bash
export MIRAGE=/path/to/this/checkout M=/path/to/DeepSeek-V4.1-Flash
D=$MIRAGE/demo/deepseek_v4_1_sglang
ARGS="--bs 8 --spec off --prompt-tokens 4000 --max-total-tokens 81920 --reps 1 --profile --profile-steps 64 --profile-tokens 256"
NS="nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx \
    --cuda-graph-trace=graph --trace-fork-before-exec=true"
STATIC="PYTHONPATH=$D/boot DSV41_STATIC_ROOT=$MIRAGE DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_MAX_POS=4607 \
    SGLANG_DSV4_KV_LAYOUT=v41_new"
# the measured set: mHC + all-reduce in the kernels (MHC, PDL, flags), the tcgen05 index scores, I1, I1b
SET="DSV41_STATIC_PDL=1 DSV41_STATIC_MHC=1 SA_DRY=0 HC_DRY_MOE=1 HC_DRY_ATTN=0 \
    DSV41_STATIC_FLAGS='-DHCP_LOOP -DHCP_SSQ=2 -DHCP_VPOLL -DSTATIC_NO_START_BARRIER -DSTATIC_RESET_PARITY -DI1_BF16 -DI1B_FQ' \
    DSV41_STATIC_IS_TCGEN05=1"
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
- For the noise of one side, `python $D/bench/pair_ab.py . static` compares `static_r1` with `static_r2` replay by
  replay. It pairs only runs that made the same tokens. SGLang and the static layers make different tokens, so it cannot
  compare the two sides.

## The switches

| Switch | What it changes |
|---|---|
| `DSV41_STATIC_MHC=1` and the HCP flags | mHC and the all-reduce inside the kernels |
| `DSV41_STATIC_IS_TCGEN05=1` | the indexer's scores with tcgen05 |
| `-DI1_BF16`, `-DI1B_FQ` | fewer register spills on the reuse layers |

None of them changes the outputs: the golden checks match line for line, and the greedy tokens match.

The branch also has these switches, all off by default and not in the set. Each cost below was measured on the whole
model, against the set with S2-1 on:
- H1 (`DSV41_X_FLAG`): +63 us a step.
- S2-2 (`DSV41_S2_DEFER`): +42 us a step.
- I1c (`DSV41_I1C`) and C12 (`-DMOE_ROWS_POLL`): no gain.
- S2-1 (`DSV41_S2_ARGMAX`): a faster greedy argmax in SGLang's sampler, roughly 17-23 us a step between graph replays (not
  measured on its own). It is left out because it changes SGLang's sampler, not the static layers.
