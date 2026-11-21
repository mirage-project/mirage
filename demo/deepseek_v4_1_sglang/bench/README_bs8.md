# DeepSeek-V4.1-Flash at bs 8: the static layers against SGLang

This is a temporary branch for one measurement. It holds the static attention and MoE layers with mHC and the all-reduce
inside the kernels, and the optimizations of 11-17 to 11-20. Every switch is off by default. The command below turns on
the set that is being measured.

Decode steps of 8 rows run the static layers (T = 8), and with `DSV41_STATIC_ROWS=1,2,4` (in `SET` below, the same for
every batch size) so do steps of 1 to 7 rows, the way GLM-5.3 does it (#821). SGLang keeps its own CUDA graphs of those
batch sizes, unchanged. The static layers read the step's rows and treat their other rows as SGLang treats a padding row
of a graph (position 0, request slot 0, KV slot 0). The MoE layers have builds of 1, 2 and 4 live rows, whose idle rows
read no expert. A step of 8 rows runs exactly the 8-row build it ran without the switch. SGLang runs the batch sizes above
8 and the prefill with its own layers, so a run at bs > 8 still compares SGLang with SGLang.

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
# the measured set: mHC + all-reduce in the kernels (MHC, PDL, flags), the tcgen05 index scores, I1, I1b, the post's fast
# end and the polls without sleeps, one module for each kind of layer; since 11-20 night also Engram, the LM head and the
# greedy argmax in the kernels (the last line)
SET="DSV41_STATIC_PDL=1 DSV41_STATIC_MHC=1 SA_DRY=0 HC_DRY_MOE=1 HC_DRY_ATTN=0 \
    DSV41_STATIC_FLAGS='-DHCP_LOOP -DHCP_SSQ=2 -DHCP_VPOLL -DSTATIC_NO_START_BARRIER -DSTATIC_RESET_PARITY -DI1_BF16 -DI1B_FQ -DHCP_FAST_END -DSTATIC_CNT_POLL_NS=0 -DSTATIC_POLL_NS=0' \
    DSV41_STATIC_IS_TCGEN05=1 DSV41_SHARE_MODULES=1 \
    DSV41_ENGRAM=2 DSV41_HEAD=2 DSV41_EG_KV14=1 DSV41_HEAD_ARGMAX=1 DSV41_EG_ROWS=1 DSV41_FILL_FUSED=1 DSV41_STATIC_ROWS=1,2,4"
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
| `-DHCP_FAST_END` | the mHC post re-arms its slots right after reading them and adds its count without a release |
| `-DSTATIC_CNT_POLL_NS=0 -DSTATIC_POLL_NS=0` | the kernels poll their counters and data without sleeping |
| `DSV41_SHARE_MODULES=1` | the layers of one kind (same generated code and flags) share one kernel module, so its code stays in L2 between them instead of coming from HBM for every layer |
| `DSV41_ENGRAM=2` | the Engram of layers 1 and 14 in our kernels: each GPU's 6400 columns of wkv and their multicast gather in the attention kernel before the Engram layer, the gate in the MoE kernel before it (SGLang replicates wkv and runs it between the kernels) |
| `DSV41_EG_KV14=1` | layer 14's wkv in attention L1's kernel instead of L13's (a reuse kernel has no idle SMs for it) |
| `DSV41_EG_ROWS=1` | the n-gram rows looked up in that attention kernel too (SGLang's own table, read in place as its PEER_READ lookup does), not by SGLang's gather between the kernels |
| `DSV41_HEAD=2` | the end of the step in the last MoE kernel: its post computes the head's input (the streams' collapse and the final norm), then the LM head (bf16 GEMM over this GPU's vocab rows, multicast gather into SGLang's fp32 logits buffer) |
| `DSV41_STATIC_ROWS=1,2,4` | decode steps of 1 to 7 rows on the static layers (above); no change at 8 rows |
| `DSV41_FILL_FUSED=1` | the step's tables (positions, window rows, page tables), which the first static layer copies from SGLang's metadata, in one kernel instead of 6 small kernels and copies before attention L0 |
| `DSV41_HEAD_ARGMAX=1` | the head's gather also computes each token's greedy next token (torch.argmax's rules); for a replayed all-greedy decode batch SGLang takes those ids (its in-graph greedy helper's rules) instead of running its 30 us argmax between the graph replays |

None of them changes the outputs. Up to I1b the golden checks match line for line. For the post's fast end, the polls and
the module sharing the greedy tokens match (SGLang at TP2 locally and the 40-layer runs below), and a shared module's
outputs are bit for bit those of the layer's own module. Engram and the head are checked against SGLang's own on the
40-layer model in check mode (DSV41_STATIC_CHECK=1, DSV41_HEAD_CHECK=1): R'' relRMS <= 6.4e-5, logits relRMS <= 9.5e-5
with the same argmax, the kernel's ids equal to torch.argmax of its logits. The Engram and head arithmetic is not bit for
bit SGLang's (other fp32 orders), so unforced runs may make other tokens; the comparison forces both sides to SGLang's.

The argmax counts as ours: it runs in the last static kernel, on the logits that kernel computes. S2-1 below, which only
makes SGLang's own sampler faster, is still left out.

The numbers below are of bs 8 and were measured without `DSV41_STATIC_ROWS`, which does not change a step of 8 rows (the
same build; the same tokens locally). Our runs of this set on 11-20 night (4 x B300, both sides forced to SGLang's own greedy tokens, each job one machine): job
1120_200601 SGLang 4372.5 us, the static layers 3632.7 us, +20.4 %; without DSV41_FILL_FUSED 3656.7 us (+19.6 %; the
same-token pair -24.2 us). Job 1120_175418 (SGLang 4351.3 us), without DSV41_FILL_FUSED: 3650.1 us, +19.2 %, and
without DSV41_EG_ROWS too 3662.8 us (+18.8 %). SGLang's own period differs by about 30 us from machine to machine.
Without the last line (Engram and the head), on 11-20: SGLang 4382.5 us, the static layers 3752.3 us, +16.8 %. Without `DSV41_SHARE_MODULES` the set gave 4074.5 us (+7.6 %), and without the fast end
and the polls too 4093.8 us (+7.1 %).

The branch also has these switches, all off by default and not in the set. Each cost below was measured on the whole
model, against the set with S2-1 on:
- H1 (`DSV41_X_FLAG`): +63 us a step.
- S2-2 (`DSV41_S2_DEFER`): +42 us a step.
- I1c (`DSV41_I1C`) and C12 (`-DMOE_ROWS_POLL`): no gain.
- S2-1 (`DSV41_S2_ARGMAX`): a faster greedy argmax in SGLang's sampler, roughly 17-23 us a step between graph replays (not
  measured on its own). It is left out because it changes SGLang's sampler, not the static layers.
