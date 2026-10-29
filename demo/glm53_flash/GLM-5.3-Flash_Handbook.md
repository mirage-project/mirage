# GLM-5.3-Flash Handbook

The static megakernel implementation of GLM-5.3-Flash against the SGLang baseline.

This document covers:

- how to replace GLM-5.3-Flash's DSA, KDA and MoE layers with static megakernel layers inside SGLang and compare end to end
  against stock SGLang;
- the prefetch-aware cost model and the cumulative-P solver;
- which steps to redo on a new GPU environment or GPU model.

---

## 1. What is compared

**Method**:

- Both sides are measured end to end with nsys, in two groups by prompt length: 256 and 4000 tokens.
- **base**: stock SGLang.
- **all3ppf**: all three layer kinds replaced by static layers, with a 16 MB L2 prefetch (`STATIC_L2_PREFETCH_MB=16`). The three
  layer kinds are:
  - every DSA layer;
  - every KDA layer;
  - every MoE layer, including the next layer's attention mHC pre at its end (`MOE_STATIC_END=mhc_pre`).
- **Metric**: the interval between consecutive replays of the forward CUDA graph (the period), median.
  - Decode: time per token = period.
  - MTP: time per token = period ÷ accept length.
  - Speedup = base ÷ all3ppf − 1.

**Settings shared by both sides**:

- TP4; bf16 KV cache; `context_length` 8448.
- KV pool capped at 81920 tokens (163840 at bs 16/32), because the static MoE layers hold their own copy of the expert
  weights, about 1.8 GB per layer.

### 1.1 What is replaced

| Hook | What the static layer covers |
|---|---|
| `sglang_dsa/dsa_static_hook.py` | From the attention's input to the FFN's input: indexer, MLA, o_proj, in-kernel all-reduce, attention mHC post, FFN mHC pre |
| `sglang_kda/kda_static_hook.py` | Same, with KDA as the attention (in_proj, kda_decode, o_proj) |
| `sglang_moe/moe_static_hook.py` | Router, experts (the shared expert is in the experts queue too), all-reduce, FFN mHC post; in `mhc_pre` mode also the next layer's attention mHC pre |

Prefill and the MTP draft model are not replaced. Static layers are built per number of rows per step, T ∈ {8, 16, 32}:

- **DSA/KDA**:
  - a decode step whose row count equals T exactly uses the decode layer;
  - an MTP verify step of bs requests uses the verify layer of T = 8·bs (each request's 6 draft tokens take 8 rows);
  - any other row count falls back to SGLang.
- **MoE**: a decode step uses the smallest T not below its row count; an MTP verify step uses the verify layer of its
  bs × 6 live rows.

### 1.2 Scenarios

| Scenario | Configuration | Static layers used |
|---|---|---|
| `spec516_bs1` / `spec516_bs2` / `spec516_bs4` | MTP 5/1/6, bs 1/2/4 | verify layers, T = 8/16/32 |
| `off_bs8` / `off_bs16` / `off_bs32` | plain decode, bs 8/16/32 | decode layers, T = 8/16/32 |

---

## 2. Setup specific to this branch

- **Two venvs**:
  - mirage is built in its own venv, with z3-solver 4.16;
  - the SGLang venv (0.5.22) ships z3 4.15.
  - The static layers run inside SGLang's processes and load mirage's libz3 through `MIRAGE_Z3_LIB` (pointing to the mirage
    venv's `.../site-packages/z3/lib`).
  - Build mirage with the same Python minor version as the SGLang venv: the hooks put `<checkout>/python` straight on
    `sys.path`.
- **KDA boot file**: install it into the SGLang venv's site-packages. The DSA and MoE hooks are loaded by the chain through
  `PYTHONPATH=sglang_dsa/boot` and need no install.
  ```bash
  SITE=$($SGL_PY -c "import site; print(site.getsitepackages()[0])")
  cp $MIRAGE/demo/glm53_flash/sglang_kda/kda_static_boot.py $SITE/ && echo "import kda_static_boot" > $SITE/kda_static_boot.pth
  ```
  The file in site-packages is a copy: copy it again after changing the code.
- **Model path**: hard-coded as `MODEL` in `sglang_dsa/sgl_spec_bench.py`, `sgl_check.py` and `sgl_gsm8k.py`. The static
  layers read their weights from SGLang's `model_path`.

---

## 3. Running the comparison

`sglang_dsa/sgl_chain.py`, started with **the SGLang venv's python**, runs the runs one after another in the current
directory, one process per run, each logging to `<run name>.log`.

- It resets `*_STATIC_*`, `STATIC_L2_*`, `PYTHONPATH`, `LD_LIBRARY_PATH` and `MIRAGE_ROOT` for every run; values set outside
  are ignored. Exception: `DSA_STATIC_ROOT` is read once when the chain starts and names the mirage checkout (default: the
  checkout the chain is in).
- Before each run it waits until no other process is on `CUDA_VISIBLE_DEVICES`; `SGL_WAIT_FREE=0` turns the wait off.

```bash
export MIRAGE=/path/to/mirage SGL_PY=/path/to/sglang-venv/bin/python MIRAGE_Z3_LIB=/path/to/mirage-venv/lib/python3.12/site-packages/z3/lib
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 PATH=/usr/local/cuda/bin:$(dirname $SGL_PY):$PATH
C=$MIRAGE/demo/glm53_flash/sglang_dsa/sgl_chain.py
P() { for s in "$@"; do for p in 256 4000; do echo -n "prof_base_${s}_p$p prof_all3ppf_${s}_p$p "; done; done; }

$SGL_PY $C check_all3p check_all3p_spec_4l                        # correctness (T=8 decode + verify)
$SGL_PY $C check_t16_bs16_attn check_t16_bs16_moe4l check_t16_bs2_attn check_t16_bs2_moe4l \
           check_t32_bs32_attn check_t32_bs32_moe4l check_t32_bs4_attn check_t32_bs4_moe4l
$SGL_PY $C $(P spec516_bs1 spec516_bs2 spec516_bs4 off_bs8 off_bs16 off_bs32)   # nsys end to end
python3 $MIRAGE/demo/glm53_flash/sglang_dsa/nsys_e2e.py . | tee nsys_e2e.md        # the table
```

### 3.1 Run names

| Run name | What it does |
|---|---|
| `check_*` | `sgl_check.py`, eager mode: each step runs SGLang first, then the static layer on the same inputs, and prints the relRMS. `*_attn` checks DSA + KDA, `*_moe4l` the MoE layers 3–6 |
| `prof_{base,all3ppf}_<scenario>_p{256,4000}` | `sgl_spec_bench.py --profile`: one timed round first (the accept length is read from it), then 64 forward steps under nsys (`--cuda-graph-trace=graph`); the `.sqlite` is exported automatically |
| `base_<scenario>_mtt` / `all3ppf_<scenario>` | wall clock, 4000-token prompts, median of 3 rounds: a cross-check of the nsys result |
| `gsm8k_off` / `gsm8k_516` / `all3ppf_gsm8k_off` / `all3ppf_gsm8k_516` | GSM8K, 5-shot, 200 questions; needs `GSM8K_TEST_JSONL` |

### 3.2 Variables for manual runs (without the chain)

| Variable | Meaning |
|---|---|
| `PYTHONPATH=<checkout>/demo/glm53_flash/sglang_dsa/boot` | **Required**: without it the DSA and MoE hooks are not loaded (the KDA hook comes through the `.pth` in the venv) |
| `LD_LIBRARY_PATH=$MIRAGE_Z3_LIB` | Lets mirage's core load z3 4.16 |
| `DSA/KDA/MOE_STATIC_LAYERS` | `all` or layer ids; unset: nothing replaced |
| `DSA_STATIC_CHECK`, `MOE_STATIC_CHECK` | **Both default to 1; set them to 0 for performance runs**: in check mode the static layers are not used during CUDA graph capture |
| `DSA/KDA/MOE_STATIC_TOKENS` | The T values to build, e.g. `8,16,32` |
| `DSA_STATIC_FUSED_PLAN{,16,32}`, `KDA_STATIC_PLAN{,16,32}`, `MOE_STATIC_PLAN{,16,32}` | Plan file per T; unset: round robin |
| `MOE_STATIC_END` | `mhc` or `mhc_pre` (the last layer is always `mhc`) |
| `STATIC_L2_PREFETCH_MB` | L2 prefetch budget (MB) |
| `KDA_STATIC_ROOT` | Required whenever `KDA_STATIC_LAYERS` is set |
| `MIRAGE_STATIC_BUILD_CACHE` | A shared build cache. Default: under `<checkout>/build/` |

### 3.3 Reading the results

- **Whether the static layers took effect**: the `.log` has one line per replaced layer,
  `... replaces SGLang's ... (..., CUDA graph capture; ...)`.
- **The `nsys_e2e.py` table**: one row per scenario and prompt length, with both sides' time per token, the speedup, the
  period median and mean, the accept length and the number of periods.
  - The forward graph is the one with the longest median replay. `--one <db>` lists every graph of a trace.
- **Do not measure end to end with `--cuda-graph-trace=node`**: per-node records break the static kernels' PDL overlap. One
  earlier bs1 MTP measurement:
  - static layers: period 7170.0 µs under the node trace, while the same configuration's bench gives 5548.3 µs per verify
    step;
  - SGLang: period 8087.6 µs under the node trace, bench 9765.7 µs.
  - The bench numbers include scheduling overhead and should be the larger ones; for the static layers the node trace is
    much larger instead, so the node trace slows the static layers down.
- **Typical check values** (earlier runs):

| Layer | Metric | Typical value |
|---|---|---|
| KDA decode | `x_ffn` relRMS | about 1e-3 |
| KDA verify | `x_ffn` relRMS | about 3e-3–6e-3 |
| KDA | conv and ring | bit-equal about 99.8%–99.9% (lowest seen 99.76%) |
| DSA | `x_ffn` relRMS | about 2e-2 (max about 6e-2), the usual level so far; an order of magnitude above KDA, possibly because the indexer's FP8 scores make top-k pick slightly different positions |
| MoE | static-exact vs SGLang-exact | the first should not exceed the second, e.g. 3.8e-2 vs 5.1e-2 |

---

## 4. Plan files

`DSA_PLAN` / `KDA_PLAN` / `MOE_PLAN` in `sgl_chain.py` decide which plan each run uses. They are the best plans measured on
B200 (148 SMs); a missing file means round robin:

| Layer | T=8 | T=16 | T=32 |
|---|---|---|---|
| DSA decode | `sglang_dsa/dsa_fused_plan_{short,long}.json` | `dsa_fused_plan16_{short,long}.json` | round robin |
| KDA decode | round robin | round robin | round robin |
| MoE (`mhc_pre`) | `sglang_moe/moe_plan_mhc_pre.json` | `moe_plan16_mhc_pre.json` | round robin |

Check runs such as `*_moe4l` end with `mhc`; at T=16 they use `moe_plan16_mhc.json`.

- **DSA short/long**: prompts of at most 1000 tokens use short, all others long.
- **Verify layers**: DSA's verify layers always run round robin; KDA's and MoE's verify layers use the T=8 plan at T=8 and
  round robin at T≥16. The chain sets no T=8 plan for KDA, so KDA's verify layer is in fact round robin too.
- **A plan file belongs to one graph**: it is valid only for the graph it was made for (its nodes and grid candidates), its T
  and its SM count.
  - When the number of SM lists differs from the GPU's SM count, compilation fails with
    `the plan has N SM lists, the GPU has M SMs`.
  - A node with one task per SM must also have grid x equal to the SM count.

---

## 5. Cost model and search strategy

Discussions and implementation:

| Number | Content |
|---|---|
| [#801](https://github.com/mirage-project/mirage/issues/801) | The prefetch-aware cost model: tasks already do input-independent work before their inputs are ready (e.g. a GEMM issuing its weight TMA loads), so a fixed-duration model ranks plans wrongly |
| [#805](https://github.com/mirage-project/mirage/issues/805) | Follow-up to #801: each key fitted by a one-breakpoint piecewise linear regression, without the fixed thresholds α/β |
| [#807](https://github.com/mirage-project/mirage/pull/807) | The implementation of that cost model (`cost_records.py`, `plan_gen.py`, `search.Costs`) |
| [#802](https://github.com/mirage-project/mirage/issues/802) | A faster search: no per-task × per-SM variables but a cumulative constraint; SMs assigned afterwards by interval graph colouring |
| [#809](https://github.com/mirage-project/mirage/issues/809) | Placement rules (same_sm, cluster_pairs, same_group, near); cumulative-P first, per-SM-P as the fallback |

### 5.1 Cost records (`python/mirage/mpk/cost_records.py`)

**The prefetch model** (#801):

- `end = max(enter + pre, ready) + post`, equivalently `d = post + max(0, pre − max(w, 0))`.
- The fit (#805): a one-breakpoint piecewise linear regression per key (`fit_prefetch`). `pre` is searched in 0.05 µs steps,
  then in 0.01 µs steps around the best value; `post` is a weighted quantile; the objective is the least pinball loss at
  quantile 0.75. The 0.75 quantile rather than the median, because a node waits for its slowest task.
- When the samples do not lie on both sides of the knee, the knee cannot be placed. `pre` then falls back to a GEMM's up-front
  weight bytes ÷ the fitted bytes per µs (0 for other task types), clipped to the range the samples allow.

**Reading records**: `search.Costs(files, code=..., gpu=...)` uses only the records whose code and GPU name both match. A
demo's `--search` takes the current code_tag and the current device name by default. So after any change to a static
megakernel header the old records are no longer used by default; `--costs-code any`, or a given tag, mixes old records in.

### 5.2 Search (`search.search_plan`)

1. **Options**: each node's grid candidates (`compiler.candidate_grids`, from `changeable_grid_dims`). Candidates without
   records are dropped.
2. **Durations**:
   - Each task's time is the model's `post`; `pre` goes to solver rule 7: the task holds its SM from its start and can do
     `pre` before its inputs arrive.
   - A consumer's time also depends on the grid its producer chose (e.g. how many K parts it adds).
3. **Model**: the tasks, the dependencies the slices give, the GPU sets with the same work, and the experts queue (pool).
4. **Solve**:
   - First `solver_cumulative` (#802): no SM variables, at most as many tasks hold an SM at any time as there are SMs; after
     the solve the SMs are assigned by interval colouring in start order, which is exact.
   - Placement rules 8–11 (#809) are supported. When the cumulative model cannot express them it raises `NeedsPerSM` and
     `solver.py`'s per-SM model is used instead.
5. **Start**:
   - The current plan (round robin or `--plan`) is the starting point. The result is either that plan or a plan the model
     predicts faster.
   - `--time` is the solver's time limit, default 300 s; DSA used 1800 s so far.
6. **Output**: `<out>/plan.json` (`{"grids", "lists"}`) and `<out>/plan_view.json` (the predicted timeline).

GLM-5.3-Flash's three graphs currently use **no placement rules**: none of the three demos passes `placement` to
`search_plan`, and the only node that adds cluster pairs automatically is `sum_rmsnorm` (used by K3), which these graphs do
not have. The model is optimistic for plans unlike its training samples: an earlier DSA short search predicted 56.5 µs and
measured 63.2 µs, no faster than the current plan (62.9 µs). So **a searched plan must be measured in an A/B run, and only a
plan measured faster is kept**.

### 5.3 Commands

The three single-GPU demos (`dsa_fused_demo.py`, `kda_fused_demo.py`, `moe_demo.py`) all take:

- `--record FILE`: the timing build; this run's records appended to FILE;
- `--collect FILE`: every plan of the collection run once with `--record`;
- `--search FILE ...`: a search with these records, then one run of the plan found, with its time printed;
- `--plan FILE`: a run of the given plan.

T comes from the environment variable `STATIC_TOKENS`. The arguments below match the SGLang runs:

```bash
M=/path/to/GLM-5.3-Flash; D=$MIRAGE/demo/glm53_flash
SERVE="--model-dir $M --gpus 1 --tp 4 --reps 22 --nvcc-flag=-DSTATIC_RESET_IN_KERNEL --nvcc-flag=-DSTATIC_PDL_TRIGGER=1"
export STATIC_L2_PREFETCH_MB=16          # as all3ppf
SHORT="256 290 330 370 410 450 490 512"  # the contexts while decoding after a 256-token prompt
LONG="4000 4040 4080 4120 4160 4200 4240 4256"
```

- `--gpus 1 --tp 4` runs TP shard 0 only: each GPU's tasks are those of TP4, but the records miss the few µs the all-reduce
  waits for the other GPUs.
- DSA records carry no context length, so short and long must go into separate files.
- The earlier B200 plans were collected at contexts 20–275 and 3346–3600.

---

## 6. A new GPU

**What adapts on its own**:

- The build arch follows the device capability (`sm_103a` on B300).
- The SM count is read from the device.
- Nodes with one task per SM get a grid that follows the SM count.
- The default SM groups are the two halves (two dies). They matter only with same_group/near rules, which GLM-5.3-Flash does
  not use today.

**What must be redone**:

- **Cost records**: records carry the GPU name; B200 records are not used.
- **Plan files**: the current plans are laid out for 148 SMs and fail to compile on a GPU with another SM count; even with the
  same SM count they are not necessarily the best.
- **Search the best grids and the L2 prefetch budget**: the default sparse_mla split count and kda_decode RP were chosen for
  one wave of 148 SMs (e.g. T=16: 9 × 16 = 144 tasks); the 16 MB L2 prefetch budget may only suit B200.

**Steps** (all on the new GPU; each step once for every T of 8/16/32):

1. **Get SGLang running first**: replace or remove the B200 files that `DSA_PLAN` / `MOE_PLAN` in `sgl_chain.py` point to,
   so every layer runs round robin. Then run the `check_*` runs to confirm correctness.

2. **Collect costs** (one file per graph, T and context; the GPU in the file name so its records do not mix with B200's):
   ```bash
   G=b300; mkdir -p costs runs
   for T in 8 16 32; do
     for c in short long; do L=$([ $c = short ] && echo $SHORT || echo $LONG)
       STATIC_TOKENS=$T python $D/dsa_fused_demo.py $SERVE --seq-lens $L --out runs/col_dsa_t${T}_$c \
           --collect costs/${G}_dsa_t${T}_$c.json --jobs 4 --rounds 3
     done
     STATIC_TOKENS=$T python $D/kda_fused_demo.py $SERVE --out runs/col_kda_t$T --collect costs/${G}_kda_t$T.json
     STATIC_TOKENS=$T python $D/moe_demo.py $SERVE --rank 0 --end mhc_pre --out runs/col_moe_t$T \
         --collect costs/${G}_moe_t${T}_mhc_pre.json
   done
   ```
   - `--collect` writes the plans to `<out>/plans/` and builds and runs each in a process of its own: `--jobs` builds at a
     time, one GPU run at a time.
   - A plan whose run left a `result.json` is skipped, so a rerun after an interruption carries on.
   - On B200 a DSA collection (about 70 plans, 4 builds at a time) took about 5 minutes with the build cache warm. The first
     collection builds every plan and takes longer.

3. **Search** (once per file, with the same `--seq-lens` and T):
   ```bash
   STATIC_TOKENS=8 python $D/dsa_fused_demo.py $SERVE --seq-lens $LONG --out runs/srch_dsa_t8_long \
       --search costs/${G}_dsa_t8_long.json --time 1800
   ```
   - The result goes to `runs/srch_dsa_t8_long/plan.json`: one GPU's lists, which the hooks reuse for every rank.
   - The plan is then run once.

4. **Measure A/B**: on the same GPU with the same arguments, alternate between the default round robin and
   `--plan runs/srch_.../plan.json`, a few rounds each, and compare the `span ... median` in the output. Keep the searched
   plan only if it is faster.

5. **Use the new plans**:
   - Copy the kept plans into `sglang_dsa/` or `sglang_moe/` with the GPU in the name, e.g. `dsa_fused_plan_long_b300.json`.
   - Point `DSA_PLAN` / `KDA_PLAN` / `MOE_PLAN` in `sgl_chain.py` to the new files.
   - `KDA_PLAN` has entries for T=16/32 only, and `_best_plans` also skips KDA at T=8: a KDA T=8 plan needs both changed.

6. **Re-measure the L2 prefetch budget**: set `PF` in the chain to 16/32/64 MB, or remove it (no prefetch), run a set of
   `prof_all3ppf_*` runs for each, and take the shortest period.

7. **Run end to end**: the commands of section 3, then the table with `nsys_e2e.py`.
