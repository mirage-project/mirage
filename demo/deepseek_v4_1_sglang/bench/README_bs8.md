# DeepSeek-V4.1-Flash at bs 8: the static layers against SGLang

This is a temporary branch for one measurement. It holds the static attention and MoE layers with mHC and the all-reduce
inside the kernels, and the optimizations of 11-17 to 11-19. Every switch is off by default. `compare_bs8.py` turns on
the set that is being measured.

Only decode steps of exactly 8 rows run the static layers (T = 8). SGLang runs the other batch sizes and the prefill
with its own layers. A run at another batch size therefore compares SGLang with SGLang.

## Run

You need a machine with 4 x B300 and SGLang at commit `240f2a0a64` in its venv. Build mirage from this branch with the
same Python minor version, and put `nvcc` and `nsys` on `PATH`. Then, in SGLang's venv:

```bash
python demo/deepseek_v4_1_sglang/bench/compare_bs8.py /path/to/DeepSeek-V4.1-Flash
```

It runs SGLang and the static layers in turn for 2 rounds (`--rounds`) on the GPUs of `CUDA_VISIBLE_DEVICES` (default
0-3). It prints each run's step time and each round's speedup. The last line is the result:

```
median over 2 rounds: SGLang .... us, static .... us a step; speedup = SGLang / static - 1 = +x.xx %
```

The runs (`.nsys-rep`, `.sqlite`, `.log`) stay in `bs8_runs/` (`--out`).

- The static MoE keeps its own copy of the experts, so each GPU needs about 213 GiB with SGLang's weights. The script
  stops before a run when a GPU has less than 250000 MiB or holds another process.
- The model folder must be the official checkout, because the static layers read its `inference/config.json`. For a
  copy without `inference/`, set `DSV41_STATIC_CKPT` to a folder that holds that file and links to the shards.
- If mirage was built in another venv than SGLang's, set `MIRAGE_Z3_LIB` to mirage's libz3 directory, as for #819's
  `sgl_chain.py`.
- The first static run builds the kernels, which takes some minutes. Later runs reuse `build/sgl_static_dsv41/cache`.
- The script stops when a static run's log does not contain `first static decode step (L0): in a CUDA graph capture`,
  because then the static layers did not run.

## The method

The method is PR #819's. `dsv41_spec_bench.py` is #819's `sgl_spec_bench.py` (`demo/glm53_flash/sglang_dsa/` in upstream
mpk) with DSv4.1's model path and engine arguments; it also prints a hash of every row's output. Each run starts one
`sgl.Engine` and traces, under nsys with the CUDA graph trace, the first 64 forward steps of one generate call of 256 new
tokens. `period.py` takes the decode step as the period of the forward graph, from the start of one replay to the start
of the next. The speedup is SGLang's period divided by the static layers' period, minus 1.

- SGLang alone keeps its default KV layout. The static layers need `SGLANG_DSV4_KV_LAYOUT=v41_new`, and in SGLang alone
  that layout is slower.
- SGLang alone varies by 1-2 % from machine to machine, so compare only runs from one machine. The script runs both
  sides on the same GPUs.
- For the noise of one side, `python demo/deepseek_v4_1_sglang/bench/pair_ab.py bs8_runs static` compares `static_r1`
  with `static_r2` replay by replay. It pairs only runs that made the same tokens. SGLang and the static layers make
  different tokens, so it cannot compare the two sides.

## The switches

The measured set is `STATIC` in `compare_bs8.py`.

| Switch | What it changes | Result so far |
|---|---|---|
| `DSV41_STATIC_MHC=1` and the HCP flags | mHC and the all-reduce inside the kernels | 11-18 whole model: +3.6 % against SGLang |
| `DSV41_S2_ARGMAX=1` | the greedy argmax as a 2-kernel Triton reduction | -23 us a step (whole model) |
| `-DI1_BF16`, `-DI1B_FQ` | fewer register spills on the reuse layers | -0.9 us each a reuse layer (one layer, Modal B200) |
| `DSV41_I1C=2` | fewer spills on the index layers | -2.0 us a full_r2 layer (one layer, Modal B200) |
| `-DMOE_ROWS_POLL` | sum_store polls the routed rows instead of a count | MoE y -0.5 us, span ~0 (one layer, Modal B200) |
| `DSV41_X_FLAG=1 ..._ATTN_BIND=0` | H1 stage A: the MoE's front waits for the attention's x flag, not the grid | -2.2 us a MoE kernel (local TP2) |
| `DSV41_S2_DEFER=1` | greedy decode from the vocab shards, without the logits all-gather | correct, not timed |

None of them changes the outputs: the golden checks match line for line, and the greedy tokens match. The last five rows
have not been measured on the whole model yet.
