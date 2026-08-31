# Qwen3: hand-written vs compiler-generated

One command, one table. The same model built two ways, with the task
boundaries held fixed: once out of hand-written kernels, once with every task
body compiler-generated. The only thing that changes is who wrote the CUDA.

```bash
python demo/qwen3_compare/compare.py --gpu 6
```

```
=========================================================
  Qwen3 build paths, side by side (baseline: handwritten)
=========================================================
  variant          tok/s   ms/tok   vs base  tokens
  handwritten     6276.6    1.255    1.000x  baseline
  graph           1726.9    4.560    0.275x  DIFFER
=========================================================
```

## The two paths

| variant | what it builds |
| --- | --- |
| `handwritten` | `Qwen3Builder`'s imperative `*_layer` calls. Every task is a hand-written `.cuh`, and the task boundaries are the order those calls appear in. |
| `graph` | The whole model from `mirage.mpk.lowering.node`, with every task body compiler-generated. `partition_as_today` puts the task boundaries in exactly the same places the `*_layer` call sequence puts them, so the only difference against `handwritten` is who wrote the CUDA. |

`--list` prints both with the exact environment each one sets.

## What is still hand-written in the graph path

The table would overstate what the compiler produced if this were not said
plainly.

Four nodes are **opaque**: the graph cannot model them, so embedding, the
KV-cache append, attention and argmax are hand-written tasks on both sides.
On top of that, `run_batch_perf.py`'s `MPK_HANDWRITTEN` defaults to `lm_head`
plus every layer's `qkv` -- search schedules those fine, but the task it
lowers them to is far worse than the hand-written one (the `lm_head` lowers to
2374 blocks where `linear_sm100` uses 148), so the default substitutes the
hand-written task. Set `MPK_HANDWRITTEN=` to see what search alone does.

Everything else -- the o projection, gate, up, SwiGLU, down, and both residual
adds -- is compiler-generated. That is what the `graph` row measures.

## Reading the table

**tok/s** is whole-model decode throughput, and it is the only objective worth
ranking on. A task that is faster in isolation can leave the model slower --
measured: a `silu_mul` schedule 1.20x faster per task left Qwen3-0.6B 2.5%
slower end to end, because `silu_mul` is not what dominates.

**tokens** compares the decoded ids against the baseline's, and it is a weaker
signal than it looks. Decoding is greedy argmax, so in principle two builds
that compute the same thing emit identical ids. In practice the MEGAKERNEL
ITSELF is not run-to-run reproducible: `handwritten` was run twice at an
identical config here and diverged from itself at roughly the tenth token
(`2^3 + 3^2 +` one run, `2 + 3 + 4 +` the next). Reduction order across
workers is not fixed, bf16 rounding then flips a near-tie, and the sequences
part for good.

So `DIFFER` is not evidence that a build computes the wrong thing, and `same`
is not evidence that it computes the right one. Read the column as a cheap
smoke test and the sample text as the real signal: coherent English means the
model works, garbage means it does not. The numeric gate is
`tests/runtime_python/test_mode/test_model_graph.py::test_lowered_mlp_matches_torch`
(rel < 0.02 against torch).

## Why each build is a subprocess

A process gets one megakernel and one CUDA context, so both cannot be built in
one interpreter. Each is a fresh `python tests/ci-tests/run_batch_perf.py`
with a different environment.

That is also what keeps the comparison honest: both model sources go through
that script's `run_and_report`, so every number in the table is produced by
identical timing code. `compare.py` contains no model-building logic of its
own, deliberately -- a second copy of that wiring would drift from the first.

Budget a few minutes per build.

## Related

- `tests/ci-tests/run_batch_perf.py` -- the single-variant bench this drives.
- `python/mirage/mpk/lowering/` -- the graph, the partitioner, the schedule
  search, and the lowering the `graph` path uses.
