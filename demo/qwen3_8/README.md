# Qwen3.8-27B BF16 text inference (experimental)

This adapter targets the dense
[Qwen/Qwen3.8-27B checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0).
The checkpoint declares the `Qwen3_5ForConditionalGeneration` architecture,
so the internal Python package and CUDA task names still use `qwen3_5`.
The exposed builder alias and full-model geometry are 27B only.

`run.py` now builds an MPK `offline` graph and invokes it once. It passes prompt
lengths, tokens, steps, input/output tokens and KV page-table buffers to MPK.
The scheduler consumes one token per request and iteration and the attention
cache comes from `KVCachePlan`. The graph uses one 1024-token KV page per
request, eager attention, and BF16 weights. B1 generated 16/16 exact HF tokens
on the pinned B200 with both 16- and 128-token prompts. After the attention
correction, B8 with 128- and 256-token prompts generated 128/128 exact HF
tokens over 16 steps in each same-device run and matched all 64 final decoder
hidden states. The 128-token run also matched captured attention and GDN
tensors. Set
`MPK_QWEN_EXPERIMENTAL_B8=1` to run B8; the default offline API
rejects it pending validation beyond cache length 271 and more inputs. B8 ignores EOS to
keep GDN state indexed by stable request slots. These counts use tokenwise HF
prefill, the same B200, CUDA 12.8, PyTorch 2.7.0+cu128 and Transformers 5.14.1.
`run_fixed.py` preserves the earlier fixed-step numerical probe.

The final 27B-only fixed-step graph also passed the B200 full-model suite
against separately frozen HF references using matching prefill modes:
3,194/3,194 tensor checks met the 2% relative RMS and 5% peak bounds
(maximum relative RMS 1.5907%); 12/12 boundary and cache next-token checks
and all eight 16-token continuations matched. The retained cross-mode report
has 3,190/3,194 accepted checks and fails its original gate. Native HF itself
changes logits between block and tokenwise prefill, and one Chinese continuation
changes tokens, so cross-mode results must not be described as model parity.
The same-mode suite validates the fixed-step probe; the offline single-call
path has the independent operator and prompt/logit checks above.

The earlier H100 SDPA validation passed the tested output-logit and greedy-token
checks, with three intermediate states exceeding the strict numerical limit.
Those historical measurements precede the B200 fixed-batch reduction changes;
the revised kernels have not been revalidated on H100. See the measured results
below for the tested environments and scope.

| Property | Qwen3.8-27B |
| --- | --- |
| Decoder layers | 64: 48 Gated DeltaNet, 16 full attention |
| Hidden / MLP width | 5120 / 17408 |
| GDN heads | 16 Q/K, 48 V; three V heads share each Q/K head |
| Full attention heads | 24 Q, 4 KV; head dimension 256 |
| Vocabulary | 248320; separate embedding and LM head weights |
| Precision | BF16 weights and activations; FP32 recurrent states |

The preserved probe interface supports synchronized fixed batches of 1–32 requests on one
B200, with a 1024-token cache per request. H100 retains batch 1. It includes text inference, greedy generation,
reset, one-token MPK decoding, and optional 64-token MPK prefill on Hopper.
`prefill_mode="block"` uses PyTorch CUDA prefill; `"stream"` calls MPK once per
input token. Vision, MoE routing, quantized weights, multi-GPU inference, and
continuous batching are outside this adapter's scope.

The new grouped GDN tasks read the checkpoint's compact Q/K tensors directly;
weights are not expanded threefold. Batch-one gate projections are padded to a 64-column GEMM tile;
multiple requests use contiguous gate rows and 32-column SM100 tiles. Full attention supports the six-query-head GQA mapping and uses the
24-head FlashAttention split heuristic for SDPA prefill and decoding. Narrow
projections and the MLP down projection reproduce the pinned H100 cuBLAS split-K
schedule on SM90; those choices were checked at every input length from 1 to
1024. SM100 decoding uses the native Blackwell linear tasks instead.

## Run

Use an allocated H100 or B200 and the CUDA/PyTorch environment used for
verification: CUDA 12.8, PyTorch 2.7.0+cu128, Transformers 5.14.1.
Numerical results below were measured on specific 132-SM H100 and 148-SM B200
devices; they do not establish the same rounding behavior on other GPUs or
library versions.
The model checkpoint is about 55.6 GB; leave additional device memory for
runtime metadata, scratch space and caches. The CPU-first loader also needs
host memory for text weights.

Download the pinned revision into your chosen storage directory:

```bash
python demo/qwen3_8/download_model.py --output-dir /path/to/Qwen3.8-27B
```

This also writes `download_manifest.json` for the validation commands below.
Use `--verify-only` to validate an existing copy without downloading it again.

Build the native extension from this checkout. After changing C++ task
registration, `--force` is necessary to relink the extension against the new
static runtime library:

```bash
python setup.py build_ext --inplace --force
export PYTHONPATH="$PWD/python"
python demo/qwen3_8/run.py \
  --model /path/to/Qwen3.8-27B \
  --output-dir /path/to/new-kernel-directory \
  --tokens-out /path/to/offline-tokens.json \
  --prompt 'Explain why the sky looks blue in two sentences.'
```

The offline demo uses eager attention and the checkpoint's text EOS token.
`--ignore-eos` runs until the requested token count. `Qwen38Builder` is
registered under `Qwen3.8-27B` and `Qwen/Qwen3.8-27B`; this demo constructs
it with the resolved local checkpoint path. Use
`run_fixed.py` for the earlier fixed-step SDPA and MPK chunk-prefill probes.
The optional token JSON records exact IDs for comparison with the independent
HF oracle from `test_qwen38_model_gpu.py`; decoded text alone is insufficient
to establish token equality. `--logits-out /path/to/logits.pt` saves the final
BF16 vocabulary logits for numerical comparison.
Set `MPK_QWEN_EXPERIMENTAL_B8=1` and add `--batch-size 8` to exercise the
experimental B8 one-call offline graph. It repeats the
same prompt eight times and ignores EOS; the benchmark's B8 path supplies
eight distinct, equal-length prompt rows. The B8 graph defaults to request-row
SiLU and residual-add tasks and 128-column linear tiles where divisible.
The corrected offline request-slot graph has a same-device token check recorded
in `benchmark/qwen38_sglang/results/b8_offset_fix_20261122/README.md`.
Longer B8 contexts should be checked with the same tokenwise prefill mode
before the offline path is accepted. Freeze the HF tokens in one
process, then compile and verify MPK in another:

```bash
export MPK_QWEN_EXPERIMENTAL_B8=1  # Only when validating B8.
python tests/qwen3_5/verify_qwen_offline_gpu.py --architecture qwen38 \
  --phase freeze --model /path/to/Qwen3.8-27B \
  --prompts-json /path/to/b8-prompts.json --oracle /path/to/b8-hf-stream.json
python tests/qwen3_5/verify_qwen_offline_gpu.py --architecture qwen38 \
  --phase verify --model /path/to/Qwen3.8-27B \
  --prompts-json /path/to/b8-prompts.json --oracle /path/to/b8-hf-stream.json \
  --compiled-dir /path/to/b8-offline-kernel
```

Checkpoint shards normally must resolve inside the checkpoint directory.
When explicitly distributing a checkpoint across caller-owned storage,
`--shard-root /path/to/other/storage` permits symlinks into that root.
Absolute shard names and `..` traversal in the index remain invalid.

## Fixed batches on B200

```python
from mirage.mpk.models.qwen3_5.model import Qwen35Model

# prompt_rows is a list of four integer token lists of equal length.
model = Qwen35Model.from_pretrained(model_directory, batch_size=4,
                                  attn_implementation="eager")
try:
    model.compile(kernel_directory)  # Compile separately for each batch size.
    outputs = model.generate(prompt_rows, max_new_tokens=64, eos_token_ids=[])
    model.reset()
    logits = model.prefill(prompt_rows, mode="block")  # [4, vocab]
    logits = model.step(model.tensors["next_token"].tolist())
finally:
    model.close()
```

`batch_size` accepts integers from 1 through 32; size 1 preserves the original
flat token-list and scalar-step interface. Larger batches require SM100 and
eager attention. Prefill/generate accept `[batch, length]` host token lists;
step accepts exactly one integer token per request. All prompt rows must have
the same length; there is no padding mask or dynamic request admission. The
1024-token capacity applies to each row, including cached decode tokens.
EOS stops each returned continuation independently. Finished slots continue
executing privately until every row finishes or the generation limit is
reached; their extra tokens are discarded. Reset starts a new whole batch.

Projection and MLP GEMMs operate on `[batch, hidden]` matrices sharing one
weight allocation. Norm, convolution, GDN, RoPE, KV store, attention and
argmax use a request axis in the same MPK graph. GDN and KV buffers are private
to each row. Every decode step launches MPK once for the entire batch.
`block` prefill uses vectorized PyTorch CUDA operations; `stream` uses only
MPK, once per prompt column. The 64-token MPK chunk graph still requires
batch 1 and Hopper. This fixed-batch path does not implement continuous batching.

Validation first freezes native HF reference tensors, then compiles and runs
MPK in a separate process on the same single B200:

```bash
python tests/qwen3_5/test_batch_model_gpu.py --phase freeze \
  --model /path/to/Qwen3.8-27B --oracle-dir /path/to/batch-oracle
python tests/qwen3_5/test_batch_model_gpu.py --phase verify \
  --model /path/to/Qwen3.8-27B --oracle-dir /path/to/batch-oracle \
  --output-dir /path/to/batch-validation --stop-on-numerical-failure
```

The default sizes are 1, 2, 3, 4, 8, 16 and 32. The suite compares each row's
logits and selected GDN/KV states against HF, checks exact permutation/replay
and forbids PyTorch arithmetic during decode. It exercises streaming, block
prefill followed by decode, cached multi-token chunks and a 1024-token case.
Execution/isolation (`passed`) and HF numerical acceptance
(`numerical_acceptance`) are separate report fields. `strict_acceptance`
additionally requires every greedy prediction to match. Numerical thresholds
remain 2% relative RMS and 5% reference peak plus 1e-5; bitwise equality is
reported separately. The command exits unsuccessfully on a numerical or token
failure. The stop flag also prevents compiling subsequent batch sizes.

The fixed-batch numerical path follows the pinned native reduction schedules:
RMSNorm and Q/K normalization account for the number of rows in PyTorch's
reduction, aligned Blackwell QK uses BF16 Tensor Cores, and unaligned 27B QK
uses batch- and length-dependent contiguous or interleaved FP32 accumulators. The 27B value product also follows
cuBLAS's partial-first 64-element WMMA tile and measured GEMVX schedules,
including the batch-dependent short-cache reduction and the odd cache-length
ranges 59–67 and 121–135. Attention receives the batch size explicitly so the
short-cache schedule changes at the same batch boundary as the native backend.
`profile_blackwell_values_gpu.py` records the native choice for every cache
length from 1 through 1024. These schedules are specific to the tested
PyTorch/CUDA/GPU environment, not a promise of identical rounding on other
backends. Saved-input operator tests isolate arithmetic from accumulated
model error; the independent frozen-HF full-model tests remain the acceptance
criterion.

The original B=1 full-model suite also compares tokenwise MPK prefill against
the whole-prompt HF oracle. These are different numerical paths even inside
native HF: a direct native replay measured 3.61% and 2.75% relative RMS on the
math and code prompts, respectively, and a different Chinese continuation.
The original oracle and cross-mode results are retained unchanged. To test
implementation parity with the same prefill path, freeze an additional native
HF stream oracle and supply it explicitly:

```bash
python tests/qwen3_5/test_qwen38_model_gpu.py --phase stream-oracle \
  --model /path/to/Qwen3.8-27B --manifest /path/to/download_manifest.json \
  --prefill-mode block --reference /path/to/block-oracle/oracle.pt \
  --output-dir /path/to/stream-oracle
python tests/qwen3_5/test_qwen38_model_gpu.py --phase verify \
  --model /path/to/Qwen3.8-27B --manifest /path/to/download_manifest.json \
  --prefill-mode block --reference /path/to/block-oracle/oracle.pt \
  --stream-reference /path/to/stream-oracle/oracle.pt \
  --compiled-run /path/to/batch-validation/b1 --output-dir /path/to/full-validation
```

The stream freezer must first reproduce all four original block continuations
and their logits bitwise. It records the original oracle's hash and freezes
native tokenwise logits and greedy histories independently of MPK. Verification
keeps the original `passed`, `checks`, and `generations` fields, and adds
`same_mode_passed`/`same_mode_summary`, `stream_checks`, and `stream_generations`.
With an explicit stream reference, the exit status uses same-mode acceptance:
all original boundary/state/block-generation checks plus stream checks against
native stream HF. Both the numerical thresholds and exact-token requirement
remain unchanged. A failing cross-mode result is never relabeled as passing.

Latest B200 B=1 regression after the short QK/PV dispatch fixes: same-mode
acceptance passed **3194/3194** tensor checks (2859 bitwise), with maximum
relative RMS **1.591%**; all 12 boundary next-token cases and all eight
16-token continuations matched. The original cross-mode view remains failed:
3190/3194 tensor checks pass. Its four remaining failures compare stream MPK
against block HF. The independent batch-one suite also passed 72/72 checks,
all bitwise, with exact greedy predictions. Reports and frozen inputs are
backed up in `artifacts/qwen38-mpk-batch-20261116/` beside this checkout.
That earlier production revision then passed every frozen-HF fixed-batch case:

| Batch | Tensor checks, all bitwise | Exact greedy predictions |
| --- | ---: | ---: |
| 1 | 72 | 8 |
| 2 | 180 | 20 |
| 3 | 216 | 24 |
| 4 | 288 | 32 |
| 8 | 576 | 64 |
| 16 | 1152 | 128 |
| 32 | 2304 | 256 |
| Total | 4788 | 532 |

Each batch used a fresh compiler process and its unchanged native HF oracle.
The cases use distinct rows, tokenwise and cached/block prefill, and exact
permutation/reset replay; B=2 additionally covers 1024 cached tokens. These
results validate the seven listed batch sizes under that workload. Operator
tests also cover every batch size from 1 through 32.

The newer B32 Blackwell attention dispatch was checked on an otherwise idle,
exclusively monitored B200. With BF16 eager attention and TF32 disabled, its
block-prefill MPK continuations matched a same-GPU native HF replay at prompt
lengths 128, 512 and 960: **6144/6144 greedy tokens**, across all 96 request
rows (32 per length, 64 tokens per row). Both phases passed; the GPU guard and
occupancy monitor recorded no competing process. Before execution, the runner
matched 28 source paths and SHA-256 digests; the verified production
`qwen35_attention.cuh` SHA-256 was
`8e3c9539d8a70de06560652f945ec138063643f4c4c447819df5deb75820467e`.
The paired reports are backed up under
`artifacts/qwen38-token-diagnosis-20261118/verify-b32-cross-b200-20261119-05/`
beside this checkout. This establishes greedy-token parity for these three
workloads, not bitwise equality of every intermediate tensor or arbitrary
lengths. The benchmark tables below remain historical.

For the shared B200 B32 benchmark, a healthy GPU passes admission when eleven
samples show at most 10 GiB of total used memory and at most 5% GPU utilization.
Existing low-use processes do not block admission. The runner binds one GPU by
UUID and invalidates a timed stage if a new process enters or a resident
process materially grows its GPU memory; each stage records its own baseline.
The guard cannot detect a resident process that resumes compute without
allocating more memory.

An initial B32 measurement completed on one B200 with 2 warmups and 5 measured
repetitions per prompt length. The runner exited 78 because its teardown
monitor flagged the already finished MPK process as a competing process. A
post-hoc audit verified that the flag was written after the complete report,
that every measured MPK output token matched same-GPU HF, and that the source
hashes matched. These medians are **exploratory**, not a clean benchmark pass:

| Prompt tokens | Total HF / MPK (s) | TTFT HF / MPK (s) | TPOT HF / MPK (ms) | Output HF / MPK (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.75 / 13.32 | 0.54 / 0.55 | 51.0 / 202.6 | 546.3 / 153.8 |
| 512 | 5.67 / 16.08 | 2.01 / 2.03 | 58.2 / 223.0 | 361.1 / 127.4 |
| 960 | 8.11 / 19.36 | 3.84 / 3.86 | 67.7 / 246.0 | 252.6 / 105.8 |

The original exit status remains 78. Later retries were stopped when other
tasks entered the selected GPU. The timing evidence and limitations are in
`artifacts/qwen38-b32-benchmark-20261119/benchmark-b32-20261119T021137Z-1051740/adjudication.json`
beside this checkout.

`benchmark_qwen38_gpu.py --backend mpk --batch-sizes 1 2 4 8 16 32`
now runs real MPK batches. Supply a completed `--hf-reference`, checkpoint
`--manifest`, and the prior `--validation-report`. It compiles each size unless
`--compiled-root` points to validated `b1/`, `b2/`, ... directories. Reuse
checks the batch size, GPU UUID, attention backend and production source hashes.
Compilation, weight loading and correctness checks are outside timed regions.

## Validation

CPU tests use the pinned official configuration and independent grouped-GDN
and GQA references. The isolated GPU suite exercises the production templates
at 27B dimensions, including cached state, partial 64-token tiles and long KV
caches. The full-model suite freezes an independent native HF oracle, exits,
and loads MPK in a separate process so two 27B models need not fit together.

```bash
python tests/qwen3_5/test_qwen38_isolated_gpu.py --output-dir /path/to/isolated-run
python tests/qwen3_5/test_qwen38_model_gpu.py \
  --phase oracle --attn-implementation sdpa --model /path/to/Qwen3.8-27B \
  --manifest /path/to/Qwen3.8-27B/download_manifest.json --output-dir /path/to/oracle-run
python tests/qwen3_5/test_qwen38_model_gpu.py \
  --phase verify --attn-implementation sdpa --model /path/to/Qwen3.8-27B \
  --manifest /path/to/Qwen3.8-27B/download_manifest.json \
  --reference /path/to/oracle-run/oracle.pt --output-dir /path/to/mpk-run
```

Set `QWEN35_EXPECTED_GPU_UUID` to the allocated GPU UUID for these tests. The
manifest records the repository, pinned revision, and each file's SHA-256.
Both phases verify those hashes. Numerical limits are fixed at 2% relative RMS
and 5% reference peak plus 1e-5; greedy continuations must match exactly.
Tests cover lengths 1, 2, 63, 64, 65, 127, 128, 129, 1023 and 1024, cached
chunk continuation, subsequent decode, four natural prompts, reset/replay,
invalid-input rejection, and absence of PyTorch arithmetic fallback in MPK
prefill. `compile_qwen38_gpu.py` uses synthetic weights only to test the full
graph's compilation and execution; it is not checkpoint numerical acceptance.

The basic isolated command runs 71 checks. The extended 3,687-check run also
supplied `--norm-fixture` and `--projection-fixture` with captured real
operands, plus `--down-model` with the checkpoint directory. Those optional
fixtures contain checkpoint-derived tensors and are kept with validation
artifacts, outside the source repository. The full-model suite independently
loads the checkpoint and always runs its 3,194 tensor comparisons.

Measured results on the pinned checkpoint and H100 environment:

| Check | Result |
| --- | --- |
| CPU adapter, contract, layout and compile-option tests | 61/61 passed |
| Isolated GPU kernels, including saved real inputs and all projection lengths | 3,687/3,687 passed; 3,663 bitwise exact |
| SDPA logits, boundary/cache cases and forced generation histories | 170/170 passed; maximum relative RMS 1.0044% |
| SDPA greedy tokens | All 12 boundary/cache cases and 8 natural continuations matched |
| SDPA strict comparison of all tensors | 3,191/3,194 passed; **strict suite failed** |
| Eager logits | 169/170 passed; maximum relative RMS 2.3386% |
| Eager greedy tokens | All 12 boundary/cache cases and 8 natural continuations matched |
| Eager strict comparison of all tensors | 3,145/3,194 passed; **strict suite failed** |

The eight continuations comprise four prompts tested with both MPK and
PyTorch prefill, each generating 16 tokens. Both backends passed exact
reset/replay, invalid-input atomicity, and the MPK no-PyTorch-arithmetic check.

SDPA's remaining failures are hidden outputs at layers 59–61 (zero-based),
at the eighth step of the seven-single-token streaming case. Their largest
relative RMS error is 5.3471%; the final logits still meet the unchanged
output tolerance. Eager additionally misses the logits limit on the initial
math-prompt prediction with MPK prefill. These failures remain in the test
reports and are not waived. The adapter remains experimental until these
failures are resolved.

The B200 fixed-batch performance measurements below are separate from these
H100 correctness results.


## Historical Catalyst B200 validation before batch fixes (2026-11-16)

The Catalyst run uses one 148-SM B200, CUDA 12.8, PyTorch 2.7.0+cu128 and
Transformers 5.14.1. Its checkpoint has the same pinned revision and file
hashes as the H100 run. The validation environment, model and build are
independent of other jobs on the host.

Use `--attn-implementation eager --prefill-mode block` on B200. This runs
PyTorch block prefill followed by MPK decode; `--prefill-mode stream` uses
MPK for each prompt token. The default demo's SDPA runtime and 64-token MPK
prefill remain restricted to SM90. Passing the isolated SDPA arithmetic
tests does not establish support for the complete B200 SDPA graph.

The full-model test now accepts `--prefill-mode block` in both its oracle
and verify phases. For this mode the eight continuations test `stream` and
`block` prefill, not 64-token MPK prefill. The independent oracle is frozen
on the same B200; the H100 reference is not reused. Numerical tolerances
and exact-token requirements are unchanged.

| Check | B200 result |
| --- | --- |
| CPU adapter, contract, layout and compile-option tests | 61/61 passed |
| Basic isolated GPU kernels at 27B dimensions | 71/71 passed; 41 bitwise exact |
| Full graph with synthetic zero weights | Compiled and executed; peak allocated GPU memory 54,284,533,248 bytes |
| Logits against the same-device HF oracle | 167/170 passed; maximum relative RMS 3.6858% |
| Greedy tokens | 12/12 boundary/cache cases and 8/8 continuations matched |
| Strict comparison of all tensors | 3,162/3,194 passed; 2,284 bitwise exact; **strict suite failed** |
| Reset/replay, invalid-input atomicity and MPK decode fallback guard | Passed |

The basic isolated test covers GDN, cached state, eager/SDPA attention,
RMSNorm and Q/K normalization plus RoPE. It does not exercise every native
Blackwell GEMM shape. The full graph smoke test uses zero weights and is
not evidence of checkpoint numerical parity. B200 performance measurements are reported separately below.

The 32 strict failures comprise 3 logits, 27 hidden outputs and 2 KV-cache
comparisons. The logits failures are the second decode step after a
65-token block prefill, and the initial math/code predictions after streaming
MPK prefill. The largest hidden-output relative RMS is 9.8492%. All eight
16-token continuations (four prompts with both stream and block prefill)
match the native HF greedy tokens despite these numerical failures.

This earlier full suite completed and returned a failing exit status; no
thresholds were relaxed. It precedes the fixed-batch reduction fixes above.
The oracle, complete report and first-logits
failure tensors are preserved under `artifacts/qwen38-b200-20261116/` beside
the checkout. This earlier result does not validate the default
SDPA/MPK-chunk demo on B200 or the subsequently added multi-request path.


## Historical benchmark before MPK batch support (2026-11-16)

All runs used the same 148-SM B200 and pinned checkpoint/environment as the
validation above. HF batches are genuine parallel requests in tensors shaped
`[batch, sequence]`, with distinct synthetic text prompts of equal length.
At the time of this earlier run, the MPK adapter supported **batch 1 only**.
These historical measurements precede the fixed-batch implementation above.

The workload generates exactly 64 greedy tokens without EOS stopping, at
input lengths 128, 512 and 960. Each configuration has 2 warmups and 5 timed
repetitions. Times use synchronized wall clock around the inference loop,
with CPU delivery of each generated token. Loading, compilation,
tokenization and correctness checks are excluded. The reported decode rate
is aggregate `batch * 63 / decode_seconds`; per-sequence token latency is
`decode_seconds / 63`. End-to-end output rate also includes prefill/reset.

Both paths use eager attention, BF16 weights and FP32 recurrent state, with
TF32, torch.compile and CUDA graphs disabled. The HF baseline uses the native
PyTorch GDN fallback, not fused FLA/causal-conv kernels or vLLM. One model is
resident at a time. HF retains 921,460,264 bytes (0.86 GiB) of unused vision
weights on GPU; MPK loads text weights only. Memory figures are PyTorch peak
allocated bytes, not total CUDA-process memory.

Median aggregate decode throughput:

| HF batch | Decode tok/s, input 128 | Input 512 | Input 960 | Peak allocated GiB, input 960 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23.8 | 23.5 | 23.9 | 51.47 |
| 2 | 47.1 | 46.8 | 46.8 | 51.96 |
| 4 | 92.4 | 92.4 | 92.6 | 52.94 |
| 8 | 185.7 | 185.1 | 185.9 | 54.93 |
| 16 | 377.0 | 371.2 | 333.8 | 58.89 |
| 32 | 626.1 | 548.2 | 470.7 | 66.80 |

All 18 HF configurations completed without OOM. Their output tokens match
native HF `generate` for the same batch, and all repeated outputs are exact.
This does not require different batch sizes to have identical floating-point
rounding. The maximum measured TPOT coefficient of variation is 1.90%.

For the 512-token input, increasing batch from 16 to 32 changes median
per-sequence decode latency from 43.11 to 58.37 ms/token, while aggregate
decode throughput rises from 371.2 to 548.2 tok/s. TTFT rises from 1.040 to
2.011 seconds. These measurements describe the tested fixed-batch workload;
they are not a production serving or optimized-HF performance claim.

The benchmark carries prior strict numerical failures into its report. The
current command above supports MPK batches; this earlier run only measured
MPK batch 1, using the then-current compiled graph.


MPK measurements use PyTorch block prefill plus the previously compiled
Blackwell MPK decode graph. Median results for batch 1:

| Input tokens | TTFT ms | Decode ms/token | Decode tok/s | End-to-end output tok/s | Greedy tokens vs HF |
| --- | ---: | ---: | ---: | ---: | --- |
| 128 | 169.28 | 26.14 | 38.25 | 35.23 | 64/64 exact |
| 512 | 207.41 | 29.57 | 33.82 | 30.92 | **Mismatch** |
| 960 | 251.34 | 33.39 | 29.95 | 27.18 | **Mismatch** |

All MPK repetitions are deterministic, but only the 128-token input matches
HF's full 64-token continuation. The 512- and 960-token prompts diverge.
Consequently these are performance observations of an experimental path,
not a speedup claim at established equivalent numerical correctness. The
strict numerical failure was present when these timings were collected. This
historical run contains no MPK batch-greater-than-one timings.

Raw reports, 105 measured samples, summary CSV, charts and reproduction
scripts are preserved in `artifacts/qwen38-b200-benchmark-20261116/` beside
the checkout. The benchmark finished and released its single GPU.
