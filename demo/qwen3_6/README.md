# Qwen3.6-35B-A3B BF16 text adapter

This adapter uses the same MPK hybrid GDN/full-attention graph as Qwen3.8,
with the repository's SM100 top-8 MoE routing and grouped expert kernels.
It supports text-only greedy decoding on one B200 at B1, with a
1024-token context limit. The fixed B8 graph is experimental. Set
`MPK_QWEN_EXPERIMENTAL_B8=1` to run it explicitly; the default
offline API rejects B8. Vision inputs, MTP, quantization, continuous
batching, and MoE chunk prefill are outside this graph.
After the attention and MoE corrections, B1 with a 128-token prompt matched
HF for 16/16 generated tokens and all 40 final decoder hidden states. B8 with
a 128-token prompt matched 128/128 generated tokens across 16 decode steps.
The B8 256-token prompt also matched 128/128 generated tokens; all 40 final
hidden states, routed and shared expert outputs, and final logits matched
elementwise on the same B200 as HF. The MoE top-8 task follows PyTorch's tie
ordering, FP32 tree sum, and per-value division before BF16 rounding; its
expert sum uses PyTorch's four FP32 accumulators across eight slots. Attention
Q·K/P×V dispatch has been checked through cache length 271. Longer contexts
and additional prompts remain to be checked, so B8 stays behind the
experimental switch.
The HF oracle uses BF16 weights, eager attention, and the explicit
`grouped_mm` expert backend. MPK rounds each weighted expert result to BF16,
sums the top-8 slots in FP32 with the CUDA reduction order, then adds the
shared expert at the BF16 boundary.
Run the oracle's two phases with the same
PyTorch, Transformers, CUDA device, checkpoint, and prompt. The script records
these identities and rejects a mismatched verification run.

The official [configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/config.json)
and [HF MoE implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
define the checkpoint geometry, routed expert order, BF16 router weights, and
the separately gated shared expert.

Download the pinned checkpoint and verify each shard's size and SHA-256:

```bash
python demo/qwen3_8/download_model.py --model qwen36 \
  --output-dir /path/to/Qwen3.6-35B-A3B
```

```bash
python setup.py build_ext --inplace --force
export PYTHONPATH="$PWD/python"
python demo/qwen3_6/run.py \
  --model /path/to/Qwen3.6-35B-A3B \
  --output-dir /path/to/compiled-qwen36 \
  --prompt 'Explain the delta rule.' \
  --max-new-tokens 16 --tokens-out /path/to/tokens.json
```

`Qwen36WeightLoader.inspect()` validates all text tensor names, shapes, shard
index entries, and dtypes without loading the 35B model. The loader ignores
the vision tower and MTP tensor namespaces. The standalone MoE probe matched
HF's routed expert sum at B1/B8; the complete B1 offline graph matched the
same-device grouped_mm HF oracle for 16/16 generated tokens with a 16-token
prompt.
For a full-model exact-token gate, use the two-process
`tests/qwen3_5/verify_qwen_offline_gpu.py --architecture qwen36` command with `--phase freeze` followed by
`--phase verify`, the same `--prompts-json`, `--model`, `--oracle`, and
`--max-new-tokens`, and `--compiled-dir` in the verify phase. The JSON prompt
file contains one or eight equal-length integer token rows. The oracle records
the checkpoint config hash, prompt hash, device, and exact generated IDs.
For eight rows, set `MPK_QWEN_EXPERIMENTAL_B8=1` during verification.
