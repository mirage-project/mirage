# Qwen3.8-27B BF16 text decode on MPK

This demo loads the dense text decoder of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)
and runs greedy, fixed-batch inference on one B200. The checkpoint declares
`Qwen3_5ForConditionalGeneration`, so the Python package and CUDA task names
use `qwen3_5`. The implementation is experimental: decode performance is far
from optimal, and the tested scope does not include vision or continuous
batching.

The 64-layer text decoder has 48 Gated DeltaNet (GDN) layers and 16 full
attention layers. GDN keeps a three-token convolution history and an FP32
recurrent state per request. Prefill initializes both this state and the
attention KV cache; MPK then executes the fixed-batch decode graph.

## Run

Use the B200 CUDA/PyTorch environment required by this checkout. Download the
pinned checkpoint, build the native extension, and run the offline demo:

```bash
python demo/qwen3_8/download_model.py --output-dir /path/to/Qwen3.8-27B
python setup.py build_ext --inplace --force
export PYTHONPATH="$PWD/python"
python demo/qwen3_8/run.py \
  --model /path/to/Qwen3.8-27B \
  --output-dir /path/to/compiled-qwen38 \
  --tokens-out /path/to/tokens.json \
  --prompt 'Explain why the sky looks blue in two sentences.'
```

`run_fixed.py` exercises the fixed-step model API directly. Compile separately
for each batch size. For B8 offline experiments, set
`MPK_QWEN_EXPERIMENTAL_B8=1`; the default offline API rejects B8 pending
broader validation. The fixed-batch path supports B1–B32 within a 1024-token
cache per request. B8/B16 use the partitioned decode graph on B200; B4/B32 use
the generic graph.

```python
from mirage.mpk.models.qwen3_5.model import Qwen35Model

model = Qwen35Model.from_pretrained(model_directory, batch_size=4,
                                    attn_implementation="eager")
try:
    model.compile(kernel_directory)
    output_ids = model.generate(prompt_rows, max_new_tokens=64, eos_token_ids=[])
finally:
    model.close()
```

## Verify

The focused CPU contracts are in `tests/qwen3_5/test_qwen38.py`,
`test_batch_contract.py`, and `test_model_contract.py`. For a same-device HF
greedy-token check, freeze an HF oracle and verify the MPK run with the same
checkpoint, prompts, and B200:

```bash
python tests/qwen3_5/verify_qwen_offline_gpu.py --architecture qwen38 \
  --phase freeze --model /path/to/Qwen3.8-27B \
  --prompts-json /path/to/prompts.json --oracle /path/to/hf-oracle.json
python tests/qwen3_5/verify_qwen_offline_gpu.py --architecture qwen38 \
  --phase verify --model /path/to/Qwen3.8-27B \
  --prompts-json /path/to/prompts.json --oracle /path/to/hf-oracle.json \
  --compiled-dir /path/to/compiled-verify
```

For the four-backend decode-only comparison in the draft PR, see
`benchmark/qwen38_sglang/README.md`. SGLang and vLLM are installed separately;
the repository contains a comparison harness, not either serving engine.
