# DeepSeek-V4.1-Flash static layers in SGLang

These files run the attention and MoE layers of DeepSeek-V4.1-Flash on the static megakernel, inside SGLang's decode steps. The
configuration is TP4 and 8 rows per step. SGLang keeps the prefill, the steps with another row count, the all-reduce and mHC. No file
in SGLang changes: `boot/sitecustomize.py` patches SGLang when each process starts.

| File | Purpose |
|---|---|
| `boot/sitecustomize.py` | Loads the hooks. Put `boot/` on `PYTHONPATH`. |
| `dsv41_attn_hook.py`, `dsv41_attn_layer.py` | The attention layers (`../../deepseek_v4_1_moe/sglang/` has the same two files for MoE). |
| `plans/` | The plans of the layers at `DSV41_STATIC_MAX_POS=4607`. At another value, the layers use the demo's hand plan. |
| `make_prompts.py` | Makes the prompts of the whole-step comparison. |
| `sgl_generate.py` | 8 real prompts, for a check of the answers. |
| `sgl_gsm8k.py` | GSM8K accuracy. |

## Setup

- SGLang at commit `240f2a0a64`. `SGLANG_DSV4_KV_LAYOUT=v41_new` is necessary for the static layers: they read and write SGLang's KV
  pools in that layout.
- Build mirage with the same Python minor version as the SGLang venv. The layers load mirage's z3 library: put its directory on
  `LD_LIBRARY_PATH`.
- `DSV41_STATIC_CKPT` is a directory with `inference/config.json` and the checkpoint's shards. The default is SGLang's model path.
- `export MIRAGE=/path/to/mirage M=/path/to/DeepSeek-V4.1-Flash`

The commands below use:

```bash
D=$MIRAGE/demo/deepseek_v4_1_csa2/sglang
STATIC="PYTHONPATH=$D/boot DSV41_STATIC_ROOT=$MIRAGE DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_CKPT=$M"
LAYOUT="SGLANG_DSV4_KV_LAYOUT=v41_new"
```

## Check the layers against SGLang

Each decode step runs SGLang's layer, then the static layer on the same input. Rank 0 prints the relRMS of y.

```bash
env $STATIC $LAYOUT DSV41_STATIC_CHECK=1 python $D/sgl_generate.py --model $M --no-autotune --tokens 10
```

## Compare one decode step with SGLang

Plain decode, 8 rows per step, autotune off on the two sides. Alternate the two commands for at least 3 rounds on the same machine, and
compare the median decode time that `one_batch` prints.

```bash
python $D/make_prompts.py --model $M --tokens 4000 --out prompts.txt
ARGS="--model-path $M --tp 4 --batch-size 8 --input-len 4000 --output-len 256 --context-length 8448 --max-total-tokens 81920 \
      --mem-fraction-static 0.7 --disable-radix-cache --disable-flashinfer-autotune --trust-remote-code --prompt-filename prompts.txt"
env $STATIC $LAYOUT DSV41_STATIC_MAX_POS=4607 python -m sglang.benchmark.one_batch $ARGS   # the static layers
python -m sglang.benchmark.one_batch $ARGS                                                  # SGLang alone
```

`DSV41_STATIC_MAX_POS` must be at least the prompt length plus the output length. Each value builds its own kernels.

## GSM8K

```bash
env $STATIC $LAYOUT DSV41_STATIC_MAX_POS=4607 python $D/sgl_gsm8k.py --model $M --data-path test.jsonl
python $D/sgl_gsm8k.py --model $M --data-path test.jsonl
```
