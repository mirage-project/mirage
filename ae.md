# Artifact Evaluation of Mirage

## Get Started
```
conda activate mirage
export MIRAGE_ROOT=/path/to/mirage
cd $MIRAGE_ROOT
pip install .
```

## Detailed Instructions

### Micro-benchmark Evaluation
For baseline implementation, refer to https://github.com/jiazhihao/mirage_baselines.

To reproduce the benchmark results shown in the paper, run the python scripts under `$MIRAGE_ROOT/benchmark`. The optimization time may take up to 4 hours. In order to save time, you can use our cached results in `$MIRAGE_ROOT/benchmark/saved_mugraphs` to avoid running from scratch.

To evaluate the optimization time, run Mirage from scratch, and look at the line
```
[Search] Second step finished. Time elapsed: xx.xxsec
```
which reports the optimization time.

Specifically, followings are how to reproduce each benchmark result:


#### GQA
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/group_query_attention.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/gqa_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/group_query_attention.py --bs {batch_size}
```


#### QKNorm:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/qknorm_gqa.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/qknorm_gqa_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/qknorm_gqa.py --bs {batch_size}
```

#### RMSNorm:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/rmsnorm_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py --bs {batch_size}
```

#### RMSNorm:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/rmsnorm_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py --bs {batch_size}
```

#### LoRA:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/lora.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/lora_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/lora.py --bs {batch_size}
```

#### GatedMLP
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/gated_mlp.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/gated_mlp_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/gated_mlp.py --bs {batch_size}
```

#### nTrans
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/norm_transformer.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/ntrans_bs{batch_size}.py
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/norm_transformer.py --bs {batch_size}
```

### End-to-end Evaluation
Baseline implementations are in `$MIRAGE_ROOT/demo/pytorch`.

To reproduce the end-to-end evaluation results shown in the paper, run the python scripts under `$MIRAGE_ROOT/benchmark/end-to-end`. Optimization usually takes less than 1 min.

#### Chameleon-7B
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/chameleon.py --bs {batch_size}
```

#### LLaMA-3-8B
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/llama.py --bs {batch_size}
```

#### GPT-3-7B-LoRA
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/lora.py --bs {batch_size}
```

#### nGPT-1B
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/ngpt.py --bs {batch_size}
```
