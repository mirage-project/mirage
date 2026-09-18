# OpenAI-compatible text server

The upstream SM100 two-stage top-k/top-p sampler remains available through
`sampling_partial_layer()` / `sampling_reduce_layer()` and Qwen3's compiled
`do_sample` graph configuration. In `online_pinned` mode, Qwen3 uses the serving
sampler instead: each request supplies its own settings, seed, and penalty
history, without the SM100 graph sampler's fixed candidate limit.

The server accepts the upstream `--do-sample --temperature ... --top_p ...
--top_k ... --seed ...` flags as defaults for omitted request fields. Hyphenated
`--top-p` and `--top-k` aliases are also accepted. Explicit request values take
precedence, including `temperature=0` for greedy decoding. Without `--do-sample`,
HTTP sampling defaults are unchanged. `--sampling-topk-max` is retained for
command-line compatibility and does not limit the serving sampler.

Raw benchmark artifacts under `benchmark/results/` are local files excluded
from Git.

Launch a single-GPU persistent-kernel server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mirage.engine.launch_server \
  --model Qwen/Qwen3-0.6B --port 8000 \
  --max-seq-length 2048 --max-num-batched-requests 4 \
  --max-num-batched-tokens 8 --page-size 4096 --max-num-pages 16
```

`max_seq_length` covers the complete templated prompt plus generated tokens.
The KV page pool must cover every configured concurrent sequence at maximum
length. `--max-pending-requests` bounds accepted requests (default 128), with
HTTP 429 on overload. Prefix caching is not implemented; every chat request
prefills its entire history. Multi-GPU tensor-parallel HTTP serving is not
currently supported; run one server per GPU for independent replicas.

## Python OpenAI client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
messages = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "My name is Ada."},
]
first = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B", messages=messages,
    temperature=0, max_completion_tokens=64,
)
messages.append(first.choices[0].message.model_dump(exclude_none=True))
messages.append({"role": "user", "content": "What is my name?"})
for chunk in client.chat.completions.create(
    model="Qwen/Qwen3-0.6B", messages=messages,
    temperature=0.7, top_p=0.9, seed=42,
    max_completion_tokens=64, stream=True,
    stream_options={"include_usage": True},
):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Supported contract

- `POST /v1/chat/completions`: full ordered message history, with system, user,
  assistant, developer, and tool-result messages. Text strings and text content
  parts are supported. Assistant tool-call history must have matching tool
  results. Function arguments are JSON strings on the API boundary.
- Developer messages use the explicit `--developer-role system` adapter by
  default. This preserves their content as system messages, but cannot reproduce
  a separate developer priority level in models trained without that role.
  `native` passes the role through to a supporting template; `reject` refuses it.
- Tool history is rendered using the model's chat template. This does not enable
  new tool-call generation: `tools`, `tool_choice`, and structured tool-call
  output parsing are outside this text-generation contract.
- `POST /v1/completions`: one raw string prompt, without chat templating.
- `GET /v1/models`, `GET /health`.
- `temperature` (0–2, default 1), `top_p` (0–1, exclusive of zero), `seed`
  (nonnegative signed 64-bit), `frequency_penalty` and `presence_penalty` (-2–2),
  `logit_bias` (-100–100, at most 256 vocabulary entries).
- `max_completion_tokens` or `max_tokens`: positive output budget. Both are
  accepted if equal, otherwise rejected. Omission uses remaining context space.
- `stop`: a nonempty string or up to four nonempty strings. Matching text is
  excluded, including when a stop string crosses token boundaries.
- `stream`, `stream_options.include_usage`, `n=1`, and optional `user` metadata
  (accepted but not persisted).
- `top_k` (0 disables) and `repetition_penalty` (positive, 1 disables) are Mirage
  extensions, passed using `extra_body` in clients that do not expose them.

Unknown fields are rejected with an OpenAI-shaped 400 response. This includes
multimodal inputs, logprobs, multiple choices, structured output constraints,
Responses API fields, and tool-generation options. No credentials are required;
place authentication at your deployment boundary if needed.

## Generation semantics

Bias is applied first, then repetition penalty, frequency/presence penalties,
temperature, top-k, and top-p. Frequency/presence count only the generated portion
of the current request; repetition counts prompt and generated tokens. Greedy
sampling (`temperature=0`) applies bias and penalties but ignores top-k/top-p.
Ties prefer lower token IDs. The nucleus includes the token that crosses the
probability threshold, after top-k renormalization.

A seed and generated-token position determine random draws independently of
request IDs and batch slots. Identical seeds do not guarantee identical output
across model, kernel, or hardware changes that alter logits. GPU sampling uses
FP32 scores and a radix selection implementation; performance tuning remains
possible without changing the API or transport.

Finish reasons distinguish EOS/stop strings (`stop`) from exhausted output
budgets (`length`). Usage counts the complete templated prompt and sampled
completion tokens, including EOS or tokens containing a matched stop string.
Text streaming uses cumulative decoding and withholds incomplete Unicode and
possible stop-string prefixes. Cancelling/disconnecting requests signals the
GPU and releases rows through the existing completion acknowledgment protocol.

## Tests

```bash
python -m pytest tests/engine -q
MIRAGE_TEST_GPUS=0,1,2,3 python -m pytest tests/engine/test_sampling_gpu.py -q
python tests/engine/live_server_check.py
```

The first command runs CPU API/engine tests. GPU sampler tests compile the actual
serving CUDA code and test probability distributions, masks, penalties, and
batch-order-independent seeds on each selected GPU. The live check builds a
Qwen3-0.6B runner and tests real persistent-kernel generation through HTTP.
