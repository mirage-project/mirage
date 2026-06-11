---
name: add-mpk-model
description: Guide for adding a new model (e.g., Llama4, DeepSeek V3) to the MPK persistent kernel. Enforces a bottom-up modular build: discover catalog, pick fused variants, build the smallest composite first, test it against its PyTorch reference, then climb upward.
---

You are helping the user add a new model to MPK.

The **preferred path** is a `python/mirage/mpk/models/<model>/modeling.py` file that defines an `nn.Module` tree (each block is an `MPKModule` subclass composing catalog layers from `mirage.mpk.layers`), paired with a `demo/<model>/demo_new.py` driver. References: `python/mirage/mpk/models/qwen3/modeling.py` + `demo/qwen3/demo_new.py` (dense GQA + paged attention), and `python/mirage/mpk/models/deepseek_v3/modeling.py` + `demo/deepseek_v3/demo_new.py` (MLA + MoE).

The **legacy path** — a custom `GraphBuilder` subclass calling `pk.foo_layer()` methods directly — still works (see `python/mirage/mpk/models/deepseek_v3/builder.py` + `demo/deepseek_v3/demo.py`) but is not recommended for new models.

---

## The Mandatory Workflow: Bottom-Up Modularity

**Building a new model top-down (write the whole `ForCausalLM`, then run the end-to-end demo, then debug a failing token) is the wrong way.** When something is wrong, the failure surfaces hundreds of MB of tensor operations away from its cause and the search space is the whole model.

**Build bottom-up instead, with a numerical comparison at every level.** Each composite you write owns BOTH a `forward()` (eager PyTorch reference) and a `compile()` (MPK task registration), AND a test file that runs the composite standalone in `test_mode` and asserts `torch.testing.assert_close(forward_out, compile_out)`. You do not move up a level until the current level's test passes.

This makes every bug have one possible cause: the composite you just wrote.

### Stage 0 — Catalog discovery (no code yet)

1. List every layer the model uses: embedding, normalization, projection layers (q/k/v/o, gate/up/down), routing, expert linears, activation, attention, RoPE, lm_head, sampler.
2. For each one, search `python/mirage/mpk/layers/` for a matching `MPKModule`. The catalog inventory is in the appendix below.
3. **Pick the most fused variant available.** Examples:
   - Prefer `LinearWithResidual` over `Linear` + `layers.add` when there's a residual.
   - Prefer `MoEPermute` + `FP8GroupGEMM*` + `MoEUnpermute` (group-GEMM path) over `MoEW13FP8` + `MoEW2FP8` (per-expert path) on Blackwell when applicable.
   - Use `PagedAttention` (handles prefill + decode in one task) instead of separate prefill/decode kernels.
   - For MLA decode use `MLADecode` + `MLAReduce`. For chunked prefill use `MLAPrefillTP8Chunked`.
4. Flag every layer that has NO catalog match. For each one, stop here and invoke `/add-mpk-task` to implement it first. **Do not start composing until every leaf you need exists in the catalog.**
5. Write down (in a temporary scratch file or comments) the dependency tree you intend to build, e.g.:

   ```
   Qwen3MLP  := Linear(gate+up via shuffle_tensors) → SiluMul → LinearWithResidual(down)
   Qwen3Attention := Linear(qkv via shuffle_tensors) → PagedAttention → LinearWithResidual(o_proj)
   Qwen3DecoderLayer := RMSNorm → Qwen3Attention → RMSNorm → Qwen3MLP
   Qwen3Model := Embed → [DecoderLayer * N] → RMSNorm
   Qwen3ForCausalLM := Qwen3Model → Linear(lm_head) → ArgmaxPartial → ArgmaxReduce
   ```

Now you build this tree leaf-up.

### Stage 1 — Smallest meaningful composite

Pick the smallest composite in your tree that has a non-trivial PyTorch reference. For most LLMs this is the **MLP block** (3 linears + activation) or the **attention block** (qkv projection + attention + o projection + residual). Implement it as a single `MPKModule` subclass:

```python
class MyMLP(MPKModule):
    def __init__(self, config, *, prefix=""):
        super().__init__(prefix=prefix)
        # nn.Parameter weights — match HF state_dict naming via _load_from_state_dict
        self.gate_proj_weight = nn.Parameter(...)
        ...

    def _load_from_state_dict(self, state_dict, prefix, ...):
        # Pop HF keys, copy into self.<name>_weight.
        ...

    def forward(self, x, residual):
        """Eager PyTorch reference — the correctness oracle for this composite."""
        gate = F.linear(x, self.gate_proj_weight)
        up   = F.linear(x, self.up_proj_weight)
        silu = (F.silu(gate.float()) * up.float()).to(x.dtype)
        return F.linear(silu, self.down_proj_weight) + residual

    def compile(self, x_dt, *, residual_dt, output):
        """MPK task wiring — compose catalog modules / pk primitives."""
        pk = current_pk()
        # ... see "Composing in compile()" below ...
        return output
```

Write a unit test alongside it:

```python
# tests/runtime_python/models/<model>/test_my_mlp.py
def test_my_mlp_testmode():
    cfg = SimpleNamespace(hidden_size=4096, intermediate_size=11008)
    m   = MyMLP(cfg).to("cuda", torch.bfloat16)
    x   = torch.randn(8, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    r   = torch.randn(8, cfg.hidden_size, dtype=torch.bfloat16, device="cuda") * 0.01
    ref = m.forward(x, r)

    # Build a tiny test-mode PK, attach inputs, call m.compile(), run.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers,
                  max_num_batched_tokens=8, max_num_batched_requests=8)
    pk = PersistentKernel(**params)
    x_dt = pk.attach_input(x, name="x")
    r_dt = pk.attach_input(r, name="r")
    out  = torch.zeros_like(ref)
    with pk.compile_scope():
        m.compile(x_dt, residual_dt=r_dt, output=out)
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
```

Run it on a free GPU. **Don't proceed until it passes.** If the diff is large:
- Check tensor layouts (especially fused `[gate|up]` halved layout for silu_mul, GQA-interleaved fused QKV for paged attention).
- Check that the catalog modules' `forward()` and `compile()` themselves are consistent (catalog test at `tests/runtime_python/layers/test_<leaf>.py` should already PASS — if it doesn't, that's a catalog bug, not your composite's bug).

### Stage 2 — Next level up: attention block, decoder layer

After MLP passes, write `MyAttention`, then `MyDecoderLayer = RMSNorm → MyAttention → RMSNorm → MyMLP`. Same drill: both `forward()` and `compile()`, then a test that runs the whole decoder layer standalone and asserts numerical match. Use a tiny KV cache pool, 1-2 tokens, and `test_mode=True` so the runtime quits after one task graph iteration.

For the test, you'll need to set up the meta-tensors (`qo_indptr_buffer`, `paged_kv_indptr_buffer`, etc.) that the attention kernel reads. See `demo/qwen3/demo_new.py` for the minimal viable setup; copy the meta-tensor block into your test.

### Stage 3 — Reduced-layer model

Build `MyModel` and `MyForCausalLM`. Add a `--layers 0-3` flag to your `demo_new.py` that overrides `config.num_hidden_layers` before instantiation. Run with random or real weights. End-to-end smoke check: does `pk.compile()` finish? Does `pk()` finish without `cudaErrorIllegalAddress`? Are the output tokens finite?

### Stage 4 — Full model, full prompt

Only after the reduced-layer smoke passes do you run the full model. Compare the generated token stream against the legacy `demo.py --use-mirage` driver (if it exists) on the same prompt. Outputs should match token-by-token.

### Where to put tests

- `tests/runtime_python/layers/test_<leaf>.py` — catalog leaf tests (already exist; you reuse them).
- `tests/runtime_python/models/<model>/test_<composite>_testmode.py` — your composite tests.
- `tests/runtime_python/test_mode/` — multi-block pipelines that don't correspond to a single composite.
- The demo's `--skip-weight-load` + `--layers 0-3` flags are the stage-3 smoke.

---

## Composing in `compile()`

Two patterns appear in every catalog composite:

**Pattern A: chain catalog modules via their `compile()` methods.** Each returns a DTensor (or writes through `output=`). Pass it to the next.

```python
def compile(self, x_dt, *, residual_dt, output):
    pk = current_pk()
    mid    = self.gate_up.compile(x_dt)           # Linear (fused via shuffle_tensors)
    activ  = self.silu_mul.compile(mid)
    return self.down.compile(activ, residual=residual_dt, output=output)  # LinearWithResidual
```

**Pattern B: drop down to `pk.<primitive>` when the catalog doesn't cover the wiring.** Common case: `pk.shuffle_tensors(...)` to fuse two weight tensors at compile time before feeding them to a Linear. This is fine; the catalog is not exhaustive.

```python
def compile(self, x_dt, *, residual_dt, output):
    pk = current_pk()
    w_gate_dt = pk.attach_input(self.gate_proj_weight, name=f"{self.prefix}gate_proj_weight")
    w_up_dt   = pk.attach_input(self.up_proj_weight,   name=f"{self.prefix}up_proj_weight")
    num_tasks = _grid_for_linear(2 * self.intermediate_size)
    w_gateup_dt = pk.shuffle_tensors(
        inputs=[w_gate_dt, w_up_dt], shuffled_dim=0,
        num_groups=num_tasks // 2, name=f"{self.prefix}gateup_proj",
    )
    mlp_mid = pk.new_tensor(
        dims=(pk.max_num_batched_tokens, 2 * self.intermediate_size),
        dtype=_mi_bf16, name=f"{self.prefix}per_layer_mlp_mid",
    )
    pk.linear_layer(input=x_dt, weight=w_gateup_dt, output=mlp_mid,
                    grid_dim=(num_tasks, 1, 1), block_dim=(128, 1, 1))
    ...
```

**Naming convention**: `prefix` strings use `_` separator (no dots — dots are illegal in C++ identifiers used as MPK tensor names). Build prefixes by concatenation: `f"{prefix}self_attn_"` etc.

**Allocate per-layer intermediates** inside each `DecoderLayer.compile()` call. The old "shared across all 36 layers" pattern is no longer required (a kernel bug that forced it was fixed).

---

## `current_pk()` and `pk.compile_scope()`

The `current_pk()` helper resolves the active `PersistentKernel` via a `contextvars.ContextVar`. It MUST be called inside a `with pk.compile_scope():` block at the model root — typically the demo driver. Calling `current_pk()` outside the scope raises a clear `RuntimeError`.

```python
# in demo_new.py
pk = mi.PersistentKernel(...)
input_tokens_dt = pk.attach_input(input_tokens, name="input_token")
with pk.compile_scope():
    model.compile(input_tokens_dt, output_tokens=output_tokens)  # current_pk() resolves inside
pk.compile(output_dir=args.output_dir)
```

---

## Driver responsibilities

The `modeling.py` does NOT handle these — the driver does:

- Allocating meta-tensors (`qo_indptr_buffer`, `paged_kv_indptr_buffer`, `paged_kv_indices_buffer`, `paged_kv_last_page_len_buffer`, `step`, `tokens`, `input_tokens`, `output_tokens`, `num_new_tokens`, `prompt_lengths`).
- Allocating the KV cache pool. Shape depends on attention type:
  - Standard GQA: `(num_layers, max_num_pages, page_size, num_kv_heads, head_dim)` for both `k_cache` and `v_cache`.
  - MLA: single combined `(num_layers, max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim)` pool registered as both `k_cache` and `v_cache` (the modeling slices it).
- Constructing `PersistentKernel` with `meta_tensors=`, `kv_cache=`, `target_cc`, etc.
- Tokenizing the prompt, populating `tokens` / `prompt_lengths`.
- Loading HF weights, dequantizing FP8 if needed, any pre-shuffle/absorption that doesn't fit in `_load_from_state_dict`.
- Pre-padding `lm_head.weight` to the argmax-partial grid stride (256-multiple).
- Wrapping `model.compile()` in `with pk.compile_scope():`.
- Running `pk()` and reading `tokens` for the decoded output.

See `demo/qwen3/demo_new.py` for the full skeleton.

---

## File layout & reference implementations

```
python/mirage/mpk/models/<model>/modeling.py   # MPKModule tree
demo/<model>/demo_new.py                       # driver
```

Read these references — your `modeling.py` should follow the same conventions verbatim (sub-block `prefix` chaining, `_load_from_state_dict` HF-key mapping, per-layer intermediate allocation inside each `DecoderLayer.compile()`):

- **Dense GQA + paged attention**: `python/mirage/mpk/models/qwen3/modeling.py` + `demo/qwen3/demo_new.py`
- **MLA + MoE + reduced-layer flag**: `python/mirage/mpk/models/deepseek_v3/modeling.py` + `demo/deepseek_v3/demo_new.py`
