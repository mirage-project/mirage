# SGLang v0.5.21 (ROCm image) - Kimi K3 decoder layer map for the hook

Source read: the `sglang` package of lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261009.
Paths below are relative to `sglang/srt/` unless they start with `sglang/kernels/`.
Config numbers come from the K3 checkpoint's `config.json`. Scope: TP8, no DP attention, no SP-MoE, is_hip,
decode target-verify (8 tokens, 1 request). Anything not read directly in code is marked **(inferred)**.

K3 facts used (k3_config.json): 93 layers, hidden 7168, `attn_res_block_size`=12, `first_k_dense_replace`=1 (layer 0 = dense
MLP, layers 1..92 = MoE), KDA = layers whose `idx+1` is in `kda_layers` (`configs/kimi_linear.py:158-162`), so MLA layers are
3,7,11,...,87,91,92. Block-write layers (`idx % 12 == 0`, `models/kimi_k3.py:2488`) = 0,12,24,36,48,60,72,84, all of them KDA.
Bank width NB = cdiv(93,12) = 8 (`models/kimi_k3.py:2792`). rms_norm_eps 1e-5. num_experts 896, top-16, latent 3584.

---------------------------------------------------------------------------------------------------------------------------

## 1. `KimiK3DecoderLayer.forward` and the two stage boundaries

`models/kimi_k3.py:2636-2659`:

```python
hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch, capture_gathered=...)   # aggregation 1
hidden_states = self._run_self_attn(hidden_states, positions, forward_batch, zero_allocator)       # KDA / MLA, TP-partial o_proj
hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)                            # record partial as pending
hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)                            # TP all-reduce + aggregation 2
if self._ffn_writes_stream:                                                                         # every MoE layer of K3
    hidden_states = self.mlp(hidden_states, prefix_sum=residual_batch.written_residual(forward_batch))
else:                                                                                               # layer 0 (dense MLP)
    hidden_states = self.mlp(hidden_states)
return self.ffn_boundary.finish(hidden_states, forward_batch)
```

### 1.0 Objects and per-forward state

* `AttnBank` (`layers/layer_boundary/residual/attn_bank.py:30-54`): holder with one field `current`. One instance per model,
  created in `KimiK3LinearModel.__init__` (`models/kimi_k3.py:2705`) and passed to every layer's constructor
  (`models/kimi_k3.py:2417`, `2752`). The layer does NOT keep it as an attribute; it goes into `AttnBankState`
  (`models/kimi_k3.py:2532-2544`).
* Per forward, `KimiK3LinearModel.forward` calls `residual_batch.start(forward_batch)` (`models/kimi_k3.py:2771`, which sets
  `forward_batch.residual_stream = ResidualStream()`, `layers/layer_boundary/residual/batch.py:22-28`) and
  `self.attn_bank.open(hidden_states, NB, None)` (`models/kimi_k3.py:2791-2799`) -> `AttnResidual(...)`.
* `AttnResidual` (`layers/attn_residual.py:443-551`):
  * `block_residual`: `hidden_states.new_empty((T, NB, H))` (`layers/attn_residual.py:459-461`), bf16, **token-major**
    `[T=8, NB=8, H=7168]`, stride(0)=NB*H=57344, stride(1)=H=7168, allocated fresh every forward (inside the graph's memory
    pool when captured - (inferred) from it being a plain `new_empty` in the forward).
  * `num_valid_blocks` (nvb): Python int, number of valid bank rows, +1 on each write (`layers/attn_residual.py:480-482`).
    Because it is a Python int, its value per layer is baked into the captured graph (inferred; it is the same every step).
  * Rows in use: row j < nvb is bank row j; the "prefix" row (current partial sum) is scored as row nvb.
* `ResidualStream` (`layers/layer_boundary/residual/stream.py:80-232`): fields `residual` (the written stream / running
  prefix tensor) and `pending` (a `Contribution`: producer output not yet added). Lives in `forward_batch.residual_stream`.
* `residual_batch.written_residual(forward_batch)` = `forward_batch.residual_stream.residual`
  (`layers/layer_boundary/residual/batch.py:38-41`). Called right after `ffn_boundary.prepare`, it is the **new prefix u**
  (attention output added into the running prefix), see 1.3.

Stage declaration (`models/kimi_k3.py:2520-2584`): with `use_attn_residuals`, `AttnBankState(...).residual_ops()` gives
`attn_readout=_AttnReadout`, `attn_update=PLAIN_ADD`, `ffn_readout=_FfnReadout`, `ffn_update=PLAIN_ADD`
(`layers/layer_boundary/residual/attn_bank.py:182-188`); the layer replaces the FFN update by `REPLACE_AT_EXIT` when
`_ffn_writes_stream` (`models/kimi_k3.py:2568-2572`). Norms: `input_layernorm` goes with the attention stage,
`post_attention_layernorm` with the FFN stage (`models/kimi_k3.py:2573-2584`).

`_ffn_writes_stream` (`models/kimi_k3.py:2512-2517`) = use_attn_residuals and MoE layer and `mlp.use_latent_moe` and not
(DP attention without EP a2a) -> True for layers 1..92 (latent MoE: `latent_moe_use_norm`/`routed_expert_hidden_size` set in
the config; `use_latent_moe = config.routed_expert_hidden_size is not None`, `models/kimi_k3.py:441`).

Which path the boundary takes on ROCm TP8 (ORDINARY batch variant):
* `variant_for` (`layers/layer_boundary/construction.py:263-275`): no SP, `input_scattered` needs `_is_cuda or _is_npu`
  (`layers/layer_boundary/adapters/attention.py:93-101`), no CP -> `BatchVariant.ORDINARY`.
* The attention output contract: owes the ATTN_TP sum, `always_partial=True` (`layers/layer_boundary/factories.py:462-487`,
  `declare_attn` default `ALWAYS_PARTIAL` at `factories.py:334-377`).
* No generic add+norm fusions run on any K3 boundary: both bank reads have `is_plain_norm = False`
  (`residual/attn_bank.py:196`), and `_bind_entry_path` passes fusions/carried fusions only when the read is a plain norm
  (`layers/layer_boundary/boundary.py:281-287`, `331`).
* `all_reduce_fusion` is False on ROCm: `_fuses_attn_all_reduce` needs `k3_ar_fusion.enabled()`
  (`models/kimi_k3.py:2371-2380`), whose auto-probe returns None unless the device SM is 100/103
  (`layers/communication/k3_ar_fusion.py:58-74`); even with `SGLANG_K3_AR_FUSION=1` it needs a `CustomAllReduceV2` with
  multicast (`k3_ar_fusion.py:83-104`), which `dispatch_custom_allreduce` only builds on CUDA
  (`distributed/device_communicators/custom_all_reduce.py:358-371`). So `_ffn_input_fusions()` returns `()`
  (`models/kimi_k3.py:2586-2597`) and `_k3_all_reduce_add` (`models/kimi_k3.py:369-374`) never runs on MI355X.

### 1.1 `attn_boundary.prepare` (aggregation 1)

Call chain: `StageBoundary.prepare` (`layers/layer_boundary/stage.py:139-170`) -> `_prepare_attention` (`stage.py:337-409`)
-> `_prepare_input` -> `_prepare` (`stage.py:86-125`) -> `entry.prepare` = `_run_entry` (`layers/layer_boundary/prepare.py:281-340`)
-> step `_update_read` (`prepare.py:343-383`, selected at `boundary.py:589-603` for owes=None, no gather) -> the readout.

What arrives depends on the previous layer's FFN:
* After a MoE layer (REPLACE_AT_EXIT): the stream is "written": `residual = y_prev`, `pending=None`
  (`layers/layer_boundary/exit.py:130-134`). `stream.input` returns `(y_prev, None)` (`stream.py:180-186`), so
  `_update_read` calls `read.read(y_prev, input_layernorm)` (`prepare.py:374-376`) -> `_AttnReadout.read`
  (`attn_bank.py:233-239`) -> `read_attn_input(residual)` -> `_attn_aggregate(y_prev, None, norm)`
  (`attn_bank.py:117-134`) -> `AttnResidual.forward(y_prev, None, self_attention_res_proj, self_attention_res_norm,
  input_layernorm, rows=None, write=is_block_write_layer)` (`attn_residual.py:484-551`).
* After layer 0 (dense MLP, PLAIN_ADD): pending = TP-partial MLP output carried as `UnreducedOutput` (deferred sum,
  `exit.py:221-233`); `_run_entry` completes it with an all-reduce (`prepare.py:325-339`, no carried fusions because the read
  is not a plain norm), then `read.update_and_read(PLAIN_ADD, mlp_out, residual=u0, norm)` (`prepare.py:377-383`) ->
  `AttnResidual.forward(mlp_out, u0, ...)`; this folds the add: prefix = bf16(u0 + mlp_out) inside the hip kernel
  (`attn_residual.py:536-546`, `360-384`). (The "deferred" decision: `exit.py:184-252`, `337-360` - inferred to be taken for
  layer 0 on TP8 since `may_defer_to_next` holds for a dense FFN that is not complete, `factories.py:209-222`.)
* Layer 0: stream empty; `_update_read(enters_stack=True)` runs `init_residual` (identity, `attn_bank.py:202-203`) then
  `read.read(embedding)` (`prepare.py:365-376`).

Inside `AttnResidual.forward` (`attn_residual.py:484-551`) on ROCm:
* nvb == 0 (layer 0 only): `write(hidden)` copies the embedding into bank row 0, returns
  `(input_layernorm(hidden), hidden)` (`attn_residual.py:504-509`).
* nvb >= 1 and no pending add (every layer after a MoE layer): `_aggregate(prefix=y_prev, bank, nvb, ...,
  write_bank_row=fused_write)` (`attn_residual.py:522-534`) -> `_use_hip_fused(7168, nvb)` is True for nvb <= 8
  (`attn_residual.py:53-63`; register gate `next_pow2(nvb)*8192 <= 65536`, `sglang/kernels/ops/attention/attn_res_hip.py:23-34`)
  -> `_aggregate_hip(...)` (`attn_residual.py:283-315`) -> one Triton launch `_agg_kernel[(T,)]`
  (`sglang/kernels/ops/attention/attn_res_hip.py:37-133`, launcher `136-212`).
* Block-write layer: `fused_write=True` on the hip path (`attn_residual.py:518-521`): the kernel stores the prefix row into
  `bank[:, nvb, :]` before scoring (`attn_res_hip.py:102-103`), then `_count_write` makes nvb+1 (`attn_residual.py:547-548`).
  The aggregation itself still mixes rows 0..nvb-1 plus the prefix (so the written row is mixed once, as the prefix).
* Return `(normed, prefix)`. `_attn_aggregate` returns `(normed, None if writes_block else prefix)` (`attn_bank.py:125-134`).
  `_prepare` then does `stream.write(residual)` (`stage.py:124`): **after aggregation 1 on a block-write layer the running
  prefix is None (the block restarts); on other layers it is the same tensor object y_prev** (no copy).

Equation of aggregation 1 (hip kernel, per token t; fp32 inside, `attn_res_hip.py:90-133`), with
`cw = fp32(self_attention_res_norm.weight) * fp32(self_attention_res_proj.weight)` cached per module
(`attn_residual.py:66-83`):
```
p      = prefix row (bf16; = bf16(residual + addend) when there is a pending add, written to prefix_out)
s_j    = dot(b_j, cw) / sqrt(mean(b_j^2) + eps_score)     j = 0..nvb-1  (b_j = bank[t, j, :] as fp32)
s_p    = dot(p,   cw) / sqrt(mean(p^2)   + eps_score)
m      = max(max_j s_j, s_p);  w_j = exp(s_j - m), w_p = exp(s_p - m);  inv = 1/(sum_j w_j + w_p)
acc    = p * (w_p*inv) + sum_j (w_j*inv) * b_j                        (fp32, NOT rounded before the norm)
x_attn = bf16( acc / sqrt(mean(acc^2) + eps_in) * fp32(input_layernorm.weight) )
```
eps_score = self_attention_res_norm eps, eps_in = input_layernorm eps (both 1e-5).
Note: the eager reference `aggregate_stream_torch` (`attn_residual.py:318-336`) and the non-HIP 2-kernel path round the mixture
to bf16 before `out_norm` (`attn_residual.py:203-206`, `280`); the HIP kernel does not.

### 1.2 Attention + `attn_boundary.finish`

* `_run_self_attn` (`models/kimi_k3.py:2599-2634`): for MLA (`prepare_qkv_latent` exists) it sets `AttentionInputs` in the
  attention-TP context, calls `self.self_attn(...)`, clears it.
* KDA returns `self.o_proj(core_attn_out)[0]` (`models/kimi_k3.py:2119-2123`; the symmetric-buffer branch 2119-2122 is only
  for `all_reduce_fusion`). o_proj is a `RowParallelLinear` with `reduce_results=False` (`models/kimi_k3.py:1743`, MLA
  `2161`); stage boundaries refuse any producer with `reduce_results=True` (`layers/layer_boundary/stage.py:412-432`). So the
  attention output is the **TP-partial** o_proj output [8, 7168] bf16 on each rank.
* `attn_boundary.finish` (`stage.py:172-188`): attention plans have `finishes_directly=True`
  (`layers/layer_boundary/construction.py:352-353`) -> `stream.record(partial, PLAIN_ADD, declared_sum=SumGroup.ATTN_TP)`
  (`stream.py:132-158`) -> `pending = Contribution(partial, PLAIN_ADD, DeclaredSum(ATTN_TP))`, returns an `OwedOutput`
  handle. No compute.

### 1.3 `ffn_boundary.prepare` (o_proj all-reduce + aggregation 2)

Chain: `_prepare_input` -> `_prepare` (`stage.py:86-125`): pending present -> `call["update"]=PLAIN_ADD`; `stream.input`
returns `(partial, residual)` (`stream.py:180-184`); `_run_entry` sees a DeclaredSum of the expected group and runs the step
(`prepare.py:309-340`); the step is `_reduce_update_read(gathers_residual=False, fusions=(), read_fusions=(), group=ATTN_TP,
read=_FfnReadout)` (`boundary.py:604-629`):
1. `read_fusions` empty (would hold `_k3_all_reduce_add` only with `all_reduce_fusion`), `fusions` empty
   (`prepare.py:117-137`).
2. **The o_proj TP all-reduce is here**: `attn_tp_all_reduce(partial, forward_batch, may_quantize=True)`
   (`prepare.py:137-141`) -> `attention_tensor_model_parallel_all_reduce` (`layers/layer_boundary/ops.py:182-191`; the
   quantized variant is used only for non-decode modes when `--enable-quant-communications` is set, and TARGET_VERIFY is not
   `is_decode_or_idle`, `model_executor/forward_batch_info.py:254-255` - off by default (inferred)).
   On HIP the group all-reduce picks `ca` (custom AR; aiter `AiterCustomAllreduce` or sglang's 1-stage per
   `custom_all_reduce.py:371-380`), then quick-reduce `qr`, else NCCL (`distributed/parallel_state.py:955-995`, `1021-1066`).
3. `read.update_and_read(PLAIN_ADD, a, residual, post_attention_layernorm)` (`prepare.py:148-155`) -> `_FfnReadout.update_and_read`
   (`attn_bank.py:269-271`) -> `update_and_read_ffn_input(a, residual, norm)` (`attn_bank.py:136-144`) ->
   `AttnResidual.forward(a, residual, mlp_res_proj, mlp_res_norm, post_attention_layernorm, rows=None, write=False)`.
   * non-write layer (residual = y_prev): `_aggregate_fused_add(prefix_a=y_prev, prefix_b=a, ...)`
     (`attn_residual.py:536-546`, `360-384`) -> hip kernel with `HAS_ADD`: `u = bf16(fp32(y_prev) + fp32(a))` written to a
     new tensor `prefix_out` (`attn_res_hip.py:92-101`, `attn_residual.py:300`), then the same score/softmax/mix/norm.
   * block-write layer (residual = None after aggregation 1): `_aggregate(a, ...)` with prefix = `a` itself
     (`attn_residual.py:522-534`): **u = a** (the all-reduced attention output alone; the old prefix lives in the bank row
     just written).
   * The FFN read never writes the bank (`write=False`, `attn_bank.py:143`).
4. `_prepare` writes `stream.write(u)` (`stage.py:124`) -> stream state "written": `residual=u`, `pending=None` (the
   consumed contribution's tensors are released, `stream.py:123-130`).

Returned from `ffn_boundary.prepare`: `h = x_ffn` [8, 7168] bf16 (MLP input). `written_residual(forward_batch)` = `u`.

### 1.4 MLP call and `ffn_boundary.finish`

* MoE layers: `self.mlp(h, prefix_sum=u)` -> returns `y = bf16(bf16(up(latent) + shared) + u)` (section 4); this is the next
  stream.
* `ffn_boundary.finish(y)` (`stage.py:172-188`) -> `exit(forward_batch).finish(y)` -> `FfnExit.finish` (`exit.py:539-549`).
  The FFN output contract is "complete" for REPLACE_AT_EXIT (`factories.py:207`), so `_decide` takes `complete_now` with
  `owes_sum=False` (`exit.py:184-205`); `_complete_now` (`exit.py:254-292`): no sum, output move `keep_output`
  (`ops.py:334-340`; inferred from produced layout == residual layout), then `ReplaceAtExit.update(y, u)` returns `y`
  (`residual/add_norm.py:283-300`) and drops `u`; `_record_output` sets `stream.residual=None` then `stream.write(y)`
  (`exit.py:130-134`). Forward returns `y` (same object as `stream.residual`, which the next prepare checks,
  `stream.py:160-169`).
* Layer 0 (dense `KimiK3MLP`, PLAIN_ADD): output TP-partial (`down_proj` `reduce_results=False`, `models/kimi_k3.py:316`);
  the exit defers the TP sum into layer 1's attention entry as `UnreducedOutput` (see 1.1) (inferred: `_defers_sum` true on
  TP8 without EP/LoRA, `exit.py:337-395`).
* After the last layer: `residual_batch.final_norm(...)` with `AttnBankOutputRead` (`models/kimi_k3.py:2721-2741`,
  `2865-2877`; `attn_bank.py:274-322`) = one more bank aggregation with `output_attn_res_proj/norm` and `model.norm`.

### 1.5 Summary per layer type (TP8, ROCm, verify)

| step | KDA non-write layer | KDA block-write layer (0,12,..,84) | MLA layer (never a write layer in K3) |
|---|---|---|---|
| attn prepare in | stream residual y_prev | y_prev (layer 0: embedding) | y_prev |
| bank | read rows 0..nvb-1 | write row nvb := y_prev, nvb += 1 | read |
| x_attn | agg1(bank, y_prev) | agg1(bank old rows, y_prev) | agg1 |
| stream after attn prepare | residual = y_prev | residual = None | residual = y_prev |
| attention out | TP-partial o_proj | same | TP-partial gated o_proj |
| ffn prepare | AR(a); u=bf16(y_prev+a); h=agg2 | AR(a); u=a; h=agg2 (nvb now +1) | as non-write |
| MoE | y = bf16(bf16(moe(h)) + u) | same | same |

---------------------------------------------------------------------------------------------------------------------------

## 2. Aggregation 2 (FFN readout) - exact equations

Weights (per layer, replicated, bf16 unless loaded otherwise): `mlp_res_proj` = `ReplicatedLinear(7168 -> 1, bias=False,
quant_config=None)` (`models/kimi_k3.py:2501-2507`), `mlp_res_norm` = `RMSNorm(7168, eps)` (`models/kimi_k3.py:2493`),
`post_attention_layernorm` = `RMSNorm(7168, eps)` (`models/kimi_k3.py:2480-2482`). Score vector
`cw2 = fp32(mlp_res_norm.weight) * fp32(mlp_res_proj.weight.squeeze())` (`attn_residual.py:81`, fp32 for the triton/hip path,
`attn_residual.py:299`). Kernel: `sglang/kernels/ops/attention/attn_res_hip.py:86-133`, one program per token, num_warps=4.

Inputs: `a` = all-reduced attention output (bf16), `r` = stream residual after aggregation 1 (`y_prev`, or None on a write
layer), bank `B[t, 0..nvb-1, :]` (nvb already includes the row written by this layer's aggregation 1).

```
u      = bf16( fp32(r) + fp32(a) )          if r is not None   (HAS_ADD; rounded to bf16 BEFORE scoring, attn_res_hip.py:93-101)
u      = a                                  on a block-write layer
s_j    = dot(B_j, cw2) * rsqrt(mean(B_j^2) + eps_mlp_res_norm)     j = 0..nvb-1      (fp32)
s_u    = dot(u,   cw2) * rsqrt(mean(u^2)   + eps_mlp_res_norm)
m      = max(s_0..s_{nvb-1}, s_u);  w_j = exp(s_j - m);  w_u = exp(s_u - m);  inv = 1 / (sum w_j + w_u)
acc    = u*(w_u*inv) + sum_j (w_j*inv)*B_j                                             (fp32, not rounded)
h      = bf16( acc * rsqrt(mean(acc^2) + eps_post) * fp32(post_attention_layernorm.weight) )
new prefix (stream residual, `written_residual`) = u
```
Softmax is over the nvb bank rows plus the current partial u (u is the last row, index nvb), `attn_res_hip.py:111-127`.
Note the score uses `RMSNorm(v)` without a bf16 round of the normalized vector and folds the norm weight into cw (fp32); the
eager reference instead runs `score_norm` (bf16 out) then `score_proj` (`attn_residual.py:332-333`) - not bit-identical
(inferred from the code shapes; not measured).

---------------------------------------------------------------------------------------------------------------------------

## 3. Where to hook

Replace, inside `KimiK3DecoderLayer.forward` (`models/kimi_k3.py:2636-2659`), the three calls
`_run_self_attn` (2645-2647), `attn_boundary.finish` (2648) and `ffn_boundary.prepare` (2649) by one call of our kernel,
keeping `attn_boundary.prepare` (2642-2644, aggregation 1 incl. the bank write) and the MoE/`ffn_boundary.finish` after it.
This is the v0.5.21 counterpart of the v0.5.19 hook already used by mpk-kernels
(`box_mirror/sglang_0519_src/sglang/srt/models/kimi_k3.py:2483-2540`, calling
`mirage_kda/mpk-kernels/mpk_kernels/integrations/sglang_k3_kda.py:192-230`), which sat right after aggregation 1.

What the kernel needs at the hook (all available right after `attn_boundary.prepare`):
* `x_attn` = return value of `attn_boundary.prepare` [8, 7168] bf16.
* `prefix` = `residual_batch.stream_of(forward_batch).residual` (`residual/batch.py:31-35`): y_prev, or **None on a block-write
  layer** (pass zeros, as the v0.5.19 integration did with `S.zeros`, `sglang_k3_kda.py:219`; bf16(0 + a) = a, so the result
  equals SGLang's `u = a`).
* bank = `AttnBank.require()` -> `.block_residual` [8, 8, 7168] (strides 57344, 7168) and `.num_valid_blocks`
  (already incremented on a write layer). The layer has no attribute for the bank: either store `attn_bank` in `__init__`
  (constructor arg at `models/kimi_k3.py:2417`) or reach it via `self.ffn_boundary.declaration.read.state.bank`
  (`_FfnReadout.state`, `attn_bank.py:199-200`; `StageDeclaration.read`, `factories.py:285`) (inferred path, both
  objects are plain attributes).
* aggregation-2 weights: `self.mlp_res_proj.weight`, `self.mlp_res_norm.weight`, `self.post_attention_layernorm.weight`
  (`cw2` may be reused from `get_cw(self.mlp_res_proj, self.mlp_res_norm)`, `attn_residual.py:66-83`).

State that must be left exactly as SGLang leaves it after `ffn_boundary.prepare`:
1. `forward_batch.residual_stream`: state "written" with `residual = u` (our new prefix) and `pending = None`. After
   `attn_boundary.prepare` the stream is already `pending=None` (`stage.py:124`), so one call
   `stream_of(forward_batch).write(u)` (`stream.py:123-130`) reproduces it. Do NOT call `set_written` (it replaces the
   stream object, `residual/batch.py:167-169`; harmless here but not what SGLang does).
2. Bank: nothing. Aggregation 2 never writes the bank (`attn_bank.py:143`); the block-write row and `num_valid_blocks += 1`
   were done by `attn_boundary.prepare`, which stays SGLang's. Our kernel must read nvb rows (the count after that prepare).
3. Attention caches: the KDA conv/recurrent state, ReplaySSM rings and intermediate conv windows (section 5.1), or the MLA KV
   rows of the 8 tokens (section 5.2), must be written as SGLang's verify path writes them, because the post-verify commit
   (accept length) reads them.
4. The MLA attention-TP context (`get_attn_tp_context().set_attn_inputs`, `models/kimi_k3.py:2614-2632`) is only consumed
   inside SGLang's MLA forward; skipping `_run_self_attn` needs no cleanup (it is cleared at 2631-2632 anyway).

Then:
```python
h = self.attn_boundary.prepare(hidden_states, forward_batch, capture_gathered=capture_gathered)
if <ours applies>:                      # forward_batch.forward_mode.is_target_verify() and batch_size == 1 and h.shape[0] == 8
    stream = residual_batch.stream_of(forward_batch)
    bank = <attn_bank>.require()
    h, u = our_attn_layer(h, bank.block_residual, bank.num_valid_blocks, stream.residual or zeros, ...)
    stream.write(u)                     # == state after ffn_boundary.prepare
else:
    ... SGLang's 3 calls ...
y = self.mlp(h, prefix_sum=residual_batch.written_residual(forward_batch))   # or our MoE kernel: y = bf16(bf16(moe(h)) + u)
return self.ffn_boundary.finish(y, forward_batch)                             # writes y as the next stream
```
If our MoE kernel replaces `self.mlp`, `ffn_boundary.finish(y)` must still run (it sets `stream.residual = y` and does no
compute for REPLACE_AT_EXIT layers, 1.4). For layer 0 (dense MLP) keep `self.mlp(h)` + finish (TP-partial, deferred sum).

How the layer knows it is verify: `forward_batch.forward_mode.is_target_verify()` (`model_executor/forward_batch_info.py:257-258`);
note `is_extend()` is also True for TARGET_VERIFY (`forward_batch_info.py:224-230`) and `is_decode()` is False. The decoder
layer itself never branches on the mode; KDA does (`models/kimi_k3.py:2070-2094`). `forward_batch.batch_size` is the request
count (used the same way by the v0.5.19 integration, `sglang_k3_kda.py:200-201`).

---------------------------------------------------------------------------------------------------------------------------

## 4. `KimiK3MoE.forward` / `_forward_fused`

* `forward(hidden_states, *, prefix_sum=None)` (`models/kimi_k3.py:1458-1474`): takes `_forward_fused` when rows > 0 and
  `_eligible_for_fused_front` (`models/kimi_k3.py:741-756`: latent MoE, shared experts present, `_front_w` built, not an EP
  pair, no a2a backend, shared down weight bf16/fp16), else `_forward_unfused` (`1066-1242`).
* Inputs: `hidden_states` = `h` (output of aggregation 2, [8, 7168] bf16); `prefix_sum` = `u` from
  `written_residual(forward_batch)` (`models/kimi_k3.py:2653-2656`).
* `_forward_fused` (`models/kimi_k3.py:1324-1456`) on ROCm (`k3_ar_fusion` off, `alt_stream=None` on HIP,
  `models/kimi_k3.py:2701`):
  1. `fused = _k3_bf16_gemm(h, _front_w, out_dtype=None)` (bf16 front, `_front_fp32` is False on HIP, `760-769`), split into
     `gate_up, router_logits, routed_input` (`1346-1352`); router logits made contiguous on HIP unless `SGLANG_AITER_K3_OPT`
     (`1353-1354`); `routed_input` made a dense bf16 copy when `_moe_front_needs_dense_bf16` (`1355-1358`, `1244-1263`).
  2. One buffer `buf = [latent (8 x 3584) | shared (8 x 7168)]` (`1359-1373`, views at `1374-1375`).
  3. `_forward_shared` (shared down GEMM into `shared_output`, `1306-1317`), `_forward_routed` (top-k + experts into
     `latent`, `1281-1291`), then `buf = tensor_model_parallel_all_reduce(buf)` - one TP all-reduce of both partial sums
     (`1433-1446`, `998-1000`).
  4. `latent = _latent_norm(latent)` = `routed_expert_norm(latent)` (`1450-1451`, `858-861`), `out = routed_expert_up_proj(latent)` (`1452`).
  5. `return _add3(out, shared_output, prefix_sum, prefetch_bc=True)` (`1456`) = `bf16(bf16(out + shared) + prefix_sum)`
     (`models/kimi_k3.py:338-355`; the add3 JIT kernel keeps the same double rounding).
* Output semantics: **the output already includes prefix_sum (u); it is the next residual stream y** (docstring
  `models/kimi_k3.py:1463-1467`; REPLACE_AT_EXIT writes it as the stream, 1.4). The unfused path ends the same way
  (`_add3` at `1238`, or `out + prefix_sum` at `1240` without shared experts).
* So the MoE kernel contract matches (c): `y = bf16(bf16(moe(x)) + prefix)` with `bf16(moe(x)) = bf16(up(norm(AR(latent))) +
  AR(shared))`, where `up(...)` is itself a bf16 GEMM output.

---------------------------------------------------------------------------------------------------------------------------

Sections 5 and 6: paths relative to the `sglang/` package root (`srt/...` or `kernels/...`); `K` = `srt/models/kimi_k3.py`,
`DS` = `srt/models/deepseek_v2.py`, `MX` = `srt/layers/quantization/mxfp4.py`, `TK` = `srt/layers/moe/topk.py`,
`AR` = `srt/layers/moe/moe_runner/aiter.py`. Server flags the code cannot settle (which env vars / `--attention-backend` the
box server runs with) are called out where they change the path.

## 5. Attention state during target verify

### 5.1 KDA (`KimiK3DeltaAttention`, K:1477; 12 heads/rank x 128, conv kernel 4, `gate_lower_bound` -5.0)

Op order in `forward` (K:2063-2123) for TARGET_VERIFY on HIP:
* `defer_f_b` needs `is_decode()` (K:2070-2072) -> False in verify. `fused_onorm` False on HIP (K:2088-2094; the HIP
  fused-decode stash `_prepare_fused_decode` K:1889-1945 needs `SGLANG_ROCM_K3_KDA_FUSED_BACKEND=aiter` and serves decode only).
* In-proj (`forward_qkvbfg_fused`, K:1988): `do_fuse_qkvbfg = quant_config is None and ...` (K:1537) is False for the
  compressed-tensors checkpoint, so the ROCm one-GEMM `[q,k,v,g|f_a|b]` branch (K:1997-2013, needs `_qkvgbfa_sizes`) is off,
  HIP has no alt stream, and the plain path runs (K:2038-2046): `fused_qkvg_proj(x)` -> [8, 6144] split qkv 4608 | g 1536;
  `bfa = kimi_k3_tiny_gemm(x, _bfa_w)` (`[f_a 128 | b 12 | pad 4]`); `forget_gate = tiny_gemm(bfa[:, :128], _bfa_f_b_w)`
  -> [8, 1536]; `beta = bfa[:, 128:140]` (raw b, sigmoid later). `tiny_gemm` = `tiny_gemm_bf16` for small m else `F.linear`
  (`kernels/ops/gemm/kimi_k3.py:13-34`).
* Verify shapes: `a = forget_gate.unflatten(-1,(-1,128)).unsqueeze(0)` -> [1,8,12,128]; `b = beta.unsqueeze(0)` -> [1,8,12]
  (K:2082-2085).
* `self.attn(...)` (K:2102) -> `RadixLinearAttention.forward` (`srt/layers/radix_linear_attention.py:80`) ->
  `HybridLinearAttnBackend.forward` -> `forward_extend` for non-decode modes
  (`srt/layers/attention/hybrid_linear_attn_backend.py:1457-1502`) -> `KDAAttnBackend.forward_extend`
  (`srt/layers/attention/linear/kda_backend.py:1012`) -> `_forward_target_verify` (`kda_backend.py:1023-1024`, body 1186).
* `o_norm` = `FusedRMSNormGated(128, eps=1e-5, activation="sigmoid")` (K:1735-1737):
  `y = x * rsqrt(mean_128(x^2) + eps) * w * sigmoid(g)` per head, g = the g slice of the in-proj
  (`kernels/ops/attention/fla/fused_norm_gate.py:84-123`); run in the model (K:2114-2116). Then `[8,1536] -> o_proj`
  (TP-partial, K:2117-2123).

Verify backend selection:
* `--linear-attn-verify-backend` generic default `triton` (`srt/layers/attention/linear/utils.py:96-99`), but the K3 hook pins
  `nv_cutedsl` whenever a speculative algorithm is set, no platform check (`srt/arg_groups/kimi_k3_hook.py:25-33`); the
  dispatcher maps `nv_cutedsl` to `TritonKDAKernel` (`kda_backend.py:172-173`); the CuTe path needs CUDA SM10x
  (`kda_backend.py:1702-1707`).
* Fused chain verify `fused_kda_conv_gating_verify` needs `SGLANG_OPT_FUSED_KDA_VERIFY=1` (default off, `srt/environ.py:1415`),
  verify backend == `triton` (`kda_backend.py:510-513`), gfx95 + bs<=16 + T in {6,8} + bf16 + fp32 conv weights
  (`kda_backend.py:1502-1511`), no ReplaySSM (`1497-1500`). With the pin it is off unless `--linear-attn-verify-backend triton`.
* Default HIP verify chain (`kda_backend.py:1310-1407`): `causal_conv1d_update` (Triton,
  `kernels/ops/mamba/causal_conv1d_triton.py:1021`, kernel at 607) -> split q,k,v [1,8,12,128] ->
  `TritonKDAKernel.target_verify` (`srt/layers/attention/linear/kernels/kda_triton.py:159`) ->
  `fused_sigmoid_gating_delta_rule_update(is_kda=True, disable_state_update=True, use_qk_l2norm_in_kernel=True,
  lower_bound=-5.0)` (`kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py`).

State pools (`HybridReqToTokenPool` -> `MambaPool`, `srt/mem_cache/memory_pool.py:494`, built at
`srt/mem_cache/kv_cache_configurator.py:1494`; shapes from `KimiLinearStateShape.create`, `srt/configs/mamba_utils.py:323-363`,
called from `srt/configs/kimi_linear.py:172-182`). L = 69 KDA layers (dense index via `mamba_map`, `memory_pool.py:1441, 1651`;
per-layer view `req_to_token_pool.mamba2_layer_cache(layer_id)`):

| tensor | shape | dtype | where |
|---|---|---|---|
| `conv[0]` | [69, size+1, 3, 4608] (K-1=3 steps x 3*1536 channels; transposed vs GDN) | bf16 (`SGLANG_MAMBA_CONV_DTYPE`, `mamba_utils.py:68`) | `memory_pool.py:709` |
| `temporal` (recurrent state) | [69, size+1, 12, 128(V), 128(K)], K contiguous (`fused_sigmoid_gating_recurrent.py:262-273`) | fp32 default (`mamba_utils.py:71`) | `memory_pool.py:736` |
| `intermediate_ssm` | [69, spec+1, 8, 12, 128, 128] | ssm dtype | `memory_pool.py:870` |
| `intermediate_conv_window[0]` | [69, spec+1, 8, 3, 4608] (dense layout, KDA disables window dedup, `memory_pool.py:127-142`) | bf16 | `memory_pool.py:938` |

`spec = max_num_reqs` (`kv_cache_configurator.py:1502`).

Indices: slot = `forward_metadata.mamba_cache_indices = req_index_to_mamba_index_mapping[req_pool_indices]`
(`memory_pool.py:1607`, `hybrid_linear_attn_backend.py:187-191`); scratch row = `intermediate_state_indices`, positional
(batch row i -> row i; `srt/layers/attention/linear/utils.py:125-157`, `kda_backend.py:519-524`); the v0.5.19 integration read
it as `verify_intermediate_state_indices` on the KDA backend (`mirage_kda/mpk-kernels/mpk_kernels/integrations/sglang_k3_kda.py:155`);
the same attribute exists in v0.5.21 (`KDAAttnBackend.verify_intermediate_state_indices`, `kda_backend.py:524-525`, per-batch
selection `select_verify_intermediate_state_indices` at `kda_backend.py:1245`). `query_start_loc = arange(0, T+1, 8)`
(`hybrid_linear_attn_backend.py:285-292`; graph replay 883-900).

ReplaySSM rings (off by default, `enable_linear_replayssm_spec=False`, `srt/arg_groups/fields/exec_.py:491`; ring length
`linear_replayssm_cache_len` default 16, power of 2 and >= 2x drafts, `exec_.py:484`, `memory_pool.py:809-815`):
`replayssm_rawv` [69, size+1, 12, Lr, 128] bf16 (post-conv v), `replayssm_rawk` same shape bf16 (post-conv k before L2 norm),
`replayssm_g` [69, size+1, 12, Lr, 128] fp32 (per-K log decay), `replayssm_beta` [69, size+1, 12, Lr] fp32 (sigmoid(b))
(`memory_pool.py:798, 821-835`); `intermediate_ssm` is then None (`868-869`). Ring writes at ring[slot][head][step]
(`fused_sigmoid_gating_recurrent.py:371-418`); on HIP the unfused Triton kernel writes them (`kda_backend.py:1237-1245`,
fused+ring path needs SM90/SM100, `1495-1500`). The v0.5.19 mpk integration required this flag
(`sglang_k3_kda.py:160-162`).

What target verify writes (no ReplaySSM):
* conv kernel (`causal_conv1d_triton.py:607`, `num_accept_tokens=None`): reads the old 3-step window (697-712),
  **overwrites `conv[slot]` in place with the last 3 pre-conv inputs** (759-785), writes the window after each step into
  `intermediate_conv_window[row][step]` (980-991); output `silu(bias + sum w*x)` fp32 -> bf16 (997).
* recurrence kernel per token (`fused_sigmoid_gating_recurrent.py`): `x = a + dt_bias` (331); `g = -5*sigmoid(exp(A_log)*x)`
  per K channel (334); `beta = sigmoid(b)` (361); `q,k *= rsqrt(sum x^2 + 1e-6)` (436-439); `q *= 128^-0.5` (441, 619);
  `h *= exp(g)[k]` (455); `v -= sum_k h*k` (460); `v *= beta` (463); `h += k (x) v` (479); `o = sum_k h*q` (482); store `h` to
  `intermediate_ssm[row][step]` (486-497). **`temporal` is not written** (`DISABLE_STATE_UPDATE`, 517). BV=32, num_warps=1,
  grid (4, N, 12) (22-66, 635-636).
* Commit after acceptance: `commit_mamba_states_after_verify` (`srt/speculative/spec_utils.py:933`) ->
  `update_mamba_state_after_mtp_verify` (`spec_utils.py:1168` -> `hybrid_linear_attn_backend.py:1505`) ->
  `scatter_mamba_states_after_mtp_verify` (`kernels/ops/mamba/mamba_state_scatter_triton.py:695`):
  `temporal[slot] = intermediate_ssm[row, accept_len-1]`, `conv[slot] = intermediate_conv_window[row, accept_len-1]`
  (`spec_utils.py:871-930`). With ReplaySSM: `commit_kda_replayssm_after_verify`
  (`kernels/ops/attention/fla/kda_replayssm_spec_decode.py:336`, via `spec_utils.py:1089-1144`) folds the accepted ring
  prefix into `temporal`, then scatters the conv window. The fused-accept branch (`hybrid_linear_attn_backend.py:1563-1588`)
  is CUDA-only (`kda_backend.py:544-570`).

=> Our KDA kernel must, for the 8 tokens: read `conv[slot]` + `temporal[slot]`; write `intermediate_conv_window[row][0..7]`,
`intermediate_ssm[row][0..7]` (or the 4 rings if ReplaySSM is on); overwrite `conv[slot]` with the last 3 pre-conv inputs
(what SGLang's conv kernel does in verify); leave `temporal[slot]` untouched. (Whether the commit later re-reads `conv[slot]`
or only the window: the scatter overwrites it from the window, so the in-place write is not read back - inferred.)

### 5.2 MLA (`KimiK3MLAAttention`, K:2125, subclass of `DeepseekV2AttentionMLA`, DS:1977)

Backend on ROCm: `get_default_attn_backend` returns `aiter` on HIP only if kv heads per rank is 16 or 128, else `triton`
(`srt/arg_groups/model_override_base.py:327-334`); K3 has `num_key_value_heads`=96 -> 12/rank
(`srt/configs/model_config.py:1591-1602`) -> **triton by default**; aiter only with `--attention-backend aiter`.
With aiter: `speculative_attention_mode` default "prefill" (`srt/arg_groups/fields/spec.py:129-136`) so verify uses the
prefill backend string (DS:2254-2281); `handle_attention_aiter` -> `MLA` for verify
(`srt/models/deepseek_common/attention_backend_handler.py:190-203`) -> `MLA_ROCM` (32-43) ->
`forward_absorb_rocm_prepare` / `core` (`srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:494, 807`).
K3's `dispatch_attn_forward_method` only changes the GGUF split case (K:2301-2305).

Absorbed forward:
* `fused_qkv_a_proj_with_mqa(x)` -> `[q_lora 1536 | c_kv 512 | k_pe 64]` (DS:2033-2040, 2474-2499); `q_a_layernorm`
  RMSNorm(1536), `kv_a_layernorm` RMSNorm(512), eps 1e-5 (DS:2041, 2127); fused `fused_qk_rmsnorm_bf16` with aiter
  (`forward_mla_rocm.py:573-581`).
* `q_b_proj` -> [8, 12, 192] -> `q_nope` [8,12,128], `q_pe` [8,12,64]; `k_pe = latent[..., 512:]` [8,1,64]
  (DS:2501-2513, `forward_mla.py:186-193`).
* **No RoPE**: K3 passes `skip_rope=True` (K:2162-2163) -> `rotary_emb=None` (DS:2129-2145); q_pe/k_pe unrotated, positions
  unused (`forward_mla_rocm.py:1104-1120`). Softmax scale `192^-0.5` (DS:2023), no mscale.
* Absorb: `q_nope_out = bmm(q_nope, w_kc)` -> [8,12,512] (`rocm_absorb_q_bmm`, `forward_mla_rocm.py:197-247`); after
  attention `bmm(o, w_vc)` -> [8, 12*128] (`rocm_absorb_v_bmm`, 250+).
* Core (aiter path, `forward_mla_rocm.py:989-1010`): `q = cat(q_nope_out, q_pe)` [8,12,576]; `k = cat(c_kv_normed, k_pe)`
  [8,1,576]; `v = c_kv`; `attn_mqa(q, k, v, save_kv_cache=True)` -> `full_attn_backend.forward_extend`
  (`hybrid_linear_attn_backend.py:1427-1455`).
* Output gate (config `mla_use_output_gate=true`, K:2143): `g_proj` ColumnParallel 7168 -> 1536/rank (K:2238-2249).
  `forward` stores the attention input `hidden_states` (K:2331-2345); `_gated_o_proj_forward` wraps `o_proj.forward`
  (K:2266-2287): `gate = g_proj(hidden_states)` (K:2321-2329), then **`x = x * sigmoid(gate)` on the [8,1536] attention output
  before `o_proj`**, via `mla_output_gate.kimi_k3_mla_output_gate` = `bf16(x * bf16(sigmoid(g)))` when covered
  (`kernels/ops/attention/mla_output_gate.py:34-53`; HIP build of that JIT kernel not verified - inferred), else
  `x * torch.sigmoid(gate)`. `o_proj` RowParallel `reduce_results=False` (K:2161).

KV cache:
* `HybridLinearKVPool` wraps `MLATokenToKVPool` (24 layers, layer-id map) (`srt/mem_cache/memory_pool.py:4204-4222`).
  `kv_buffer[l]` = `(size + page_size, 1, 576)` (`memory_pool.py:4655-4685`), row 0 = padding slot; bf16 with
  `kv_cache_dtype="auto"` (`srt/arg_groups/fields/model.py:189-213`). `page_size` default 1 on HIP
  (`srt/arg_groups/overrides.py:1280-1303`, 64 only with `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`); the mamba no_buffer
  strategy requires 1 (`srt/arg_groups/mamba_hook.py:157`).
* Write: aiter `forward_extend` (MLA, no DCP) calls `token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)`
  (`srt/layers/attention/aiter_backend.py:3756-3762`) -> `kv_buffer[l][out_cache_loc] = k` as 576-wide rows
  (`memory_pool.py:4754-4790`).
* `out_cache_loc` = the 8 slots `req_to_token[req, seq_len : seq_len+8]` (`srt/speculative/eagle_utils.py:571-581`) or the
  prepared locations (`srt/speculative/eagle_worker_common.py:255-320`); `req_to_token` int32 [reqs, ctx]
  (`memory_pool.py:397-398`).
* aiter verify metadata (`aiter_backend.py:2285-2422`, graph replay 3111-3160): `kv_lens = seq_lens + 8` (seq_lens excludes the
  drafts), `qo_indptr = arange(0, (bs+1)*8, 8)`, `kv_indptr = cumsum(kv_lens)`, `kv_indices` = token slots from
  `req_to_token`, `kv_last_page_len` all ones (2425-2429), `max_q_len = 8`; persistent-kernel metadata when
  `SGLANG_AITER_MLA_PERSIST` (default on, line 98; 601-626), 32 KV splits (633-636).
* aiter kernel (`aiter_backend.py:3983-4035`): bf16 KV -> `_mla_decode_fwd_with_head_pad` (1439-1500): q zero-padded 12 -> 16
  heads, `aiter.mla.mla_decode_fwd(q[8,16,576], K_Buffer.view(-1,1,1,576), o, qo_indptr, kv_indptr, kv_indices,
  kv_last_page_lens, max_seqlen_q=8, sm_scale, ..., num_kv_splits=32)`, sliced back to 12 heads. Causal mask among the 8
  queries inside aiter (inferred). fp8 KV: same call with q cast to fp8 (1419-1424, 1458-1478).
* After acceptance: no MLA commit step found; accepted rows stay addressed through `req_to_token` (inferred; the spec-worker
  free path was not traced).

=> Our MLA kernel must write the 8 KV rows (576 bf16 each: normed c_kv 512 | k_pe 64, unrotated) at `out_cache_loc`, attend over
`req_to_token[req, 0 : seq_len+8]` with causal masking within the 8, and apply the gate before `o_proj`.

---------------------------------------------------------------------------------------------------------------------------

## 6. Weight access (ROCm, compressed-tensors MXFP4 checkpoint)

Checkpoint (read from `/mnt/shared/models/Kimi-K3` headers): `quant_method=compressed-tensors`, `mxfp4-pack-quantized`, group
32; `ignore` = self_attn, shared_experts, dense `mlp.*_proj`, lm_head, vision -> all BF16; routed experts U8 MXFP4 + U8 E8M0
scales; router gate, `routed_expert_down/up_proj`, `routed_expert_norm` BF16; `e_score_correction_bias` F32; conv weights,
`A_log`, `dt_bias`, KDA `o_norm` F32 in the file. **No FP8 / `weight_scale_inv` tensors**, so `_get_k3_dense_weight`
(K:191-201) returns the bf16 weight unchanged and every HIP FP8 fold is skipped. Ignored linears ->
`UnquantizedLinearMethod` (`srt/layers/quantization/compressed_tensors/compressed_tensors.py:189-195`), whose
`process_weights_after_loading` does nothing off CPU-AMX (`srt/layers/quantization/unquant.py:490-492`); with
`SGLANG_USE_AITER=1` unquantized GEMMs run through aiter `tgemm.mm` / triton a16w16 (`unquant.py:514-517`).

### 6.1 KDA (per rank, TP8)

| attribute | shape | dtype | source |
|---|---|---|---|
| `fused_qkvg_proj.weight` | [6144, 7168], rows q 1536 \| k 1536 \| v 1536 \| g 1536 | bf16 | K:1545-1557, loader K:3070-3075, 3187-3204 |
| `split_sizes` | [4608, 1536] | - | K:1558-1561 |
| `b_proj.weight` | [12, 7168] | bf16 | K:1562-1571 |
| `f_a_proj.weight` (replicated) | [128, 7168] | bf16 | K:1572-1578 |
| `f_b_proj.weight` | [1536, 128] | bf16 | K:1579-1586 |
| `_bfa_w` (merged after load) | [144, 7168] = f_a 128 \| b 12 \| 4 zero rows; f_a/b weights become views | bf16 | `_merge_bfa_weights` K:1790-1829, K:254-273 |
| `_bfa_f_b_w` | = `f_b_proj.weight` [1536, 128] | bf16 | K:1827 |
| `qkv_conv1d.weight` | [4608, 1, 4]; `attn.conv_weights` = [4608, 4] | fp32 | K:1700-1708, 1756 |
| `A_log` | [1, 1, 12, 1] (loader narrows the checkpoint [128] at rank*12) | fp32 | K:1713-1733 |
| `dt_bias` | [1536] | fp32 | K:1692-1698 |
| `o_norm.weight` | [128] | default dtype bf16 (inferred; file value F32) | K:1735-1737 |
| `o_proj.weight` | [7168, 1536], `reduce_results=False` | bf16 | K:1738-1754 |

`post_load_weights` (K:3260): per KDA layer `_merge_bfa_weights()` then `_prepare_fused_decode()` (K:3350-3352); `_warm_cw`
precomputes the attention-residual `cw` vectors for `self_attention_res_*`, `mlp_res_*`, `output_attn_res_*` (K:3324-3335).
No transposes or dequant of KDA weights on HIP.

### 6.2 MLA (per rank)

| attribute | shape | dtype | source |
|---|---|---|---|
| `fused_qkv_a_proj_with_mqa.weight` (replicated) | [2112, 7168] = q_a 1536 \| kv_a 512+64 | bf16 | DS:2033-2039, loader K:3150-3171 |
| `q_a_layernorm.weight` / `kv_a_layernorm.weight` | [1536] / [512] | bf16 | DS:2041, 2127 |
| `q_b_proj.weight` | [2304, 1536] (12 x 192) | bf16 | DS:2042-2049 |
| `kv_b_proj.weight` | [3072, 512] (12 x (128+128)) | bf16 | DS:2109-2116 |
| `w_kc` | logical [12, 128, 512], strides (65536, 1, 128) | bf16 | `post_load_weights` K:3298-3301 |
| `w_vc` | logical [12, 512, 128], strides (65536, 1, 512) | bf16 | K:3302 |
| `g_proj.weight` | [1536, 7168] | bf16 | K:2238-2249 |
| `o_proj.weight` | [7168, 1536], `reduce_results=False` | bf16 | DS:2118-2125, K:2232-2235 |

`w_kc/w_vc`: `kv_b.unflatten(0, (-1, 256))` -> [12, 256, 512], split [128 | 128] on dim 1 (K:3298-3300); `w_scale` stays 1.0
on HIP (K:3303-3317).

### 6.3 MoE (per rank; 896 experts, top-16, latent 3584, intermediate 3072 -> 384/rank, 2 shared experts -> 768/rank)

* Router `gate` = `MoEGate(quant_config=None)`, weight [896, 7168] bf16 (K:455, DS:503-512); `e_score_correction_bias` fp32
  (DS:513-529, forced again K:3346-3349). `topk`: grouped, `num_expert_group=1`, `topk_group=1`, sigmoid scoring, correction
  bias, renormalize, `routed_scaling_factor` 1.0 (K:495-523). With `_use_aiter` the bias is cast to the logits dtype (bf16)
  (TK:2616-2624) and `aiter_biased_grouped_topk` runs (TK:1919-1952; `moe_route_radix4` only with `SGLANG_K3_RADIX4_TOPK`).
* `shared_experts` = `KimiK3MLP` on the TP group: `gate_up_proj` [1536, 7168] (gate 0:768, up 768:1536), `down_proj`
  [7168, 768] `reduce_results=False`, act `SituAndMul(beta=4.0, linear_beta=25.0)` (K:582-617, 303-326).
* Latent projections (replicated, bf16; `latent_quant_config` None for compressed-tensors, K:631-639):
  `routed_expert_down_proj` [3584, 7168] (K:640-646), `routed_expert_norm` RMSNorm(3584) eps 1e-5 (K:647-651),
  `routed_expert_up_proj` [7168, 3584] (K:652-658).
* Merged front `_front_w` (`_merge_front_weights`, K:688-736, called from `post_load_weights` K:3343-3345):
  `[shared gate 768 | shared up 768 | router 896 | latent down 3584]` = **[6016, 7168] bf16**, no padding; the module weights
  become views (K:704-709, 728); `_front_sizes = [1536, 896, 3584]`.
* Routed experts: `Mxfp4Config(is_checkpoint_mxfp4_serialized=True)` -> `Mxfp4MoEMethod` (K:466-470, MX:373-375), `FusedMoE`
  with `reduce_results=False` (`srt/layers/moe/fused_moe_triton/layer.py:338, 1587`), `gate_up_interleaved=False` (K:485).
  Intermediate padded to 128 with `SGLANG_AITER_K3_OPT`, else 256 (-> 384 or 512/rank); hidden 3584 unpadded
  (MX:444-633, aiter branch 517-528). Tensors: `w13_weight` [896, 2I, 1792] u8 (gate rows then up rows),
  `w13_weight_scale` [896, 2I, 112] u8 E8M0, `w2_weight` [896, 3584, I/2] u8, `w2_weight_scale` [896, 3584, I/32] u8, no bias.
  `process_weights_after_loading` (`_use_aiter`, MX:944-1040): for SiTU with `AITER_SITUV2_A8W4` set or `AITER_SITUV2_A4W4`
  unset (MX:150-154): `shuffle_weight_a16w4(w13, 16, True)` + `shuffle_scale_a16w4`, `shuffle_weight_a16w4(w2, 16, False)`;
  else `shuffle_weight`/`shuffle_scale`; then `is_shuffled = True` on both (MX:1029-1030). Call: `_apply_aiter` (MX:1738-1802)
  -> `aiter.fused_moe.fused_moe` with `PER_1X32`, `Situv2`, beta 4.0 / linear_beta 25.0 (AR:108, 302-309, 375-394); the
  gfx950 small-M kernel does not apply (`kernels/ops/moe/smallm_moe_gfx950/__init__.py:32, 296-332`).
  Box measurement (`amd_runs/nx30_sgl.full`, MoE-layer harness, not the server): `w13 (896, 768, 1792) uint8 shuffled True`
  (I=384, so `SGLANG_AITER_K3_OPT` on) and aiter picked `flydsl_moe1_afp8_wfp4_...` + `opus_moe2_afp8_wfp4_atomic...`
  (FP8 activations inside aiter, A8W4).
* Path for verify T=8: `_forward_fused` (eligibility K:741-756 holds; harness log "fused front True, front fp32 False");
  `_forward_mega_experts` (K:771) unused (`_use_mega_moe` False with no a2a, K:531). Front GEMM via `_k3_bf16_gemm` =
  `F.linear` (K:204-242). Routed experts write into the `latent` slice via the zero-copy context (copied if not)
  (K:1281-1291); FusedMoE trims to `[..., :3584]` (`fused_moe_triton/layer.py:1580-1585`). The only MoE collective is the
  one all-reduce of `[latent_partial | shared_partial]` (K:1446, 998-1000).

Full MoE formula (bf16 at each arrow):
`y = bf16( bf16( up_proj( RMSNorm_3584( AR(L) ) ) + AR(S) ) + u )`, with
`L = aiter fused_moe(routed_input = bf16(h @ W_down^T), top-16 renormalized sigmoid+bias routing)`,
`S = bf16( SiTU(h @ W_gu_shard^T) @ W_d_shard^T )`, AR = bf16 sum over 8 ranks, `u` = `written_residual`.

Unsettled by code: which of `SGLANG_USE_AITER`, `SGLANG_AITER_K3_OPT`, `AITER_SITUV2_*`, `SGLANG_ROCM_K3_KDA_FUSED_BACKEND`,
`--attention-backend`, `--linear-attn-verify-backend`, `--enable-linear-replayssm-spec` the box server runs with. Read the
server launch line before matching numerics.
