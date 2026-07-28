# Phase 0 — compute graph (draw.io) → model plan

Turns the user's compute-graph file into the op inventory + classification + milestone
plan that the other phases execute. Output artifact: the model plan doc —
`.claude/skills/v2-model-support/references/V2_<MODEL>_MASTER_PLAN.md` if it should travel
with the repo, or `scratch/` (git-ignored) for throwaway drafts. Mirror
`references/V2_DSV3_DECODE_MASTER_PLAN.md` (the proven, archived instance).

## 1. The draw.io convention

> STATUS NOTE: no model compute-graph `.drawio` file was found in the tree or home dirs
> when this skill was authored (only docs illustration `.drawio.svg` exports under
> `docs/source/images/`). The convention below is therefore DEFINED (derived from what
> the builder graph needs), not reverse-engineered from a found file. If the user hands
> you a drawio that deviates, extract the same fields and note the mapping in the plan.

A drawio file is XML: `<mxfile><diagram>...<mxGraphModel><root><mxCell .../></root>`.
Newer saves may compress the `<diagram>` payload (base64 + raw-deflate) — if the inner
text is not XML, inflate it first:

```python
import base64, zlib, xml.etree.ElementTree as ET
from urllib.parse import unquote
tree = ET.parse(path); dia = tree.find(".//diagram")
inner = dia.text or ""
if "<mxGraphModel" not in inner:  # compressed payload
    inner = unquote(zlib.decompress(base64.b64decode(inner), -15).decode())
model = ET.fromstring(inner)
nodes = {c.get("id"): c for c in model.iter("mxCell") if c.get("vertex") == "1"}
edges = [c for c in model.iter("mxCell") if c.get("edge") == "1"]  # source/target ids
```

**Node (op) label convention** — one op per vertex, label lines (HTML `<br>` or `\n`):
```
line 1: <op_name>                      e.g. l{i}.attn.q_b_proj
line 2: kind=<op_kind>                 e.g. kind=linear_fp8 | rmsnorm | rope | attn_mla
                                            | router_topk | group_gemm | silu_mul | AR
line 3: in=<shapes> out=<shapes>       GLOBAL (unsharded) shapes, dtype suffix
                                       e.g. in=[1,1536]bf16 out=[1,9216]bf16
line 4: w=<hf_key> shard=<rule>        shard= col|row|rep|ep(dim=k)  (omit if no weight)
line 5: (optional) notes: fused-candidate / collective=AR / scratch=...
```
**Edge convention**: edge label = tensor name (+shape if not inferable). Edge direction
= dataflow. An edge into an op with no label = the activation stream (`self.x`-class).
Container/group cells (a `<mxCell style="group">` around a layer) mark layer boundaries;
if absent, infer layers from the `l{i}.` name prefix.

Strip HTML from `value` attributes (`re.sub(r"<[^>]+>", "\n", v)`) before parsing lines.

## 2. The op inventory table (the plan's core)

One row per DISTINCT op template (not per layer — layers repeat). Columns:

```
name | kind | GLOBAL shapes | PER-RANK shapes @TP/EP | dtype(+scale layout) |
weight key(s) + conversion | collective | v2 status | port kind | grid intuition
```

`v2 status` ∈ {PRESENT (enum + `tasks/blackwell_v2/*.cuh` exist), ABSENT, PRESENT-but-
builder-not-routed}. Check BOTH `runtime_header.h` (`grep '_V2 ='`; DSv3-era range
242/243 + 326–355) and the wrapper in `persistent_kernel.py` (`"..._v2" if
self.use_v2_runtime`) — a registered kernel the builder never selects is still ABSENT
from the model's point of view (the DSv3 FFN-mega was exactly this: kernel proven,
builder re-route was the actual work).

`port kind` ∈:
- **leaf** — role-split trivial (memset/elementwise/reduce-per-row). Consumer-only body
  with the dep-prefix; hours of work.
- **collective** — contains cross-rank NVSHMEM sync (AR, global argmax). HARD: the v1
  body is 256-thread + `__syncthreads()`; a v2 consumer role has 128 threads → restride
  + v2-safe barrier (named `bar.sync`/tag-flag) rewrite, validated on a TP2 micrograph.
- **megakernel-shape (Form-2)** — one task per worker (`num_tasks == num_workers`,
  hard-asserted), co-resident, self-syncing via an in-op GMEM count-barrier (monotonic:
  `need = num_tasks*(iter+1)`, never reset → scratch zeroed at step 0 ONLY). Needs the
  `task_offset = bid.x` metadata line in `runtime.cc` (union with `merge_task_offset` —
  v2 megas read offset 0, v1 megas offset 4).

## 3. Per-rank shape derivation (the sharding algebra)

Given TP world W, EP degree E, routed_tp = W/E, rank r, ep_rank = r // routed_tp:

| rule | weight shape effect | activation effect | collective after |
|---|---|---|---|
| rep (None) | unchanged | unchanged | none |
| col (dim=0 of [N,K]) | N → N/W | out width → N/W (local heads/columns) | none (stay sharded) |
| row (dim=1 of [N,K]) | K → K/W | out = FULL-width PARTIAL per rank | **AllReduce(+residual once)** |
| ep + col/row on inner dim | E_total → E_total/E local slice; inner dim / routed_tp | routed act sharded within EP group | AR over routed group |
| vocab-parallel lm_head | vocab → padded local shard | partial argmax per rank | cross-rank argmax |

Two derived facts to record per row: (a) does the op END in a partial that a downstream
collective must combine (row-parallel), and (b) is any residual/bias added — it must be
added EXACTLY ONCE, post-AllReduce (`Σ_r(partial_r + res) = Σpartial + W·res` is the
classic TP bug; bind a ZERO residual into the kernel and fold the real one in the AR).

## 4. Worked example — DSv3 decode at TP8 EP2 (the numbers to crib)

Global: hidden=7168, 64 q-heads, q_lora=1536, kv_lora=512, rope=64, vocab=129280,
256 routed experts (top-8) + 1 shared, routed inter=2048, shared inter=2048, dense
inter=18432, 61 layers (0-2 dense, 3-60 MoE). Per-rank at TP8 EP2 (16 local heads,
routed_tp=4, 128 local experts, routed inter/4 = 512, dense inter/8 = 2304):

| op | per-rank weight (fp8 payload + f32 scale) | notes |
|---|---|---|
| qkv_a_proj | (2176,7168) + scale (17,56) | replicated; 2176 = 1536 q_a + 512 c_latent + 64 k_pe + 64 zero-pad (128-row MMA tile) |
| q_a/kv_a/input LN | bf16 1536 / 512 / 7168 | concat trick: one (9216,) buffer to stay under MAX_INPUTS_PER_TASK=14 |
| q_b_proj (absorbed) | (9216,1536) + (72,12) | 9216 = 16 heads × 576([nope512|pe64]); col-sharded |
| kv_b_v_bmm_dense (W_UV) | (16,128,512) + (16,1,4) | per-head BMM repack, head-dim sharded |
| o_proj_original | (7168,2048) + (56,16) | ROW-parallel (2048 = 16×128) ⇒ partial + AR+residual |
| router gate | bf16 (256,7168) + e_score bias | replicated |
| experts.w13 | [128, 1024, 7168] u8 | EP slice + inter col-shard (1024 = 2×512) |
| experts.w2 | [128, 7168, 512] u8 | EP slice + inter row-shard ⇒ AR after combine |
| shared gate_up / down | (512,7168) / (7168,256) | TP8-sharded shared expert |
| dense gate_up / down | (4608,7168) / (7168,2304) | layers 0-2; 4608 = 2×2304 |
| lm_head | replicated (129280→pad,7168) bf16 or vocab-parallel shard | flag ↔ cache key |

FP8 **scale layouts** (three distinct — do not conflate):
- dense per-128-block f32 `weight_scale_inv` `[N/128, K/128]`, kernel reads
  `Wsc[(n>>7)*nk + g]` plain fp32;
- MoE pow2/UE8M0 requantized scales, v2 packs all four into ONE f32 buffer in
  `MEGA_SC_` order `w13|wgu|w2|wdn` sized `E*8*56 + 4*56 + E*56*4 + 56*2` — offsets
  MUST mirror `dsv3_ffn_v2_spec.h` (crib `builder.py:~3056-3131` and the reference
  packing in `tests/runtime_python/blackwell_v2/dsv3_ffn_harness.py`);
- BMM per-head scales: packed UE8M0 `(H,512,1)` vs dense f32 `(H,1,4)` variants.

MLA **absorbed vs unabsorbed** weight forms (three coexist; conversion emits all):
- decode (absorbed): fused `q_b_proj [H*576, q_lora]`, fused KV cache row = 576
  (`[nope_512_abs | pe_64]`), W_UV as the `kv_b_v_bmm*` repack;
- absorbed prefill: `q_b_nope [H*512,q_lora]` / `q_b_pe [H*64,q_lora]`;
- chunked unabsorbed prefill: per-head `q_nope_128`/`q_pe_64` + `kv_b_k.weight`/
  `kv_b_v.weight` decompression. Decode-on-v2 needs only the FIRST form, but the
  conversion + cache must keep emitting whatever the v1/prefill paths consume.

## 5. Classification decision (per op row)

```
has v2 enum + kernel + wrapper switch?  ──yes──▶ REUSE (verify builder actually routes it;
   │                                             check contracts: v2 linear family M<=16)
   no
   │  is it on the FIRST-e2e slice's critical path?
   │      no ──▶ defer (M4/M5 item) or route around (e.g. --disable-vocab-parallel-lm-head
   │             kept nvshmem_global_argmax OFF the DSv3 M3 path — still absent today)
   │      yes
   ▼
NEW KERNEL → port kind {leaf | collective | mega} → a Phase-b spec row:
   roles / SMEM regions (+alignment=1024 for extern pools) / sync (tag-flag, no
   op-private mbarriers) / correctness reference (the v1 kernel on identical bytes)
   / validate step (harness op branch, TP2 micrograph for collectives, in-MPK smoke
   for megas). FUSED-BLOCK candidates are marked but NOT built until the chain is green.
```

## 6. Milestones + risk section of the plan doc

End the plan with: (a) the v2-ABSENT set sized (DSv3 was: 5 new tasks + 1 re-route for
the MoE steady state, +1 Form-2 task for dense layers); (b) the M0→M5 ladder with an
exit criterion per milestone; (c) a ranked where-it-breaks list — for DSv3 that was:
1 collective sync-mismatch, 2 `task_offset` metadata omission, 3 missing consumer body
(§1.1), 4 SMEM budget/`__align__(1024)`, 5 EP-locality of reused megas (les/nle slice),
6 iteration-lockstep. Rank yours the same way; the order has predicted the actual
failure sites. Cross-check the plan with Codex + `ablation-logic-reviewer` (the DSv3
plan's 3 load-bearing conclusions were reviewer+Codex-vetted before any build agent ran).
