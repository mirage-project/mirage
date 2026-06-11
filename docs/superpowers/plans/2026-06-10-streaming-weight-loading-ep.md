# Streaming Weight Loading + EP-aware DSV3 Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MPKModule.load_weights` stream weights one-at-a-time (bounded host RAM, loud on missing keys), and migrate the catalog DeepSeek V3 loader onto that streaming path with EP-aware expert loading.

**Architecture:** Replace the `list(weights)` + group-then-recurse body in `MPKModule.load_weights` with a streaming loop driven by a stateless `resolve_weight` hook (models override the hook, not the loop). DSV3 — which needs stateful fp8 weight↔scale pairing and per-layer expert stacking — overrides `load_weights` with its own streaming loop reusing shared base helpers, plus a `process_weights` for cross-parameter MLA absorption. EP filtering happens in the expert `weight_loader` (non-local mmap views are never touched ⇒ no disk read / no GPU residency).

**Tech Stack:** Python, PyTorch, safetensors (mmap-backed `get_tensor`), MPK catalog (`MPKModule`). Tests run on a GPU node via `test-on-gpu`; pure-CPU unit tests can run there too.

**Spec:** `docs/superpowers/specs/2026-06-10-streaming-weight-loading-ep-design.md`

**Phasing:** A = streaming core + Qwen3 (testable via Qwen3-8B). B = EP-aware MoE leaves (unit-testable). C = DSV3 migration (testable via `demo_new.py`). Each phase leaves the tree green.

---

## Shared contract (defined in Phase A, used everywhere)

```python
# python/mirage/mpk/layers/_base.py
SKIP_WEIGHT = object()   # sentinel: key is deliberately not loaded (e.g. non-local expert)

# resolve_weight(name, params) -> one of:
#   None                              -> key not recognized by this model (UNEXPECTED, fatal)
#   SKIP_WEIGHT                       -> recognized but intentionally skipped (counts as consumed)
#   (param, loader_or_None, kwargs)   -> apply: (loader or param.data.copy_)(param, tensor, **kwargs)
# where `params` is the dict(self.named_parameters()) built once per load.
```

---

## Phase A — Streaming core + Qwen3

### Task A1: Streaming `MPKModule.load_weights` + `resolve_weight` hook + missing-key assertion

**Files:**
- Modify: `python/mirage/mpk/layers/_base.py:92-172` (replace `load_weights` body; add `resolve_weight`, `_assert_fully_loaded`)
- Test: `tests/runtime_python/test_mode/test_streaming_load_weights.py` (create)

- [ ] **Step 1: Write the failing test** (pure CPU; no CUDA needed)

```python
# tests/runtime_python/test_mode/test_streaming_load_weights.py
import weakref
import pytest
import torch
import torch.nn as nn
from mirage.mpk.layers._base import MPKModule


class _Leaf(MPKModule):
    def __init__(self, prefix=""):
        super().__init__(prefix=prefix)
        self.weight = nn.Parameter(torch.empty(4, 4))


class _Model(MPKModule):
    def __init__(self, prefix=""):
        super().__init__(prefix=prefix)
        self.a = _Leaf()
        self.b = _Leaf()


def _tracking_iter(n, alive):
    """Yield (name, tensor); record peak number of yielded tensors kept alive."""
    refs = []
    for i in range(n):
        t = torch.ones(4, 4)
        refs.append(weakref.ref(t))
        yield f"{'a' if i % 2 == 0 else 'b'}.weight", t
        alive.append(sum(1 for r in refs if r() is not None))


def test_streaming_releases_each_tensor():
    m = _Model()
    alive = []
    consumed = m.load_weights(_tracking_iter(2, alive))
    assert consumed == {"a.weight", "b.weight"}
    # Process-and-release: never more than 1 source tensor alive at once.
    assert max(alive) <= 1, f"held {max(alive)} tensors live (expected streaming)"


def test_missing_key_raises():
    m = _Model()
    with pytest.raises(ValueError, match="a.weight"):
        m.load_weights(iter([("b.weight", torch.ones(4, 4))]))  # 'a' never provided


def test_unexpected_key_raises():
    m = _Model()
    with pytest.raises(ValueError, match="zzz.weight"):
        m.load_weights(iter([
            ("a.weight", torch.ones(4, 4)),
            ("b.weight", torch.ones(4, 4)),
            ("zzz.weight", torch.ones(4, 4)),
        ]))
```

- [ ] **Step 2: Run test to verify it fails**

Write to `run.sh`, then `test-on-gpu`:
```bash
echo 'MIRAGE_SKIP_GPU_CHECK=1 python -m pytest tests/runtime_python/test_mode/test_streaming_load_weights.py -v' > .scratch/run.sh
MIRAGE_SRC=$(pwd) MIRAGE_SKIP_GPU_CHECK=1 test-on-gpu gpu2 .scratch/run.sh
```
Expected: FAIL — current `load_weights` holds all tensors (`max(alive)` == 2) and does not raise on missing/unexpected keys.

- [ ] **Step 3: Replace the `load_weights` body and add hooks**

Replace `python/mirage/mpk/layers/_base.py:127-172` (the body from `weights_list = list(weights)` through `return consumed`) with:

```python
        """  # keep the existing docstring above this line unchanged
        params = dict(self.named_parameters())   # full-path -> Parameter
        consumed: Set[str] = set()
        loaded_paths: Set[str] = set()

        for name, tensor in weights:           # STREAMING: one tensor at a time
            res = self.resolve_weight(name, params)
            if res is None:
                raise ValueError(
                    f"{type(self).__name__}.load_weights: unexpected checkpoint "
                    f"key {name!r} (no matching parameter)."
                )
            if res is SKIP_WEIGHT:
                consumed.add(name)
                continue
            param, loader, kwargs = res
            if loader is not None:
                loader(param, tensor, **kwargs)
            else:
                param.data.copy_(tensor)
            consumed.add(name)
            loaded_paths.add(kwargs.get("_path", name))
            del tensor                         # drop ref so mmap pages are reclaimable

        self._assert_fully_loaded(params, loaded_paths)
        return consumed

    def resolve_weight(self, name, params):
        """Map an HF key to (param, loader_or_None, kwargs), None, or SKIP_WEIGHT.

        Default: 1:1 by deepest ``named_modules`` prefix → that leaf's parameter,
        using the parameter's ``weight_loader`` callback if present.
        Override on a model to remap names / fuse / route experts; fall back to
        ``super().resolve_weight`` for the default path.
        """
        routing = [(self, "")]
        for mod_name, mod in self.named_modules():
            if isinstance(mod, MPKModule) and mod is not self and mod_name:
                routing.append((mod, mod_name + "."))
        routing.sort(key=lambda mp: -len(mp[1]))
        for mod, prefix in routing:
            if prefix and not name.startswith(prefix):
                continue
            local = name[len(prefix):]
            if local in mod._parameters and mod._parameters[local] is not None:
                param = mod._parameters[local]
                loader = getattr(param, "weight_loader", None)
                return (param, loader, {"_path": name})
        return None

    def _assert_fully_loaded(self, params, loaded_paths):
        """Raise if any parameter expected by this module was never loaded."""
        optional = getattr(self, "_optional_param_paths", frozenset())
        missing = (set(params) - loaded_paths) - set(optional)
        if missing:
            raise ValueError(
                f"{type(self).__name__}.load_weights: parameters never loaded: "
                f"{sorted(missing)}"
            )
```

Add at module top (near other module-level names in `_base.py`):
```python
SKIP_WEIGHT = object()
```

Note: leaf `weight_loader` callbacks (e.g. `ColumnParallelLinear._weight_loader`) take `(param, loaded_weight)` and will receive the extra `_path` kwarg. To avoid touching every leaf, strip private kwargs before calling:

Change the apply line above from `loader(param, tensor, **kwargs)` to:
```python
                call_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
                loader(param, tensor, **call_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run the same command as Step 2. Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/mirage/mpk/layers/_base.py tests/runtime_python/test_mode/test_streaming_load_weights.py
git commit -m "feat: streaming MPKModule.load_weights with resolve_weight hook + missing-key assert"
```

---

### Task A2: Qwen3 q_norm/k_norm via `resolve_weight` override

**Files:**
- Modify: `python/mirage/mpk/models/qwen3/modeling.py:279-305` (delete the `load_weights` override; add `resolve_weight`)
- Test: extend `tests/runtime_python/test_mode/test_streaming_load_weights.py`

- [ ] **Step 1: Write the failing test**

Append to the test file:
```python
def test_qwen3_qnorm_knorm_remap():
    from mirage.mpk.models.qwen3.modeling import Qwen3Attention
    # Build a minimal attention via the real config path is heavy; instead
    # assert the resolver remaps the two special keys to the attn leaves.
    # (Construct requires current_pk(); this test runs inside a built PK in CI.)
    pytest.importorskip("mirage")
    # Smoke: the mapping table contains the two remaps.
    assert Qwen3Attention._QNORM_REMAP == {
        "q_norm.weight": "attn.q_norm",
        "k_norm.weight": "attn.k_norm",
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
echo 'MIRAGE_SKIP_GPU_CHECK=1 python -m pytest tests/runtime_python/test_mode/test_streaming_load_weights.py::test_qwen3_qnorm_knorm_remap -v' > .scratch/run.sh
MIRAGE_SRC=$(pwd) MIRAGE_SKIP_GPU_CHECK=1 test-on-gpu gpu2 .scratch/run.sh
```
Expected: FAIL — `_QNORM_REMAP` does not exist.

- [ ] **Step 3: Replace the override**

Delete `Qwen3Attention.load_weights` (`modeling.py:279-305`). Add in its place:
```python
    _QNORM_REMAP = {
        "q_norm.weight": "attn.q_norm",
        "k_norm.weight": "attn.k_norm",
    }

    def resolve_weight(self, name, params):
        # HF Qwen3RMSNorm q_norm/k_norm (.weight) -> PagedAttention raw params.
        remapped = self._QNORM_REMAP.get(name)
        if remapped is not None:
            return super().resolve_weight(remapped, params)
        return super().resolve_weight(name, params)
```

- [ ] **Step 4: Run test to verify it passes**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/mirage/mpk/models/qwen3/modeling.py tests/runtime_python/test_mode/test_streaming_load_weights.py
git commit -m "refactor: Qwen3 q_norm/k_norm remap via resolve_weight hook"
```

---

### Task A3: Qwen3-8B end-to-end regression (CLAUDE.md rule 2)

**Files:** none (verification only)

- [ ] **Step 1: Run Qwen3-8B megakernel**

```bash
cat > .scratch/run.sh <<'EOF'
python demo/qwen3/demo.py --output-dir ./output/output_1 --use-mirage \
  --model /mnt/shared/models/Qwen3-8B --max-num-batched-requests 1 > output/run.log 2>&1
EOF
MIRAGE_SRC=$(pwd) test-on-gpu gpu1 .scratch/run.sh
```

- [ ] **Step 2: Verify output + perf**

Read `output/run.log`. Expected: coherent generated text AND ~4.3 ms/token on 1×B200. If output is garbage → the missing-key assert or routing regressed; debug before proceeding.

- [ ] **Step 3: Commit (marker only, no code)** — skip; nothing to commit.

---

## Phase B — EP-aware MoE expert leaves

### Task B1: Local-sized expert params + EP-aware `weight_loader` on `MoEW13`/`MoEW2`

**Files:**
- Modify: `python/mirage/mpk/layers/moe/w13.py` (`_MoEW13Base.__init__`, `MoEW13BF16.__init__`)
- Modify: `python/mirage/mpk/layers/moe/w2.py` (mirror)
- Test: `tests/runtime_python/test_mode/test_moe_ep_loader.py` (create)

- [ ] **Step 1: Write the failing test** (pure CPU)

```python
# tests/runtime_python/test_mode/test_moe_ep_loader.py
import torch
from mirage.mpk.layers.moe.w13 import MoEW13BF16


def _make(num_experts, ep_size, ep_rank, hidden=8, inter=4):
    w = MoEW13BF16(
        num_experts=num_experts, num_experts_per_tok=2,
        hidden_size=hidden, intermediate_size=inter,
        ep_size=ep_size, ep_rank=ep_rank,
    )
    return w


def test_local_expert_param_shape():
    w = _make(num_experts=8, ep_size=2, ep_rank=1)
    # 8 experts / ep_size 2 = 4 local experts on this rank.
    assert w.weight.shape[0] == 4
    assert w.num_local_experts == 4
    assert w.local_expert_start == 4   # rank 1 owns experts 4..7


def test_loader_writes_local_skips_remote():
    w = _make(num_experts=8, ep_size=2, ep_rank=0)  # owns experts 0..3
    inter, hidden = 4, 8
    # global expert 5 is NON-local on rank 0 -> loader returns False, no write.
    src = torch.full((inter, hidden), 9.0, dtype=torch.bfloat16)  # a gate slab
    ok = w.weight_loader(w.weight, src, expert_id=5, slot="gate")
    assert ok is False
    # global expert 2 IS local -> writes into local slot 2, gate half.
    ok = w.weight_loader(w.weight, src, expert_id=2, slot="gate")
    assert ok is True
    assert torch.equal(w.weight[2, :inter], src)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
echo 'MIRAGE_SKIP_GPU_CHECK=1 python -m pytest tests/runtime_python/test_mode/test_moe_ep_loader.py -v' > .scratch/run.sh
MIRAGE_SRC=$(pwd) MIRAGE_SKIP_GPU_CHECK=1 test-on-gpu gpu2 .scratch/run.sh
```
Expected: FAIL — `MoEW13BF16` has no `ep_size`/`ep_rank` params, no `weight_loader`, full-size weight.

- [ ] **Step 3: Implement EP sizing + loader**

In `w13.py`, change `_MoEW13Base.__init__` (`:78-91`) signature to accept `ep_size=1, ep_rank=0` and compute local range:
```python
    def __init__(self, num_experts, num_experts_per_tok, hidden_size,
                 intermediate_size, *, ep_size=1, ep_rank=0, prefix=""):
        super().__init__(prefix=prefix)
        if num_experts % ep_size != 0:
            raise ValueError(
                f"MoEW13: num_experts ({num_experts}) % ep_size ({ep_size}) != 0"
            )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_local_experts = num_experts // ep_size
        self.local_expert_start = ep_rank * self.num_local_experts
```

Note: `auto_grid_dim`/`compile` use `self.num_experts` for the *grid* but the kernel iterates the *local* expert tensor; review those call sites and switch to `self.num_local_experts` where they index the weight tensor (grid heuristics that cap at 8 can keep `num_experts` or `num_local_experts` — pick `num_local_experts` since the loaded tensor is local). Confirm against `w13.py:_w13_auto_grid` usage.

In `MoEW13BF16.__init__` (`:112-129`), accept+forward `ep_size`/`ep_rank` and allocate local:
```python
    def __init__(self, num_experts, num_experts_per_tok, hidden_size,
                 intermediate_size, *, ep_size=1, ep_rank=0, prefix=""):
        super().__init__(num_experts, num_experts_per_tok, hidden_size,
                         intermediate_size, ep_size=ep_size, ep_rank=ep_rank,
                         prefix=prefix)
        self.weight = nn.Parameter(torch.empty(
            self.num_local_experts, 2 * intermediate_size, hidden_size,
            dtype=torch.bfloat16,
        ))

    def weight_loader(self, param, loaded_weight, *, expert_id, slot):
        """Write one expert's gate|up slab into the local w13 slot.
        Returns False (and writes nothing) if expert_id is not local to this rank.
        `slot` is 'gate' or 'up'; loaded_weight is (intermediate, hidden)."""
        local = expert_id - self.local_expert_start
        if not (0 <= local < self.num_local_experts):
            return False
        inter = self.intermediate_size
        row0 = 0 if slot == "gate" else inter
        param.data[local, row0:row0 + inter].copy_(loaded_weight)
        return True
```

Mirror in `w2.py`: `MoEW2BF16.weight_loader(param, loaded_weight, *, expert_id)` writes the whole `(hidden, inter)` down-proj into `param.data[local]`; local param shape `(num_local_experts, hidden, intermediate)`.

- [ ] **Step 4: Run test to verify it passes**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/mirage/mpk/layers/moe/w13.py python/mirage/mpk/layers/moe/w2.py tests/runtime_python/test_mode/test_moe_ep_loader.py
git commit -m "feat: EP-aware MoEW13/MoEW2 (local expert params + per-expert weight_loader)"
```

---

## Phase C — DeepSeek V3 catalog migration

### Task C1: Wire EP into `DeepseekV3MoEMLP`

**Files:**
- Modify: `python/mirage/mpk/models/deepseek_v3/modeling.py:620-688` (`DeepseekV3MoEMLP.__init__`)

- [ ] **Step 1: Compute EP range from ParallelConfig**

In `DeepseekV3MoEMLP.__init__`, after reading `self.num_experts` (`:624`), add:
```python
        pc = current_pk().parallel_config
        self.ep_size = pc.ep_size
        self.ep_rank = pc.ep_rank
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"DeepseekV3MoEMLP: n_routed_experts ({self.num_experts}) % "
                f"ep_size ({self.ep_size}) != 0"
            )
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
```

Pass `ep_size=self.ep_size, ep_rank=self.ep_rank` into the `MoEW13(...)` (`:653`) and `MoEW2(...)` (`:665`) constructors. Update `MoETopkRouting(...)` (`:640`) args `local_num_experts=self.num_local_experts, local_expert_start=self.local_expert_start` (currently hardcoded `num_experts`/`0`).

- [ ] **Step 2: Verify import/construction**

```bash
echo 'MIRAGE_SKIP_GPU_CHECK=1 python -c "import mirage.mpk.models.deepseek_v3.modeling"' > .scratch/run.sh
MIRAGE_SRC=$(pwd) MIRAGE_SKIP_GPU_CHECK=1 test-on-gpu gpu2 .scratch/run.sh
```
Expected: import OK (no syntax/name errors).

- [ ] **Step 3: Commit**

```bash
git add python/mirage/mpk/models/deepseek_v3/modeling.py
git commit -m "feat: wire EP (local experts) into DeepseekV3MoEMLP from ParallelConfig"
```

---

### Task C2: DSV3 streaming `load_weights` (mappings + fp8 pairing + EP experts)

**Files:**
- Modify: `python/mirage/mpk/models/deepseek_v3/modeling.py` — delete the three `_load_from_state_dict` hooks (`:197-218`, `:511-...`, `:691-736`); add `DeepseekV3ForCausalLM.load_weights` (`:1103+`)
- Reuse: `demo/deepseek_v3/models/convert.py` (`dequantize_fp8`, `is_fp8`) via the existing import path
- Test: `tests/runtime_python/test_mode/test_dsv3_ep_load.py` (create)

- [ ] **Step 1: Write the failing test** (pure CPU; synthetic tiny config)

```python
# tests/runtime_python/test_mode/test_dsv3_ep_load.py
import torch
from mirage.mpk.layers.moe.w13 import MoEW13BF16
from mirage.mpk.layers.moe.w2 import MoEW2BF16


def test_ep_expert_stacking_skips_remote():
    # rank 0 of ep_size 2 over 4 experts -> owns experts 0,1.
    w13 = MoEW13BF16(num_experts=4, num_experts_per_tok=2, hidden_size=8,
                     intermediate_size=4, ep_size=2, ep_rank=0)
    w2 = MoEW2BF16(num_experts=4, num_experts_per_tok=2, hidden_size=8,
                   intermediate_size=4, ep_size=2, ep_rank=0)
    gate = torch.ones(4, 8, dtype=torch.bfloat16)
    up = torch.full((4, 8), 2.0, dtype=torch.bfloat16)
    down = torch.full((8, 4), 3.0, dtype=torch.bfloat16)
    for e in range(4):
        w13.weight_loader(w13.weight, gate, expert_id=e, slot="gate")
        w13.weight_loader(w13.weight, up, expert_id=e, slot="up")
        w2.weight_loader(w2.weight, down, expert_id=e)
    # Only local experts 0,1 stored; tensor has exactly 2 slots.
    assert w13.weight.shape[0] == 2 and w2.weight.shape[0] == 2
    assert torch.equal(w13.weight[0, :4], gate) and torch.equal(w13.weight[0, 4:], up)
    assert torch.equal(w2.weight[1], down)
```

(The full `load_weights` is exercised end-to-end by the demo in Task C5; this unit test pins the EP stacking contract C2 depends on.)

- [ ] **Step 2: Run test to verify it fails**

```bash
echo 'MIRAGE_SKIP_GPU_CHECK=1 python -m pytest tests/runtime_python/test_mode/test_dsv3_ep_load.py -v' > .scratch/run.sh
MIRAGE_SRC=$(pwd) MIRAGE_SKIP_GPU_CHECK=1 test-on-gpu gpu2 .scratch/run.sh
```
Expected: FAIL until B1 is present (it is) — if B1 done, this passes; if so, proceed to wire `load_weights`. (This task's real deliverable is the loader below.)

- [ ] **Step 3: Implement `DeepseekV3ForCausalLM.load_weights`**

Delete the three `_load_from_state_dict` methods in `modeling.py`. Add to `DeepseekV3ForCausalLM`:

```python
    def load_weights(self, weights):
        """Streaming HF→catalog loader. Per (name, tensor):
          * fp8 weights are buffered until their `<name>_scale_inv` partner
            arrives, then dequantized to bf16 (convert.dequantize_fp8);
          * routed-expert keys are EP-filtered + stacked into local w13/w2;
          * raw q_b/kv_b/o_proj are stashed for process_weights() absorption;
          * everything else routes 1:1 via resolve_weight.
        Returns consumed key set; asserts no fp8 scale left unpaired.
        """
        import re
        from convert import dequantize_fp8, is_fp8   # demo convert.py (already on sys.path)

        params = dict(self.named_parameters())
        consumed, loaded = set(), set()
        fp8_pending = {}      # base_name -> fp8 weight tensor awaiting its scale
        scale_pending = {}    # base_name -> scale tensor awaiting its weight
        expert_re = re.compile(r"\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

        def _finalize(name, w):
            self._route_one(name, w, params, consumed, loaded, expert_re)

        for name, tensor in weights:
            if name.endswith("_scale_inv"):
                base = name[: -len("_scale_inv")]
                if base in fp8_pending:
                    w = dequantize_fp8(fp8_pending.pop(base), tensor,
                                       target_dtype=torch.bfloat16)
                    consumed.add(name)
                    _finalize(base, w)
                else:
                    scale_pending[base] = tensor
                    consumed.add(name)
                continue
            if is_fp8(tensor):
                if name in scale_pending:
                    w = dequantize_fp8(tensor, scale_pending.pop(name),
                                       target_dtype=torch.bfloat16)
                    _finalize(name, w)
                else:
                    fp8_pending[name] = tensor
                continue
            _finalize(name, tensor.to(torch.bfloat16))

        if fp8_pending or scale_pending:
            raise ValueError(
                f"DSV3.load_weights: unpaired fp8 tensors: "
                f"{sorted(fp8_pending) + sorted(scale_pending)}"
            )
        self.process_weights()
        return consumed

    def _route_one(self, name, w, params, consumed, loaded, expert_re):
        """Route one (already-bf16) tensor `w` for HF key `name`."""
        # 1. Routed experts -> EP-filtered stack into the owning layer's w13/w2.
        m = expert_re.search(name)
        if m:
            expert_id, proj = int(m.group(1)), m.group(2)
            moe = self._moe_for_key(name)          # find DeepseekV3MoEMLP by layer idx
            if proj == "down":
                ok = moe.w2.weight_loader(moe.w2.weight, w, expert_id=expert_id)
            else:
                slot = "gate" if proj == "gate" else "up"
                ok = moe.w13.weight_loader(moe.w13.weight, w, expert_id=expert_id,
                                           slot=slot)
            consumed.add(name)                      # consumed even if non-local (skipped)
            return
        # 2. q_b/kv_b/o_proj: stash raw for process_weights absorption.
        if name.endswith((".q_b_proj.weight", ".kv_b_proj.weight",
                          ".o_proj.weight")):
            self._stash_mla_raw(name, w)
            consumed.add(name)
            return
        # 3. Name remaps for catalog param names.
        remapped = self._REMAP_SUFFIX(name)        # e.g. q_a_proj.weight -> q_a_proj_weight
        res = self.resolve_weight(remapped, params)
        if res is None:
            raise ValueError(f"DSV3.load_weights: unexpected key {name!r}")
        param, loader, kwargs = res
        call = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        (loader(param, w, **call) if loader else param.data.copy_(w))
        consumed.add(name); loaded.add(kwargs.get("_path", remapped))
```

Note: the helpers `_moe_for_key`, `_stash_mla_raw`, `_REMAP_SUFFIX`, and `resolve_weight` overrides need concrete definitions matching the catalog param names (`q_a_proj_weight`, `kv_a_proj_with_mqa_weight`, `gate_weight`, `routing.bias`, shared-expert params). Define them on `DeepseekV3ForCausalLM`/`DeepseekV3MoEMLP` mirroring the current `_load_from_state_dict` key tables (`modeling.py:203-214`, `:697-732`) and the demo remaps (`demo_new.py:204-319`). `_moe_for_key` parses `model.layers.{i}.` and returns `self.model.layers[i].mlp`.

- [ ] **Step 4: Run test to verify it passes**

Same command as Step 2. Expected: PASS (unit contract). Full loader correctness is verified in C5.

- [ ] **Step 5: Commit**

```bash
git add python/mirage/mpk/models/deepseek_v3/modeling.py tests/runtime_python/test_mode/test_dsv3_ep_load.py
git commit -m "feat: DSV3 streaming load_weights (fp8 pairing + EP experts + remaps)"
```

---

### Task C3: DSV3 `process_weights` — MLA absorption

**Files:**
- Modify: `python/mirage/mpk/models/deepseek_v3/modeling.py` — add `process_weights` to `DeepseekV3MLA` (and ensure `DeepseekV3ForCausalLM.process_weights` recurses)
- Reuse: `convert.absorb_kv_into_q`, `get_model_params`

- [ ] **Step 1: Implement absorption from stashed raw weights**

On `DeepseekV3MLA`, add (mirrors `demo_new.py:232-267` but reads the stashed raw tensors set by `_stash_mla_raw`):
```python
    def process_weights(self):
        from convert import absorb_kv_into_q, get_model_params
        mp = get_model_params(self.config.to_dict())
        q_w = self._raw_q_b.float(); kv_w = self._raw_kv_b.float()
        absorbed = absorb_kv_into_q(q_w, kv_w, mp).to(torch.bfloat16)
        self.q_b_proj_weight.data.copy_(absorbed.contiguous())
        # Fuse W_UV into o_proj.
        nh = self.num_heads; qk_nope = self.qk_nope_head_dim
        v_dim = mp["v_head_dim"]; klr = self.kv_lora_rank
        W_UV = kv_w.reshape(nh, qk_nope + v_dim, klr)[:, qk_nope:, :]
        o = self._raw_o.to(torch.bfloat16); hdim = o.shape[0]
        o_fused = torch.einsum("dhn,hnk->dhk",
                               o.reshape(hdim, nh, v_dim).float(), W_UV.float())
        self.o_proj_weight.data.copy_(
            o_fused.reshape(hdim, nh * klr).to(torch.bfloat16).contiguous())
        del self._raw_q_b, self._raw_kv_b, self._raw_o   # free
```

`_stash_mla_raw` (on the ForCausalLM, routing by layer) sets `mla._raw_q_b / _raw_kv_b / _raw_o`. Mark `q_b_proj_weight`/`o_proj_weight` as filled for the missing-key assert (they're filled here, not in load) by adding their paths to `_optional_param_paths` OR recording them in `loaded` during `_stash_mla_raw`. Use `_optional_param_paths` to keep the assert honest about params filled in `process_weights`.

- [ ] **Step 2: Verify import/build**

```bash
echo 'MIRAGE_SKIP_GPU_CHECK=1 python -c "import mirage.mpk.models.deepseek_v3.modeling"' > .scratch/run.sh
MIRAGE_SRC=$(pwd) MIRAGE_SKIP_GPU_CHECK=1 test-on-gpu gpu2 .scratch/run.sh
```
Expected: import OK.

- [ ] **Step 3: Commit**

```bash
git add python/mirage/mpk/models/deepseek_v3/modeling.py
git commit -m "feat: DSV3 process_weights MLA absorption (kv_b->q_b, W_UV->o_proj)"
```

---

### Task C4: Switch `demo_new.py` to streaming load

**Files:**
- Modify: `demo/deepseek_v3/demo_new.py:155-358` (delete `_maybe_dequant`, `_load_hf_weights_with_absorption`), `:506-513` (call site)

- [ ] **Step 1: Replace the load call**

Replace the construction+load block (`demo_new.py:506-513`) with:
```python
        model = DeepseekV3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)
        from mirage.mpk.weight_loader import (
            find_safetensors_files, safetensors_weights_iterator,
        )
        files = find_safetensors_files(model_path)
        model.load_weights(safetensors_weights_iterator(files))  # streams + process_weights
```

Delete `_maybe_dequant` (`:155-165`) and `_load_hf_weights_with_absorption` (`:168-358`). Keep `convert` imports only if still used elsewhere; otherwise remove the now-orphaned import (`:50`).

Note on `layer_indices`: the old driver supported loading a subset of layers. The streaming iterator yields all keys; layer subsetting now relies on `num_hidden_layers_override` (HFConfig) so the model only *has* those layers and non-existent layer keys become "unexpected". If subset-without-override is still needed, add a `skip_fn` to the iterator that drops keys for layers ≥ override — but prefer `num_hidden_layers_override`.

- [ ] **Step 2: Verify the demo imports/parses**

```bash
echo 'MIRAGE_SKIP_GPU_CHECK=1 python -c "import ast; ast.parse(open(\"demo/deepseek_v3/demo_new.py\").read())"' > .scratch/run.sh
MIRAGE_SRC=$(pwd) MIRAGE_SKIP_GPU_CHECK=1 test-on-gpu gpu2 .scratch/run.sh
```
Expected: no parse error.

- [ ] **Step 3: Commit**

```bash
git add demo/deepseek_v3/demo_new.py
git commit -m "refactor: DSV3 demo_new uses streaming load_weights, delete preprocessing driver"
```

---

### Task C5: DSV3 demo correctness (single-GPU, reduced layers)

**Files:** none (verification only)

- [ ] **Step 1: Run reduced-layer DSV3 demo**

```bash
cat > .scratch/run.sh <<'EOF'
python demo/deepseek_v3/demo_new.py --num-hidden-layers-override 4 > output/dsv3_new.log 2>&1
EOF
MIRAGE_SRC=$(pwd) test-on-gpu gpu2 .scratch/run.sh
```
(Adjust the flag name to demo_new.py's actual arg; reduced layers avoid the known full-model host OOM.)

- [ ] **Step 2: Compare against current main behavior**

Expected: the demo runs to completion and produces the same token output as the pre-refactor driver for the same prompt + layer count (capture a baseline before starting Phase C if not already saved). If output differs → debug load_weights/process_weights mapping; the MLA absorption math must match `demo_new.py:232-267` exactly.

- [ ] **Step 3: Commit** — skip; verification only.

---

## Self-review notes (author)

- **Spec coverage:** §4.1 streaming + assert → A1. §4.1 Qwen3 composite → A2. §4.2 EP experts → B1, C1. §4.3 DSV3 stream/fp8/experts → C2; MLA absorption → C3; demo swap → C4. §6 tests → A1/A3/B1/C2/C5. §3 non-goals (no build_from_config, ep_size=1 gated) respected (C-tasks never touch build_from_config).
- **Open risk flagged in C2/C3:** `_moe_for_key`, `_REMAP_SUFFIX`, `_stash_mla_raw`, and the `resolve_weight` remap tables are specified by *reference* to the exact current key tables (`modeling.py:203-214`, `:697-732`, `demo_new.py:204-319`) rather than fully transcribed — the executor must transcribe those key→param maps verbatim. This is the one place to be most careful.
- **Memory note:** MLA absorption stashes per-layer raw q_b/kv_b/o_proj until `process_weights` (MLA-only, modest vs experts). If tighter bound needed later, do per-layer inline absorption — follow-up.
