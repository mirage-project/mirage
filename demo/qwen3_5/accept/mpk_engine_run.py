#!/usr/bin/env python3
"""MPK engine adapter + driver for the AC-3 correctness harness (M2-I9).

Token-ids-in / token-ids-out over ``MODE_OFFLINE``: this is the concrete
``EngineAdapter`` the harness (``accept/harness/engine_adapter.py``) was written
against, plus a CLI that materialises the ``bs<N>.json`` dumps
``run_ac3.py --engine-dump-dir`` consumes.

Chat template
-------------
There is nothing to re-apply. The reference (``accept/reference/reference_outputs.json``)
persists the exact ``input_ids`` ``generate_reference.py`` fed to
``model.generate`` -- i.e. the output of
``tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)``
-- and the harness hands those verbatim to the adapter
(``ac3_types.PromptRequest``: "never re-tokenized"). MPK is driven from those
ids, so the template is identical *by construction*, not by re-derivation.
``--verify-chat-template`` re-derives them from ``.pm/eval/prompts.jsonl`` with the
model's own tokenizer and asserts byte-equality anyway, and writes the result to
``chat_template_check.json`` as evidence.

Run protocol (why it is shaped this way)
----------------------------------------
* **Exactly-64.** ``MODE_OFFLINE`` retires a request on the *global*
  ``step + 1 >= max_seq_length`` (``persistent_kernel.cuh:281``), so with
  ``max_seq_length = max(prompt_len) + 64`` every request generates **at least**
  64 tokens (a short prompt generates more). We slice the first 64 written at
  ``tokens[r, prompt_len : prompt_len + 64]``. Greedy decode is causal, so the
  extra tail cannot change those 64 ids; reporting exactly ``num_generated``
  keeps ``ENGINE_TOO_LONG`` meaningful instead of tripping it on surplus compute.
* **EOS off (``eos_token_id = -1``).** All 10 reference prompts have
  ``hit_eos = false`` / ``num_generated = 64``, so honouring EOS cannot change a
  matching run -- but it *can* truncate a diverging one into a bare
  ``ENGINE_TOO_SHORT`` with no per-position evidence. Ignoring it is strictly
  more informative and never more lenient: an MPK-emitted EOS id still lands in
  the token stream and still fails the position compare.
* **Waves, not rolling admission.** Each batch is one wave of at most
  ``batch_size`` requests with ``total_num_requests == wave size``, so the
  offline scheduler never admits a request into a slot another request has
  vacated. See ``HAZARD-COMPACTION`` below -- this is the adapter exercising the
  batch size it was asked for, not a workaround for a failing gate.

HAZARD-WAVE-RESET (raised in M2-I9, root-caused and RETIRED in M3-I2a)
---------------------------------------------------------------------
M2-I9 saw the megakernel wedge on the second in-process wave at bs=4 and
attributed it to ``init_request_func`` leaving some task-graph queue state
un-reset. **That attribution was wrong.** M3-I2a re-ran the failing wave pair on
M2-I9's own compiled kernel and could not reproduce it, then ran 62 in-process
launches across bs 4/8/16 with the prompt geometry changing at every launch
boundary -- all clean, all byte-identical.

The real precondition is SM residency. MPK's workers and schedulers spin-wait on
each other and never yield an SM, and the launch config claims the whole GPU
(``get_configurations_from_gpu``: 128 workers at one SM each, plus
``4 * (sm_count - workers)`` schedulers packed 4-per-SM into the remaining 20 --
exactly the 148 SMs of a B200). One block of any other process is enough to stop
the grid from becoming co-resident, and a partially resident grid deadlocks. The
positive control reproduces M2-I9's signature exactly: waves 0-4 run clean, a
co-tenant lands on the GPU, wave 5 wedges at ``step=[0,0,0,0]``.

So in-process multi-wave is supported; what it needs is an EXCLUSIVE GPU. The
launcher now probes co-residency before every launch and raises instead of
wedging (``MPK_SKIP_RESIDENCY_CHECK=1`` opts out). ``--prompt-ids`` still works
for bisection, but it is no longer required as a workaround.

A wedged megakernel does NOT die on SIGTERM -- it holds its CUDA context and
spins the GPU at 100% -- so always ``kill -9`` and verify with
``nvidia-smi --query-compute-apps``.

HAZARD-COMPACTION (found in M2-I9, reported, not silently avoided)
------------------------------------------------------------------
``prepare_next_batch`` compacts surviving requests toward slot 0 when one
retires (``persistent_kernel.cuh:311-362``) and migrates their KV pages with
them (``paged_kv_indices_snapshot``). The **GDN conv/recurrent state pools are
not part of that migration**: they are ``[max_num_batched_requests, ...]``
tensors indexed by the *batch slot*, so a survivor that moves from slot 1 to
slot 0 starts reading the retired request's state.

CORRECTION (M3-I9). This paragraph used to end "this only fires ... i.e. rolling
admission with ``total_num_requests > max_num_batched_requests``, which the wave
protocol above does not use". The trigger is right; the ``i.e.`` is wrong, and
so is the implied all-clear. A request retiring while another is active needs no
rolling admission at all -- it happens *inside* a single wave, because
``max_seq_length`` retires on the global absolute step while slot-order-greedy
admission advances the low slots first. Replaying the shipped waves
(``opt/m3i9/protocol_sim.py``) gives **1 / 12 / 69 live-slot migrations at bs
4 / 8 / 16**.

Most of those are harmless because the request has already written the 64 tokens
the harness reports. The ones that are not are the duplicate padding slots --
and they are exactly the slots whose ``slot_isolation_checks`` report
``identical: false`` in every committed bs16 / bs8-w1 / bs4-w2 dump: 14 of 14,
no exceptions (``opt/m3i9/compaction_audit.py``). Those mismatches have been
read as a slot-isolation question; they are this hazard reaching an answer.
The competing explanation -- that a differently *chopped* prefill gives a
different answer -- is refuted by the committed report itself: 9 of the 10
prompts are prefilled with two to four different chunkings across batch sizes,
in placements that never migrate, and all 25 (prompt, batch size) sequences are
byte-identical.

``assert_no_rolling_admission`` still holds the line it was written for, but it
is a *sufficient* condition, not the necessary one: it passes for every wave
above while live slots are being moved. Every wave's exposure is now recorded in
the timings artifact (``compaction.live_slot_migrations`` /
``compaction.straddling_slots``) so it stops being invisible. The fix is not a
different slot order -- the lowest live slot always has budget priority, so it
always retires first whatever the order -- but equalising the per-iteration
advance; see ``opt/m3i9/README.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "harness"))
sys.path.insert(0, str(HERE))

import admission_policy  # noqa: E402  -- THE admission-cap policy authority
from ac3_types import EngineSequence, PromptRequest  # noqa: E402
from engine_adapter import EngineAdapter  # noqa: E402

MAX_NEW_TOKENS = 64
# One MoE-router pass covers 16 rows (builder.MOE_ROUTER_ROWS_PER_TILE). Until
# M3-I5b that was a hard cap and this default WAS the cap; the kernel now loops
# over row tiles, so mbt is a free scheduling knob again. The default is
# deliberately UNCHANGED at 16 so every M2/M3 measurement stays comparable --
# raising it is the A/B in demo/qwen3_5/accept/opt/m3i5b/prep.md, not a default.
DEFAULT_MBT = 16
DEFAULT_PAGE_SIZE = 256


def log(msg: str) -> None:
    print(f"[mpk_engine] {msg}", flush=True)


# ----------------------------------------------------------------------------
# reference / prompt plumbing
# ----------------------------------------------------------------------------
def load_reference_requests(ref_path: Path) -> List[PromptRequest]:
    with open(ref_path, "r") as f:
        raw = json.load(f)
    return [
        PromptRequest(prompt_id=pid, input_ids=list(r["input_ids"]))
        for pid, r in raw["results"].items()
    ]


def verify_chat_template(ref_path: Path, prompts_path: Path, snapshot: str,
                         out_path: Optional[Path]) -> dict:
    """Re-derive input_ids from the pinned prompt file with the model's own
    tokenizer and assert they equal the reference's. Evidence that MPK is fed a
    chat-templated prompt identical to the reference's."""
    from transformers import AutoTokenizer

    with open(ref_path, "r") as f:
        ref = json.load(f)["results"]
    tok = AutoTokenizer.from_pretrained(snapshot)

    rows = []
    with open(prompts_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    report = {"template_call": ("tokenizer.apply_chat_template(messages, "
                               "tokenize=True, add_generation_prompt=True)"),
              "snapshot": str(snapshot), "prompts": {}, "all_equal": True}
    for row in rows:
        pid = row["id"]
        enc = tok.apply_chat_template(row["messages"], tokenize=True,
                                      add_generation_prompt=True,
                                      return_tensors="pt")
        ids = (enc.input_ids[0].tolist() if hasattr(enc, "input_ids")
               else enc[0].tolist())
        equal = ids == ref[pid]["input_ids"]
        report["prompts"][pid] = {"equal": equal, "len_rederived": len(ids),
                                  "len_reference": len(ref[pid]["input_ids"])}
        if not equal:
            report["all_equal"] = False
            report["prompts"][pid]["rederived"] = ids
            report["prompts"][pid]["reference"] = ref[pid]["input_ids"]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


# ----------------------------------------------------------------------------
# the adapter
# ----------------------------------------------------------------------------
class MPKOfflineAdapter(EngineAdapter):
    """Drives the full 40-layer Qwen3.5 megakernel in ``MODE_OFFLINE``.

    One compiled kernel per ``(max_num_batched_tokens, batch_size)`` pair --
    both are baked into the graph -- reused across every wave and every prompt
    at that batch size.
    """

    def __init__(self, *, model_name: str, model_path: Optional[str] = None,
                 mbt: int = DEFAULT_MBT, page_size: int = DEFAULT_PAGE_SIZE,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 kernel_dir: Optional[Path] = None,
                 reuse_kernel: bool = False,
                 expose_logits: bool = False,
                 pinned_max_seq_length: Optional[int] = None,
                 slot_order: str = "wave",
                 audit_compaction: bool = True,
                 per_request_token_cap="policy"):
        self.model_name = model_name
        self.model_path = model_path
        self.mbt = mbt
        self.page_size = page_size
        self.max_new_tokens = max_new_tokens
        self.kernel_dir = kernel_dir
        self.reuse_kernel = reuse_kernel
        self.expose_logits = expose_logits
        # Pinning max_seq_length makes the compiled kernel identical for every
        # wave of a batch size, so the decode length is a property of the
        # protocol rather than of the wave a prompt landed in -- and one
        # compilation serves every wave (and every process, when a run is split
        # across processes for bisection).
        self.pinned_max_seq_length = pinned_max_seq_length
        # M3-I9 admission-protocol knobs. "wave" is the shipped behaviour, byte
        # for byte: sort the wave ascending, then pad by repetition. The other
        # values are experiment arms (opt/m3i9/README.md).
        if slot_order not in ("wave", "sorted-padded", "descending"):
            raise ValueError(f"unknown slot_order {slot_order!r}")
        self.slot_order = slot_order
        self.audit_compaction = audit_compaction
        # The per-request per-iteration admission cap. THE POLICY LIVES IN
        # admission_policy.py -- this class only stores the REQUEST and resolves
        # it there once the batch size is known. "policy" (the default) is the
        # shipped behaviour; "none"/None reproduces the pre-policy runtime;
        # "auto" / an int force an arm. The knob is compile-time, so the resolved
        # value is part of a kernel directory's identity (admission_policy.py).
        self.per_request_token_cap = admission_policy.validate(per_request_token_cap)
        self._mpk = None
        self._builder = None
        self._bs = None
        self._meta = {}
        self.timings: List[dict] = []

    # -- graph construction -------------------------------------------------
    def _build(self, batch_size: int, max_seq_length: int, total_requests: int):
        import torch
        import mirage as mi
        from mirage.mpk.models.qwen3_5.builder import Qwen35Builder

        assert_no_rolling_admission(total_requests, batch_size)

        num_pages_per_req = -(-max_seq_length // self.page_size)
        max_num_pages = max(batch_size * num_pages_per_req + 4, 8)

        num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
        # MPK_NUM_WORKERS -- explicit override for the megakernel WIDTH, added
        # 2026-07-31. Upstream PR #743 ("increased num_workers for B200") raises
        # the sm_cnt>=144 tier from 128 to 136 workers, which on this 148-SM part
        # trades 8 scheduler SMs for 8 worker SMs. That matters here because
        # M4-I8's floors are max(critical path, total task work / num_workers) and
        # bs16 is the one batch size where the WORK BOUND binds (5976.0 us vs a cp
        # of 5638.7): dividing by 136 instead of 128 drops it to ~5624.5 and flips
        # bs16 back to critical-path-bound, where our levers actually work.
        # Set explicitly so an A/B arm NAMES the knob rather than inheriting it --
        # and note this changes the GENERATED code (argmax grid_dim and the
        # (mbt, num_workers) partial buffers both scale with it), so a kernel dir
        # must NOT be reused across values (the M3-I7 / M3-I9 trap).
        _w_override = os.environ.get("MPK_NUM_WORKERS", "").strip()
        if _w_override:
            _sm = torch.cuda.get_device_properties(0).multi_processor_count
            num_workers = int(_w_override)
            num_schedulers = 4 * (_sm - num_workers)
            if num_schedulers <= 0:
                raise SystemExit(
                    f"MPK_NUM_WORKERS={num_workers} leaves no SMs for schedulers "
                    f"on a {_sm}-SM device")
            print(f"[worker-override] num_workers={num_workers} "
                  f"num_schedulers={num_schedulers} (sm_cnt={_sm})")
        dev = "cuda"
        step = torch.zeros(total_requests, dtype=torch.int32, device=dev)
        tokens = torch.zeros((total_requests, max_seq_length), dtype=torch.long,
                             device=dev)
        prompt_lengths = torch.zeros(total_requests, dtype=torch.int32, device=dev)
        num_new_tokens = torch.ones(total_requests, dtype=torch.int32, device=dev)
        input_tokens = torch.zeros((self.mbt, 1), dtype=torch.long, device=dev)
        output_tokens = torch.zeros((self.mbt, 1), dtype=torch.long, device=dev)
        meta = {
            "step": step,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": torch.zeros(batch_size + 1, dtype=torch.int32, device=dev),
            "paged_kv_indptr_buffer": torch.zeros(batch_size + 1, dtype=torch.int32, device=dev),
            "paged_kv_indices_buffer": torch.zeros(max_num_pages, dtype=torch.int32, device=dev),
            "paged_kv_last_page_len_buffer": torch.zeros(batch_size, dtype=torch.int32, device=dev),
            "paged_kv_indices_snapshot": torch.zeros(max_num_pages, dtype=torch.int32, device=dev),
        }

        torch.set_default_dtype(torch.bfloat16)
        mpk = mi.PersistentKernel(
            mode="offline", world_size=1, mpi_rank=0,
            num_workers=num_workers, num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=max_seq_length,
            max_num_batched_requests=batch_size,
            max_num_batched_tokens=self.mbt,
            max_num_pages=max_num_pages,
            page_size=self.page_size,
            # See module docstring: all 10 reference prompts run the full 64
            # tokens without EOS, so disabling it cannot mask a mismatch -- it
            # only stops a diverging run from truncating into an evidence-free
            # ENGINE_TOO_SHORT.
            eos_token_id=-1,
            meta_tensors=meta,
            profiler_tensor=None,
            trace_name="",
            spec_decode_config=None,
            use_cutlass_kernel=True,
            **self._cap_kwargs(batch_size),
        )
        builder = Qwen35Builder(mpk)
        builder.expose_logits = self.expose_logits
        t0 = time.time()
        builder.build_from_model(model_name=self.model_name,
                                 model_path=self.model_path)
        log(f"graph assembled in {time.time() - t0:.1f}s "
            f"(bs={batch_size}, mbt={self.mbt}, msl={max_seq_length})")

        out_dir = str(self.kernel_dir) if self.kernel_dir else None
        t0 = time.time()
        if self.reuse_kernel and self.kernel_dir and \
                (self.kernel_dir / "task_graph_rank0.json").exists():
            mpk.load_mpk_kernel(output_dir=out_dir)
            log(f"reused compiled kernel from {out_dir} ({time.time() - t0:.1f}s)")
        else:
            mpk.compile(output_dir=out_dir)
            log(f"megakernel compiled in {time.time() - t0:.1f}s")

        self._mpk, self._builder, self._bs = mpk, builder, batch_size
        self._meta = meta
        self._max_seq_length = max_seq_length
        self._total_requests = total_requests

    # -- the EngineAdapter contract ----------------------------------------
    def run(self, requests: List[PromptRequest],
            batch_size: int) -> Dict[str, EngineSequence]:
        import torch

        plens = [len(r.input_ids) for r in requests]
        # ONE max_seq_length for every wave and every batch size: the kernel is
        # compiled against it, and a uniform value keeps the decode length a
        # property of the protocol rather than of the wave a prompt landed in.
        max_seq_length = (self.pinned_max_seq_length
                          if self.pinned_max_seq_length is not None
                          else max(plens) + self.max_new_tokens)
        assert max_seq_length >= max(plens) + self.max_new_tokens, (
            f"pinned max_seq_length {max_seq_length} cannot deliver "
            f"{self.max_new_tokens} new tokens for a {max(plens)}-token prompt")
        # Ascending prompt length inside a wave.
        #
        # This used to claim "the longest prompt retires first, so putting it in
        # the highest slot means no survivor is ever compacted downward". M3-I9
        # replayed `prepare_next_batch` and refuted it: slot-order-greedy
        # admission gives the token budget to the LOWEST live slot first, so the
        # lowest slot always advances fastest and therefore always retires
        # first, whatever the prompt lengths are. The shipped waves move live
        # slots 1 / 12 / 69 times at bs 4 / 8 / 16 (opt/m3i9/compaction_audit.py).
        # No slot ORDER can prevent that; only equalising the per-iteration
        # advance can (opt/m3i9/README.md, "the runtime knob"). The sort is kept
        # because it is what every committed AC-3 artifact was produced with.
        ordered = sorted(requests, key=lambda r: len(r.input_ids))
        if self.slot_order == "descending":
            ordered = ordered[::-1]
        waves = [ordered[i:i + batch_size]
                 for i in range(0, len(ordered), batch_size)]
        # Every wave is padded back up to exactly `batch_size` live requests, by
        # repeating prompts from the same wave. Two reasons: the kernel bakes in
        # the request geometry (so it cannot change between waves), and a
        # duplicated prompt in a different slot is a free per-wave check that
        # slot state really is isolated -- `dup_checks` records whether the
        # copies agreed.
        self.dup_checks: List[dict] = []

        out: Dict[str, EngineSequence] = {}
        for w_idx, wave in enumerate(waves):
            slots = list(wave)
            while len(slots) < batch_size:
                slots.append(wave[len(slots) % len(wave)])
            if self.slot_order == "sorted-padded":
                # M3-I9 P1: sort the PADDED slot list too, so the repeated short
                # prompts stop sitting in the high slots. Predicted +11.4% wave
                # time at bs16 and no rebuild needed -- but it also changes which
                # slot reports each prompt, so it is off by default and is an
                # experiment arm, not a shipping default.
                slots = sorted(slots, key=lambda r: len(r.input_ids))

            if self._mpk is None:
                self._build(batch_size, max_seq_length, batch_size)
            elif (self._bs != batch_size or
                  self._max_seq_length != max_seq_length):
                raise RuntimeError("wave geometry changed after the build")

            meta = self._meta
            meta["tokens"].zero_()
            meta["step"].zero_()
            meta["num_new_tokens"].fill_(1)
            for r_i, req in enumerate(slots):
                ids = req.input_ids
                meta["tokens"][r_i, :len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device="cuda")
                meta["prompt_lengths"][r_i] = len(ids)
            # Re-run init_kernel so step / request_ids / qo_indptr / the page
            # queue / next_request_id start this wave clean
            # (persistent_kernel.cuh:143 `init_kernel`, reachable from Python as
            # the launcher module's `init_request_func`). GDN state needs no
            # reset: `step == 0` makes the kernel treat it as zero
            # (v1-architecture.md 3.3), and prefill overwrites the KV pages.
            self._reset_runtime()

            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            self._mpk()
            ender.record()
            self._watchdog(meta, slots, w_idx, batch_size)
            torch.cuda.synchronize()
            wall_ms = starter.elapsed_time(ender)

            steps = meta["step"].tolist()
            toks = meta["tokens"].cpu().tolist()
            decode_steps = 0
            per_slot: Dict[int, List[int]] = {}
            for r_i, req in enumerate(slots):
                plen = len(req.input_ids)
                gen = toks[r_i][plen:plen + self.max_new_tokens]
                per_slot[r_i] = gen
                if len(gen) < self.max_new_tokens:
                    log(f"WARNING {req.prompt_id}: only {len(gen)} tokens "
                        f"(step={steps[r_i]}, msl={max_seq_length})")
                decode_steps = max(decode_steps, steps[r_i] - plen)
            # First slot holding each prompt is the reported one; any repeat is
            # an isolation check, never a second chance.
            first_slot: Dict[str, int] = {}
            for r_i, req in enumerate(slots):
                if req.prompt_id not in first_slot:
                    first_slot[req.prompt_id] = r_i
                    if req in wave:
                        out[req.prompt_id] = EngineSequence(
                            token_ids=per_slot[r_i], topk_logits=None)
                else:
                    ref = per_slot[first_slot[req.prompt_id]]
                    agree = per_slot[r_i] == ref
                    # M3-I9: WHERE they diverge is the evidence, not THAT they do.
                    # If the cause is HAZARD-COMPACTION the copy must agree up to
                    # the token produced by the iteration that migrated its slot,
                    # and the predicted index differs per slot (60/54/46/35/19/0
                    # for slots 10..15 of the bs16 wave). If the cause were the
                    # different prefill chunking the copy gets, position 0 would
                    # already differ. A boolean cannot tell those apart.
                    first_div = next(
                        (k for k, (x, y) in enumerate(zip(ref, per_slot[r_i]))
                         if x != y),
                        min(len(ref), len(per_slot[r_i])) if not agree else None)
                    self.dup_checks.append({
                        "batch_size": batch_size, "wave": w_idx,
                        "prompt_id": req.prompt_id,
                        "slots": [first_slot[req.prompt_id], r_i],
                        "identical": agree,
                        "first_divergence": first_div})
                    if not agree:
                        log(f"SLOT-ISOLATION MISMATCH {req.prompt_id} "
                            f"slots {first_slot[req.prompt_id]} vs {r_i}")
            self.timings.append({
                "batch_size": batch_size, "wave": w_idx,
                "num_requests": len(slots),
                "num_distinct_prompts": len(wave),
                "prompt_ids": [r.prompt_id for r in slots],
                "wall_ms": wall_ms,
                "max_decode_steps": decode_steps,
                "ms_per_decode_step": (wall_ms / decode_steps) if decode_steps else None,
                "tokens_per_s": ((len(wave) * decode_steps) / (wall_ms / 1000.0))
                                if decode_steps and wall_ms else None,
                "max_seq_length": max_seq_length,
                "max_num_batched_tokens": self.mbt,
                "slot_order": self.slot_order,
                "compaction": self._compaction_audit(
                    [len(r.input_ids) for r in slots], max_seq_length),
            })
            log(f"bs={batch_size} wave={w_idx} slots={len(slots)} "
                f"distinct={len(wave)} wall={wall_ms:.1f}ms "
                f"steps={decode_steps}")
        return out

    def _cap_kwargs(self, batch_size: int) -> dict:
        """The per-request per-iteration token cap this run compiles with.

        `prepare_next_batch` step 3 otherwise gives a prefilling request
        `min(remaining, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens)` -- the whole
        remaining budget. The cap adds one more `min` against a compile-time
        `MPK_MAX_TOKENS_PER_REQUEST` whose default is
        `MPK_MAX_NUM_BATCHED_TOKENS`, i.e. the pre-policy behaviour byte for
        byte. The policy and its evidence live in admission_policy.py; the spec
        of the runtime knob is opt/m3i9/README.md.
        """
        log(admission_policy.describe(self.per_request_token_cap, self.mbt,
                                      batch_size))
        cap = admission_policy.resolve_int(self.per_request_token_cap, self.mbt,
                                          batch_size)
        if cap is None:
            return {}
        import inspect
        # Interrogate the CLASS, not `mirage.PersistentKernel`.  That package
        # attribute is monkeypatchable and is in fact patched right here:
        # `opt/profile_wave.py`'s `ProfiledAdapter._build` rebinds it to a plain
        # function for the duration of this very call, so probing it asked
        # `function.__init__` for a `max_tokens_per_request` parameter, never
        # found one, and raised NotImplementedError on a build that has the knob.
        # Symptom: profile_wave.py failed at every batch size where the policy
        # resolves to a cap -- bs >= 4 -- i.e. exactly the AC-4 batch sizes,
        # while bs 1/2 (cap None, early return above) worked. Found by M4-I5.
        from mirage.mpk.persistent_kernel import PersistentKernel as _PK
        if "max_tokens_per_request" not in inspect.signature(
                _PK.__init__).parameters:
            raise NotImplementedError(
                "--per-request-token-cap needs the runtime knob specced in "
                "demo/qwen3_5/accept/opt/m3i9/README.md ('the runtime knob'): "
                "MPK_MAX_TOKENS_PER_REQUEST in persistent_kernel.cuh plus the "
                "PersistentKernel(max_tokens_per_request=...) parameter. This "
                "build does not have it, and running without it would measure "
                "the UNMODIFIED schedule under a flag that claims otherwise.")
        return {"max_tokens_per_request": cap}

    def _compaction_audit(self, plens: List[int], max_seq_length: int) -> Optional[dict]:
        """Replay this wave's admission and record HAZARD-COMPACTION exposure.

        Diagnostic only -- it never touches the run. It exists because the
        hazard's stated precondition (rolling admission) is not the real one:
        a wave with `total_num_requests == batch_size` still migrates live slots
        whenever one request retires while another is active, which is every
        AC-3 wave at bs >= 4. `straddling_slots` is the subset whose reported
        `max_new_tokens` window is still being written when it migrates -- those
        are the answers the hazard can actually reach, and they are exactly the
        duplicate slots whose `slot_isolation_checks` report `identical: false`.
        """
        if not self.audit_compaction:
            return None
        try:
            sys.path.insert(0, str(HERE / "opt" / "m3i9"))
            from protocol_sim import audit, simulate  # noqa: E402
        except Exception as exc:            # tooling absent -> never break a run
            return {"unavailable": repr(exc)}
        cap = admission_policy.resolve_int(self.per_request_token_cap, self.mbt,
                                           len(plens))
        sim = simulate(plens, self.mbt, max_seq_length, cap=cap)
        a = audit(sim, self.max_new_tokens)
        rec = {"predicted_iterations": sim["n_iterations"],
               # the COMPILE-TIME value this wave's kernel carries, recorded so a
               # kernel-dir reuse across cap values is visible in the artifact
               # rather than only in the replay (admission_policy.py).
               "per_request_token_cap_requested":
                   (self.per_request_token_cap
                    if self.per_request_token_cap is not None else "none"),
               "per_request_token_cap_compiled": cap,
               "live_slot_migrations": a["n_moves"],
               "migrated_slots": a["moved_requests"],
               "straddling_slots": a["straddling_requests"]}
        if a["straddling_requests"]:
            log(f"HAZARD-COMPACTION exposure: slots {a['straddling_requests']} "
                f"migrate while their reported window is still being written "
                f"({a['n_moves']} live-slot migrations this wave)")
        return rec

    def _watchdog(self, meta, slots, w_idx: int, batch_size: int,
                  timeout_s: float = 120.0, poll_s: float = 5.0) -> None:
        """Poll the runtime's own per-request `step` on a SIDE stream while the
        persistent kernel runs on the default one, so a stall is reported with
        the exact per-request progress instead of hanging silently.

        Reading a device tensor with the default stream would queue behind the
        megakernel and block; a separate stream copies concurrently.
        """
        import torch

        side = torch.cuda.Stream()
        done_evt = torch.cuda.Event()
        done_evt.record()
        t0 = time.time()
        last = None
        stalled_for = 0.0
        while not done_evt.query():
            if time.time() - t0 > timeout_s:
                with torch.cuda.stream(side):
                    steps = meta["step"].to("cpu", non_blocking=False).tolist()
                    qo = meta["qo_indptr_buffer"].to("cpu").tolist()
                    kvp = meta["paged_kv_indptr_buffer"].to("cpu").tolist()
                    lpl = meta["paged_kv_last_page_len_buffer"].to("cpu").tolist()
                side.synchronize()
                plens = [len(r.input_ids) for r in slots]
                phase = ["prefill" if s < p else "decode"
                         for s, p in zip(steps, plens)]
                raise RuntimeError(
                    f"WATCHDOG: bs={batch_size} wave={w_idx} made no progress "
                    f"for {stalled_for:.0f}s (total {time.time() - t0:.0f}s). "
                    f"step={steps} prompt_len={plens} phase={phase} "
                    f"qo_indptr={qo} kv_indptr={kvp} last_page_len={lpl}")
            time.sleep(poll_s)
            with torch.cuda.stream(side):
                cur = meta["step"].to("cpu").tolist()
            side.synchronize()
            if cur == last:
                stalled_for += poll_s
            else:
                stalled_for = 0.0
                log(f"  [watchdog] bs={batch_size} wave={w_idx} step={cur}")
            last = cur

    def _reset_runtime(self) -> None:
        fn = getattr(self._mpk, "init_request_func", None)
        if fn is None:
            mod = sys.modules.get("__mirage_launcher")
            fn = getattr(mod, "init_request_func", None) if mod else None
        if fn is None:
            raise RuntimeError(
                "launcher exposes no init_request_func: cannot reset the "
                "runtime between waves (persistent_kernel.py:125)")
        fn()


def assert_no_rolling_admission(total_requests: int, batch_size: int) -> None:
    """See HAZARD-COMPACTION in the module docstring: GDN state pools are
    slot-indexed and are NOT migrated when the offline scheduler compacts
    surviving requests toward slot 0. Rolling admission is therefore unsafe for
    this model until that is fixed."""
    if total_requests > batch_size:
        raise RuntimeError(
            f"total_num_requests={total_requests} > max_num_batched_requests="
            f"{batch_size}: the offline scheduler would recycle slots, and the "
            "GDN conv/recurrent state pools are slot-indexed without migration "
            "(HAZARD-COMPACTION). Split into waves of at most batch_size.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--reference",
                    default=str(HERE / "reference" / "reference_outputs.json"))
    ap.add_argument("--prompts-file", default=None,
                    help="Only needed for --verify-chat-template.")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--out-dir", required=True,
                    help="Writes bs<N>.json (+ timings) here for run_ac3.py "
                         "--engine-dump-dir.")
    ap.add_argument("--kernel-dir", default=None,
                    help="Compiled-kernel output dir (reused across waves).")
    ap.add_argument("--reuse-kernel", action="store_true",
                    help="Load a previously compiled kernel from --kernel-dir.")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--mbt", type=int, default=DEFAULT_MBT,
                    help="max_num_batched_tokens (>= longest prompt for "
                         "single-chunk prefill; >= batch size for decode).")
    ap.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    ap.add_argument("--prompt-ids", default=None,
                    help="Comma-separated subset (bisection / re-runs).")
    ap.add_argument("--verify-chat-template", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="Pin max_seq_length so every wave of every batch size "
                         "compiles the SAME kernel, which keeps the decode "
                         "length a property of the protocol and lets one "
                         "compilation serve every wave.")
    ap.add_argument("--slot-order", default="wave",
                    choices=("wave", "sorted-padded", "descending"),
                    help="M3-I9 admission arm. 'wave' (default) is the shipped "
                         "protocol, byte for byte. 'sorted-padded' also sorts "
                         "the padded slot list (predicted +11.4%% wave time at "
                         "bs16, but it changes which slot reports each prompt). "
                         "See opt/m3i9/README.md.")
    ap.add_argument("--per-request-token-cap", default="policy",
                    help="Cap the tokens one request may contribute to one "
                         f"iteration: {admission_policy.CAP_CHOICES_DOC}. "
                         "DEFAULT 'policy' = the shipped policy in "
                         "admission_policy.py (auto at bs>="
                         f"{admission_policy.CAP_MIN_BATCH_SIZE}); 'none' "
                         "forces the pre-policy uncapped runtime and is what "
                         "reproduces every artifact captured before the policy "
                         "landed. Compile-time knob: a kernel dir is only valid "
                         "for the cap it was built with. Needs the runtime knob "
                         "(opt/m3i9/README.md); the adapter raises rather than "
                         "silently measuring the unmodified schedule.")
    ap.add_argument("--no-audit-compaction", action="store_true",
                    help="skip the HAZARD-COMPACTION exposure replay recorded "
                         "in the timings artifact (diagnostic only)")
    ap.add_argument("--dump-name", default=None,
                    help="Override the dump filename (default bs<N>.json); use "
                         "per-wave names when splitting a run across "
                         "processes.")
    args = ap.parse_args(argv)

    ref_path = Path(args.reference)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_chat_template:
        from mirage.mpk.models.qwen3_5.weight_loader import resolve_snapshot
        prompts = Path(args.prompts_file) if args.prompts_file else None
        if prompts is None:
            log("--verify-chat-template needs --prompts-file; skipping")
        else:
            rep = verify_chat_template(ref_path, prompts,
                                       resolve_snapshot(args.model, args.model_path),
                                       out_dir / "chat_template_check.json")
            log(f"chat-template re-derivation all_equal={rep['all_equal']}")
            if not rep["all_equal"]:
                return 2

    requests = load_reference_requests(ref_path)
    if args.prompt_ids:
        wanted = {s.strip() for s in args.prompt_ids.split(",") if s.strip()}
        requests = [r for r in requests if r.prompt_id in wanted]
    log(f"{len(requests)} prompts, lengths "
        f"{sorted(len(r.input_ids) for r in requests)}")

    adapter = MPKOfflineAdapter(
        model_name=args.model, model_path=args.model_path, mbt=args.mbt,
        page_size=args.page_size, max_new_tokens=args.max_new_tokens,
        kernel_dir=Path(args.kernel_dir) if args.kernel_dir else None,
        reuse_kernel=args.reuse_kernel,
        pinned_max_seq_length=args.max_seq_length,
        slot_order=args.slot_order,
        audit_compaction=not args.no_audit_compaction,
        per_request_token_cap=args.per_request_token_cap,
    )
    result = adapter.run(requests, args.batch_size)

    dump = {pid: {"token_ids": seq.token_ids} for pid, seq in result.items()}
    dump_path = out_dir / (args.dump_name or f"bs{args.batch_size}.json")
    with open(dump_path, "w") as f:
        json.dump(dump, f, indent=2)
    log(f"wrote {dump_path} ({len(dump)} prompts)")

    tname = (f"timings_{args.dump_name}" if args.dump_name
             else f"timings_bs{args.batch_size}.json")
    with open(out_dir / tname, "w") as f:
        json.dump({"batch_size": args.batch_size,
                   "note": ("informational only -- no perf claim; mbt is fixed "
                            "across batch sizes for a uniform correctness "
                            "config, which inflates per-step cost at small bs"),
                   "admission_policy": dict(
                       admission_policy.summary(),
                       requested=(adapter.per_request_token_cap
                                  if adapter.per_request_token_cap is not None
                                  else "none"),
                       compiled_cap=admission_policy.resolve_int(
                           adapter.per_request_token_cap, args.mbt,
                           args.batch_size)),
                   "waves": adapter.timings,
                   "slot_isolation_checks": adapter.dup_checks}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
