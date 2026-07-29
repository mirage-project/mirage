#!/usr/bin/env python3
"""THE authoritative per-request admission-cap policy for the Qwen3.5 MPK runtime.

One place, one rule. Every caller -- ``mpk_engine_run.py`` and its CLI, the AC-3
harness, ``harness/gate_ac3_stable.py``, the opt/ probes, and the M4 final gate --
derives the shipped policy from this module. ``docs/qwen35/bench-protocol.md``
documents it and points here; it does not restate the batch-size table, because a
policy written down in two places is a divergence bug waiting for the first run
that reads only one of them.

WHAT THE KNOB IS
----------------
``MPK_MAX_TOKENS_PER_REQUEST`` caps how many tokens ONE request may contribute to
ONE megakernel iteration. It is a **compile-time define** (``persistent_kernel.py``
turns ``PersistentKernel(max_tokens_per_request=k)`` into
``-DMPK_MAX_TOKENS_PER_REQUEST=k``), so a kernel directory is only valid for the
cap value it was compiled with. Two arms that share a ``--kernel-dir`` under
``--reuse-kernel`` run ONE binary and differ only in the CPU-side admission
replay -- the trap that made M3-I9 under-report the win and M3-I7's first pass
report the arms as identical. Never share a kernel dir across cap values.

Its default in ``persistent_kernel.cuh`` is ``MPK_MAX_NUM_BATCHED_TOKENS``, i.e.
uncapped: without the cap, ``prepare_next_batch`` step 3 hands a prefilling
request ``min(remaining, mbt - num_tokens)`` -- the whole remaining budget.

WHY IT IS ON (the mechanism, measured; M3-I7 gate + M4-I4)
----------------------------------------------------------
Uncapped, the lowest live slot takes the whole ``mbt`` budget, so requests
prefill almost one at a time. That costs in two separable ways:

* **iteration count** -- a request that finishes prefill early starts decoding
  and eats budget the later prefills needed, so the wave takes more iterations to
  push the same tokens. This term is large only where ``mbt`` is the binding
  constraint: at bs16 the admission replay counts 1887 iterations uncapped
  against 1279 capped (1.48x of a measured 1.83x), while at bs<=8 the two counts
  differ by under 4% -- at bs8/msl=132 they are identical and the cap is still
  1.17x faster.
* **graph width per iteration** -- one request contributing a 16-token chunk
  produces a NARROWER task graph than four requests contributing 4 tokens each,
  and MPK's cost is set by the widest per-slot chunk rather than the token total
  (``opt/m3i9/cost_model.py``). This term is what pays at bs2/bs4/bs8, where the
  iteration counts are essentially equal, and it is the whole reason the cap helps
  below bs16 at all.

Both terms are prefill-side, which is why the decode-side argument M3-I9 used to
exclude bs<16 ("the cap only changes prefill chunk boundaries") did not settle
the question. The mechanism recorded before M4-I4 -- "uncapped admission
serialises prefill, 1887 iterations against 1279" -- describes only the first
term, i.e. only bs16.

THE POLICY
----------
``auto`` at every batch size from ``CAP_MIN_BATCH_SIZE`` up:

    cap = auto = max(1, mbt // batch_size)

At ``batch_size == 1`` that is ``mbt`` itself, so the extra ``min`` can never
fire and a capped bs1 build is semantically identical to an uncapped one -- the
policy is a no-op there by construction, not by exception. ``CAP_MIN_BATCH_SIZE``
is the smallest batch size where the cap is MEASURED to pay; it moves only on
evidence, and the measurement that set its current value is cited below.

Evidence: ``demo/qwen3_5/accept/opt/m4i4/README.md`` (A/B at both geometries,
per-rep, arms interleaved in one GPU claim, per-arm kernels) and
``opt/m3i7/tables/cap_policy.json``.
"""
from __future__ import annotations

from typing import Optional, Union

# ---------------------------------------------------------------------------
# THE POLICY. Change it here and nowhere else.
# ---------------------------------------------------------------------------

#: The cap mode the shipped policy uses where it applies. ``"auto"`` is
#: ``max(1, mbt // batch_size)`` -- an equal share of the iteration's token
#: budget per live request, which is both the packing optimum and the widest
#: task graph the budget allows.
CAP_MODE = "auto"

#: Smallest batch size the policy caps -- MEASURED (M4-I4, both geometries, 3
#: reps, arms interleaved in one GPU claim): at the pinned 256/1024 workload the
#: cap makes prefill 1.283x / 1.447x / 1.727x / 2.714x faster at bs 2 / 4 / 8 / 16
#: for +1.4% / +4.0% / +14.1% / +83.4% e2e, with non-overlapping per-rep sets at
#: every one of those batch sizes. bs1 is excluded because ``auto`` there equals
#: ``mbt``, so the cap provably cannot bind -- measured at exactly 1.000x, medians
#: identical to 0.1 ms. Moving this constant is a policy change and needs its own
#: A/B at both geometries plus the AC-3 gate.
CAP_MIN_BATCH_SIZE = 2

#: Batch sizes the pinned benchmark protocol runs (AC-3/AC-4/AC-5).
PROTOCOL_BATCH_SIZES = (1, 2, 4, 8, 16)

#: What ``--per-request-token-cap`` accepts, and what each value means.
#:   "policy" -- use this module's policy for the run's batch size (the default)
#:   "none"   -- force uncapped; reproduces every pre-policy artifact
#:   "auto"   -- force max(1, mbt // bs) even where the policy would not cap
#:   <int>    -- force an explicit cap (experiment arms)
CAP_CHOICES_DOC = "policy | none | auto | <positive int>"

CapRequest = Union[str, int, None]
CapResolved = Union[str, int, None]


def auto_cap(mbt: int, batch_size: int) -> int:
    """``auto``'s arithmetic: an equal share of the iteration budget."""
    if mbt < 1 or batch_size < 1:
        raise ValueError(f"auto_cap(mbt={mbt}, batch_size={batch_size}): both >= 1")
    return max(1, mbt // batch_size)


def policy_cap(batch_size: int) -> Optional[str]:
    """The shipped policy for one batch size: ``"auto"`` or ``None`` (uncapped)."""
    if batch_size < 1:
        raise ValueError(f"policy_cap(batch_size={batch_size}): >= 1")
    return CAP_MODE if batch_size >= CAP_MIN_BATCH_SIZE else None


def validate(requested: CapRequest) -> CapRequest:
    """Accept a ``--per-request-token-cap`` value, or raise.

    Returns the request NORMALISED (``"none"`` -> ``None``, numeric string ->
    ``int``) but NOT resolved: resolution needs the batch size, which the CLI
    does not always know at parse time.
    """
    if requested is None or requested == "none":
        return None
    if requested in ("policy", "auto"):
        return requested
    if isinstance(requested, bool):                       # bool is an int subclass
        raise ValueError(f"per_request_token_cap={requested!r}: {CAP_CHOICES_DOC}")
    if isinstance(requested, int):
        if requested < 1:
            raise ValueError(f"per_request_token_cap={requested!r}: {CAP_CHOICES_DOC}")
        return requested
    if isinstance(requested, str):
        try:
            value = int(requested)
        except ValueError:
            raise ValueError(
                f"per_request_token_cap={requested!r}: {CAP_CHOICES_DOC}") from None
        if value < 1:
            raise ValueError(f"per_request_token_cap={requested!r}: {CAP_CHOICES_DOC}")
        return value
    raise ValueError(f"per_request_token_cap={requested!r}: {CAP_CHOICES_DOC}")


def resolve(requested: CapRequest, batch_size: int) -> CapResolved:
    """Turn a validated request into ``None`` (uncapped) / ``"auto"`` / an int.

    ``"policy"`` is the only value that depends on ``batch_size``.
    """
    req = validate(requested)
    if req == "policy":
        return policy_cap(batch_size)
    return req


def resolve_int(requested: CapRequest, mbt: int, batch_size: int) -> Optional[int]:
    """The compile-time value to pass as ``max_tokens_per_request``, or ``None``."""
    res = resolve(requested, batch_size)
    if res is None:
        return None
    if res == "auto":
        return auto_cap(mbt, batch_size)
    return int(res)


def describe(requested: CapRequest, mbt: int, batch_size: int) -> str:
    """One log line: what was asked for, what the policy resolved it to, and why."""
    req = validate(requested)
    value = resolve_int(req, mbt, batch_size)
    if value is None:
        why = ("policy: uncapped below bs%d" % CAP_MIN_BATCH_SIZE
               if req == "policy" else "explicitly uncapped")
        return (f"admission cap: OFF (bs={batch_size}, mbt={mbt}) "
                f"[requested={req if req is not None else 'none'}; {why}]")
    noop = " (== mbt, semantically a no-op)" if value >= mbt else ""
    return (f"admission cap: MPK_MAX_TOKENS_PER_REQUEST={value}{noop} "
            f"(bs={batch_size}, mbt={mbt}) "
            f"[requested={req}; {'shipped policy' if req == 'policy' else 'forced'}]")


def summary() -> dict:
    """Machine-readable snapshot of the policy, for run metadata."""
    return {
        "cap_mode": CAP_MODE,
        "cap_min_batch_size": CAP_MIN_BATCH_SIZE,
        "per_bs": {bs: policy_cap(bs) for bs in PROTOCOL_BATCH_SIZES},
        "authority": "demo/qwen3_5/accept/admission_policy.py",
    }


if __name__ == "__main__":
    # Bare invocation prints ONLY the machine-readable summary, so drivers can
    # embed it in their run_meta.json (harness/gate_ac3_stable.sh does).
    import json
    import sys as _sys
    if "--describe" in _sys.argv[1:]:
        mbt = 16
        for _bs in PROTOCOL_BATCH_SIZES:
            print(describe("policy", mbt, _bs))
    else:
        print(json.dumps(summary()))
