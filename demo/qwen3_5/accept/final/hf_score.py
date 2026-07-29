#!/usr/bin/env python3
"""Collect the AC-3(a) coherence INPUTS that need the HF reference model:
the decoded text of every continuation and its teacher-forced perplexity under
the same HF model / same pinned revision the AC-3 reference was generated with.

This is the only part of the AC-3 path that needs a GPU, and it deliberately
produces DATA rather than verdicts: the verdicts are computed by the pure
``score_ac3.py`` / ``ac3_criteria.py``, which are unit-tested without a GPU.

DEFINITIONS
-----------
For a prompt with reference ``input_ids`` P and a continuation C (|C| = 64):

    nll_sum = -sum_i log softmax(logits(P + C)[|P| + i - 1])[C_i]
    ppl     = exp(nll_sum / |C|)

i.e. plain teacher-forced perplexity of the continuation given the prompt, the
same quantity for the engine's continuation and for the reference's own
continuation, computed by the same model in the same process.  AC-3(a) compares
them as a ratio, so any constant offset from the model or its quantization
cancels.

* The model is loaded exactly the way ``reference/generate_reference.py`` loads
  it (same auto-class ladder, ``dtype="auto"``, ``device_map="cuda:0"``, pinned
  ``--revision``), so the "HF reference model" in AC-3(a) is the same object that
  produced the reference.
* log-softmax is taken in float32 for numerical headroom; the model's own dtype
  is untouched.
* Continuations are DEDUPLICATED by content hash: identical token sequences are
  scored once.  That is exact, not an approximation, and it is what keeps the
  stage cheap when 3 reps x 5 batch sizes mostly agree.

OUTPUT (``coherence_inputs.json``)
    {"reference": {pid: {ppl, nll_sum, n, text, token_ids_sha256}},
     "engine":   [{batch_size, prompt_id, token_ids_sha256, reps: [...], ppl,
                   nll_sum, n, text}], ...}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "final/coherence_inputs/v1"


def sha_ids(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def load_model(model_id: str, revision: str, model_path):
    """Same ladder as reference/generate_reference.py:load_model_and_tokenizer."""
    from transformers import AutoTokenizer
    src = model_path or model_id
    tok = AutoTokenizer.from_pretrained(src, revision=None if model_path else revision)
    last = None
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                 "Qwen3_5MoeForConditionalGeneration"):
        try:
            import transformers
            cls = getattr(transformers, name)
            kw = {"dtype": "auto", "device_map": "cuda:0"}
            if not model_path:
                kw["revision"] = revision
            m = cls.from_pretrained(src, **kw)
            print(f"[hf] loaded via {name}", flush=True)
            return m, tok, name
        except Exception as e:                            # noqa: BLE001
            last = f"{name}: {type(e).__name__}: {e}"
            print(f"[hf] {last}", flush=True)
    raise SystemExit(f"INTEGRITY: could not load the HF reference model ({last})")


def score_one(model, prompt_ids, cont_ids) -> tuple:
    import torch

    ids = torch.tensor([list(prompt_ids) + list(cont_ids)], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    p = len(prompt_ids)
    idx = torch.arange(p - 1, p - 1 + len(cont_ids), device=ids.device)
    tgt = torch.tensor(list(cont_ids), device=ids.device)
    nll = -lp[idx, tgt]
    s = float(nll.sum().item())
    return s, float(torch.exp(torch.tensor(s / len(cont_ids))).item())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reps-root", required=True,
                    help="the cold-rep tree from gate_ac3_stable.sh")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--expect-new-tokens", type=int, default=64)
    ap.add_argument("--output-json", required=True)
    a = ap.parse_args(argv)

    ref_doc = json.loads(Path(a.reference).read_text())
    ref = ref_doc["results"]
    batch_sizes = [int(x) for x in a.batch_sizes.split(",") if x.strip()]

    # ---- gather every distinct (bs, prompt, continuation) from the reps ----
    wanted: dict = {}
    reps_root = Path(a.reps_root)
    for d in sorted(reps_root.glob("bs*_r*")):
        try:
            bs = int(d.name.split("_r", 1)[0][2:])
        except (IndexError, ValueError):
            continue
        if bs not in batch_sizes:
            continue
        dump = d / f"bs{bs}.json"
        if not dump.exists():
            continue
        try:
            got = json.loads(dump.read_text())
        except Exception:                                 # noqa: BLE001
            continue
        for pid, e in got.items():
            ids = e.get("token_ids")
            if not ids or pid not in ref:
                continue
            key = (bs, pid, sha_ids(ids))
            w = wanted.setdefault(key, {"batch_size": bs, "prompt_id": pid,
                                        "token_ids_sha256": key[2],
                                        "token_ids": list(ids), "reps": []})
            w["reps"].append(d.name)
    print(f"[hf] {len(wanted)} distinct (bs, prompt, continuation) to score "
          f"across {len(batch_sizes)} batch size(s)", flush=True)

    model, tok, via = load_model(a.model, a.revision, a.model_path)
    vocab_len = len(tok)

    out = {"schema": SCHEMA, "model_id": a.model, "revision": a.revision,
           "loaded_via": via, "tokenizer_vocab_len": vocab_len,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reps_root": str(reps_root),
           "definition": "teacher-forced continuation perplexity given the "
                         "reference input_ids; log-softmax in float32",
           "reference": {}, "engine": []}
    try:
        out["git_sha"] = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:                                     # noqa: BLE001
        out["git_sha"] = None
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": p.name, "uuid": str(getattr(p, "uuid", "") or ""),
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except Exception:                                     # noqa: BLE001
        out["device"] = None

    # ---- the reference continuations (the per-prompt bar) ----------------
    for pid in sorted(ref):
        e = ref[pid]
        cont = e["output_ids"][:a.expect_new_tokens]
        nll, ppl = score_one(model, e["input_ids"], cont)
        out["reference"][pid] = {
            "n": len(cont), "nll_sum": nll, "ppl": ppl,
            "token_ids_sha256": sha_ids(cont),
            "text": tok.decode(cont, skip_special_tokens=False)}
        print(f"[hf] ref  {pid:<14} ppl={ppl:9.4f}", flush=True)

    # ---- the engine continuations ---------------------------------------
    for key in sorted(wanted, key=lambda k: (k[0], k[1])):
        w = wanted[key]
        pid = w["prompt_id"]
        nll, ppl = score_one(model, ref[pid]["input_ids"], w["token_ids"])
        bad_ids = [t for t in w["token_ids"] if not (0 <= t < vocab_len)]
        out["engine"].append({
            "batch_size": w["batch_size"], "prompt_id": pid,
            "token_ids_sha256": w["token_ids_sha256"], "reps": sorted(w["reps"]),
            "n": len(w["token_ids"]), "nll_sum": nll, "ppl": ppl,
            "invalid_token_ids": bad_ids,
            "text": tok.decode(w["token_ids"], skip_special_tokens=False)})
        rp = out["reference"][pid]["ppl"]
        print(f"[hf] bs{w['batch_size']:<3} {pid:<14} ppl={ppl:9.4f} "
              f"ratio={ppl / rp:6.3f} reps={len(w['reps'])}", flush=True)

    p = Path(a.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"[hf] wrote {len(out['engine'])} engine + {len(out['reference'])} "
          f"reference records -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
