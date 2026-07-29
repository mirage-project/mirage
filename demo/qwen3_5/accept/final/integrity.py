#!/usr/bin/env python3
"""Stage 0 of the final gate: INTEGRITY.  Runs first, fails fast, exit 1 on any
violation.  Pure stdlib; needs no GPU (``tests/test_integrity.py``).

WHAT IT PROVES BEFORE ANY MEASUREMENT HAPPENS
---------------------------------------------
1. The invocation is the PINNED one.  ``.pm/accept.sh`` is the authority for
   the model id, the batch-size set, the prompt file, the 64-token horizon, the
   workload floors and the 1.25x AC-5 factor; this stage re-derives those values
   BY PARSING ``.pm/accept.sh`` ITSELF and refuses if the flags it was handed
   disagree.  A weakened caller therefore cannot smuggle looser bounds past the
   gate: the gate reads the pinned file, not the caller's word for it.
2. The pinned prompt set is untampered -- sha256 recomputed here against the
   digest recorded in ``.pm/accept.sh``, independently of accept.sh's own check.
3. The AC-3 reference artifact is the pinned one (model id + revision + 10
   prompts x 64 tokens + real top-k logit data) and its sha256 is recorded.
4. The exactness-diagnostic baseline (``results/dumps_final``) is present and
   complete for every batch size, with per-case digests recorded.
5. The vLLM baseline table is present, binding-valid, and its identity is
   recorded (AC-4's comparator identity).
6. The pinned workload satisfies accept.sh's own floors (input >= MIN_INPUT_LEN,
   output >= MIN_OUTPUT_LEN) -- the mechanical link between the goal's bounds
   and the protocol's pinned 256/1024 choice.
7. The tree is CLEAN and its commit sha, branch and the sha256 of every tool
   file the gate will execute are recorded (AC-6: "from a clean tree").
8. The sha256 of ``.pm/goal.md``, ``.pm/accept.sh``, ``.pm/immutable`` are
   recorded, so a moved goal is visible in the artifact rather than inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "final/integrity/v1"

TOOL_FILES = (
    "final.sh",
    "final/integrity.py", "final/ac3_criteria.py", "final/score_ac3.py",
    "final/score_perf.py", "final/report.py", "final/hf_score.py",
    "final/collect_ac3.sh", "final/collect_perf.sh", "final/remote_setup.sh",
    "final/mechanisms.json",
    "harness/gate_ac3_stable.sh", "harness/gate_ac3_stable.py",
    "mpk_engine_run.py", "bench_vllm.py", "admission_policy.py",
    "opt/m3i7/scripts/make_matched_reference.py",
    "opt/m3i9/make_synthetic_prompts.py",
    "opt/m3i7/scripts/gpu_guard_i7.sh",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_accept_sh(path: Path) -> dict:
    """Pull the pinned constants out of ``.pm/accept.sh``.

    Deliberately reads the PINNED file rather than trusting the flags this
    process was handed.  Any shape change in accept.sh shows up here as a
    missing key and fails the gate closed, which is the right outcome: the gate
    must not guess what the pinned contract says.
    """
    txt = path.read_text()
    want = {"MODEL_ID": str, "BATCH_SIZES": str, "PROMPTS": str,
            "PROMPTS_SHA": str, "CORRECT_NEW_TOKENS": int, "MIN_INPUT_LEN": int,
            "MIN_OUTPUT_LEN": int, "E2E_FACTOR_MAX": float, "BASELINE": str,
            "HARNESS": str}
    out = {}
    for key, cast in want.items():
        m = re.search(rf'^{key}=(?:"([^"]*)"|(\S+))', txt, re.M)
        if not m:
            raise SystemExit(f"INTEGRITY: {path} does not define {key} -- the gate "
                             f"refuses to guess the pinned contract")
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        out[key] = cast(raw)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-dir", required=True, help="workspace/demo/qwen3_5/accept")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--agent-root", default=None,
                    help="the directory holding .pm/ (required for a BINDING run)")
    ap.add_argument("--baseline-dir", required=True,
                    help="baselines/vllm-<ver>-<date> (the pinned vLLM table)")
    ap.add_argument("--bench-vllm", required=True)
    # what the caller (accept.sh) claims the contract is; cross-checked below
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-sizes", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--correct-new-tokens", type=int, required=True)
    ap.add_argument("--min-input-len", type=int, required=True)
    ap.add_argument("--min-output-len", type=int, required=True)
    ap.add_argument("--e2e-factor-max", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--workload-input-len", type=int, default=256)
    ap.add_argument("--workload-output-len", type=int, default=1024)
    ap.add_argument("--output-json", required=True)
    a = ap.parse_args(argv)

    acc = Path(a.accept_dir).resolve()
    repo = Path(a.repo_root).resolve()
    rep = {"schema": SCHEMA,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "binding": True, "violations": [], "notes": [], "recorded": {}}
    V = rep["violations"]

    # ---- 1/2. the pinned contract, read from .pm/accept.sh --------------
    pinned = None
    if a.agent_root:
        accept_sh = Path(a.agent_root) / ".pm" / "accept.sh"
        if not accept_sh.exists():
            V.append(f"--agent-root given but {accept_sh} is absent")
        else:
            pinned = parse_accept_sh(accept_sh)
            rep["recorded"]["pinned_contract"] = pinned
            claimed = {"MODEL_ID": a.model, "BATCH_SIZES": a.batch_sizes,
                       "PROMPTS": a.prompts,
                       "CORRECT_NEW_TOKENS": a.correct_new_tokens,
                       "MIN_INPUT_LEN": a.min_input_len,
                       "MIN_OUTPUT_LEN": a.min_output_len,
                       "E2E_FACTOR_MAX": float(a.e2e_factor_max),
                       "BASELINE": a.baseline}
            for k, v in claimed.items():
                if pinned[k] != v:
                    V.append(f"invocation disagrees with the pinned {k}: "
                             f"caller {v!r} != .pm/accept.sh {pinned[k]!r}")
            # the harness path accept.sh execs must be THIS file's own gate
            expect_harness = "workspace/demo/qwen3_5/accept/final.sh"
            if pinned["HARNESS"] != expect_harness:
                V.append(f".pm/accept.sh execs {pinned['HARNESS']!r}, this gate lives "
                         f"at {expect_harness!r}")
            for f in (".pm/goal.md", ".pm/accept.sh", ".pm/immutable"):
                p = Path(a.agent_root) / f
                rep["recorded"].setdefault("pm_file_sha256", {})[f] = (
                    sha256_file(p) if p.exists() else None)
            prompts = Path(a.agent_root) / pinned["PROMPTS"]
            if not prompts.exists():
                V.append(f"pinned prompt set missing: {prompts}")
            else:
                got = sha256_file(prompts)
                rep["recorded"]["prompts"] = {"path": str(prompts), "sha256": got,
                                              "expected": pinned["PROMPTS_SHA"]}
                if got != pinned["PROMPTS_SHA"]:
                    V.append(f"prompt set digest mismatch (tampered): {got} != "
                             f"{pinned['PROMPTS_SHA']}")
                ids = [json.loads(line)["id"]
                       for line in prompts.read_text().splitlines() if line.strip()]
                rep["recorded"]["prompts"]["ids"] = ids
                if len(ids) != 10:
                    V.append(f"pinned prompt set has {len(ids)} prompts, expected 10")
    else:
        rep["binding"] = False
        rep["notes"].append("NON-BINDING: no --agent-root, so the pinned contract in "
                            ".pm/accept.sh could not be read or cross-checked")

    # ---- 3. the AC-3 reference artifact ---------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_perf import load_bench_constants          # noqa: E402
    consts = load_bench_constants(Path(a.bench_vllm))
    rep["recorded"]["pinned_bench_constants"] = {
        k: (sorted(v) if isinstance(v, set) else v) for k, v in consts.items()}

    # The admission-cap policy is code with ONE authority
    # (accept/admission_policy.py).  Recorded, never restated here, so the report
    # says which policy the run was scored against.
    try:
        sys.path.insert(0, str(acc))
        import admission_policy                          # noqa: E402
        rep["recorded"]["admission_policy"] = admission_policy.summary()
    except Exception as e:                               # noqa: BLE001
        V.append(f"cannot import the admission-cap authority "
                 f"accept/admission_policy.py: {type(e).__name__}: {e}")

    ref_path = acc / "reference" / "reference_outputs.json"
    if not ref_path.exists():
        V.append(f"AC-3 reference artifact missing: {ref_path}")
    else:
        doc = json.loads(ref_path.read_text())
        meta, res = doc.get("meta") or {}, doc.get("results") or {}
        n_tok = a.correct_new_tokens
        topk_ok = all(len(e.get("topk_ids_per_step") or []) == n_tok
                      and len(e.get("topk_logits_per_step") or []) == n_tok
                      for e in res.values())
        rep["recorded"]["reference"] = {
            "path": str(ref_path), "sha256": sha256_file(ref_path),
            "model_id": meta.get("model_id"), "revision": meta.get("revision"),
            "max_new_tokens": meta.get("max_new_tokens"),
            "greedy": meta.get("greedy"), "n_prompts": len(res),
            "prompt_ids": sorted(res), "topk_present_every_step": topk_ok}
        if meta.get("model_id") != a.model:
            V.append(f"reference model {meta.get('model_id')!r} != pinned {a.model!r}")
        if meta.get("revision") != consts["REVISION_DEFAULT"]:
            V.append(f"reference revision {meta.get('revision')!r} != pinned "
                     f"{consts['REVISION_DEFAULT']!r}")
        if meta.get("greedy") is not True or meta.get("do_sample") is not False:
            V.append("reference was not generated greedily (goal AC-2)")
        if len(res) != 10:
            V.append(f"reference has {len(res)} prompts, expected 10")
        bad = {pid: len(e.get("output_ids") or []) for pid, e in res.items()
               if len(e.get("output_ids") or []) != n_tok
               or e.get("num_generated") != n_tok}
        if bad:
            V.append(f"reference prompts are not exactly {n_tok} generated tokens: {bad}")
        if not topk_ok:
            V.append("reference lacks per-step top-k logits at every position -- "
                     "AC-3(b)'s near-tie test cannot be evaluated without them")
        if "prompts" in rep["recorded"] and sorted(res) != sorted(
                rep["recorded"]["prompts"].get("ids") or []):
            V.append("reference prompt ids differ from the pinned prompt set's ids")

    # ---- 4. the exactness-diagnostic baseline ---------------------------
    dumps = acc / "results" / "dumps_final"
    bss = [int(x) for x in a.batch_sizes.split()] if " " in a.batch_sizes else \
          [int(x) for x in a.batch_sizes.split(",") if x.strip()]
    rep["recorded"]["batch_sizes"] = bss
    dd = {}
    for bs in bss:
        p = dumps / f"bs{bs}.json"
        if not p.exists():
            V.append(f"exactness baseline missing: {p} (AC-3(c) requires the "
                     f"diagnostic to be computable)")
            continue
        d = json.loads(p.read_text())
        dd[str(bs)] = {"sha256": sha256_file(p), "n_cases": len(d),
                       "per_case_md5": {k: hashlib.md5(
                           json.dumps(v["token_ids"]).encode()).hexdigest()
                           for k, v in sorted(d.items())}}
        if len(d) != 10:
            V.append(f"{p} holds {len(d)} cases, expected 10")
    rep["recorded"]["exactness_baseline"] = dd

    # ---- 5. the pinned vLLM comparator ---------------------------------
    bdir = Path(a.baseline_dir)
    if not bdir.exists():
        V.append(f"pinned vLLM baseline directory missing: {bdir}")
    else:
        sm_path = bdir / "full" / "summary.json"
        info = {"dir": str(bdir)}
        if sm_path.exists():
            sm = (json.loads(sm_path.read_text()).get("shared_meta") or {})
            cli = sm.get("cli_args") or {}
            info.update({"vllm_version": (sm.get("versions") or {}).get("vllm"),
                         "model_id": sm.get("model_id"), "revision": sm.get("revision"),
                         "input_len": cli.get("input_len"),
                         "output_len": cli.get("output_len"),
                         "language_model_only": cli.get("language_model_only"),
                         "reps": cli.get("reps")})
            if cli.get("input_len") != a.workload_input_len or \
                    cli.get("output_len") != a.workload_output_len:
                V.append(f"pinned baseline workload {cli.get('input_len')}/"
                         f"{cli.get('output_len')} != the gate's pinned workload "
                         f"{a.workload_input_len}/{a.workload_output_len}")
        else:
            V.append(f"pinned baseline summary missing: {sm_path}")
        merged = {}
        for bs in bss:
            mp = bdir / f"bs{bs}.merged.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                merged[str(bs)] = {"binding_valid": m.get("binding_valid"),
                                   "decode": m.get("decode_tokens_per_second_median"),
                                   "e2e": m.get("e2e_wall_seconds_median")}
                if not m.get("binding_valid"):
                    V.append(f"{mp.name}: pinned capture is not binding-valid")
            elif (bdir / "full" / f"bs{bs}.json").exists():
                merged[str(bs)] = {"source": "full", "binding_valid": None}
            else:
                V.append(f"pinned baseline has no data for bs{bs}")
        info["per_bs"] = merged
        rep["recorded"]["vllm_pinned_baseline"] = info

    # ---- 6. the workload satisfies accept.sh's floors -------------------
    rep["recorded"]["workload"] = {"input_len": a.workload_input_len,
                                   "output_len": a.workload_output_len,
                                   "min_input_len": a.min_input_len,
                                   "min_output_len": a.min_output_len,
                                   "source": "docs/qwen35/bench-protocol.md 2 "
                                             "(pinned 256/1024)"}
    if a.workload_input_len < a.min_input_len:
        V.append(f"pinned workload input_len {a.workload_input_len} < accept.sh's "
                 f"MIN_INPUT_LEN {a.min_input_len}")
    if a.workload_output_len < a.min_output_len:
        V.append(f"pinned workload output_len {a.workload_output_len} < accept.sh's "
                 f"MIN_OUTPUT_LEN {a.min_output_len}")
    if a.workload_input_len != consts["BINDING_INPUT_LEN"] or \
            a.workload_output_len != consts["BINDING_OUTPUT_LEN"]:
        V.append(f"the gate's workload {a.workload_input_len}/{a.workload_output_len} "
                 f"!= bench_vllm.py's pinned "
                 f"{consts['BINDING_INPUT_LEN']}/{consts['BINDING_OUTPUT_LEN']}")

    # ---- 7. clean tree + tool provenance -------------------------------
    def git(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, timeout=120)
        return r.stdout.strip(), r.returncode

    porcelain, rc = git("status", "--porcelain")
    sha, _ = git("rev-parse", "HEAD")
    branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    rep["recorded"]["git"] = {"repo": str(repo), "sha": sha, "branch": branch,
                              "clean": (rc == 0 and porcelain == ""),
                              "dirty_paths": [l for l in porcelain.splitlines()][:50]}
    if rc != 0:
        V.append(f"git status failed in {repo}")
    elif porcelain:
        V.append(f"tree is NOT clean ({len(porcelain.splitlines())} path(s)) -- AC-6 "
                 f"requires a clean tree; first: {porcelain.splitlines()[0]!r}")

    prov = {}
    for rel in TOOL_FILES:
        p = acc / rel
        prov[rel] = sha256_file(p) if p.exists() else None
        if prov[rel] is None:
            V.append(f"gate tool file missing: {p}")
    rep["recorded"]["tool_sha256"] = prov

    rep["verdict"] = "PASS" if not V else "FAIL"
    out = Path(a.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))

    print(f"=== INTEGRITY {rep['verdict']}"
          f"{' (NON-BINDING)' if not rep['binding'] else ''} ===")
    print(f"  repo {sha[:12]} branch {branch} clean="
          f"{rep['recorded']['git']['clean']}")
    for n in rep["notes"]:
        print(f"  note: {n}")
    for v in V:
        print(f"  VIOLATION {v}")
    print(f"  report -> {out}")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
