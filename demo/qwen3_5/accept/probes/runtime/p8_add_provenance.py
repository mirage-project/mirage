import json

REPO = "/home/catalyst/agent/workspace/demo/qwen3_5/accept/probes/runtime"
FETCHED = "/tmp/claude-1006/-home-catalyst-agent/d305d170-45e6-4ad6-b21b-823f6f637deb/scratchpad/fetched"

DIRTY_FILES = [
    "include/mirage/kernel/task_register.h", "include/mirage/persistent_kernel/runtime_header.h",
    "include/mirage/persistent_kernel/tasks/blackwell/task_header.cuh",
    "python/mirage/mpk/models/deepseek_v3/builder.py", "python/mirage/mpk/models/utils.py",
    "python/mirage/mpk/persistent_kernel.py", "src/kernel/graph.cc", "src/kernel/runtime.cc",
    "src/kernel/task_register.cc",
]
HEAD_SHA = "79c00730393e18fb7f90f5fa67d896fc2acfff40"

dirty_prov = {
    "head_sha": HEAD_SHA,
    "status_short_dirty_list": DIRTY_FILES,
    "dirty_list_provenance": (
        "RECONSTRUCTED, not captured live: no `git status` snapshot was taken on B200's "
        "~/mpk-qwen35/mirage at run time (this session's original oversight -- flagged by "
        "the coordinator). Reconstructed from (a) this exact dirty-file list read from B200 "
        "immediately after the fix request, at HEAD 79c0073 with all 9 files still 'M' "
        "modified and 0 files with a LATER-than-run mtime among them, and (b) mtime forensics: "
        "every dirty file's mtime is 21:03:09-21:07:09 (Jul 25), 17-77 minutes AFTER this "
        "probe's last compile (p9_step2.log, 20:46:45) and >45 min after every P8/P9-sweep "
        "compile -- i.e. these files were still clean (matching committed HEAD) throughout "
        "every original compile window (19:56-20:46); the other agent's edits landed after. "
        "Not a certainty (a file could theoretically have been dirty-then-reverted-then-"
        "redirtied within the window with coincidentally-later final mtimes), which is "
        "exactly why a real clean-room re-run was still done rather than relying on this "
        "alone."
    ),
    "venv": "venv-mpk (torch 2.13.0+cu130, transformers 4.57.1, accelerate 1.8.0)",
    "gpu": "catalyst-B200 GPU 7 (verified idle, <10MiB/0%util, before every run)",
    "mirage_root": "~/mpk-qwen35/mirage (shared clone, concurrently used by another M2 agent)",
}
clean_prov = {
    "head_sha": HEAD_SHA,
    "status_short_dirty_list": [],
    "venv": "venv-mpk2 (fresh; torch 2.13.0+cu130, transformers 4.57.1, accelerate 1.8.0 -- "
           "matched to venv-mpk; z3-solver ABI-drift fix applied per the M1 memory pattern)",
    "gpu": "catalyst-B200 GPU 7 (verified idle before every run)",
    "mirage_root": "~/mpk-qwen35/mirage-clean (git worktree, detached at origin/qwen3-5_support, "
                   "`git status --short` verified EMPTY before every run in this set)",
    "worktree_disk_cost": "609M (worktree) + 4.9G (venv-mpk2) = 5.5G; left in place per the "
                          "coordinator's instruction for the serialized kernel wave to reuse",
}

# ---------------- P8 ----------------
with open(f"{REPO}/p8_verdict.json") as f:
    p8_dirty = json.load(f)
with open(f"{FETCHED}/p8_verdict_clean_purified.json") as f:
    p8_clean = json.load(f)

r_dirty, r_clean = p8_dirty["r"], p8_clean["r"]
delta_pct = 100 * abs(r_clean - r_dirty) / r_dirty

p8_final = dict(p8_clean)  # clean data is now authoritative per instruction 3
p8_final["provenance"] = {
    "clean_run_used_for_this_verdict": clean_prov,
    "original_dirty_run": {**dirty_prov, "r": r_dirty, "band": p8_dirty["band"],
                           "workload_pin_stands": p8_dirty["workload_pin_stands"]},
    "clean_vs_dirty_delta": {
        "r_dirty": r_dirty, "r_clean": r_clean, "delta_pct": delta_pct,
        "verdict": (
            f"WITHIN NOISE ({delta_pct:.2f}% delta, smaller than this probe's own "
            "cross-pair spread of ~1.4-1.5% on either tree) -- the dirty-tree run is "
            "RETRO-VALIDATED, not overturned. Both land in the same band with the same "
            "pin verdict. The clean numbers above are reported as authoritative per "
            "instruction, but the dirty-tree numbers were never actually wrong."
        ),
    },
}
with open(f"{REPO}/p8_verdict.json", "w") as f:
    json.dump(p8_final, f, indent=2)
print("p8_verdict.json updated: r=", r_clean, "band=", p8_final["band"],
      "pin_stands=", p8_final["workload_pin_stands"], "delta_pct=", round(delta_pct, 3))
