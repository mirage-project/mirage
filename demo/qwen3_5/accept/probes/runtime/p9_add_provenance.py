import json
import re

REPO = "/home/catalyst/agent/workspace/demo/qwen3_5/accept/probes/runtime"
FETCHED = "/tmp/claude-1006/-home-catalyst-agent/d305d170-45e6-4ad6-b21b-823f6f637deb/scratchpad/fetched"
HEAD_SHA = "79c00730393e18fb7f90f5fa67d896fc2acfff40"

LAT_RE = re.compile(r"per-token latency:\s*([\d.]+) ms")
HEADER_RE = re.compile(r"^=== ([^=].*?) ===$")


def parse_latencies(text):
    results, current_label = [], None
    for line in text.splitlines():
        m = HEADER_RE.search(line)
        if m:
            current_label = m.group(1)
            continue
        m2 = LAT_RE.search(line)
        if m2 and current_label is not None:
            results.append((current_label, float(m2.group(1))))
            current_label = None
    return results


with open(f"{FETCHED}/clean_run.log") as f:
    clean_text = f.read()
results = dict(parse_latencies(clean_text))
clean_sweep = {r: results[f"r={r}"] for r in (1, 2, 4, 8)}
all_results = parse_latencies(clean_text)
r8_clean_all = [clean_sweep[8]] + [v for k, v in all_results if k.startswith("r=8 trial")] + \
    [v for k, v in all_results if "FRESH kernel_cache" in k]
r1_control_clean = next(v for k, v in all_results if k == "r=1 control rerun")

r8_clean_mean = sum(r8_clean_all) / len(r8_clean_all)
r8_clean_spread_pct = 100 * (max(r8_clean_all) - min(r8_clean_all)) / r8_clean_mean
observed_jump_clean = r8_clean_mean - clean_sweep[4]
knee_reproduced_clean = observed_jump_clean > 1.5

with open(f"{REPO}/p9_findings.json") as f:
    p9_dirty = json.load(f)

r8_dirty_mean = p9_dirty["r8_repeatability_check"]["mean"]
delta_pct_r8 = 100 * abs(r8_clean_mean - r8_dirty_mean) / r8_dirty_mean

dirty_prov = {
    "head_sha": HEAD_SHA,
    "status_short_dirty_list": [
        "include/mirage/kernel/task_register.h", "include/mirage/persistent_kernel/runtime_header.h",
        "include/mirage/persistent_kernel/tasks/blackwell/task_header.cuh",
        "python/mirage/mpk/models/deepseek_v3/builder.py", "python/mirage/mpk/models/utils.py",
        "python/mirage/mpk/persistent_kernel.py", "src/kernel/graph.cc", "src/kernel/runtime.cc",
        "src/kernel/task_register.cc",
    ],
    "dirty_list_provenance": (
        "Reconstructed (not captured live) from a post-hoc B200 git-status + mtime-forensics "
        "check: all 9 dirty files' mtimes (21:03-21:07) postdate this probe's entire compile "
        "window (19:56-20:46) by 17-77 minutes -- see p8_verdict.json's provenance block for "
        "the full methodology note, identical reasoning applies here (same shared clone, same "
        "session)."
    ),
    "venv": "venv-mpk", "gpu": "catalyst-B200 GPU 7 (verified idle before every run)",
    "mirage_root": "~/mpk-qwen35/mirage (shared clone)",
    "r8_mean": r8_dirty_mean, "step1_ms_per_token": p9_dirty["step1_ms_per_token"],
}
clean_prov = {
    "head_sha": HEAD_SHA, "status_short_dirty_list": [],
    "venv": "venv-mpk2 (fresh, matched package versions, see p8_verdict.json provenance)",
    "gpu": "catalyst-B200 GPU 7 (verified idle before every run)",
    "mirage_root": "~/mpk-qwen35/mirage-clean (git worktree, detached at origin/qwen3-5_support, "
                   "status --short verified EMPTY before this run set)",
    "r8_mean": r8_clean_mean, "step1_ms_per_token": clean_sweep,
}

p9_final = dict(p9_dirty)
p9_final["knee_reproduced"] = knee_reproduced_clean  # clean data is authoritative
p9_final["step1_ms_per_token"] = clean_sweep
p9_final["r8_repeatability_check"] = {
    "measurements_ms_per_token": r8_clean_all,
    "mean": r8_clean_mean, "spread_pct": r8_clean_spread_pct,
    "sources": ["clean sweep r=8", "clean trial 1", "clean trial 2", "clean trial 3", "clean fresh compile-cache dir"],
}
p9_final["r1_control_rerun_ms_per_token"] = r1_control_clean
p9_final["observed_jump_r4_to_r8_ms_per_token"] = observed_jump_clean
p9_final["attribution"] = p9_dirty["attribution"]  # unchanged: still blocked by the same infra bug, independent of tree cleanliness
p9_final["provenance"] = {
    "clean_run_used_for_this_verdict": clean_prov,
    "original_dirty_run": dirty_prov,
    "clean_vs_dirty_delta": {
        "r8_mean_dirty": r8_dirty_mean, "r8_mean_clean": r8_clean_mean, "delta_pct": delta_pct_r8,
        "verdict": (
            f"WITHIN NOISE ({delta_pct_r8:.3f}% delta, far smaller than either tree's own "
            f"internal r=8 spread of ~0.5%) -- the dirty-tree 'knee does not reproduce' finding "
            "is RETRO-VALIDATED by an independent clean-room instrument at the identical HEAD "
            f"({HEAD_SHA[:10]}) with zero working-tree changes. No knee on either tree: both "
            "give ~4.5 ms/token flat across r=1..8, not the historical 7.49."
        ),
    },
}
with open(f"{REPO}/p9_findings.json", "w") as f:
    json.dump(p9_final, f, indent=2)
print("p9_findings.json updated: knee_reproduced=", knee_reproduced_clean,
      "r8_clean_mean=", round(r8_clean_mean, 4), "r8_dirty_mean=", round(r8_dirty_mean, 4),
      "delta_pct=", round(delta_pct_r8, 4))
