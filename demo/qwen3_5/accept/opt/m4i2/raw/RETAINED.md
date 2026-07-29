# Retained raws (too large for the repo)

The profiler event buffers behind `tables/stage_wallspan.txt` are ~990 MB total,
so they live outside the repo:

```
/home/catalyst/mpk-artifacts/m4i2/prof_raw/
    raw_{A,B}_bs{1,16}.npz        profile_wave.py --save-raw event buffers
    meta_{A,B}_bs{1,16}.json      the run metadata each npz needs
    task_names_{A,B}.json         task-type id -> name for that run
    SHA256SUMS.txt                checksums of the four npz
```

`raw/stage/conc_*.json` in this directory is the derived output and is what the
tables are built from, so the tables reproduce with no GPU:

```
python3 scripts/stage_tables.py --stage-dir <dir with conc_*.json> --out <out>
```

To re-derive `conc_*.json` from the retained buffers (also no GPU — pure CPU work
over the npz), from `demo/qwen3_5/accept/opt`:

```
R=/home/catalyst/mpk-artifacts/m4i2/prof_raw
for a in A B; do for bs in 1 16; do
  python3 concurrency.py $R/raw_${a}_bs${bs}.npz $R/meta_${a}_bs${bs}.json \
                         $R/task_names_${a}.json conc_${a}_bs${bs}.json
done; done
```

Arm A = `MPK_FP8_DENSE_BASELINE=1` (slice 128 + the golden path). Arm B = default
(per-shape slices + the ferret v011 fast path).

The e2e A/B's own per-rep records are committed in full (`tables/ab_per_rep.csv`,
`tables/m4i2_tables.json`), and the AC-3 gate report and re-pinned report are in
`raw/ac3/`. Per-rep KV/GDN fingerprints (`fp_*.npz`) and the cold kernel dirs were
transient by design — `gate_ac3_stable.sh` deletes kernel dirs unless
`--keep-kernels` and `/raid` sat at 100% throughout this run; every fingerprint's
`state_sig` is recorded in `raw/ac3/gate_ac3_stable.json`, which is what the
stability verdict rests on.
