# M3-I2b REDUCED A/B (no-profiler mpk_engine timing path, events=0)

3 reps/cell, median of ms_per_decode_step (profile_wave.py's own field).

| bs | base step_us (spread%) | v1 step_us (spread%) | delta% | base tok/s | v1 tok/s | I1 profiled baseline us |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15161 (1.2%) | 11814 (0.8%) | +28.3 | 66.0 | 84.6 | 15264 |
| 2 | 15906 (1.4%) | 12092 (1.7%) | +31.5 | 125.7 | 165.4 | 15648 |
| 4 | 15837 (1.3%) | 12294 (1.1%) | +28.8 | 252.6 | 325.4 | 15645 |
| 8 | 19817 (1.3%) | 16436 (1.1%) | +20.6 | 403.7 | 486.7 | 18618 |
| 16 | 43052 (29.0%) | 36399 (0.2%) | +18.3 | 232.3 | 274.7 | 22005 |

## Per-rep detail + AC-3 token-id consistency

- base bs1: n=3 step_us=[15160.6, 15134.2, 15308.6] tok_s=[66.0, 66.1, 65.3] sha=consistent
- base bs2: n=3 step_us=[15905.9, 16006.3, 15779.2] tok_s=[125.7, 125.0, 126.7] sha=consistent
- base bs4: n=3 step_us=[15762.6, 15971.1, 15837.4] tok_s=[253.8, 250.5, 252.6] sha=consistent
- base bs8: n=3 step_us=[19816.6, 19797.8, 20057.8] tok_s=[403.7, 404.1, 398.8] sha=consistent
- base bs16: n=3 step_us=[42927.2, 55406.8, 43052.3] tok_s=[233.0, 180.5, 232.3] sha=consistent
- v1 bs1: n=3 step_us=[11813.9, 11829.4, 11733.4] tok_s=[84.6, 84.5, 85.2] sha=consistent
- v1 bs2: n=3 step_us=[12077.3, 12278.4, 12092.3] tok_s=[165.6, 162.9, 165.4] sha=consistent
- v1 bs4: n=3 step_us=[12287.0, 12426.5, 12294.3] tok_s=[325.5, 321.9, 325.4] sha=consistent
- v1 bs8: n=3 step_us=[16359.0, 16544.7, 16435.8] tok_s=[489.0, 483.5, 486.7] sha=consistent
- v1 bs16: n=3 step_us=[36391.3, 36459.9, 36399.2] tok_s=[274.8, 274.3, 274.7] sha=consistent

## v1-vs-prediction check (bs1)

- v1 bs1 median step_us = 11813.9 (min 11733.4, max 11829.4)
- predictions.md P5 refined: 11200-12000 us central, 11200-13500 us full plausible range
- in central band (11.2-12.0ms)? True
- in full plausible band (11.2-13.5ms)? True
