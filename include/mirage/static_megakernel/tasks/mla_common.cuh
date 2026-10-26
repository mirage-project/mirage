// tasks/mla_common.cuh -- what the Kimi K3 MLA tasks (mla_norm, mla_qb,
// mla_absorb, mla_attention) share. Their polls load all of a thread's 16-B
// vectors together each round and spin without a sleep (10-11 A/B, layer 47, 37
// KV rows, 2 runs each: no sleep 39.1 / 39.1 us vs a 256-ns sleep 43.5 / 45.0;
// one vector at a time 41.5 / 41.3 vs together 39.8 / 41.0). Keep a thread's
// polled vectors few: 18 of them in registers spilled the whole kernel (10-11:
// 37 -> 58 us).
#pragma once
#include "../core.cuh"
// (a dry pass of mla_attention's chunk + tail code before the query, to fetch
// its instructions early, LOST 10-12: 39.2 vs 38.2 us at 37 rows; the query was
// seen 3.7 us later. The front (fused_a's ~10 us) is HBM-bound: ~60 MB of
// weights stream in its window at ~6.3 TB/s; moving g_proj off it (plan g_late)
// or the gate poll to the attention's end did not help.)
