"""Grids for the GLM-5.3-Flash test graphs: every gemm_tile node gets the candidate grid (compiler.candidate_grids) with
the most tasks up to `cap` (ties: the fewer K parts), every other node its layer default. MI355X 11-07 glmdsa4 / glmdense1:
the layer defaults (one 128-row task per block, one K part) left the projections on 3 .. 48 CUs (e.g. [wk | gate |
weights_proj] 3 tasks, 33 us at T 1).

    grids = wide_gemm_grids(mpk, cap=256)   ->   simple_plan(mpk, grids)
"""
from mirage.mpk import compiler


def wide_gemm_grids(pk, cap: int = 256, only=None, act_aware=None, max_kparts: int = 14) -> dict:
    """act_aware (default on; MPK_GLM_ACTGRID=0 off): the grid with the fewest bytes per CU - a task of R rows and K part KP reads R KP
    2 B of weights plus T KP 2 B of activation (every task loads its whole activation slice into LDS), ceil(tasks / 256) tasks per
    CU; ties: more tasks, then fewer K parts. K parts > 1 only for "slots" GEMMs (mode 0: the reader adds the parts). MI355X 11-07
    glmkda5: the KDA projection at T 64 on (204, 1) - every 32-row task loaded the 512 KB activation, 104 MB per launch, ~30 us
    before its first MFMA; glmkda8: (51, 4) T 64 130.2 vs 150.2 us, T 32 94.4 vs 98.6). Applied at T >= 32 only; glmgrid2 / 3: dense T 64 100.6 vs 118.5, DSA T 64 188.7 vs 198.7"""
    import os
    if act_aware is None:
        act_aware = os.environ.get("MPK_GLM_ACTGRID", "1") == "1"   # (default on, glmgrid2 / 3; MPK_GLM_ACTGRID=0: off)
    out = {}
    for n in compiler.graph_nodes(pk):
        if n.name != "gemm_tile" or (only is not None and n.graph_idx not in only):
            continue
        cands = [tuple(g) for g in compiler.candidate_grids(n)]
        # (at most max_kparts K parts: gemm_tile's finish paths take at most 14 - GLM-5.3's H 6144 offers up to 48)
        fit = [g for g in cands if g[0] * g[1] * g[2] <= cap and g[1] <= max_kparts]
        if len(n.params) > 4 and n.params[4] == 2:   # (GEMM_FP8: each wave's K columns whole 128-column scale blocks)
            ksplit = lambda g: 1 if n.params[2] // g[0] >= 64 else 4 // (n.params[2] // g[0] // 16)
            fit = [g for g in fit if (n.params[1] // g[1]) % (128 * ksplit(g)) == 0]
        if not fit:
            continue
        T = compiler.dims(n.inputs[0])[0]
        pks = int(os.environ.get("MPK_PA_KSPLIT", "0"))   # (push_all in K parts: tasks/gemm_tile.cuh, each part its own slot)
        if pks and n.params[0] == 6 and T >= int(os.environ.get("MPK_PA_KSPLIT_MIN_T", "32")):
            fk = [g for g in fit if g[1] == pks]
            if fk:
                out[n.graph_idx] = max(fk, key=lambda g: g[0] * g[1] * g[2])
                continue
        if act_aware and T >= 32:   # (T 8: (51, 4) 67.2 vs (204, 1) 64.3 us - the reader's polls of the K parts, glmkda8)
            mode, K, N = n.params[0], n.params[1], n.params[2]
            # (at most 8 K parts: gemm_tile's finish paths take at most 14 - glmgrid2: DSA's [q_a | kv_a] at 16 / 32 parts)
            fit = [g for g in fit if (g[1] == 1 or mode == 0) and g[1] <= 8] or fit
            wb = 1 if len(n.params) > 4 and n.params[4] == 2 else 2   # (the weight's bytes per element: GEMM_FP8 1)
            cost = lambda g: -(-(g[0] * g[1] * g[2]) // 256) * ((N // g[0]) * (K // g[1]) * wb + T * (K // g[1]) * 2)
            out[n.graph_idx] = min(fit, key=lambda g: (cost(g), -(g[0] * g[1] * g[2]), g[1]))
        else:
            out[n.graph_idx] = max(fit, key=lambda g: (g[0] * g[1] * g[2], -g[1]))
    return out
