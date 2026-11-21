"""fill_tables.py (11-20 night, DSV41_FILL_FUSED=1): StepTables.fill in one kernel. The first static layer of a step copies
SGLang's metadata of the step into the static layers' tables: positions, the window's rows and output location, the request
slots, and per compress ratio the page table (ratio 1: each page of 256 rows as two entries of 128: 2 p, 2 p + 1). As torch
ops that is 6 or more small kernels before attention L0's kernel in the graph; here one program per row does all of them
(the same values: plain copies and the same integer arithmetic). A step of n < T rows (11-21, DSV41_STATIC_ROWS): rows n ..
T - 1 get what SGLang puts in a padding row of a CUDA graph (position 0, request slot 0, KV slot 0, the window's rows 0 then
-1, the page table's row 0s)."""
import triton
import triton.language as tl

MAXP = 4   # page tables (compress ratios) a launch


@triton.jit
def _fill(pos_in, pos_out, wr_in, wr_in_s, wr_out, W: tl.constexpr, WP: tl.constexpr, wl_in, wl_out, rs_in, rs_out,
          pt, pt_s, ptw,
          p0, m0, x0, p0s, p1, m1, x1, p1s, p2, m2, x2, p2s, p3, m3, x3, p3s, n, NP: tl.constexpr, BM: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    live = t < n   # (rows n .. T - 1: SGLang's padding row)
    tl.store(pos_out + t, tl.load(pos_in + t, mask=live, other=0).to(tl.int32))
    tl.store(wl_out + t, tl.load(wl_in + t, mask=live, other=0).to(tl.int32))
    tl.store(rs_out + t, tl.load(rs_in + t, mask=live, other=0).to(tl.int64))
    c = tl.arange(0, WP)
    wr = tl.load(wr_in + t * wr_in_s + c, mask=(c < W) & live, other=0).to(tl.int32)
    wr = tl.where(live, wr, tl.where(c == 0, 0, -1))
    tl.store(wr_out + t * W + c, wr, mask=c < W)
    j = tl.arange(0, BM)
    for k in tl.static_range(NP):
        p = p0 if k == 0 else (p1 if k == 1 else (p2 if k == 2 else p3))
        m = m0 if k == 0 else (m1 if k == 1 else (m2 if k == 2 else m3))
        x = x0 if k == 0 else (x1 if k == 1 else (x2 if k == 2 else x3))
        ps = p0s if k == 0 else (p1s if k == 1 else (p2s if k == 2 else p3s))
        for b in range(0, m, BM):
            e = b + j
            src = tl.where(x != 0, e // 2, e)
            v = tl.load(pt + t * pt_s + src, mask=(e < m) & (src < ptw) & live, other=0).to(tl.int32)
            v = tl.where(x != 0, v * 2 + (e % 2), v)
            tl.store(p + t * ps + e, v, mask=e < m)


def fill(tables, core, positions, req_pool_indices, T: int, n: int = None):
    """StepTables.fill's work in one launch (tables: the StepTables), for a step of n <= T rows (default T)."""
    n = T if n is None else n
    t = tables.t
    wr = core.swa_page_indices
    pt = core.page_table
    assert wr.stride(1) == 1 and pt.stride(1) == 1
    W = t["window_rows"].shape[1]
    outs = []
    for r, pages in tables.pages.items():
        e = 2 * pt.shape[1] if r == 1 else pt.shape[1]
        outs.append((pages, min(e, pages.shape[1]), 1 if r == 1 else 0))
    assert len(outs) <= MAXP
    args = []
    for k in range(MAXP):
        if k < len(outs):
            pg, m, x = outs[k]
            args += [pg, m, x, pg.stride(0)]
        else:
            args += [t["positions"], 0, 0, 0]
    _fill[(T,)](positions.view(-1), t["positions"], wr, wr.stride(0), t["window_rows"], W, triton.next_power_of_2(W),
                core.swa_out_cache_loc, t["window_loc"], req_pool_indices, t["req_slots"], pt, pt.stride(0), pt.shape[1],
                *args, n, NP=len(outs), BM=128)
