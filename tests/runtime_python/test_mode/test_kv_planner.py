"""
Unit tests for plan_kv_groups.
"""

from contextlib import contextmanager

import torch

from mirage.mpk.kv_planner import (
    KVEventLog,
    KVSpec,
    KVStream,
    KVUnificationError,
    pages_per_request,
    plan_kv_groups,
    build_kv_cache,
    _resolve_pool_size,
)


@contextmanager
def _raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} was not raised")


def _by_spec(plan):
    out = {}
    for g in plan.groups:
        out.setdefault(g.spec_name, []).append(g)
    return out


class _GptOssCfg:
    """gpt-oss-20b's shape: 24 layers alternating sliding/full, 8 KV heads of 64."""
    num_key_value_heads = 8
    head_dim = 64
    sliding_window = 128
    layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention"
                   for i in range(24)]


def _gpt_oss_plan(page_size):
    """The real model's streams, planned but not allocated -- these tests are
    about page geometry, not about owning a pool."""
    from mirage.mpk.models.gpt_oss.builder import kv_streams
    return plan_kv_groups([s._spec() for s in
                           kv_streams(_GptOssCfg(), page_size=page_size)])


def test_four_streams_at_mixed_compression_share_one_page():
    specs = [
        KVSpec("c4_main", per_entry_bytes=584, layer_ids=(0,),
               compress_ratio=4, preferred_block_size=256),
        KVSpec("c128_main", per_entry_bytes=584, layer_ids=(1,),
               compress_ratio=128, preferred_block_size=256),
        KVSpec("c4_indexer", per_entry_bytes=132, layer_ids=(2,),
               compress_ratio=4, preferred_block_size=256),
        KVSpec("swa", per_entry_bytes=584, layer_ids=(3,),
               window_size=128, preferred_block_size=64),
    ]
    plan = plan_kv_groups(specs)
    assert plan.target_page_bytes == 37376
    got = {g.spec_name: g.block_size for g in plan.groups}
    assert got == {
        "c4_main": 256,      # 64 entries x4 — the page-size anchor
        "c128_main": 8192,   # 64 entries x128, an exact 32x ratio of the page
        "c4_indexer": 1088,  # 272 entries x4, floored to the 64-token tile
        "swa": 64,           # 64 entries x1
    }
    # A tightest fit gives 283 entries = 1132 tokens, and 1132 % 64 = 44.
    pad = {g.spec_name: g.padding_bytes_per_page for g in plan.groups}
    assert pad["c4_main"] == 0 and pad["swa"] == 0 and pad["c128_main"] == 0
    assert pad["c4_indexer"] == 37376 - 272 * 132   # 1472 B, 3.9% of the page
    for g in plan.groups:
        assert g.block_size % 64 == 0, f"{g.spec_name} is not tile-legal"


def test_an_exact_multiple_page_is_lossless_and_tile_safe():
    # A stream whose natural page divides the shared one keeps its block
    # size scaled by that integer: the tile floor removes nothing.
    specs = [
        KVSpec("fat", per_entry_bytes=2048, layer_ids=(0,),
               preferred_block_size=256),                    # anchor, 512 KiB
        KVSpec("thin", per_entry_bytes=512, layer_ids=(1,),
               preferred_block_size=256),                    # 128 KiB, 4x under
    ]
    plan = plan_kv_groups(specs)
    by = {g.spec_name: g for g in plan.groups}
    assert plan.target_page_bytes == 256 * 2048
    assert by["fat"].block_size == 256
    assert by["thin"].block_size == 256 * 4        # scaled, not re-packed
    assert by["fat"].padding_bytes_per_page == 0
    assert by["thin"].padding_bytes_per_page == 0  # exact ratio wastes nothing
    for g in plan.groups:
        assert g.block_size % 64 == 0


def test_block_size_is_floored_to_the_declared_tile():
    # Same stream, two tiles: a bigger tile costs padding.
    def plan_with(tile):
        return plan_kv_groups([
            KVSpec("anchor", per_entry_bytes=584, layer_ids=(0,),
                   compress_ratio=4, preferred_block_size=256),
            KVSpec("idx", per_entry_bytes=132, layer_ids=(1,),
                   compress_ratio=4, preferred_block_size=256,
                   block_size_multiple_of=tile),
        ])

    got = {t: {g.spec_name: g for g in plan_with(t).groups} for t in (16, 64)}
    # 283 entries fit; tile 64 needs groups of 16 -> 272, tile 16 -> 280.
    assert got[64]["idx"].block_size == 1088 and got[64]["idx"].entries_per_page == 272
    assert got[16]["idx"].block_size == 1120 and got[16]["idx"].entries_per_page == 280
    assert got[16]["idx"].padding_bytes_per_page < got[64]["idx"].padding_bytes_per_page
    assert got[64]["idx"].block_size % 64 == 0
    assert got[16]["idx"].block_size % 16 == 0


def test_a_stream_that_cannot_fit_one_tile_is_refused():
    # 8 entries fit, but a 64-token tile needs 64.
    specs = [
        KVSpec("fat", per_entry_bytes=8, layer_ids=(0,),
               preferred_block_size=800),
        KVSpec("thin", per_entry_bytes=800, layer_ids=(1,),
               preferred_block_size=8),
    ]
    with _raises(KVUnificationError):
        plan_kv_groups(specs, target_page_bytes=6400)


def test_an_illegal_anchor_page_size_is_refused_not_floored():
    # 100 is not a multiple of the 64-token tile.
    with _raises(KVUnificationError):
        _gpt_oss_plan(100)
    for legal in (64, 128, 4096):
        assert _gpt_oss_plan(legal).groups[0].block_size == legal


def test_the_report_names_the_anchor_and_flags_a_dead_window():
    # A block whose first recycle lands past the run: the window is inert
    # here, so the report says so without telling the user to shrink it.
    dead = _gpt_oss_plan(4096).describe(max_seq_length=512)
    assert "anchor 'sliding_attention'" in dead
    assert "never recycles here" in dead and "WARNING" in dead
    assert "not a reason to lower the block size" in dead
    # The same plan over a long enough run does recycle, so no warning.
    assert "WARNING" not in _gpt_oss_plan(4096).describe(max_seq_length=131072)
    # At the tile it recycles early, and the report says from which step.
    live = _gpt_oss_plan(64).describe(max_seq_length=512)
    assert "never recycles" not in live and "recycles from step" in live


def test_gpt_oss_real_config_plan():
    # 24 layers alternating sliding/full, 8 KV heads of 64 on both, so the
    # two streams unify onto one page with no padding: 2 groups of 12 slots.
    plan = _gpt_oss_plan(64)
    assert plan.target_page_bytes == 64 * 2 * 8 * 64 * 2
    assert plan.num_slots == 12 and len(plan.groups) == 2
    by = _by_spec(plan)
    assert set(by) == {"sliding_attention", "full_attention"}
    for g in plan.groups:
        assert g.block_size == 64            # both streams keep the page size
        assert g.padding_bytes_per_page == 0
        assert None not in g.layer_ids       # 12 and 12, nothing padded
    # A layer's group is its attention kind, its slot its index within it.
    for layer_id, kind in enumerate(_GptOssCfg.layer_types):
        group_id, slot_id = plan._layer_info(layer_id)
        assert plan.groups[group_id].spec_name == kind
        assert slot_id == layer_id // 2


def test_gpt_oss_groups_carry_the_window():
    plan = _gpt_oss_plan(64)
    windows = {g.spec_name: g.window_size for g in plan.groups}
    assert windows == {"sliding_attention": 128, "full_attention": 0}
    # group_specs is what PersistentKernel actually reads.
    specs = plan.group_specs()
    assert [s.window_size for s in specs] == [g.window_size for g in plan.groups]


def test_pages_per_request_bounded_by_the_window():
    # Full attention holds the whole sequence.
    assert pages_per_request(64, 0, 512) == 8
    # A window holds the window plus its partial pages, and stops growing.
    assert pages_per_request(64, 128, 512) == 3
    assert pages_per_request(64, 128, 8192) == 3
    # Pages are allocated for a batch's last token but recycled against its
    # first, so a wide batch holds one more.
    assert pages_per_request(64, 128, 512, max_num_batched_tokens=8) == 4
    # window=0 and a window bigger than the sequence recycles nothing.
    assert pages_per_request(64, 4096, 512) == 8
    assert pages_per_request(64, 0, 512, 8) == 8


def test_page_id_bytes_is_the_whole_column():
    # A page id is that page at every slot: slots x page bytes
    small = _gpt_oss_plan(64)
    big = _gpt_oss_plan(4096)
    assert small.page_id_bytes == 12 * 128 * 1024
    assert big.page_id_bytes == 64 * small.page_id_bytes
    assert small.budget_bytes(16) == 24 * 1024**2
    assert big.budget_bytes(16) == 1536 * 1024**2


def test_pages_for_budget_rounds_down_and_round_trips():
    plan = _gpt_oss_plan(64)          # 1.5 MiB per page id
    assert plan.pages_for_budget(24 * 1024**2) == 16
    assert plan.pages_for_budget(24 * 1024**2 - 1) == 15   # never over-commit
    assert plan.pages_for_budget(0) == 0
    for n in (1, 7, 100):
        assert plan.pages_for_budget(plan.budget_bytes(n)) == n


def test_resolve_kv_budget_parses_sizes():
    from mirage.mpk.kv_planner import resolve_kv_budget

    assert resolve_kv_budget("24GiB") == 24 * 1024**3
    assert resolve_kv_budget("512MiB") == 512 * 1024**2
    assert resolve_kv_budget("1GB") == 1000**3        # decimal suffix
    assert resolve_kv_budget(25165824) == 25165824    # an int is raw bytes
    # Anything without a unit is refused, fractions included.
    for ambiguous in ("24", 0.6, "60%"):
        with _raises(ValueError):
            resolve_kv_budget(ambiguous)


def test_a_small_budget_lands_under_the_floor():
    plan = _gpt_oss_plan(64)        # 1.5 MiB per page id
    assert plan.pages_needed(1, 512, 8) == 12         # 4 sliding + 8 full
    assert plan.pages_for_budget(64 * 1024**2) == 42
    # resolve_pool_size is what refuses this; see the tests at the bottom,
    # which stub mem_get_info rather than needing a device.
    assert plan.pages_for_budget(8 * 1024**2) == 5    # below the floor


def test_group_size_picks_slots_per_group():
    # Slots per group: pad up when the layer counts are close, otherwise
    # split on a usable gcd. Padding strands (S-k)/S of every page the group
    # holds, so it is only worth it when the counts are near-equal.
    for note, n_a, n_b, slots, groups, padded in [
        ("12 vs 13: close, pad rather than split", 12, 13, 13, (1, 1), 1),
        ("20 vs 4 at 5:1: gcd 4, zero padding", 20, 4, 4, (5, 1), 0),
        ("20 vs 30: gcd 10 beats a min-based 20", 20, 30, 10, (2, 3), 0),
    ]:
        specs = [
            KVSpec("a", per_entry_bytes=512, layer_ids=tuple(range(n_a)),
                   window_size=128, preferred_block_size=64),
            KVSpec("b", per_entry_bytes=512,
                   layer_ids=tuple(range(n_a, n_a + n_b)),
                   preferred_block_size=64),
        ]
        plan = plan_kv_groups(specs)
        by = _by_spec(plan)
        assert plan.num_slots == slots, note
        assert (len(by["a"]), len(by["b"])) == groups, note
        assert sum(g.layer_ids.count(None) for g in plan.groups) == padded, note

def test_allocate_pool_slots_are_per_layer_and_do_not_alias():
    # per_entry_bytes must match what the caller actually stores: an (8, 16)
    # bf16 entry is 256 B. Going through the pool makes the two agree by
    # construction — declaring 64 here would overrun the page budget.
    specs = [
        KVSpec("full", per_entry_bytes=256, layer_ids=(0, 1),
              preferred_block_size=64),
        KVSpec("window", per_entry_bytes=256, layer_ids=(2, 3),
              window_size=128, preferred_block_size=64),
    ]
    plan = plan_kv_groups(specs)
    assert plan.num_slots == 2
    layout = [("kv", (8, 16), torch.bfloat16)]
    pool, views = plan._allocate_pool({"full": layout, "window": layout},
                                     max_num_pages=32, device="cpu")
    by = {g.spec_name: g.group_id for g in plan.groups}
    cache = views[by["full"]]["kv"]
    assert tuple(cache.shape) == (2, 32, 64, 8, 16)
    assert cache.dtype == torch.bfloat16
    # Slots are layers: slicing [slot_id] is what a builder attaches, and two
    # slots must not alias each other.
    cache[0].fill_(1.0)
    cache[1].fill_(2.0)
    assert (cache[0] == 1.0).all()
    assert (cache[1] == 2.0).all()
    # Two streams reading a page the same way get the same bytes; they are
    # told apart by which page ids they hold, not by separate allocations.
    assert views[by["window"]]["kv"].data_ptr() == cache.data_ptr()


def test_allocate_pool_handles_streams_with_different_entry_sizes():
    # Entries of 8 B and 800 B carve the same 6400 B page into 800 and 8
    # slots respectively. One pool serves both; this is the case a single
    # shared entry layout could not express.
    # tile 1: synthetic entry sizes, exercising pool geometry rather than
    # anything a real kernel would accept.
    fat = KVSpec("fat", per_entry_bytes=8, layer_ids=(0,),
                preferred_block_size=64, block_size_multiple_of=1)
    thin = KVSpec("thin", per_entry_bytes=800, layer_ids=(1,),
                 preferred_block_size=8, block_size_multiple_of=1)
    plan = plan_kv_groups([fat, thin], target_page_bytes=6400)
    by = {g.spec_name: g for g in plan.groups}
    assert by["fat"].entries_per_page == 800
    assert by["thin"].entries_per_page == 8

    pool, views = plan._allocate_pool(
        {"fat": [("kv", (4,), torch.bfloat16)],      # 8 B entries
         "thin": [("kv", (400,), torch.bfloat16)]},  # 800 B entries
        max_num_pages=16, device="cpu")
    assert tuple(pool.shape) == (plan.num_slots, 16, 6400)
    assert tuple(views[by["fat"].group_id]["kv"].shape) == \
        (plan.num_slots, 16, 800, 4)
    assert tuple(views[by["thin"].group_id]["kv"].shape) == \
        (plan.num_slots, 16, 8, 400)
    # Same page stride for both, since a page is a page.
    stride = plan.elems_per_page(torch.bfloat16)
    assert views[by["fat"].group_id]["kv"].stride(1) == stride
    assert views[by["thin"].group_id]["kv"].stride(1) == stride

    # A layout claiming more than the page holds is refused, not truncated.
    with _raises(AssertionError):
        plan._allocate_pool(
            {"fat": [("kv", (4,), torch.bfloat16)],
             "thin": [("kv", (4000,), torch.bfloat16)]},
            max_num_pages=16, device="cpu")


def test_allocate_pool_shares_one_allocation_across_streams():
    # Two streams with different entry sizes read the SAME bytes: a page id
    # is owned by one stream at a time, so nothing is stranded. Entries that
    # do not tile the page leave bounded padding, and every view is strided
    # by the whole page rather than by its packed entry span.
    specs = [
        KVSpec("main", per_entry_bytes=584, layer_ids=(0, 1),
               compress_ratio=4, preferred_block_size=256),
        KVSpec("indexer", per_entry_bytes=132, layer_ids=(2, 3),
               compress_ratio=4, preferred_block_size=256),
    ]
    plan = plan_kv_groups(specs)
    assert plan.target_page_bytes == 37376
    by = {g.spec_name: g for g in plan.groups}
    assert by["main"].entries_per_page == 64        # 584 B x 64, no padding
    assert by["indexer"].entries_per_page == 272    # floored to the tile

    pages = 8
    pool, views = plan._allocate_pool(
        {"main": [("kv", (292,), torch.bfloat16)],
         "indexer": [("kv", (66,), torch.bfloat16)]},
        max_num_pages=pages, device="cpu")
    assert tuple(pool.shape) == (plan.num_slots, pages, 37376)
    main = views[by["main"].group_id]["kv"]
    idx = views[by["indexer"].group_id]["kv"]
    assert tuple(main.shape) == (plan.num_slots, pages, 64, 292)
    assert tuple(idx.shape) == (plan.num_slots, pages, 272, 66)
    # Both are views on the one allocation, not copies of it.
    assert main.data_ptr() == idx.data_ptr() == pool.data_ptr()
    # One addressing rule for padded and unpadded streams alike.
    stride = plan.elems_per_page(torch.bfloat16)
    assert stride == 37376 // 2
    assert main.stride(1) == idx.stride(1) == stride
    assert idx.stride(1) > 272 * 66     # strictly wider than packed

    with _raises(KeyError):
        plan._allocate_pool({"main": [("kv", (292,), torch.bfloat16)]},
                           max_num_pages=pages, device="cpu")


def test_allocate_pool_multi_component_page_shares_one_page_id():
    # A GQA stream stores K and V. They are two COMPONENTS of one page, so a
    # single page id covers both — no second page table, no second draw from
    # the free list. 8 kv heads x 64 dim bf16 => 1024 B per token per
    # component, 2048 B for K+V.
    spec = KVSpec("gqa", per_entry_bytes=2048, layer_ids=(0, 1),
                  preferred_block_size=64)
    plan = plan_kv_groups([spec])
    (g,) = plan.groups
    assert plan.target_page_bytes == 64 * 2048
    assert g.entries_per_page == 64

    pool, views = plan._allocate_pool(
        {"gqa": [("k", (8, 64), torch.bfloat16),
                 ("v", (8, 64), torch.bfloat16)]},
        max_num_pages=8, device="cpu")
    k, v = views[g.group_id]["k"], views[g.group_id]["v"]
    # Each component keeps exactly the per-layer cache shape used today.
    assert tuple(k.shape) == tuple(v.shape) == (2, 8, 64, 8, 64)
    # V starts halfway into the page; both are page-strided by the whole page.
    assert v.data_ptr() - pool.data_ptr() == 64 * 1024
    assert k.stride(1) == v.stride(1) == plan.elems_per_page(torch.bfloat16)
    # Writing one component must not disturb the other.
    k.fill_(1.0)
    v.fill_(2.0)
    assert (k == 1.0).all() and (v == 2.0).all()


def test_assert_in_pool_catches_a_detached_copy():
    # A copy keeps the shape, dtype and values; only its storage differs.
    spec = KVSpec("gqa", per_entry_bytes=2048, layer_ids=(0, 1),
                  preferred_block_size=64)
    plan = plan_kv_groups([spec])
    _pool, views = plan._allocate_pool(
        {"gqa": [("k", (8, 64), torch.bfloat16),
                 ("v", (8, 64), torch.bfloat16)]},
        max_num_pages=8, device="cpu")
    view = views[0]["k"][0]
    assert plan._assert_in_pool(view, "k") is view

    copy = view.contiguous()
    assert copy.shape == view.shape and copy.dtype == view.dtype
    assert torch.equal(copy, view)
    with _raises(AssertionError):
        plan._assert_in_pool(copy, "k")


def test_kernel_entry_multiple_constraint():
    spec = KVSpec("x", per_entry_bytes=100, layer_ids=(0,),
                  block_size_multiple_of=16)
    plan = plan_kv_groups([spec], target_page_bytes=4096)
    (g,) = plan.groups
    assert g.entries_per_page == 32  # floor(4096/100)=40 -> down to 32
    assert g.block_size == 32


def test_group_specs_feed_persistent_kernel():
    specs = [
        KVSpec("full", per_entry_bytes=584, layer_ids=(0,),
               preferred_block_size=64),
        KVSpec("sw", per_entry_bytes=584, layer_ids=(1,), window_size=128,
               preferred_block_size=64),
    ]
    plan = plan_kv_groups(specs)
    gs = plan.group_specs()
    assert [g.block_size for g in gs] == [64, 64]


def test_build_meta_tensors():
    specs = [
        KVSpec("full", per_entry_bytes=64, layer_ids=(0, 1),
               preferred_block_size=64),
        KVSpec("sw", per_entry_bytes=64, layer_ids=(2, 3), window_size=128,
               preferred_block_size=64),
    ]
    plan = plan_kv_groups(specs)
    assert len(plan.groups) == 2
    meta = plan.build_meta_tensors(max_num_pages=32, max_seq_length=64,
                                   max_num_batched_requests=4, device="cpu")
    assert set(meta.keys()) == {
        f"paged_kv_{field}_buffer_{g}"
        for g in range(2)
        for field in ("indptr", "indices", "last_page_len")
    }
    for g in range(2):
        assert meta[f"paged_kv_indptr_buffer_{g}"].shape == (5,)
        assert meta[f"paged_kv_last_page_len_buffer_{g}"].shape == (4,)
        assert meta[f"paged_kv_indptr_buffer_{g}"].dtype == torch.int32
        # span = max(pages, requests * ceil(seq/block)) = max(32, 4*1) = 32
        assert meta[f"paged_kv_indices_buffer_{g}"].shape == (32,)
    # With it, by page-table SPAN instead: recycled slots keep their place,
    # so the buffer must cover the whole sequence, not just the live pages.
    meta = plan.build_meta_tensors(max_num_pages=8, max_num_batched_requests=2,
                                   max_seq_length=512, device="cpu")
    assert meta["paged_kv_indices_buffer_0"].shape[0] == 2 * (512 // 64)

class _FakePK:
    """Duck-typed PersistentKernel stand-in: KVEventLog only needs
    meta_tensors."""

    def __init__(self):
        self.meta_tensors = {}


def test_kv_event_log_roundtrip():
    specs = [KVSpec("full", per_entry_bytes=64, layer_ids=(0,),
                    preferred_block_size=64)]
    plan = plan_kv_groups(specs)
    pk = _FakePK()
    ev = KVEventLog(pk, plan, capacity=64, device="cpu")
    assert pk.meta_tensors["kv_event_log"] is ev.log
    # Hand-craft a symmetric alloc/free sequence for group 0: pages 0 and 1
    # allocated, an iteration marker, then both freed.
    events = [
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (3, -1, -1, -1),
        (2, 0, 0, 0),
        (2, 0, 0, 1),
    ]
    ev.log[0] = len(events)
    for i, (t, g, r, p) in enumerate(events):
        ev.log[4 * i + 1], ev.log[4 * i + 2], ev.log[4 * i + 3], \
            ev.log[4 * i + 4] = t, g, r, p
    result = ev.verify()
    assert result == {"iterations": 1, "compactions": 0,
                      "per_group": [{"allocs": 2, "frees": 2}]}


def test_kv_event_log_replay_catches_leak_and_double_alloc():
    def _make_log(events):
        log = torch.zeros(64, dtype=torch.int32)
        log[0] = len(events)
        for i, (t, g, r, p) in enumerate(events):
            log[4 * i + 1], log[4 * i + 2], log[4 * i + 3], log[4 * i + 4] = t, g, r, p
        return log

    with _raises(AssertionError):
        KVEventLog.replay(_make_log([(1, 0, 0, 0)]), num_groups=1)  # leak

    with _raises(AssertionError):
        KVEventLog.replay(
            _make_log([(1, 0, 0, 0), (1, 0, 0, 0)]), num_groups=1)  # double-alloc

    with _raises(AssertionError):
        KVEventLog.replay(
            _make_log([(2, 0, 0, 0)]), num_groups=1)  # free of unowned page

    # well-formed 2 groups, symmetric
    good = _make_log([(1, 0, 0, 0), (1, 1, 0, 5), (2, 0, 0, 0), (2, 1, 0, 5)])
    result = KVEventLog.replay(good, num_groups=2)
    assert result == {"iterations": 0, "compactions": 0, "per_group": [
        {"allocs": 1, "frees": 1}, {"allocs": 1, "frees": 1}]}

    # A MOVE carries no group id (-1) and must not be range-checked as one.
    moved = _make_log([(1, 0, 0, 0), (4, -1, 1, 0), (2, 0, 0, 0)])
    result = KVEventLog.replay(moved, num_groups=1)
    assert result["compactions"] == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"{fn.__name__} OK")
    print(f"PASSED: {len(fns)} planner tests")


# ── the declaration surface models actually use ───────────────────────────


class _StubMPK:
    """Stands in for PersistentKernel: attach() only needs attach_input."""

    def __init__(self):
        self.attached = []

    def attach_input(self, torch_tensor, name):
        self.attached.append(name)
        return name


def _gpt_oss_shaped_streams(h=8, d=64, page=64):
    kv = [("k", (h, d), torch.bfloat16), ("v", (h, d), torch.bfloat16)]
    return [
        KVStream("sliding_attention", layers=(0, 2), window=128,
                 components=kv, preferred_block_size=page),
        KVStream("full_attention", layers=(1, 3),
                 components=kv, preferred_block_size=page),
    ]


def test_kvstream_derives_the_byte_count_the_hand_formula_gave():
    h, d = 8, 64
    stream = KVStream("s", layers=(0,),
                      components=[("k", (h, d), torch.bfloat16),
                                  ("v", (h, d), torch.bfloat16)])
    # what every builder used to write out by hand: K + V, bf16
    assert stream.per_entry_bytes == 2 * h * d * 2


def test_build_kv_cache_matches_the_equivalent_kvspec_plan():
    """The new surface is a re-spelling, not a different planner."""
    with _free_memory(64 << 30):
        new = build_kv_cache(_gpt_oss_shaped_streams(),
                             max_num_pages=4, device="cpu", verbose=False)
    old = plan_kv_groups([
        KVSpec("sliding_attention", per_entry_bytes=2 * 8 * 64 * 2,
               layer_ids=(0, 2), window_size=128, preferred_block_size=64),
        KVSpec("full_attention", per_entry_bytes=2 * 8 * 64 * 2,
               layer_ids=(1, 3), preferred_block_size=64),
    ])
    assert new.target_page_bytes == old.target_page_bytes
    assert new.num_slots == old.num_slots
    assert [g.block_size for g in new.groups] == [g.block_size for g in old.groups]


def test_attach_hands_out_pool_views_and_the_group_id():
    with _free_memory(64 << 30):
        plan = build_kv_cache(_gpt_oss_shaped_streams(),
                              max_num_pages=4, device="cpu", verbose=False)
    mpk = _StubMPK()
    seen = {}
    for layer in (0, 1, 2, 3):
        got = plan.attach(mpk, layer)
        assert set(got) == {"k_cache", "v_cache", "group_id"}
        seen[layer] = got["group_id"]
    # layers of one stream share a page table, the two streams do not
    assert seen[0] == seen[2] and seen[1] == seen[3] and seen[0] != seen[1]
    assert len(mpk.attached) == 8


def test_attach_refuses_a_cache_copied_out_of_the_pool():
    """The check attach() folds in is the only thing that catches the
    .contiguous() trap: the copy has the right shape, dtype and values."""
    with _free_memory(64 << 30):
        plan = build_kv_cache(_gpt_oss_shaped_streams(),
                              max_num_pages=4, device="cpu", verbose=False)
    view = plan.views(0)["k"][0]
    copy = view.contiguous()
    assert copy.shape == view.shape and copy.dtype == view.dtype
    assert torch.equal(copy, view)
    with _raises(AssertionError):
        plan._assert_in_pool(copy, "copied")


def test_materialize_refuses_a_plan_that_never_declared_layouts():
    plan = plan_kv_groups([
        KVSpec("s", per_entry_bytes=2048, layer_ids=(0,),
               preferred_block_size=64)])
    with _raises(RuntimeError):
        plan._materialize(max_num_pages=2, device="cpu")


# ── resolve_pool_size ─────────────────────────────────────────────────────


@contextmanager
def _free_memory(free_bytes, total_bytes=None):
    """Pin what the device reports free. resolve_pool_size refuses a pool that
    does not fit, and that branch is otherwise only reachable by owning a
    particular card."""
    real = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda device=0: (
        free_bytes, total_bytes if total_bytes is not None else free_bytes)
    try:
        yield
    finally:
        torch.cuda.mem_get_info = real


def _one_stream_plan(page_size=64):
    """Planned but NOT sized -- these tests drive the sizing step itself."""
    return plan_kv_groups([
        KVSpec("attention", per_entry_bytes=2 * 8 * 64 * 2,
               layer_ids=tuple(range(4)), preferred_block_size=page_size)])


def test_resolve_pool_size_takes_exactly_one_of_the_two_knobs():
    plan = _one_stream_plan()
    with _raises(ValueError):        # neither
        _resolve_pool_size(plan, max_seq_length=512, verbose=False)
    with _raises(ValueError):        # both
        _resolve_pool_size(plan, kv_budget="1GiB", max_num_pages=64,
                          max_seq_length=512, verbose=False)


def test_resolve_pool_size_publishes_the_count_on_the_plan():
    """Both sizing sites read plan.max_num_pages, not the return value."""
    plan = _one_stream_plan()
    assert plan.max_num_pages is None
    with _free_memory(64 << 30):
        got = _resolve_pool_size(plan, max_num_pages=4096, max_seq_length=512,
                                verbose=False)
    assert got == 4096 and plan.max_num_pages == 4096


def test_resolve_pool_size_refuses_a_pool_below_one_requests_floor():
    plan = _one_stream_plan()
    floor = plan.pages_needed(1, 8192, 1)
    with _free_memory(64 << 30), _raises(ValueError):
        _resolve_pool_size(plan, max_num_pages=floor - 1,
                          max_seq_length=8192, verbose=False)
    with _free_memory(64 << 30):     # the floor itself is allowed
        _resolve_pool_size(plan, max_num_pages=floor, max_seq_length=8192,
                          verbose=False)


def test_resolve_pool_size_refuses_a_pool_that_does_not_fit_in_free_memory():
    """Caught before the allocation, so the failure names the pool rather than
    arriving as a CUDA OOM from inside allocate."""
    plan = _one_stream_plan()
    pages = 1 << 20
    need = plan.budget_bytes(pages)
    with _free_memory(need - 1), _raises(ValueError):
        _resolve_pool_size(plan, max_num_pages=pages, max_seq_length=512,
                          verbose=False)
    with _free_memory(need):
        assert _resolve_pool_size(plan, max_num_pages=pages,
                                 max_seq_length=512, verbose=False) == pages


def test_build_kv_cache_requires_max_seq_length_with_a_budget():
    """A budget is sized to hold a request of some length; without one there
    is nothing to size to, and an undersized pool surfaces as a run-time
    deadlock rather than an error here."""
    with _raises(ValueError):
        build_kv_cache(_gpt_oss_shaped_streams(), kv_budget="1GiB",
                       device="cpu", verbose=False)


def test_resolve_pool_size_skips_the_floor_check_without_a_length():
    """An explicit page count and no length: nothing to compare, so the pool
    is taken as given. This is the pre-KV2 contract the unmigrated demos
    still run on."""
    plan = _one_stream_plan()
    tiny = 1
    assert tiny < plan.pages_needed(1, 8192, 1)
    with _free_memory(64 << 30):
        assert _resolve_pool_size(plan, max_num_pages=tiny,
                                  verbose=False) == tiny
