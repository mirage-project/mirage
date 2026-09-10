"""
Unit tests for plan_kv_groups.
"""

from contextlib import contextmanager

import torch

from mirage.mpk.models.gpt_oss.builder import (
    kv_streams as kv_streams_gpt_oss,
)
from mirage.mpk.kv_planner import (
    KVEventLog,
    KVKind,
    KVSpec,
    KVStream,
    KVUnificationError,
    pages_per_request,
    plan_kv_groups,
    build_kv_cache,
    _merge_identical_streams,
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
    return plan_kv_groups([s._spec() for s in kv_streams_gpt_oss(_GptOssCfg())],
                          block_size=page_size)


def test_four_streams_at_mixed_compression_share_one_page():
    specs = [
        KVSpec("c4_main", per_entry_bytes=584, layer_ids=(0,),
               compress_ratio=4),
        KVSpec("c128_main", per_entry_bytes=584, layer_ids=(1,),
               compress_ratio=128),
        KVSpec("c4_indexer", per_entry_bytes=132, layer_ids=(2,),
               compress_ratio=4),
        KVSpec("swa", per_entry_bytes=584, layer_ids=(3,),
               window_size=128),
    ]
    # 37376 B is what c4_main and swa both need at 64 entries; given
    # explicitly since no single block_size (token count) yields it for
    # both when their compress ratios differ.
    plan = plan_kv_groups(specs, target_page_bytes=37376)
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
        KVSpec("fat", per_entry_bytes=2048, layer_ids=(0,)),   # 512 KiB @ 256
        KVSpec("thin", per_entry_bytes=512, layer_ids=(1,)),   # 128 KiB, 4x under
    ]
    plan = plan_kv_groups(specs, block_size=256)
    by = {g.spec_name: g for g in plan.groups}
    assert plan.target_page_bytes == 256 * 2048
    assert by["fat"].block_size == 256
    assert by["thin"].block_size == 256 * 4        # scaled, not re-packed
    assert by["fat"].padding_bytes_per_page == 0
    assert by["thin"].padding_bytes_per_page == 0  # exact ratio wastes nothing
    for g in plan.groups:
        assert g.block_size % 64 == 0


def test_block_size_pads_a_non_exact_spec_instead_of_repacking_it():
    # indexer's native page (also 256 tokens) is not an exact divisor of
    # main's, so the block_size path pads it at 64 entries rather than
    # greedily repacking to 272 the way explicit target_page_bytes would
    # (see test_allocate_pool_shares_one_allocation_across_streams).
    specs = [
        KVSpec("main", per_entry_bytes=584, layer_ids=(0,), compress_ratio=4),
        KVSpec("indexer", per_entry_bytes=132, layer_ids=(1,),
               compress_ratio=4),
    ]
    plan = plan_kv_groups(specs, block_size=256)
    assert plan.target_page_bytes == 37376
    by = {g.spec_name: g for g in plan.groups}
    assert by["main"].block_size == 256 and by["main"].padding_bytes_per_page == 0
    assert by["indexer"].block_size == 256          # native, NOT 1088
    assert by["indexer"].entries_per_page == 64     # native, NOT 272
    assert by["indexer"].padding_bytes_per_page == 37376 - 64 * 132


def test_block_size_and_target_page_bytes_are_exclusive():
    with _raises(ValueError):
        plan_kv_groups([KVSpec("s", per_entry_bytes=64, layer_ids=(0,))],
                       block_size=64, target_page_bytes=4096)
    # either alone, or neither, is fine
    plan_kv_groups([KVSpec("s", per_entry_bytes=64, layer_ids=(0,))],
                   block_size=64)
    plan_kv_groups([KVSpec("s", per_entry_bytes=64, layer_ids=(0,))],
                   target_page_bytes=4096)
    plan_kv_groups([KVSpec("s", per_entry_bytes=64, layer_ids=(0,))])


def test_block_size_is_floored_to_the_declared_tile():
    # Same stream, two tiles: a bigger tile costs padding. Explicit
    # target_page_bytes so idx is greedily repacked (_fit_block_size), not
    # padded at its native size -- the tradeoff tested is within repacking.
    def plan_with(tile):
        return plan_kv_groups([
            KVSpec("anchor", per_entry_bytes=584, layer_ids=(0,),
                   compress_ratio=4),
            KVSpec("idx", per_entry_bytes=132, layer_ids=(1,),
                   compress_ratio=4, block_size_multiple_of=tile),
        ], target_page_bytes=37376)

    got = {t: {g.spec_name: g for g in plan_with(t).groups} for t in (16, 64)}
    # 283 entries fit; tile 64 needs groups of 16 -> 272, tile 16 -> 280.
    assert got[64]["idx"].block_size == 1088 and got[64]["idx"].entries_per_page == 272
    assert got[16]["idx"].block_size == 1120 and got[16]["idx"].entries_per_page == 280
    assert got[16]["idx"].padding_bytes_per_page < got[64]["idx"].padding_bytes_per_page
    assert got[64]["idx"].block_size % 64 == 0
    assert got[16]["idx"].block_size % 16 == 0


def test_page_wastes_bytes_warns_only_when_a_spec_does_not_tile_exactly():
    import warnings

    exact = [
        KVSpec("a", per_entry_bytes=512, layer_ids=(0,)),
        KVSpec("b", per_entry_bytes=256, layer_ids=(1,)),   # divides "a"'s page
    ]
    ragged = [
        KVSpec("a", per_entry_bytes=584, layer_ids=(0,), compress_ratio=4),
        KVSpec("idx", per_entry_bytes=132, layer_ids=(1,), compress_ratio=4),
    ]
    for specs, expect_warning in [(exact, False), (ragged, True)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan_kv_groups(specs, block_size=256)
        if expect_warning:
            assert len(caught) == 1
            msg = str(caught[0].message)
            assert "'idx'" in msg and "%" in msg and "do not tile" in msg
        else:
            assert not caught, [str(w.message) for w in caught]


def test_a_stream_that_cannot_fit_one_tile_is_refused():
    # 8 entries fit, but a 64-token tile needs 64.
    specs = [
        KVSpec("fat", per_entry_bytes=8, layer_ids=(0,)),
        KVSpec("thin", per_entry_bytes=800, layer_ids=(1,)),
    ]
    with _raises(KVUnificationError):
        plan_kv_groups(specs, target_page_bytes=6400)


def test_a_block_size_off_the_tile_floors_instead_of_refusing():
    # 100 is not a multiple of the 64-token tile, so _native_fit floors it to
    # 64 for both streams before either is picked as the anchor, so neither
    # pays padding. It does not raise (KVUnificationError is reserved for a
    # spec that can't fit even one tile's worth of entries at all).
    plan = _gpt_oss_plan(100)
    assert plan.groups[0].block_size == 64
    assert plan.groups[0].padding_bytes_per_page == 0
    for legal in (64, 128, 4096):
        assert _gpt_oss_plan(legal).groups[0].block_size == legal
        assert _gpt_oss_plan(legal).groups[0].padding_bytes_per_page == 0


def test_the_report_flags_a_dead_window():
    # A block whose first recycle lands past the run: the window is inert
    # here, so the report says so without telling the user to shrink it.
    dead = _gpt_oss_plan(4096).describe(max_seq_length=512)
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
                   window_size=128),
            KVSpec("b", per_entry_bytes=512,
                   layer_ids=tuple(range(n_a, n_a + n_b))),
        ]
        plan = plan_kv_groups(specs)
        by = _by_spec(plan)
        assert plan.num_slots == slots, note
        assert (len(by["a"]), len(by["b"])) == groups, note
        assert sum(g.layer_ids.count(None) for g in plan.groups) == padded, note


def test_group_size_warns_only_when_a_tiny_stream_forces_fragmentation():
    # The three shapes above are deliberate, bounded-waste outcomes -- none
    # should warn. A stream too small to share a usable gcd (e.g. a 1-layer
    # speculative-decode draft) forces group_size down to itself and
    # fragments every other stream; that should warn by name, even though
    # gcd-with-1 is what _group_size itself treats as "clean".
    import warnings

    for note, n_a, n_b, expect_warning in [
        ("12 vs 13: close, pad rather than split", 12, 13, False),
        ("20 vs 4 at 5:1: gcd 4, zero padding", 20, 4, False),
        ("20 vs 30: gcd 10 beats a min-based 20", 20, 30, False),
        ("61 vs 1: no gcd but 1, draft starves the target", 61, 1, True),
    ]:
        specs = [
            KVSpec("a", per_entry_bytes=512, layer_ids=tuple(range(n_a))),
            KVSpec("b", per_entry_bytes=512,
                   layer_ids=tuple(range(n_a, n_a + n_b))),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan_kv_groups(specs)
        if expect_warning:
            assert len(caught) == 1, note
            msg = str(caught[0].message)
            assert "'a'" in msg and "'b'" in msg and "61 group" in msg, note
        else:
            assert not caught, (note, [str(w.message) for w in caught])


# ── bounded specs (fixed one entry per request, e.g. a Mamba/KDA state) ─────


def test_bounded_spec_never_exceeds_one_entry_even_when_not_the_anchor():
    """The bug this exists for: a bounded state sharing a page with a bigger
    spec used to get greedily packed to several entries and report ZERO
    padding, because _fit_block_size assumed every entry it could fit was
    useful. Only entry 0 of a bounded state is ever read or written, so that
    was capacity mislabeled as in-use. Checked on both fitting paths."""
    bounded = KVSpec("state", per_entry_bytes=8192, layer_ids=(0,),
                     compress_ratio=4096, bounded=True)
    attn = KVSpec("attn", per_entry_bytes=1152, layer_ids=(1,))

    # target_page_bytes path (_fit_block_size): 73728 // 8192 = 9 entries
    # would fit if packed like a normal spec.
    plan = plan_kv_groups([bounded, attn], target_page_bytes=73728)
    g = {g.spec_name: g for g in plan.groups}["state"]
    assert g.entries_per_page == 1
    assert g.padding_bytes_per_page == 73728 - 8192, (
        "padding must be the honest remainder, not 0")

    # block_size path (_fit_to_anchor): bounded's own native bytes (8192) is
    # smaller than attn's here, so bounded is NOT the anchor and would have
    # scaled up under the old "exact ratio -> zero padding" rule.
    plan = plan_kv_groups([
        KVSpec("state", per_entry_bytes=1024, layer_ids=(0,),
              compress_ratio=4096, bounded=True),
        KVSpec("attn", per_entry_bytes=1024, layer_ids=(1,)),
    ], block_size=256)
    g = {g.spec_name: g for g in plan.groups}["state"]
    assert g.entries_per_page == 1, (
        "an exact scale-up ratio must not apply to a bounded spec")

    # And the default path (neither kwarg given): before `bounded` existed,
    # this shape could only be planned by hitting KVUnificationError here and
    # manually retrying with an explicit target_page_bytes.
    plan = plan_kv_groups([
        KVSpec("state", per_entry_bytes=2_170_880, layer_ids=(0,),
              compress_ratio=4096, bounded=True),
        KVSpec("attn", per_entry_bytes=1152, layer_ids=(1,)),
    ])
    g = {g.spec_name: g for g in plan.groups}["state"]
    assert g.entries_per_page == 1 and g.padding_bytes_per_page == 0


def test_bounded_spec_holds_exactly_one_page_per_request():
    """The whole point of `bounded`: no matter the run length (as long as it
    stays under max_seq_length, which compress_ratio is set to), a bounded
    group never grows past 1 page -- unlike a normal spec, which needs more
    pages as the sequence grows."""
    bounded = KVSpec("state", per_entry_bytes=8192, layer_ids=(0,),
                     compress_ratio=4096, bounded=True)
    plan = plan_kv_groups([bounded], target_page_bytes=8192)
    g = plan.groups[0]
    for seq_len in (1, 4096):
        assert pages_per_request(g.block_size, g.window_size, seq_len) == 1


def test_bounded_and_window_are_exclusive():
    """A fixed one-entry state is never recycled by a sliding window --
    checked at both entry points a caller could hit: the raw KVSpec, and the
    KVStream._spec() a model builder actually goes through."""
    with _raises(AssertionError):
        KVSpec("state", per_entry_bytes=8192, layer_ids=(0,),
              bounded=True, window_size=128)
    with _raises(ValueError):
        KVStream("state", layers=(0,),
                 components=[("s", (2048,), torch.bfloat16)],
                 kind=KVKind.BOUNDED, window=128)._spec()


def test_kimi_kda_shaped_bounded_state_end_to_end():
    """The real motivating shape: a KDA (gated-delta) state next to MLA, the
    numbers from the Kimi Linear investigation. kda is declared once per
    request's whole run (compress_ratio=max_seq_length), mla scales normally
    off the same shared page."""
    kda_bytes = 12288 * 3 * 2 + 32 * 128 * 128 * 4   # conv bf16 + recurrent fp32
    kda = KVStream("kda", layers=tuple(range(21)), compress_ratio=4096,
                  kind=KVKind.BOUNDED,
                  components=[("conv", (12288, 3), torch.bfloat16),
                              ("recur", (32, 128, 128), torch.float32)])
    mla = KVStream("mla", layers=tuple(range(21, 27)),
                  components=[("kv", (576,), torch.bfloat16)])
    with _free_memory(64 << 30):
        plan = build_kv_cache([kda, mla], target_page_bytes=kda_bytes,
                              max_num_pages=256, max_seq_length=4096,
                              device="cpu", verbose=False)
    by = {g.spec_name: g for g in plan.groups}
    assert by["kda"].entries_per_page == 1
    assert by["kda"].padding_bytes_per_page == 0   # kda set the page itself
    assert by["mla"].entries_per_page == 1856      # measured, kept as a pin
    assert plan.pages_needed(1, 4096, 1) == 13      # 1 (kda) + 12 (mla)

    mpk = _StubMPK()
    kv = plan.attach(mpk, 0)
    assert plan.groups[kv["group_id"]].spec_name == "kda"


def test_allocate_pool_slots_are_per_layer_and_do_not_alias():
    # per_entry_bytes must match what the caller actually stores: an (8, 16)
    # bf16 entry is 256 B. Going through the pool makes the two agree by
    # construction — declaring 64 here would overrun the page budget.
    specs = [
        KVSpec("full", per_entry_bytes=256, layer_ids=(0, 1)),
        KVSpec("window", per_entry_bytes=256, layer_ids=(2, 3),
              window_size=128),
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
                block_size_multiple_of=1)
    thin = KVSpec("thin", per_entry_bytes=800, layer_ids=(1,),
                 block_size_multiple_of=1)
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
    # is owned by one stream at a time, so nothing is stranded. Every view is
    # strided by the whole page, not its packed entry span. Explicit
    # target_page_bytes so indexer is greedily repacked (this test is about
    # allocation/view mechanics, not which fitting rule ran).
    specs = [
        KVSpec("main", per_entry_bytes=584, layer_ids=(0, 1),
               compress_ratio=4),
        KVSpec("indexer", per_entry_bytes=132, layer_ids=(2, 3),
               compress_ratio=4),
    ]
    plan = plan_kv_groups(specs, target_page_bytes=37376)
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
    spec = KVSpec("gqa", per_entry_bytes=2048, layer_ids=(0, 1))
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
    spec = KVSpec("gqa", per_entry_bytes=2048, layer_ids=(0, 1))
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
        KVSpec("full", per_entry_bytes=584, layer_ids=(0,)),
        KVSpec("sw", per_entry_bytes=584, layer_ids=(1,), window_size=128),
    ]
    plan = plan_kv_groups(specs)
    gs = plan.group_specs()
    assert [g.block_size for g in gs] == [64, 64]


def test_build_meta_tensors():
    specs = [
        KVSpec("full", per_entry_bytes=64, layer_ids=(0, 1)),
        KVSpec("sw", per_entry_bytes=64, layer_ids=(2, 3), window_size=128),
    ]
    plan = plan_kv_groups(specs)
    assert len(plan.groups) == 2
    meta = plan.build_meta_tensors(max_num_pages=32, max_seq_length=64,
                                   max_num_batched_requests=4, device="cpu")
    assert set(meta.keys()) == {
        f"paged_kv_{field}_buffer_{g}"
        for g in range(2)
        for field in ("indptr", "indices", "last_page_len")
    } | {f"paged_kv_indices_snapshot_{g}" for g in range(2)}
    # the snapshot is compaction's scratch copy of the index buffer, so it
    # must match it exactly -- PersistentKernel does not size it in
    # online_pinned mode, which allocates nothing of its own
    for g in range(2):
        assert (meta[f"paged_kv_indices_snapshot_{g}"].shape
                == meta[f"paged_kv_indices_buffer_{g}"].shape)
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
    specs = [KVSpec("full", per_entry_bytes=64, layer_ids=(0,))]
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


def _gpt_oss_shaped_streams(h=8, d=64):
    kv = [("k", (h, d), torch.bfloat16), ("v", (h, d), torch.bfloat16)]
    return [
        KVStream("sliding_attention", layers=(0, 2), window=128,
                 components=kv),
        KVStream("full_attention", layers=(1, 3), components=kv),
    ]


def test_attach_hands_out_pool_views_and_the_group_id():
    with _free_memory(64 << 30):
        plan = build_kv_cache(_gpt_oss_shaped_streams(),
                              max_num_pages=4, device="cpu", verbose=False)
    mpk = _StubMPK()
    seen = {}
    for layer in (0, 1, 2, 3):
        got = plan.attach(mpk, layer)
        # window_size rides along ONLY for the windowed stream (gpt-oss's
        # even layers); absent means 0, which is what the layer functions
        # already default to. It comes from the same group the page table
        # does, so the kernel's mask and the scheduler's recycling cannot
        # disagree.
        want = {"k_cache", "v_cache", "group_id"}
        if layer % 2 == 0:
            want |= {"window_size"}
            assert got["window_size"] == 128
        assert set(got) == want
        seen[layer] = got["group_id"]
    # layers of one stream share a page table, the two streams do not
    assert seen[0] == seen[2] and seen[1] == seen[3] and seen[0] != seen[1]
    assert len(mpk.attached) == 8


def _dsv4_shaped_streams(layers=(0, 1)):
    """One layer carrying THREE caches, the DeepSeek-V4 shape: a windowed
    cache, a compressed one, and an indexer. vLLM's sparse-MLA reads the first
    two in ONE kernel launch with two block tables, so attach() has to be able
    to hand them over together."""
    return [
        KVStream("swa", layers=layers, window=128,
                 components=[("k", (8, 64), torch.bfloat16),
                             ("v", (8, 64), torch.bfloat16)]),
        KVStream("c4", layers=layers, compress_ratio=4,
                 components=[("kv", (576,), torch.bfloat16)]),
        KVStream("indexer", layers=layers, compress_ratio=4,
                 components=[("kv", (132,), torch.uint8)]),
    ]


def test_one_layer_can_carry_several_caches_and_attach_returns_them_all():
    """The capability this exists for: a task reading a windowed cache and a
    compressed cache needs both, plus a page table for each, from one call.
    No attach_prefix is declared anywhere -- each stream's OWN name is the
    namespace, since names are already required unique."""
    with _free_memory(64 << 30):
        plan = build_kv_cache(_dsv4_shaped_streams(), max_num_pages=8,
                              target_page_bytes=128 * 1024, device="cpu",
                              verbose=False)
    mpk = _StubMPK()
    got = plan.attach(mpk, 0)
    assert set(got) == {
        "swa_k_cache", "swa_v_cache", "swa_group_id", "swa_window_size",
        "c4_kv_cache", "c4_group_id",
        "indexer_kv_cache", "indexer_group_id",
    }
    assert got["swa_window_size"] == 128
    # _layer_info used to return the FIRST group holding a layer, silently
    # handing back one cache and hiding the rest; it must refuse instead.
    with _raises(KeyError):
        plan._layer_info(0)
    # three DIFFERENT page tables, so a task can read all three at once
    assert len({got["swa_group_id"], got["c4_group_id"],
                got["indexer_group_id"]}) == 3
    # every cache got its own graph input; a collision here would have made
    # attach_input silently overwrite one in the kernel-reuse tensor map
    assert len(mpk.attached) == len(set(mpk.attached)) == 4


def test_identical_geometry_on_one_layer_is_refused():
    """Two caches with the SAME page geometry on the SAME layer are refused
    regardless of their (always-distinct) names -- nothing needs this shape
    today, and if something ever does, the fix is a real page-describing
    difference, not silently splitting into two groups."""
    kv = [("k", (8, 64), torch.bfloat16)]
    with _raises(ValueError):
        build_kv_cache([KVStream("a", layers=(0,), components=kv),
                        KVStream("b", layers=(0,), components=kv)],
                       max_num_pages=4, device="cpu", verbose=False)
    # a real geometry difference (here: window) makes them legitimately two
    # caches
    with _free_memory(64 << 30):
        build_kv_cache([KVStream("a", layers=(0,), components=kv),
                        KVStream("b", layers=(0,), components=kv,
                                 window=128)],
                       max_num_pages=4, device="cpu", verbose=False)


def test_two_streams_cannot_share_a_name():
    """_layouts is keyed by stream name, so a repeat silently loses a layout."""
    kv = [("k", (8, 64), torch.bfloat16)]
    with _raises(ValueError):
        build_kv_cache([KVStream("dup", layers=(0,), components=kv),
                        KVStream("dup", layers=(1,), components=kv)],
                       max_num_pages=4, device="cpu", verbose=False)


def test_merged_streams_still_attach_with_bare_keys_per_layer():
    """Two streams identical but for name/layers fold into ONE group (the
    DSv3 mla+mtp win), but attach() resolves per LAYER, not per group: since
    layer 0 and layer 1 each carry only one of them, both still get bare
    keys, exactly as they would unmerged. The merge is an internal pooling
    decision that a builder's attach() calls never have to know about."""
    kv = [("k", (8, 64), torch.bfloat16)]
    streams = [KVStream("a", layers=(0,), components=kv),
               KVStream("b", layers=(1,), components=kv)]
    merged, name_of = _merge_identical_streams(streams)
    assert len(merged) == 1, "identical layout, disjoint layers should merge"
    assert name_of == {"a": "a+b", "b": "a+b"}

    with _free_memory(64 << 30):
        plan = build_kv_cache(streams, max_num_pages=4, device="cpu",
                              verbose=False)
    mpk = _StubMPK()
    assert set(plan.attach(mpk, 0)) == {"k_cache", "group_id"}
    assert set(plan.attach(mpk, 1)) == {"k_cache", "group_id"}


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
        KVSpec("s", per_entry_bytes=2048, layer_ids=(0,))])
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


def _one_stream_plan():
    """Planned but NOT sized -- these tests drive the sizing step itself."""
    return plan_kv_groups([
        KVSpec("attention", per_entry_bytes=2 * 8 * 64 * 2,
               layer_ids=tuple(range(4)))])


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


# ── merging streams that lay a page out identically ────────────────────────


def _dsv3_shaped_streams():
    """DeepSeek-V3's shape: 61 attention layers and 1 MTP layer. MTP is a
    speculative draft module built from a plain DeepseekV2DecoderLayer, so its
    KV is the same thing per token as the main layers'."""
    entry = [("kv", (1, 576), torch.bfloat16)]     # MLA: 512 latent + 64 rope
    return [
        KVStream("attention", layers=tuple(range(61)), components=entry),
        KVStream("mtp", layers=(61,), components=entry),
    ]


def test_identical_streams_merge_into_one_group():
    """Declared as two streams, _group_size sees counts [61, 1]: gcd is 1, so
    it returns 1 and the plan is 62 single-slot groups. Merged first, it sees
    [62] and returns 62 -- one group, no padding."""
    unmerged = plan_kv_groups([s._spec() for s in _dsv3_shaped_streams()])
    assert len(unmerged.groups) == 62 and unmerged.num_slots == 1

    merged, _ = _merge_identical_streams(_dsv3_shaped_streams())
    assert [s.name for s in merged] == ["attention+mtp"]
    plan = plan_kv_groups([s._spec() for s in merged])
    assert len(plan.groups) == 1 and plan.num_slots == 62
    # Every layer still resolves, and to a distinct slot.
    slots = [plan._layer_info(i) for i in range(62)]
    assert sorted(s for _, s in slots) == list(range(62))
    assert {g for g, _ in slots} == {0}


def test_merged_stream_sorts_its_layers():
    """Slot assignment must not depend on which stream was declared first."""
    a, b = _dsv3_shaped_streams()
    forward = _merge_identical_streams([a, b])[0][0].layers
    backward = _merge_identical_streams([b, a])[0][0].layers
    assert forward == backward == tuple(range(62))


def test_gpt_oss_streams_do_not_merge():
    """The counter-example that keeps the merge honest: gpt-oss's two streams
    agree on everything but the window, and a window is what a group recycles
    against. Fusing them would give the full-attention layers a windowed page
    table and free pages still being read."""
    streams = kv_streams_gpt_oss(_GptOssCfg())
    assert len(_merge_identical_streams(streams)[0]) == 2
    plan = plan_kv_groups(
        [s._spec() for s in _merge_identical_streams(streams)[0]])
    assert len(plan.groups) == 2 and plan.num_slots == 12


def test_equal_byte_size_is_not_equal_layout():
    """Both of these are 1024 bytes per entry, and merging them would hand a
    layer a view of the wrong shape. This is why the merge runs on KVStream
    and not inside plan_kv_groups, which sees only per_entry_bytes."""
    one = KVStream("one", layers=(0,),
                   components=[("k", (8, 64), torch.bfloat16)])
    two = KVStream("two", layers=(1,),
                   components=[("k", (4, 64), torch.bfloat16),
                               ("v", (4, 64), torch.bfloat16)])
    assert one.per_entry_bytes == two.per_entry_bytes == 1024
    assert len(_merge_identical_streams([one, two])[0]) == 2


def test_merge_key_covers_every_declared_field():
    """Two streams differing in ANY page-describing field must stay apart.
    Asserts the table covers every KVStream field, so adding a field fails
    here until given a value to differ by -- a key that silently missed a
    field would fuse streams that aren't alike, uncaught downstream.
    """
    from dataclasses import fields as _fields

    base = dict(components=[("k", (4, 64), torch.bfloat16)], window=0,
                compress_ratio=1, block_size_multiple_of=None,
                kind=KVKind.PAGED)
    others = dict(components=[("k", (8, 64), torch.bfloat16)], window=128,
                  compress_ratio=2, block_size_multiple_of=64,
                  kind=KVKind.FLAT)

    declared = {f.name for f in _fields(KVStream)} - {"name", "layers"}
    assert declared == set(base) == set(others), (
        f"KVStream fields {sorted(declared)} are not all covered by this "
        f"test's table {sorted(base)}")

    for field, other in others.items():
        a = KVStream("a", layers=(0,), **base)
        b = KVStream("b", layers=(1,), **{**base, field: other})
        assert len(_merge_identical_streams([a, b])[0]) == 2, (
            f"streams differing only in {field!r} were merged")

    # ...and two that differ in nothing but name and layers do merge, so the
    # loop above is not passing because merging is broken outright.
    a = KVStream("a", layers=(0,), **base)
    b = KVStream("b", layers=(1,), **base)
    assert len(_merge_identical_streams([a, b])[0]) == 1


def test_merged_stream_allocates_and_every_layer_gets_its_view():
    """The integration risk of merging: the group's spec_name becomes the
    merged name, and _layouts is keyed by stream name. If those stopped
    lining up it would surface only here, at allocation."""
    plan = build_kv_cache(_dsv3_shaped_streams(), max_num_pages=4,
                          device="cpu", verbose=False)
    assert plan.groups[0].spec_name == "attention+mtp"
    for layer in range(62):
        group_id, slot = plan._layer_info(layer)
        view = plan._views[group_id]["kv"][slot]
        assert tuple(view.shape[2:]) == (1, 576)
        assert view.dtype == torch.bfloat16
    # Distinct layers must not alias each other.
    a = plan._views[0]["kv"][plan._layer_info(0)[1]]
    b = plan._views[0]["kv"][plan._layer_info(61)[1]]
    a.fill_(1.0); b.fill_(2.0)
    assert a.flatten()[0].item() == 1.0 and b.flatten()[0].item() == 2.0


def test_deepseek_v3_declares_one_group_over_61_layers_plus_mtp():
    """DeepSeek-V3 61 MLA layers + the MTP predictor land in ONE group
    (declared apart, folded by layout)."""
    from mirage.mpk.models.deepseek_v3.builder import kv_streams

    class _Config:
        num_hidden_layers = 61
        num_nextn_predict_layers = 1

    streams = kv_streams(_Config())
    assert [s.name for s in streams] == ["mla", "mtp"]
    assert streams[0].per_entry_bytes == 576 * 2      # 512 latent + 64 rope
    assert streams[1].layers == (61,)

    plan = build_kv_cache(streams, max_num_pages=4, device="cpu",
                          verbose=False)
    assert len(plan.groups) == 1, (
        f"MLA and MTP must share a page table, got "
        f"{[(g.group_id, g.spec_name) for g in plan.groups]}")
    group = plan.groups[0]
    assert group.spec_name == "mla+mtp"
    assert len(group.layer_ids) == 62 and 61 in group.layer_ids
    assert group.block_size == 64 and group.entries_per_page == 64
    assert group.padding_bytes_per_page == 0

    # One entry per token, so a page holds block_size rows and the per-layer
    # view is strided by exactly one page.
    view = plan._views[0]["kv"][plan._layer_info(3)[1]]
    assert tuple(view.shape) == (plan.max_num_pages, 64, 576)
    assert view.stride() == (64 * 576, 576, 1)


def test_deepseek_v3_leaves_the_mtp_slot_out_when_mtp_is_off():
    """A slot costs 1/62 of every page whether or not MTP runs, so the demo
    passes 0 when --mtp is 0."""
    from mirage.mpk.models.deepseek_v3.builder import kv_streams

    class _Config:
        num_hidden_layers = 61
        num_nextn_predict_layers = 1

    streams = kv_streams(_Config(), num_mtp_layers=0)
    assert [s.name for s in streams] == ["mla"]
    plan = build_kv_cache(streams, max_num_pages=4, device="cpu",
                          verbose=False)
    assert plan.num_slots == 61


# ── unpaged streams ───────────────────────────────────────────────────────
# Two halves: what the plan does for a flat stream, and what it does NOT do to
# everyone else's plan.


def _flat_streams(layers=(0, 1), h=8, d=64):
    kv = [("k", (h, d), torch.bfloat16), ("v", (h, d), torch.bfloat16)]
    return [KVStream("flat", layers=layers, components=kv,
                     kind=KVKind.FLAT)]


def test_unpaged_stream_gets_storage_but_no_group_or_page_table():
    with _free_memory(64 << 30):
        plan = build_kv_cache(_flat_streams(), kv_budget="1GiB",
                              max_seq_length=256, device="cpu", verbose=False)
    assert plan.flat_streams and not plan._layouts
    assert plan.groups == () and plan.num_slots == 0
    assert plan.target_page_bytes == 0
    assert plan.page_id_bytes == 0
    assert plan.flat_bytes == 2 * 256 * (2 * 8 * 64 * 2)
    assert plan.budget_bytes(plan.max_num_pages) == plan.flat_bytes


def test_an_unpaged_stream_leaves_the_paged_plan_alone():
    """The containment property, and the reason this is a stream flag rather
    than a planner special case: an unpaged stream must not move a single
    number in the plan the paged streams would have got."""
    with _free_memory(64 << 30):
        alone = build_kv_cache(_gpt_oss_shaped_streams(), max_num_pages=4,
                               device="cpu", verbose=False)
        together = build_kv_cache(
            _gpt_oss_shaped_streams() + _flat_streams(layers=(8, 9)),
            max_num_pages=4, max_seq_length=64, device="cpu", verbose=False)
    assert together.target_page_bytes == alone.target_page_bytes
    assert together.num_slots == alone.num_slots
    assert [(g.spec_name, g.block_size, g.window_size, g.entries_per_page)
            for g in together.groups] == \
           [(g.spec_name, g.block_size, g.window_size, g.entries_per_page)
            for g in alone.groups]
    # ...and the flat stream is nonetheless there and budgeted
    assert together.flat_bytes > 0
    assert together.budget_bytes(4) == alone.budget_bytes(4) + \
        together.flat_bytes


def test_unpaged_streams_refuse_every_knob_that_describes_a_page():
    """All three describe a page. Accepting one on a stream that has no page
    would silently ignore it -- a declared 512-token window that frees nothing
    reads as an allocation the model is not getting."""
    from dataclasses import fields as _fields

    kv = [("k", (8, 64), torch.bfloat16)]
    offenders = dict(window=128, compress_ratio=2, block_size_multiple_of=64)
    paging_knobs = {f.name for f in _fields(KVStream)} - {
        "name", "layers", "components", "kind"}
    assert paging_knobs == set(offenders), (
        f"KVStream paging knobs {sorted(paging_knobs)} are not all covered by "
        f"this test's table {sorted(offenders)}")

    for knob, value in offenders.items():
        stream = KVStream("flat", layers=(0,), components=kv,
                          kind=KVKind.FLAT, **{knob: value})
        with _raises(ValueError):
            stream._flat(capacity=128)
    # the same stream without the knob is accepted, so the loop above is not
    # passing because _flat refuses everything
    KVStream("flat", layers=(0,), components=kv, kind=KVKind.FLAT)._flat(128)


def test_attach_hands_out_flat_views_with_no_group_id():
    with _free_memory(64 << 30):
        plan = build_kv_cache(_flat_streams(), kv_budget="1GiB",
                              max_seq_length=256, device="cpu", verbose=False)
    mpk = _StubMPK()
    for layer in (0, 1):
        got = plan.attach(mpk, layer)
        # same key set a paged layer gets; group_id None, not 0 (0 is real)
        assert set(got) == {"k_cache", "v_cache", "group_id"}
        assert got["group_id"] is None
    assert len(mpk.attached) == 4
    k0 = plan._flat_views[("flat", 0)]["k"]
    assert k0.shape == (256, 8, 64) and k0.dtype == torch.bfloat16
    # the two layers do not alias, and K and V within a layer do not either
    assert k0.data_ptr() != plan._flat_views[("flat", 1)]["k"].data_ptr()
    assert k0.data_ptr() != plan._flat_views[("flat", 0)]["v"].data_ptr()


def test_attach_refuses_a_flat_cache_copied_off_the_plan():
    with _free_memory(64 << 30):
        plan = build_kv_cache(_flat_streams(), kv_budget="1GiB",
                              max_seq_length=256, device="cpu", verbose=False)
    view = plan._flat_views[("flat", 0)]["k"]
    copy = view.clone()
    assert copy.shape == view.shape and torch.equal(copy, view)
    with _raises(AssertionError):
        plan._assert_in_flat(copy, "copied")
    # ...and a view still on the buffer but no longer one token row per step
    strided = view[:, :4, :]
    assert strided.data_ptr() == view.data_ptr()
    with _raises(AssertionError):
        plan._assert_in_flat(strided, "strided")
    # the honest view passes, so the two above are not failing for free
    plan._assert_in_flat(view, "view")


# ── the declaration is mandatory (three-state kv_streams) ─────────────────


def test_every_registered_builder_declares_its_kv_streams():
    """A builder that inherits the base kv_streams has not declared anything."""
    from mirage.mpk.model_registry import _MODEL_BUILDERS
    from mirage.mpk.models.graph_builder import GraphBuilder
    import mirage.mpk.models  # noqa: F401  (registers the builders)

    assert _MODEL_BUILDERS, "no builders registered; the import above moved"
    undeclared = sorted({
        cls.__name__ for cls in _MODEL_BUILDERS.values()
        if getattr(cls, "kv_streams", None) is GraphBuilder.kv_streams})
    assert not undeclared, (
        f"{undeclared} inherit GraphBuilder.kv_streams; a model whose kernels "
        f"read the cache flat declares KVStream(..., kind=KVKind.FLAT)")

    with _raises(NotImplementedError):
        GraphBuilder.kv_streams(None)


def test_dflash_declares_one_unpaged_stream_at_its_own_layer_ids():
    from mirage.mpk.models.dflash.builder import dflash_kv_streams

    cfg = {"num_hidden_layers": 6, "num_key_value_heads": 8, "head_dim": 128}
    streams = dflash_kv_streams(cfg, layer_id_base=61)
    assert len(streams) == 1 and not streams[0].paged
    # off the target's ids, as Eagle3's draft stream is
    assert streams[0].layers == tuple(range(61, 67))
    assert streams[0].per_entry_bytes == 2 * 8 * 128 * 2
    # a window is NOT declared: the draft writes verifier context at absolute
    # slots, so SWA is a compute-time limit, not an allocation.
    assert streams[0].window == 0


def test_an_unpaged_plan_builds_a_persistent_kernel():
    """A model with no paged KV presents ZERO groups, and PersistentKernel
    takes them -- checked against the real constructor and capacity check,
    since MPK_NUM_KV_GROUPS becomes a -D and dimensions device-side arrays."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("needs a device: PersistentKernel attaches cuda tensors")
    from mirage.mpk.persistent_kernel import PersistentKernel
    from mirage.mpk.models.inkling.builder import inkling_kv_streams

    class _Cfg:
        num_hidden_layers = 66
        head_dim = 128

    S = 2048
    plan = build_kv_cache(inkling_kv_streams(_Cfg()),
                          kv_budget="4GiB", max_seq_length=S, verbose=False)
    meta = dict(plan.build_meta_tensors(max_seq_length=S,
                                        max_num_batched_requests=1))
    meta.update({
        "step": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "tokens": torch.zeros(1, S, dtype=torch.int64, device="cuda"),
        "input_tokens": torch.zeros(1, 1, dtype=torch.int64, device="cuda"),
        "output_tokens": torch.zeros(1, 1, dtype=torch.int64, device="cuda"),
        "num_new_tokens": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "prompt_lengths": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "qo_indptr_buffer": torch.zeros(2, dtype=torch.int32, device="cuda"),
    })
    pk = PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0, num_workers=96,
        num_local_schedulers=4, num_remote_schedulers=0, max_seq_length=S,
        max_num_batched_requests=1, max_num_batched_tokens=1,
        max_num_pages=plan.max_num_pages,
        kv_groups=plan.group_specs(), meta_tensors=meta, profiler_tensor=None,
        trace_name=None, spec_decode_config=None, use_cutlass_kernel=False)
    assert pk.kv_groups == []
    pk._check_kv_capacity()          # nothing demands a page, and it fits

    kv = plan.attach(pk, 0)
    assert kv["group_id"] is None
    k = kv["k_cache"]
    assert k.num_dims == 3
    assert [k.dim(i) for i in range(3)] == [S, 16, 128]

