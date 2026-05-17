"""
View / split / narrow IR-level construction tests.

These are pure Python tests that exercise Graph::view / Graph::split /
Graph::narrow without compiling or running anything on the GPU. They
verify:
  - View DTensors carry the correct base_guid, view_offset, shape.
  - Multi-level views flatten to root storage at construction time.
  - data_offset arithmetic for split (contiguous slabs) is correct.
  - Error paths raise.
"""

import sys

import mirage


def _row_major_byte_stride(dims, dtype_size, dim_idx):
    """Compute the row-major byte stride at dim_idx."""
    stride = dtype_size
    for i in range(len(dims) - 1, dim_idx, -1):
        stride *= dims[i]
    return stride


def _expect(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_view_basics():
    print("[test_view_basics]")
    g = mirage.new_kernel_graph()
    # Storage tensor T: shape (4, 8), bfloat16 (2 bytes/element).
    T = g.new_input(dims=(4, 8), dtype=mirage.bfloat16)
    _expect(T.is_virtual is False, "fresh storage tensor must not be virtual")
    _expect(T.base_guid == 0, "fresh storage tensor must have base_guid == 0")
    _expect(T.view_offset == 0, "fresh storage tensor must have view_offset == 0")
    _expect(T.shape == (4, 8), f"shape mismatch: {T.shape}")

    # view() — same number of elements, different shape.
    V = g.view(T, [2, 16])
    _expect(V.is_virtual, "view() result must be virtual")
    _expect(V.base_guid == T.guid, "view.base_guid must point to root storage")
    _expect(V.view_offset == 0, "pure reshape preserves view_offset = 0")
    _expect(V.shape == (2, 16), f"view shape mismatch: {V.shape}")
    _expect(V.guid != T.guid, "view must have a fresh guid")

    # view shape must match total elements.
    try:
        g.view(T, [3, 8])
        print("FAIL: view with mismatched element count should raise")
        sys.exit(1)
    except RuntimeError as e:
        _expect("total element count" in str(e), f"unexpected error: {e}")

    print("  OK")


def test_split_along_outer_dim():
    print("[test_split_along_outer_dim]")
    g = mirage.new_kernel_graph()
    # Shape (6, 4) bfloat16; split into 3 equal slabs along dim 0.
    T = g.new_input(dims=(6, 4), dtype=mirage.bfloat16)
    slabs = g.split(T, 3, 0)
    _expect(len(slabs) == 3, f"expected 3 slabs, got {len(slabs)}")
    dtype_size = 2  # bfloat16
    for i, sl in enumerate(slabs):
        _expect(sl.shape == (2, 4), f"slab[{i}] shape {sl.shape}")
        _expect(sl.base_guid == T.guid, f"slab[{i}].base_guid")
        expected = i * 2 * 4 * dtype_size  # i * slab_rows * cols * sizeof(bf16)
        _expect(sl.view_offset == expected,
                f"slab[{i}].view_offset = {sl.view_offset} != {expected}")
    print("  OK")


def test_split_along_inner_dim():
    print("[test_split_along_inner_dim]")
    g = mirage.new_kernel_graph()
    # Shape (4, 12) bfloat16; split into [4, 4, 4] along dim 1.
    # Each slab has shape (4, 4); slabs are NOT contiguous (parent stride
    # for dim 0 is still 12 bf16 = 24 bytes, while a slab's logical row is
    # only 4 bf16 = 8 bytes). View_offset is in bytes along dim 1.
    T = g.new_input(dims=(4, 12), dtype=mirage.bfloat16)
    slabs = g.split(T, [4, 4, 4], 1)
    _expect(len(slabs) == 3, f"expected 3 slabs, got {len(slabs)}")
    dtype_size = 2
    for i, sl in enumerate(slabs):
        _expect(sl.shape == (4, 4), f"slab[{i}] shape {sl.shape}")
        _expect(sl.base_guid == T.guid, f"slab[{i}].base_guid")
        expected = i * 4 * dtype_size  # i * slab_cols * sizeof(bf16)
        _expect(sl.view_offset == expected,
                f"slab[{i}].view_offset = {sl.view_offset} != {expected}")
    print("  OK")


def test_narrow():
    print("[test_narrow]")
    g = mirage.new_kernel_graph()
    T = g.new_input(dims=(8, 6), dtype=mirage.bfloat16)
    n = g.narrow(T, 0, 2, 3)  # rows [2, 5)
    _expect(n.shape == (3, 6), f"narrow shape {n.shape}")
    _expect(n.base_guid == T.guid, "narrow.base_guid")
    _expect(n.view_offset == 2 * 6 * 2, f"narrow.view_offset = {n.view_offset}")

    # Bounds check.
    try:
        g.narrow(T, 0, 7, 5)  # 7+5 > 8
        print("FAIL: out-of-range narrow should raise")
        sys.exit(1)
    except RuntimeError as e:
        _expect("out of range" in str(e), f"unexpected error: {e}")
    print("  OK")


def test_multi_level_view_flattens_to_root():
    print("[test_multi_level_view_flattens_to_root]")
    g = mirage.new_kernel_graph()
    T = g.new_input(dims=(4, 8), dtype=mirage.bfloat16)
    # T.split → V; V.view → W; W.narrow → X. base_guid must all be T.guid.
    parts = g.split(T, 2, 0)  # 2 slabs of (2, 8)
    V = parts[1]  # second slab, view_offset = 2 * 8 * 2 = 32 bytes
    _expect(V.view_offset == 32, f"V.view_offset = {V.view_offset}")

    # Pure reshape: same element count, same view_offset.
    W = g.view(V, [4, 4])
    _expect(W.base_guid == T.guid, "W must flatten to root")
    _expect(W.view_offset == V.view_offset, "view() preserves view_offset")

    # Narrow along dim 0 of W (shape 4, 4): take rows [1, 3).
    X = g.narrow(W, 0, 1, 2)
    _expect(X.base_guid == T.guid, "X must flatten to root")
    # X.view_offset = W.view_offset + 1 * 4 * sizeof(bf16)
    _expect(X.view_offset == V.view_offset + 1 * 4 * 2,
            f"X.view_offset = {X.view_offset}")
    _expect(X.shape == (2, 4), f"X.shape = {X.shape}")
    print("  OK")


def test_split_then_view_independent_guids():
    print("[test_split_then_view_independent_guids]")
    g = mirage.new_kernel_graph()
    T = g.new_input(dims=(6, 4), dtype=mirage.bfloat16)
    slabs = g.split(T, 3, 0)
    guids = {sl.guid for sl in slabs}
    _expect(len(guids) == 3, "each split slab must have a unique guid")
    _expect(T.guid not in guids, "split slab guids must differ from parent")
    print("  OK")


def main():
    test_view_basics()
    test_split_along_outer_dim()
    test_split_along_inner_dim()
    test_narrow()
    test_multi_level_view_flattens_to_root()
    test_split_then_view_independent_guids()
    print("ALL VIEW CONSTRUCTION TESTS PASSED")


if __name__ == "__main__":
    main()
