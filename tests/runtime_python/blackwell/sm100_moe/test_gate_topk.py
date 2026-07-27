import sys

import torch
from torch.nn import functional as F

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

NUM_EXPERTS_LIST = [128, 256]
NUM_TOPKS = [8]
# One PASS of the kernel covers WARP_SIZE * VPT / num_experts * 8 rows -- 8 at
# 256 experts, 16 at 128 (TopkConstants picks VPT=8 in both cases). Before
# M3-I5b every row past the first pass was silently dropped (M2-I9), which is
# why this test only ever ran ONE row. The sizes past the pass width are the
# regression that keeps the row-tile loop honest; the odd ones also exercise
# the partial-warp shuffle mask at 128 experts (THREADS_PER_ROW = 16).
BATCH_SIZES = [1, 8, 9, 16, 17, 33]

WEIGHT_RTOL = 1e-2
WEIGHT_ATOL = 1e-2

# ============================================================================
# Tie-aware comparison
#
# torch.randn synthetic bf16 logits hit genuine top-8 boundary ties (the k-th
# selected logit and the best-unselected logit exactly equal, bit-for-bit in
# bf16) at several (batch_size, num_experts) cells in the coverage sweep
# below -- confirmed on hardware: the per-bs coverage run showed small,
# localized routing-index diffs at a handful of (expert, token) cells, and
# the window run's own diagnostic printed a "top-8 boundary tie" for one of
# them.
#
# A CPU simulation of the kernel's own documented algorithm (successive
# argmax with a lower-index tie-break, exactly as coded in
# topk_softmax_sm100.cuh: "Argmax reduce across subgroup with index
# tie-breaker (prefer lower index)"), run at the real (batch_size,
# num_experts) coverage cells, showed the ambiguity is NOT limited to the
# literal (k-1, k) boundary pair: a value can repeat among two or more
# experts ENTIRELY INSIDE the top-k (e.g. ranks 4, 5, 6 all sharing one
# bf16 value). That leaves the SELECTED SET unambiguous but the specific
# RANK LABEL within the tied group ambiguous -- MPK deterministically
# assigns ranks by ascending expert index, but torch.topk's own tie-break
# is not documented or guaranteed to agree (confirmed empirically: it does
# not, in general). Mechanistically this is the identical phenomenon as the
# boundary case (an exact bf16 value collision resolved differently by two
# independent implementations, on a never-shipped comparison path --
# M2-I9/P5 history: MPK's tie-break is verified lower-index on both sides
# against its own semantics), so it gets the identical treatment. A token
# is therefore classified as TIE if ANY adjacent pair in the sorted top-
# (k+1) window is exactly equal (equivalently: any duplicate value among
# the top-k selected values and the best-unselected one) -- this subsumes
# the literal boundary case as the i=k-1 instance of the same check.
#
# At a TIE token, accept either expert choice, provided:
#   (a) the selected SET's logit multiset equals the reference top-k logit
#       multiset (bitwise on bf16 values), and
#   (b) the renormalized weight matches the reference renormalization
#       recomputed on MPK's OWN chosen set (not torch's).
# Non-tie tokens (no duplicate anywhere in the top-(k+1) window) keep the
# original, strict, index-exact comparison: this rewrite must not accept a
# genuine off-by-one wrong-expert bug that ISN'T tie-explained -- see
# `_self_check_catches_non_tie_bug` below, which is the negative control for
# exactly that failure mode.
# ============================================================================


def compute_reference(gating_output_ref, num_topk):
    """Reference: select topk then softmax over those values (unchanged from
    the pre-existing test's semantics)."""
    gating_output_f = gating_output_ref.to(torch.float)
    norm_gating_output = gating_output_f - gating_output_f.amax(dim=1, keepdim=True)
    torch_softmax = F.softmax(norm_gating_output, dim=1, dtype=torch.float)
    torch_topk_values, torch_topk_indices = torch.topk(torch_softmax, num_topk, dim=1)
    torch_topk_weights = torch_topk_values / torch_topk_values.sum(dim=-1, keepdim=True)
    return torch_softmax, torch_topk_indices, torch_topk_weights


def find_tie_tokens(gating_output_ref, num_topk):
    """A token is TIE iff there is an exact duplicate value anywhere among
    the top-(num_topk + 1) sorted logits -- the k-th/(k+1)-th "boundary"
    pair the spec calls out, OR a duplicate fully inside the top-k (SET
    unambiguous, in-group RANK ambiguous). Since the array is sorted
    descending, "any duplicate in the first k+1 entries" is equivalent to
    "some ADJACENT pair in that window is equal", checked directly below.
    bf16 -> fp32 is a lossless upcast (no rounding), so sorting/comparing in
    fp32 here is an exact bitwise comparison on the original bf16 values."""
    logits_f = gating_output_ref.to(torch.float)
    sorted_logits, _ = torch.sort(logits_f, dim=1, descending=True)
    adjacent_equal = sorted_logits[:, :num_topk] == sorted_logits[:, 1 : num_topk + 1]
    is_tie = adjacent_equal.any(dim=1)
    return is_tie, sorted_logits


def kernel_selected_experts(mpk_routing_indices, num_topk, batch_size, num_expert):
    """Invert mpk_routing_indices [num_expert, batch] (expert -> 1-based
    rank, 0 if unselected) into [batch, num_topk] (rank -> expert), -1 where
    a rank was never filled (a row-coverage bug)."""
    device = mpk_routing_indices.device
    sel = torch.full((batch_size, num_topk), -1, dtype=torch.long, device=device)
    nz = mpk_routing_indices.nonzero()
    if nz.numel() > 0:
        ranks = mpk_routing_indices[nz[:, 0], nz[:, 1]].long() - 1
        sel[nz[:, 1], ranks] = nz[:, 0]
    return sel


def tie_aware_compare(
    gating_output_ref,
    torch_softmax,
    torch_topk_indices,
    torch_topk_weights,
    mpk_routing_indices,
    topk_weights,
    num_topk,
    batch_size,
    num_expert,
):
    """Per-token tie-aware comparison. Raises AssertionError (message
    includes the tied logit values, so a failure log is self-diagnosing) on
    the first genuine mismatch. Returns (n_tie, is_tie, kernel_sel) on
    success."""
    is_tie, sorted_logits = find_tie_tokens(gating_output_ref, num_topk)
    kernel_sel = kernel_selected_experts(
        mpk_routing_indices, num_topk, batch_size, num_expert
    )

    for t in range(batch_size):
        kernel_experts_t = kernel_sel[t]
        if (kernel_experts_t < 0).any():
            raise AssertionError(
                f"token {t}: kernel left a rank unfilled -- row not fully "
                f"covered by the row-tile loop: {kernel_experts_t.tolist()}"
            )

        ref_experts_t = torch_topk_indices[t]

        if not bool(is_tie[t]):
            # NON-tie: exact rank-order index equality + the existing weight
            # tolerance (both unchanged from the pre-existing test).
            if not torch.equal(kernel_experts_t, ref_experts_t):
                # Not classified as a boundary tie -- but report whether the
                # specific differing slot(s) are secretly value-tied anyway
                # (an INTERNAL, non-boundary tie the spec doesn't cover), so
                # a human can tell "real defect" from "tie our definition
                # missed" at a glance instead of re-deriving it from raw
                # logits.
                diverge = (kernel_experts_t != ref_experts_t).nonzero().flatten()
                notes = []
                for r in diverge.tolist():
                    mv = gating_output_ref[t, kernel_experts_t[r]].item()
                    rv = gating_output_ref[t, ref_experts_t[r]].item()
                    tag = (
                        "VALUE-EQUAL (internal tie?)"
                        if mv == rv
                        else "DIFFERENT VALUES"
                    )
                    notes.append(
                        f"    rank {r}: mpk expert {kernel_experts_t[r].item()} "
                        f"(logit={mv:.6f}) vs ref expert {ref_experts_t[r].item()} "
                        f"(logit={rv:.6f}) -- {tag}"
                    )
                raise AssertionError(
                    f"token {t}: NON-tie boundary (k-1,k logits = "
                    f"{sorted_logits[t, num_topk - 1].item():.6f}, "
                    f"{sorted_logits[t, num_topk].item():.6f}, not equal) but "
                    f"expert selection differs -- NOT tie-explained\n"
                    f"  mpk rank-order = {kernel_experts_t.tolist()}\n"
                    f"  ref rank-order = {ref_experts_t.tolist()}\n"
                    + "\n".join(notes)
                )
            kernel_w = topk_weights[t]
            ref_w = torch_topk_weights[t]
            if not torch.allclose(kernel_w, ref_w, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL):
                raise AssertionError(
                    f"token {t}: NON-tie weight mismatch beyond tolerance "
                    f"(rtol={WEIGHT_RTOL}, atol={WEIGHT_ATOL})\n"
                    f"  mpk = {kernel_w.tolist()}\n  ref = {ref_w.tolist()}"
                )
        else:
            # TIE: (a) value-multiset equality, (b) renorm recomputed on
            # MPK's own chosen set (not compared cross-set against ref).
            ref_multiset, _ = torch.sort(sorted_logits[t, :num_topk])
            kernel_multiset, _ = torch.sort(
                gating_output_ref[t, kernel_experts_t].to(torch.float)
            )
            tie_val = sorted_logits[t, num_topk - 1].item()
            if not torch.equal(kernel_multiset, ref_multiset):
                raise AssertionError(
                    f"token {t}: TIE boundary (value={tie_val:.6f}) but the "
                    f"selected SET's logit multiset differs from the "
                    f"reference top-k multiset -- NOT tie-explained\n"
                    f"  mpk experts  = {kernel_experts_t.tolist()}\n"
                    f"  mpk multiset = {kernel_multiset.tolist()}\n"
                    f"  ref multiset = {ref_multiset.tolist()}"
                )
            selected_softmax = torch_softmax[t, kernel_experts_t]
            recomputed_w = selected_softmax / selected_softmax.sum()
            kernel_w = topk_weights[t]
            if not torch.allclose(
                recomputed_w, kernel_w, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL
            ):
                raise AssertionError(
                    f"token {t}: TIE boundary (value={tie_val:.6f}) weight "
                    f"mismatch against the reference renormalized on MPK's "
                    f"own set (rtol={WEIGHT_RTOL}, atol={WEIGHT_ATOL})\n"
                    f"  mpk        = {kernel_w.tolist()}\n"
                    f"  recomputed = {recomputed_w.tolist()}"
                )

    n_tie = int(is_tie.sum().item())
    return n_tie, is_tie, kernel_sel


def tie_aware_mask_check(
    is_tie, torch_topk_indices, kernel_sel, mpk_active_ids, num_expert, batch_size
):
    """Active-expert-mask check, made tie-aware: the expected active set uses
    the reference's own pick for non-tie tokens (must match exactly) and
    MPK's own already-validated-equivalent pick for tie tokens, since torch's
    arbitrary tie-break is not itself something we are validating."""
    device = mpk_active_ids.device
    expected_mask = torch.zeros((num_expert,), device=device, dtype=torch.int32)
    for t in range(is_tie.shape[0]):
        experts_t = kernel_sel[t] if bool(is_tie[t]) else torch_topk_indices[t]
        expected_mask[experts_t] = 1

    num_active = int(mpk_active_ids[-1].item())
    recon_mask = torch.zeros((num_expert,), device=device, dtype=torch.int32)
    if num_active > 0:
        active_ids = mpk_active_ids[:num_active].to(torch.long)
        recon_mask.index_fill_(0, active_ids, 1)

    if not torch.equal(recon_mask, expected_mask):
        diff = (recon_mask != expected_mask).nonzero().flatten().tolist()
        raise AssertionError(
            f"active-expert mask mismatch at expert(s) {diff}\n"
            f"  recon    = {recon_mask.tolist()}\n"
            f"  expected = {expected_mask.tolist()}"
        )
    return num_active


# ============================================================================
# Negative self-check
#
# Guards against the exact failure mode this rewrite exists to avoid: a
# tie-aware relaxation that quietly also swallows a genuine off-by-one bug.
# Pure CPU/tensor logic, no CUDA extension needed -- run at import (always,
# before the GPU sweep) and standalone via `--self-check-only`.
# ============================================================================


def _build_routing(topk_indices, num_expert, batch_size, num_topk, device):
    r = torch.zeros((num_expert, batch_size), dtype=torch.int32, device=device)
    for t in range(batch_size):
        for k_idx in range(num_topk):
            r[topk_indices[t, k_idx], t] = k_idx + 1
    return r


def _self_check_catches_non_tie_bug():
    """Build a small CPU-only synthetic batch with:
      row0: strictly decreasing, no ties anywhere;
      row1: a BOUNDARY tie (the k-th selected logit == the best-unselected
            logit, i.e. the literal (k-1, k) pair);
      row2: an INTERNAL tie -- two experts tied well inside the top-k (ranks
            1 and 2 of 4), nowhere near the boundary. This is the case a
            naive "only check position (k-1, k)" implementation MISSES: the
            SET is unambiguous but the specific rank label within the tied
            pair is not, so a naive strict rank-order comparison would
            spuriously fail on real hardware data even though nothing is
            wrong -- this is exactly what an end-to-end CPU simulation of
            the kernel's own documented tie-break, run at the real coverage
            cells, surfaced empirically (dominant over the literal boundary
            case at these problem sizes) before this test ever touched a
            GPU.
    Confirm the correct output passes on all three rows, then confirm three
    distinct corruptions are REJECTED:
      1. a wrong (non-tied) expert substituted on the NON-tie row;
      2. a wrong (non-tied) expert substituted on the BOUNDARY-tie row, at a
         rank that is NOT part of the tie;
      3. a wrong (non-tied) expert substituted on the INTERNAL-tie row, at a
         rank that is NOT part of the tie.
    In each case the multiset check must catch the real defect regardless of
    whether the row also happens to contain a (legitimate) tie elsewhere.
    """
    device = "cpu"
    num_expert = 16
    num_topk = 4
    batch_size = 3

    row0 = torch.arange(num_expert, 0, -1, dtype=torch.float32)  # 16..1, no ties
    row1 = row0.clone()
    row1[num_topk] = row1[num_topk - 1]  # boundary tie: ranks 3,4 both == 13
    row2 = row0.clone()
    row2[2] = row2[1]  # internal tie: ranks 1,2 both == 15 (nowhere near rank 4)

    gating = torch.stack([row0, row1, row2]).to(torch.bfloat16)
    torch_softmax, torch_topk_indices, torch_topk_weights = compute_reference(
        gating, num_topk
    )

    is_tie, _ = find_tie_tokens(gating, num_topk)
    assert not bool(is_tie[0]), "self-check setup bug: row 0 must NOT be a tie"
    assert bool(is_tie[1]), "self-check setup bug: row 1 must BE a boundary tie"
    assert bool(is_tie[2]), "self-check setup bug: row 2 must BE an internal tie"

    good_routing = _build_routing(
        torch_topk_indices, num_expert, batch_size, num_topk, device
    )
    good_weights = torch_topk_weights.clone().to(torch.float32)

    # 0. The untouched "correct" output must pass on all three rows
    #    (exercises the non-tie branch and the tie branch -- both boundary
    #    and internal -- on their success path).
    tie_aware_compare(
        gating,
        torch_softmax,
        torch_topk_indices,
        torch_topk_weights,
        good_routing,
        good_weights,
        num_topk,
        batch_size,
        num_expert,
    )

    def _corrupt_and_expect_catch(row, victim_rank, label):
        bad_routing = good_routing.clone()
        victim_expert = torch_topk_indices[row, victim_rank].item()
        unselected = [
            e for e in range(num_expert) if e not in torch_topk_indices[row].tolist()
        ]
        wrong_expert = unselected[0]
        bad_routing[victim_expert, row] = 0
        bad_routing[wrong_expert, row] = victim_rank + 1

        caught = False
        try:
            tie_aware_compare(
                gating,
                torch_softmax,
                torch_topk_indices,
                torch_topk_weights,
                bad_routing,
                good_weights,
                num_topk,
                batch_size,
                num_expert,
            )
        except AssertionError:
            caught = True
        if not caught:
            raise AssertionError(
                f"SELF-CHECK FAILED ({label}): tie_aware_compare accepted a "
                f"non-tied wrong-expert substitution -- the tie-aware "
                f"relaxation is too permissive and would mask a genuine "
                f"kernel defect."
            )

    # 1. NON-tie row, lowest-ranked slot.
    _corrupt_and_expect_catch(0, num_topk - 1, "non-tie row")
    # 2. BOUNDARY-tie row, first (unambiguous) rank.
    _corrupt_and_expect_catch(1, 0, "boundary-tie row")
    # 3. INTERNAL-tie row, last (unambiguous) rank -- the tie is at ranks
    #    1,2; rank 3 is untouched by it.
    _corrupt_and_expect_catch(2, num_topk - 1, "internal-tie row")

    print(
        "  self-check: wrong-expert corruption on the non-tie row, the "
        "boundary-tie row, AND the internal-tie row are all correctly "
        "REJECTED"
    )


def main():
    if "--self-check-only" in sys.argv:
        _self_check_catches_non_tie_bug()
        print("\nSelf-check OK (--self-check-only): exiting without the GPU sweep.")
        return

    # Cheap, CPU-only, run before anything touches the GPU or the compiled
    # extension.
    _self_check_catches_non_tie_bug()

    import runtime_kernel_blackwell

    g = torch.Generator(device="cuda").manual_seed(1234)
    results = []

    for batch_size in BATCH_SIZES:
        for num_expert in NUM_EXPERTS_LIST:
            for num_topk in NUM_TOPKS:
                case = f"bs={batch_size:2d} experts={num_expert:3d} topk={num_topk}"
                print(
                    f"\n=== Testing batch_size = {batch_size} "
                    f"num_experts = {num_expert} num_topk = {num_topk} ==="
                )

                # Random gating outputs (pre-softmax logits) should be using
                # bfloat16 but the bfloat16 range is a bit small for randn so
                # we test with float here
                gating_output = torch.randn(
                    (batch_size, num_expert),
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=g,
                )

                topk_weights = torch.empty(
                    batch_size, num_topk, device="cuda", dtype=torch.float
                )
                mpk_routing_indices = torch.zeros(
                    (num_expert, batch_size), device="cuda", dtype=torch.int32
                )
                mpk_active_ids = torch.empty(
                    (num_expert + 1,), device="cuda", dtype=torch.int32
                )

                # Preserve a copy of inputs for reference before kernel
                # mutates input
                gating_output_ref = gating_output.clone()

                # Run fused topk softmax
                runtime_kernel_blackwell.topk_softmax_sm100(
                    gating_output, topk_weights, mpk_routing_indices, mpk_active_ids
                )

                torch_softmax, torch_topk_indices, torch_topk_weights = (
                    compute_reference(gating_output_ref, num_topk)
                )

                is_tie_expected, _ = find_tie_tokens(gating_output_ref, num_topk)
                n_tie_expected = int(is_tie_expected.sum().item())

                try:
                    n_tie, is_tie, kernel_sel = tie_aware_compare(
                        gating_output_ref,
                        torch_softmax,
                        torch_topk_indices,
                        torch_topk_weights,
                        mpk_routing_indices,
                        topk_weights,
                        num_topk,
                        batch_size,
                        num_expert,
                    )
                    tie_aware_mask_check(
                        is_tie,
                        torch_topk_indices,
                        kernel_sel,
                        mpk_active_ids,
                        num_expert,
                        batch_size,
                    )

                    num_active = int(mpk_active_ids[-1].item())
                    print(f"Active experts: {num_active}")
                    if num_active > 0:
                        print(
                            f"Active expert IDs: "
                            f"{mpk_active_ids[:num_active].cpu().tolist()}"
                        )
                    print(
                        f"Test passed! ({n_tie} of {batch_size} tokens were "
                        f"boundary ties)"
                    )
                    results.append((case, "PASS", n_tie, ""))
                except AssertionError as e:
                    print(f"Test FAILED: {e}")
                    results.append((case, "FAIL", n_tie_expected, str(e)))

                # Warm-up + benchmark: unchanged, runs regardless of the
                # correctness verdict above (matches the pre-existing
                # behavior of this file; abort-on-first-failure stays OFF so
                # every (bs, experts) cell is exercised and timed).
                for _ in range(16):
                    runtime_kernel_blackwell.topk_softmax_sm100(
                        gating_output, topk_weights, mpk_routing_indices, mpk_active_ids
                    )

                torch.cuda.synchronize()
                starter, ender = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                repetitions = 1000
                starter.record()
                for rep in range(repetitions):
                    runtime_kernel_blackwell.topk_softmax_sm100(
                        gating_output, topk_weights, mpk_routing_indices, mpk_active_ids
                    )
                ender.record()
                torch.cuda.synchronize()
                total_time = starter.elapsed_time(ender)
                avg_time = total_time / repetitions
                print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for case, status, n_tie, _ in results:
        print(f"  [{status}] {case}  tie_tokens={n_tie}")

    failures = [r for r in results if r[1] != "PASS"]
    if failures:
        print(f"\n{len(failures)} of {len(results)} case(s) FAILED:")
        for case, _, _, detail in failures:
            print(f"\n--- {case} ---\n{detail}")
        sys.exit(1)

    print(f"\nAll {len(results)} cases PASSED.")


if __name__ == "__main__":
    main()
