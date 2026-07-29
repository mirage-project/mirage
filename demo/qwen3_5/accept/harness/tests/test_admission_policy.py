"""M4-I4: the admission-cap policy has exactly ONE authority, and it is arithmetic.

Three things this pins, all CPU-only:

1. **Resolution semantics.** `policy | none | auto | <int>` resolve as documented,
   and junk raises rather than silently running an unintended configuration.
2. **No second copy of the policy.** The runtime entry points must DERIVE from
   `admission_policy` instead of restating a batch-size list. A grep-level test is
   crude, but the failure it guards against is exactly a textual second copy that
   drifts (M3-I9's policy lived only in prose and the two statements disagreed for
   two days).
3. **The mechanism the policy rests on.** The CPU-side admission replay
   (`opt/m3i9/protocol_sim.py`) must reproduce, for the pinned 256/1024 geometry,
   the iteration-count asymmetry the landing is argued from: a large win at bs16,
   a small one at bs8, and ~none at bs<=4 -- which is why the bs4 win has to come
   from graph width per iteration, not from iteration count. If a change to the
   replay or to `mbt` ever flattens that, the policy's stated mechanism is wrong
   and the A/B has to be re-run.

Run directly: `python3 tests/test_admission_policy.py`.
"""
import sys
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent
ACC = HARNESS.parent
sys.path.insert(0, str(ACC))
sys.path.insert(0, str(ACC / "opt" / "m3i9"))

import admission_policy as ap  # noqa: E402
from protocol_sim import simulate  # noqa: E402

MBT = 16                      # the pinned max_num_batched_tokens (bench-protocol.md)


class ResolutionTest(unittest.TestCase):
    def test_auto_cap_is_an_equal_share_floored_at_one(self):
        self.assertEqual([ap.auto_cap(MBT, bs) for bs in (1, 2, 4, 8, 16)],
                         [16, 8, 4, 2, 1])
        self.assertEqual(ap.auto_cap(MBT, 32), 1)      # floor, never 0

    def test_policy_caps_from_cap_min_batch_size_up(self):
        for bs in ap.PROTOCOL_BATCH_SIZES:
            expect = ap.CAP_MODE if bs >= ap.CAP_MIN_BATCH_SIZE else None
            self.assertEqual(ap.policy_cap(bs), expect, f"bs={bs}")

    def test_bs1_policy_would_be_a_no_op_even_if_it_were_capped(self):
        # auto at bs1 is mbt itself, so the extra min() in prepare_next_batch can
        # never fire. Excluding bs1 is bookkeeping, not a perf claim.
        self.assertEqual(ap.auto_cap(MBT, 1), MBT)

    def test_none_and_the_string_none_are_the_pre_policy_runtime(self):
        for value in (None, "none"):
            self.assertIsNone(ap.resolve(value, 16))
            self.assertIsNone(ap.resolve_int(value, MBT, 16))

    def test_policy_resolves_per_batch_size(self):
        self.assertEqual(ap.resolve_int("policy", MBT, 16), 1)
        self.assertEqual(ap.resolve_int("policy", MBT, 8), 2)
        self.assertEqual(ap.resolve_int("policy", MBT, 4), 4)
        below = ap.CAP_MIN_BATCH_SIZE - 1
        if below >= 1:
            self.assertIsNone(ap.resolve_int("policy", MBT, below))

    def test_auto_forces_the_cap_even_where_the_policy_would_not(self):
        self.assertEqual(ap.resolve_int("auto", MBT, 1), MBT)

    def test_explicit_int_passes_through_as_int(self):
        self.assertEqual(ap.resolve_int(3, MBT, 8), 3)
        self.assertEqual(ap.resolve_int("3", MBT, 8), 3)

    def test_junk_raises_rather_than_running_something_else(self):
        for bad in ("", "yes", "0", 0, -1, True, 1.5, [], "auto2"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ap.validate(bad)

    def test_describe_names_the_compiled_value_and_the_source(self):
        on = ap.describe("policy", MBT, 16)
        self.assertIn("MPK_MAX_TOKENS_PER_REQUEST=1", on)
        self.assertIn("shipped policy", on)
        self.assertIn("OFF", ap.describe("none", MBT, 16))
        self.assertIn("no-op", ap.describe("auto", MBT, 1))

    def test_summary_is_self_describing(self):
        s = ap.summary()
        self.assertEqual(s["authority"],
                         "demo/qwen3_5/accept/admission_policy.py")
        self.assertEqual(set(s["per_bs"]), set(ap.PROTOCOL_BATCH_SIZES))


class SingleAuthorityTest(unittest.TestCase):
    """The policy must not be restated anywhere that can drift away from it."""

    ENTRY_POINTS = ("mpk_engine_run.py", "harness/gate_ac3_stable.py")

    def test_entry_points_import_the_authority(self):
        for rel in self.ENTRY_POINTS:
            src = (ACC / rel).read_text()
            self.assertIn("admission_policy", src, rel)

    def test_entry_points_do_not_recompute_auto(self):
        # `max(1, mbt // bs)` living in two files is the divergence bug this
        # module exists to prevent; only admission_policy.py may spell it.
        for rel in self.ENTRY_POINTS:
            src = (ACC / rel).read_text()
            self.assertNotIn("max(1, self.mbt //", src, rel)
            self.assertNotIn("max(1, mbt //", src, rel)

    def test_engine_cli_default_is_the_policy(self):
        src = (ACC / "mpk_engine_run.py").read_text()
        self.assertIn('ap.add_argument("--per-request-token-cap", default="policy"',
                      src)

    def test_the_gate_certifies_the_shipped_configuration(self):
        src = (ACC / "harness" / "gate_ac3_stable.py").read_text()
        self.assertIn('p.add_argument("--per-request-token-cap", default="policy")',
                      src)
        sh = (ACC / "harness" / "gate_ac3_stable.sh").read_text()
        self.assertIn('CAP="policy"', sh)


class MechanismTest(unittest.TestCase):
    """The replay must still show the asymmetry the landing is argued from."""

    GEOMETRY = dict(plen=256, msl=1280)      # the pinned 256/1024 benchmark shape

    def _iters(self, bs, cap):
        return simulate([self.GEOMETRY["plen"]] * bs, MBT,
                        self.GEOMETRY["msl"], cap=cap)["n_iterations"]

    def test_iteration_count_win_is_bs16_only(self):
        ratios = {}
        for bs in ap.PROTOCOL_BATCH_SIZES:
            uncapped = self._iters(bs, None)
            capped = self._iters(bs, ap.auto_cap(MBT, bs))
            ratios[bs] = uncapped / capped
        self.assertAlmostEqual(ratios[1], 1.0, places=6)      # provable no-op
        self.assertLess(ratios[2], 1.01)
        self.assertLess(ratios[4], 1.01)
        self.assertGreater(ratios[8], 1.02)                   # small but real
        self.assertLess(ratios[8], 1.10)
        self.assertGreater(ratios[16], 1.40)                  # the big term

    def test_the_cap_shrinks_the_widest_chunk_at_every_capped_bs(self):
        # The second mechanism: same token budget, spread over more requests.
        # This is what pays at bs4/bs8, where the iteration counts barely move.
        for bs in ap.PROTOCOL_BATCH_SIZES:
            if bs < 2:
                continue
            sim_u = simulate([256] * bs, MBT, 1280, cap=None)
            sim_c = simulate([256] * bs, MBT, 1280, cap=ap.auto_cap(MBT, bs))
            # during prefill the uncapped arm hands one slot the whole budget
            self.assertEqual(max(i["max_chunk"] for i in sim_u["iters"]), MBT, bs)
            self.assertEqual(max(i["max_chunk"] for i in sim_c["iters"]),
                             ap.auto_cap(MBT, bs), bs)
            # ... and therefore activates fewer slots per iteration
            act_u = max(i["n_active"] for i in sim_u["iters"])
            act_c = max(i["n_active"] for i in sim_c["iters"])
            self.assertLessEqual(act_u, act_c, bs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
