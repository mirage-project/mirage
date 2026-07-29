#!/usr/bin/env python3
"""Tests for ``final/integrity.py``.  No GPU.

The interesting property is that the gate reads the PINNED ``.pm/accept.sh`` and
refuses when the flags it was handed disagree -- so a caller cannot hand it looser
bounds.  These tests exercise that with a synthetic pinned file (the real one is
mode 0555 and must not be touched), plus a smoke run against the real accept dir.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]
ACC = FINAL.parent
REPO = ACC.parents[2]
INTEG = FINAL / "integrity.py"
sys.path.insert(0, str(FINAL))
from integrity import parse_accept_sh, sha256_file      # noqa: E402

PINNED_SHAPE = '''#!/bin/bash
set -euo pipefail
MODEL_ID="Qwen/Qwen3.5-35B-A3B-FP8"
BATCH_SIZES="1 2 4 8 16"
PROMPTS=".pm/eval/prompts.jsonl"
PROMPTS_SHA="deadbeef"
CORRECT_NEW_TOKENS=64
MIN_INPUT_LEN=64
MIN_OUTPUT_LEN=256
E2E_FACTOR_MAX="1.25"
BASELINE="vllm"
HARNESS="workspace/demo/qwen3_5/accept/final.sh"
'''


class ParseAcceptShTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="final_integ_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parses_the_pinned_shape(self):
        p = self.tmp / "accept.sh"
        p.write_text(PINNED_SHAPE)
        got = parse_accept_sh(p)
        self.assertEqual(got["MODEL_ID"], "Qwen/Qwen3.5-35B-A3B-FP8")
        self.assertEqual(got["BATCH_SIZES"], "1 2 4 8 16")
        self.assertEqual(got["CORRECT_NEW_TOKENS"], 64)
        self.assertEqual(got["E2E_FACTOR_MAX"], 1.25)
        self.assertEqual(got["HARNESS"], "workspace/demo/qwen3_5/accept/final.sh")

    def test_a_missing_key_is_refused_not_guessed(self):
        p = self.tmp / "accept.sh"
        p.write_text(PINNED_SHAPE.replace('E2E_FACTOR_MAX="1.25"\n', ""))
        with self.assertRaises(SystemExit):
            parse_accept_sh(p)

    def test_the_real_pinned_accept_sh_still_has_the_shape_we_parse(self):
        """If .pm/accept.sh is ever re-pinned in a different shape this fails
        loudly instead of the gate silently reading defaults."""
        for cand in (REPO.parent / "agent" / ".pm" / "accept.sh",
                     Path.home() / "agent" / ".pm" / "accept.sh"):
            if cand.exists():
                got = parse_accept_sh(cand)
                self.assertEqual(got["CORRECT_NEW_TOKENS"], 64)
                self.assertEqual(got["MIN_OUTPUT_LEN"], 256)
                self.assertEqual(got["E2E_FACTOR_MAX"], 1.25)
                self.assertEqual(got["BASELINE"], "vllm")
                self.assertEqual(got["HARNESS"],
                                 "workspace/demo/qwen3_5/accept/final.sh")
                return
        self.skipTest("no .pm/accept.sh reachable from here")


class IntegritySmokeTest(unittest.TestCase):
    """A full run against the real accept dir WITHOUT --agent-root: it must
    produce a well-formed, explicitly NON-BINDING report that records the tree's
    sha and every tool digest."""

    def test_smoke(self):
        tmp = Path(tempfile.mkdtemp(prefix="final_integ_smoke_"))
        try:
            out = tmp / "integrity.json"
            p = subprocess.run(
                [sys.executable, str(INTEG), "--accept-dir", str(ACC),
                 "--repo-root", str(REPO),
                 "--baseline-dir", str(ACC / "baselines" / "vllm-0.25.1-20260725"),
                 "--bench-vllm", str(ACC / "bench_vllm.py"),
                 "--model", "Qwen/Qwen3.5-35B-A3B-FP8",
                 "--batch-sizes", "1 2 4 8 16",
                 "--prompts", ".pm/eval/prompts.jsonl",
                 "--correct-new-tokens", "64", "--min-input-len", "64",
                 "--min-output-len", "256", "--e2e-factor-max", "1.25",
                 "--baseline", "vllm", "--output-json", str(out)],
                capture_output=True, text=True)
            self.assertTrue(out.exists(), p.stdout + p.stderr)
            rep = json.loads(out.read_text())
            self.assertFalse(rep["binding"])          # no --agent-root
            self.assertIn("git", rep["recorded"])
            self.assertEqual(rep["recorded"]["reference"]["n_prompts"], 10)
            self.assertTrue(rep["recorded"]["reference"]["topk_present_every_step"])
            self.assertEqual(len(rep["recorded"]["exactness_baseline"]), 5)
            # every tool the gate executes must have a digest recorded
            self.assertTrue(all(v for v in rep["recorded"]["tool_sha256"].values()),
                            json.dumps(rep["recorded"]["tool_sha256"], indent=1))
            # the pinned vLLM comparator identity is recorded
            self.assertEqual(rep["recorded"]["vllm_pinned_baseline"]["input_len"], 256)
            self.assertEqual(rep["recorded"]["vllm_pinned_baseline"]["output_len"], 1024)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sha256_file_matches_sha256sum(self):
        p = ACC / "bench_vllm.py"
        want = subprocess.run(["sha256sum", str(p)], capture_output=True,
                              text=True).stdout.split()[0]
        self.assertEqual(sha256_file(p), want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
