"""CPU-only compile/reuse precision contracts; requires the native extension."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from mirage.mpk.persistent_kernel import PersistentKernel, get_compile_command


class CompilePrecisionTests(unittest.TestCase):
    def kernel(self, fast_math):
        pk = PersistentKernel.__new__(PersistentKernel)
        pk.__finalized__ = True  # This metadata-only object owns no runtime.
        for name in ("max_seq_length", "max_num_batched_requests",
                     "max_num_batched_tokens", "max_tokens_per_request",
                     "max_num_pages", "page_size", "world_size"):
            setattr(pk, name, 1)
        pk.mode, pk.mpi_rank, pk.target_cc = "offline", 0, 100
        pk.use_fast_math = fast_math
        pk._model_tensors, pk.meta_tensors = {}, {}
        pk.kv_groups, pk.spec_decode_config = [SimpleNamespace()], None
        return pk

    def test_compile_flag_is_opt_out(self):
        for fast_math in (True, False):
            cmd = get_compile_command(
                self.kernel(fast_math), 100, "nvcc", "kernel.cu", "/python",
                "/mirage", "/include", "/deps", None, None, None, None,
                "kernel.so", False, False)
            self.assertEqual("-use_fast_math" in cmd, fast_math)

    def test_kernel_reuse_rejects_precision_mismatch(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "metadata.json")
            self.kernel(False)._save_kernel_metadata(path)
            self.kernel(False)._validate_kernel_compatibility(path)
            with self.assertRaisesRegex(ValueError, "use_fast_math"):
                self.kernel(True)._validate_kernel_compatibility(path)

    def test_legacy_metadata_means_fast_math(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            self.kernel(True)._save_kernel_metadata(str(path))
            metadata = json.loads(path.read_text())
            del metadata["use_fast_math"]
            path.write_text(json.dumps(metadata))
            self.kernel(True)._validate_kernel_compatibility(str(path))
            with self.assertRaisesRegex(ValueError, "use_fast_math"):
                self.kernel(False)._validate_kernel_compatibility(str(path))


if __name__ == "__main__":
    unittest.main()
