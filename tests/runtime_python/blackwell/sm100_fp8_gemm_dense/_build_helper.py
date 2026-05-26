"""Shared on-demand `setup.py build_ext` helper for sm100_fp8_gemm_dense tests.

Each of the 5 tests in this directory wants the same .so available before
import. Centralise the rebuild check so changes to the wrapper or sm100
headers only have to invalidate the cached .so in one place.
"""
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SO = "runtime_kernel_blackwell_fp8_gemm_dense"


def ensure_extension_built(so_name: str = _DEFAULT_SO, force: bool = False) -> None:
    """Build the C++/CUDA extension in this test dir if the .so is missing.

    Args:
        so_name: base name of the extension (no .cpython-...so suffix).
        force: remove any pre-built artifacts and rebuild from scratch.
    """
    import glob
    so_glob = os.path.join(THIS_DIR, f"{so_name}.cpython-*.so")
    if force:
        build_dir = os.path.join(THIS_DIR, "build")
        if os.path.isdir(build_dir):
            import shutil
            shutil.rmtree(build_dir)
        for so in glob.glob(so_glob):
            os.remove(so)
    if not glob.glob(so_glob):
        print(f"Building C++ extension: {so_name}", flush=True)
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=THIS_DIR)
