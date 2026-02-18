# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Packaging tests for the mirage Python package.

Pure tests verify the current environment works (import succeeds).
Impure tests create fresh venvs and test both editable and non-editable
installs end-to-end, including bundled native library layout.
"""

import importlib.util
import os
import pathlib
import subprocess
import textwrap
import venv

import pytest

# Maintenance surface: if a native library is added or renamed upstream,
# this tuple must be updated to match.
REQUIRED_NATIVE_LIBS: tuple[str, ...] = (
    "libabstract_subexpr",
    "libformal_verifier",
)


def _clean_env() -> dict[str, str]:
    """Return the current environment with Python path variables stripped.

    Prevents the test runner's PYTHONPATH/PYTHONHOME from leaking into
    venv subprocesses, which could mask packaging bugs (e.g. missing
    bundled libs found via a leaked source-tree PYTHONPATH).
    """
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}


@pytest.fixture(scope="session")
def repo_root(pytestconfig: pytest.Config) -> pathlib.Path:
    """Return the pytest rootdir as a Path, requiring a repo checkout."""
    root: pathlib.Path = pytestconfig.rootpath
    if not (root / "setup.py").is_file():
        pytest.skip(f"repo checkout not available under pytest rootdir: {root}")
    return root


def _create_venv(base: pathlib.Path, name: str) -> pathlib.Path:
    """Create a venv at `base/name` with an upgraded pip.

    Returns the path to the venv's Python interpreter.

    NOTE: assumes Unix layout (bin/python).  This project only
    targets Linux (CUDA requirement), so Windows is not supported.
    """
    venv_dir: pathlib.Path = base / name
    venv.create(venv_dir, with_pip=True)
    python: pathlib.Path = venv_dir / "bin" / "python"
    subprocess.check_call(
        [python, "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_clean_env(),
    )
    return python


def _pip_install(python: pathlib.Path, repo_root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run `pip install` with `args` inside the venv identified by `python`."""
    return subprocess.run(
        [python, "-m", "pip", "install", *args],
        cwd=repo_root,
        env=_clean_env(),
        text=True,
        timeout=600,
        check=True,
    )


class TestImport:
    """Verify that mirage is importable in the current environment."""

    def test_import_mirage(self) -> None:
        """import mirage must succeed if the package is on sys.path."""
        spec = importlib.util.find_spec("mirage")
        if spec is None:
            pytest.skip("mirage is not on sys.path")
        import mirage  # noqa: F401


@pytest.mark.slow
@pytest.mark.impure
class TestNonEditableInstall:
    """Verify that ``pip install .`` produces a working, self-contained package.

    Creates a fresh venv with no build tree, so native libs are only
    discoverable if they are properly bundled inside the package.
    """

    def test_install_layout_and_import(self, tmp_path: pathlib.Path, repo_root: pathlib.Path) -> None:
        """Non-editable install must bundle native libs and import cleanly."""
        python: pathlib.Path = _create_venv(tmp_path, "venv_non_editable")
        _pip_install(python, repo_root, ".")

        # Run layout + import check inside the fresh venv.  There is no
        # build tree, so _find_native_lib() can only succeed if the libs
        # are bundled in the package directory.
        #
        # Layout check runs first (via find_spec, without importing) so
        # that diagnostic output is available even if import crashes.
        check_script = textwrap.dedent("""\
            import importlib.util
            import pathlib

            spec = importlib.util.find_spec("mirage")
            assert spec is not None, "mirage not found on sys.path in venv"
            assert spec.submodule_search_locations is not None

            pkg_dir = pathlib.Path(spec.submodule_search_locations[0])
            lib_dir = pkg_dir / "lib"
            assert lib_dir.is_dir(), (
                f"lib/ directory not found in installed package: {{pkg_dir}}"
            )

            required = {required!r}
            for name in required:
                found = any(
                    name in f.name
                    for f in lib_dir.iterdir()
                    if f.suffix == ".so"
                )
                assert found, f"{{name}}.so not bundled in {{lib_dir}}"

            print(f"layout ok: {{lib_dir}}")

            import mirage
            print("import ok")
        """).format(required=list(REQUIRED_NATIVE_LIBS))

        result = subprocess.run(
            [python, "-c", check_script],
            env=_clean_env(),
            timeout=30,
        )
        assert result.returncode == 0, "Layout or import check failed (see output above)"


@pytest.mark.slow
@pytest.mark.impure
class TestEditableInstall:
    """Verify that ``pip install -e .`` produces a working package."""

    def test_editable_import(self, tmp_path: pathlib.Path, repo_root: pathlib.Path) -> None:
        """``import mirage`` must succeed from an editable install."""
        python: pathlib.Path = _create_venv(tmp_path, "venv_editable")
        _pip_install(python, repo_root, "-e", ".")

        result = subprocess.run(
            [python, "-c", "import mirage; print('editable import ok')"],
            env=_clean_env(),
            timeout=30,
        )
        assert result.returncode == 0, "import mirage failed from editable install (see output above)"
