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

These tests verify that the installed package layout is self-contained,
that native shared libraries are discoverable regardless of install
mode (editable vs non-editable), and primary libs loadable (by proxy by
importing mirage, note this isnt an exhaustive check its very basic).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import subprocess
import textwrap
import venv

import pytest

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent.parent
assert (REPO_ROOT / "setup.py").is_file(), (
    f"REPO_ROOT sanity check failed: {REPO_ROOT / 'setup.py'} does not exist. "
    f"Did the test file move? REPO_ROOT is derived from __file__ via "
    f"parent.parent.parent and must point to the repository root."
)

# Maintenance surface: if a native library is added or renamed upstream,
# this tuple must be updated to match.  There is no clean way to derive
# these from setup.py (it runs cargo/cmake unconditionally at import
# time, making it unparseable) or from pyproject.toml (which does not
# list the Rust crate outputs).
REQUIRED_NATIVE_LIBS: tuple[str, ...] = (
    "libabstract_subexpr",
    "libformal_verifier",
)


def _find_package_dir() -> pathlib.Path | None:
    """Locate the installed mirage package directory without importing it.

    `importlib.util.find_spec` searches `sys.path` for the package
    and returns its metadata without executing ``__init__.py``, so this
    works even when DSO loading would fail.
    """
    spec: importlib.machinery.ModuleSpec | None = importlib.util.find_spec("mirage")
    if spec is None or spec.submodule_search_locations is None:
        return None
    return pathlib.Path(spec.submodule_search_locations[0])


def _lib_exists_in_dir(directory: pathlib.Path, lib_name: str) -> bool:
    """Return True if `directory` contains a `.so` whose name includes `lib_name`."""
    return any(lib_name in entry.name for entry in directory.iterdir() if entry.suffix == ".so")


def _create_venv(base: pathlib.Path, name: str) -> pathlib.Path:
    """Create a venv at `base/name` with an upgraded pip.

    Returns the path to the venv's Python interpreter.
    """
    venv_dir: pathlib.Path = base / name
    venv.create(venv_dir, with_pip=True)
    # NOTE: assumes Unix layout, afaik we only support linux.
    python: pathlib.Path = venv_dir / "bin" / "python"
    subprocess.check_call(
        [python, "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return python


def _pip_install(python: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run `pip install` with `args` inside the venv identified by `python`."""
    return subprocess.run(
        [python, "-m", "pip", "install", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )


class TestInstalledLayout:
    """Verify the installed package bundles its native dependencies.

    Locates the package without importing it (which would trigger lib
    loading), then verifies that native libs are bundled. Skipped when the
    package is not installed.

    Requires package on sys path.
    """

    @pytest.fixture()
    def pkg_dir(self) -> pathlib.Path:
        """Locate the installed package directory, skipping if not installed.

        Also skips for editable installs (source tree on PYTHONPATH) where
        native libs live in build/ rather than being bundled in the package.
        """
        found: pathlib.Path | None = _find_package_dir()
        if found is None:
            pytest.skip("mirage is not installed / not on sys.path")
        if REPO_ROOT in found.parents or found == REPO_ROOT:
            pytest.skip("editable install -- native libs live in build/, not package tree")
        return found

    def test_native_libs_bundled(self, pkg_dir: pathlib.Path) -> None:
        """Required libs must exist within the installed package tree.

        Checks package root and `lib/` subdir.
        """
        lib_subdir: pathlib.Path = pkg_dir / "lib"
        pkg_entries: list[str] = sorted(p.name for p in pkg_dir.iterdir())

        for lib_name in REQUIRED_NATIVE_LIBS:
            found: bool = _lib_exists_in_dir(pkg_dir, lib_name) or (
                lib_subdir.is_dir() and _lib_exists_in_dir(lib_subdir, lib_name)
            )
            assert found, (
                f"{lib_name}.so is not bundled inside the installed package. "
                f"Package dir: {pkg_dir}\n"
                f"Contents: {pkg_entries}\n"
                f"Native libraries must be included in the package (e.g. under "
                "mirage/lib/) for non-editable installs to work."
            )


# Hermetic build systems should test the install themselves and pip-in-venv
# would fail due to missing FHS library paths, among many other issues with
# the impurity of pip.
@pytest.mark.slow
@pytest.mark.impure
class TestNonEditableInstall:
    """Verify that `pip install` (non-editable) produces a working package."""

    def test_pkg_install(self, tmp_path: pathlib.Path) -> None:
        """``import mirage`` must succeed from a non-editable install."""
        python: pathlib.Path = _create_venv(tmp_path, "venv_non_editable")

        result: subprocess.CompletedProcess[str] = _pip_install(python, ".")

        result = subprocess.run(
            [
                python,
                "-c",
                textwrap.dedent("""\
                    import mirage
                    print(f"mirage.__file__: {mirage.__file__}")
                    print("import successful")
                """),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"import mirage failed from non-editable install.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_pkg_install_editable(self, tmp_path: pathlib.Path) -> None:
        """`pip install -e .` must also continue to work."""
        python: pathlib.Path = _create_venv(tmp_path, "venv_editable")

        result: subprocess.CompletedProcess[str] = _pip_install(python, "-e", ".")

        result = subprocess.run(
            [python, "-c", "import mirage; print('editable import ok')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"import mirage failed from editable install.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
