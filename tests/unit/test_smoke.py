"""Smoke tests verifying package import and version metadata."""

import tomllib
from pathlib import Path

import vlm_rag


def test_package_import() -> None:
    """Verify that the package can be imported and has a valid version."""
    assert hasattr(vlm_rag, "__version__")
    assert isinstance(vlm_rag.__version__, str)
    assert vlm_rag.__version__ == "0.1.0"


def test_version_matches_pyproject() -> None:
    """Verify that the runtime __version__ matches pyproject.toml."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    expected_version = pyproject_data["project"]["version"]
    assert vlm_rag.__version__ == expected_version
