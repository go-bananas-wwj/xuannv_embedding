from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_module_help_is_available_without_optional_dependencies() -> None:
    """Deleting the package CLI module must make this user-visible command fail."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "xuannv_embedding.cli", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "region-agnostic monthly geospatial embeddings" in result.stdout


@pytest.mark.parametrize(
    "module",
    ["xuannv_embedding.training.cli", "xuannv_embedding.export.cli"],
)
def test_runtime_help_does_not_import_optional_raster_stack(module: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    code = f"""
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'rasterio' or name.startswith('rasterio.'):
        raise ImportError('rasterio intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from {module} import main
main(['--help'])
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
