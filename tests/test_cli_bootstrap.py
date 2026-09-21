from __future__ import annotations

import argparse
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


def _registered_data_commands() -> list[str]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from xuannv_embedding.cli import build_parser
    finally:
        sys.path.pop(0)
    actions = [
        action
        for action in build_parser()._subparsers._group_actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    data = actions[0].choices["data"]
    nested = [
        action
        for action in data._subparsers._group_actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    return sorted(nested[0].choices)


@pytest.mark.parametrize("command", _registered_data_commands())
def test_every_data_subcommand_help_runs(command: str) -> None:
    """每个 `xuannv data` 子命令都是延迟导入的，help 是唯一便宜的导入门禁。

    子命令在顶层 parser 注册、实现在 data_process.cli 里 dispatch，二者不一致或被
    dispatch 的模块 import 失败时，除了 help 之外没有任何测试会触发它。
    """
    pytest.importorskip("rasterio")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "xuannv_embedding.cli", "data", command, "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("command", ["train", "export", "diagnose", "downstream"])
def test_top_level_subcommand_help_runs(command: str) -> None:
    pytest.importorskip("rasterio")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "xuannv_embedding.cli", command, "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
