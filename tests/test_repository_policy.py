from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from xuannv_embedding.utils.repository_policy import (
    PolicyError,
    validate_markdown_links,
    validate_tree,
)


def test_tree_policy_rejects_large_and_forbidden_files(tmp_path: Path) -> None:
    safe = tmp_path / "safe.py"
    safe.write_text("pass\n", encoding="utf-8")
    validate_tree(tmp_path, [safe], max_file_bytes=10, max_tree_bytes=10)

    forbidden = tmp_path / "weight.pt"
    forbidden.write_bytes(b"x")
    with pytest.raises(PolicyError, match="禁止文件类型"):
        validate_tree(tmp_path, [forbidden], max_file_bytes=10, max_tree_bytes=10)

    large = tmp_path / "large.txt"
    large.write_bytes(b"x" * 11)
    with pytest.raises(PolicyError, match="单文件"):
        validate_tree(tmp_path, [large], max_file_bytes=10, max_tree_bytes=20)


@pytest.mark.parametrize(
    "name",
    [
        "preview.png",
        "photo.jpeg",
        "animation.gif",
        "audio.wav",
        "table.xlsx",
        "bundle.zip",
        "bundle.tar",
        "records.parquet",
        "cache.sqlite",
    ],
)
def test_tree_policy_rejects_non_source_binary_and_data_formats(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")

    with pytest.raises(PolicyError, match="禁止文件类型"):
        validate_tree(tmp_path, [path])


def test_markdown_policy_checks_relative_targets(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("# Target\n", encoding="utf-8")
    source = tmp_path / "README.md"
    source.write_text("[ok](docs/target.md) [web](https://example.com)\n", encoding="utf-8")
    validate_markdown_links(tmp_path, [source, target])

    source.write_text("[missing](docs/missing.md)\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="失效本地链接"):
        validate_markdown_links(tmp_path, [source])


def test_accelerator_extras_pin_the_locally_validated_torch_pair() -> None:
    """torch 主版本对每个加速器平台都必须锁定在已验收的 2.6.0。"""
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    # CUDA 走同一个 torch pin，仅 wheel index 不同，因此没有独立 extra。
    assert "torch==2.6.0" in metadata["dependencies"]
    assert metadata["optional-dependencies"]["npu"] == ["torch-npu==2.6.0.post5"]
