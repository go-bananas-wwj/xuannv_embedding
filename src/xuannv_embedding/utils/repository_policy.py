"""公开生产仓的文件、体积、配置与文档链接门禁。"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from xuannv_embedding.config import Config

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TREE_BYTES = 25 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".docx",
    ".log",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pdf",
    ".pt",
    ".pth",
    ".pptx",
    ".safetensors",
    ".tif",
    ".tiff",
}
FORBIDDEN_NAMES = {".env", "credentials.json", "secrets.json"}
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


class PolicyError(ValueError):
    """仓库内容违反公开生产门禁。"""


def validate_tree(
    root: Path,
    tracked_files: list[Path],
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_tree_bytes: int = MAX_TREE_BYTES,
) -> None:
    total = 0
    for path in tracked_files:
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file():
            raise PolicyError(f"tracked 文件不存在: {path}")
        relative = absolute.relative_to(root) if absolute.is_relative_to(root) else absolute
        if absolute.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise PolicyError(f"禁止文件类型: {relative}")
        if absolute.name.lower() in FORBIDDEN_NAMES or absolute.name.endswith(".lock"):
            raise PolicyError(f"禁止文件名: {relative}")
        size = absolute.stat().st_size
        if size > max_file_bytes:
            raise PolicyError(f"单文件超过 {max_file_bytes} bytes: {relative} ({size})")
        total += size
    if total > max_tree_bytes:
        raise PolicyError(f"tracked tree 超过 {max_tree_bytes} bytes: {total}")


def validate_markdown_links(root: Path, markdown_files: list[Path]) -> None:
    failures: list[str] = []
    for path in markdown_files:
        absolute = path if path.is_absolute() else root / path
        text = absolute.read_text(encoding="utf-8")
        for raw_target in _MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local = unquote(target.split("#", maxsplit=1)[0])
            destination = (
                root / local.lstrip("/") if local.startswith("/") else absolute.parent / local
            )
            if not destination.exists():
                failures.append(f"{absolute.relative_to(root)} -> {target}")
    if failures:
        raise PolicyError("失效本地链接:\n" + "\n".join(sorted(failures)))


def tracked_files(root: Path) -> list[Path]:
    output = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    )
    return [Path(value) for value in output if value]


def validate_repository(root: Path) -> dict[str, int]:
    files = tracked_files(root)
    validate_tree(root, files)
    markdown = [path for path in files if path.suffix.lower() == ".md"]
    validate_markdown_links(root, markdown)
    configs = sorted((root / "configs" / "production").glob("*.yaml")) + sorted(
        (root / "configs" / "examples").glob("*.yaml")
    )
    for path in configs:
        Config.from_yaml(path)
    return {
        "tracked_files": len(files),
        "tracked_bytes": sum((root / path).stat().st_size for path in files),
        "markdown_files": len(markdown),
        "validated_configs": len(configs),
    }
