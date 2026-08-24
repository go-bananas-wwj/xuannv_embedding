"""统一的 ``xuannv`` 命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from xuannv_embedding import __version__


def build_parser() -> argparse.ArgumentParser:
    """构建不依赖可选运行组件的顶层命令解析器。"""
    parser = argparse.ArgumentParser(
        prog="xuannv",
        description="region-agnostic monthly geospatial embeddings.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行统一命令行入口。"""
    build_parser().parse_args(argv)
    return 0
