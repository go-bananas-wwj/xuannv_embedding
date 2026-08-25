"""统一的 ``xuannv`` 命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from xuannv_embedding import __version__
from xuannv_embedding.downstream.heads import STANDARD_HEAD_NAMES


def _add_downstream_commands(subparsers: argparse._SubParsersAction) -> None:
    downstream = subparsers.add_parser("downstream", help="训练或评测标准下游头")
    actions = downstream.add_subparsers(dest="downstream_command", required=True)

    train = actions.add_parser("train", help="按固定空间 fold 训练下游头")
    train.add_argument("--dataset", type=Path, required=True, help="NPZ embedding/label 数据集")
    train.add_argument("--folds", type=Path, required=True, help="空间 fold JSON")
    train.add_argument("--fold", type=int, default=0)
    train.add_argument("--head", choices=STANDARD_HEAD_NAMES, required=True)
    train.add_argument("--shot", choices=("full", "5", "10", "50"), default="full")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.set_defaults(handler=_run_downstream_train)

    evaluate = actions.add_parser("evaluate", help="使用 validation 阈值评测固定 test split")
    evaluate.add_argument("--dataset", type=Path, required=True, help="同一 NPZ 数据集")
    evaluate.add_argument("--folds", type=Path, required=True, help="同一空间 fold JSON")
    evaluate.add_argument("--fold", type=int, default=0)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(handler=_run_downstream_evaluate)


def _run_downstream_train(args: argparse.Namespace) -> int:
    from xuannv_embedding.downstream.workflow import train_downstream

    train_downstream(args)
    return 0


def _run_downstream_evaluate(args: argparse.Namespace) -> int:
    from xuannv_embedding.downstream.workflow import evaluate_downstream

    evaluate_downstream(args)
    return 0


def _add_data_commands(subparsers: argparse._SubParsersAction) -> None:
    data = subparsers.add_parser("data", help="网格、采样、物化、预处理与审计")
    actions = data.add_subparsers(dest="data_command", required=True)
    descriptions = {
        "grid": "构建全国 1280 m 父网格",
        "registry": "生成全国采样 registry",
        "partition": "生成确定性的全国十等分",
        "materialize": "从冻结 catalog 物化多源栅格",
        "preprocess": "对齐并切分多源栅格",
        "manifest": "生成带摘要的 manifest v1",
        "validate": "审计 manifest 或父网格包",
        "local-index": "索引本地 V2 月度 ZIP 与高分场景",
        "preflight": "训练前审计 V2 数据合同与像元质量",
        "statistics": "计算 V2 训练划分的 stored-DN 波段统计量",
        "local-zarr-cache": "将本地 ZIP 顺序重打包为 smoke Zarr cache",
    }
    for command, help_text in descriptions.items():
        action = actions.add_parser(command, help=help_text, add_help=False)
        action.set_defaults(handler=_run_data, data_command=command)


def _run_data(args: argparse.Namespace) -> int:
    from xuannv_embedding.data_process.cli import dispatch

    return dispatch(args.data_command, args.forwarded_args)


def _run_train(args: argparse.Namespace) -> int:
    from xuannv_embedding.training.cli import main as train_main

    return train_main(args.forwarded_args)


def _run_export(args: argparse.Namespace) -> int:
    from xuannv_embedding.export.cli import main as export_main

    return export_main(args.forwarded_args)


def build_parser() -> argparse.ArgumentParser:
    """构建不依赖可选运行组件的顶层命令解析器。"""
    parser = argparse.ArgumentParser(
        prog="xuannv",
        description="region-agnostic monthly geospatial embeddings.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    _add_data_commands(subparsers)
    _add_downstream_commands(subparsers)
    train = subparsers.add_parser("train", help="运行 P10C 训练或发布 smoke", add_help=False)
    train.set_defaults(handler=_run_train)
    export = subparsers.add_parser(
        "export", help="从严格 checkpoint 导出 embedding", add_help=False
    )
    export.set_defaults(handler=_run_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行统一命令行入口。"""
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.command in {"data", "train", "export"}:
        args.forwarded_args = unknown
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    handler = getattr(args, "handler", None)
    return int(handler(args)) if handler is not None else 0
