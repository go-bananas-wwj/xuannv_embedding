"""Measured preparation and same-embedding task-request costs, with explicit stage scope."""

from __future__ import annotations

import json
from pathlib import Path

from xuannv_embedding.downstream.journal_report import _table
from xuannv_embedding.export.context import sha


def build(study, paper):
    study, paper = Path(study), Path(paper)
    root, goal = study / "jstars-revision", study.parent.parent
    sources = []

    def read(path, complete=True):
        value = json.loads(path.read_text())
        if complete and value.get("state") != "complete":
            raise ValueError("Cost stage has no completed measurement")
        sources.append({"path": str(path), "sha256": sha(path)})
        return value

    public = read(root / "resource-profile/public-timed/result.json")
    visuals = read(root / "resource-profile/public-visual/timing/result.json")
    projection = read(root / "resource-profile/regional-projections/result.json")
    queries = read(root / "resource-profile/monthly-queries/result.json")
    native = read(root / "resource-profile/native-independent/cost/result.json")
    semantic = read(
        goal / "waves/semantic_residual_v2/runtime-receipts/train_highres_haidian.0.json", False
    )
    structure = read(
        goal
        / "waves/decoupled_geometry_v5/runtime-receipts"
        / "train_decoupled_geometry_wide_highres_haidian.0.json",
        False,
    )
    if semantic["returncode"] or structure["returncode"]:
        raise ValueError("Regional fit process receipt is unsuccessful")
    stages = [
        (
            "公共双尺度视觉编码",
            "Public visual encoding at two scales",
            visuals["total_seconds"],
            "NPU",
            "0",
        ),
        (
            "公共时序拟合及导出",
            "Public temporal fit and export",
            public["total_seconds"],
            "NPU",
            "1200",
        ),
        ("公共月份查询", "Public monthly queries", queries["seconds"], "NPU", "0"),
        (
            "区域投影准备",
            "Regional projection preparation",
            projection["additive_seconds"],
            "CPU",
            "--",
        ),
        ("语义分支拟合", "Semantic fusion fit", semantic["elapsed_seconds"], "NPU", "2000"),
        ("结构分支拟合", "Structural fusion fit", structure["elapsed_seconds"], "NPU", "2000"),
        (
            "地方编码、融合推理与引导",
            "Local encoding, fusion inference and guidance",
            native["encoding_seconds"],
            "NPU",
            "0",
        ),
    ]
    requests = []
    for dataset, head, budget in [
        ("clcd", "mlp", 5),
        ("clcd", "mlp", 20),
        ("clsm", "mlp", 5),
        ("clsm", "conv3x3", 20),
        ("clcd", "regression", 20),
        ("clsm", "regression", 20),
        ("clcd", "retrieval", 3),
        ("clsm", "retrieval", 3),
    ]:
        path = root / "reuse-v2/jobs" / f"{dataset}_native_{head}_{budget}/result.json"
        value = read(path)
        requests.append(
            dict(
                dataset=dataset,
                head=head,
                budget=budget,
                targets=len(value["rows"]),
                prepare=value["prepare_seconds"],
                fit=value["fit_and_calibrate_seconds"],
                query=value["full_query_seconds"],
                source={"path": str(path), "sha256": sha(path)},
            )
        )
    record = (
        root
        / "reuse-v2/jobs/clcd_native_mlp_20/readout/calibration/records"
        / "C_clcd_crop_20260922_20_mlp.json"
    )
    model = json.loads(record.read_text())
    model_identity = read(Path(model["readout"]) / "identity.json", False)
    hardware = model_identity["runtime"]
    for english in [False, True]:
        out = paper / "tables" / ("en" if english else "")
        rows = [[x[int(english)], x[3], x[4], f"{x[2]/60:.2f}"] for x in stages]
        rows.append(
            [
                "阶段累计" if not english else "Sum of stage times",
                "--",
                "--",
                f"{sum(x[2] for x in stages)/60:.2f}",
            ]
        )
        _table(
            out / "jstars_resources.tex",
            (
                "海淀320图块的区域构建阶段耗时。输入为已准备RGB栅格、公共观测张量及统计量；"
                "I/O计入进程墙钟，各NPU阶段使用一张同型号设备。区域投影行补计无标签采样、拟合及"
                "上下文监督投影，编码阶段已包含的既定投影应用计数一次。"
                if not english
                else "Haidian preparation over 320 tiles from prepared RGB rasters, "
                "public observation "
                "tensors, and statistics. Process wall times include I/O; each NPU stage uses one "
                "device of the same type. Additional projection preparation counts sampling, "
                "fitting, and context-target projection once. The sum is stage accounting."
            ),
            "tab:jstars-resources",
            "llrr",
            (
                ["阶段", "设备", "参数更新", "分钟"]
                if not english
                else ["Stage", "Device", "Updates", "Minutes"]
            ),
            rows,
        )
        rows = []
        for r in requests:
            budget = (
                (f'{r["budget"]} components' if english else f'{r["budget"]}正类连通片')
                if r["head"] == "retrieval"
                else (f'{r["budget"]} tiles' if english else f'{r["budget"]}图块')
            )
            rows.append(
                [
                    r["dataset"].upper(),
                    {
                        "mlp": "MLP",
                        "conv3x3": "CNN",
                        "regression": "Ridge",
                        "retrieval": "Prototype",
                    }[r["head"]],
                    str(r["targets"]),
                    budget,
                    f'{r["prepare"]:.2f}',
                    f'{r["fit"]:.2f}',
                    f'{r["query"]:.2f}',
                ]
            )
        _table(
            out / "jstars_request_costs.tex",
            (
                "同一嵌入的读取调用耗时（秒）。目标数表示一次登记调用包含的任务数；分类为NPU，"
                "回归与原型检索为CPU。准备列包括实际缓存与锁等待状态，同一参考网格的准备由任务共享；"
                "适配包含登记验证步骤。"
                if not english
                else "Same-embedding request times (seconds). Targets counts the tasks in each "
                "registered request. Classification uses NPU; regression and prototypes use CPU. "
                "Preparation reflects actual cache/lock state and is shared by reference grid. "
                "Adaptation includes the registered validation step."
            ),
            "tab:jstars-request-costs",
            "lllrrrr",
            (
                ["参考", "读取器", "目标数", "支持预算", "准备", "适配", "查询"]
                if not english
                else [
                    "Reference",
                    "Reader",
                    "Targets",
                    "Support budget",
                    "Prepare",
                    "Adapt",
                    "Query",
                ]
            ),
            rows,
        )
    old = json.loads((paper / "notes/jstars-resource-ledger.json").read_text())
    old.update(
        sources=sources,
        measured_stages=[dict(name=x[1], seconds=x[2], device=x[3], updates=x[4]) for x in stages],
        cumulative_stage_seconds=sum(x[2] for x in stages),
        task_requests=requests,
        hardware=hardware,
        input_scope=(
            "Prepared RGB rasters, public observation tensors and summaries; "
            "inherited global pretraining is credited separately"
        ),
        frozen_visual_pretraining_updates_in_this_region=0,
    )
    (paper / "notes/jstars-resource-ledger.json").write_text(
        json.dumps(old, ensure_ascii=False, indent=2) + "\n"
    )
    return old
