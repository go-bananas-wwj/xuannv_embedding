"""Source-bound bilingual evidence for task reuse, composition and measured resources."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream.journal_report import _table
from xuannv_embedding.export.context import sha

NAMES = {
    "native": ("完整融合", "Viewwise fusion"),
    "public": ("同方法公共", "Matched public"),
    "AEF": ("AEF", "AEF"),
    "Raw180": ("原始统计180维", "Raw statistics (180D)"),
    "degraded": ("同权重10米降质", "Fixed-weight 10-m degradation"),
}


def build(root, paper):
    root, paper = Path(root), Path(paper)
    sources, auxiliary, mixed = [], [], []

    def read(path):
        sources.append({"path": str(path), "sha256": sha(path)})
        return json.loads(path.read_text())

    registered = read(root / "auxiliary-jobs.json")
    for job in registered:
        p = Path(job["output"]) / "result.json"
        if p.exists():
            result = read(p)
            if result["state"] == "complete":
                if (
                    result["identity"]["spec"] != job["spec"]["sha256"]
                    or result["model"] != job["model"]
                    or result["head"] != job["head"]
                    or result["budget"] != job["budget"]
                ):
                    raise ValueError("Auxiliary receipt does not match registration")
                auxiliary.append({"registration": job, "result": result})
    for model in NAMES:
        p = root / "mixed-composition" / model / "result.json"
        if p.exists():
            result = read(p)
            if result["state"] == "complete":
                mixed.append(result)
    summary = []
    for value in auxiliary:
        r, job = value["result"], value["registration"]
        for source in sorted({x["source"] for x in r["rows"]}):
            rows = [x for x in r["rows"] if x["source"] == source]
            metric = {"regression": "mae", "retrieval": "ap"}.get(job["head"], "f1")
            summary.append(
                dict(
                    source=source,
                    model=job["model"],
                    head=job["head"],
                    budget=job["budget"],
                    metric=metric,
                    mean=float(np.mean([x["metrics"][metric] for x in rows])),
                    tasks=len(rows),
                    prepare_seconds=r["prepare_seconds"],
                    fit_seconds=r["fit_and_calibrate_seconds"],
                    query_seconds=r["full_query_seconds"],
                    peak_process_rss_bytes=r["peak_process_rss_bytes"],
                )
            )
    for english in [False, True]:
        out = paper / "tables" / ("en" if english else "")
        out.mkdir(exist_ok=True)
        language = int(english)
        rows = []
        for source in ["clcd", "clsm", "osm", "worldcover"]:
            for model in NAMES:
                entries = [x for x in summary if x["source"] == source and x["model"] == model]
                if not entries:
                    continue
                values = [source.upper(), NAMES[model][language]]
                for head, budget in [("mlp", 5), ("mlp", 20), ("conv3x3", 5), ("conv3x3", 20)]:
                    item = [x for x in entries if x["head"] == head and x["budget"] == budget]
                    values.append(f"{100*item[0]['mean']:.2f}" if item else "--")
                rows.append(values)
        _table(
            out / "jstars_reuse_classification.tex",
            (
                "同一表示的多参考分类。F1为百分数，各参考内按任务等权平均；5/20为每任务支持图块预算。"
                "本组重新拟合各预算读取器，与主表的历史冻结头分别报告。"
                if not english
                else "Same-representation classification across reference sources. F1 is "
                "averaged equally across targets within each source (percent). Columns "
                "use 5/20 support tiles per task. Readers are fitted for each budget; "
                "the principal mapping table separately retains its original readers."
            ),
            "tab:jstars-reuse-classification",
            "llrrrr",
            ["参考" if not english else "Reference", "表示" if not english else "Representation"]
            + ["MLP5", "MLP20", "CNN5", "CNN20"],
            rows,
        )
        rows = []
        for source in ["clcd", "clsm", "osm", "worldcover"]:
            for model in NAMES:
                entries = [x for x in summary if x["source"] == source and x["model"] == model]
                if not entries:
                    continue
                values = [source.upper(), NAMES[model][language]]
                for head, budget in [("regression", 5), ("regression", 20), ("retrieval", 3)]:
                    item = [x for x in entries if x["head"] == head and x["budget"] == budget]
                    values.append(f"{item[0]['mean']:.4f}" if item else "--")
                rows.append(values)
        _table(
            out / "jstars_reuse_tasks.tex",
            (
                "固定表示的覆盖比例回归与样例检索。MAE按参考内任务等权平均，越小越好；检索报告AP，"
                "每任务使用三个正类连通片样例，越大越好。覆盖比例在已登记的物理聚合网格评价。"
                if not english
                else "Coverage regression and example retrieval from fixed representations. "
                "MAE is averaged equally within each reference source (lower is better). "
                "Retrieval uses three positive connected components per task and reports "
                "AP (higher is better). Coverage uses the registered aggregation grids."
            ),
            "tab:jstars-reuse-tasks",
            "llrrr",
            ["参考" if not english else "Reference", "表示" if not english else "Representation"]
            + ["MAE5", "MAE20", "AP3"],
            rows,
        )
        rows = []
        for result in mixed:
            for record in result["rows"]:
                metrics = record["metrics"]["outside_support_and_calibration"]
                values = [NAMES[result["model"]][language], str(record["budget_tiles"])]
                for stratum in ["all_cells", "mixed_cells"]:
                    values.extend(f"{metrics[stratum][k]:.4f}" for k in ["mae", "rmse"])
                rows.append(values)
        _table(
            out / "jstars_mixed.tex",
            (
                "10米建筑覆盖比例：排除支持与校准图块后的MAE/RMSE。混合像元满足参考建筑比例"
                "$0<f<1$；参考由CLSM机器提取建筑矢量在2米网格采样后聚合获得。"
                if not english
                else "Building fractions on the 10-m grid: MAE/RMSE outside support and "
                "calibration tiles. Mixed cells satisfy reference fraction $0<f<1$. "
                "CLSM machine-derived footprints are sampled on a 2-m grid and aggregated."
            ),
            "tab:jstars-mixed",
            "lrrrrr",
            ["表示" if not english else "Representation", "支持" if not english else "Tiles"]
            + ["MAE(all)", "RMSE(all)", "MAE(mixed)", "RMSE(mixed)"],
            rows,
        )
    ledger = dict(
        sources=sources,
        expected_auxiliary_jobs=len(registered),
        completed_auxiliary_jobs=len(auxiliary),
        mixed_models=[r["model"] for r in mixed],
        task_means=summary,
        metric_scope="Regional map-reference agreement; OSM/WorldCover are regional weak targets",
    )
    (paper / "notes/jstars-reuse-ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n"
    )
    return ledger


def coverage_figure(root, paper):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, paper = Path(root), Path(paper)
    points = []
    for fraction, path in [
        (0, root / "whole-source-zero/evaluation/result.json"),
        (0.25, root / "raw-missingness/coverage25/evaluation/result.json"),
        (0.5, root / "raw-missingness/coverage50/evaluation/result.json"),
        (1, root.parent / "full-region/selected_method/result.json"),
    ]:
        result = json.loads(path.read_text())
        if result["state"] != "complete":
            raise ValueError("Coverage point incomplete")
        key = "all_region" if fraction == 1 else "metrics"
        for head in ["mlp", "conv3x3"]:
            rows = [x for x in result["rows"] if x["head"] == head]
            assert len(rows) == 4
            for scope in [key, "outside_support_and_calibration"]:
                points.append(
                    {
                        "fraction": fraction,
                        "head": head,
                        "scope": "all_region" if scope == key else scope,
                        "mean_f1": 100 * float(np.mean([x[scope]["f1"] for x in rows])),
                        "source": {"path": str(path), "sha256": sha(path)},
                    }
                )
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.6, 3.0), constrained_layout=True)
    for head, color, name in [("mlp", "#31688e", "MLP"), ("conv3x3", "#a34314", "CNN")]:
        for scope, style, suffix in [
            ("all_region", "-", "all region"),
            ("outside_support_and_calibration", "--", "query only"),
        ]:
            selected = [x for x in points if x["head"] == head and x["scope"] == scope]
            ax.plot(
                [100 * x["fraction"] for x in selected],
                [x["mean_f1"] for x in selected],
                style,
                color=color,
                marker="o",
                label=f"{name}: {suffix}",
                linewidth=1.5,
                markersize=4,
            )
    ax.set(xlabel="Retained local coverage (%)", ylabel="Mean F1 (%)", xticks=[0, 25, 50, 100])
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.savefig(paper / "figures/jstars_source_coverage.pdf")
    plt.close(fig)
    (paper / "notes/jstars-coverage-curve.json").write_text(json.dumps(points, indent=2) + "\n")


def mixed_cases(root, paper):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    from xuannv_embedding.downstream.frozen_readouts import load_readout
    from xuannv_embedding.training.frozen_fusion import verified

    root, paper = Path(root), Path(paper)
    spec = json.loads((root / "mixed-composition/spec.json").read_text())
    data = json.loads(verified(spec["reference"]).read_text())
    receipt = json.loads((root / "mixed-composition/reference.json").read_text())
    truth = np.load(verified(receipt["targets"]), mmap_mode="r")
    excluded = set(data["train_positions"]) | set(data["calibration_positions"])
    candidates = [i for i in range(len(truth)) if i not in excluded]
    ordered = sorted(candidates, key=lambda i: (float(truth[i].mean()), i))
    selected = [ordered[round(q * (len(ordered) - 1))] for q in [0.25, 0.5, 0.75]]
    raw_case = json.loads((root / "fine-detail/degraded/case.json").read_text())
    raw_source = json.loads(verified(raw_case["sources"]["context"]).read_text())
    feature = json.loads(verified(raw_source["feature_spec"]).read_text())
    original = json.loads(verified(feature["rasters"]).read_text())
    changed = json.loads((root / "fine-detail/degraded/encoded/context/status.json").read_text())
    degraded = json.loads(verified(changed["rasters"]).read_text())
    rasters = [{r["patch_id"]: r for r in manifest["records"]} for manifest in [original, degraded]]
    models, arrays, sources = {}, {}, []
    for name in ["AEF", "native", "degraded"]:
        descriptor = spec["models"][name]
        if "feature_receipt" in descriptor:
            status = json.loads(Path(descriptor["feature_receipt"]).read_text())
            ref = status["features"]
        else:
            ref = descriptor["features"]
        arrays[name] = np.load(verified(ref), mmap_mode="r")
        if descriptor["layout"] == "nchw":
            arrays[name] = arrays[name].transpose(0, 2, 3, 1)
        p = root / "mixed-composition" / name / "readout-20"
        models[name] = load_readout(p, sha(p / "identity.json"))
        sources.append({"features": ref, "reader": str(p), "identity": sha(p / "identity.json")})
    fig, axes = plt.subplots(3, 6, figsize=(13, 6.7), constrained_layout=True)
    titles = [
        "Local RGB median",
        "10-m degraded RGB",
        "CLSM fraction",
        "AEF",
        "Viewwise",
        "Degraded viewwise",
    ]
    records = []
    for row, i in enumerate(selected):
        record = data["records"][i]
        images, image_sources = [], []
        for lookup in rasters:
            ref = lookup[record["patch_id"]]["raster"]
            image_sources.append(ref)
            image = np.zeros((3, 640, 640), np.float32)
            with rasterio.open(verified(ref)) as src:
                for band in range(3):
                    reproject(
                        rasterio.band(src, band + 1),
                        image[band],
                        src_crs=src.crs,
                        src_transform=src.transform,
                        dst_crs=data["crs"],
                        dst_transform=from_bounds(*record["bounds"], 640, 640),
                        resampling=Resampling.bilinear,
                    )
            images.append(image.transpose(1, 2, 0)[160:480, 160:480])
        lo, hi = np.percentile(images[0], [2, 98])
        results = [
            models[name]
            .predict(np.asarray(arrays[name][i], np.float32).reshape(-1, 64))
            .reshape(128, 128)[32:96, 32:96]
            for name in ["AEF", "native", "degraded"]
        ]
        panels = [*images, truth[i, 32:96, 32:96], *results]
        for column, (ax, values) in enumerate(zip(axes[row], panels)):
            if column < 2:
                ax.imshow(np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1))
            else:
                mapped = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
            ax.set(xticks=[], yticks=[])
            if row == 0:
                ax.set_title(titles[column], fontsize=9)
            ax.plot([0.06, 0.21625], [0.06, 0.06], color="white", lw=2, transform=ax.transAxes)
            ax.text(
                0.138125,
                0.095,
                "100 m",
                ha="center",
                color="white",
                fontsize=7,
                transform=ax.transAxes,
            )
        axes[row, 0].set_ylabel(f"Case {row + 1}", fontsize=9)
        records.append(
            dict(
                position=i,
                tile=record,
                reference_mean=float(truth[i].mean()),
                image_sources=image_sources,
                shared_rgb_display_percentiles=[float(lo), float(hi)],
                crop_output_rows=[32, 96],
                crop_output_columns=[32, 96],
            )
        )
    fig.colorbar(mapped, ax=axes[:, 2:].ravel().tolist(), shrink=0.7, label="Building fraction")
    fig.savefig(paper / "figures/jstars_building_composition.pdf")
    plt.close(fig)
    manifest = dict(
        selection=(
            "25th,50th,75th percentile reference mean among tiles outside "
            "all training and calibration positions; central640m crop"
        ),
        reference=receipt["targets"],
        models=sources,
        cases=records,
        code_sha256=sha(__file__),
    )
    (paper / "notes/jstars-building-cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def per_target_tables(root, paper):
    root, paper = Path(root), Path(paper)
    registry = json.loads((root / "auxiliary-jobs.json").read_text())
    results, values, targets, sources = [], {}, set(), []
    for job in registry:
        path = Path(job["output"]) / "result.json"
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        if receipt["state"] != "complete":
            continue
        if receipt["identity"]["spec"] != job["spec"]["sha256"]:
            raise ValueError("Per-target receipt specification changed")
        sources.append({"path": str(path), "sha256": sha(path)})
        for row in receipt["rows"]:
            key = (row["source"], row["task"], job["model"], job["head"], job["budget"])
            if key in values:
                raise ValueError("Duplicate task/budget result")
            values[key] = row["metrics"]
            targets.add((row["source"], row["task"]))
            results.append({**row, "representation": job["model"]})
    short = {
        "native": ("融合", "Fusion"),
        "public": ("公共", "Public"),
        "AEF": ("AEF", "AEF"),
        "Raw180": ("Raw180", "Raw180"),
        "degraded": ("降质", "Degraded"),
    }
    for english in [False, True]:
        out = paper / "tables" / ("en" if english else "")
        for family, settings, header, caption in [
            (
                "classification",
                [("mlp", 5, "f1"), ("mlp", 20, "f1"), ("conv3x3", 5, "f1"), ("conv3x3", 20, "f1")],
                ["MLP5", "MLP20", "CNN5", "CNN20"],
                (
                    "All registered classification targets and support budgets: F1 (percent)."
                    if english
                    else "全部登记分类目标与5/20支持预算：F1（百分数）。"
                ),
            ),
            (
                "readouts",
                [("regression", 5, "mae"), ("regression", 20, "mae"), ("retrieval", 3, "ap")],
                ["MAE5", "MAE20", "AP3"],
                (
                    "All registered coverage-regression and retrieval targets. Dashes indicate "
                    "task families not registered for the reference source."
                    if english
                    else "全部登记覆盖比例回归和检索目标。横线表示该参考未登记相应任务类型。"
                ),
            ),
        ]:
            rows = []
            for source, task in sorted(targets):
                for model in NAMES:
                    metrics = [values.get((source, task, model, h, b)) for h, b, _ in settings]
                    if not any(metrics):
                        continue
                    label = task.removeprefix(source + "_").replace("_", r"\_")
                    row = [source.upper(), label, short[model][int(english)]]
                    for entry, (_, _, key) in zip(metrics, settings):
                        if entry is None:
                            row.append("--")
                        else:
                            value = entry[key] * (100 if key == "f1" else 1)
                            row.append(f"{value:.2f}" if key == "f1" else f"{value:.4f}")
                    rows.append(row)
            text = [
                r"\small\begin{longtable}{lll" + "r" * len(settings) + "}",
                r"\caption{" + caption + r"}\\",
                r"\toprule",
                " & ".join(
                    (
                        ["Reference", "Target", "Representation"]
                        if english
                        else ["参考", "目标", "表示"]
                    )
                    + header
                )
                + r"\\\midrule\endhead",
                *[" & ".join(row) + r"\\" for row in rows],
                r"\bottomrule\end{longtable}\normalsize",
                "",
            ]
            (out / ("supplement_" + family + ".tex")).write_text("\n".join(text))
    ledger = dict(
        sources=sources, rows=results, completed_jobs=len(sources), expected_jobs=len(registry)
    )
    (paper / "notes/jstars-task-results.json").write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger
