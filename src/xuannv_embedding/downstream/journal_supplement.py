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
    "Raw180": ("原始观测180维", "Raw observations (180D)"),
    "degraded": ("同权重10米降质", "Fixed-weight 10-m degradation"),
}


def task_registry(root):
    candidate = Path(root) / "reuse-v2/jobs.json"
    return candidate if candidate.exists() else Path(root) / "auxiliary-jobs.json"


def composition_root(root):
    candidate = Path(root) / "mixed-composition-v2"
    return candidate if (candidate / "spec.json").exists() else Path(root) / "mixed-composition"


def build(root, paper):
    root, paper = Path(root), Path(paper)
    sources, auxiliary, mixed = [], [], []

    def read(path):
        sources.append({"path": str(path), "sha256": sha(path)})
        return json.loads(path.read_text())

    registered = read(task_registry(root))
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
        p = composition_root(root) / model / "result.json"
        if p.exists():
            result = read(p)
            if result["state"] == "complete":
                mixed.append(result)
    common = None
    if mixed:
        from xuannv_embedding.downstream.journal_uncertainty import common_query_metrics

        spec = json.loads((composition_root(root) / "spec.json").read_text())
        reference_data = json.loads(Path(spec["reference"]["path"]).read_text())
        common = common_query_metrics({x["model"]: x for x in mixed}, reference_data["indices"])
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
                metrics = next(
                    x["metrics"]
                    for x in common["models"][result["model"]]["rows"]
                    if x["budget_tiles"] == record["budget_tiles"]
                )
                values = [NAMES[result["model"]][language], str(record["budget_tiles"])]
                for stratum in ["all_cells", "mixed_cells"]:
                    values.extend(f"{metrics[stratum][k]:.4f}" for k in ["mae", "rmse"])
                rows.append(values)
        _table(
            out / "jstars_mixed.tex",
            (
                "10米建筑覆盖比例：固定查询域的MAE/RMSE，排除20个候选支持图块及固定校准图块。"
                "混合像元满足参考建筑比例"
                "$0<f<1$；参考由CLSM机器提取建筑矢量在2米网格采样后聚合获得。"
                if not english
                else "Building fractions on the 10-m grid: MAE/RMSE on fixed query cells "
                "outside the maximum 20-support-tile reservation and calibration. "
                "Mixed cells satisfy reference fraction $0<f<1$. "
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
        task_registry=str(task_registry(root)),
        composition_study=str(composition_root(root)),
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
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
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
    spec = json.loads((composition_root(root) / "spec.json").read_text())
    data = json.loads(verified(spec["reference"]).read_text())
    receipt = json.loads((composition_root(root) / "reference.json").read_text())
    truth = np.load(verified(receipt["targets"]), mmap_mode="r")
    excluded = set(data["train_positions"]) | set(data["calibration_positions"])
    candidates = []
    for i in range(len(truth)):
        crop = truth[i, 32:96, 32:96]
        mixed = (crop > 0) & (crop < 1)
        if i not in excluded and float(crop.mean()) >= 0.05 and float(mixed.mean()) >= 0.1:
            candidates.append(i)
    if len(candidates) < 3:
        raise ValueError("Insufficient reference-defined building-composition cases")
    ordered = sorted(candidates, key=lambda i: (float(truth[i, 32:96, 32:96].mean()), i))
    selected = [ordered[round(0.75 * (len(ordered) - 1))]]
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
        p = composition_root(root) / name / "readout-20"
        models[name] = load_readout(p, sha(p / "identity.json"))
        sources.append({"features": ref, "reader": str(p), "identity": sha(p / "identity.json")})
    plt.rcParams.update({"font.size": 14, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.5), constrained_layout=True, squeeze=False)
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
                ax.set_title(titles[column], fontsize=14)
            ax.plot([0.06, 0.21625], [0.06, 0.06], color="white", lw=2, transform=ax.transAxes)
            ax.text(
                0.138125,
                0.095,
                "100 m",
                ha="center",
                color="white",
                fontsize=12,
                transform=ax.transAxes,
            )
        axes[row, 0].set_ylabel(f"Case {row + 1}", fontsize=14)
        records.append(
            dict(
                position=i,
                tile=record,
                reference_mean=float(truth[i, 32:96, 32:96].mean()),
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
            "75th percentile central640m reference mean for a dense-building "
            "structural illustration; tiles "
            "outside all training and calibration positions; eligible central crops "
            "have building fraction mean>=0.05 and mixed-cell share>=0.1; no prediction scores"
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
    registry = json.loads(task_registry(root).read_text())
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
    archive = task_registry(root).parent / "compiled-task-results.json"
    archive.write_text(json.dumps(ledger, indent=2) + "\n")
    compact = dict(
        sources=sources,
        full_results={"path": str(archive), "sha256": sha(archive)},
        completed_jobs=len(sources),
        expected_jobs=len(registry),
        result_rows=len(results),
        targets=sorted(targets),
        code_sha256=sha(__file__),
    )
    (paper / "notes/jstars-task-results.json").write_text(json.dumps(compact, indent=2) + "\n")
    return ledger


def temporal_tables(root, paper):
    root, paper = Path(root), Path(paper)
    paths = [
        ("six", root.parent / "full-region/selected_method/result.json"),
        ("spread", root / "temporal-source/three_spread/evaluation/result.json"),
        ("consecutive", root / "temporal-source/three_consecutive/evaluation/result.json"),
        ("one", root / "temporal-source/one_observation/evaluation/result.json"),
    ]
    results = []
    for name, path in paths:
        value = json.loads(path.read_text())
        if value["state"] != "complete":
            raise ValueError("Whole-source temporal condition incomplete")
        key = "all_region" if name == "six" else "metrics"
        numbers = []
        for scope in [key, "outside_support_and_calibration"]:
            for head in ["mlp", "conv3x3"]:
                rows = [x for x in value["rows"] if x["head"] == head]
                assert len(rows) == 4
                numbers.append(f"{100*np.mean([x[scope]['f1'] for x in rows]):.3f}")
        results.append(
            dict(name=name, values=numbers, source={"path": str(path), "sha256": sha(path)})
        )
    labels = {
        "six": ("六份实际观测", "Six actual observations"),
        "spread": ("三份分散观测", "Three spread observations"),
        "consecutive": ("三份连续观测", "Three consecutive observations"),
        "one": ("一份实际观测", "One actual observation"),
    }
    for english in [False, True]:
        _table(
            paper / "tables" / ("en" if english else "") / "jstars_temporal.tex",
            (
                "Whole-source temporal retention with fixed model and reader parameters. "
                "The temporal RGB median and derived local features are recomputed from "
                "retained raw observations. Values are four-target mean F1 (percent)."
                if english
                else "完整地方来源的时间保留：固定模型、投影和读取器，"
                "从保留原始观测重新生成中值合成及派生特征。"
                "报告四类平均F1（百分数）。"
            ),
            "tab:jstars-temporal",
            "lrrrr",
            (
                ["Condition", "All MLP", "All CNN", "Query MLP", "Query CNN"]
                if english
                else ["观测条件", "全区MLP", "全区CNN", "查询MLP", "查询CNN"]
            ),
            [[labels[x["name"]][int(english)], *x["values"]] for x in results],
        )
    (paper / "notes/jstars-temporal-ledger.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def fine_detail_tables(root, paper):
    root, paper = Path(root), Path(paper)
    results = []
    for name, path, key in [
        ("native", root.parent / "full-region/selected_method/result.json", "all_region"),
        ("degraded", root / "fine-detail/degraded/evaluation/result.json", "metrics"),
    ]:
        value = json.loads(path.read_text())
        if value["state"] != "complete":
            raise ValueError("Fixed-weight fine-detail intervention incomplete")
        results.append(
            dict(
                name=name,
                result=value,
                all_region_key=key,
                source={"path": str(path), "sha256": sha(path)},
            )
        )
    for english in [False, True]:
        rows = []
        for entry in results:
            row = [NAMES[entry["name"]][int(english)]]
            for scope in [entry["all_region_key"], "outside_support_and_calibration"]:
                for head in ["mlp", "conv3x3"]:
                    values = [x[scope]["f1"] for x in entry["result"]["rows"] if x["head"] == head]
                    assert len(values) == 4
                    row.append(f"{100*np.mean(values):.3f}")
            rows.append(row)
        _table(
            paper / "tables" / ("en" if english else "") / "jstars_fine_detail.tex",
            (
                "Same-scene 10-m degradation with frozen weights, PCA, regional parameters, "
                "readers, and thresholds. Dates, footprints, and color normalization are "
                "retained. Four-target mean F1 (percent)."
                if english
                else "同景10米降质：固定视觉权重、PCA、区域融合参数、读取器及阈值；"
                "日期、有效覆盖与颜色归一化一致。报告四类平均F1（百分数）。"
            ),
            "tab:jstars-fine-detail",
            "lrrrr",
            (
                ["Input", "All MLP", "All CNN", "Query MLP", "Query CNN"]
                if english
                else ["输入", "全区MLP", "全区CNN", "查询MLP", "查询CNN"]
            ),
            rows,
        )
    ledger = dict(
        sources=[x["source"] for x in results],
        scope=(
            "Same fixed native-trained model and readers; "
            "degrade imagery before all local encoders"
        ),
        native_product_spacing_m=3,
        degraded_spacing_m=10,
        before_and_after_statistical_fits=False,
    )
    (paper / "notes/jstars-fine-detail-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger


def temporal_figure(root, paper):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root, paper = Path(root), Path(paper)
    points = {}
    sources = []
    for name, path in [
        ("six", root.parent / "full-region/selected_method/result.json"),
        ("three_spread", root / "temporal-source/three_spread/evaluation/result.json"),
        ("three_consecutive", root / "temporal-source/three_consecutive/evaluation/result.json"),
        ("one", root / "temporal-source/one_observation/evaluation/result.json"),
    ]:
        result = json.loads(path.read_text())
        if result["state"] != "complete":
            raise ValueError("Temporal curve requires completed whole-source conditions")
        key = "all_region" if name == "six" else "metrics"
        points[name] = {
            h: 100 * float(np.mean([x[key]["f1"] for x in result["rows"] if x["head"] == h]))
            for h in ["mlp", "conv3x3"]
        }
        sources.append({"path": str(path), "sha256": sha(path)})
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    handles = []
    for head, color, name in [("mlp", "#31688e", "MLP"), ("conv3x3", "#a34314", "CNN")]:
        line = ax.plot(
            [1, 3, 6],
            [points[n][head] for n in ["one", "three_spread", "six"]],
            "o-",
            color=color,
            label=name,
            linewidth=1.4,
            markersize=4,
        )[0]
        handles.append(line)
        ax.scatter([3], [points["three_consecutive"][head]], marker="^", color=color, s=40)
    handles.append(Line2D([], [], marker="^", color="black", linestyle="", label="3 consecutive"))
    ax.set(xlabel="Actual local observations", ylabel="Mean F1 (%)", xticks=[1, 3, 6])
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, frameon=False, fontsize=7.5)
    fig.savefig(paper / "figures/jstars_temporal.pdf")
    plt.close(fig)
    (paper / "notes/jstars-temporal-curve.json").write_text(
        json.dumps(dict(points=points, sources=sources), indent=2) + "\n"
    )


def fixed_component_tables(root, paper):
    root, paper = Path(root), Path(paper)
    entries = [
        (
            ("完整融合", "Complete fusion"),
            root.parent / "full-region/selected_method/result.json",
            "all_region",
        ),
        (
            ("关闭结构补充", "Structural increment off"),
            root / "guidance-controls/without_geometry_fixed/evaluation/result.json",
            "metrics",
        ),
        (("AEF", "AEF"), root.parent / "full-region/AEF/result.json", "all_region"),
        (
            ("AEF及相同引导", "AEF with the same guide"),
            root / "guidance-controls/aef_guidance_nchw/evaluation/result.json",
            "metrics",
        ),
    ]
    values, sources = [], []
    for labels, path, scope in entries:
        result = json.loads(path.read_text())
        if result["state"] != "complete":
            raise ValueError("Fixed component comparison incomplete")
        numbers = []
        for head in ["mlp", "conv3x3"]:
            rows = [x for x in result["rows"] if x["head"] == head]
            assert len(rows) == 4
            numbers.append(f"{100*np.mean([x[scope]['f1'] for x in rows]):.3f}")
        for head in ["mlp", "conv3x3"]:
            forest = next(
                x for x in result["rows"] if x["head"] == head and x["task"] == "clcd_forest"
            )
            numbers.append(f"{100*forest[scope]['boundary_f1_1px']:.3f}")
        values.append((labels, numbers))
        sources.append({"path": str(path), "sha256": sha(path)})
    for english in [False, True]:
        _table(
            paper / "tables" / ("en" if english else "") / "jstars_fixed_components.tex",
            (
                "固定读取器的组件干预，报告全区四类平均F1及林地边界F1（百分数）。"
                "结构关闭固定完整流程读取器；AEF引导固定AEF读取器。干预导出采用批量2；"
                "边界容差为一个CLCD原生参考像元（30米）。"
                if not english
                else "Fixed-reader component interventions: all-region four-target mean F1 and "
                "forest boundary F1 (percent). Structural removal retains complete-fusion "
                "readers; AEF guidance retains AEF readers. Intervention exports use batch2. "
                "Boundary "
                "tolerance is one native CLCD reference cell (30m)."
            ),
            "tab:jstars-fixed-components",
            "lrrrr",
            (
                ["条件", "平均MLP", "平均CNN", "边界MLP", "边界CNN"]
                if not english
                else ["Condition", "Mean MLP", "Mean CNN", "Forest BF1 MLP", "Forest BF1 CNN"]
            ),
            [[labels[int(english)], *numbers] for labels, numbers in values],
        )
    (paper / "notes/jstars-fixed-components.json").write_text(
        json.dumps(dict(sources=sources), indent=2) + "\n"
    )
    return sources


def label_budgets(root, paper):
    from xuannv_embedding.training.frozen_fusion import verified

    root, paper = Path(root), Path(paper)
    registry = json.loads(task_registry(root).read_text())
    records = []
    for dataset in sorted({x["dataset"] for x in registry}):
        row = next(
            x
            for x in registry
            if x["dataset"] == dataset
            and x["model"] == "native"
            and x["head"] == "mlp"
            and x["budget"] == 20
        )
        job = json.loads(verified(row["spec"]).read_text())
        data = json.loads(verified(job["reference"]).read_text())
        valid = np.load(verified(data["valid"]), mmap_mode="r")
        val = data["calibration_positions"]
        with np.load(verified(data["labels"])) as labels:
            counts = {
                task: int(((labels[task][val] >= 0) & valid[val]).sum())
                for tasks in data["tasks"]["C"].values()
                for task in tasks
            }
        records.append(
            dict(
                dataset=dataset,
                reference=job["reference"],
                calibration_version=data.get("calibration_version"),
                calibration_tiles=len(val),
                calibration_label_cells_per_target=counts,
                support_budgets=[5, 20],
                support_unit="Fitting tiles per target",
                retrieval_budget=3,
                retrieval_unit=(
                    "Positive connected components; "
                    "prototype parameters come from training examples"
                ),
            )
        )
    (paper / "notes/jstars-label-budgets.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
