"""Frozen 5m/10m cloud comparison on an immutable pilot; never replaces production QA."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from affine import Affine
from rasterio.warp import Resampling, reproject

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_quality import quality_masks, transfer_invalid
from xuannv_embedding.data_process.v5_rasters import BRANCH_BANDS, read_native
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def downsample_for_cloud(values, valid, transform, crs):
    """Area-average 5m to 10m; reject coarse cells with any missing contributing pixel."""
    if values.ndim != 3 or values.shape[1:] != valid.shape or any(n % 2 for n in valid.shape):
        raise ValueError("expected aligned even native dimensions and a spatial validity mask")
    if not np.all(np.isfinite(values[:, valid])):
        raise ValueError("nonfinite valid input")
    shape = tuple(n // 2 for n in valid.shape)
    destination_transform = transform * Affine.scale(2, 2)
    geometry = {
        "src_transform": transform,
        "src_crs": crs,
        "dst_transform": destination_transform,
        "dst_crs": crs,
        "resampling": Resampling.average,
    }
    fraction = np.zeros(shape, dtype="f4")
    reproject(valid.astype("f4"), fraction, **geometry)
    coarse_valid = fraction >= 1 - 1e-6
    output = np.zeros((len(values), *shape), dtype="f4")
    for i, band in enumerate(values):
        source = np.where(valid, band, np.nan).astype("f4")
        reproject(source, output[i], src_nodata=np.nan, dst_nodata=0, **geometry)
    output[:, ~coarse_valid] = 0
    return output, coarse_valid, destination_transform


def mask_difference(baseline, candidate, data_valid):
    if baseline.shape != candidate.shape or baseline.shape != data_valid.shape:
        raise ValueError("comparison grids disagree")
    if np.any((baseline | candidate) & ~data_valid):
        raise ValueError("comparison promotes NoData to valid")
    count = int(data_valid.sum())
    return {
        "native_valid_pixels": int(baseline.sum()),
        "candidate_valid_pixels": int(candidate.sum()),
        "newly_valid_pixels": int((candidate & ~baseline).sum()),
        "newly_invalid_pixels": int((baseline & ~candidate).sum()),
        "data_valid_pixels": count,
        "disagreement_fraction_of_data": (
            float((baseline != candidate).sum() / count) if count else None
        ),
    }


def _render(
    values, valid, native_classes, coarse_classes, native_mask, candidate_mask, path, title
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # B5/B4/B3 are identified red/green/blue bands; stretch is display-only.
    image = values[[4, 3, 2]].copy()
    for band in image:
        low, high = np.percentile(band[valid], [2, 98]) if valid.any() else (0, 1)
        band[:] = np.clip((band - low) / max(float(high - low), 1e-6), 0, 1)
    image[:, ~valid] = 0
    colors = ListedColormap(["#28965a", "#ffffff", "#68cce7", "#555555"])
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    panels = [
        (image.transpose(1, 2, 0), "Native RGB", None),
        (native_classes, "5m cloud classes", colors),
        (coarse_classes, "10m cloud classes", colors),
        (native_mask, "5m QA + 30m buffer", "gray"),
        (candidate_mask, "10m QA + 30m buffer → 5m", "gray"),
        (
            candidate_mask.astype("i1") - native_mask.astype("i1"),
            "Changed: blue removed / red added",
            "coolwarm",
        ),
    ]
    for axis, (array, label, cmap) in zip(axes, panels, strict=True):
        limits = (
            {"vmin": 0, "vmax": 3}
            if cmap is colors
            else {"vmin": -1, "vmax": 1} if label.startswith("Changed") else {"vmin": 0, "vmax": 1}
        )
        axis.imshow(array, cmap=cmap, interpolation="nearest", extent=(0, 1280, 0, 1280), **limits)
        axis.set_title(label, fontsize=9)
        axis.set_xticks([0, 640, 1280])
        axis.set_yticks([0, 640, 1280])
        axis.grid(alpha=0.15)
    fig.suptitle(title + " | diagnostic, no reference cloud labels", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compare_jilin_resolution(dataset_root, report_root, quality_root, model_dir, *, device_id=7):
    from xuannv_embedding.data_process.omnicloudmask_npu import AscendOmniCloudMaskV4Predictor

    snapshot = quality_root / "catalog.snapshot.parquet"
    baseline_lock = json.loads((quality_root / "source.lock.json").read_text())
    if baseline_lock.get("limit") is None or sha256(snapshot) != baseline_lock["catalog_sha256"]:
        raise ValueError("an immutable completed pilot snapshot is required")
    files = pd.read_parquet(snapshot).set_index("observation_id")
    order = pd.read_parquet(quality_root / "observation_order.parquet")
    models = [model_dir / f"ocm_v4_model_{i}_96_910b4.om" for i in (0, 1)]
    if {p.name: sha256(p) for p in models} != baseline_lock["models_sha256"]:
        raise ValueError("comparison model weights differ from baseline")
    for name, digest in baseline_lock["code_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("baseline inference contract changed")
    classes = zarr.open_group(str(quality_root / "classes.zarr"), mode="r")["classes"]
    masks = zarr.open_group(str(quality_root / "valid_masks.zarr"), mode="r")
    if not np.asarray(masks["completed"][:]).all() or len(order) != len(classes):
        raise ValueError("baseline cloud pilot is incomplete")
    directory = report_root / "diagnostics/jilin_resolution_5m_10m_v1"
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "baseline_source_lock_sha256": sha256(quality_root / "source.lock.json"),
        "catalog_sha256": sha256(snapshot),
        "order_sha256": sha256(quality_root / "observation_order.parquet"),
        "native_gsd": 5,
        "candidate_gsd": 10,
        "cloud_buffer_m": 30,
        "code_sha256": sha256(Path(__file__)),
        "model_sha256": baseline_lock["models_sha256"],
        "input_bands": ["B5", "B4", "B6"],
        "resampling": "area average; all contributing data must be valid",
        "scope": "fixed pilot diagnostics only; no production QA replacement",
    }
    lock_path = directory / "source.lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != fingerprint:
        raise ValueError("comparison inputs changed; choose a new diagnostic version")
    write_json(lock_path, fingerprint)
    output = zarr.open_group(str(directory / "comparison.zarr"), mode="a")
    for name, shape, dtype in [
        ("classes_10m", (len(order), 128, 128), "u1"),
        ("valid_5m_from_10m", (len(order), 256, 256), "bool"),
    ]:
        output.require_dataset(name, shape=shape, chunks=(1, *shape[1:]), dtype=dtype)
    rows = []
    predictor = AscendOmniCloudMaskV4Predictor(model_paths=models, device_id=device_id)
    try:
        for index, item in enumerate(order.itertuples()):
            row = files.loc[item.observation_id]
            if int(row.year) not in (2020, 2021) or row.product_id != "jilin1_ms_5m":
                raise ValueError("comparison observation does not match pilot product/year")
            if sha256(Path(row.path)) != row.file_sha256:
                raise ValueError("pilot source pixels changed")
            frame = read_native(Path(row.path), BRANCH_BANDS["jilin1_ms_5m"])
            valid = np.all(frame.valid, axis=0)
            rgb = frame.values[[4, 3, 5]].copy()
            rgb[:, ~valid] = 0
            native, _ = predictor.predict_batch([rgb])
            if not np.array_equal(native[0], classes[index]):
                raise ValueError("native rerun differs from stored pilot baseline")
            native_qa = quality_masks(native[0], valid, gsd=5)
            if not np.array_equal(native_qa["valid"], masks["valid"][index]):
                raise ValueError("stored baseline mask disagrees with frozen policy")
            coarse, coarse_valid, transform = downsample_for_cloud(
                rgb, valid, Affine(*frame.transform), frame.crs
            )
            candidate, _ = predictor.predict_batch([coarse])
            coarse_qa = quality_masks(candidate[0], coarse_valid, gsd=10)
            invalid = transfer_invalid(
                ~coarse_qa["valid"],
                src_transform=transform,
                src_crs=frame.crs,
                dst_transform=Affine(*frame.transform),
                dst_crs=frame.crs,
                shape=valid.shape,
            )
            final = valid & ~invalid
            output["classes_10m"][index] = candidate[0]
            output["valid_5m_from_10m"][index] = final
            metrics = mask_difference(native_qa["valid"], final, valid)
            filename = f"jilin_{index:02d}.png"
            _render(
                frame.values,
                valid,
                native[0],
                candidate[0],
                native_qa["valid"],
                final,
                directory / filename,
                f"{row.sensor} | {row.acquired_at} | {row.patch_id}",
            )
            rows.append(
                {
                    "observation_id": item.observation_id,
                    "patch_id": row.patch_id,
                    "year": int(row.year),
                    "image": filename,
                    **metrics,
                }
            )
            write_json(
                directory / "progress.json",
                {
                    "processed": len(rows),
                    "selected": len(order),
                    "status": "running",
                    "updated_at": now(),
                },
            )
    finally:
        predictor.close()
    atomic_parquet(pd.DataFrame(rows), directory / "comparison.parquet")
    pd.DataFrame(rows).to_csv(directory / "samples.csv", index=False)
    body = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>吉林一号云分辨率对照</title>',
        "<style>body{font-family:system-ui;margin:24px}img{width:100%}"
        "article{margin:30px 0}</style>",
        "<h1>吉林一号：5米与10米冻结云推理对照</h1>",
        "<p>固定原始样例，原生5米重跑与旧结果逐像元一致才进行比较。10米结果投影回原5米网格。",
        "掩膜变化没有人工云真值评分，不能解读为准确率提高，不用于替换正式质量产物。</p>",
    ]
    body += [
        f'<article>{html.escape(r["observation_id"])}'
        f'<img src="{r["image"]}" loading="lazy"></article>'
        for r in rows
    ]
    (directory / "index.html").write_text("\n".join(body + ["</html>"]))
    totals = {
        k: sum(r[k] for r in rows)
        for k in (
            "native_valid_pixels",
            "candidate_valid_pixels",
            "newly_valid_pixels",
            "newly_invalid_pixels",
            "data_valid_pixels",
        )
    }
    result = {
        "status": "diagnostic_comparison_complete",
        "processed": len(rows),
        "unique_positions": len({r["patch_id"] for r in rows}),
        **totals,
        "accuracy_gain_established": False,
        "production_qa_replaced": False,
        "fingerprint": fingerprint,
        "finished_at": now(),
    }
    write_json(directory / "progress.json", result)
    return result
