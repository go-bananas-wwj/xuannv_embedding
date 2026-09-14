import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


def fixture(tmp_path, monkeypatch):
    from test_v5_dense_integrity import make_registry

    from xuannv_embedding.data_process import v5_quarter_index as module
    from xuannv_embedding.data_process.v5_dense_integrity import inspect_dense_archive

    data, report = tmp_path / "data", tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    report.mkdir()
    registry = make_registry()
    registry.to_parquet(data / "registry/national_62000.parquet", index=False)
    roots = {}
    for product, bands, size in [
        ("s1_local", 2, 128),
        ("s2_local", 10, 128),
        ("landsat_local", 6, 43),
    ]:
        archive = tmp_path / (product + ".zip")
        with MemoryFile() as file:
            with file.open(
                driver="GTiff",
                width=size,
                height=size,
                count=bands,
                dtype="uint16",
                crs="EPSG:32650",
                transform=from_origin(500000, 3001280, 1280 / size, 1280 / size),
            ) as dst:
                dst.write(np.ones((bands, size, size), "u2"))
            blob = file.read()
        with ZipFile(archive, "w") as z:
            z.writestr("one.tif", blob)
        root = tmp_path / product
        roots[product] = str(root)
        for year in [2020, 2021]:
            for month in range(1, 13):
                inspect_dense_archive(
                    archive, registry, root, product=product, year=year, month=month
                )
    monkeypatch.setattr(
        module,
        "load_highres_views",
        lambda *a: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}),
    )
    monkeypatch.setattr(module, "load_target_view", lambda *a: {})
    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(
            {
                "schema": "quarter_source_index_v1",
                "dense_audit_roots": roots,
                "highres": {
                    f: {"eligibility_root": str(tmp_path / f), "quality_root": str(tmp_path / f)}
                    for f in ["gaofen", "jilin1"]
                },
                "target_reader_root": str(tmp_path / "target"),
            }
        )
    )
    return data, report, config


def test_real_dense_audits_become_quarter_references_without_inventing_quality(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process.v5_quarter_index import build_quarter_index

    data, report, config = fixture(tmp_path, monkeypatch)
    result = build_quarter_index(data, report, config)
    assert result["quarter_samples"] == 8 and result["dense_observations"] == 72
    assert result["sample_index_gate_passed"] is False and result["usable_base_observations"] == 0
    root = Path(result["output"])
    samples = pd.read_parquet(root / "quarter_samples.parquet")
    assert samples.dense_inventory_ids.map(len).eq(9).all()
    assert samples.dense_observation_ids.map(len).eq(0).all()
    assert samples.missing_monthly_sources.map(len).eq(0).all()
    assert samples.target_year.equals(samples.year)
    raw = pd.read_parquet(root / "dense_observations.parquet")
    assert raw.contract_status.eq("pending").all() and raw.quality_status.eq("pending").all()
    assert raw.file_sha256.notna().all() and raw.member_name.eq("one.tif").all()


def test_quarter_index_reuses_sealed_output_and_rejects_changed_monthly_inventory(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process.v5_quarter_index import build_quarter_index
    from xuannv_embedding.data_process.v5_sources import sha256

    data, report, config = fixture(tmp_path, monkeypatch)
    first = build_quarter_index(data, report, config)
    root = Path(first["output"])
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in root.iterdir()}
    assert build_quarter_index(data, report, config)["reused"]
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}
    entry = json.loads(config.read_text())
    p = Path(entry["dense_audit_roots"]["s1_local"]) / "s1_local_2020_01.parquet"
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dense audit"):
        build_quarter_index(data, report, config)


def test_quarter_cli_requires_explicit_source_manifest(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_quarter_index

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(v5_quarter_index, "build_quarter_index", lambda *a: calls.append(a))
    args = ["--stage", "sample-index"]
    for k in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + k, str(tmp_path / k)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--sample-inputs", str(tmp_path / "inputs.json")]
    assert v5_cli.main(args) == 0 and calls[0][-1] == tmp_path / "inputs.json"
    with pytest.raises(ValueError, match="complete"):
        v5_cli.main(args + ["--max-patches", "1"])


def test_one_year_without_highres_keeps_same_quarter_file_schema(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_quarter_index as module

    data, report, config = fixture(tmp_path, monkeypatch)
    scene = pd.DataFrame(
        [
            dict(
                patch_id="location",
                split="train",
                family="jilin1",
                year=2020,
                acquired_at="2020-12-31",
                scene_group_id="late",
                available=True,
                candidate_qualified=True,
                strict_pixel_fusion_candidate=False,
                alignment_status="unknown",
                clear_fraction=0.2,
            )
        ]
    )
    monkeypatch.setattr(module, "load_highres_views", lambda *a: (scene, scene, scene, {}))
    result = module.build_quarter_index(data, report, config)
    samples = pd.read_parquet(Path(result["output"]) / "quarter_samples.parquet")
    assert samples.loc[samples.year.eq(2020)].highres_scene_ids.map(len).eq(1).all()
    assert samples.loc[samples.year.eq(2021)].highres_scene_ids.map(len).eq(0).all()


def test_highres_adapter_checks_sealed_member_lists(tmp_path, monkeypatch):
    from test_v5_candidate_statistics import fixture as candidate_fixture

    from xuannv_embedding.data_process import v5_quarter_index as module
    from xuannv_embedding.data_process.v5_sources import sha256

    Reader, root, _, _ = candidate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "HighresQualityReader", Reader)
    config = {"jilin1": {"eligibility_root": str(root), "quality_root": str(tmp_path / "qa")}}
    scene, pool, selected, files = module.load_highres_views(
        tmp_path / "data", config, Reader().registry.reset_index()
    )
    assert scene.loc[scene.scene_group_id.eq("s1"), "candidate_qualified"].item()
    assert "s2" not in set(pool.scene_group_id)
    assert set(pool.split) == {"train", "val", "test"}
    path = root / "scene_groups.parquet"
    frame = pd.read_parquet(path)
    frame.at[1, "eligible_branch_ids"] = []
    frame.to_parquet(path, index=False)
    # Even a self-consistent output checksum cannot substitute for branch membership.
    seal_path = root / "output.lock.json"
    seal = json.loads(seal_path.read_text())
    seal["outputs_sha256"]["scene_groups.parquet"] = sha256(path)
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="members"):
        module.load_highres_views(tmp_path / "data", config, Reader().registry.reset_index())
