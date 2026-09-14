import numpy as np
import pytest

from xuannv_embedding.data_process.v5_negative_corrections import corrected_overlay
from xuannv_embedding.data_process.v5_negative_rules import TASKS


def test_corrected_slope_can_revoke_and_add_only_evidence_backed_water_negatives():
    shape = (128, 128)
    states = {name: np.zeros(shape, "u1") for name in TASKS}
    wc = np.zeros(shape, "u1")
    valid = np.ones(shape, bool)
    slope = np.zeros(shape, "f4")
    slope[0, 0] = 25
    slope[0, 1] = 25
    slope_valid = valid.copy()
    slope_valid[0, 1] = False
    states["water_area"][0, 2] = 1
    slope[0, 2] = 30
    overlay = corrected_overlay(
        states,
        wc,
        valid,
        slope,
        slope_valid,
        {"erosion_pixels": 2, "steep_slope_degrees": 20.0, "negative_confidence": 224},
    )
    assert overlay["states"].shape == (4, 128, 128)
    assert overlay["states"][2, 0, 0] == 3 and overlay["confidence"][2, 0, 0] == 224
    assert overlay["states"][2, 0, 1] == 0 and overlay["states"][2, 0, 2] == 0
    assert not overlay["states"][[0, 1, 3]].any()
    slope[:] = 0
    assert not corrected_overlay(
        states,
        wc,
        valid,
        slope,
        valid,
        {"erosion_pixels": 2, "steep_slope_degrees": 20.0, "negative_confidence": 224},
    )["states"].any()


def test_negative_corrections_cli_rejects_partial_scope(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_negative_corrections

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    monkeypatch.setattr(
        v5_negative_corrections, "build_negative_corrections", lambda *a: calls.append(a)
    )
    argv = ["--stage", "target-negative-corrections"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(argv) == 0
    assert calls == [(tmp_path / "dataset-root", tmp_path / "report-root")]
    with pytest.raises(ValueError):
        v5_cli.main(argv + ["--max-patches", "32"])


def test_negative_correction_materialization_checks_actual_source_chunks_and_reuses_lock(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from pathlib import Path

    import pandas as pd
    import zarr

    from xuannv_embedding.data_process import v5_negative_corrections as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data = tmp_path / "data"
    report = tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    report.mkdir()
    registry = pd.DataFrame({"patch_id": ["a"], "split": ["train"]})
    registry.to_parquet(data / "registry/national_62000.parquet", index=False)
    parameters = {"erosion_pixels": 2, "steep_slope_degrees": 20.0, "negative_confidence": 224}
    roots = {}
    paths = {}
    manifest = []
    shape = (128, 128)
    states = {t: np.zeros(shape, "u1") for t in TASKS}
    wc = np.zeros(shape, "u1")
    valid = np.ones(shape, bool)
    slope = np.zeros(shape, "f4")
    slope[0, 0] = 30
    overlay = corrected_overlay(states, wc, valid, slope, valid, parameters)
    for family in ["osm", "static", "reliable_negative"]:
        paths[family] = tmp_path / (family + ".zarr")
        roots[family] = zarr.open_group(str(paths[family]), mode="w")
        roots[family].attrs["patch_ids"] = ["a"]
        manifest.append({"family": family, "path": str(paths[family])})
    for year in [2020, 2021]:
        for t in TASKS:
            roots["osm"].create_dataset(f"{year}/states/{t}", data=states[t][None])
        roots["static"].create_dataset(f"targets/worldcover_{year}", data=wc[None])
        roots["static"].create_dataset(f"valid_masks/worldcover_{year}", data=valid[None])
        for group in ["states", "confidence"]:
            for i, t in enumerate(TASKS):
                roots["reliable_negative"].create_dataset(
                    f"{year}/{group}/{t}", data=overlay[group][i][None]
                )
    roots["static"].create_dataset("targets/dem_slope", data=slope[None])
    roots["static"].create_dataset("valid_masks/dem_slope", data=valid[None])
    pd.DataFrame(manifest).to_parquet(data / "targets/manifest.parquet", index=False)
    (report / "target_value_audit.parquet").write_bytes(b"fixture")
    write_json(report / "osm_temporal_progress.json", {"fixture": True})
    fp = {
        "registry_sha256": sha256(data / "registry/national_62000.parquet"),
        "manifest_sha256": sha256(data / "targets/manifest.parquet"),
        "value_audit_sha256": sha256(report / "target_value_audit.parquet"),
        "temporal_audit_sha256": sha256(report / "osm_temporal_progress.json"),
        "code_sha256": sha256(Path(module.__file__).with_name("v5_negative_rules.py")),
        "metadata_sha256": {f: sha256(p / ".zattrs") for f, p in paths.items()},
        "parameters": parameters,
    }
    for year in [2020, 2021]:
        required = [("osm", f"{year}/states/{t}") for t in TASKS]
        required += [
            ("static", f"{g}/{t}")
            for g in ["targets", "valid_masks"]
            for t in [f"worldcover_{year}", "dem_slope"]
        ]
        required += [
            ("reliable_negative", f"{year}/{g}/{t}")
            for g in ["states", "confidence"]
            for t in TASKS
        ]
        digest = hashlib.sha256()
        for f, a in required:
            digest.update(roots[f][a][:].tobytes())
        write_json(
            data / "quality/targets/negative_rules/full/chunks" / f"{year}_000000.json",
            {
                "fingerprint": {
                    **fp,
                    "year": year,
                    "start": 0,
                    "stop": 1,
                    "decoded_chunk_sha256": digest.hexdigest(),
                }
            },
        )
    write_json(
        report / "negative_rule_audit_full.json",
        {
            "status": "negative_rule_audit_finished",
            "scope": "full",
            "failed_targets": 0,
            "processed_targets": 8,
            "selected_targets": 8,
            "fingerprint": fp,
        },
    )
    new_slope = np.zeros(shape, "f4")
    new_slope[0, 1] = 30

    class DEM:
        def __init__(self, *a):
            pass

        def read(self, *a):
            return {"slope": new_slope.copy(), "slope_valid": valid.copy()}

    monkeypatch.setattr(module, "CorrectedDEMReader", DEM)
    demdir = data / "targets/corrections/dem/v1"
    demdir.mkdir(parents=True)
    pd.DataFrame(
        {"patch_id": ["a"], "index": [0], "split": ["train"], "correction_file": ["patch.npz"]}
    ).to_parquet(demdir / "manifest.parquet", index=False)
    write_json(
        demdir / "corrections.lock.json", {"manifest_sha256": sha256(demdir / "manifest.parquet")}
    )
    summary = module.build_negative_corrections(data, report)
    assert summary["added_negative_pixels"] == 2 and summary["removed_negative_pixels"] == 2
    assert summary["corrected_position_years"] == 2
    lock = Path(summary["output"]) / "corrections.lock.json"
    first = lock.read_bytes()
    module.build_negative_corrections(data, report)
    assert lock.read_bytes() == first
    reader = module.CorrectedNegativeReader(data, report)
    result = reader.read("a", 2020)
    assert result["states"][2, 0, 0] == 0 and result["states"][2, 0, 1] == 3
    assert roots["reliable_negative"]["2020/states/water_area"][0, 0, 0] == 3
    roots["osm"]["2020/states/road_all"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="evidence pixels"):
        reader.read("a", 2020)
    roots["osm"]["2020/states/road_all"][0, 0, 0] = 0
    row = pd.read_parquet(Path(summary["output"]) / "manifest.parquet").iloc[0]
    (Path(summary["output"]) / row.correction_file).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="correction file"):
        reader.read("a", 2020)
    assert json.loads(lock.read_text())["training_authorized"] is False
