import numpy as np
import pytest

from xuannv_embedding.data_process.v5_dem_corrections import (
    correct_dem,
    payload_sha256,
    validate_payload,
)
from xuannv_embedding.data_process.v5_dem_geometry import legacy_slope, slope_support


def test_corrected_dem_never_supervises_slope_derived_from_missing_elevation():
    elevation = np.indices((128, 128))[1].astype("f4")
    valid = np.ones(elevation.shape, bool)
    valid[64, 64] = False
    result = correct_dem(elevation, valid)
    assert np.array_equal(result["elevation_valid"], valid)
    assert np.array_equal(result["slope_valid"], slope_support(valid))
    assert not result["slope_valid"][64, 63]
    assert result["slope_valid"][63, 63]
    assert result["slope"][64, 63] == 0
    assert result["elevation"][64, 64] == 0
    assert result["slope"][0, 0] == legacy_slope(elevation, valid)[0, 0]
    validate_payload(result)


def test_correction_keeps_valid_negative_elevation_and_rejects_corrupt_values():
    elevation = np.full((128, 128), -25, "f4")
    valid = np.ones(elevation.shape, bool)
    result = correct_dem(elevation, valid)
    assert np.all(result["elevation"] == -25)
    assert not result["slope"].any()
    before = payload_sha256(result)
    result["elevation"][0, 0] += 1
    assert payload_sha256(result) != before
    result["elevation"][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_payload(result)
    valid[:] = False
    result = correct_dem(elevation, valid)
    assert not any(a.any() for a in result.values())


def test_dem_correction_cli_exposes_only_data_processing(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_dem_corrections

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    monkeypatch.setattr(
        v5_dem_corrections, "build_dem_corrections", lambda *a, **k: calls.append((a, k))
    )
    argv = ["--stage", "dem-corrections"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(argv) == 0
    assert calls == [((tmp_path / "dataset-root", tmp_path / "report-root"), {})]


def test_full_correction_layer_is_sparse_reusable_and_rejects_mutated_baseline(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from pathlib import Path

    import pandas as pd
    import zarr

    from xuannv_embedding.data_process import v5_dem_corrections as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json
    from xuannv_embedding.data_process.v5_target_geometry import compare_target

    data, report = tmp_path / "data", tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    report.mkdir()
    registry = pd.DataFrame(
        {
            "patch_id": ["a", "b"],
            "split": ["train", "test"],
            "grid_epsg": [32650, 32650],
            "utm_bounds": [[0, 0, 1280, 1280], [1280, 0, 2560, 1280]],
        }
    )
    regpath = data / "registry/national_62000.parquet"
    registry.to_parquet(regpath, index=False)
    basepath = tmp_path / "baseline.zarr"
    root = zarr.open_group(str(basepath), mode="w")
    root.attrs["patch_ids"] = ["a", "b"]
    elevation = np.indices((128, 128))[1].astype("f4")
    valid = np.ones((128, 128), bool)
    old_valid = valid.copy()
    old_valid[-1, 0] = False
    old_elevation = np.where(old_valid, elevation, 0).astype("f4")
    original = {
        "targets/dem_elevation": np.stack([old_elevation, elevation]),
        "targets/dem_slope": np.stack(
            [legacy_slope(old_elevation, old_valid), legacy_slope(elevation, valid)]
        ),
        "valid_masks/dem_elevation": np.stack([old_valid, valid]),
        "valid_masks/dem_slope": np.stack([old_valid, valid]),
    }
    for name, arr in original.items():
        root.create_dataset(name, data=arr)
    metadata = sha256(basepath / ".zattrs")
    manifest = pd.DataFrame(
        [
            {
                "family": "static",
                "array": name,
                "path": str(basepath),
                "registry_order_verified": True,
                "temporal_mode": "static",
                "source_metadata_sha256": metadata,
            }
            for name in original
        ]
    )
    mpath = data / "targets/manifest.parquet"
    manifest.to_parquet(mpath, index=False)
    values = pd.DataFrame(
        [
            {
                "family": "static",
                "array": name,
                "path": str(basepath),
                "decoded_values_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            }
            for name, arr in original.items()
        ]
    )
    values.to_parquet(report / "target_value_audit.parquet", index=False)
    part = tmp_path / "source.zip.part1"
    part.write_bytes(b"fixed_source")
    fp = {
        "registry_sha256": sha256(regpath),
        "manifest_sha256": sha256(mpath),
        "sources": [{"path": str(part), "sha256": sha256(part)}],
        "code_sha256": {
            name: sha256(Path(module.__file__).with_name(name))
            for name in ["v5_dem_geometry.py", "v5_target_geometry.py"]
        },
    }
    observations = []
    for i, pid in enumerate(["a", "b"]):
        for name, fresh in [
            ("dem_elevation", elevation),
            ("dem_slope", legacy_slope(elevation, valid)),
        ]:
            observations.append(
                {
                    "patch_id": pid,
                    "target": name,
                    **compare_target(
                        fresh,
                        valid,
                        original["targets/" + name][i],
                        original["valid_masks/" + name][i],
                        categorical=False,
                    ),
                }
            )
    opath = data / "quality/targets/geometry/dem/full/observations.parquet"
    opath.parent.mkdir(parents=True)
    pd.DataFrame(observations).to_parquet(opath, index=False)
    write_json(
        report / "target_geometry_dem_full.json",
        {
            "status": "dem_geometry_audit_finished",
            "scope": "full",
            "processed_targets": 4,
            "selected_targets": 4,
            "failed_targets": 2,
            "fingerprint": fp,
            "output": str(opath),
        },
    )

    class Source:
        def __init__(self, *args):
            pass

        def reconstruct(self, *args):
            return elevation.copy(), valid.copy(), ["actual.tif"]

        def close(self):
            pass

    monkeypatch.setattr(module, "DEMSource", Source)
    result = module.build_dem_corrections(data, report)
    assert result["processed_positions"] == 2 and result["corrected_positions"] == 1
    assert result["original_unsupported_slope_pixels"] == 2
    assert result["added_elevation_pixels"] == 1
    reader = module.CorrectedDEMReader(data)
    new = reader.read("a")
    unchanged = reader.read("b")
    assert new["elevation_valid"].all() and new["slope_valid"].all()
    assert np.array_equal(new["slope"], legacy_slope(elevation, valid))
    assert np.array_equal(unchanged["elevation"], elevation)
    for name, arr in original.items():
        assert np.array_equal(root[name][:], arr)
    lock = Path(result["output"]) / "corrections.lock.json"
    first = lock.read_bytes()
    assert module.build_dem_corrections(data, report)["corrected_positions"] == 1
    assert lock.read_bytes() == first
    root["targets/dem_elevation"][1, 0, 0] = 123
    with pytest.raises(ValueError, match="baseline"):
        reader.read("b")
    with pytest.raises(ValueError):
        module.build_dem_corrections(data, report)
    root["targets/dem_elevation"][1, 0, 0] = 0
    record = pd.read_parquet(Path(result["output"]) / "manifest.parquet").iloc[0]
    bundle = Path(result["output"]) / record.correction_file
    bundle.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="correction"):
        reader.read("a")
    state = json.loads(lock.read_text())
    assert state["training_authorized"] is False
