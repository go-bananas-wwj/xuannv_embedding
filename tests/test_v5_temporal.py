import numpy as np
import pytest


def test_current_osm_and_unknown_pixels_never_become_historical_supervision():
    from xuannv_embedding.data_process.v5_temporal import inspect_osm_temporal_block

    result = inspect_osm_temporal_block(
        states=np.array([0, 1, 0, 1], dtype="u1"),
        confidence=np.array([0, 255, 0, 255], dtype="u1"),
        targets=np.array([0, 12, 0, 255], dtype="u1"),
        source_bits=np.array([2, 1, 0, 3], dtype="u1"),
    )
    assert result["current_only_unknown_pixels"] == 1
    assert result["historical_positive_pixels"] == 2
    assert result["unknown_pixels"] == 2


@pytest.mark.parametrize(
    "field,value", [("states", 1), ("confidence", 1), ("targets", 1), ("source_bits", 4)]
)
def test_temporal_audit_rejects_cross_array_and_undated_evidence(field, value):
    from xuannv_embedding.data_process.v5_temporal import inspect_osm_temporal_block

    block = {
        k: np.array([0], dtype="u1") for k in ("states", "confidence", "targets", "source_bits")
    }
    block["source_bits"][0] = 2
    block[field][0] = value
    with pytest.raises(ValueError):
        inspect_osm_temporal_block(**block)


def test_temporal_audit_checks_actual_values_against_locked_value_audit(tmp_path):
    import hashlib
    import json

    import pandas as pd
    import zarr

    from xuannv_embedding.data_process.v5_sources import sha256, write_json
    from xuannv_embedding.data_process.v5_temporal import audit_osm_temporal

    data, report, source = tmp_path / "data", tmp_path / "report", tmp_path / "osm.zarr"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    registry = data / "registry/national_62000.parquet"
    pd.DataFrame({"patch_id": ["a", "b"]}).to_parquet(registry)
    root = zarr.open_group(str(source), mode="w")
    root.attrs.update(
        patch_ids=["a", "b"],
        external_evidence=None,
        target_names=["building"],
        structure_names=[],
        indexes={
            "2020": {"path": "index2020", "sha256": "a" * 64},
            "2021": {"path": "index2021", "sha256": "b" * 64},
        },
    )
    entries, values = [], []
    for year in (2020, 2021):
        for field, raw in {
            "states": [1, 0],
            "confidence": [255, 0],
            "targets": [42, 0],
            "source_bits": [1, 2],
        }.items():
            arr = np.array(raw, dtype="u1").reshape(2, 1, 1)
            name = f"{year}/{field}/building"
            root.create_dataset(name, data=arr, chunks=(1, 1, 1))
            entries.append(
                {
                    "family": "osm",
                    "path": str(source),
                    "array": name,
                    "registry_order_verified": True,
                    "source_metadata_sha256": sha256(source / ".zattrs"),
                }
            )
            values.append(
                {
                    "family": "osm",
                    "path": str(source),
                    "array": name,
                    "decoded_values_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                    "status": "values_checked_provenance_pending",
                    "issues": "[]",
                }
            )
    manifest = data / "targets/manifest.parquet"
    pd.DataFrame(entries).to_parquet(manifest)
    for row in values:
        row["source_manifest_sha256"] = sha256(manifest)
    report.mkdir()
    pd.DataFrame(values).to_parquet(report / "target_value_audit.parquet")
    write_json(
        report / "target_source_audit.json",
        {
            "status": "source_audit_finished",
            "sources": [
                {
                    "family": "osm_historical",
                    "year": 2020,
                    "path": "index2020",
                    "actual_sha256": "a" * 64,
                    "status": "matches_original_sha256",
                },
                {
                    "family": "osm_historical",
                    "year": 2021,
                    "path": "index2021",
                    "actual_sha256": "b" * 64,
                    "status": "matches_original_sha256",
                },
            ],
        },
    )
    result = audit_osm_temporal(data, report)
    assert result["checked_groups"] == 2 and result["failed_groups"] == 0
    assert result["historical_positive_pixels"] == 2
    # Both values are semantically valid; changing them must still invalidate the old receipt.
    root["2020/targets/building"][0] = 43
    result = audit_osm_temporal(data, report)
    assert result["failed_groups"] == 1
    rows = pd.read_parquet(report / "osm_temporal_audit.parquet")
    assert "value_fingerprint_changed" in json.loads(rows.iloc[0].issues)
    assert not (data / "locks/acceptance.json").exists()


def test_temporal_cli_routes_to_data_audit_without_training(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_temporal

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *args: None)
    monkeypatch.setattr(v5_temporal, "audit_osm_temporal", lambda *args: calls.append(args))
    argv = ["--stage", "target-temporal"]
    for key in ("source-root", "dataset-root", "report-root", "base-root"):
        argv += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(argv) == 0
    assert calls == [(tmp_path / "dataset-root", tmp_path / "report-root")]
