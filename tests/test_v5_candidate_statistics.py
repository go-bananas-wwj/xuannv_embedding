import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def fixture(tmp_path, monkeypatch):
    from test_v5_highres_eligibility import inputs

    from xuannv_embedding.data_process import v5_highres_eligibility as eligibility
    from xuannv_embedding.data_process.v5_rasters import NativeRaster
    from xuannv_embedding.data_process.v5_sources import sha256

    registry, rows = inputs()
    registry = pd.DataFrame(
        [
            dict(patch_id=p, split=s, longitude=100, latitude=30)
            for p, s in [("p", "train"), ("v", "val"), ("t", "test")]
        ]
    )
    rows.at[4, "patch_id"], rows.at[4, "split"] = "v", "val"
    rows.at[5, "patch_id"], rows.at[5, "split"] = "t", "test"
    rows.at[0, "file_sha256"] = rows.at[1, "file_sha256"]
    rows.at[0, "quality_status"] = "qa_missing"
    rows.at[0, "valid_pixels_by_band"] = [0, 0]
    rows.at[1, "valid_pixels_by_band"] = [4, 4]
    rows.at[3, "valid_pixels_by_band"] = [4, 0]
    source = tmp_path / "qa.json"
    source.write_text("{}")
    files = {str(source): sha256(source)}
    audit = pd.DataFrame(
        [
            dict(
                observation_id="n2",
                patch_id="p",
                split="train",
                sensor="sensor",
                year=2020,
                status="over_limit",
                pairs="[]",
                scope="relative",
            )
        ]
    )
    inventory_columns = [
        "observation_id",
        "scene_group_id",
        "product_id",
        "patch_id",
        "sensor",
        "year",
        "split",
        "path",
        "file_sha256",
    ]
    calls = []

    class Reader:
        family = "jilin1"
        configuration = {}

        def __init__(self, *args):
            self.files = files.copy()
            self.registry = registry.set_index("patch_id")
            self.inventory = rows[inventory_columns].copy()

        def verify_unchanged(self):
            if sha256(source) != files[str(source)]:
                raise ValueError("source changed")

        def read(self, row):
            calls.append(row.observation_id)
            assert row.observation_id in ["b1", "b3"]
            values = np.arange(8, dtype="f4").reshape(2, 2, 2)
            valid = np.ones_like(values, dtype=bool)
            if row.observation_id == "b3":
                values += 10
                valid[1] = False
            return NativeRaster(values, valid, ("B1", "B2"), (1, 0, 0, 0, -1, 2), "EPSG:32650"), {
                "source": row.file_sha256,
                "mask": valid.tolist(),
            }

    monkeypatch.setattr(eligibility, "HighresQualityReader", Reader)
    monkeypatch.setattr(eligibility, "normalize_branches", lambda *a: rows.copy())
    monkeypatch.setattr(eligibility, "read_audit", lambda *a: (audit, {}))
    result = eligibility.run_highres_eligibility(
        tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa"
    )
    return Reader, Path(result["output"]), rows, calls


def test_candidate_selection_prefers_valid_alias_and_preserves_exclusion_reasons(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process.v5_candidate_statistics import CandidateSelection

    Reader, root, _, _ = fixture(tmp_path, monkeypatch)
    selection = CandidateSelection(Reader(), root)
    assert selection.rows.observation_id.tolist() == ["b1", "b3"]
    excluded = selection.excluded.set_index("observation_id")
    assert "qa_missing" in excluded.loc["b0", "reason"]
    assert "native_spectral_over_limit" in excluded.loc["b2", "reason"]
    assert excluded.loc[["b4", "b5"], "reason"].str.contains("non_training_split").all()
    selection.verify_unchanged()


def test_candidate_selection_rejects_other_qa_snapshot_and_cross_split_source(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process.v5_candidate_statistics import CandidateSelection

    Reader, root, rows, _ = fixture(tmp_path, monkeypatch)
    reader = Reader()
    reader.configuration = {"different": True}
    with pytest.raises(ValueError, match="QA snapshot"):
        CandidateSelection(reader, root)
    reader = Reader()
    reader.inventory.loc[reader.inventory.observation_id.eq("b4"), "file_sha256"] = "1"
    with pytest.raises(ValueError, match="cross-split"):
        CandidateSelection(reader, root)
    rows.at[1, "path"] = "changed.tif"
    with pytest.raises(ValueError, match="source rows"):
        CandidateSelection(Reader(), root)


def test_candidate_selection_rejects_modified_outputs_and_inputs_after_open(tmp_path, monkeypatch):
    from xuannv_embedding.data_process.v5_candidate_statistics import CandidateSelection

    Reader, root, _, _ = fixture(tmp_path, monkeypatch)
    selection = CandidateSelection(Reader(), root)
    (root / "branches.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="candidate evidence"):
        selection.verify_unchanged()
    with pytest.raises(ValueError, match="candidate evidence"):
        CandidateSelection(Reader(), root)


def test_candidate_statistics_uses_only_selected_valid_pixels_and_replays(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_highres_statistics as module
    from xuannv_embedding.data_process.v5_sources import sha256

    Reader, root, _, calls = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "HighresQualityReader", Reader)
    args = (tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa")
    result = module.run_highres_statistics(*args, eligibility_root=root, reservoir_size=32)
    assert result["processed"] == 2 and result["excluded_observations"] == 4
    assert result["scope"] == "frozen_eligible_candidate_view"
    output = Path(result["output"])
    stats = json.loads((output / "statistics.json").read_text())
    product = stats["products"]["jilin1_ms_5m/sensor"]["statistics"]
    assert product["count"] == [8, 4]
    assert product["mean"] == pytest.approx([6.5, 5.5])
    assert not stats["normalization_authorized"]
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    module.run_highres_statistics(*args, eligibility_root=root, reservoir_size=32)
    assert calls == ["b1", "b3", "b1", "b3"]
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}
    fingerprint = json.loads((output / "inputs.lock.json").read_text())["fingerprint"]
    assert fingerprint["eligibility_view"]["output_lock_sha256"] == sha256(
        root / "output.lock.json"
    )


def test_candidate_statistics_cli_passes_explicit_view(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_highres_statistics

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(
        v5_highres_statistics, "run_highres_statistics", lambda *a, **k: calls.append(k)
    )
    args = ["--stage", "highres-statistics", "--sensor-family", "jilin1"]
    for key in [
        "source-root",
        "dataset-root",
        "report-root",
        "base-root",
        "quality-root",
        "eligibility-root",
    ]:
        args.extend(["--" + key, str(tmp_path / key)])
    assert v5_cli.main(args) == 0
    assert calls[0]["eligibility_root"] == tmp_path / "eligibility-root"


def test_candidate_statistics_does_not_publish_if_view_changes_during_read(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_highres_statistics as module

    Reader, root, _, _ = fixture(tmp_path, monkeypatch)
    original = Reader.read

    def changed(self, row):
        frame = original(self, row)
        (root / "branches.parquet").write_bytes(b"changed during read")
        return frame

    monkeypatch.setattr(Reader, "read", changed)
    monkeypatch.setattr(module, "HighresQualityReader", Reader)
    with pytest.raises(ValueError, match="candidate evidence"):
        module.run_highres_statistics(
            tmp_path / "data",
            tmp_path / "report",
            "jilin1",
            tmp_path / "qa",
            eligibility_root=root,
        )
    assert not list((tmp_path / "data/statistics").rglob("output.lock.json"))
