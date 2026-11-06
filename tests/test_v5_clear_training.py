from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def candidate_rows():
    return pd.DataFrame(
        [
            {
                "observation_id": f"{sensor}-{year}-{i}",
                "patch_id": f"{sensor}-{i}",
                "sensor": sensor,
                "year": year,
                "split": "train" if i < 15 else "test",
                "reference_clear_fraction": 1.0 if i != 0 else 0.94,
                "path": f"source-{sensor}-{year}-{i}.tif",
                "file_sha256": "source-hash",
                "alignment_status": "unknown",
            }
            for sensor in ["sensor-a", "sensor-b"]
            for year in [2020, 2021]
            for i in range(20)
        ]
    )


def test_clear_training_selection_is_fixed_balanced_and_ignores_measured_alignment():
    from xuannv_embedding.data_process.v5_clear_training import select_training_rows

    rows = candidate_rows()
    selected = select_training_rows(rows)
    assert len(selected) == 16
    assert selected.groupby(["sensor", "year"]).size().eq(4).all()
    assert selected.split.eq("train").all() and selected.reference_clear_fraction.ge(0.95).all()
    assert not selected.duplicated(["sensor", "patch_id"]).any()
    modified = rows.sample(frac=1, random_state=19).copy()
    modified["alignment_status"] = "passed"
    assert (
        select_training_rows(modified).observation_id.tolist() == selected.observation_id.tolist()
    )
    assert "alignment_status" not in selected.columns


def test_clear_training_selection_never_replaces_missing_train_samples_with_test():
    from xuannv_embedding.data_process.v5_clear_training import select_training_rows

    rows = candidate_rows()
    rows.loc[(rows.sensor == "sensor-a") & (rows.year == 2021), "split"] = "test"
    with pytest.raises(ValueError, match="four distinct training positions"):
        select_training_rows(rows)
    rows = candidate_rows()
    rows.loc[0, "reference_clear_fraction"] = np.nan
    with pytest.raises(ValueError, match="quality fraction"):
        select_training_rows(rows)
    with pytest.raises(ValueError, match="duplicate"):
        select_training_rows(pd.concat([candidate_rows(), candidate_rows().iloc[:1]]))


def prepared_qa(tmp_path):
    import shutil

    import zarr
    from test_v5_clear_intraband import gaofen_qa

    from xuannv_embedding.data_process.v5_intraband import calibrate_family

    data, qa, rows = gaofen_qa(tmp_path, count=8)
    table = pd.read_parquet(qa / "observation_quality.parquet")
    table.loc[4:, "year"] = 2021
    table["clear_fraction"] = 1.0
    table["ms_valid_pixels"] = 160 * 160
    table.to_parquet(qa / "observation_quality.parquet", index=False)
    masks = zarr.open_group(str(qa / "valid_masks.zarr"), mode="a")
    classes = zarr.open_group(str(qa / "classes.zarr"), mode="a")
    classes["classes"][:] = 0
    for name in ["ms_valid_packed", "before_buffer_packed"]:
        masks[name][:] = masks["data_valid_packed"][:]
    target = data / "quality/cloud/gaofen"
    target.mkdir(parents=True)
    shutil.copy(qa / "observation_quality.parquet", target / "observation_quality.parquet")
    report = tmp_path / "report"
    assert calibrate_family(data, report, "gaofen", version="v5")["status"] == "passed"
    return data, qa, report


def test_clear_training_calibration_freezes_before_matching_and_reuses_verified_results(
    tmp_path, monkeypatch
):
    import json

    from xuannv_embedding.data_process import v5_clear_training as module
    from xuannv_embedding.data_process.v5_sources import sha256

    data, qa, report = prepared_qa(tmp_path)
    original = module.calibrate_texture
    seen = []

    def calibrated(*args, **kwargs):
        frozen = list((data / "quality/alignment/clear_training/gaofen").glob("*/inputs.parquet"))
        assert len(frozen) == 1
        seen.append(sha256(frozen[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "calibrate_texture", calibrated)
    first = module.calibrate_training(data, report, "gaofen", qa)
    assert first["status"] == "passed" and first["processed"] == 8
    assert len(seen) == 8 and len(set(seen)) == 1
    path = Path(first["output"])
    before, mtime = sha256(path), path.stat().st_mtime_ns
    monkeypatch.setattr(
        module, "calibrate_texture", lambda *a, **k: pytest.fail("cache reran matching")
    )
    second = module.calibrate_training(data, report, "gaofen", qa)
    assert second["reused"] and sha256(path) == before and path.stat().st_mtime_ns == mtime
    assert not json.loads(path.read_text())["pixel_fusion_authorized"]
    progress = json.loads((report / "clear_training_calibration_gaofen_progress.json").read_text())
    assert progress["execution_status"] == "finished"
    original_bytes = path.read_bytes()
    modified = json.loads(original_bytes)
    modified["results"][0]["known_shift_calibration"]["status"] = "failed"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="published clear training output"):
        module.calibrate_training(data, report, "gaofen", qa)
    path.write_bytes(original_bytes)
    table = pd.read_parquet(qa / "observation_quality.parquet")
    Path(table.iloc[0].ms_path).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source pixels changed"):
        module.calibrate_training(data, report, "gaofen", qa)


def test_clear_training_calibration_rejects_source_changes_during_measurement(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process import v5_clear_training as module

    data, qa, report = prepared_qa(tmp_path)
    source = Path(pd.read_parquet(qa / "observation_quality.parquet").iloc[0].ms_path)
    original = module.calibrate_texture
    changed = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if not changed:
            source.write_bytes(source.read_bytes() + b"changed")
            changed.append(True)
        return result

    monkeypatch.setattr(module, "calibrate_texture", mutate)
    with pytest.raises(ValueError, match="source file changed"):
        module.calibrate_training(data, report, "gaofen", qa)
    assert not list((data / "quality/alignment/clear_training/gaofen").glob("*/calibration.json"))


def test_clear_training_calibration_is_available_in_the_single_cli(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_clear_training, v5_cli

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(
        v5_clear_training, "calibrate_training", lambda *a, **k: calls.append((a, k))
    )
    args = ["--stage", "clear-training-calibration", "--alignment-version", "v5"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--sensor-family", "gaofen"]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--quality-root", str(tmp_path / "qa")]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".clear-training-calibration.gaofen.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0
    assert calls[0][1] == {"version": "v5"}
