import json
from pathlib import Path

import pytest


def test_candidate_reuses_frozen_training_sources_and_seals_results(tmp_path, monkeypatch):
    from test_v5_clear_training import prepared_qa

    from xuannv_embedding.data_process import v5_adaptive_calibration as module
    from xuannv_embedding.data_process.v5_clear_training import calibrate_training
    from xuannv_embedding.data_process.v5_sources import sha256

    data, qa, report = prepared_qa(tmp_path)
    original = calibrate_training(data, report, "gaofen", qa)
    baseline_root = Path(original["output"]).parent
    measure = module.evaluate_texture
    calls = []

    def check_frozen(*args, **kwargs):
        frozen = list(
            (data / "quality/alignment/adaptive_calibration/gaofen").glob("*/inputs.lock.json")
        )
        assert len(frozen) == 1
        calls.append(sha256(frozen[0]))
        return measure(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_texture", check_frozen)
    result = module.calibrate_adaptive(data, report, "gaofen", qa, baseline_root)
    assert result["status"] == "passed" and result["processed"] == 8
    assert len(calls) == 8 and len(set(calls)) == 1
    root = Path(result["output"])
    contents = json.loads((root / "calibration.json").read_text())
    assert not contents["pixel_fusion_authorized"]
    assert contents["negative_controls_rejected"] == 24
    files = {p: (sha256(p), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}
    monkeypatch.setattr(module, "evaluate_texture", lambda *a, **k: pytest.fail("cache reran"))
    replay = module.calibrate_adaptive(data, report, "gaofen", qa, baseline_root)
    assert replay["reused"]
    assert files == {p: (sha256(p), p.stat().st_mtime_ns) for p in files}
    path = root / "calibration.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="output seal"):
        module.calibrate_adaptive(data, report, "gaofen", qa, baseline_root)


def test_candidate_rejects_changed_frozen_input_during_measurement(tmp_path, monkeypatch):
    from test_v5_clear_training import prepared_qa

    from xuannv_embedding.data_process import v5_adaptive_calibration as module
    from xuannv_embedding.data_process.v5_clear_training import calibrate_training

    data, qa, report = prepared_qa(tmp_path)
    original = calibrate_training(data, report, "gaofen", qa)
    measure = module.evaluate_texture

    def corrupt(*args, **kwargs):
        result = measure(*args, **kwargs)
        for p in (data / "quality/alignment/adaptive_calibration/gaofen").glob(
            "*/inputs.lock.json"
        ):
            p.write_text("{}")
        return result

    monkeypatch.setattr(module, "evaluate_texture", corrupt)
    with pytest.raises(ValueError, match="frozen inputs"):
        module.calibrate_adaptive(data, report, "gaofen", qa, Path(original["output"]).parent)
    assert not list(
        (data / "quality/alignment/adaptive_calibration/gaofen").glob("*/output.lock.json")
    )


def test_adaptive_calibration_cli_requires_roots_and_family_lock(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_adaptive_calibration, v5_cli

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(v5_adaptive_calibration, "calibrate_adaptive", lambda *a: calls.append(a))
    args = ["--stage", "adaptive-alignment-calibration"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    for extra in [[], ["--sensor-family", "gaofen"], ["--quality-root", str(tmp_path / "qa")]]:
        args += extra
        with pytest.raises(SystemExit):
            v5_cli.main(args)
    args += ["--clear-calibration-root", str(tmp_path / "baseline")]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".adaptive-alignment-calibration.gaofen.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0
    assert calls[0][-1] == tmp_path / "baseline"
