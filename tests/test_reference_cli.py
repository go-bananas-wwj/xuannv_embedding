from pathlib import Path

from xuannv_embedding.downstream import temporal_mean, worldcover_reference
from xuannv_embedding.training import experiment


def test_reference_commands_dispatch_without_shadowing_cache_preparation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(worldcover_reference, "prepare", lambda spec: calls.append(("wc", spec)))
    monkeypatch.setattr(temporal_mean, "export", lambda spec: calls.append(("mean", spec)))
    monkeypatch.setattr(experiment, "prepare", lambda *a, **kw: calls.append(("cache", a, kw)))
    spec = tmp_path / "spec.json"
    experiment.main(["prepare-worldcover", "--spec", str(spec)])
    experiment.main(["export-temporal-mean", "--spec", str(spec)])
    experiment.main(["prepare", "--config", str(spec), "--output", str(tmp_path / "cache")])
    assert calls[0] == ("wc", spec)
    assert calls[1] == ("mean", spec)
    assert calls[2][0] == "cache"
    assert calls[2][1][0] == Path(spec)
