import json

import pytest

from xuannv_embedding.export.context import sha
from xuannv_embedding.training import temporal_fusion


def test_public_cost_profile_requires_the_registered_terminal_training_step(tmp_path, monkeypatch):
    from xuannv_embedding.export.journal_resources import public_profile

    model = tmp_path / "model"
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({"steps": 1200, "output": str(model)}))
    spec = tmp_path / "profile.json"
    spec.write_text(
        json.dumps(
            {
                "schema": "journal-public-cost-v1",
                "fit": {"path": str(fit), "sha256": sha(fit)},
                "output": str(tmp_path / "profile"),
            }
        )
    )

    def incomplete(path):
        model.mkdir()
        (model / "status.json").write_text(json.dumps({"state": "complete", "step": 1000}))

    monkeypatch.setattr(temporal_fusion, "train", incomplete)
    with pytest.raises(ValueError, match="terminal"):
        public_profile(spec)
    assert not (tmp_path / "profile/result.json").exists()


def test_native_cost_profile_does_not_time_a_partially_encoded_cache_as_fresh(tmp_path):
    from xuannv_embedding.export.journal_resources import native_profile

    case = tmp_path / "case.json"
    output = tmp_path / "native"
    (output / "encoded/detail").mkdir(parents=True)
    case.write_text(json.dumps({"output": str(output)}))
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "journal-native-cost-v1",
                "case": {"path": str(case), "sha256": sha(case)},
                "output": str(tmp_path / "cost"),
            }
        )
    )
    with pytest.raises(ValueError, match="unencoded"):
        native_profile(profile, "cpu")
