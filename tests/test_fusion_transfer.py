import json

import pytest

from xuannv_embedding.training.fusion_transfer import main, read_spec, verified


def test_fixed_transfer_rejects_unknown_configuration_keys(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema": "fixed-regional-fusion-transfer-v1", "unknown": True}))
    with pytest.raises(ValueError, match="specification"):
        read_spec(path)


def test_fixed_transfer_rejects_changed_source_identity(tmp_path):
    path = tmp_path / "source"
    path.write_text("input")
    with pytest.raises(ValueError, match="identity"):
        verified({"path": str(path), "sha256": "0" * 64})


def test_fixed_transfer_help_has_no_training_or_search_phase(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "{prepare,semantic,features,report}" in help_text
    assert "train" not in help_text
