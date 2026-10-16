import hashlib

import pytest

from xuannv_embedding.export.standalone import verified_file


def test_bundle_verifies_content_and_rejects_external_reference(tmp_path):
    (tmp_path / "config.yaml").write_text("model: sample\n")
    reference = {
        "path": "config.yaml",
        "sha256": hashlib.sha256((tmp_path / "config.yaml").read_bytes()).hexdigest(),
    }
    assert verified_file(tmp_path, reference) == tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="inside"):
        verified_file(tmp_path, {**reference, "path": "../config.yaml"})
    (tmp_path / "config.yaml").write_text("altered")
    with pytest.raises(ValueError, match="identity"):
        verified_file(tmp_path, reference)
