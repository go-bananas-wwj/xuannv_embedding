import json

import pytest

from xuannv_embedding.data_process.retention import apply_manifest, file_identity


def manifest(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    payload = root / "old.npy"
    payload.write_bytes(b"derived array")
    source = root / "source.tif"
    source.write_bytes(b"original")
    data = {
        "protocol": "artifact-retention-v1",
        "allowed_roots": [str(root)],
        "protected": [str(source)],
        "requirements": [file_identity(source)],
        "files": [{**file_identity(payload), "role": "derived_features"}],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    return path, payload, source


def test_prune_removes_only_registered_derived_payload_and_preserves_source(tmp_path):
    plan, payload, source = manifest(tmp_path)
    result = apply_manifest(plan, tmp_path / "journal.jsonl")
    assert not payload.exists()
    assert source.read_bytes() == b"original"
    assert result["deleted_files"] == 1
    records = [json.loads(x) for x in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [r["state"] for r in records] == ["pending", "deleted"]


@pytest.mark.parametrize("mutation", ["replace", "protect", "outside", "role", "source"])
def test_prune_prevalidates_entire_manifest_before_any_delete(tmp_path, mutation):
    plan, payload, source = manifest(tmp_path)
    data = json.loads(plan.read_text())
    second = payload.with_name("second.npy")
    second.write_bytes(b"second")
    data["files"].insert(0, {**file_identity(second), "role": "derived_features"})
    if mutation == "replace":
        payload.rename(payload.with_suffix(".old"))
        payload.write_bytes(b"new")
    elif mutation == "protect":
        data["protected"].append(str(payload))
    elif mutation == "outside":
        outside = tmp_path / "unrelated.npy"
        outside.write_bytes(b"unrelated")
        data["files"][1] = {**file_identity(outside), "role": "derived_features"}
    elif mutation == "role":
        data["files"][1]["role"] = "original"
    else:
        source.unlink()
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        apply_manifest(plan, tmp_path / "journal.jsonl")
    assert second.exists()


def test_prune_rejects_symlink_ancestor_even_if_resolved_path_is_allowed(tmp_path):
    plan, payload, source = manifest(tmp_path)
    alias = payload.parent / "alias"
    alias.symlink_to(payload.parent, target_is_directory=True)
    data = json.loads(plan.read_text())
    data["files"] = [
        {**file_identity(payload), "path": str(alias / payload.name), "role": "derived_features"}
    ]
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        apply_manifest(plan, tmp_path / "journal.jsonl")
    assert payload.exists() and source.exists()


def test_hardlinked_payload_is_not_reported_as_reclaimed_when_another_link_survives(tmp_path):
    plan, payload, _ = manifest(tmp_path)
    survivor = payload.with_name("still-used.npy")
    survivor.hardlink_to(payload)
    data = json.loads(plan.read_text())
    data["files"] = [{**file_identity(payload), "role": "derived_features"}]
    plan.write_text(json.dumps(data))
    result = apply_manifest(plan, tmp_path / "journal.jsonl")
    assert survivor.exists() and result["reclaimed_allocated_bytes"] == 0
