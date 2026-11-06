from types import SimpleNamespace

import pandas as pd
import pytest

from xuannv_embedding.data_process.v5_sources import sha256, write_json


def snapshot(tmp_path):
    old = tmp_path / "audit"
    old.mkdir()
    source = dict(
        observation_id="n0",
        patch_id="p",
        sensor="JL",
        year=2020,
        split="train",
        path="native.tif",
        file_sha256="source",
    )
    qa = {
        **source,
        "scene_group_id": "s0",
        "band_ids": ["B1", "B4"],
        "quality_status": "model_inferred_needs_visual_review",
        "acquired_at": "2020-01-01",
    }
    prior_qa = tmp_path / "old" / "observation_quality.parquet"
    prior_qa.parent.mkdir()
    pd.DataFrame([qa]).to_parquet(prior_qa)
    config = tmp_path / "configuration.lock.json"
    write_json(config, {"algorithm": "fixed"})
    registry = tmp_path / "national_62000.parquet"
    registry.write_bytes(b"fixed registry")
    files = {str(p): sha256(p) for p in [prior_qa, config, registry]}
    pd.DataFrame([source]).to_parquet(old / "inputs.parquet")
    audit = {
        **source,
        "status": "over_limit",
        "pairs": "[]",
        "scope": "native bands only",
        "quality_mask_sha256": "mask",
    }
    pd.DataFrame([audit]).to_parquet(old / "observations.parquet")
    lock = {
        "inputs_sha256": sha256(old / "inputs.parquet"),
        "fingerprint": {"family": "jilin1"},
        "snapshot": {"quality_inputs_sha256": files},
    }
    write_json(old / "inputs.lock.json", lock)
    write_json(
        old / "output.lock.json",
        {
            "inputs_lock_sha256": sha256(old / "inputs.lock.json"),
            "observations_sha256": sha256(old / "observations.parquet"),
        },
    )
    root = tmp_path / "new"
    (root / "receipts").mkdir(parents=True)
    receipt = root / "receipts/s0.json"
    write_json(receipt, {"mask_arrays": {"n0/valid": "mask"}})
    current_qa = root / "observation_quality.parquet"
    pd.DataFrame([qa]).to_parquet(current_qa)
    reader = SimpleNamespace(
        files={str(p): sha256(p) for p in [current_qa, config, registry]},
        inventory=pd.DataFrame([source]),
        qa=SimpleNamespace(
            table=pd.DataFrame([qa]).set_index("observation_id"),
            root=root,
            receipts={"s0": sha256(receipt)},
        ),
    )
    return old, reader, receipt


def test_expanded_snapshot_inherits_native_failure_with_provenance(tmp_path):
    from xuannv_embedding.data_process.v5_highres_eligibility import read_audit

    old, reader, receipt = snapshot(tmp_path)
    rows, files = read_audit(old, "jilin1", reader)
    assert rows.status.tolist() == ["over_limit"]
    assert rows.inheritance_scope.tolist() == ["source_and_mask_unchanged_across_QA_snapshots"]
    assert rows.inherited_from.tolist() == [str(old)]
    assert files[str(receipt)] == sha256(receipt)
    assert str(tmp_path / "old/observation_quality.parquet") in files


@pytest.mark.parametrize(
    "field,value",
    [("file_sha256", "changed"), ("path", "other.tif"), ("split", "test"), ("year", 2021)],
)
def test_changed_native_source_is_not_silently_reinherited(tmp_path, field, value):
    from xuannv_embedding.data_process.v5_highres_eligibility import read_audit

    old, reader, _ = snapshot(tmp_path)
    reader.inventory.loc[0, field] = value
    with pytest.raises(ValueError, match="inheritance source"):
        read_audit(old, "jilin1", reader)


def test_changed_mask_is_rejected_even_with_valid_new_receipt(tmp_path):
    from xuannv_embedding.data_process.v5_highres_eligibility import read_audit

    old, reader, receipt = snapshot(tmp_path)
    write_json(receipt, {"mask_arrays": {"n0/valid": "different mask"}})
    reader.qa.receipts["s0"] = sha256(receipt)
    with pytest.raises(ValueError, match="inheritance mask"):
        read_audit(old, "jilin1", reader)


def test_receipt_and_old_source_seals_are_rechecked(tmp_path):
    from xuannv_embedding.data_process.v5_highres_eligibility import read_audit

    old, reader, receipt = snapshot(tmp_path)
    original = receipt.read_bytes()
    receipt.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inheritance receipt"):
        read_audit(old, "jilin1", reader)
    receipt.write_bytes(original)
    (tmp_path / "old/observation_quality.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="historical QA input"):
        read_audit(old, "jilin1", reader)


@pytest.mark.parametrize(
    "field,value",
    [("band_ids", ["B4", "B1"]), ("acquired_at", "2020-02-01"), ("quality_status", "qa_missing")],
)
def test_changed_quality_contract_cannot_inherit(tmp_path, field, value):
    from xuannv_embedding.data_process.v5_highres_eligibility import read_audit

    old, reader, _ = snapshot(tmp_path)
    reader.qa.table.at["n0", field] = value
    with pytest.raises(ValueError, match="inheritance QA identity"):
        read_audit(old, "jilin1", reader)


def test_configuration_change_and_missing_source_fail_closed(tmp_path):
    from xuannv_embedding.data_process.v5_highres_eligibility import read_audit

    old, reader, _ = snapshot(tmp_path)
    config = str(tmp_path / "configuration.lock.json")
    reader.files[config] = "different"
    with pytest.raises(ValueError, match="inheritance configuration"):
        read_audit(old, "jilin1", reader)
    reader.files[config] = sha256(tmp_path / "configuration.lock.json")
    reader.inventory = reader.inventory.iloc[:0]
    with pytest.raises(ValueError, match="inheritance source"):
        read_audit(old, "jilin1", reader)
