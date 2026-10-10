import copy
import json

import numpy as np
import pytest
import torch

from xuannv_embedding.data.observation_ablation import ablate_inputs, mask_audit
from xuannv_embedding.downstream.multitask import _features, export_month_index


def batch_fixture():
    frames = torch.ones(1, 3, 1, 2, 2)
    return {
        "patch_ids": ["tile"],
        "timestamps": torch.tensor([[202512, 202601, 202602]]),
        "source_frames": {"optical": frames, "radar": frames.clone()},
        "source_masks": {"optical": torch.ones(1, 3), "radar": torch.ones(1, 3)},
        "highres_frames": {"detail": frames.clone(), "static": torch.ones(1, 1, 2, 2)},
        "highres_masks": {"detail": frames.clone(), "static": torch.ones(1, 1, 2, 2)},
        "targets": {"truth": frames},
    }


def test_source_ablation_zeros_all_months_without_mutating_original_or_targets():
    batch = batch_fixture()
    masked = ablate_inputs(batch, ["optical", "detail"])
    for group, source in [("source", "optical"), ("highres", "detail")]:
        assert not masked[group + "_frames"][source].count_nonzero()
        assert not masked[group + "_masks"][source].count_nonzero()
    assert batch["targets"]["truth"].eq(1).all()
    assert batch["source_masks"]["optical"].eq(1).all()
    assert masked["source_frames"]["radar"].eq(1).all()


def test_prefix_ablation_keeps_current_month_and_drops_future_and_undated_inputs():
    masked = ablate_inputs(batch_fixture(), [], last_month=1)
    for group in ("source", "highres"):
        for name, values in masked[group + "_frames"].items():
            if name == "static":
                assert not values.count_nonzero()
            else:
                assert values[:, :2].eq(1).all()
                assert not values[:, 2:].count_nonzero()
                assert not masked[group + "_masks"][name][:, 2:].count_nonzero()


def test_ablation_audit_records_availability_and_detects_changed_masks():
    batch = batch_fixture()
    original = mask_audit(batch)
    masked = mask_audit(ablate_inputs(batch, ["detail"]))
    assert original[0]["patch_id"] == "tile"
    assert original[0]["sha256"] != masked[0]["sha256"]
    assert masked[0]["availability"]["highres:detail"] == [0, 0, 0]
    assert original[0]["availability"]["highres:detail"] == [1, 1, 1]
    assert mask_audit(copy.deepcopy(batch)) == original


@pytest.mark.parametrize(
    "sources,month",
    [(["missing"], None), (["optical", "optical"], None), ([], 3), ([], -1), ([], True)],
)
def test_ablation_rejects_invalid_source_and_time_requests(sources, month):
    with pytest.raises(ValueError):
        ablate_inputs(batch_fixture(), sources, last_month=month)


def test_month_selector_requires_observed_month_and_explicit_array_provenance():
    manifest = {"months": ["2026-01", "2026-02"]}
    assert export_month_index({}, manifest) == 1
    assert export_month_index({"month_index": 0}, manifest) == 0
    with pytest.raises(ValueError, match="array"):
        export_month_index({"month_index": 0, "array": "somewhere"}, manifest)
    manifest["input_ablation"] = {"last_visible_month_index": 0}
    with pytest.raises(ValueError, match="future"):
        export_month_index({}, manifest)


def test_monthly_features_read_requested_month_instead_of_last(tmp_path):
    values = np.empty((2, 64, 128, 128), dtype=np.float32)
    values[0], values[1] = 0.25, 0.75
    path = tmp_path / "tile.npz"
    np.savez_compressed(path, embedding=values, timestamps=np.array([202601, 202602]))
    records = [{"patch_id": "tile", "bounds": [0, 0, 1, 1], "path": str(path)}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"months": ["2026-01", "2026-02"], "records": records}))
    features, identity = _features(
        {"manifest": str(manifest), "month_index": 0}, records, [0], tmp_path
    )
    assert np.all(features == 0.25)
    assert identity["month"] == "2026-01"
