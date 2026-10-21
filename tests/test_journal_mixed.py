import numpy as np
import pytest

from xuannv_embedding.downstream.journal_mixed import footprint_fraction


def test_two_metre_reference_measures_composition_inside_ten_metre_cells():
    mask = np.zeros((10, 10), np.uint8)
    mask[:2, :2] = 1
    result = footprint_fraction(mask, 5)
    np.testing.assert_array_equal(result, np.array([[0.16, 0], [0, 0]], np.float32))


def test_fraction_reference_rejects_incomplete_cells():
    with pytest.raises(ValueError):
        footprint_fraction(np.ones((11, 10), np.uint8), 5)


def test_composition_readout_preserves_pixel_targets_and_excludes_support_for_query(tmp_path):
    import json

    from xuannv_embedding.downstream.journal_mixed import score_model
    from xuannv_embedding.export.context import dump
    from xuannv_embedding.training.shared_fusion_search import ref

    rng = np.random.default_rng(41)
    np.save(tmp_path / "features.npy", rng.normal(size=(23, 16, 16, 64)).astype("float32"))
    np.save(tmp_path / "targets.npy", np.full((23, 16, 16), 0.25, np.float32))
    data = {
        "records": [{"patch_id": str(i)} for i in range(23)],
        "indices": list(range(23)),
        "train_positions": list(range(20)),
        "calibration_positions": [20, 21],
    }
    dump(tmp_path / "data.json", data)
    spec = {
        "reference": ref(tmp_path / "data.json"),
        "output": str(tmp_path),
        "_spec_sha256": "a" * 64,
    }
    target = {"targets": ref(tmp_path / "targets.npy")}
    assert score_model(
        spec, "model", {"features": ref(tmp_path / "features.npy"), "layout": "nhwc"}, target
    )
    result = json.loads((tmp_path / "model/result.json").read_text())
    assert result["rows"][0]["metrics"]["all_region"]["mixed_cells"]["observations"] == 5888
    assert result["rows"][0]["metrics"]["all_region"]["mixed_cells"]["rmse"] == 0
    assert (
        result["rows"][0]["metrics"]["outside_support_and_calibration"]["all_cells"]["observations"]
        == 4096
    )
    assert (
        result["rows"][1]["metrics"]["outside_support_and_calibration"]["all_cells"]["observations"]
        == 256
    )
