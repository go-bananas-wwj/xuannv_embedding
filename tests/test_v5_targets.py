import numpy as np
import pytest

from xuannv_embedding.data_process.v5_targets import inspect_label_block


def test_valid_static_labels_reject_nodata_as_a_class():
    values = np.array([0, 10, 80], dtype="u1")
    assert (
        inspect_label_block(
            "static", "targets/worldcover_2020", values, np.array([False, True, True])
        )["valid_pixels"]
        == 2
    )
    with pytest.raises(ValueError, match="class"):
        inspect_label_block("static", "targets/worldcover_2020", values, np.ones(3, bool))


def test_nonfinite_invalid_pixels_are_separate_from_valid_errors():
    values = np.array([2.0, np.nan])
    result = inspect_label_block("static", "targets/dem_elevation", values, np.array([True, False]))
    assert result["nonfinite_pixels"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        inspect_label_block("static", "targets/dem_elevation", values, np.ones(2, bool))


def test_osm_unknown_is_preserved_and_overlay_cannot_overwrite_positive():
    states = np.array([0, 1, 2, 3], dtype="u1")
    result = inspect_label_block("osm", "2020/states/building", states)
    assert result["value_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    with pytest.raises(ValueError, match="overlay"):
        inspect_label_block(
            "reliable_negative", "2020/states/building", np.array([3]), base_states=np.array([1])
        )
    with pytest.raises(ValueError, match="state"):
        inspect_label_block("osm", "2020/states/building", np.array([4]))


def test_value_audit_reads_all_chunks_and_does_not_approve_provenance(tmp_path):
    import pandas as pd
    import zarr

    from xuannv_embedding.data_process.v5_targets import audit_target_values

    data = tmp_path / "data"
    report = tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    pd.DataFrame({"patch_id": ["a", "b"]}).to_parquet(data / "registry/national_62000.parquet")
    source = tmp_path / "labels.zarr"
    root = zarr.open_group(str(source), mode="w")
    root.create_dataset(
        "targets/clcd_2020", data=np.array([[[1]], [[2]]], dtype="u1"), chunks=(1, 1, 1)
    )
    root.create_dataset("valid_masks/clcd_2020", data=np.ones((2, 1, 1), dtype=bool))
    pd.DataFrame(
        [
            {
                "family": "static",
                "array": "targets/clcd_2020",
                "path": str(source),
                "registry_order_verified": True,
            }
        ]
    ).to_parquet(data / "targets/manifest.parquet")
    result = audit_target_values(data, report)
    assert result["processed_arrays"] == 1
    assert result["failed_arrays"] == 0
    root["targets/clcd_2020"][1] = 0
    repeated = audit_target_values(data, report)
    assert repeated["failed_arrays"] == 1  # Changed pixels invalidate the prior check.
    assert not (data / "locks/acceptance.json").exists()
