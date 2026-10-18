import numpy as np
import pytest

from xuannv_embedding.export.public_statistics import summarize_public


def inputs():
    bands = {
        "s2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        "s1": ["vv", "vh"],
        "landsat": ["blue", "green", "red", "nir08", "swir16", "swir22"],
    }
    frames = {k: np.full((2, len(v), 2, 2), 0.1, np.float32) for k, v in bands.items()}
    valid = {k: np.ones((2, 2, 2), bool) for k in bands}
    frames["s2"][:, bands["s2"].index("B08")] = 0.6
    frames["s2"][:, bands["s2"].index("B04")] = 0.2
    frames["s2"][:, bands["s2"].index("B03")] = 0.3
    return frames, valid, bands


def test_public_statistics_uses_named_red_nir_green_not_adjacent_red_edge_bands():
    frames, valid, bands = inputs()
    result, names = summarize_public(frames, valid, bands)
    assert result.shape == (95, 2, 2)
    np.testing.assert_allclose(result[names.index("s2:ndvi:mean")], 0.5, atol=1e-6)
    np.testing.assert_allclose(result[names.index("s2:ndwi:mean")], -1 / 3, atol=1e-6)
    order = list(reversed(range(11)))
    frames["s2"] = frames["s2"][:, order]
    bands["s2"] = [bands["s2"][i] for i in order]
    permuted, names = summarize_public(frames, valid, bands)
    np.testing.assert_array_equal(result, permuted)


def test_public_statistics_keeps_missing_months_out_of_statistics():
    frames, valid, bands = inputs()
    frames["s2"][1] = 1000
    valid["s2"][1] = False
    result, names = summarize_public(frames, valid, bands)
    np.testing.assert_allclose(result[names.index("s2:B04:mean")], 0.2)
    np.testing.assert_allclose(result[names.index("s2:availability")], 0.5)
    valid["s2"][:] = False
    result, names = summarize_public(frames, valid, bands)
    assert np.isfinite(result).all()
    assert (result[[i for i, n in enumerate(names) if n.startswith("s2:")]] == 0).all()


def test_public_statistics_rejects_ambiguous_band_contract():
    frames, valid, bands = inputs()
    bands["s2"][0] = "B04"
    with pytest.raises(ValueError, match="band"):
        summarize_public(frames, valid, bands)


def test_offset_calibration_precedes_named_normalized_difference_indices():
    from xuannv_embedding.export.public_statistics import calibrate_public

    frames, valid, bands = inputs()
    frames["s2"] = frames["s2"] * 10000 + 1000
    radiometry = {
        s: {
            b: {"scale": 0.0001 if s == "s2" else 1, "offset": -0.1 if s == "s2" else 0}
            for b in names
        }
        for s, names in bands.items()
    }
    calibrated = calibrate_public(frames, bands, radiometry)
    result, names = summarize_public(calibrated, valid, bands)
    np.testing.assert_allclose(result[names.index("s2:ndvi:mean")], 0.5, atol=1e-6)


def test_negative_reflectance_is_not_a_valid_normalized_difference_observation():
    frames, valid, bands = inputs()
    frames["s2"][:, bands["s2"].index("B04")] = -0.2
    result, names = summarize_public(frames, valid, bands)
    assert (result[names.index("s2:ndvi:mean")] == 0).all()


def test_normalized_public_export_requires_no_highres_data_or_map_targets(tmp_path):
    import json

    import torch

    from xuannv_embedding.export.context import dump, sha
    from xuannv_embedding.export.public_statistics import export

    def ref(path):
        return {"path": str(path), "sha256": sha(path)}

    frames, masks, bands = inputs()
    sample = {
        "source_frames": {k: torch.from_numpy(v) for k, v in frames.items()},
        "target_masks": {k + "_recon": torch.from_numpy(v) for k, v in masks.items()},
    }
    sample_path = tmp_path / "sample.pt"
    torch.save(sample, sample_path)
    inputs_path = tmp_path / "inputs.json"
    dump(inputs_path, {"records": [{"input": ref(sample_path)}]})
    statistics = tmp_path / "statistics.json"
    dump(statistics, {s: {"mean": [0] * len(b), "std": [1] * len(b)} for s, b in bands.items()})
    spec = tmp_path / "spec.json"
    dump(
        spec,
        {
            "schema": "named-public-statistics-v2",
            "inputs": ref(inputs_path),
            "bands": bands,
            "statistics": ref(statistics),
            "kind": "normalized_tensor",
            "radiometry": {
                s: {b: {"scale": 1, "offset": 0} for b in names} for s, names in bands.items()
            },
            "output": str(tmp_path / "export"),
        },
    )
    export(spec)
    exported = np.load(tmp_path / "export/annual.npy")
    expected, _ = summarize_public(frames, masks, bands)
    np.testing.assert_array_equal(exported[0], expected)
    assert json.loads((tmp_path / "export/status.json").read_text())["map_labels_used"] is False
    export(spec)
