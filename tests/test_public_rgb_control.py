import json
from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.data_process import public_rgb_control
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS


def ref(p):
    return {"path": str(p), "sha256": sha(p)}


def fixture_spec(tmp_path):
    p = tmp_path / "observations.npz"
    values = np.ones((2, 11, 2, 2), np.float32) * 8000
    values[1, 0] = 1100
    values[1, 1] = 1200
    values[1, 2] = 1300
    valid = np.ones((2, 2, 2), bool)
    valid[1, 0, 0] = False
    np.savez_compressed(p, s2=values, s2_valid=valid)
    record = {"patch_id": "tile", "bounds": [0, 0, 20, 20], "crs": "EPSG:32650"}
    inputs = tmp_path / "inputs.json"
    dump(inputs, {"records": [{**record, "input": ref(p)}]})
    index = tmp_path / "index.json"
    dump(index, {"records": [record]})
    stats = tmp_path / "stats.json"
    dump(stats, {"s2": {"mean": [0] * 12, "std": [10000] * 12}})
    original = tmp_path / "observation-spec.json"
    dump(
        original,
        {
            "schema": "named-public-statistics-v2",
            "kind": "physical_npz",
            "inputs": ref(inputs),
            "statistics": ref(stats),
            "bands": PUBLIC_BANDS,
            "radiometry": {
                "s2": {b: {"scale": 0.0001, "offset": -0.1} for b in PUBLIC_BANDS["s2"]}
            },
        },
    )
    spec = {
        "schema": "public-s2-rgb-control-v1",
        "observations": ref(original),
        "index": ref(index),
        "input_months": ["2024-01", "2024-02"],
        "months": ["2024-02"],
        "display_max_reflectance": 0.3,
        "encoder_resolution_m": 5,
        "output": str(tmp_path / "rgb"),
    }
    path = tmp_path / "spec.json"
    dump(path, spec)
    return path, spec


def test_public_rgb_uses_named_bands_physical_offsets_and_only_selected_observations(tmp_path):
    import rasterio

    path, spec = fixture_spec(tmp_path)
    manifest = public_rgb_control.prepare(path)
    with rasterio.open(Path(spec["output"]) / "000000.tif") as src:
        np.testing.assert_allclose(src.read()[:, 1, 1], [25.5, 17, 8.5], atol=1e-4)
        assert not src.dataset_mask()[0, 0]
    result = json.loads(Path(manifest).read_text())
    with rasterio.open(result["records"][0]["raster"]["path"]) as src:
        assert src.shape == (4, 4)
    assert public_rgb_control.prepare(path) == manifest


def test_public_rgb_rejects_evaluation_fields_and_unavailable_months(tmp_path):
    path, spec = fixture_spec(tmp_path)
    dump(path, {**spec, "evaluation": "must-not-be-read"})
    with pytest.raises(ValueError, match="specification"):
        public_rgb_control.prepare(path)
    dump(path, {**spec, "months": ["2024-03"]})
    with pytest.raises(ValueError, match="month"):
        public_rgb_control.prepare(path)


def test_public_rgb_tensor_path_needs_no_local_imagery_or_label_targets(tmp_path):
    import rasterio
    import torch

    path, spec = fixture_spec(tmp_path)
    original = json.loads(Path(spec["observations"]["path"]).read_text())
    input_path = Path(original["inputs"]["path"])
    records = json.loads(input_path.read_text())
    with np.load(records["records"][0]["input"]["path"]) as stored:
        values = stored["s2"].copy()
        valid = stored["s2_valid"].copy()
    tensor = tmp_path / "public_only.pt"
    frames = np.concatenate([values / 10000, np.full((2, 1, 2, 2), 999, np.float32)], 1)
    torch.save(
        {
            "source_frames": {"s2": torch.from_numpy(frames)},
            "target_masks": {"s2_recon": torch.from_numpy(valid.astype(np.float32))},
        },
        tensor,
    )
    records["records"][0]["input"] = ref(tensor)
    dump(input_path, records)
    original["kind"] = "normalized_tensor"
    original["inputs"] = ref(input_path)
    dump(spec["observations"]["path"], original)
    spec["observations"] = ref(spec["observations"]["path"])
    dump(path, spec)
    public_rgb_control.prepare(path)
    with rasterio.open(Path(spec["output"]) / "000000.tif") as source:
        np.testing.assert_allclose(source.read()[:, 1, 1], [25.5, 17, 8.5], atol=1e-4)
