import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.export import rgb_guided_refinement as guided
from xuannv_embedding.export.context import sha


def bundle(tmp_path):
    root = tmp_path / "source"
    package = root / "anyup"
    package.mkdir(parents=True)
    (root / "LICENSE").write_text("CC-BY-4.0 test fixture")
    (package / "__init__.py").write_text("")
    (package / "model.py").write_text(
        "import torch\n"
        "class AnyUp(torch.nn.Module):\n"
        " def __init__(self,use_natten=False):\n"
        "  super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))\n"
        " def forward(self,image,features,output_size=None,q_chunk_size=None):\n"
        "  return features.roll(1,-1)*self.weight\n"
    )
    record = {
        "repository": "https://github.com/wimmerth/anyup",
        "commit": "a" * 40,
        "source_path": str(root),
        "license": "CC-BY-4.0",
        "files": {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()},
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(record))
    weights = tmp_path / "weights.pth"
    torch.save({"weight": torch.tensor(1.0)}, weights)
    return {"path": str(path), "sha256": sha(path)}, {"path": str(weights), "sha256": sha(weights)}


def spec(tmp_path):
    ref = {"path": str(tmp_path / "input"), "sha256": "a" * 64}
    return {
        "schema": "rgb-guided-refinement-v1",
        "index": ref,
        "features": {**ref, "layout": "nchw", "index_sha256": "a" * 64},
        "rgb_cache": ref,
        "upsampler": {
            "family": "anyup_multi_backbone",
            "source_bundle": ref,
            "checkpoint": ref,
            "use_natten": False,
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "RGB_divisor": 255,
        },
        "q_chunk_size": 256,
        "output": str(tmp_path / "out"),
        "reserve_gib": 0,
        "budgets": [{"name": "guided025", "strength": 0.25}],
    }


def test_rgb_reconstruction_preserves_subcell_order_and_masks_missing():
    image = torch.arange(3 * 10 * 15).reshape(1, 3, 10, 15).float() % 256
    mask = torch.ones(1, 1, 10, 15)
    mask[:, :, 0, 0] = 0
    packed = torch.nn.functional.pixel_unshuffle(image, 5)
    packed[:, 0, 0, 0] = float("nan")
    coverage = torch.nn.functional.pixel_unshuffle(mask, 5)
    norm = spec(Path("/tmp"))["normalization"]
    guide, confidence = guided.reconstruct_guide(packed, coverage, norm)
    expected = (image / 255 - torch.tensor(norm["mean"])[None, :, None, None]) / torch.tensor(
        norm["std"]
    )[None, :, None, None]
    expected[:, :, 0, 0] = 0
    torch.testing.assert_close(guide, expected)
    assert confidence.shape == (1, 1, 2, 3)
    assert confidence[0, 0, 0, 0] == pytest.approx(24 / 25)


@pytest.mark.parametrize("kind", ["mask_nan", "mask_range", "rgb_nan", "rgb_range", "shape"])
def test_invalid_observed_rgb_is_rejected(kind):
    rgb, mask = torch.ones(1, 75, 4, 4), torch.ones(1, 25, 4, 4)
    if kind == "mask_nan":
        mask[0, 0, 0, 0] = float("nan")
    elif kind == "mask_range":
        mask[0, 0, 0, 0] = 2
    elif kind == "rgb_nan":
        rgb[0, 0, 0, 0] = float("nan")
    elif kind == "rgb_range":
        rgb[0, 0, 0, 0] = 256
    else:
        mask = mask[:, :24]
    with pytest.raises(ValueError):
        guided.reconstruct_guide(rgb, mask, spec(Path("/tmp"))["normalization"])


def test_missing_rgb_exact_fallback_and_bounded_interpolation():
    base = torch.randn(2, 64, 4, 4)
    refined = base.roll(1, -1)
    coverage = torch.rand(2, 1, 4, 4)
    coverage[:, :, 0] = 0
    result = guided.blend(base, refined, coverage, 0.75)
    assert torch.equal(result[:, :, 0], base[:, :, 0])
    torch.testing.assert_close(result, base + 0.75 * coverage * (refined - base))
    assert torch.equal(guided.blend(base, refined, coverage, 0), base)
    assert (result >= base.amin((2, 3), keepdim=True) - 1e-6).all()
    assert (result <= base.amax((2, 3), keepdim=True) + 1e-6).all()
    with pytest.raises(ValueError):
        guided.blend(base, refined, coverage, float("nan"))
    with pytest.raises(ValueError):
        guided.blend(base, refined * float("nan"), coverage, 0.5)


def test_official_loader_is_frozen_strict_and_source_isolated(tmp_path):
    source, weights = bundle(tmp_path)
    model = guided.load_anyup(source, weights)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    x = torch.randn(1, 64, 4, 4)
    torch.testing.assert_close(model(torch.zeros(1, 3, 20, 20), x), x.roll(1, -1))
    torch.save({"other": torch.tensor(1.0)}, weights["path"])
    weights["sha256"] = sha(weights["path"])
    with pytest.raises(RuntimeError):
        guided.load_anyup(source, weights)


def test_source_mutation_or_unregistered_python_is_rejected(tmp_path):
    source, weights = bundle(tmp_path)
    record = json.loads(Path(source["path"]).read_text())
    root = Path(record["source_path"])
    (root / "anyup" / "extra.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="source"):
        guided.load_anyup(source, weights)
    (root / "anyup" / "extra.py").unlink()
    (root / "anyup" / "model.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="source"):
        guided.load_anyup(source, weights)


@pytest.mark.parametrize("change", ["unknown", "traversal", "strength", "norm", "natten", "bool"])
def test_specification_is_strict(tmp_path, change):
    value = copy.deepcopy(spec(tmp_path))
    assert guided.validate_specification(value) == value
    if change == "unknown":
        value["_base_"] = "x"
    elif change == "traversal":
        value["budgets"][0]["name"] = "../x"
    elif change == "strength":
        value["budgets"][0]["strength"] = 1.1
    elif change == "norm":
        value["normalization"]["std"][0] = 0
    elif change == "natten":
        value["upsampler"]["use_natten"] = True
    else:
        value["q_chunk_size"] = True
    with pytest.raises(ValueError):
        guided.validate_specification(value)


def test_duplicate_specification_keys_are_rejected(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError):
        guided.specification(path)


def export_fixture(tmp_path):
    value = spec(tmp_path)
    source, weights = bundle(tmp_path)
    value["upsampler"].update(source_bundle=source, checkpoint=weights)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [{"patch_id": "a"}, {"patch_id": "b"}]}))
    value["index"] = {"path": str(index), "sha256": sha(index)}

    def array(name, a):
        path = tmp_path / (name + ".npy")
        np.save(path, a.astype(np.float16))
        return {
            "path": str(path),
            "sha256": sha(path),
            "layout": "nchw",
            "index_sha256": sha(index),
        }

    rng = np.random.default_rng(41)
    value["features"] = array("base", rng.normal(size=(2, 64, 4, 4)))
    coverage = np.ones((2, 25, 4, 4))
    coverage[:, :, 0] = 0
    cache = {
        "state": "complete",
        "schema": "subpixel-rgb-cache-data-v1",
        "index": value["index"],
        "registered_tiles": 2,
        "output_shape": [4, 4],
        "factor": 5,
        "working_pixel_m": 2,
        "output_pixel_m": 10,
        "primary_or_weak_maps_loaded": False,
        "rgb": array("rgb", rng.uniform(0, 255, (2, 75, 4, 4))),
        "coverage": array("coverage", coverage),
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache))
    value["rgb_cache"] = {"path": str(path), "sha256": sha(path)}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value))
    return path, value


def test_export_preserves_coordinates_missing_rgb_and_rejects_mutated_receipt(tmp_path):
    path, value = export_fixture(tmp_path)
    result = guided.export(path, batch_size=3)
    assert len(result) == 1 and result[0]["state"] == "complete"
    root = Path(value["output"]) / "representations/guided025"
    base = np.load(value["features"]["path"]).astype(np.float32)
    expected = base + 0.25 * (np.roll(base, 1, axis=-1) - base)
    expected[:, :, 0] = base[:, :, 0]
    np.testing.assert_array_equal(np.load(root / "annual.npy"), expected.astype(np.float16))
    assert guided.export(path, batch_size=3) == result
    assert result[0]["missing_rgb_pixels"] == 8
    status = json.loads((root / "status.json").read_text())
    status["index_sha256"] = "0" * 64
    (root / "status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="Cached"):
        guided.export(path, batch_size=3)


def test_base_geographic_index_mismatch_is_rejected(tmp_path):
    _, value = export_fixture(tmp_path)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"records": [{"patch_id": "c"}, {"patch_id": "d"}]}))
    value["index"] = {"path": str(other), "sha256": sha(other)}
    with pytest.raises(ValueError):
        guided.SourceData(value)


@pytest.mark.parametrize(
    "key,value", [("factor", 4), ("working_pixel_m", 3), ("primary_or_weak_maps_loaded", True)]
)
def test_guide_grid_or_map_lineage_mismatch_is_rejected(tmp_path, key, value):
    _, definition = export_fixture(tmp_path)
    path = Path(definition["rgb_cache"]["path"])
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    definition["rgb_cache"]["sha256"] = sha(path)
    with pytest.raises(ValueError):
        guided.SourceData(definition)


def test_refinement_batch_size_is_bound_to_export_identity(tmp_path):
    path, _ = export_fixture(tmp_path)
    guided.export(path, batch_size=1)
    with pytest.raises(ValueError):
        guided.export(path, batch_size=2)
