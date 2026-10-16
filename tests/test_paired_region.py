import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.training.paired_region import (
    cache_data,
    prepare_static_targets,
    public_control_config,
    spatial_five_folds,
)


def test_cache_data_serializes_nested_configuration_paths():
    from xuannv_embedding.config import Config

    config = Config.from_yaml(Path("configs/examples/china_p10c_pilot.yaml"))
    restored = json.loads(json.dumps(cache_data(config)))
    assert isinstance(restored["datasets"][0]["manifest_path"], str)
    assert restored["months"] == config.data.months


def test_spatial_folds_cover_all_tiles_once_and_keep_blocks_together():
    bounds = [
        (x * 1280, y * 1280, (x + 1) * 1280, (y + 1) * 1280) for y in range(20) for x in range(16)
    ]
    folds = spatial_five_folds(bounds, seed=41)
    assert folds == spatial_five_folds(bounds, seed=41)
    assert sorted(i for f in folds for i in f["test"]) == list(range(320))
    for fold in folds:
        a, b, c = (set(fold[k]) for k in ["train", "validation", "test"])
        assert len(a) == 192 and len(b) == len(c) == 64
        assert not (a & b or a & c or b & c)
        for y in range(0, 20, 2):
            for x in range(0, 16, 2):
                block = {y * 16 + x, y * 16 + x + 1, (y + 1) * 16 + x, (y + 1) * 16 + x + 1}
                assert any(block <= group for group in [a, b, c])


def test_spatial_folds_reject_duplicates_and_nonfinite_bounds():
    with pytest.raises(ValueError):
        spatial_five_folds([(0, 0, 1280, 1280)] * 20)
    with pytest.raises(ValueError):
        spatial_five_folds([(0, 0, np.nan, 1280)] * 20)


def test_public_control_drops_transformer_without_changing_common_architecture():
    full = {
        "model": {
            "input_sources": {"s2": {"role": "temporal"}, "aerial": {"role": "highres"}},
            "target_heads": {"s2_recon": {"source": "s2"}, "worldcover": {"source": "worldcover"}},
            "highres_transformer": {"dim": 16},
            "stp": {"space_dim": 32},
        },
        "data": {
            "datasets": [
                {"source_map": {"s2": "s2", "aerial": "aerial", "worldcover": "worldcover"}}
            ]
        },
    }
    original = copy.deepcopy(full)
    public = public_control_config(full)
    assert full == original
    assert public["model"]["highres_transformer"] is None
    assert set(public["model"]["input_sources"]) == {"s2"}
    assert public["model"]["stp"] == full["model"]["stp"]
    assert public["model"]["target_heads"] == full["model"]["target_heads"]


def test_static_targets_keep_unknown_pixels_and_do_not_change_source_sample():
    label = torch.tensor([[[0, 1], [2, 11]]]).repeat(2, 1, 1)
    sample = {"targets": {"worldcover": label}, "target_masks": {"worldcover": (label > 0).float()}}
    prepared = prepare_static_targets(sample)
    assert "worldcover" in sample["targets"]
    assert "esa_worldcover" not in sample["targets"]
    assert torch.equal(prepared["targets"]["esa_worldcover"], label[0])
    assert torch.equal(prepared["targets"]["osm_landcover"], label)
    assert prepared["target_masks"]["esa_worldcover"][0, 0] == 0
    sample["targets"]["worldcover"][1, 0, 0] = 3
    with pytest.raises(ValueError, match="static"):
        prepare_static_targets(sample)
