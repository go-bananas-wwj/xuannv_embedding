from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xuannv_embedding.export.manuscript_controls import binary_coverage, pca_projection
from xuannv_embedding.export.manuscript_model import coverage_mask


def test_partial_coverage_control_preserves_missing_positions():
    c = torch.tensor([0.0, 0.25, 1.0])
    assert binary_coverage(c).tolist() == [0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        binary_coverage(torch.tensor([float("nan")]))


def test_missingness_masks_are_nested_and_content_independent():
    a, b = coverage_mask((128, 128), 0.25, 3), coverage_mask((128, 128), 0.5, 3)
    assert torch.all(a <= b) and a.mean() == 0.25 and b.mean() == 0.5
    assert coverage_mask((32, 32), 0.0, 3).sum() == 0


def test_simple_projection_is_fixed_width_finite_and_deterministic():
    x = np.random.default_rng(41).normal(size=(200, 164)).astype("float32")
    x[:, 2] = 0
    mean, scale, components = pca_projection(x)
    assert components.shape == (164, 64)
    assert np.isfinite(((x - mean) / scale) @ components).all()
    np.testing.assert_array_equal(components, pca_projection(x)[2])


@pytest.mark.parametrize("kind", ["guided", "binary_coverage"])
def test_retired_control_rebuilds_exactly_without_changing_frozen_identity(
    tmp_path, monkeypatch, kind
):
    from test_viewwise_guided_refinement import export_fixture

    from xuannv_embedding.export import manuscript_controls as controls
    from xuannv_embedding.export.context import dump, sha

    guide, settings = export_fixture(tmp_path)
    monkeypatch.setattr(controls.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**15))
    spec = {
        "schema": "manuscript-frozen-control-v1",
        "guide": {"path": str(guide), "sha256": sha(guide)},
        "features": settings["features"],
        "kind": kind,
        "output": str(tmp_path / "control"),
        "reserve_gib": 100,
    }
    path = tmp_path / "control.json"
    dump(path, spec)
    first = controls.export(path, "cpu")
    Path(first["features"]["path"]).unlink()
    second = controls.export(path, "cpu")
    assert first == second
