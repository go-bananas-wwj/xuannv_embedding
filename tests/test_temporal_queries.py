import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_temporal_fusion_training import fixture_spec, ref

from xuannv_embedding.export.context import dump
from xuannv_embedding.export.temporal_queries import export_queries, masked_resize
from xuannv_embedding.training import temporal_fusion


def setup(tmp_path):
    training = fixture_spec(tmp_path)
    spec = json.loads(training.read_text())
    spec["static"]["local"] = None
    dump(training, spec)
    temporal_fusion.train(training)
    annual = temporal_fusion.export(training)
    plan = tmp_path / "queries.json"
    dump(
        plan,
        {
            "schema": "temporal-public-queries-v1",
            "training_spec": ref(training),
            "checkpoint": ref(tmp_path / "run/model.weights.pt"),
            "annual_reference": ref(annual),
            "output_shape": [4, 4],
            "minimum_coverage": 0.05,
            "maximum_absolute_replay_error": 1e-6,
            "maximum_relative_replay_rmse": 1e-6,
            "device": "cpu",
            "reserve_gib": 0,
            "output": str(tmp_path / "queries"),
        },
    )
    return plan


def test_temporal_queries_resume_without_map_targets_and_match_full_export(tmp_path):
    plan = setup(tmp_path)
    # Inference must not depend on any map target files.
    (tmp_path / "weak.json").unlink()
    for p in tmp_path.glob("weak_*.npz"):
        p.unlink()
    assert export_queries(plan, max_tiles=1)["state"] == "checkpointed"
    result = export_queries(plan)
    assert result["state"] == "complete" and result["months"] == list(range(202401, 202413))
    assert result["annual_replay_max_absolute_error"] == 0
    x = np.load(result["queries"]["path"])
    assert x.shape == (2, 12, 64, 4, 4) and x.dtype == np.float32 and np.isfinite(x).all()
    assert export_queries(plan) == result
    (tmp_path / "queries/manifest.json").unlink()
    assert export_queries(plan) == result
    with Path(result["queries"]["path"]).open("ab") as f:
        f.write(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        export_queries(plan)


def test_masked_resize_uses_native_pixel_centers_and_ignores_unobserved_values():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4).expand(1, 2, 64, 4, 4).clone()
    mask = torch.ones(1, 2, 1, 4, 4)
    out, coverage = masked_resize(x, mask, (2, 2), minimum_coverage=0.05)
    assert torch.equal(out[0, 0, 0], torch.tensor([[2.5, 4.5], [10.5, 12.5]]))
    assert torch.equal(coverage, torch.ones_like(coverage))
    mask[..., 0, 0] = 0
    x[..., 0, 0] = float("nan")
    partial, fraction = masked_resize(x, mask, (2, 2), minimum_coverage=0.05)
    torch.testing.assert_close(partial[0, 0, 0, 0, 0], torch.tensor(10 / 3))
    assert fraction[0, 0, 0, 0, 0] == 0.75
    mask.zero_()
    x.fill_(float("nan"))
    out, coverage = masked_resize(x, mask, (2, 2), minimum_coverage=0.05)
    assert torch.count_nonzero(out) == torch.count_nonzero(coverage) == 0


def test_temporal_queries_reject_committed_prefix_tampering(tmp_path):
    plan = setup(tmp_path)
    export_queries(plan, max_tiles=1)
    p = tmp_path / "queries/payload.partial/monthly.npy"
    x = np.load(p, mmap_mode="r+")
    x[0, 0, 0, 0, 0] += 1
    x.flush()
    with pytest.raises(ValueError, match="prefix"):
        export_queries(plan)


def test_temporal_queries_reject_unknown_fields_and_local_teacher(tmp_path):
    plan = setup(tmp_path)
    s = json.loads(plan.read_text())
    dump(plan, {**s, "labels": "never-read"})
    with pytest.raises(ValueError, match="specification"):
        export_queries(plan)
    training = Path(s["training_spec"]["path"])
    t = json.loads(training.read_text())
    t["static"]["local"] = t["static"]["public"]
    dump(training, t)
    s["training_spec"] = ref(training)
    dump(plan, s)
    with pytest.raises(ValueError, match="public"):
        export_queries(plan)


def test_temporal_queries_reject_repeated_calendar_bins_before_teacher_loading(tmp_path):
    plan = setup(tmp_path)
    s = json.loads(plan.read_text())
    training = Path(s["training_spec"]["path"])
    t = json.loads(training.read_text())
    monthly = Path(t["monthly"]["path"])
    metadata = json.loads(monthly.read_text())
    metadata["months"] = [202401] * 12
    dump(monthly, metadata)
    t["monthly"] = ref(monthly)
    dump(training, t)
    s["training_spec"] = ref(training)
    dump(plan, s)
    with pytest.raises(ValueError, match="consecutive"):
        export_queries(plan)
