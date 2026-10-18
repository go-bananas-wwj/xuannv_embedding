import json

import numpy as np
import pytest

from xuannv_embedding.export.context import dump
from xuannv_embedding.training.shared_fusion_search import (
    export_candidate,
    read_plan,
    reader_jobs,
    ref,
)


def fixture_inputs(tmp_path):
    rng = np.random.default_rng(5)
    records = [
        {"patch_id": f"p{i}", "bounds": [i * 20, 0, i * 20 + 20, 20], "crs": "EPSG:32650"}
        for i in range(22)
    ]
    index = tmp_path / "index.json"
    dump(index, {"crs": "EPSG:32650", "months": ["2024-01"], "records": records})
    region = {
        "name": "region_a",
        "index": ref(index),
        "task_source": "reference",
        "support_seed": 41,
        "evaluation": {"path": str(tmp_path / "must-not-read-labels"), "sha256": "0" * 64},
    }
    for name, channels in [("public", 64), ("semantic", 64), ("raw", 95)]:
        path = tmp_path / f"{name}.npy"
        values = rng.normal(size=(22, channels, 2, 2)).astype(np.float32)
        if name == "semantic":
            values[0] = 0
        np.save(path, values)
        region[name] = {**ref(path), "layout": "nchw"}
    plan = {"seed": 41, "fit_samples": 88, "output": str(tmp_path / "wave")}
    recipe = {
        "name": "aligned_a25",
        "kind": "anchored",
        "strength": 0.25,
        "kernel": 1,
        "ridge": 0.01,
    }
    return plan, region, recipe, records


def test_shared_export_never_opens_evaluation_maps_and_keeps_public_fallback(tmp_path):
    plan, region, recipe, _ = fixture_inputs(tmp_path)
    path = export_candidate(plan, region, recipe)
    result = np.load(path)
    original = np.load(region["public"]["path"])
    assert result.shape == (22, 64, 2, 2)
    np.testing.assert_array_equal(result[0], original[0])
    assert json.loads((path.parent / "status.json").read_text())["map_labels_used"] is False
    assert export_candidate(plan, region, recipe) == path


def test_shared_reader_registration_keeps_masks_supports_and_all_registered_tasks(tmp_path):
    plan, region, recipe, records = fixture_inputs(tmp_path)
    feature = export_candidate(plan, region, recipe)
    valid = tmp_path / "valid.npy"
    np.save(valid, np.ones((22, 2, 2), bool))
    labels = tmp_path / "labels.npz"
    target = np.zeros((22, 2, 2), np.int8)
    target[:, 0, 0] = 1
    np.savez_compressed(labels, target=target)
    data = {
        "state": "complete",
        "protocol": "full-region-data-v1",
        "records": records,
        "indices": list(range(22)),
        "train_positions": list(range(20)),
        "calibration_positions": [20, 21],
        "valid": ref(valid),
        "labels": ref(labels),
        "features": {},
        "size": 2,
        "crs": "EPSG:32650",
        "tasks": {"C": {"reference": ["target"]}, "R": {}, "Q": {}},
    }
    datafile = tmp_path / "data.json"
    dump(datafile, data)
    region["evaluation"] = ref(datafile)
    jobs = reader_jobs(plan, region, recipe, feature)
    assert len(jobs) == 2
    data_output = feature.parent.parent / "data.json"
    result = json.loads(data_output.read_text())
    assert result["valid"] == data["valid"] and result["labels"] == data["labels"]
    assert result["train_positions"] == data["train_positions"]
    assert result["calibration_positions"] == data["calibration_positions"]
    assert result["tasks"]["C"] == data["tasks"]["C"]
    assert data_output.exists()
    data["records"] = list(reversed(records))
    dump(datafile, data)
    region["evaluation"] = ref(datafile)
    with pytest.raises(ValueError, match="order"):
        reader_jobs(plan, region, recipe, feature)


def test_joint_projection_retains_64_dimensions_without_reference_access(tmp_path):
    plan, region, _, _ = fixture_inputs(tmp_path)
    recipe = {
        "name": "joint",
        "kind": "standardized_joint",
        "sources": ["public", "semantic", "raw"],
        "role": "candidate",
    }
    path = export_candidate(plan, region, recipe)
    values = np.load(path)
    assert values.shape == (22, 64, 2, 2)
    assert np.isfinite(values).all()
    assert json.loads((path.parent / "status.json").read_text())["map_labels_used"] is False
    assert export_candidate(plan, region, recipe) == path


def test_tail_export_never_opens_reference_maps(tmp_path):
    plan, region, _, _ = fixture_inputs(tmp_path)
    recipe = {"name": "tail16", "kind": "tail", "strength": 0.25, "semantic_dimensions": 16}
    path = export_candidate(plan, region, recipe)
    assert np.load(path).shape == (22, 64, 2, 2)
    state = json.loads((path.parent / "status.json").read_text())
    assert state["public_coordinates_retained"] == 64 and state["map_labels_used"] is False


def test_shared_plan_rejects_duplicated_regions_before_any_scoring(tmp_path):
    regions, anchor = tmp_path / "regions.json", tmp_path / "anchor.json"
    dump(regions, {"regions": [{"name": "a"}, {"name": "a"}, {"name": "b"}]})
    values = {r: {h: {"aef": 0.6, "public": 0.5} for h in ["mlp", "conv3x3"]} for r in ["a", "b"]}
    dump(anchor, {"state": "complete", "values": values, "selection_uses_validation_only": True})
    plan = tmp_path / "plan.json"
    dump(
        plan,
        {
            "schema": "shared-fusion-search-v1",
            "regions": ref(regions),
            "anchor": ref(anchor),
            "recipes": [
                {"name": "x", "kind": "anchored", "strength": 0.25, "kernel": 1, "ridge": 0.01}
            ],
            "output": str(tmp_path / "wave"),
            "seed": 41,
            "fit_samples": 128,
        },
    )
    with pytest.raises(ValueError, match="regions"):
        read_plan(plan)


def test_selection_never_promotes_public_only_control_and_prefers_eligible_candidates():
    from xuannv_embedding.training.shared_fusion_search import select_shared_winner

    def record(name, role, minimum, eligible):
        return {
            "recipe": {"name": name, "role": role},
            "minimum_gain": minimum,
            "mean_gain": minimum,
            "validation_eligible_for_query": eligible,
        }

    control = record("raw", "control", 0.2, True)
    insufficient = record("too_small_mean", "candidate", 0.004, False)
    accepted = record("balanced", "candidate", 0.003, True)
    assert select_shared_winner([control, insufficient, accepted])["recipe"]["name"] == "balanced"
    assert select_shared_winner([control, insufficient])["recipe"]["name"] == "too_small_mean"
