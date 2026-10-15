from collections import Counter
from types import SimpleNamespace

import pytest

from xuannv_embedding.training.annual_experiments import coverage_stratum, stratified_indices


def test_common_corrected_targets_do_not_change_the_legacy_model_input_policy():
    from dataclasses import replace

    import torch
    from test_annual_model import annual_config, raw_batch

    from xuannv_embedding.training.annual_experiments import prepare_evaluation_batch

    legacy = annual_config()
    legacy = replace(legacy, training=replace(legacy.training, highres_visible_weight=0.0))
    exact = replace(legacy, model=replace(legacy.model, annual_exact_mask_coverage=True))
    raw = raw_batch(32)
    a = prepare_evaluation_batch(raw, legacy, 11, exact_targets=True)
    b = prepare_evaluation_batch(raw, exact, 11, exact_targets=True)
    for target_a, target_b in zip(a["highres_targets"][0], b["highres_targets"][0], strict=True):
        for part in ["heldout", "visible"]:
            assert torch.equal(
                target_a["partitions"][part]["mask"], target_b["partitions"][part]["mask"]
            )
            assert torch.equal(
                target_a["partitions"][part]["values"], target_b["partitions"][part]["values"]
            )
    assert any(
        not torch.equal(x["mask"], y["mask"])
        for x, y in zip(a["highres_observations"][0], b["highres_observations"][0], strict=True)
    )


def test_cluster_bootstrap_preserves_pairing_and_groups_adjacent_patches():
    import numpy as np

    from xuannv_embedding.downstream.annual_ablation import (
        folds_for,
        paired_group_bootstrap,
        spatial_group,
    )

    assert spatial_group("parent_32650:32:64") == spatial_group("parent_32650:33:65")
    assert spatial_group("parent_32650:32:64") != spatial_group("parent_32650:48:64")
    assert (
        folds_for(["a", "b", "a", "c", "d", "e"])[0] == folds_for(["a", "b", "a", "c", "d", "e"])[2]
    )
    a = np.arange(10, dtype=float).reshape(5, 2) + 1
    result = paired_group_bootstrap(
        a, a, ["a", "b", "c", "d", "e"], lambda x: x[0] / x.sum(), draws=100
    )
    assert result["difference"] == 0 and result["CI95"] == [0, 0]
    assert result["groups"] == 5
    sparse = paired_group_bootstrap(a[:1], a[:1], ["one-city"], lambda x: x[0] / x.sum(), draws=100)
    assert sparse["CI95"] is None and sparse["groups"] == 1


def test_confirmation_bootstrap_uses_frozen_two_degree_groups(tmp_path):
    import json

    from xuannv_embedding.downstream.annual_ablation import semantic_groups

    ids = ["parent_32650:0:0", "parent_32650:32:0"]
    path = tmp_path / "groups.json"
    path.write_text(json.dumps([["block_1", "test", patch_id, 0] for patch_id in ids]))
    assert len(set(semantic_groups({}, ids))) == 2
    plan = {"confirmation_only": True, "cities": str(path)}
    assert semantic_groups(plan, ids) == ["block_1", "block_1"]
    with pytest.raises(ValueError, match="do not cover"):
        semantic_groups(plan, ids + ["parent_32650:64:0"])


def test_five_meter_counts_measure_regions_and_one_pixel_boundary_tolerance():
    import torch

    from xuannv_embedding.downstream.annual_ablation import city_counts, city_metrics

    truth = torch.zeros(1, 16, 16, dtype=torch.uint8)
    truth[:, 4:12, 7:9] = 1
    exact = city_metrics(city_counts(truth.bool(), truth).sum(0))
    assert exact == {"F1": 1, "IoU": 1, "boundaryF1": 1}
    shifted = city_metrics(city_counts(truth.roll(1, -1).bool(), truth).sum(0))
    assert shifted["F1"] < 1 and shifted["boundaryF1"] == 1


def test_confirmation_probe_scores_only_annual_spatial_holdout(tmp_path, monkeypatch):
    import json

    import numpy as np
    import torch

    from xuannv_embedding.downstream.annual_ablation import semantic_probe
    from xuannv_embedding.training.annual_experiments import digest

    labels = tmp_path / "labels"
    features = tmp_path / "original" / "semantic"
    labels.mkdir()
    features.mkdir(parents=True)
    (tmp_path / "original" / "export.json").write_text(json.dumps({"checkpoint_sha256": "f" * 64}))
    records = []
    for index in range(10):
        patch_id = f"parent_32650:{index * 16}:0"
        path = labels / f"{index}.npz"
        target = np.array([[0, 1], [0, 1]], dtype=np.int64)
        np.savez(path, water=target, landcover=target)
        records.append({"patch_id": patch_id, "file": path.name, "sha256": digest(path)})
        np.savez(
            features / f"{patch_id}.npz",
            emb=np.ones((64, 2, 2), np.float32),
            valid=np.ones((2, 2)),
            split="train" if index < 8 else "test",
        )
    (labels / "labels.json").write_text(json.dumps({"records": records}))
    monkeypatch.setattr(torch.optim.LBFGS, "step", lambda self, closure: closure())
    semantic_probe(
        {"labels": str(labels), "confirmation_only": True},
        tmp_path,
        "original",
        torch.device("cpu"),
    )
    result = json.loads((tmp_path / "original" / "semantic_probe" / "metrics.json").read_text())
    assert result["confirmation_only"] and not result["development_only"]
    confusion = np.load(tmp_path / "original" / "semantic_probe" / "water.npy")
    train = np.asarray(result["annual_splits"]) == "train"
    assert confusion[train].sum() == 0
    assert confusion[~train].sum() == 8
    assert "pretraining_seen" not in result["tasks"]["water"]


@pytest.mark.parametrize("guard_original", [False, True, "control_best"])
def test_selection_rejects_semantic_regression_despite_fine_detail_gain(tmp_path, guard_original):
    import json

    from xuannv_embedding.downstream.annual_ablation import SELECTION_PROTOCOL, select_candidate

    (tmp_path / "selection_protocol.json").write_text(json.dumps({"protocol": SELECTION_PROTOCOL}))
    root = tmp_path / "evaluation"
    arms = {
        "control": {},
        "A_context": {"context": True},
        "AB_visible01": {"context": True, "visible": 0.1},
    }
    for tag, water, landcover, gain in [
        ("control", 0.85, 0.43, 0),
        ("A_context", 0.80, 0.45, 0.1),
        ("AB_visible01", 0.86, 0.44, 0.01),
        ("original", 0.89, 0.47, 0.02),
    ]:
        semantic = {
            "tasks": {"water": {"all": {"F1": water}}, "landcover": {"all": {"mIoU": landcover}}}
        }
        city = {
            "tasks": {
                "road:conv3x3": {"all": {"F1": 0.23 + gain, "boundaryF1": 0.4 + gain}},
                "building:conv3x3": {"all": {"F1": 0.25 + gain}},
                "water:conv3x3": {"all": {"F1": 0.7 + gain}},
            }
        }
        for suite, value in [("semantic_probe", semantic), ("city_probe", city)]:
            destination = root / tag / suite
            destination.mkdir(parents=True)
            (destination / "metrics.json").write_text(json.dumps(value))
    plan = {"arms": {tag: {"changes": changes} for tag, changes in arms.items()}}
    if guard_original:
        plan["product_reference"] = "original"
    if guard_original == "control_best":
        p = root / "control" / "semantic_probe" / "metrics.json"
        value = json.loads(p.read_text())
        value["tasks"]["water"]["all"]["F1"] = 0.95
        value["tasks"]["landcover"]["all"]["mIoU"] = 0.5
        p.write_text(json.dumps(value))
        p = root / "control" / "city_probe" / "metrics.json"
        value = json.loads(p.read_text())
        for row in value["tasks"].values():
            row["all"] = {key: number + 0.05 for key, number in row["all"].items()}
        p.write_text(json.dumps(value))
    selected = select_candidate(plan, root)
    assert not selected["scores"]["A_context"]["eligible"]
    expected = (
        "control"
        if guard_original == "control_best"
        else None if guard_original else "AB_visible01"
    )
    assert selected["selected"] == expected
    with pytest.raises(ValueError, match="already selected"):
        select_candidate(plan, root)


@pytest.mark.parametrize("threshold_free", [False, True])
def test_fine_confirmation_never_trains_heads_on_holdout_labels(
    tmp_path, monkeypatch, threshold_free
):
    import json

    import numpy as np
    import torch

    import xuannv_embedding.downstream.annual_ablation as evaluation
    from xuannv_embedding.training.annual_experiments import digest

    features = tmp_path / "original" / "city5m"
    features.mkdir(parents=True)
    (features.parent / "export.json").write_text(json.dumps({"checkpoint_sha256": "a" * 64}))
    osm = tmp_path / "osm"
    osm.mkdir()
    cache = tmp_path / "osm_labels"
    cache.mkdir()
    rows = []
    for index in range(10):
        patch_id = f"parent_32650:{index * 16}:0"
        split = "train" if index < 8 else "test"
        rows.append([str(index), split, patch_id, 0])
        source = osm / f"{patch_id}.json"
        source.write_text("{}")
        target = np.zeros((256, 256), dtype=np.uint8)
        target[100:116, 100:116] = 1
        np.savez(
            cache / f"{patch_id}.npz",
            n_building=20,
            source_sha256=digest(source),
            building=target,
            road=target,
            water=target,
        )
        np.savez(
            features / f"{patch_id}.npz",
            emb=np.zeros((64, 256, 256), np.float16),
            valid=np.ones((256, 256), bool),
            split=split,
            highres_obs=0,
        )
    cities = tmp_path / "cities.json"
    cities.write_text(json.dumps(rows))
    seen = []

    def fake_train(features, labels, train, kind, fold, steps):
        seen.append(train.tolist())
        head = torch.nn.Conv2d(64, 1, 1)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.fill_(1)
        return head, torch.zeros(1, 64, 1, 1), torch.ones(1, 64, 1, 1)

    monkeypatch.setattr(evaluation, "train_city_head", fake_train)
    evaluation.city_probe(
        {
            "cities": str(cities),
            "osm": str(osm),
            "probe_steps": 1,
            "confirmation_only": True,
            "probe_threshold_free": threshold_free,
        },
        tmp_path,
        "original",
        torch.device("cpu"),
    )
    metadata = json.loads((features.parent / "city_probe" / "metrics.json").read_text())
    train_rows = np.asarray(metadata["annual_splits"]) == "train"
    assert len(seen) == 6
    assert all(np.all(train_rows[index]) for index in seen)
    assert metadata["tasks"]["road:conv3x3"]["all"]["patches"] == 2
    assert metadata["confirmation_only"] and not metadata["development_only"]
    if threshold_free:
        metrics = metadata["tasks"]["road:conv3x3"]["all"]
        assert metrics["AP"] == pytest.approx(256 / 65536)
        assert metrics["ROC_AUC"] == pytest.approx(0.5)
    stats = np.load(features.parent / "city_probe" / "road_conv3x3.npy")
    assert stats[train_rows].sum() == 0 and stats[~train_rows].sum() > 0


def records():
    return [
        SimpleNamespace(
            patch_id=f"p{index}",
            grid={"parent_key": "32650:0:0"},
            provenance={
                "observations": {
                    "s2": [1],
                    "GF6_PAN_c1": [1] if index < 20 else [],
                    "optical": [1] if index < 10 else [],
                }
            },
        )
        for index in range(100)
    ]


def test_hash_sampling_preserves_coverage_proportions_and_excludes_development_patches():
    source = records()
    selected = stratified_indices(source, 40, "seed")
    assert len(set(selected)) == 40
    assert Counter(coverage_stratum(source[index]) for index in selected) == {
        "32650:ms1:pan1": 4,
        "32650:ms0:pan1": 4,
        "32650:ms0:pan0": 32,
    }
    assert selected == stratified_indices(source, 40, "seed")
    assert selected != stratified_indices(source, 40, "different-seed")
    excluded = {source[index].patch_id for index in selected}
    alternative = stratified_indices(source, 40, "seed", excluded)
    assert not set(alternative).intersection(selected)
    with pytest.raises(ValueError, match="eligible"):
        stratified_indices(source, 61, "seed", excluded)
