from collections import Counter
from types import SimpleNamespace

import pytest

from xuannv_embedding.training.annual_experiments import coverage_stratum, stratified_indices


def _evaluation_cache_worker(source, destination, barrier, writes, results):
    import time
    from pathlib import Path

    import xuannv_embedding.utils.manifest as manifests
    from xuannv_embedding.training.annual_experiments import evaluation_manifest

    original = manifests.write_manifest

    def slow_write(*args, **kwargs):
        with writes.get_lock():
            writes.value += 1
        time.sleep(0.2)
        return original(*args, **kwargs)

    manifests.write_manifest = slow_write
    barrier.wait(timeout=60)
    try:
        path = evaluation_manifest(Path(source), Path(destination), {"p1"})
        records = manifests.IndexedManifest(path)
        before = path.stat().st_mtime_ns
        time.sleep(0.1)
        ids = [row.patch_id for row in records]
        results.put({"ids": ids, "mtime": before, "after": path.stat().st_mtime_ns})
    except Exception as exc:
        results.put({"error": repr(exc)})


@pytest.mark.parametrize("preexisting_links", [False, True])
def test_evaluation_cache_serializes_concurrent_publishers_and_keeps_readers_stable(
    tmp_path, preexisting_links
):
    import multiprocessing

    from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

    source = tmp_path / "source" / "2021.train.manifest.jsonl"
    source.parent.mkdir()
    (source.parent / "statistics").mkdir()
    for name in ["summary.json", "sources.json"]:
        (source.parent / name).write_text("{}")
    write_manifest(
        source,
        [ManifestRecord(f"p{i}", "annual_2021", {"s2": None}) for i in range(2)],
        months=[f"2021-{month:02}" for month in range(1, 13)],
    )
    destination = tmp_path / "cache" / source.name
    if preexisting_links:
        destination.parent.mkdir()
        for name in ["summary.json", "sources.json", "statistics"]:
            (destination.parent / name).symlink_to(
                source.parent / name, target_is_directory=name == "statistics"
            )
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    writes = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_evaluation_cache_worker,
            args=(str(source), str(destination), barrier, writes, results),
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    try:
        rows = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        assert all(row.get("ids") == ["p1"] for row in rows), rows
        assert writes.value == 1
        assert len({row["mtime"] for row in rows}) == 1
        assert all(row["mtime"] == row["after"] for row in rows)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def test_completed_evaluation_cache_is_read_only_and_rejects_other_identities(tmp_path):
    from xuannv_embedding.training.annual_experiments import evaluation_manifest
    from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

    source = tmp_path / "source" / "2021.test.manifest.jsonl"
    source.parent.mkdir()
    (source.parent / "statistics").mkdir()
    for name in ["summary.json", "sources.json"]:
        (source.parent / name).write_text("{}")
    write_manifest(
        source,
        [ManifestRecord(f"p{i}", "annual_2021", {"s2": None}) for i in range(2)],
        months=["2021-01"],
    )
    destination = tmp_path / "cache" / source.name
    evaluation_manifest(source, destination, {"p1"})
    before = destination.read_bytes(), destination.stat().st_mtime_ns
    assert evaluation_manifest(source, destination, {"p1"}) == destination
    assert (destination.read_bytes(), destination.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="identity differs"):
        evaluation_manifest(source, destination, {"p0"})
    assert (destination.read_bytes(), destination.stat().st_mtime_ns) == before


def test_prepare_from_legacy_config_preserves_native_pan_and_resets_control_targets(tmp_path):
    import json
    from pathlib import Path

    import torch
    import yaml
    from test_annual_model import raw_batch

    from xuannv_embedding.config import Config
    from xuannv_embedding.training.annual import prepare_annual_batch
    from xuannv_embedding.training.annual_experiments import prepare
    from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

    data = tmp_path / "data"
    data.mkdir()
    (data / "statistics").mkdir()
    for name in ["summary.json", "sources.json"]:
        (data / name).write_text("{}")
    for split in ["train", "validation"]:
        write_manifest(
            data / f"2020.{split}.manifest.jsonl",
            [
                ManifestRecord(
                    patch_id=f"parent_32650:{i}:0",
                    region="national2020",
                    sources={"s2": []},
                    grid={"parent_key": f"32650:{i}:0"},
                    provenance={"year": 2020, "observations": {"s2": []}},
                )
                for i in range(2)
            ],
            months=[f"2020-{i:02d}" for i in range(1, 13)],
        )
    raw = yaml.safe_load(Path("configs/production/annual5m_100m_v1.yaml").read_text())
    raw["training"]["highres_visible_weight"] = 0.9
    raw["data"]["datasets"] = raw["data"]["datasets"][:1]
    raw["data"]["datasets"][0].update(
        manifest_path=str(data / "2020.train.manifest.jsonl"),
        patch_grid_path=str(data / "2020.train.manifest.jsonl"),
        statistics_dir=str(data / "statistics"),
    )
    raw["training"]["annual_feature_anchor_weight"] = 10.0
    raw["training"]["annual_feature_anchor_mode"] = "highpass_10m"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"identity-only fixture; prepare never loads weights")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "labels.json").write_text('{"records": []}')
    cities = tmp_path / "cities.json"
    cities.write_text("[]")
    output = tmp_path / "experiment"
    prepare(
        SimpleNamespace(
            config=config_path,
            initialize_from=checkpoint,
            teacher_config=config_path,
            teacher_checkpoint=checkpoint,
            labels=labels,
            cities=cities,
            osm=tmp_path / "osm",
            output=output,
            seed=42,
            epochs=8,
            samples_per_year=1,
            validation_samples=1,
            probe_steps=1,
            lr=3e-5,
            workers=0,
        )
    )
    plan = json.loads((output / "plan.json").read_text())
    assert plan["exact_mask_coverage_targets"]
    config = Config.from_yaml(Path(plan["arms"]["control"]["config"]))
    assert config.training.highres_visible_weight is None
    batch = prepare_annual_batch(raw_batch(128), config, training=False)
    for observation in batch["highres_observations"][0]:
        assert torch.all(observation["mask"] > 0)
    for arm in plan["arms"].values():
        prepared = Config.from_yaml(Path(arm["config"]))
        assert prepared.model.annual_exact_mask_coverage
        assert prepared.training.annual_feature_anchor_weight == 0


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
@pytest.mark.parametrize("deterministic", [False, True])
def test_fine_confirmation_never_trains_heads_on_holdout_labels(
    tmp_path, monkeypatch, threshold_free, deterministic
):
    import json

    import numpy as np
    import torch

    import xuannv_embedding.downstream.annual_ablation as evaluation
    from xuannv_embedding.training.annual_experiments import digest

    features = tmp_path / "original" / "city5m"
    features.mkdir(parents=True)
    (features.parent / "export.json").write_text(
        json.dumps({"checkpoint_sha256": "a" * 64, "evaluation_source_code_sha256": "b" * 64})
    )
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
        if deterministic:
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark
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
        deterministic=deterministic,
    )
    probe_dir = features.parent / ("city_probe_deterministic" if deterministic else "city_probe")
    metadata = json.loads((probe_dir / "metrics.json").read_text())
    train_rows = np.asarray(metadata["annual_splits"]) == "train"
    assert len(seen) == 6
    assert all(np.all(train_rows[index]) for index in seen)
    assert metadata["tasks"]["road:conv3x3"]["all"]["patches"] == 2
    assert metadata["confirmation_only"] and not metadata["development_only"]
    assert ("numerical_policy" in metadata) == deterministic
    if deterministic:
        assert metadata["feature_export_source_code_sha256"] == "b" * 64
        assert metadata["numerical_policy"]["deterministic_algorithms"]
    if threshold_free:
        metrics = metadata["tasks"]["road:conv3x3"]["all"]
        assert metrics["AP"] == pytest.approx(256 / 65536)
        assert metrics["ROC_AUC"] == pytest.approx(0.5)
    stats = np.load(probe_dir / "road_conv3x3.npy")
    assert stats[train_rows].sum() == 0 and stats[~train_rows].sum() > 0
    if deterministic:
        previous_metrics = (probe_dir / "metrics.json").read_bytes()
        with pytest.raises(ValueError, match="new result directory"):
            evaluation.city_probe(
                {"cities": str(cities), "osm": str(osm), "confirmation_only": True},
                tmp_path,
                "original",
                torch.device("cpu"),
                deterministic=True,
            )
        assert len(seen) == 6 and (probe_dir / "metrics.json").read_bytes() == previous_metrics


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
