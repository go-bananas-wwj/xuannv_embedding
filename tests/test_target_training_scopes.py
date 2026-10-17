import copy
import json
from pathlib import Path

import pytest
from test_annual_targets import target_bundle  # noqa: F401
from test_multisource_targets import local_bundle  # noqa: F401

from xuannv_embedding.data.target_contract import TARGET_TASKS, training_positions
from xuannv_embedding.data.targets import AnnualTargetReader
from xuannv_embedding.data_process.local_targets import prepare_local_targets
from xuannv_embedding.data_process.target_audit import digest


@pytest.mark.parametrize("target_bundle", [3], indirect=True)
@pytest.mark.parametrize("scope", ["expanded", "allowed"])
def test_larger_caches_cover_new_training_positions_without_exposing_evaluation(
    local_bundle, scope  # noqa: F811
):
    root, screen, screen_sha, index = local_bundle
    parent = screen.parent / "prepared"
    out = screen.with_name("larger_" + scope)
    prepare_local_targets(parent, root, out, workers=1, scope=scope)
    manifest = json.loads((out / "local_targets/manifest.json").read_text())
    assert manifest["scope"] == scope and manifest["positions"] == 2
    assert digest(screen / "preparation.json") == screen_sha
    with AnnualTargetReader(
        root, out, preparation_sha256=digest(out / "preparation.json"), scope=scope
    ) as reader:
        assert reader.statistics["scope"] == scope
        assert reader.statistics["unique_positions"] == 2
        assert not reader.preparation["formal_training_released"]
        assert reader.roles["train"] == {row["patch_id"] for row in index[:2]}
        for row in index[:2]:
            for year in [2020, 2021]:
                assert set(reader.read(row["patch_id"], year, tasks=TARGET_TASKS)) == set(
                    TARGET_TASKS
                )
        with pytest.raises(ValueError, match="outside"):
            reader.read(index[-1]["patch_id"], 2020)
        with pytest.raises(ValueError, match="sealed"):
            reader.read(index[-1]["patch_id"], 2020, role="development")
    with pytest.raises(ValueError, match="scope"):
        AnnualTargetReader(
            root, out, preparation_sha256=digest(out / "preparation.json"), scope="screen"
        )


def test_every_scope_rejects_duplicate_positions_and_evaluation_overlap():
    split = {
        "roles": {"development": [{"patch_id": "held"}]},
        "screen_patch_ids": ["train"],
        "expanded_patch_ids": ["train", "extra"],
        "allowed_train_patch_ids": ["train", "extra"],
    }
    assert training_positions(split, "allowed") == ["train", "extra"]
    for key, scope in [
        ("screen_patch_ids", "screen"),
        ("expanded_patch_ids", "expanded"),
        ("allowed_train_patch_ids", "allowed"),
    ]:
        for bad in [["train", "held"], ["train", "train"]]:
            changed = copy.deepcopy(split)
            changed[key] = bad
            with pytest.raises(ValueError):
                training_positions(changed, scope)


@pytest.mark.parametrize("scope,count", [("screen", 1), ("expanded", 2), ("allowed", 3)])
def test_training_pool_uses_exact_frozen_positions_and_paired_years(tmp_path, scope, count):
    import yaml
    from test_annual_model import annual_config

    from xuannv_embedding.data_process.target_audit import write_json
    from xuannv_embedding.training.target_experiments import prepare_training_pool
    from xuannv_embedding.utils.manifest import IndexedManifest, ManifestRecord, write_manifest

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    ids = [f"parent_32643:{100+i}:3200" for i in range(4)]
    split = {
        "screen_patch_ids": ids[:1],
        "expanded_patch_ids": ids[:2],
        "allowed_train_patch_ids": ids[:3],
        "roles": {"development": [{"patch_id": ids[3]}]},
    }
    write_json(prepared / "experiment_splits.json", split)
    write_json(
        prepared / "preparation.json",
        {
            "splits_sha256": digest(prepared / "experiment_splits.json"),
            "formal_training_released": False,
        },
    )
    raw = annual_config().to_dict()
    template = copy.deepcopy(raw["data"]["datasets"][0])
    raw["data"]["datasets"] = []
    bindings = []
    for year in [2020, 2021]:
        folder = tmp_path / str(year)
        folder.mkdir()
        (folder / "statistics").mkdir()
        for name in ["summary.json", "sources.json"]:
            (folder / name).write_text("{}")
        path = folder / f"{year}.train.manifest.jsonl"
        receipt = write_manifest(
            path,
            [
                ManifestRecord(
                    patch_id=p,
                    region=f"national{year}",
                    sources={"s2": None},
                    provenance={"year": year},
                )
                for p in ids
            ],
            months=[f"{year}-{m:02}" for m in range(1, 13)],
            generator_version="fixture",
        )
        bindings.append({"path": str(path), "sha256": receipt.sha256})
        dataset = copy.deepcopy(template)
        dataset.update(
            region=f"national{year}",
            manifest_path=str(path),
            patch_grid_path=str(path),
            statistics_dir=str(folder / "statistics"),
            source_map={name: name for name in raw["model"]["input_sources"]},
        )
        raw["data"]["datasets"].append(dataset)
    initial = tmp_path / "original.yaml"
    initial.write_text(yaml.safe_dump(raw))
    write_json(
        prepared / "pretraining_split_binding.json",
        {"config_sha256": digest(initial), "original_pretraining_manifests": bindings},
    )
    result = prepare_training_pool(prepared, initial, tmp_path / "pool", scope=scope)
    assert not result["training_started"] and not result["formal_training_released"]
    assert result["unique_positions"] == count
    for row in result["manifests"]:
        assert {r.patch_id for r in IndexedManifest(Path(row["path"]))} == set(ids[:count])
        assert row["records"] == count
