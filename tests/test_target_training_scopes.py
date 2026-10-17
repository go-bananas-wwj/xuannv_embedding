import copy
import json

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
