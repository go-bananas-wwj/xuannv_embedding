import copy
import json

import numpy as np
import pytest

from xuannv_embedding.training.experiment_folds import derive_fold_cache


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "parent"
    root.mkdir()
    records = [
        {"index": i, "path": f"/samples/{i}.pt", "sha256": str(i), "bounds": [x, y, x + 1, y + 1]}
        for i, (x, y) in enumerate((x, y) for x in range(20) for y in range(10))
    ]
    split = {f"group{k}": list(range(k * 40, (k + 1) * 40)) for k in range(5)}
    split["pilot_train"] = [0]
    (root / "cache.json").write_text(json.dumps({"records": records, "split": split}))
    return root


def test_fold_reuses_samples_and_rotates_groups_with_spatial_buffer(cache, tmp_path):
    original = (cache / "cache.json").read_bytes()
    for fold in range(5):
        out = tmp_path / f"fold{fold}"
        derive_fold_cache(cache, out, fold)
        data = json.loads((out / "cache.json").read_text())
        parent = json.loads(original)
        assert data["records"] == parent["records"]
        split = data["split"]
        assert split["test"] == parent["split"][f"group{fold}"]
        assert split["validation"] == parent["split"][f"group{(fold - 1) % 5}"]
        assert "pilot_train" not in split
        parts = [set(split[k]) for k in ["train", "validation", "test", "buffer"]]
        assert set.union(*parts) == set(range(200))
        assert sum(map(len, parts)) == 200
        centers = np.array([r["bounds"][:2] for r in data["records"]])
        distance = np.abs(
            centers[split["train"]][:, None] - centers[split["test"] + split["validation"]]
        )
        assert distance.max(axis=-1).min() > 1
        assert (cache / "cache.json").read_bytes() == original
        with pytest.raises(FileExistsError):
            derive_fold_cache(cache, out, fold)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "nonfinite", "unequal", "index"])
def test_fold_rejects_invalid_geometry_or_group_membership(cache, tmp_path, problem):
    data = copy.deepcopy(json.loads((cache / "cache.json").read_text()))
    if problem == "duplicate":
        data["split"]["group1"].append(0)
    elif problem == "missing":
        data["split"]["group0"].pop()
    elif problem == "nonfinite":
        data["records"][0]["bounds"][0] = float("nan")
    elif problem == "unequal":
        data["records"][0]["bounds"][2] += 1
    else:
        data["records"][0]["index"] = 9
    (cache / "cache.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        derive_fold_cache(cache, tmp_path / "invalid", 1)
    assert not (tmp_path / "invalid").exists()


def test_fold_rejects_invalid_fold_number(cache, tmp_path):
    with pytest.raises(ValueError):
        derive_fold_cache(cache, tmp_path / "invalid", 5)
