import json

import numpy as np
import pytest

from xuannv_embedding.downstream.multitask_features import FeatureSelection, read_features
from xuannv_embedding.export.context import sha


def setup_exports(tmp_path, *, kind="monthly", channels=3):
    records = [{"patch_id": f"p{i}", "bounds": [i * 20, 0, i * 20 + 20, 20]} for i in range(4)]
    cache = {
        "data": {"patch_size": 2, "months": ["2026-04", "2026-05"]},
        "split": {"train": [0], "validation": [1], "test": [2], "buffer": [3]},
        "records": records,
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache))
    months = ["annual_2025"] if kind == "annual" else cache["data"]["months"]
    if kind == "raw":
        months = ["2026-05"]
    exported, digests = [], {}
    for i, record in enumerate(records):
        path = tmp_path / f"p{i}.npz"
        exported.append({**record, "path": str(path)})
        # The reader must not even open test/buffer feature files by default.
        if i >= 2:
            continue
        array = np.stack(
            [np.full((channels, 2, 2), i + j + 0.25, np.float32) for j in range(len(months))]
        )
        data = {"embedding": array}
        if kind == "monthly":
            data["timestamps"] = np.array([202604, 202605])
        np.savez(path, **data)
        digests[record["patch_id"]] = sha(path)
    manifest = {
        "months": months,
        "cache_sha256": sha(cache_path),
        "split": cache["split"],
        "records": exported,
    }
    if kind == "raw":
        manifest["kind"] = "raw"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    selection = FeatureSelection(
        kind=kind,
        period="2025" if kind == "annual" else "2026-05",
        evaluation_month="2026-05",
        channels=channels,
    )
    return dict(
        manifest_path=path,
        cache_path=cache_path,
        manifest_sha256=sha(path),
        cache_sha256=sha(cache_path),
        tile_sha256=digests,
        selection=selection,
    )


@pytest.mark.parametrize("kind,channels", [("monthly", 3), ("annual", 64), ("raw", 9)])
def test_preserves_dimensions_periods_and_reads_only_requested_split(tmp_path, kind, channels):
    args = setup_exports(tmp_path, kind=kind, channels=channels)
    result = read_features(**args)
    assert result.values.shape == (2, 2, 2, channels)
    assert result.indices == (0, 1)
    assert result.valid.all()
    expected = 1.25 if kind == "monthly" else 0.25
    np.testing.assert_array_equal(result.values[0], np.full((2, 2, channels), expected))
    assert result.identity["feature_period"] == args["selection"].period
    assert result.identity["evaluation_month"] == "2026-05"
    assert result.identity["temporal_resolution"] == ("annual" if kind == "annual" else "monthly")
    assert result.identity["test_records_read"] is False


def rewrite_manifest(args, edit):
    path = args["manifest_path"]
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    args["manifest_sha256"] = sha(path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["records"][1].update(bounds=[99, 0, 119, 20]),
        lambda d: d["records"][1].update(patch_id="p0"),
        lambda d: d["split"].update(validation=[2], test=[1]),
        lambda d: d.update(cache_sha256="0" * 64),
        lambda d: d.update(months=["2026-05", "2026-04"]),
        lambda d: d.update(input_ablation={"last_visible_month_index": 0}),
    ],
)
def test_rejects_grid_split_period_or_future_cutoff_mismatch(tmp_path, edit):
    args = setup_exports(tmp_path)
    rewrite_manifest(args, edit)
    with pytest.raises(ValueError):
        read_features(**args)


def test_manifest_or_tile_changes_cannot_silently_replace_registered_features(tmp_path):
    args = setup_exports(tmp_path)
    path = args["manifest_path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest"):
        read_features(**args)
    args["manifest_sha256"] = sha(path)
    tile = tmp_path / "p0.npz"
    tile.write_bytes(tile.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest"):
        read_features(**args)


@pytest.mark.parametrize("corruption", ["timestamp", "channels", "nonfinite"])
def test_rejects_rehashed_but_invalid_tile_arrays(tmp_path, corruption):
    args = setup_exports(tmp_path)
    path = tmp_path / "p0.npz"
    with np.load(path) as source:
        data = {k: source[k] for k in source.files}
    if corruption == "timestamp":
        data["timestamps"][1] = 202604
    elif corruption == "channels":
        data["embedding"] = data["embedding"][:, :1]
    else:
        data["embedding"][1, 0, 0, 0] = np.nan
    np.savez(path, **data)
    args["tile_sha256"]["p0"] = sha(path)
    with pytest.raises(ValueError):
        read_features(**args)


def test_annual_product_cannot_be_relabelled_as_monthly(tmp_path):
    args = setup_exports(tmp_path, kind="annual")
    args["selection"] = FeatureSelection("monthly", "2026-05", "2026-05", 3)
    with pytest.raises(ValueError):
        read_features(**args)


def test_explicit_mask_is_preserved_and_invalid_values_are_not_valid_zeroes(tmp_path):
    args = setup_exports(tmp_path, kind="annual")
    path = tmp_path / "p0.npz"
    with np.load(path) as source:
        data = {k: source[k] for k in source.files}
    data["valid_mask"] = np.array([[False, True], [True, True]])
    data["embedding"][0, :, 0, 0] = np.nan
    np.savez(path, **data)
    args["tile_sha256"]["p0"] = sha(path)
    result = read_features(**args)
    assert result.valid[0].sum() == 3
    np.testing.assert_array_equal(result.values[0, 0, 0], np.zeros(3))
    assert not result.valid[0, 0, 0]
    assert result.identity["valid_pixels"] == [3, 4]


def test_nonboolean_mask_and_partial_digest_inventory_are_rejected(tmp_path):
    args = setup_exports(tmp_path)
    args["tile_sha256"].pop("p1")
    with pytest.raises(ValueError, match="digest"):
        read_features(**args)
    args = setup_exports(tmp_path)
    path = tmp_path / "p0.npz"
    with np.load(path) as source:
        data = {k: source[k] for k in source.files}
    data["valid_mask"] = np.ones((2, 2), dtype=float)
    np.savez(path, **data)
    args["tile_sha256"]["p0"] = sha(path)
    with pytest.raises(ValueError, match="mask"):
        read_features(**args)


def test_test_split_requires_explicit_request_and_is_recorded(tmp_path):
    args = setup_exports(tmp_path)
    path = tmp_path / "p2.npz"
    np.savez(path, embedding=np.ones((2, 3, 2, 2)), timestamps=[202604, 202605])
    args["tile_sha256"]["p2"] = sha(path)
    result = read_features(**args, splits=("test",))
    assert result.indices == (2,)
    assert result.identity["test_records_read"] is True
    with pytest.raises(ValueError):
        read_features(**args, splits=("train", "train"))


def test_auxiliary_partition_alias_cannot_bypass_explicit_test_request(tmp_path):
    args = setup_exports(tmp_path)
    cache = json.loads(args["cache_path"].read_text())
    cache["split"]["group0"] = [2]
    args["cache_path"].write_text(json.dumps(cache))
    args["cache_sha256"] = sha(args["cache_path"])
    rewrite_manifest(
        args,
        lambda d: d.update(split=cache["split"], cache_sha256=args["cache_sha256"]),
    )
    assert read_features(**args).indices == (0, 1)
    with pytest.raises(ValueError, match="splits"):
        read_features(**args, splits=("group0",))


def test_disk_output_matches_memory_and_never_overwrites_an_existing_bundle(tmp_path):
    args = setup_exports(tmp_path)
    memory = read_features(**args)
    out = tmp_path / "bundle"
    disk = read_features(**args, output=out)
    np.testing.assert_array_equal(np.load(out / "features.npy"), memory.values)
    np.testing.assert_array_equal(np.load(out / "valid.npy"), memory.valid)
    assert json.loads((out / "identity.json").read_text()) == disk.identity == memory.identity
    with pytest.raises(FileExistsError):
        read_features(**args, output=out)
