from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, shift

from xuannv_embedding.data_process.image_quality import read_frame, translation
from xuannv_embedding.data_process.release_quality import (
    apply_pan,
    merge_release,
    mux_identity,
    original_product,
    pan_clear_mask,
    validate_release,
)
from xuannv_embedding.utils.manifest import ManifestRecord, load_manifest, write_manifest


def test_pan_cloud_guard_preserves_native_grid_and_blocks_invalid():
    classes = np.zeros((160, 160), dtype=np.uint8)
    classes[80, 80] = 1
    classes[10, 10] = 255
    valid = np.ones((640, 640), dtype=bool)
    valid[0, 0] = False
    clear = pan_clear_mask(classes, valid)
    assert clear.shape == valid.shape
    assert not clear[320:324, 320:324].any()
    assert not clear[316:320, 320:324].any()
    assert not clear[40:44, 40:44].any()
    assert not clear[0, 0]
    assert clear[300, 300]
    assert not pan_clear_mask(np.ones_like(classes), valid).any()
    with pytest.raises(ValueError, match="native MUX/PAN"):
        pan_clear_mask(classes[:80], valid)


def test_registration_detects_translation_and_does_not_certify_missing_texture():
    rng = np.random.default_rng(123)
    reference = gaussian_filter(rng.normal(size=(128, 128)), 2)
    moving = shift(reference, (3, -2), order=1, mode="constant")
    valid = np.ones_like(reference, dtype=bool)
    valid[:10] = valid[-10:] = False
    result = translation(reference, moving, valid)
    assert result["status"] == "measured"
    assert result["shift_yx_pixels"] == [-3, 2]
    assert result["peak_ncc"] > 0.8
    assert translation(reference, moving, np.zeros_like(valid))["status"] == "insufficient_overlap"
    flat = np.ones_like(reference)
    assert translation(flat, flat, valid)["status"] == "insufficient_texture"


def test_registration_mask_downsampling_requires_almost_full_support(tmp_path):
    profile = dict(
        driver="GTiff",
        count=1,
        height=256,
        width=256,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_origin(512000, 3585280, 5, 5),
    )
    values = np.ones((256, 256), dtype=np.uint8)
    with rasterio.open(tmp_path / "values.tif", "w", **profile) as ds:
        ds.write(values, 1)
    values[0, 0] = 0
    with rasterio.open(tmp_path / "mask.tif", "w", **profile) as ds:
        ds.write(values, 1)
    _, valid = read_frame(tmp_path, {"path": "values.tif", "mask": "mask.tif"})
    assert not valid[0, 0]  # A 75% valid coarse cell must not round to uint8=1.
    assert valid[1, 1]


def test_scene_identity_does_not_invent_monthly_lineage():
    scene = "GF6_PMS_E110.0_N30.0_20200105_L1A123"
    assert mux_identity({"archive_member": scene + "-MSS_ORTHO.tif"}) == scene
    assert original_product({"source": "GF6_PAN_c1", "scene_id": scene}) == scene
    assert original_product({"source": "s2", "path": "monthly.tif"}) is None
    assert (
        original_product({"source": "JL1_test", "path": "20200101_JL1_A_L3B_5m_aabb.tif"})
        == "JL1_A_L3B"
    )


def test_merge_quarantines_leakage_and_offsets_before_fitting_train_stats(tmp_path):
    base, output = tmp_path / "base", tmp_path / "v6"
    base.mkdir()
    output.mkdir()
    source = "GF6_PAN_c1"
    schemas = {
        "s1": {"channels": 1, "role": "temporal"},
        source: {"channels": 1, "role": "highres"},
    }
    (base / "sources.json").write_text(json.dumps(schemas))
    rows, highres, fingerprints = [], {}, {}
    for split, priority in (("train", 0), ("test", 2)):
        parent = f"32650:{400 + priority}:2800"

        def observation(name, src, mean, month=1, scene=None):
            path = f"{split}/{name}.tif"
            item = dict(
                path=path,
                source=src,
                date=f"2020-{month:02}-01",
                parent_key=parent,
                sha256=hashlib.sha256(path.encode()).hexdigest(),
                valid_fraction=1.0,
                band_counts=[100],
                band_mean=[mean],
                band_variance=[4],
            )
            if scene:
                item.update(
                    scene_id=scene,
                    status="accepted",
                    pixel_sha256=path,
                    pan_mux_registration={"status": "measured"},
                )
                highres[path] = item
            rows.append((path, src, priority, item["sha256"], scene))
            return item

        lowres = [
            observation(str(m), "s1", 2 if split == "train" else 200, m) for m in range(1, 13)
        ]
        pan = [observation("shared", source, 500, scene="shared_scene")]
        pan += [observation("unique", source, 7 if split == "train" else 200, scene=split)]
        if split == "train":
            pan += [observation("offset", source, 999, scene="offset_scene")]
            pan += [observation("unresolved", source, 999, scene="unknown_scene")]
        obs = {"s1": lowres, source: pan, "landsat": [observation("ls", "landsat", 999)]}
        record = ManifestRecord(
            patch_id=parent,
            region="annual_2020",
            sources={k: [o["path"] for o in v] for k, v in obs.items()},
            grid={"parent_key": parent},
            quality={},
            provenance={"split": split, "year": 2020, "observations": obs},
        )
        path = base / f"2020.{split}.manifest.jsonl"
        write_manifest(
            path, [record], months=[f"2020-{m:02}" for m in range(1, 13)], generator_version="test"
        )
        fingerprints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(output / "audit.sqlite") as db:
        db.execute(
            "CREATE TABLE observations(path TEXT,source TEXT,priority INTEGER,"
            "sha TEXT,product TEXT)"
        )
        db.executemany("INSERT INTO observations VALUES (?,?,?,?,?)", rows)
    for name, records in (
        ("pan-final.sqlite", highres),
        ("lineage.sqlite", {}),
        (
            "registration-full.sqlite",
            {
                p: {
                    "path": p,
                    "status": (
                        "large_relative_offset_review"
                        if "offset" in p
                        else "inconclusive" if "unresolved" in p else "measured"
                    ),
                }
                for p in highres
            },
        ),
    ):
        with sqlite3.connect(output / name) as db:
            db.execute("CREATE TABLE results(path TEXT,payload TEXT)")
            db.executemany(
                "INSERT INTO results VALUES (?,?)", [(p, json.dumps(r)) for p, r in records.items()]
            )
    for name, content in {
        "run.json": {
            "input": str(base),
            "data_root": str(tmp_path),
            "input_manifests": fingerprints,
        },
        "visual-review.json": {"completed": True},
        "registration-full-summary.json": {"status": "complete"},
    }.items():
        (output / name).write_text(json.dumps(content))
    report = merge_release(output)
    assert not report["training_ready"]
    assert report["exclusions"]["original_product_cross_split"] == 1
    assert report["exclusions"]["image_review_quarantine"] == 1
    assert report["exclusions"]["highres_relative_registration_unresolved"] == 1
    assert report["exclusions"]["landsat_cloud_qa_unavailable"] == 2
    stats = json.loads((output / "statistics" / f"{source}_stats.json").read_text())
    assert stats["num_files"] == 1 and stats["mean"] == [7]
    assert stats["std"] == [2] and stats["fit_split"] == "train"
    records = load_manifest(output / "2020.train.manifest.jsonl").records
    assert records[0].sources[source] == ["train/unique.tif"]
    assert "landsat" not in records[0].sources
    assert (
        records[0].provenance["observations"][source][0]["relative_registration"]["status"]
        == "measured"
    )
    with sqlite3.connect(output / "registration-full.sqlite") as db:
        db.execute("DELETE FROM results WHERE path='train/unique.tif'")
    with pytest.raises(ValueError, match="coverage differs"):
        merge_release(output)
    (output / "loader-check.json").write_text(
        json.dumps(
            {
                "loader_passed": True,
                "manifests": {name: {"samples_read": 1} for name in fingerprints},
            }
        )
    )
    (output / "statistics" / f"{source}_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_release(output)


def test_pan_screening_writes_clear_pixel_statistics_and_rejects_changed_payload(tmp_path):
    rng = np.random.default_rng(9)
    coarse = 1000 + 200 * gaussian_filter(rng.normal(size=(160, 160)), 1)
    pan = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1).astype("uint16")
    pan[0, 0] = 0

    def write(name, values, spacing):
        with rasterio.open(
            tmp_path / name,
            "w",
            driver="GTiff",
            count=len(values),
            dtype=values.dtype,
            height=values.shape[1],
            width=values.shape[2],
            crs="EPSG:32650",
            transform=from_origin(512000, 3585280, spacing, spacing),
        ) as ds:
            ds.write(values)

    write("pan.tif", pan[None], 2)
    write("numeric.tif", (pan > 0).astype("uint8")[None], 2)
    mux = np.stack([coarse] * 4).astype("float32")
    write("mux.tif", mux, 8)
    classes = np.zeros((160, 160), dtype="uint8")
    classes[:, :30] = 1
    write("cloud.tif", np.stack([classes == 0, classes]).astype("uint8"), 8)
    original = {
        "path": "pan.tif",
        "mask": "numeric.tif",
        "sha256": hashlib.sha256((tmp_path / "pan.tif").read_bytes()).hexdigest(),
    }
    prediction = {
        "status": "accepted",
        "mask": "cloud.tif",
        "mux": {
            "path": "mux.tif",
            "sha256": hashlib.sha256((tmp_path / "mux.tif").read_bytes()).hexdigest(),
        },
    }
    result = apply_pan((original, prediction, tmp_path, tmp_path / "output"))
    assert result["status"] == "accepted"
    assert result["pan_mux_registration"]["status"] == "measured"
    clear = pan_clear_mask(classes, pan > 0)
    assert result["band_counts"] == [int(clear.sum())]
    assert result["band_mean"] == pytest.approx([pan[clear].mean()])
    assert result["band_variance"] == pytest.approx([pan[clear].var()])
    with rasterio.open(tmp_path / result["mask"]) as ds:
        np.testing.assert_array_equal(ds.read(1), clear)
    original["sha256"] = "0" * 64
    rejected = apply_pan((original, prediction, tmp_path, tmp_path / "output"))
    assert rejected["status"] == "excluded" and "changed" in rejected["error"]
