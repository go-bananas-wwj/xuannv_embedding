import json
from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.export import semantic_shards
from xuannv_embedding.export.context import dump, sha


def ref(p):
    return {"path": str(p), "sha256": sha(p)}


def fixture_plan(tmp_path):
    rows = [
        {"patch_id": f"p{i}", "bounds": [i * 32, 0, i * 32 + 32, 32], "crs": "EPSG:32650"}
        for i in range(4)
    ]
    cache = tmp_path / "cache.json"
    dump(cache, {"state": "complete", "months": ["2024-01"], "records": rows})
    rasters = []
    for i, row in enumerate(rows):
        p = tmp_path / f"source{i}.bin"
        p.write_bytes(str(i).encode())
        rasters.append({**row, "raster": ref(p)})
    inputs = tmp_path / "rasters.json"
    dump(inputs, {"schema": "semantic-raster-inputs-v1", "records": rasters, "dependencies": []})
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"frozen encoder identity")
    stats = tmp_path / "statistics.json"
    dump(stats, {"mean": [0] * 3, "std": [1] * 3})
    source = tmp_path / "source.json"
    dump(
        source,
        {
            "schema": "regional-semantic-features-v1",
            "cache": ref(cache),
            "statistics": ref(stats),
            "weights": ref(weights),
            "model": "vit_large_patch16_dinov3.sat493m",
            "source": "rgb",
            "output": str(tmp_path / "merged"),
            "seed": 41,
            "fit_samples": 64,
            "components": 64,
            "rasters": ref(inputs),
            "output_shape": [2, 2],
            "max_pixels": 128,
            "crop_batch": 1,
        },
    )
    plan = tmp_path / "plan-spec.json"
    dump(
        plan,
        {
            "schema": "regional-semantic-shards-v1",
            "source_spec": ref(source),
            "shards": 2,
            "output": str(tmp_path / "shards"),
        },
    )
    return plan


def complete_shards(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    for shard in plan["shards"]:
        spec = json.loads(Path(shard["spec"]["path"]).read_text())
        root = Path(spec["output"])
        root.mkdir(parents=True, exist_ok=True)
        native = np.stack([np.full((1024, 2, 2), i + 1, np.float16) for i in shard["positions"]])
        np.save(root / "native.npy", native)
        np.save(root / "native-mask.npy", np.ones((len(native), 1, 2, 2), np.float32))
        dump(root / "identity.json", {"spec_sha256": sha(shard["spec"]["path"])})
        dump(
            root / "native-status.json",
            {
                "stage": "complete",
                "completed": len(native),
                "source_shape": [32, 32],
                "native_shape": list(native.shape),
            },
        )
    return plan


def test_semantic_shards_restore_original_order_without_fitting_independent_projections(tmp_path):
    plan = semantic_shards.prepare(fixture_plan(tmp_path))
    complete_shards(plan)
    output = semantic_shards.merge(plan)
    result = np.load(Path(output) / "native.npy")
    np.testing.assert_array_equal(result[:, 0, 0, 0], [1, 2, 3, 4])
    assert result.shape == (4, 1024, 2, 2)
    assert not (Path(output) / "projection.npz").exists()
    assert json.loads((Path(output) / "native-status.json").read_text())["completed"] == 4


def test_semantic_shards_reject_overlapping_positions_and_incomplete_extraction(tmp_path):
    path = semantic_shards.prepare(fixture_plan(tmp_path))
    plan = complete_shards(path)
    bad = json.loads(json.dumps(plan))
    bad["shards"][1]["positions"][0] = bad["shards"][0]["positions"][0]
    p = tmp_path / "bad.json"
    dump(p, bad)
    with pytest.raises(ValueError, match="coverage"):
        semantic_shards.merge(p)
    source = json.loads(Path(plan["shards"][0]["spec"]["path"]).read_text())
    status = Path(source["output"]) / "native-status.json"
    row = json.loads(status.read_text())
    row["stage"] = "extracting"
    dump(status, row)
    with pytest.raises(ValueError, match="incomplete"):
        semantic_shards.merge(path)


def test_semantic_shards_reject_changed_encoder_parameters_even_with_rebound_hash(tmp_path):
    path = semantic_shards.prepare(fixture_plan(tmp_path))
    plan = complete_shards(path)
    source = Path(plan["shards"][0]["spec"]["path"])
    spec = json.loads(source.read_text())
    spec["crop_batch"] = 2
    dump(source, spec)
    plan["shards"][0]["spec"] = ref(source)
    dump(path, plan)
    with pytest.raises(ValueError, match="parameters"):
        semantic_shards.merge(path)


def test_semantic_shards_never_overwrite_unowned_native_outputs(tmp_path):
    path = semantic_shards.prepare(fixture_plan(tmp_path))
    complete_shards(path)
    root = tmp_path / "merged"
    root.mkdir()
    np.save(root / "native.npy", np.ones((1, 1024, 2, 2), np.float16))
    digest = sha(root / "native.npy")
    with pytest.raises(ValueError, match="overwrite"):
        semantic_shards.merge(path)
    assert sha(root / "native.npy") == digest
