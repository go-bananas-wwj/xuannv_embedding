import json
from pathlib import Path

import numpy as np
import pytest
from test_review_readouts import cohort

from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import review_readouts as shared
from xuannv_embedding.downstream import review_retrieval as retrieval
from xuannv_embedding.export.context import sha


def test_retrieval_reuses_completed_queries_and_preserves_the_registered_domain(
    tmp_path, monkeypatch
):
    path, spec, _ = cohort(tmp_path)
    primary.calibrate(path)
    primary.score(path, sha(Path(spec["output"]) / "calibration/identity.json"))
    common = tmp_path / "shared"
    shared.prepare_shared(path, common, "calibration")
    shared.prepare_shared(path, common, "test")
    task = {
        "protocol": "review-retrieval-v1",
        "cohort": {"path": str(path), "sha256": sha(path)},
        "shared": str(common),
        "reuse": [{"spec": {"path": str(path), "sha256": sha(path)}, "model": "candidate"}],
        "model": "candidate",
        "output": str(tmp_path / "retrieval"),
    }
    task_path = tmp_path / "retrieval.json"
    task_path.write_text(json.dumps(task))
    monkeypatch.setattr(primary, "_fit", lambda *a, **k: pytest.fail("existing prototype refit"))
    result = retrieval.run(task_path)
    assert result["imported_conditions"] == 1 and result["computed_conditions"] == 0
    rows = json.loads((tmp_path / "retrieval/results.json").read_text())
    original = json.loads((Path(spec["output"]) / "test/results.json").read_text())["candidate"]
    assert rows == [r for r in original if r["family"] == "Q"]


def test_extended_retrieval_budgets_keep_nested_support_prototypes(tmp_path):
    path, spec, _ = cohort(tmp_path)
    spec["retrieval_budgets"] = [1, 3, 5, 7, 10]
    for record in spec["labels"].values():
        file = Path(record["path"])
        with np.load(file) as z:
            arrays = {k: z[k] for k in z.files}
        yy, xx = np.indices((16, 16))
        arrays["osm_building"] = ((yy % 4 < 2) & (xx % 4 < 2)).astype(np.int8)[None]
        np.savez(file, **arrays)
        record["sha256"] = sha(file)
    lock = Path(spec["lock"]["path"])
    lock.write_text(
        json.dumps({"state": "locked", "contract_sha256": primary.contract_sha256(spec)})
    )
    spec["lock"]["sha256"] = sha(lock)
    path.write_text(json.dumps(spec))
    common = tmp_path / "shared"
    shared.prepare_shared(path, common, "calibration")
    shared.prepare_shared(path, common, "test")
    contract = {
        "protocol": "review-retrieval-v1",
        "cohort": {"path": str(path), "sha256": sha(path)},
        "shared": str(common),
        "reuse": [],
        "model": "candidate",
        "output": str(tmp_path / "query"),
    }
    task = tmp_path / "query.json"
    task.write_text(json.dumps(contract))
    result = retrieval.run(task)
    assert result["computed_conditions"] == 5
    root = tmp_path / "query/readouts"
    a = primary.load_readout(
        root / "Q_osm_building_7_1", sha(root / "Q_osm_building_7_1/identity.json")
    )
    b = primary.load_readout(
        root / "Q_osm_building_7_10", sha(root / "Q_osm_building_7_10/identity.json")
    )
    np.testing.assert_array_equal(a.arrays["prototypes"], b.arrays["prototypes"][:1])
