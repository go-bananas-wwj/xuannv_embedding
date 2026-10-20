import json

import numpy as np
import pytest

from xuannv_embedding.downstream import full_region_readouts as full
from xuannv_embedding.export.context import dump
from xuannv_embedding.training import manuscript_followup as followup
from xuannv_embedding.training.shared_fusion_search import ref


@pytest.mark.parametrize("query_only_tiles", [0, 1])
def test_frozen_followup_runs_both_actual_readout_heads_against_registered_aef(
    tmp_path, monkeypatch, query_only_tiles
):
    rng = np.random.default_rng(41)
    count = 22 + query_only_tiles
    values = rng.normal(size=(count, 64, 8, 8)).astype(np.float32)
    features = tmp_path / "features.npy"
    np.save(features, values)
    baseline = tmp_path / "aef.npy"
    np.save(baseline, values.transpose(0, 2, 3, 1))
    valid = tmp_path / "valid.npy"
    np.save(valid, np.ones((count, 8, 8), bool))
    labels = tmp_path / "labels.npz"
    np.savez(labels, target=(values[:, 0] > 0).astype(np.int8))
    records = [{"patch_id": str(i), "bounds": [80 * i, 0, 80 * (i + 1), 80]} for i in range(count)]
    index = tmp_path / "index.json"
    dump(index, {"crs": "EPSG:32650", "records": records})
    data = {
        "state": "complete",
        "protocol": "full-region-data-v1",
        "indices": list(range(count)),
        "records": records,
        "train_positions": list(range(20)),
        "calibration_positions": [20, 21],
        "calibration_cohort": {"name": "fixed-fixture"},
        "features": {"aef": ref(baseline)},
        "valid": ref(valid),
        "labels": ref(labels),
        "size": 8,
        "crs": "EPSG:32650",
        "tasks": {"C": {"reference": ["target"]}, "R": {}, "Q": {}},
    }
    datafile = tmp_path / "data.json"
    dump(datafile, data)
    anchors = {}
    for head in ["mlp", "conv3x3"]:
        job = {
            "protocol": full.PROTOCOL,
            "data": ref(datafile),
            "model": "aef",
            "head": head,
            "budget": 20,
            "seed": 41,
            "device": "cpu",
            "archives": {},
            "output": str(tmp_path / "aef" / head),
        }
        path = tmp_path / f"aef-{head}.json"
        dump(path, job)
        full.run(path, "calibration")
        anchors[head] = ref(path)
    original_jobs = followup.reader_jobs

    def cpu_readers(*args):
        jobs = original_jobs(*args)
        for job in jobs:
            path = tmp_path / job["spec"]
            spec = json.loads(path.read_text())
            spec["device"] = "cpu"
            dump(path, spec)
        return jobs

    monkeypatch.setattr(followup, "reader_jobs", cpu_readers)
    path = tmp_path / "followup.json"
    dump(
        path,
        {
            "schema": "manuscript-followup-v1",
            "region": {
                "name": "region_a",
                "index": ref(index),
                "evaluation": ref(datafile),
                "task_source": "reference",
                "support_seed": 41,
                "aef_jobs": anchors,
            },
            "case": "fixture",
            "producer": {"kind": "frozen", "features": ref(features)},
            "output": str(tmp_path / "separate-reader-root"),
            "retire_arrays": False,
        },
    )
    result = followup.score(path)
    assert result["state"] == "complete"
    assert {row["head"] for row in result["rows"]} == {"mlp", "conv3x3"}
    assert result["all_support_and_scored_positions_match_AEF"] is True
    assert followup.score(path) == result
    monkeypatch.setattr(full, "_fit", lambda *a, **k: pytest.fail("query must reuse frozen heads"))
    query = followup.query(path)
    assert query["heads_refitted"] is False
    assert query["thresholds_refitted"] is False
    assert len(query["rows"]) == 2
    for row in query["rows"]:
        assert row["all_region"]["observations"] == count * 8 * 8
        assert row["outside_support_and_calibration"]["observations"] == query_only_tiles * 8 * 8
    registered_query = tmp_path / "query.json"
    dump(
        registered_query,
        {
            "schema": "manuscript-frozen-query-v1",
            "jobs": [ref(tmp_path / job["spec"]) for job in result["jobs"]],
            "output": str(tmp_path / "registered-query"),
        },
    )
    assert followup.query_jobs(registered_query) == query


@pytest.mark.parametrize("policy", [None, "registered_basic"])
def test_summary_uses_explicit_public_reference_without_changing_scores(tmp_path, policy):
    prior = tmp_path / "prior.json"
    heads = ["mlp", "conv3x3"]
    dump(
        prior,
        {
            "scores": {
                "aef": {"reference": {"region_a": dict.fromkeys(heads, 0.83)}},
                "viewwise075": {"highres": {"region_a": dict.fromkeys(heads, 0.84)}},
            },
            "reference": {"region_a": {head: {"public": 0.90} for head in heads}},
        },
    )
    followups = []
    for case, value in [("original_public", 0.80), ("original_highres", 0.85)]:
        root = tmp_path / case
        root.mkdir()
        spec = tmp_path / f"{case}.json"
        dump(spec, {"case": case, "region": {"name": "region_a"}, "output": str(root)})
        rows = [
            {
                "case": case,
                "region": "region_a",
                "head": head,
                "task": "target",
                "metrics": {"f1": value, "iou": value, "ap": value},
            }
            for head in heads
        ]
        dump(
            root / "result.json",
            {"state": "complete", "spec_sha256": ref(spec)["sha256"], "rows": rows},
        )
        followups.append(ref(spec))
    spec = tmp_path / "summary.json"
    document = {
        "schema": "manuscript-summary-v1",
        "prior": ref(prior),
        "followups": followups,
        "output": str(tmp_path / "report"),
    }
    if policy:
        document["public_reference_policy"] = policy
    dump(spec, document)
    result = followup.summarize(spec)
    assert result["selected_for_query"] == ("original_highres" if policy else None)
    assert result["scores"]["original_public"]["region_a"]["mlp"] == 0.80
    assert result["scores"]["original_highres"]["region_a"]["mlp"] == 0.85
    assert len(result["all_task_rows"]) == 4
