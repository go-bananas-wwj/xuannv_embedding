import json

from xuannv_embedding.training.evaluation_queue import ready


def test_ready_requires_complete_dependencies(tmp_path):
    p = tmp_path / "status.json"
    job = {"requires": [{"path": str(p), "state": "complete"}]}
    assert not ready(job)
    p.write_text(json.dumps({"state": "running"}))
    assert not ready(job)
    p.write_text(json.dumps({"state": "complete"}))
    assert ready(job)
