from xuannv_embedding.data_process.v5_audit import acceptance_status


def test_acceptance_never_approves_user_on_automatic_pass():
    checks = {
        "download": True,
        "catalog": True,
        "radiometry": True,
        "quality": True,
        "alignment": True,
        "targets": True,
        "statistics": True,
        "sample_index": True,
        "loader": True,
        "reproducibility": True,
        "visual_review": True,
    }
    result = acceptance_status(checks)
    assert result["status"] == "ready_for_review"
    assert result["user_accepted"] is False
    assert result["training_authorized"] is False


def test_missing_or_failed_gate_is_incomplete():
    assert acceptance_status({})["status"] == "incomplete"
    assert acceptance_status({"download": True})["status"] == "incomplete"
