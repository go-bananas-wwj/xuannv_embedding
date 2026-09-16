import pytest

from xuannv_embedding.training.experiment_export import validate_export_identity


def test_export_rejects_another_spatial_folds_cache():
    run = {"config_sha256": "a", "cache_sha256": "b"}
    with pytest.raises(ValueError, match="cache"):
        validate_export_identity(run, config_sha="a", cache_sha="c")


def test_export_requires_matching_configuration_and_cache():
    run = {"config_sha256": "a", "cache_sha256": "b"}
    validate_export_identity(run, config_sha="a", cache_sha="b")
    with pytest.raises(ValueError, match="config"):
        validate_export_identity(run, config_sha="c", cache_sha="b")
