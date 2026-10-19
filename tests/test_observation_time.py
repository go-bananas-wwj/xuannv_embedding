import numpy as np
import pytest

from xuannv_embedding.data_process.observation_time import ObservationInterval, month_support


def test_physical_month_support_does_not_overwrite_author_analysis_bin():
    observation = ObservationInterval.from_mapping(
        {
            "start": "2026-04-30",
            "end": "2026-05-01",
            "precision": "day",
            "analysis_period": "2026-05",
        }
    )
    weights = month_support([observation], [202603, 202604, 202605])
    assert observation.analysis_period == "2026-05"
    np.testing.assert_array_equal(weights, [[0, 1, 0]])


def test_year_only_static_view_stays_one_observation_with_annual_support():
    observation = ObservationInterval.from_mapping(
        {"start": "2024-01-01", "end": "2025-01-01", "precision": "year", "analysis_period": "2024"}
    )
    weights = month_support([observation], list(range(202401, 202413)))
    assert weights.shape == (1, 12)
    np.testing.assert_allclose(weights, np.full((1, 12), 1 / 12))


def test_month_precision_is_an_interval_not_an_invented_acquisition_day():
    observation = ObservationInterval.from_mapping(
        {
            "start": "2025-09-01",
            "end": "2025-10-01",
            "precision": "month",
            "analysis_period": "2025-09",
        }
    )
    np.testing.assert_array_equal(
        month_support([observation], [202508, 202509, 202510]), [[0, 1, 0]]
    )
    with pytest.raises(ValueError, match="precision"):
        ObservationInterval.from_mapping(
            {
                "start": "2025-09-01",
                "end": "2025-10-01",
                "precision": "day",
                "analysis_period": "2025-09",
            }
        )


def test_temporal_support_rejects_unknown_fields_invalid_bins_and_outside_window():
    record = {
        "start": "2024-01-01",
        "end": "2025-01-01",
        "precision": "year",
        "analysis_period": "2024",
    }
    with pytest.raises(ValueError, match="fields"):
        ObservationInterval.from_mapping({**record, "city": "must-not-branch"})
    observation = ObservationInterval.from_mapping(record)
    for months in [[202401, 202401], [202401, 202413], [202402, 202401]]:
        with pytest.raises(ValueError, match="calendar"):
            month_support([observation], months)
    with pytest.raises(ValueError, match="overlap"):
        month_support([observation], [202501, 202502])
    with pytest.raises(ValueError, match="precision"):
        ObservationInterval.from_mapping({**record, "precision": []})
