import pytest
import torch

from xuannv_embedding.training.regional_base import training_domain, validate_domains


def cache():
    return {
        "records": [{"region": "area_a"} for _ in range(3)]
        + [{"region": "area_b"} for _ in range(4)],
        "months": ["2025-05", "2025-06"],
        "domains": [
            {"region": "area_a", "months": ["2025-05", "2025-06"], "indices": [0, 1, 2]},
            {"region": "area_b", "months": ["2025-12", "2026-01"], "indices": [3, 4, 5, 6]},
        ],
    }


def test_domain_schedule_alternates_regions_without_mixing_time_windows():
    data = cache()
    validate_domains(data)
    assert training_domain(data, 1) == data["domains"][0]
    assert training_domain(data, 2) == data["domains"][1]
    assert training_domain(data, 3) == data["domains"][0]


def test_single_region_keeps_original_index_order():
    data = cache()
    del data["domains"]
    domain = training_domain(data, 1)
    assert domain["indices"] == list(range(7))
    assert domain["months"] == data["months"]


@pytest.mark.parametrize("failure", ["overlap", "missing", "invalid_month", "duplicate_region"])
def test_domain_contract_rejects_invalid_geometry_or_time(failure):
    data = cache()
    if failure == "overlap":
        data["domains"][1]["indices"][0] = 2
    elif failure == "missing":
        data["domains"][1]["indices"].pop()
    elif failure == "invalid_month":
        data["domains"][1]["months"][1] = "2026-02"
    else:
        data["domains"][1]["region"] = "area_a"
    with pytest.raises(ValueError, match="[Dd]omain"):
        validate_domains(data)


def test_shared_seed_keeps_rank_batches_inside_selected_domain():
    data = cache()
    domain = training_domain(data, 2)
    picks = torch.randperm(len(domain["indices"]), generator=torch.Generator().manual_seed(41))
    selected = [domain["indices"][int(i)] for i in picks]
    assert set(selected) == {3, 4, 5, 6}


def test_domain_rejects_sample_region_mismatch():
    data = cache()
    data["records"][3]["region"] = "area_a"
    with pytest.raises(ValueError, match="Domain region"):
        validate_domains(data)
