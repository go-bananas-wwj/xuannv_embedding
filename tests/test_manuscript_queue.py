import pytest

from xuannv_embedding.training.manuscript_queue import validate_jobs


def test_queue_rejects_cycles_and_unrelated_device_zero():
    jobs = [{"id": "a", "depends_on": ["b"]}, {"id": "b", "depends_on": ["a"]}]
    with pytest.raises(ValueError):
        validate_jobs(jobs, [1, 2])
    with pytest.raises(ValueError):
        validate_jobs([], [0, 1])


def test_queue_accepts_paired_forks_and_unique_producers():
    jobs = [
        {"id": "base", "depends_on": []},
        {"id": "public", "depends_on": ["base"]},
        {"id": "local", "depends_on": ["base"]},
    ]
    validate_jobs(jobs, [1, 2, 3])
