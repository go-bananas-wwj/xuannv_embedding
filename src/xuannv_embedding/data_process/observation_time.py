"""Physical observation intervals independent of author-assigned analysis bins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


def _next_month(value: date) -> date:
    return date(value.year + value.month // 12, value.month % 12 + 1, 1)


@dataclass(frozen=True)
class ObservationInterval:
    start: date
    end: date
    precision: str
    analysis_period: str | None

    def __post_init__(self) -> None:
        if type(self.start) is not date or type(self.end) is not date or self.end <= self.start:
            raise ValueError("Observation interval requires increasing physical dates")
        if not isinstance(self.precision, str) or self.precision not in {"day", "month", "year"}:
            raise ValueError("Unknown observation time precision")
        expected = None
        if self.precision == "day":
            expected = self.start + timedelta(days=1)
        elif self.precision == "month" and self.start.day == 1:
            expected = _next_month(self.start)
        elif self.precision == "year" and self.start.month == self.start.day == 1:
            expected = date(self.start.year + 1, 1, 1)
        if self.end != expected:
            raise ValueError("Physical interval does not match its declared precision")
        if self.analysis_period is not None:
            if not isinstance(self.analysis_period, str) or not re.fullmatch(
                r"[0-9]{4}(?:-[0-9]{2})?", self.analysis_period
            ):
                raise ValueError("Invalid author analysis period")
            fields = self.analysis_period.split("-")
            try:
                date(int(fields[0]), int(fields[1]) if len(fields) == 2 else 1, 1)
            except ValueError as error:
                raise ValueError("Invalid author analysis period") from error

    @classmethod
    def from_mapping(cls, record: dict) -> ObservationInterval:
        if not isinstance(record, dict) or set(record) != {
            "start",
            "end",
            "precision",
            "analysis_period",
        }:
            raise ValueError("Observation time fields must be explicit and known")
        try:
            start, end = date.fromisoformat(record["start"]), date.fromisoformat(record["end"])
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid physical observation date") from error
        return cls(start, end, record["precision"], record["analysis_period"])


def month_support(observations: list[ObservationInterval], months: list[int]) -> np.ndarray:
    """One row per actual view; uniform support over intersecting calendar bins.

    Year precision represents unknown timing within that year, not twelve images.
    Analysis-bin labels never replace the physical acquisition interval.
    Missing public observations remain encoded by separate pixel coverage masks.
    """
    if not observations or any(not isinstance(v, ObservationInterval) for v in observations):
        raise ValueError("Physical observations must be explicit intervals")
    if not months or any(type(v) is not int for v in months):
        raise ValueError("Invalid public calendar bins")
    try:
        starts = [date(v // 100, v % 100, 1) for v in months]
        ends = [_next_month(v) for v in starts]
    except ValueError as error:
        raise ValueError("Invalid public calendar bins") from error
    serial = [v.year * 12 + v.month - 1 for v in starts]
    if len(set(months)) != len(months) or any(b - a != 1 for a, b in zip(serial, serial[1:])):
        raise ValueError("Public calendar bins must be unique, consecutive and increasing")
    weights = np.zeros((len(observations), len(months)), dtype=np.float32)
    for i, observation in enumerate(observations):
        active = [
            min(observation.end, end) > max(observation.start, start)
            for start, end in zip(starts, ends)
        ]
        if not any(active):
            raise ValueError("Physical observation does not overlap the public calendar")
        weights[i, active] = 1.0 / sum(active)
    return weights
