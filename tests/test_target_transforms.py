import numpy as np
import pytest

from xuannv_embedding.data.target_transforms import (
    aggregate_continuous,
    aggregate_structure,
    clcd_fractions,
    dsm_slope,
    osm_evidence,
)


def test_slope_uses_metres_and_masks_missing_neighbors_and_patch_edges():
    values = np.broadcast_to(np.arange(32, dtype=np.float32) * 40, (32, 32)).copy()
    valid = np.ones_like(values, dtype=bool)
    result = dsm_slope(values, valid)
    np.testing.assert_allclose(result["values"][1:-1, 1:-1], 45, atol=1e-5)
    assert result["valid"].sum() == 900
    valid[16, 16] = False
    result = dsm_slope(values, valid)
    assert result["valid"].sum() == 895


def test_clcd_preserves_class_mixture_without_majority_or_unknown_background():
    raw = np.ones((1, 128, 128), np.uint8)
    raw[:, :, 2::4] = 8
    raw[:, :, 3::4] = 8
    valid = np.ones_like(raw, dtype=bool)
    valid[:, 0:4, 0:3] = False
    result = clcd_fractions(raw, valid)
    assert result["values"].shape == (1, 9, 32, 32)
    assert not result["valid"][0, 0, 0]
    assert result["values"][0, 0, 1, 1] == 0.5 and result["values"][0, 7, 1, 1] == 0.5
    np.testing.assert_allclose(result["values"].sum(1)[result["valid"]], 1)
    raw[0, 12, 12] = 10
    with pytest.raises(ValueError, match="CLCD"):
        clcd_fractions(raw, valid)


def test_coarse_nightlights_preserve_negative_values_and_coverage():
    values = np.full((1, 128, 128), -0.25, np.float32)
    valid = np.ones_like(values, dtype=bool)
    valid[:, 0:64, 0:48] = False
    values[~valid] = np.nan
    result = aggregate_continuous(values, valid, 64)
    assert result["values"].shape == (1, 2, 2)
    assert not result["valid"][0, 0, 0]
    assert result["values"][0, 1, 1] == -0.25


def test_osm_unknown_and_current_only_do_not_become_positives_or_negatives():
    values = np.array([255, 255, 255, 0], np.uint8)
    states = np.array([0, 0, 1, 0], np.uint8)
    confidence = np.full(4, 255, np.uint8)
    bits = np.array([0, 2, 1, 0], np.uint8)
    negative = np.array([0, 0, 3, 3], np.uint8)
    result = osm_evidence(values, states, confidence, bits, negative, np.full(4, 224, np.uint8))
    assert result["valid"].tolist() == [False, False, True, True]
    assert result["values"].tolist() == [0, 0, 1, 0]
    assert result["weight"][2] == 1
    assert result["weight"][3] == pytest.approx(224 / 255)
    bits[2] = 2
    with pytest.raises(ValueError, match="historical source"):
        osm_evidence(values, states, confidence, bits, negative, confidence)


def test_structure_aggregation_preserves_known_mass_and_does_not_fill_unknown():
    x = np.zeros((1, 1, 4, 4), np.float32)
    valid = np.zeros_like(x, dtype=bool)
    x[0, 0, 0, 0] = 1
    valid[0, 0, 0, 0] = True
    result = aggregate_structure({"values": x, "valid": valid, "weight": valid.astype(np.float32)})
    assert result["values"][0, 0, 0, 0] == 1
    assert result["weight"][0, 0, 0, 0] == 0.25
    assert result["valid"].sum() == 1
