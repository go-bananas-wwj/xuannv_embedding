import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift


def texture(seed=73, size=256):
    return gaussian_filter(np.random.default_rng(seed).normal(size=(size, size)), 1.5)


def test_interior_windows_recover_known_shifts_without_overlapping_evidence():
    from xuannv_embedding.data_process.v5_adaptive_alignment import audit_adaptive
    from xuannv_embedding.data_process.v5_alignment import PARAMETERS, audit_translation

    reference = texture()
    valid = np.zeros(reference.shape, bool)
    valid[70:186, 70:186] = True
    for delta in [(0, 0), (0, 0.5), (0, 1.5), (2, -1)]:
        moving = shift(reference, delta, order=1, mode="constant", cval=0)
        moved = shift(valid.astype(float), delta, order=1, mode="constant") >= 1 - 1e-6
        original = audit_translation(reference, moving, valid & moved, gsd=5)
        result = audit_adaptive(reference, moving, valid, moved, gsd=5)
        assert original["status"] == "uncertain"
        assert result["status"] == ("passed" if np.linalg.norm(delta) * 5 <= 5 else "over_limit")
        np.testing.assert_allclose(result["translation_yx_m"], -np.array(delta) * 5, atol=1)
        assert result["parameters"] == PARAMETERS
        origins = result["selection"]["origins_yx"]
        for i, a in enumerate(origins):
            for b in origins[i + 1 :]:
                assert abs(a[0] - b[0]) >= 64 or abs(a[1] - b[1]) >= 64


def test_layout_uses_reference_only_and_preserves_qualified_corners():
    from xuannv_embedding.data_process.v5_adaptive_alignment import audit_adaptive
    from xuannv_embedding.data_process.v5_alignment import audit_translation

    reference = texture()
    valid = np.ones(reference.shape, bool)
    result = audit_adaptive(reference, reference, valid, valid, gsd=5)
    baseline = audit_translation(reference, reference, valid, gsd=5)
    assert result["windows"] == baseline["windows"]
    assert result["selection"]["layout"] == "original_corners"
    other = audit_adaptive(reference, texture(17), valid, np.zeros_like(valid), gsd=5)
    assert other["selection"] == result["selection"]
    assert other["status"] == "uncertain"


def test_adaptive_rejects_empty_constant_periodic_unrelated_and_local_deformation():
    from xuannv_embedding.data_process.v5_adaptive_alignment import audit_adaptive

    reference = texture()
    valid = np.ones(reference.shape, bool)
    y, x = np.indices(reference.shape)
    periodic = np.sin(x * np.pi / 4) + np.cos(y * np.pi / 4)
    cases = [
        (reference, reference, np.zeros_like(valid)),
        (np.ones_like(reference), np.ones_like(reference), valid),
        (periodic, periodic, valid),
        (reference, texture(18), valid),
    ]
    moving = reference.copy()
    moving[128:] = shift(reference, (0, 3), order=1, mode="constant")[128:]
    cases.append((reference, moving, valid))
    for a, b, mask in cases:
        result = audit_adaptive(a, b, mask, mask, gsd=5)
        assert result["status"] == "uncertain"
    for gsd in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            audit_adaptive(reference, reference, valid, valid, gsd=gsd)
    with pytest.raises(ValueError):
        audit_adaptive(
            reference[:128, :128],
            reference[:128, :128],
            valid[:128, :128],
            valid[:128, :128],
            gsd=5,
        )


def test_small_clear_island_cannot_supply_three_independent_windows():
    from xuannv_embedding.data_process.v5_adaptive_alignment import audit_adaptive

    reference = texture()
    valid = np.zeros(reference.shape, bool)
    valid[96:160, 96:160] = True
    result = audit_adaptive(reference, reference, valid, valid, gsd=5)
    assert result["status"] == "uncertain"
    assert result["valid_windows"] < 3
