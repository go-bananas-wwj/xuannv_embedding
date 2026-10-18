import numpy as np
import pytest

from xuannv_embedding.export.anchored_fusion import align_samples, blend_aligned, minimax_rank


def test_tail_injection_keeps_public_directions_outside_the_injection_space():
    from xuannv_embedding.export.anchored_fusion import fit_tail, inject_tail

    rng = np.random.default_rng(14)
    u = rng.normal(size=(256, 64)).astype(np.float32)
    v = rng.normal(size=(256, 64)).astype(np.float32)
    raw = rng.normal(size=(256, 95)).astype(np.float32)
    model = fit_tail(u, v, raw, semantic_dimensions=16, seed=41)
    maps = [x[:16].T.reshape(x.shape[1], 4, 4) for x in [u, v, raw]]
    result = inject_tail(*maps, model, strength=0.25)
    delta = (result - maps[0]).reshape(64, -1)
    projection = model["basis"] @ model["basis"].T
    assert np.linalg.norm((np.eye(64) - projection) @ delta) < 1e-5
    np.testing.assert_array_equal(inject_tail(*maps, model, strength=0), maps[0])
    np.testing.assert_array_equal(
        inject_tail(maps[0], maps[1] * 0, maps[2] * 0, model, strength=0.25), maps[0]
    )


def test_missing_highres_does_not_change_the_fitted_public_or_spectral_branch():
    from xuannv_embedding.export.anchored_fusion import fit_tail, inject_tail

    rng = np.random.default_rng(12)
    u, v, raw = [rng.normal(size=(256, d)).astype(np.float32) for d in [64, 64, 95]]
    first = fit_tail(u, v, raw, semantic_dimensions=8, seed=41)
    second = fit_tail(u, v * 0, raw, semantic_dimensions=8, seed=41)
    maps = [x[:16].T.reshape(x.shape[1], 4, 4) for x in [u, v * 0, raw]]
    np.testing.assert_array_equal(
        inject_tail(*maps, first, strength=0.25), inject_tail(*maps, second, strength=0.25)
    )


def test_alignment_uses_paired_unlabelled_vectors_and_preserves_missing_source_exactly():
    rng = np.random.default_rng(41)
    semantic = rng.normal(size=(256, 64)).astype(np.float32)
    public = semantic * 0.7 + 0.1
    model = align_samples(public, semantic, ridge=1e-6)
    u, v = public[:16].T.reshape(64, 4, 4), semantic[:16].T.reshape(64, 4, 4)
    assert np.max(np.abs(blend_aligned(u, v, model, strength=1) - u)) < 1e-4
    np.testing.assert_array_equal(blend_aligned(u, v, model, strength=0), u)
    v[:, 0, 0] = 0
    fused = blend_aligned(u, v, model, strength=0.5)
    np.testing.assert_array_equal(fused[:, 0, 0], u[:, 0, 0])


def test_small_neighborhood_only_changes_the_semantic_residual_not_missing_public_cells():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(256, 64)).astype(np.float32)
    model = align_samples(x, x)
    u = rng.normal(size=(64, 3, 3)).astype(np.float32)
    v = np.zeros_like(u)
    np.testing.assert_array_equal(blend_aligned(u, v, model, strength=0.25, kernel=3), u)
    v[:, 1, 1] = 1
    result = blend_aligned(u, v, model, strength=0.25, kernel=3)
    np.testing.assert_array_equal(result[:, 0, 0], u[:, 0, 0])
    assert np.isfinite(result).all()


def test_minimax_rank_cannot_compensate_a_failed_region_with_large_other_gains():
    reference = {
        r: {h: {"aef": 0.6, "public": 0.5} for h in ["mlp", "conv3x3"]} for r in ["r1", "r2", "r3"]
    }
    balanced = {r: {h: 0.61 for h in ["mlp", "conv3x3"]} for r in reference}
    mixed = {r: {h: 0.9 for h in ["mlp", "conv3x3"]} for r in reference}
    mixed["r3"]["mlp"] = 0.59
    assert minimax_rank(balanced, reference) > minimax_rank(mixed, reference)
    del mixed["r3"]
    with pytest.raises(ValueError, match="region"):
        minimax_rank(mixed, reference)


def test_alignment_rejects_invalid_strength_or_unpaired_samples():
    with pytest.raises(ValueError):
        align_samples(np.ones((128, 64)), np.ones((127, 64)))
    model = align_samples(np.eye(64).repeat(2, axis=0), np.eye(64).repeat(2, axis=0))
    with pytest.raises(ValueError):
        blend_aligned(np.ones((64, 2, 2)), np.ones((64, 2, 2)), model, strength=-0.1)
