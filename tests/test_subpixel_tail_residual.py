from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.export import subpixel_tail_residual as tail


def test_masked_highpass_has_zero_known_mean_and_ignores_unknown_nan():
    values = np.arange(75, dtype=np.float32).reshape(75, 1, 1)
    mask = np.ones((25, 1, 1), np.float32)
    mask[0] = 0
    values[[0, 25, 50]] = np.nan
    actual = tail.highpass(values, mask, np.ones(3))
    assert np.isfinite(actual).all()
    assert not actual[[0, 25, 50]].any()
    np.testing.assert_allclose(
        (actual.reshape(3, 25, -1) * mask.reshape(1, 25, -1)).sum(1), 0, atol=2e-5
    )


def fixture_model():
    rng = np.random.default_rng(41)
    public = rng.normal(size=(256, 64)).astype(np.float32)
    raw = rng.uniform(0, 255, size=(256, 75)).astype(np.float32)
    masks = np.ones((256, 25), np.float32)
    return tail.fit_parameters(public, raw, masks), rng


def test_additive_residual_preserves_public_principal_directions_and_is_bounded():
    model, rng = fixture_model()
    public = rng.normal(size=(64, 4, 5)).astype(np.float32)
    raw = rng.uniform(0, 255, size=(75, 4, 5)).astype(np.float32)
    masks = np.ones((25, 4, 5), np.float32)
    actual = tail.transform(public, raw, masks, np.ones((1, 4, 5), bool), model, strength=0.5)
    delta = (actual - public) / model["public_scale"][:, None, None]
    protected = model["public_basis"][:, 16:].T @ delta.reshape(64, -1)
    np.testing.assert_allclose(protected, 0, atol=2e-6)
    assert np.linalg.norm(delta, axis=0).max() <= 2.000001
    assert np.max(np.abs(actual - public)) > 0


def test_missing_visual_observations_return_all_original_public_coordinates():
    model, rng = fixture_model()
    public = rng.normal(size=(64, 4, 5)).astype(np.float32)
    actual = tail.transform(
        public,
        np.full((75, 4, 5), np.nan),
        np.zeros((25, 4, 5)),
        np.ones((1, 4, 5), bool),
        model,
        strength=0.5,
    )
    np.testing.assert_array_equal(actual, public)


def test_no_texture_or_zero_strength_is_exact_public_identity():
    model, rng = fixture_model()
    public = rng.normal(size=(64, 2, 2)).astype(np.float32)
    raw = rng.uniform(0, 255, size=(75, 2, 2)).astype(np.float32)
    mask = np.ones((25, 2, 2), np.float32)
    np.testing.assert_array_equal(
        tail.transform(public, raw, mask, np.ones((1, 2, 2)), model, strength=0), public
    )
    constant = tail.fit_parameters(
        rng.normal(size=(128, 64)), np.ones((128, 75)), np.ones((128, 25))
    )
    assert not constant["private_components"].any()
    np.testing.assert_array_equal(
        tail.transform(
            public, np.ones((75, 2, 2)), mask, np.ones((1, 2, 2)), constant, strength=0.5
        ),
        public,
    )


def test_public_basis_does_not_depend_on_private_source_values():
    rng = np.random.default_rng(41)
    public = rng.normal(size=(128, 64))
    a = tail.fit_parameters(public, rng.uniform(0, 255, size=(128, 75)), np.ones((128, 25)))
    b = tail.fit_parameters(public, np.empty((0, 75)), np.empty((0, 25)))
    for key in ["public_mean", "public_scale", "public_basis", "public_eigenvalues"]:
        np.testing.assert_array_equal(a[key], b[key])


def test_reload_keeps_identical_residual_output(tmp_path):
    model, rng = fixture_model()
    path = tmp_path / "model.npz"
    np.savez(path, **model)
    with np.load(path) as archive:
        cold = {k: archive[k] for k in archive.files}
    public = rng.normal(size=(64, 2, 3))
    raw = rng.uniform(0, 255, size=(75, 2, 3))
    mask = np.ones((25, 2, 3))
    domain = np.ones((1, 2, 3))
    np.testing.assert_array_equal(
        tail.transform(public, raw, mask, domain, model, strength=0.1),
        tail.transform(public, raw, mask, domain, cold, strength=0.1),
    )


@pytest.mark.parametrize("strength", [-1, float("nan"), 2])
def test_invalid_strength_is_rejected(strength):
    model, rng = fixture_model()
    with pytest.raises(ValueError, match="strength"):
        tail.transform(
            np.zeros((64, 2, 2)),
            np.zeros((75, 2, 2)),
            np.zeros((25, 2, 2)),
            np.ones((1, 2, 2)),
            model,
            strength=strength,
        )


def tail_fixture(tmp_path):
    from test_subpixel_rgb import cache_fixture

    from xuannv_embedding.data_process import subpixel_rgb
    from xuannv_embedding.export.context import dump, sha

    path, cache_spec = cache_fixture(tmp_path)
    subpixel_rgb.prepare(path)
    cache = Path(cache_spec["output"]) / "manifest.json"
    spec = {
        "schema": "subpixel-tail-residual-v1",
        "cache": {"path": str(cache), "sha256": sha(cache)},
        "seed": 41,
        "fit_samples": 128,
        "output": str(tmp_path / "tail"),
        "reserve_gib": 0,
        "budgets": [
            {"name": "subpixel_tail010", "strength": 0.1},
            {"name": "subpixel_tail050", "strength": 0.5},
        ],
    }
    target = tmp_path / "tail.json"
    dump(target, spec)
    return target, spec


def test_complete_fit_export_and_cold_parameters_keep_constant_rgb_exact_public(tmp_path):
    path, spec = tail_fixture(tmp_path)
    status = tail.fit(path)
    assert status["registered_input_tiles"] == status["public_observed_tiles"] == 2
    assert status["public_sample_rows"] == status["private_sample_rows"] == 128
    assert status["primary_or_weak_maps_loaded"] is False
    results = tail.export(path)
    assert len(results) == 2 and tail.fit(path) == status and tail.export(path) == results
    source = tail.SourceData(tail.specification(path))
    for b in spec["budgets"]:
        output = np.load(Path(spec["output"]) / "representations" / b["name"] / "annual.npy")
        np.testing.assert_array_equal(output, source.public.astype(np.float16))
    assert tail.load_parameters(Path(spec["output"]))["private_components"].shape == (16, 75)


def test_public_statistics_are_unchanged_when_rgb_cache_is_fully_missing(tmp_path):
    import copy

    from xuannv_embedding.export.context import dump, sha

    path, spec = tail_fixture(tmp_path)
    tail.fit(path)
    original = tail.load_parameters(Path(spec["output"]))
    cache = tail.shared.read_json(spec["cache"]["path"])
    mask = Path(cache["coverage"]["path"])
    np.save(mask, np.zeros((2, 25, 8, 8), np.float16))
    cache["coverage"]["sha256"] = sha(mask)
    cache_path = tmp_path / "missing-manifest.json"
    dump(cache_path, cache)
    missing = copy.deepcopy(spec)
    missing["cache"] = {"path": str(cache_path), "sha256": sha(cache_path)}
    missing["output"] = str(tmp_path / "missing-tail")
    missing_path = tmp_path / "missing-tail.json"
    dump(missing_path, missing)
    status = tail.fit(missing_path)
    assert status["private_sample_rows"] == 0 and status["private_observed_tiles"] == 0
    other = tail.load_parameters(Path(missing["output"]))
    for key in ["public_mean", "public_scale", "public_basis", "public_eigenvalues"]:
        np.testing.assert_array_equal(original[key], other[key])
    tail.export(missing_path)


def test_runtime_rejects_unknown_config_and_changed_parameters(tmp_path):
    path, spec = tail_fixture(tmp_path)
    with pytest.raises(ValueError, match="specification"):
        tail.validate_specification({**spec, "_base_": "bad"})
    tail.fit(path)
    (Path(spec["output"]) / "model.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="parameters"):
        tail.export(path)


@pytest.mark.parametrize("command", ["subpixel-rgb-cache", "subpixel-tail-residual"])
def test_new_commands_use_the_existing_single_cli(command, capsys):
    from xuannv_embedding.training.experiment import main

    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0 and "--spec" in capsys.readouterr().out


def test_nonfinite_or_negative_public_domain_is_rejected():
    model, _ = fixture_model()
    for value in [np.nan, -1]:
        with pytest.raises(ValueError, match="domain"):
            tail.transform(
                np.ones((64, 2, 2)),
                np.ones((75, 2, 2)),
                np.ones((25, 2, 2)),
                np.full((1, 2, 2), value),
                model,
                strength=0.1,
            )


@pytest.mark.parametrize(
    "field,value", [("path", None), ("path", "relative.json"), ("sha256", "z" * 64)]
)
def test_specification_rejects_malformed_cache_reference(tmp_path, field, value):
    _, spec = tail_fixture(tmp_path)
    spec["cache"][field] = value
    with pytest.raises(ValueError, match="specification"):
        tail.validate_specification(spec)
