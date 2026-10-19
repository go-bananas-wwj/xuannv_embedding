import numpy as np
import pytest

from xuannv_embedding.export import public_visual_subspace as fusion


def test_pool_weights_only_real_observations_and_ignores_masked_nan():
    views = np.array([[[[3.0, np.nan]]], [[[9.0, 15.0]]]], np.float32)
    masks = np.array([[[[0.25, 0.0]]], [[[0.75, 1.0]]]], np.float32)
    values, coverage = fusion.pool_views(views, masks)
    np.testing.assert_allclose(values, [[[7.5, 15.0]]])
    np.testing.assert_array_equal(coverage, [[[0.75, 1.0]]])


def test_wholly_missing_views_return_zero_without_inventing_observations():
    values, mask = fusion.pool_views(np.full((3, 4, 2, 2), np.nan), np.zeros((3, 1, 2, 2)))
    assert not values.any() and not mask.any()


@pytest.mark.parametrize("mask", [-0.1, 1.1, np.nan])
def test_pool_rejects_invalid_coverage(mask):
    with pytest.raises(ValueError, match="coverage"):
        fusion.pool_views(np.ones((1, 4, 2, 2)), np.full((1, 1, 2, 2), mask))


def test_pool_rejects_nonfinite_observed_values():
    with pytest.raises(ValueError, match="observed"):
        fusion.pool_views(np.full((1, 4, 2, 2), np.nan), np.ones((1, 1, 2, 2)))


def test_stratified_samples_preserve_tile_quotas_and_do_not_depend_on_another_branch():
    arrays = [np.arange(24, dtype=np.float32).reshape(2, 3, 4) + i * 100 for i in range(3)]
    masks = [np.ones((1, 3, 4), bool) for _ in arrays]
    rows, positions = fusion.sample_rows(arrays, masks, limit=11, seed=41)
    assert rows.shape == (11, 2)
    assert np.bincount(positions[:, 0]).tolist() == [4, 4, 3]
    again, selected = fusion.sample_rows(arrays, masks, limit=11, seed=41)
    np.testing.assert_array_equal(rows, again)
    np.testing.assert_array_equal(positions, selected)


def test_empty_visual_projection_is_loadable_and_does_not_add_features():
    model = fusion.fit_projection(
        np.empty((0, 8), np.float32),
        dimensions=4,
        seed=41,
        clip=6.0,
        solver="randomized",
        allow_empty=True,
    )
    assert not model["components"].any()
    actual = fusion.project(np.zeros((8, 2, 3), np.float32), model, dimensions=4, clip=6.0)
    assert actual.shape == (4, 2, 3) and not actual.any()


def test_same_projection_reuses_leading_coordinates_and_survives_npz_reload(tmp_path):
    rng = np.random.default_rng(41)
    rows = rng.normal(size=(128, 64)).astype(np.float32)
    model = fusion.fit_projection(rows, dimensions=48, seed=41, clip=6.0, solver="eigh")
    path = tmp_path / "projection.npz"
    np.savez(path, **model)
    with np.load(path) as archive:
        loaded = {key: archive[key] for key in archive.files}
    image = rng.normal(size=(64, 3, 4)).astype(np.float32)
    full = fusion.project(image, loaded, dimensions=48, clip=6.0)
    shorter = fusion.project(image, loaded, dimensions=32, clip=6.0)
    np.testing.assert_array_equal(shorter, full[:32])
    np.testing.assert_array_equal(full, fusion.project(image, model, dimensions=48, clip=6.0))


def test_public_projection_is_independent_of_visual_availability():
    rng = np.random.default_rng(42)
    public = rng.normal(size=(100, 64)).astype(np.float32)
    first = fusion.fit_projection(public, dimensions=48, seed=41, clip=6.0, solver="eigh")
    fusion.fit_projection(
        np.empty((0, 1024)), dimensions=32, seed=41, clip=6.0, solver="randomized", allow_empty=True
    )
    second = fusion.fit_projection(public, dimensions=48, seed=41, clip=6.0, solver="eigh")
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_mask_normalized_alignment_does_not_dilute_observed_tokens():
    values = np.array([[[11.0, np.nan], [11.0, np.nan]]], np.float32)
    coverage = np.array([[[1.0, 0.0], [1.0, 0.0]]], np.float32)
    actual = fusion.masked_align(values, coverage, (32, 32), (8, 8), threshold=0.05)
    assert actual.shape == (1, 8, 8) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual[actual != 0], 11.0, atol=1e-6)
    assert np.count_nonzero(actual == 0) > 0


def test_missing_visual_branch_preserves_public_block_and_domain_mask():
    rng = np.random.default_rng(42)
    public_model = fusion.fit_projection(
        rng.normal(size=(128, 64)), dimensions=48, seed=41, clip=6.0, solver="eigh"
    )
    visual_model = fusion.fit_projection(
        np.empty((0, 1024)), dimensions=32, seed=41, clip=6.0, solver="randomized", allow_empty=True
    )
    public = rng.normal(size=(64, 4, 4)).astype(np.float32)
    domain = np.ones((1, 4, 4), bool)
    domain[:, 0, 0] = False
    output = fusion.transform_blocks(
        public,
        np.zeros((1024, 2, 2)),
        np.zeros((1, 2, 2)),
        domain,
        {"public": public_model, "visual": visual_model},
        public_dimensions=48,
        visual_dimensions=16,
        source_shape=(32, 32),
        threshold=0.05,
        clip=6.0,
    )
    expected = fusion.project(public, public_model, dimensions=48, clip=6.0)
    expected[:, 0, 0] = 0
    assert output.shape == (64, 4, 4)
    np.testing.assert_array_equal(output[:48], expected)
    assert not output[48:].any()


def test_projection_rejects_insufficient_or_invalid_samples():
    for rows in [np.zeros((4, 64)), np.full((100, 64), np.nan)]:
        with pytest.raises(ValueError, match="samples"):
            fusion.fit_projection(rows, dimensions=48, seed=41, clip=6.0, solver="eigh")


def test_json_duplicate_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "one", "schema": "two"}')
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        fusion.read_json(path)


def test_schema_rejects_unknown_or_inherited_config():
    with pytest.raises(ValueError, match="specification"):
        fusion.validate_specification({"schema": "public-visual-subspace-v1", "_base_": "x"})


def test_subspace_cli_is_routed_by_single_entrypoint(capsys):
    from xuannv_embedding.training.experiment import main

    with pytest.raises(SystemExit) as error:
        main(["public-visual-subspace", "--help"])
    assert error.value.code == 0
    assert "--spec" in capsys.readouterr().out


def fixture_spec(tmp_path):
    from xuannv_embedding.export.context import dump

    tmp_path.mkdir(parents=True, exist_ok=True)

    def reference(path):
        return {"path": str(path), "sha256": fusion.sha(path)}

    records = [
        {"patch_id": str(i), "bounds": [i * 80, 0, (i + 1) * 80, 80], "crs": "EPSG:32650"}
        for i in range(2)
    ]
    index = tmp_path / "index.json"
    dump(index, {"records": records, "crs": "EPSG:32650"})
    rng = np.random.default_rng(41)

    def array(name, values):
        path = tmp_path / (name + ".npy")
        np.save(path, values)
        return {**reference(path), "layout": "nchw", "index_sha256": fusion.sha(index)}

    public = array("public", rng.normal(size=(2, 64, 8, 8)).astype(np.float32))
    physical = array("availability", np.ones((2, 1, 8, 8), np.float32))
    visual = array("visual", rng.normal(size=(2, 1024, 8, 8)).astype(np.float32))
    coverage = array("mask", np.ones((2, 1, 8, 8), np.float32))
    columns = tmp_path / "columns.json"
    dump(columns, {"columns": ["s2:availability"]})
    months = [202400 + i for i in range(1, 13)]
    geometry = tmp_path / "geometry.json"
    dump(geometry, {"state": "complete", "months": months, "records": records})
    source_time = {
        "start": "2024-01-01",
        "end": "2025-01-01",
        "precision": "year",
        "analysis_period": "2024",
    }
    weights = tmp_path / "weights.npy"
    np.save(
        weights,
        fusion.month_support([fusion.ObservationInterval.from_mapping(source_time)], months),
    )
    contract = tmp_path / "contract.json"
    dump(
        contract,
        {
            "state": "complete",
            "months": months,
            "views": [{"slot": 0, "source_time": source_time}],
            "weights": reference(weights),
            "one_row_per_actual_view_not_month_copy": True,
        },
    )
    spec = {
        "schema": "public-visual-subspace-v1",
        "index": reference(index),
        "public": {"array": public, "domain": {"array": physical, "columns": reference(columns)}},
        "views": [
            {"slot": 0, "features": visual, "coverage": coverage, "geometry": reference(geometry)}
        ],
        "time_contract": reference(contract),
        "source_shape": [128, 128],
        "seed": 41,
        "fit_samples": 128,
        "clip": 6.0,
        "mask_threshold": 0.05,
        "output": str(tmp_path / "output"),
        "reserve_gib": 0,
        "budgets": [
            {"name": "public48_visual16", "public_dimensions": 48, "visual_dimensions": 16},
            {"name": "public32_visual32", "public_dimensions": 32, "visual_dimensions": 32},
        ],
    }
    path = tmp_path / "spec.json"
    dump(path, spec)
    return path, spec


def test_full_fit_export_and_cold_reload_preserve_shared_public_coordinates(tmp_path):
    from pathlib import Path

    path, spec = fixture_spec(tmp_path)
    fitted = fusion.fit(path)
    assert fitted["unique_input_tiles"] == 2
    assert fitted["samples"]["public"]["rows"] == 128
    assert fitted["primary_or_weak_labels_loaded"] is False
    outputs = fusion.export(path)
    assert len(outputs) == 2
    root = Path(spec["output"])
    first = np.load(root / "representations/public48_visual16/annual.npy")
    second = np.load(root / "representations/public32_visual32/annual.npy")
    np.testing.assert_array_equal(first[:, :32], second[:, :32])
    assert first.shape == second.shape == (2, 64, 8, 8)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert fusion.fit(path) == fitted
    assert fusion.export(path) == outputs
    assert not list(root.rglob("*.partial.npy"))


def test_public_parameters_are_identical_with_entire_visual_modality_missing(tmp_path):
    import copy
    from pathlib import Path

    from xuannv_embedding.export.context import dump

    path, spec = fixture_spec(tmp_path)
    fusion.fit(path)
    other = copy.deepcopy(spec)
    mask = tmp_path / "missing.npy"
    np.save(mask, np.zeros((2, 1, 8, 8), np.float32))
    other["views"][0]["coverage"].update(path=str(mask), sha256=fusion.sha(mask))
    other["output"] = str(tmp_path / "missing-output")
    missing_path = tmp_path / "missing.json"
    dump(missing_path, other)
    result = fusion.fit(missing_path)
    assert result["samples"]["visual"]["rows"] == 0
    first, second = fusion.load_bundle(Path(spec["output"])), fusion.load_bundle(
        Path(other["output"])
    )
    for key in first["public"]:
        np.testing.assert_array_equal(first["public"][key], second["public"][key])
    fusion.export(missing_path)
    values = np.load(Path(other["output"]) / "representations/public48_visual16/annual.npy")
    assert not values[:, 48:].any()


def test_changed_source_and_geography_are_rejected(tmp_path):
    from pathlib import Path

    from xuannv_embedding.export.context import dump

    path, spec = fixture_spec(tmp_path)
    geometry = Path(spec["views"][0]["geometry"]["path"])
    payload = fusion.read_json(geometry)
    payload["records"][0]["bounds"][0] += 1
    dump(geometry, payload)
    with pytest.raises(ValueError, match="identity"):
        fusion.SourceData(*fusion.specification(path))
    spec["views"][0]["geometry"]["sha256"] = fusion.sha(geometry)
    dump(path, spec)
    with pytest.raises(ValueError, match="geographic"):
        fusion.SourceData(*fusion.specification(path))


def test_changed_parameters_and_output_array_are_rejected(tmp_path):
    from pathlib import Path

    path, spec = fixture_spec(tmp_path)
    fusion.fit(path)
    fusion.export(path)
    root = Path(spec["output"])
    output = root / "representations/public48_visual16/annual.npy"
    values = np.load(output)
    values[0, 0, 0, 0] += 1
    np.save(output, values)
    with pytest.raises(ValueError, match="export changed"):
        fusion.export(path)
    (root / "projection.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="parameters"):
        fusion.load_bundle(root)


def test_constant_observed_projection_has_finite_zero_output():
    rows = np.ones((64, 64), np.float32)
    model = fusion.fit_projection(rows, dimensions=32, seed=41, clip=6.0, solver="randomized")
    assert all(np.isfinite(x).all() for x in model.values())
    assert not fusion.project(np.ones((64, 2, 2)), model, dimensions=32, clip=6.0).any()


@pytest.mark.parametrize("section,value", [("public", None), ("index", None)])
def test_malformed_nested_specification_has_explicit_error(tmp_path, section, value):
    _, spec = fixture_spec(tmp_path)
    spec[section] = value
    with pytest.raises(ValueError, match="specification"):
        fusion.validate_specification(spec)


def test_cached_export_rejects_mismatched_model_metadata(tmp_path):
    from pathlib import Path

    from xuannv_embedding.export.context import dump

    path, spec = fixture_spec(tmp_path)
    fusion.fit(path)
    fusion.export(path)
    status_path = Path(spec["output"]) / "representations/public48_visual16/status.json"
    status = fusion.read_json(status_path)
    status["model_sha256"] = "0" * 64
    dump(status_path, status)
    with pytest.raises(ValueError, match="export changed"):
        fusion.export(path)


def test_sampling_rejects_nonfinite_mask():
    with pytest.raises(ValueError, match="mask"):
        fusion.sample_rows([np.ones((2, 3, 4))], [np.full((1, 3, 4), np.nan)], limit=12, seed=41)
