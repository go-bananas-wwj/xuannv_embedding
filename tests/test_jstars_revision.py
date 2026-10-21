import numpy as np
import pytest

from xuannv_embedding.export.jstars_revision import retain_observation


def test_retention_preserves_original_gaps_and_nested_observed_counts():
    image = np.ones((3, 8, 8), np.float32) * 17
    mask = np.ones((1, 8, 8), np.float32)
    mask[:, :2, :2] = 0
    previous = np.zeros_like(mask)
    for fraction, expected in [(0, 0), (0.25, 15), (0.5, 30), (1, 60)]:
        result, kept = retain_observation(image, mask, fraction, "tile-7", 41)
        assert np.count_nonzero(kept) == expected
        assert (kept >= previous).all()
        assert not kept[:, :2, :2].any()
        assert (result[:, kept[0] == 0] == 0).all()
        previous = kept
    assert image.sum() == 3264
    assert mask.sum() == 60


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), True])
def test_retention_rejects_unregistered_fraction(fraction):
    with pytest.raises(ValueError):
        retain_observation(np.ones((3, 4, 4)), np.ones((1, 4, 4)), fraction, "a", 41)


def test_observation_selection_hides_values_and_keeps_registered_view_count():
    from xuannv_embedding.export.jstars_revision import select_observations

    frames = [(np.full((3, 4, 4), i + 1.0), np.ones((1, 4, 4))) for i in range(6)]
    selected = list(select_observations(frames, [0, 2, 4], 1, "tile", 41))
    assert len(selected) == 6
    assert [bool(m.any()) for _, m in selected] == [True, False, True, False, True, False]
    assert selected[1][0].sum() == 0
    assert frames[1][0].sum() == 96


def test_frozen_intervention_uses_saved_heads_and_thresholds_on_changed_features(
    tmp_path, monkeypatch
):
    from xuannv_embedding.downstream import full_region_readouts as full
    from xuannv_embedding.export.context import dump
    from xuannv_embedding.export.jstars_revision import score_frozen
    from xuannv_embedding.training.shared_fusion_search import ref

    rng = np.random.default_rng(41)
    values = rng.normal(size=(23, 8, 8, 64)).astype("float32")
    np.save(tmp_path / "baseline.npy", values)
    np.save(tmp_path / "changed.npy", -values.transpose(0, 3, 1, 2))
    np.save(tmp_path / "valid.npy", np.ones((23, 8, 8), bool))
    np.savez(tmp_path / "labels.npz", target=(values[..., 0] > 0).astype("int8"))
    records = [{"patch_id": str(i), "bounds": [80 * i, 0, 80 * (i + 1), 80]} for i in range(23)]
    dump(tmp_path / "index.json", {"crs": "EPSG:32650", "records": records})
    data = {
        "state": "complete",
        "protocol": "full-region-data-v1",
        "indices": list(range(23)),
        "records": records,
        "train_positions": list(range(20)),
        "calibration_positions": [20, 21],
        "features": {"original": ref(tmp_path / "baseline.npy")},
        "valid": ref(tmp_path / "valid.npy"),
        "labels": ref(tmp_path / "labels.npz"),
        "size": 8,
        "crs": "EPSG:32650",
        "tasks": {"C": {"reference": ["target"]}, "R": {}, "Q": {}},
    }
    dump(tmp_path / "data.json", data)
    readers = {}
    for head in ["mlp", "conv3x3"]:
        p = tmp_path / f"{head}.json"
        dump(
            p,
            {
                "protocol": full.PROTOCOL,
                "data": ref(tmp_path / "data.json"),
                "model": "original",
                "head": head,
                "budget": 20,
                "seed": 41,
                "device": "cpu",
                "archives": {},
                "output": str(tmp_path / head),
            },
        )
        full.run(p, "calibration")
        readers[head] = ref(p)
    monkeypatch.setattr(full, "_fit", lambda *a, **k: pytest.fail("Must not refit"))
    spec = tmp_path / "score.json"
    dump(
        spec,
        {
            "schema": "jstars-frozen-intervention-v1",
            "region": {
                "name": "region_a",
                "index": ref(tmp_path / "index.json"),
                "evaluation": ref(tmp_path / "data.json"),
                "task_source": "reference",
                "support_seed": 41,
            },
            "feature": ref(tmp_path / "changed.npy"),
            "readers": readers,
            "output": str(tmp_path / "score"),
        },
    )
    result = score_frozen(spec, "cpu")
    assert result["heads_refitted"] is False
    assert result["thresholds_refitted"] is False
    assert len(result["rows"]) == 2
    assert all(r["metrics"]["observations"] == 23 * 64 for r in result["rows"])
    assert all(r["outside_support_and_calibration"]["observations"] == 64 for r in result["rows"])
    assert score_frozen(spec, "cpu") == result


def test_no_guide_export_preserves_base_and_records_its_actual_scope(tmp_path):
    from test_viewwise_guided_refinement import export_fixture

    from xuannv_embedding.export.context import dump
    from xuannv_embedding.export.jstars_revision import export_guidance
    from xuannv_embedding.training.shared_fusion_search import ref

    guide, original = export_fixture(tmp_path)
    spec = tmp_path / "intervention.json"
    dump(
        spec,
        {
            "schema": "jstars-guidance-intervention-v1",
            "guide": ref(guide),
            "mode": "no_guide",
            "fraction": 1,
            "slots": [0],
            "seed": 41,
            "output": str(tmp_path / "intervention"),
        },
    )
    result = export_guidance(spec, "cpu")
    np.testing.assert_array_equal(
        np.load(result["features"]["path"]),
        np.load(original["features"]["path"]).astype(np.float16),
    )
    assert result["intervention_scope"] == "terminal_image_guidance"
    assert result["upstream_local_information_retained"] is True
    assert export_guidance(spec, "cpu") == result
