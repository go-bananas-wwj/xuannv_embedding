import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.export import rgb_guided_refinement as legacy
from xuannv_embedding.export import viewwise_guided_refinement as viewwise


class GuideModel(torch.nn.Module):
    def forward(self, image, features, output_size=None, q_chunk_size=None):
        weight = image.mean((1, 2, 3)).sigmoid()[:, None, None, None]
        return features * weight + features.roll(1, -1) * (1 - weight)


def frame(value, coverage=1):
    return torch.full((2, 75, 4, 4), float(value)), torch.full((2, 25, 4, 4), float(coverage))


def norm():
    return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "RGB_divisor": 255}


def test_viewwise_mean_applies_operator_before_averaging_and_counts_missing_views():
    base = torch.randn(2, 64, 4, 4)
    frames = [frame(10), frame(200, 0.5), frame(30, 0)]
    model = GuideModel()
    result = viewwise.refine_views(base, iter(frames), model, norm(), 256, 0.75, 3)
    expected = torch.zeros_like(base)
    for rgb, coverage in frames:
        guide, c = legacy.reconstruct_guide(rgb, coverage, norm())
        expected += c * (model(guide, base) - base) / 3
    torch.testing.assert_close(result, base + 0.75 * expected, rtol=0, atol=0)
    averaged_rgb = (frames[0][0] + 0.5 * frames[1][0]) / 1.5
    guide, c = legacy.reconstruct_guide(averaged_rgb, torch.full_like(frames[0][1], 0.5), norm())
    wrong_order = legacy.blend(base, model(guide, base), c, 0.75)
    assert not torch.allclose(result, wrong_order)


def test_single_view_preserves_exact_legacy_arithmetic_with_fractional_coverage():
    base = torch.randn(2, 64, 4, 4)
    rgb, coverage = frame(80, 0.37)
    model = GuideModel()
    guide, c = legacy.reconstruct_guide(rgb, coverage, norm())
    expected = legacy.blend(base, model(guide, base), c, 0.75)
    actual = viewwise.refine_views(base, iter([(rgb, coverage)]), model, norm(), 256, 0.75, 1)
    assert torch.equal(actual, expected)


def test_missing_views_return_base_and_view_count_is_not_silently_changed():
    base = torch.randn(2, 64, 4, 4)
    assert torch.equal(viewwise.refine_views(base, iter([]), None, norm(), 256, 0.75, 0), base)
    actual = viewwise.refine_views(base, iter([frame(10, 0)]), GuideModel(), norm(), 256, 0.75, 1)
    assert torch.equal(actual, base)
    with pytest.raises(ValueError, match="count"):
        viewwise.refine_views(base, iter([frame(10)]), GuideModel(), norm(), 256, 0.75, 2)


@pytest.mark.parametrize("count", [-1, True])
def test_invalid_registered_view_count_is_rejected(count):
    with pytest.raises(ValueError):
        viewwise.refine_views(torch.randn(2, 64, 4, 4), iter([]), None, norm(), 256, 0.75, count)


def test_view_batch_cannot_broadcast_across_registered_tiles():
    image, coverage = frame(10)
    frames = [(image[:1], coverage[:1])] * 2
    with pytest.raises(ValueError, match="geometry"):
        viewwise.refine_views(
            torch.randn(2, 64, 4, 4), iter(frames), GuideModel(), norm(), 256, 0.75, 2
        )


def test_nonfinite_refinement_is_rejected():
    class InvalidModel(torch.nn.Module):
        def forward(self, image, features, **kwargs):
            return features * float("nan")

    with pytest.raises(ValueError):
        viewwise.refine_views(
            torch.ones(2, 64, 4, 4), iter([frame(10)]), InvalidModel(), norm(), 256, 0.75, 1
        )


def specification(tmp_path):
    from test_rgb_guided_refinement import spec

    value = copy.deepcopy(spec(tmp_path))
    value.update(
        schema="viewwise-rgb-guided-refinement-v1",
        rgb_sources={"path": str(tmp_path / "raw.json"), "sha256": "a" * 64},
        canonical_export=None,
    )
    return value


@pytest.mark.parametrize("change", ["extra", "source_ref", "canonical", "strength"])
def test_strict_specification_rejects_implicit_or_unknown_inputs(tmp_path, change):
    value = specification(tmp_path)
    assert viewwise.validate_specification(value) == value
    if change == "extra":
        value["_base_"] = "x"
    elif change == "source_ref":
        value["rgb_sources"]["path"] = "relative"
    elif change == "canonical":
        value["canonical_export"] = {"case": "../x"}
    else:
        value["budgets"][0]["strength"] = float("nan")
    with pytest.raises(ValueError):
        viewwise.validate_specification(value)


def test_canonical_recipe_rejects_changed_model_even_when_shapes_match(tmp_path):
    value = specification(tmp_path)
    old = {k: v for k, v in value.items() if k not in {"rgb_sources", "canonical_export"}}
    old["schema"] = "rgb-guided-refinement-v1"
    old["upsampler"] = {
        **old["upsampler"],
        "checkpoint": {"path": "/tmp/other", "sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="canonical"):
        viewwise.check_canonical_recipe(value, old)


def test_duplicate_keys_are_rejected(tmp_path):
    path = Path(tmp_path) / "spec.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError):
        viewwise.specification(path)


def export_fixture(tmp_path):
    from test_rgb_guided_refinement import bundle, spec
    from test_subpixel_rgb import cache_fixture

    from xuannv_embedding.data_process import subpixel_rgb as rgb
    from xuannv_embedding.export.context import dump, sha

    raw_path, raw = cache_fixture(tmp_path)
    rgb.prepare(raw_path)
    source, weights = bundle(tmp_path)
    value = spec(tmp_path)
    value.update(
        index=raw["index"],
        features=raw["public"]["array"],
        rgb_cache={
            "path": str(Path(raw["output"]) / "manifest.json"),
            "sha256": sha(Path(raw["output"]) / "manifest.json"),
        },
        output=str(tmp_path / "old"),
    )
    value["upsampler"].update(source_bundle=source, checkpoint=weights)
    old_path = tmp_path / "old.json"
    dump(old_path, value)
    legacy.export(old_path, batch_size=2)

    def reference(path):
        return {"path": str(path), "sha256": sha(path)}

    result = {
        **value,
        "schema": "viewwise-rgb-guided-refinement-v1",
        "rgb_sources": reference(raw_path),
        "output": str(tmp_path / "new"),
        "canonical_export": {
            "spec": reference(old_path),
            "case": "guided025",
            "status": reference(tmp_path / "old/representations/guided025/status.json"),
            "implementation": {
                "refinement": reference(legacy.__file__),
                "rgb": reference(rgb.__file__),
            },
        },
    }
    path = tmp_path / "new.json"
    dump(path, result)
    return path, result


def test_singleton_all_tiles_are_checked_before_linking_and_resume_is_verified(tmp_path):
    path, spec = export_fixture(tmp_path)
    result = viewwise.export(path, batch_size=2)
    root = Path(spec["output"])
    canonical = tmp_path / "old/representations/guided025/annual.npy"
    current = root / "representations/guided025/annual.npy"
    assert current.stat().st_ino == canonical.stat().st_ino
    assert result[0]["equivalence"]
    assert result[0]["view_count"] == 1
    assert viewwise.export(path, batch_size=2) == result


def test_fresh_single_view_stream_matches_original_without_canonical_alias(tmp_path):
    from xuannv_embedding.export.context import dump

    path, spec = export_fixture(tmp_path)
    spec["canonical_export"] = None
    dump(path, spec)
    result = viewwise.export(path, batch_size=2)
    assert result[0]["view_count"] == 1 and "canonical_array" not in result[0]
    expected = np.load(tmp_path / "old/representations/guided025/annual.npy")
    actual = np.load(Path(spec["output"]) / "representations/guided025/annual.npy")
    np.testing.assert_array_equal(actual, expected)


def test_cached_head_equivalence_cannot_follow_a_changed_source(tmp_path):
    path, spec = export_fixture(tmp_path)
    data = viewwise.SourceData(spec)
    original = data.batch_views

    def changed(positions):
        for image, coverage in original(positions):
            image = image.clone()
            image[:, :, 0, 0] += 1
            yield image, coverage

    data.batch_views = changed
    root = Path(spec["output"])
    root.mkdir()
    with pytest.raises(ValueError, match="source differs"):
        viewwise.singleton_equivalence(spec, data, viewwise.identity(path, 2), root)
    assert not (root / "singleton-equivalence.json").exists()


def test_frozen_canonical_receipt_cannot_be_silently_rewritten(tmp_path):
    path, spec = export_fixture(tmp_path)
    status = Path(spec["canonical_export"]["status"]["path"])
    status.write_text(status.read_text() + " ")
    with pytest.raises(ValueError):
        viewwise.export(path, batch_size=2)


def test_completed_alias_rejects_changed_equivalence_evidence(tmp_path):
    from xuannv_embedding.export.context import dump

    path, spec = export_fixture(tmp_path)
    viewwise.export(path, batch_size=2)
    proof_path = Path(spec["output"]) / "singleton-equivalence.json"
    proof = viewwise.shared.read_json(proof_path)
    proof["source_input_digest"] = "0" * 64
    dump(proof_path, proof)
    with pytest.raises(ValueError, match="equivalence"):
        viewwise.export(path, batch_size=2)
