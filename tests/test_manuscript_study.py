import json

import pytest
import torch

from xuannv_embedding.training.manuscript_study import (
    cyclic_batch,
    learning_rates,
    phase_settings,
    validate_sources,
)


def test_cyclic_sampler_visits_every_tile_and_pairs_roles():
    a = [i for step in range(12) for i in cyclic_batch(17, step, 4)]
    assert sorted(a[:17]) == list(range(17))
    assert a == [i for step in range(12) for i in cyclic_batch(17, step, 4)]
    assert len(cyclic_batch(2, 1, 48)) == 48


def test_registered_stage_budget_and_freeze_transition():
    assert phase_settings("base", 1)["freeze"] is False
    assert phase_settings("highres", 800)["freeze"] is True
    assert phase_settings("highres", 801)["freeze"] is False
    assert phase_settings("highres", 801)["public_lr"] == 1e-5
    assert phase_settings("public", 1)["public_lr"] == 1e-5
    assert learning_rates("highres", 801)["public"] > 0
    with pytest.raises(ValueError):
        phase_settings("base", 2401)


def test_esa_target_rejects_osm_alias_and_evaluation_maps():
    wc = {
        "target": "esa_worldcover",
        "classes": 12,
        "ignore_index": 0,
        "source": {"dataset": "ESA WorldCover", "year": 2021, "version": "v200"},
    }
    osm = {
        "dataset": "OpenStreetMap",
        "unknown_is_background": False,
        "primary_reference_maps_read": False,
    }
    validate_sources(wc, osm)
    wc["source"]["dataset"] = "OpenStreetMap"
    with pytest.raises(ValueError, match="WorldCover"):
        validate_sources(wc, osm)


def test_sources_require_unknown_osm_pixels_to_remain_unknown():
    wc = {
        "target": "esa_worldcover",
        "classes": 12,
        "ignore_index": 0,
        "source": {"dataset": "ESA WorldCover", "year": 2021, "version": "v200"},
    }
    with pytest.raises(ValueError, match="OSM"):
        validate_sources(wc, {"dataset": "OpenStreetMap", "unknown_is_background": True})


def test_checkpoint_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    from xuannv_embedding.models.model import AEFOutput
    from xuannv_embedding.training import manuscript_study as study

    class Data:
        n, meta = 3, {}

        def __init__(self, spec, *, highres):
            pass

        def sample(self, i):
            frames = {s: torch.full((12, 1, 4, 4), 0.2 + 0.1 * i) for s in ["s2", "s1", "landsat"]}
            targets = {s + "_recon": x.clone() for s, x in frames.items()}
            return {
                "source_frames": frames,
                "source_masks": {s: torch.ones(12) for s in frames},
                "timestamps": torch.tensor([202401 + i for i in range(12)]),
                "highres_frames": {},
                "highres_masks": {},
                "targets": targets,
                "target_masks": {k: torch.ones(12, 4, 4) for k in targets},
                "worldcover": torch.ones(4, 4, dtype=torch.long),
                "worldcover_valid": torch.ones(4, 4),
                "osm_area": torch.ones(4, 4, dtype=torch.long),
                "osm_geometry": torch.ones(4, 4, dtype=torch.long),
            }

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = torch.nn.Conv2d(1, 64, 1)
            self.decoders = torch.nn.ModuleDict(
                {s + "_recon": torch.nn.Conv2d(64, 1, 1) for s in ["s2", "s1", "landsat"]}
            )

        def forward(self, frames, masks, times, highres, validity):
            x = frames["s2"]
            b, t, c, h, w = x.shape
            z = self.encode(x.flatten(0, 1)).reshape(b, t, 64, h, w)
            recon = {k: d(z.flatten(0, 1)).reshape(b, t, 1, h, w) for k, d in self.decoders.items()}
            return AEFOutput(z, z.mean((-2, -1)), recon)

    def construct(*args):
        torch.manual_seed(41)
        return Tiny(), study.Objective()

    monkeypatch.setattr(study, "Observations", Data)
    monkeypatch.setattr(study, "build", construct)
    monkeypatch.setattr(study, "specification", lambda p: json.loads(p.read_text()))
    monkeypatch.setenv("XUANNV_GIT_SHA", "a" * 40)
    paths = []
    for name in ["whole", "resumed"]:
        p = tmp_path / (name + ".json")
        p.write_text(
            json.dumps({"output": str(tmp_path / name), "reserve_gib": 0, "phases": {"base": 2400}})
        )
        paths.append(p)
    whole = study.train(paths[0], "base", "cpu", 8, gate_updates=2)
    stopped = study.train(paths[1], "base", "cpu", 8, gate_updates=2, stop_after_updates=1)
    assert stopped["state"] == "checkpointed"
    resumed = study.train(paths[1], "base", "cpu", 8, gate_updates=2)
    a = torch.load(whole["checkpoint"]["path"], weights_only=True)
    b = torch.load(resumed["checkpoint"]["path"], weights_only=True)
    for group in ["model", "criterion"]:
        for key in a[group]:
            torch.testing.assert_close(a[group][key], b[group][key], rtol=0, atol=0)
    assert resumed["step"] == 2
