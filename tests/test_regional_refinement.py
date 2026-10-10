import argparse
import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from xuannv_embedding.config import InputMaskingConfig, InputSourceConfig
from xuannv_embedding.export.context import sha
from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.regional_refinement import (
    StaticObjective,
    cycle_indices,
    parameter_groups,
    rename_targets,
    validate_scope,
)
from xuannv_embedding.training.runtime import TrainingSystem
from xuannv_embedding.training.static_targets import encode_classes


def test_worldcover_mapping_keeps_all_eleven_classes_and_nodata():
    codes = np.array([[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]], np.uint8)
    np.testing.assert_array_equal(encode_classes(codes), np.arange(12)[None])
    with pytest.raises(ValueError, match="class"):
        encode_classes(np.array([[17]], np.uint8))


class BaseLoss(nn.Module):
    def forward(self, output, *args):
        return {"total": output.embedding_map.square().mean()}


def test_static_objective_is_additive_and_updates_every_month_without_label_input():
    torch.manual_seed(1)
    features = torch.randn(2, 6, 4, 3, 3, requires_grad=True)
    output = AEFOutput(features, features.mean((3, 4)), {})
    objective = StaticObjective(BaseLoss(), 4, "esa_worldcover", 12)
    objective.head_only = False
    objective.weight = 0.25
    targets = {"esa_worldcover": torch.full((2, 3, 3), 11, dtype=torch.long)}
    masks = {"esa_worldcover": torch.ones(2, 3, 3)}
    result = objective(output, targets, masks)
    torch.testing.assert_close(
        result["total"], features.square().mean() + 0.25 * result["recon_esa_worldcover"]
    )
    result["recon_esa_worldcover"].backward()
    assert features.grad.abs().sum((0, 2, 3, 4)).gt(0).all()
    torch.testing.assert_close(features.grad[:, 0], features.grad[:, 5])
    assert objective.static_decoder.weight.grad.abs().sum() > 0


def test_static_objective_nodata_has_zero_loss_and_gradient():
    x = torch.randn(1, 2, 4, 2, 2, requires_grad=True)
    objective = StaticObjective(BaseLoss(), 4, "map", 12)
    result = objective(
        AEFOutput(x, x.mean((3, 4)), {}),
        {"map": torch.zeros(1, 2, 2, dtype=torch.long)},
        {"map": torch.ones(1, 2, 2)},
    )
    result["total"].backward()
    assert result["total"].item() == 0
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_full_training_scope_keeps_downstream_partitions_separate():
    document = {
        "records": [{}] * 4,
        "split": {"train": [0], "validation": [1], "test": [2], "buffer": [3]},
    }
    before = copy.deepcopy(document)
    assert validate_scope(document, [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert document == before
    for indices in [[0, 1, 2], [0, 1, 2, 2], [True, 1, 2, 3]]:
        with pytest.raises(ValueError):
            validate_scope(document, indices)


def test_rank_sampling_covers_every_tile_and_is_resume_deterministic():
    rows = [cycle_indices(320, seed=41, cycle=3, rank=r, world_size=6) for r in range(6)]
    assert set(sum(rows, [])) == set(range(320))
    assert all(len(r) == 54 for r in rows)
    assert rows[2] == cycle_indices(320, seed=41, cycle=3, rank=2, world_size=6)
    assert rows[2] != cycle_indices(320, seed=41, cycle=4, rank=2, world_size=6)


def test_target_rename_preserves_parameters_and_batch_identity():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Module()
            self.base.decoders = nn.ModuleDict({"worldcover": nn.Linear(4, 11)})

    model = Model()
    criterion = nn.Module()
    criterion.target_cfg = {"worldcover": {"weight": 0.45, "loss_type": "ce"}}
    original = model.base.decoders["worldcover"]
    rename_targets(model, criterion, {"worldcover": "osm_landcover"})
    assert model.base.decoders["osm_landcover"] is original
    assert list(criterion.target_cfg) == ["osm_landcover"]
    with pytest.raises(ValueError):
        rename_targets(model, criterion, {"missing": "new"})


def test_joint_groups_update_base_and_adapter_but_keep_original_probe_frozen():
    system = nn.Module()
    system.model = nn.Module()
    system.model.base = nn.Module()
    system.model.base.encoder = nn.Linear(4, 4)
    system.model.base.decoders = nn.ModuleDict({"map": nn.Linear(4, 4)})
    system.model.encoders = nn.ModuleDict({"hr": nn.Linear(4, 4)})
    system.criterion = StaticObjective(nn.Linear(4, 4), 4, "map", 12)
    rates = {"public": 3e-6, "adaptation": 3e-5, "static": 1e-4}
    warm = parameter_groups(system, rates, head_only=True)
    assert len(warm) == 1
    assert not any(p.requires_grad for p in system.model.parameters())
    groups = parameter_groups(system, rates, head_only=False)
    assert {g["name"] for g in groups} == set(rates)
    assert all(p.requires_grad for p in system.model.parameters())
    assert not any(p.requires_grad for p in system.criterion.base.parameters())
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in system.parameters() if p.requires_grad}


@pytest.mark.parametrize("pause", [1, 3])
def test_refinement_resume_and_export_preserve_exact_updates_and_domains(
    tmp_path, monkeypatch, pause
):
    from xuannv_embedding.training import refinement_run as runner

    torch.set_num_threads(1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Module()
            self.base.encoder = nn.Conv2d(4, 4, 1)
            self.adapter = nn.Conv2d(4, 4, 1)
            self.dropout = nn.Dropout(0.2)

        def forward(self, frames, masks, timestamps, *unused):
            assert set(frames) == {"x"}
            x = frames["x"]
            b, t, c, h, w = x.shape
            y = self.dropout(self.base.encoder(x.reshape(b * t, c, h, w)))
            y = self.adapter(y).reshape(b, t, c, h, w)
            return AEFOutput(y, y.mean((3, 4)), {})

    config = SimpleNamespace(
        model=SimpleNamespace(input_sources={"x": InputSourceConfig(channels=4, role="temporal")}),
        training=SimpleNamespace(input_masking=InputMaskingConfig(enabled=False)),
        data=SimpleNamespace(
            datasets=[SimpleNamespace(region="test")], months=["2026-01", "2026-02"]
        ),
    )
    cache = {
        "records": [],
        "data": {"patch_size": 3},
        "model_inputs": {"x": asdict(config.model.input_sources["x"])},
        "split": {"train": [0], "validation": [1], "test": [2], "buffer": [3]},
    }
    labels = {"records": [], "classes": 12, "target": "esa_worldcover"}
    for i in range(4):
        sample = {
            "patch_id": f"p{i}",
            "region": "test",
            "source_frames": {"x": torch.randn(2, 4, 3, 3)},
            "source_masks": {"x": torch.ones(2)},
            "timestamps": torch.tensor([202601, 202602]),
            "targets": {},
            "target_masks": {},
            "highres_frames": {},
            "highres_masks": {},
            "supervised_labels": {},
            "supervised_label_masks": {},
        }
        path = tmp_path / f"p{i}.pt"
        torch.save(sample, path)
        common = {"patch_id": f"p{i}", "bounds": [i, 0, i + 1, 1]}
        cache["records"].append({**common, "path": str(path), "sha256": sha(path)})
        path = tmp_path / f"p{i}.npz"
        np.savez(path, labels=np.full((3, 3), 11, np.uint8), valid=np.ones((3, 3), bool))
        labels["records"].append({**common, "path": str(path), "sha256": sha(path)})
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache))
    spec = {
        "protocol": "regional-static-refinement-v1",
        "cache": {"path": str(cache_path), "sha256": sha(cache_path)},
        "target_aliases": {},
        "training_indices": [0, 1, 2, 3],
        "evaluation_scope": "full-region-map-supervision; downstream-query-labels-seen",
        "training": {
            "seed": 41,
            "head_steps": 1,
            "joint_steps": 4,
            "world_size": 1,
            "micro_batch": 1,
            "accumulation": 2,
            "ramp_steps": 1,
            "amp": False,
            "static_weight": 0.25,
            "weight_decay": 0.05,
            "min_free_gib": 0.001,
            "save_every": 1,
            "learning_rates": {"public": 0.01, "adaptation": 0.01, "static": 0.01},
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(runner, "read_spec", lambda p: (copy.deepcopy(spec), cache, labels))
    monkeypatch.setattr(runner, "_git_sha", lambda: "a" * 40)

    def initialize(*args):
        return TrainingSystem(Model(), StaticObjective(BaseLoss(), 4, "esa_worldcover", 12)), config

    monkeypatch.setattr(runner, "initialize", initialize)
    args = argparse.Namespace(
        spec=path, output=tmp_path / "full", device="cpu", resume=None, stop_after_updates=None
    )
    runner.run(args)
    args.output = tmp_path / "resumed"
    args.stop_after_updates = pause
    runner.run(args)
    args.resume = args.output / "latest.pt"
    args.stop_after_updates = None
    runner.run(args)
    full = torch.load(tmp_path / "full/final.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/final.pt", weights_only=True)
    for component in ["model", "criterion"]:
        for name in full[component]:
            torch.testing.assert_close(
                full[component][name], resumed[component][name], rtol=0, atol=0
            )
    assert resumed["metrics"]["optimizer_steps"] == 5
    assert json.loads((args.output / "status.json").read_text())["unique_training_indices"] == list(
        range(4)
    )
    destination = tmp_path / "export"
    runner.export(
        argparse.Namespace(
            spec=path,
            checkpoint=args.output / "final.pt",
            output=destination,
            device="cpu",
            batch_size=2,
        )
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["split"] == cache["split"]
    assert manifest["representation_training_indices"] == list(range(4))
    with np.load(manifest["records"][0]["path"]) as z:
        assert z["embedding"].shape == (2, 4, 3, 3)
        assert np.isfinite(z["embedding"]).all()
