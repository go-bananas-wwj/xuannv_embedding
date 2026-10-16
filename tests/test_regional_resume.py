import argparse
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xuannv_embedding.config import InputSourceConfig
from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.masking import InputMaskingConfig
from xuannv_embedding.training.regional_base import run_training
from xuannv_embedding.training.regional_resume import validate_resume_checkpoint
from xuannv_embedding.training.runtime import TrainingSystem


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.encoder = nn.Conv2d(1, 2, 1)
        self.base.num_months = 2
        self.base.monthly_embed = SimpleNamespace()
        self.dropout = nn.Dropout(0.3)

    def forward(self, frames, masks, timestamps, *unused):
        x = frames["x"]
        b, t, c, h, w = x.shape
        z = self.dropout(self.base.encoder(x.reshape(b * t, c, h, w)))
        z = z.reshape(b, t, 2, h, w)
        return AEFOutput(z, z.mean((3, 4)), {})


class SmallObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 1)
        self.weight = 0.25

    def set_epoch(self, step):
        pass

    def forward(self, output, *unused):
        return {"total": self.projection(output.embedding_map.mean((1, 3, 4))).square().mean()}


def fixture(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    records = []
    for i in range(6):
        path = tmp_path / f"sample{i}.pt"
        torch.save(
            {
                "patch_id": f"p{i}",
                "region": "test",
                "source_frames": {"x": torch.randn(2, 1, 2, 2)},
                "source_masks": {"x": torch.ones(2)},
                "timestamps": torch.tensor([202401, 202402]),
                "highres_frames": {},
                "highres_masks": {},
                "targets": {},
                "target_masks": {},
                "supervised_labels": {},
                "supervised_label_masks": {},
            },
            path,
        )
        records.append({"path": str(path)})
    config = SimpleNamespace(
        model=SimpleNamespace(input_sources={"x": InputSourceConfig(channels=1, role="temporal")}),
        training=SimpleNamespace(
            input_masking=InputMaskingConfig(enabled=True, modality_dropout_probs={"x": 0.2})
        ),
        data=SimpleNamespace(datasets=[SimpleNamespace(region="test")]),
    )
    cache = {"months": ["2024-01", "2024-02"], "records": records}
    return config, cache


def execute(tmp_path, output, config, cache, *, stop=None, resume=None):
    torch.manual_seed(41)
    system = TrainingSystem(SmallModel(), SmallObjective())
    output.mkdir(exist_ok=True)
    spec = {
        "config": {"sha256": "a" * 64},
        "cache": {"sha256": "b" * 64},
        "output": str(output),
        "training": {
            "seed": 41,
            "steps": 4,
            "head_steps": 0,
            "warmup_steps": 1,
            "effective_batch": 48,
            "rates": {"public": 0.001, "highres": 0.001, "heads": 0.001},
            "weight_decay": 0.05,
            "min_free_gib": 0,
            "save_every": 1,
        },
    }
    path = output / "spec.json"
    path.write_text(json.dumps(spec))
    args = argparse.Namespace(spec=path, micro_batch=24, resume=resume, stop_after_updates=stop)
    run_training(system, spec, config, cache, args, torch.device("cpu"))


def test_interrupted_regional_training_restores_optimizer_sampling_and_dropout(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XUANNV_GIT_SHA", "c" * 40)
    config, cache = fixture(tmp_path)
    execute(tmp_path, tmp_path / "full", config, cache)
    execute(tmp_path, tmp_path / "resumed", config, cache, stop=2)
    assert not (tmp_path / "resumed/final.pt").exists()
    execute(tmp_path, tmp_path / "resumed", config, cache, resume=tmp_path / "resumed/latest.pt")
    full = torch.load(tmp_path / "full/final.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/final.pt", weights_only=True)
    for component in ["model", "criterion"]:
        for key in full[component]:
            assert torch.equal(full[component][key], resumed[component][key]), key
    for key, values in full["optimizer"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, resumed["optimizer"]["state"][key][name])
    assert resumed["epoch"] == 3
    assert resumed["metrics"]["regional_resume"]["recovery_mode"] == "exact"


def test_legacy_resume_keeps_optimizer_progress_and_reports_missing_random_state(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XUANNV_GIT_SHA", "c" * 40)
    config, cache = fixture(tmp_path)
    output = tmp_path / "legacy"
    execute(tmp_path, output, config, cache, stop=2)
    path = output / "latest.pt"
    state = torch.load(path, weights_only=True)
    del state["metrics"]["regional_resume"]
    torch.save(state, path)
    execute(tmp_path, output, config, cache, resume=path)
    state = torch.load(output / "final.pt", weights_only=True)
    assert state["epoch"] == 3
    assert state["metrics"]["regional_resume"]["recovery_mode"] == "legacy_rng_reinitialized"
    assert all(v["step"].item() == 4 for v in state["optimizer"]["state"].values())


def test_resume_checkpoint_rejects_nonfinite_weights_and_changed_configuration(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XUANNV_GIT_SHA", "c" * 40)
    config, cache = fixture(tmp_path)
    execute(tmp_path, tmp_path / "run", config, cache, stop=2)
    path = tmp_path / "run/latest.pt"
    source = {k: asdict(v) for k, v in config.model.input_sources.items()}
    assert (
        validate_resume_checkpoint(
            path, config_sha256="a" * 64, sources=source, regions=["test"], max_steps=4
        )
        == 2
    )
    with pytest.raises(ValueError):
        validate_resume_checkpoint(
            path, config_sha256="d" * 64, sources=source, regions=["test"], max_steps=4
        )
    state = torch.load(path, weights_only=True)
    next(iter(state["model"].values())).fill_(float("nan"))
    torch.save(state, path)
    with pytest.raises(ValueError, match="finite"):
        validate_resume_checkpoint(
            path, config_sha256="a" * 64, sources=source, regions=["test"], max_steps=4
        )
