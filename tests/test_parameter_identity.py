"""Cross-process sensor identity and semantic optimizer restoration regression tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

from xuannv_embedding.models.annual import Annual5mModel
from xuannv_embedding.training.checkpoint import (
    CheckpointError,
    load_training_checkpoint,
    save_training_checkpoint,
)
from xuannv_embedding.training.distributed_layout import validate_distributed_layout

PAN = ("pan_d", "pan_a", "pan_c", "pan_b")
IDENTITY = {
    "config_sha256": "a" * 64,
    "git_sha": "1234567890abcdef",
    "source_schema": {"s2": {"channels": 2, "role": "temporal"}},
    "regions": ["test"],
}
EXPECTED = {f"expected_{key}": value for key, value in IDENTITY.items() if key != "git_sha"}


def small_annual():
    return Annual5mModel(
        {"s2": 2, **{name: 1 for name in PAN}},
        {"s2": "temporal", **{name: "highres" for name in PAN}},
        PAN,
        {"s2_recon": "s2"},
        embed_dim=4,
        feature_dim=8,
        stem_dim=8,
        stp={},
        backbone="conv",
    )


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_named_pan_weights_preserve_forward_when_registration_order_changes():
    torch.manual_seed(17)
    model = small_annual()
    batch = {
        "source_frames": {"s2": torch.randn(1, 12, 2, 8, 8)},
        "source_masks": {"s2": torch.ones(1, 12)},
        "source_pixel_masks": {"s2": torch.ones(1, 12, 8, 8)},
        "timestamps": torch.arange(202001, 202013)[None],
        "highres_observations": [
            [
                {
                    "source": name,
                    "values": torch.randn(1, 16, 16),
                    "mask": torch.ones(16, 16),
                    "quality_weight": 1.0,
                    "date_features": torch.tensor([0.5, 0.25]),
                }
                for name in PAN
            ]
        ],
        "decode": False,
    }
    original = {key: value.clone() for key, value in model.state_dict().items()}
    for training in (False, True):
        model.train(training)
        torch.manual_seed(42)
        expected = model(**batch).embedding_map
        model.highres_projections = nn.ModuleDict(reversed(list(model.highres_projections.items())))
        model.load_state_dict(original, strict=True)
        torch.manual_seed(42)
        actual = model(**batch).embedding_map
        assert torch.equal(actual, expected)


class ParameterProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = small_annual()

    def forward(self):
        return sum(parameter.square().mean() for parameter in self.parameters())


def layout_worker():
    directory, rank, reject = Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "reject"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        module = ParameterProbe()
        if reject and rank == 1:
            module.model.highres_projections = nn.ModuleDict(
                reversed(list(module.model.highres_projections.items()))
            )
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                parameter.fill_(int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) / 2**24)
        before = tensor_digest(module)
        error = None
        try:
            fingerprint = validate_distributed_layout(module)
        except ValueError as exc:
            assert reject
            fingerprint, error = None, str(exc)
        if not reject:
            wrapped = nn.parallel.DistributedDataParallel(module, broadcast_buffers=False)
            assert tensor_digest(module) == before
            optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)
            wrapped().backward()
            optimizer.step()
        else:
            assert error is not None and tensor_digest(module) == before
        (directory / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "before": before,
                    "after": tensor_digest(module),
                    "error": error,
                    "pan_order": list(module.model.highres_projections),
                    "python_hash_seed": os.environ["PYTHONHASHSEED"],
                }
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("reject", [False, True])
def test_ddp_preserves_named_sensor_identity_across_hash_seeds(tmp_path, reject):
    processes = []
    logs = []
    try:
        for rank, seed in enumerate(("11", "29")):
            log = (tmp_path / f"worker{rank}.log").open("w+")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        str(tmp_path),
                        str(rank),
                        "reject" if reject else "accept",
                    ],
                    env={**os.environ, "PYTHONHASHSEED": seed, "OMP_NUM_THREADS": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for process, log in zip(processes, logs, strict=True):
            code = process.wait(timeout=90)
            log.seek(0)
            assert code == 0, log.read()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()
    rows = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert rows[0]["before"] == rows[1]["before"]
    if reject:
        assert all("broadcast rejected" in row["error"] for row in rows)
    else:
        assert rows[0]["pan_order"] == rows[1]["pan_order"] == list(PAN)
        assert rows[0]["fingerprint"] == rows[1]["fingerprint"]
        assert rows[0]["after"] == rows[1]["after"] != rows[0]["before"]


def sensors(reverse=False):
    items = [(name, nn.Linear(2, 2)) for name in ("sensor_a", "sensor_b")]
    return nn.ModuleDict(reversed(items) if reverse else items)


def update(module, optimizer):
    optimizer.zero_grad(set_to_none=True)
    sum((index + 1) * p.square().sum() for index, p in enumerate(module.parameters())).backward()
    optimizer.step()


def test_optimizer_rejects_same_shape_sensor_slot_swap_before_mutation(tmp_path):
    model = sensors()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    update(model, optimizer)
    path = tmp_path / "model.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=None, epoch=0, **IDENTITY
    )
    reordered = sensors(reverse=True)
    other = torch.optim.AdamW(reordered.parameters(), lr=0.01)
    before = tensor_digest(reordered)
    with pytest.raises(CheckpointError, match="names/order differ"):
        load_training_checkpoint(path, model=reordered, optimizer=other, **EXPECTED)
    assert tensor_digest(reordered) == before and not other.state
    load_training_checkpoint(path, model=reordered, **EXPECTED)
    assert tensor_digest(reordered) == tensor_digest(model)


def test_named_optimizer_resume_matches_uninterrupted_next_update(tmp_path):
    model = sensors()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    update(model, optimizer)
    path = tmp_path / "model.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=None, epoch=0, **IDENTITY
    )
    restored = sensors()
    other = torch.optim.AdamW(restored.parameters(), lr=0.01)
    state = load_training_checkpoint(path, model=restored, optimizer=other, **EXPECTED)
    assert state["optimizer_parameter_names"] == [
        [f"model.{name}" for name, _ in model.named_parameters()]
    ]
    update(model, optimizer)
    update(restored, other)
    assert tensor_digest(model) == tensor_digest(restored)


def test_legacy_multi_pan_optimizer_rejected_but_weights_remain_loadable(tmp_path):
    model = small_annual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "legacy.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=None, epoch=0, **IDENTITY
    )
    state = torch.load(path, weights_only=True)
    del state["optimizer_parameter_names"]
    torch.save(state, path)
    restored = small_annual()
    other = torch.optim.AdamW(restored.parameters(), lr=0.01)
    before = tensor_digest(restored)
    with pytest.raises(CheckpointError, match="Legacy multi-PAN"):
        load_training_checkpoint(path, model=restored, optimizer=other, **EXPECTED)
    assert tensor_digest(restored) == before
    load_training_checkpoint(path, model=restored, **EXPECTED)
    assert tensor_digest(restored) == tensor_digest(model)


def test_complete_system_names_include_frozen_teacher_and_ema(tmp_path):
    system = nn.Module()
    system.model = nn.Linear(2, 2)
    system.criterion = nn.Linear(2, 2)
    system.teacher_model = nn.Linear(2, 2).requires_grad_(False)
    system.ema_model = nn.Linear(2, 2).requires_grad_(False)
    optimizer = torch.optim.AdamW(system.parameters(), lr=0.01)
    path = tmp_path / "teacher.pt"
    with pytest.raises(CheckpointError, match="complete TrainingSystem"):
        save_training_checkpoint(
            path,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=None,
            epoch=0,
            **IDENTITY,
        )
    assert not path.exists()
    save_training_checkpoint(
        path,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        optimizer_named_parameters=system.named_parameters(),
        **IDENTITY,
    )
    state = load_training_checkpoint(
        path,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        optimizer_named_parameters=system.named_parameters(),
        **EXPECTED,
    )
    assert "teacher_model.weight" in state["optimizer_parameter_names"][0]
    assert "ema_model.weight" in state["optimizer_parameter_names"][0]


if __name__ == "__main__":
    layout_worker()
