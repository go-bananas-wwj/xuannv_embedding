import torch

from xuannv_embedding.data_process.checkpoint_retention import compact_checkpoint, update_number


def test_checkpoint_step_uses_recorded_updates_instead_of_name_or_mtime():
    assert update_number({"metrics": {"optimizer_steps": 800}, "epoch": 7}, "best.pt") == 800
    assert update_number({"epoch": 399}, "epoch_9999.pt") == 399
    assert update_number({}, "step0400.pt") == 400
    assert update_number({}, "best.pt") == -1


def test_compact_checkpoint_preserves_every_model_and_auxiliary_tensor(tmp_path):
    source = tmp_path / "latest.pt"
    destination = tmp_path / "retained.pt"
    weights = {"encoder.weight": torch.randn(3, 4), "buffer": torch.arange(5)}
    state = {
        "model": weights,
        "criterion": {"head.weight": torch.randn(2, 3)},
        "optimizer": {"exp_avg": torch.randn(1000)},
        "scheduler": {"step": 800},
        "epoch": 799,
        "config_sha256": "a" * 64,
    }
    torch.save(state, source)
    receipt = compact_checkpoint(source, destination)
    restored = torch.load(destination, weights_only=True)
    assert restored["optimizer"] is None and restored["scheduler"] is None
    assert restored["epoch"] == 799 and restored["config_sha256"] == "a" * 64
    for key, value in weights.items():
        assert torch.equal(value, restored["model"][key])
    assert torch.equal(state["criterion"]["head.weight"], restored["criterion"]["head.weight"])
    assert source.exists() and receipt["model_tensor_digest"] == receipt["verified_digest"]
