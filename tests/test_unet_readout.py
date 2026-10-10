import numpy as np
import pytest
import torch

from xuannv_embedding.downstream import unet_readout as unet
from xuannv_embedding.export.context import sha


def example():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3, 17, 19)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int8)
    valid = np.ones(y.shape, bool)
    valid[0, 0, 0] = False
    y[1, 0, 1] = -1
    q = rng.normal(size=x.shape).astype(np.float32)
    return x, y, valid, q, (q[:, 0] > 0).astype(np.int8), valid.copy()


def test_unet_snapshot_replays_and_masks_missing_context(tmp_path, monkeypatch):
    args = example()
    model = unet.fit(*args, checkpoints=(1, 2))
    expected = model.predict(args[3], args[5])
    changed = args[3].copy()
    changed.transpose(0, 2, 3, 1)[~args[5]] = 10000
    np.testing.assert_array_equal(expected, model.predict(changed, args[5]))
    assert model.metadata["fitted_pixels"] == int((args[2] & (args[1] >= 0)).sum())
    assert model.metadata["selected_steps"] in (1, 2)
    unet.save(model, tmp_path / "head")
    loaded = unet.load(tmp_path / "head", sha(tmp_path / "head/identity.json"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a: pytest.fail("query refit"))
    np.testing.assert_array_equal(loaded.predict(args[3], args[5]), expected)
    p = tmp_path / "head/parameters.npz"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="payload"):
        unet.load(tmp_path / "head", sha(tmp_path / "head/identity.json"))


def test_unet_rejects_invalid_checkpoints_and_too_small_maps():
    args = example()
    with pytest.raises(ValueError, match="checkpoints"):
        unet.fit(*args, checkpoints=(2, 1))
    small = [a[..., :4, :4] for a in args]
    with pytest.raises(ValueError, match="eight"):
        unet.fit(*small, checkpoints=(1,))
