import numpy as np
import pytest

from xuannv_embedding.downstream.fixed_reproduction import checked_positions


def test_reproduction_rejects_changed_support_pixels():
    y = np.array([0, 1, -1, 0])
    with pytest.raises(ValueError, match="positions"):
        checked_positions(y, np.array([0, 2]), {"sample_positions_sha256": "wrong"})


def test_reproduction_rejects_changed_labels():
    import hashlib

    p = np.array([0, 1])
    row = {
        "sample_positions_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
        "support_label_sha256": "wrong",
    }
    with pytest.raises(ValueError, match="labels"):
        checked_positions(np.array([0, 1]), p, row)
