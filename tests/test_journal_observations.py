import numpy as np
import torch

from xuannv_embedding.export.journal_observations import frozen_projection


def test_frozen_projection_uses_original_basis_and_masks_absent_observations():
    native = np.ones((1, 1024, 1, 1), np.float32)
    native[:, 0] = 3
    projection = {
        "mean": np.ones(1024, np.float32),
        "components": np.eye(64, 1024, dtype=np.float32),
    }
    visible = frozen_projection(
        native, np.ones((1, 1, 1, 1), np.float32), projection, (16, 16), (2, 2)
    )
    assert visible.shape == (1, 64, 2, 2)
    assert torch.equal(visible[:, 0], torch.ones((1, 2, 2)))
    assert visible[:, 1:].count_nonzero() == 0
    absent = frozen_projection(
        native, np.zeros((1, 1, 1, 1), np.float32), projection, (16, 16), (2, 2)
    )
    assert absent.count_nonzero() == 0
    assert native[0, 0, 0, 0] == 3
