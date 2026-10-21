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


def test_same_scene_degradation_removes_subcell_texture_and_preserves_coverage():
    from xuannv_embedding.export.journal_observations import degrade_observation

    image = np.tile(np.array([[0, 200], [200, 0]], np.float32), (3, 2, 2))
    mask = np.ones((1, 4, 4), np.float32)
    values, coverage = degrade_observation(image, mask, 5.0, 10.0)
    np.testing.assert_array_equal(values, np.full((3, 4, 4), 100, np.float32))
    np.testing.assert_array_equal(coverage, mask)


def test_two_metre_working_grid_can_be_degraded_to_ten_metres():
    from xuannv_embedding.export.journal_observations import degrade_observation

    image = np.repeat(np.arange(100, dtype=np.float32).reshape(1, 10, 10), 3, 0)
    values, mask = degrade_observation(image, np.ones((1, 10, 10), np.float32), 2, 10)
    assert values[0, 0, 0] == 22
    assert values[0, -1, -1] == 77
    assert mask.min() == 1
