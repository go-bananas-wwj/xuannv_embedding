import pytest
import torch

from xuannv_embedding.export.jstars_revision import binary_refine_views
from xuannv_embedding.export.viewwise_guided_refinement import refine_views


def test_binary_coverage_changes_weights_and_preserves_empty_cells_at_batch_two():
    base = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2).repeat(2, 64, 1, 1)
    rgb = torch.full((2, 75, 2, 2), 100.0)
    mask = torch.full((2, 25, 2, 2), 0.2)
    mask[:, :, 0, 0] = 0
    norm = {"mean": [0, 0, 0], "std": [1, 1, 1], "RGB_divisor": 255}

    def model(image, values, **kwargs):
        assert len(values) == 2
        return values.flip(-1)

    binary = binary_refine_views(base, [(rgb, mask)], model, norm, 256, 0.75, 1)
    continuous = refine_views(base, [(rgb, mask)], model, norm, 256, 0.75, 1)
    assert torch.equal(binary[:, :, 0, 0], base[:, :, 0, 0])
    assert binary[0, 0, 0, 1].item() == pytest.approx(0.25)
    assert continuous[0, 0, 0, 1].item() == pytest.approx(0.85)
