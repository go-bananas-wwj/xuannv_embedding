from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xuannv_embedding.export.embedding import export_embedding_batches
from xuannv_embedding.training.experiment_export import expand_export_batch, export_groups


def test_selected_groups_preserve_original_slots_without_unselected_inputs():
    groups = list(export_groups(10, [1, 3, 6, 9], 4))
    assert groups == [([1, 3], [0, 0, 0, 1], [1, 3]), ([6], [0, 0, 0, 0], [2]), ([9], [0, 0], [1])]
    batch = {"patch_ids": ["p1", "p3"], "source_frames": {"s": torch.tensor([[1], [3]])}}
    expanded = expand_export_batch(batch, groups[0][1])
    assert expanded["patch_ids"] == ["p1", "p1", "p1", "p3"]
    assert expanded["source_frames"]["s"].tolist() == [[1], [1], [1], [3]]
    assert batch["patch_ids"] == ["p1", "p3"]


def test_export_writes_only_selected_positions_from_a_padded_batch(tmp_path):
    class PositionModel(torch.nn.Module):
        def forward(self, frames, masks, timestamps, highres, highres_masks):
            n = len(timestamps)
            output = torch.arange(n).float().reshape(n, 1, 1, 1, 1)
            return SimpleNamespace(embedding_map=output)

    batch = {
        "patch_ids": ["p1", "p1", "p1", "p3"],
        "source_frames": {},
        "source_masks": {},
        "timestamps": torch.full((4, 1), 202605),
    }
    paths = export_embedding_batches(
        PositionModel(), [batch], tmp_path, device="cpu", output_indices=[1, 3]
    )
    assert [p.name for p in paths] == ["p1.npz", "p3.npz"]
    for path, expected in zip(paths, [1, 3]):
        with np.load(path) as data:
            assert data["embedding"].item() == expected
    with pytest.raises(ValueError):
        export_embedding_batches(
            PositionModel(), [batch], tmp_path / "bad", device="cpu", output_indices=[1, 1]
        )
