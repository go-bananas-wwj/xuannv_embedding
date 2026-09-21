"""Annual embedding export with native georeferencing and explicit support maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.export.embedding import _atomic_npz
from xuannv_embedding.training.annual import prepare_annual_batch
from xuannv_embedding.training.runtime import _move


def export_annual_batches(model, batches, output_root, *, config, device):
    root = Path(output_root)
    device = torch.device(device)
    model.to(device).eval()
    written = []
    with torch.inference_mode():
        for raw in batches:
            batch = _move(prepare_annual_batch(raw, config, training=False), device)
            output = model(
                batch["source_frames"],
                batch["source_masks"],
                batch["timestamps"],
                source_pixel_masks=batch["source_pixel_masks"],
                highres_observations=batch["highres_observations"],
                decode=False,
            )
            for index, metadata in enumerate(raw["metadata"]):
                patch_id = metadata["patch_id"]
                if Path(patch_id).name != patch_id or patch_id in {".", ".."}:
                    raise ValueError("Unsafe annual patch_id")
                grid = metadata["output_grid"]
                embedding = output.embedding_map[index, 0].float().cpu().numpy()
                valid = output.validity_mask[index, 0, 0].cpu().numpy().astype(bool)
                if embedding.shape != (64, *grid["shape"]) or grid["spacing_m"] != 5:
                    raise ValueError("Annual output disagrees with the 5 m product grid")
                if not np.isfinite(embedding).all():
                    raise FloatingPointError("Nonfinite annual embedding")
                arrays = {
                    "embedding": np.where(valid[None], embedding, 0),
                    "validity_mask": valid,
                    "year": np.array(metadata["year"], dtype=np.int32),
                    "transform": np.array(grid["transform"], dtype=np.float64),
                    "crs": np.array(grid["crs"]),
                    "spacing_m": np.array(5),
                }
                for name in ("ms5m", "pan2m"):
                    arrays[name + "_weighted_support"] = (
                        output.support[name][index, 0].float().cpu().numpy()
                    )
                path = root / f"{patch_id}_{metadata['year']}.npz"
                if path.exists():
                    raise FileExistsError(path)
                _atomic_npz(path, **arrays)
                written.append(path)
    return written
