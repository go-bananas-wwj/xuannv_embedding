import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from test_multitask_features import setup_exports

from xuannv_embedding.downstream.multitask_features import FeatureSelection, read_features
from xuannv_embedding.downstream.temporal_mean import export


def test_export_mean_uses_every_month_and_does_not_read_held_out_inputs(tmp_path):
    source = setup_exports(tmp_path)
    model = {k: str(v) if isinstance(v, Path) else v for k, v in source.items()}
    model["selection"] = asdict(source["selection"])
    out = tmp_path / "mean"
    spec = {
        "protocol": "temporal-mean-export-v1",
        "source": model,
        "splits": ["train", "validation"],
        "output": str(out),
    }
    path = tmp_path / "mean_spec.json"
    path.write_text(json.dumps(spec))
    result = export(path)
    assert result["state"] == "complete" and result["labels_read"] is False
    selection = FeatureSelection("temporal_mean", "2026-04/2026-05", "2026-05", 3)
    batch = read_features(
        out / "manifest.json",
        source["cache_path"],
        manifest_sha256=result["manifest_sha256"],
        cache_sha256=source["cache_sha256"],
        tile_sha256=result["tile_sha256"],
        selection=selection,
    )
    np.testing.assert_array_equal(batch.values[0], np.full((2, 2, 3), 0.75))
    np.testing.assert_array_equal(batch.values[1], np.full((2, 2, 3), 1.75))
    assert batch.identity["temporal_resolution"] == "period_mean"
