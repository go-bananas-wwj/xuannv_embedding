from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
import zarr
from torch import nn

from xuannv_embedding.export.v2_sharded import export_v2_sharded


class _Model(nn.Module):
    def forward(self, **inputs):
        intervals = inputs["output_intervals"]
        batch, outputs = intervals.shape[:2]
        return SimpleNamespace(embedding_map=torch.ones(batch, outputs, 4, 3, 3))


def _batch() -> dict[str, object]:
    return {
        "patch_ids": ["p1", "p2"],
        "macro_ids": ["m1", "m2"],
        "splits": ["train", "test"],
        "grid_epsgs": torch.tensor([32649, 32649]),
        "observation_lineage": [
            {"s2_local": ["s2:2020-01"], "s1_local": []},
            {"s2_local": [], "s1_local": ["s1:2020-01"]},
        ],
        "model_inputs": {
            "output_intervals": torch.tensor(
                [
                    [[18262.0, 18293.0], [18628.0, 18659.0]],
                    [[18262.0, 18293.0], [18628.0, 18659.0]],
                ]
            )
        },
    }


def test_v2_export_writes_interval_utm_zarr_and_lineage_catalog(tmp_path: Path) -> None:
    catalog = export_v2_sharded(
        _Model(),
        [_batch()],
        tmp_path,
        device="cpu",
        product_version="v2-test",
        data_manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        git_sha="1234567890abcdef",
        checkpoint_sha256="c" * 64,
        model_state_sha256="d" * 64,
        run_id="smoke-test",
        shard_size=2,
    )

    table = pq.read_table(catalog)
    assert table.num_rows == 4
    assert set(table["interval_id"].to_pylist()) == {
        "20200101-20200201",
        "20210101-20210201",
    }
    relative = table["shard_path"][0].as_py()
    assert relative.startswith("interval=20200101-20200201/utm=32649/")
    group = zarr.open_group(str(catalog.parent / relative), mode="r")
    assert group.attrs["data_manifest_sha256"] == "a" * 64
    assert group.attrs["checkpoint_sha256"] == "c" * 64
    assert group["embedding"].shape == (2, 4, 3, 3)
    assert table["observation_lineage_json"][0].as_py()
    assert table["shard_sha256"][0].as_py()


def test_v2_export_refuses_to_overwrite_catalog(tmp_path: Path) -> None:
    kwargs = {
        "device": "cpu",
        "product_version": "v2-test",
        "data_manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "git_sha": "1234567890abcdef",
        "checkpoint_sha256": "c" * 64,
        "model_state_sha256": "d" * 64,
        "run_id": "smoke-test",
    }
    export_v2_sharded(_Model(), [_batch()], tmp_path, **kwargs)

    try:
        export_v2_sharded(_Model(), [_batch()], tmp_path, **kwargs)
    except FileExistsError:
        pass
    else:
        raise AssertionError("应拒绝覆盖已有 catalog")
