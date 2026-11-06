"""Shared native-resolution, per-band QA reader for all six high-resolution branches."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from affine import Affine

from xuannv_embedding.data_process.v5_clear_intraband import NativeQualityReader, apply_clear_mask
from xuannv_embedding.data_process.v5_jilin_quality import _array_digest
from xuannv_embedding.data_process.v5_partial_bands import read_jilin_branch
from xuannv_embedding.data_process.v5_quality import transfer_invalid
from xuannv_embedding.data_process.v5_rasters import read_native
from xuannv_embedding.data_process.v5_sources import sha256


class HighresQualityReader:
    def __init__(self, dataset_root: Path, family: str, quality_root: Path):
        self.qa = NativeQualityReader(dataset_root, family, quality_root)
        self.family = family
        self.files = self.qa.files
        self.configuration = self.qa.configuration
        self.registry = self.qa.registry
        if family == "gaofen":
            rows = []
            for row in self.qa.table.reset_index().to_dict("records"):
                for branch in ["ms", "pan"]:
                    rows.append(
                        {
                            **{k: row[k] for k in ["patch_id", "sensor", "year", "split"]},
                            "observation_id": row["pair_id"] + ":gaofen_" + branch,
                            "scene_group_id": row["pair_id"],
                            "product_id": "gaofen_" + branch,
                            "path": row[branch + "_path"],
                            "file_sha256": row[branch + "_sha256"],
                        }
                    )
            self.inventory = pd.DataFrame(rows)
        else:
            paths = [p for p in self.files if p.endswith("/files_with_partial_bands.parquet")]
            if len(paths) != 1:
                raise ValueError("one frozen Jilin source catalog required")
            catalog = pd.read_parquet(paths[0])
            if catalog.observation_id.duplicated().any():
                raise ValueError("duplicate Jilin source observation")
            self.catalog = catalog.set_index("observation_id")
            self.inventory = self.qa.table.reset_index().merge(
                catalog[["observation_id", "path", "file_sha256"]],
                on=["observation_id", "file_sha256"],
                how="left",
                validate="one_to_one",
            )[
                [
                    "observation_id",
                    "scene_group_id",
                    "product_id",
                    "patch_id",
                    "sensor",
                    "year",
                    "split",
                    "path",
                    "file_sha256",
                ]
            ]
        if self.inventory.isna().any().any() or self.inventory.observation_id.duplicated().any():
            raise ValueError("invalid high-resolution QA inventory")
        self.inventory = self.inventory.sort_values("observation_id").reset_index(drop=True)

    def verify_unchanged(self):
        self.qa.verify_unchanged()

    def read(self, row):
        if self.family == "gaofen":
            frame, proof = self._gaofen(row)
        else:
            frame, proof = self._jilin(row)
        # The same convention is consumed by statistics and subsequent data loading.
        frame = replace(frame, values=np.where(frame.valid, frame.values, 0).astype("f4"))
        return frame, proof

    def _gaofen(self, row):
        qa = self.qa.table.loc[row.scene_group_id]
        branch = row.product_id.removeprefix("gaofen_")
        if (
            branch not in ["ms", "pan"]
            or row.path != qa[branch + "_path"]
            or row.file_sha256 != qa[branch + "_sha256"]
        ):
            raise ValueError("Gaofen branch identity changed")
        source = SimpleNamespace(
            observation_id=row.scene_group_id,
            path=qa.ms_path,
            file_sha256=qa.ms_sha256,
            **{k: getattr(row, k) for k in ["patch_id", "sensor", "year", "split"]},
        )
        _, ms, _, _, ms_proof = self.qa.read(source)
        if branch == "ms":
            return ms, {
                **ms_proof,
                "observation_id": row.observation_id,
                "unit": "native_stored_dn",
            }
        if sha256(Path(row.path)) != row.file_sha256:
            raise ValueError("Gaofen PAN source changed")
        pan = read_native(
            Path(row.path),
            ["pan"],
            contract={"verified": True, "band_ids": ["pan"], "scales": [1], "offsets": [0]},
        )
        if (
            pan.values.shape != (1, 640, 640)
            or pan.crs != ms.crs
            or not np.allclose(
                pan.transform, (2, 0, ms.transform[2], 0, -2, ms.transform[5]), rtol=0, atol=1e-9
            )
        ):
            raise ValueError("Gaofen PAN native grid differs from QA contract")
        invalid = transfer_invalid(
            ~ms.valid.all(axis=0),
            src_transform=Affine(*ms.transform),
            src_crs=ms.crs,
            dst_transform=Affine(*pan.transform),
            dst_crs=pan.crs,
            shape=pan.values.shape[1:],
        )
        expected = pan.valid[0] & ~invalid
        i = self.qa.positions[row.scene_group_id]
        mask = np.unpackbits(
            np.asarray(self.qa.masks["pan_valid_packed"][i]), axis=-1, count=640, bitorder="little"
        ).astype(bool)
        if not np.array_equal(expected, mask) or int(mask.sum()) != qa.pan_valid_pixels:
            raise ValueError("Gaofen PAN QA differs from geographic transfer")
        clear = apply_clear_mask(pan, mask[None], pan.band_ids)
        return clear, {
            "observation_id": row.observation_id,
            "file_sha256": row.file_sha256,
            "quality_mask_sha256": _array_digest(mask[None]),
            "source_ms_proof": ms_proof,
            "unit": "native_stored_dn",
        }

    def _jilin(self, row):
        source = self.catalog.loc[row.observation_id].to_dict()
        qa = self.qa.table.loc[row.observation_id]
        for key in [
            "scene_group_id",
            "product_id",
            "patch_id",
            "sensor",
            "year",
            "split",
            "file_sha256",
        ]:
            if source[key] != getattr(row, key) or qa[key] != getattr(row, key):
                raise ValueError("Jilin source and QA identity disagree")
        if source["path"] != row.path:
            raise ValueError("Jilin source path differs from frozen catalog")
        frame = read_jilin_branch(source)
        receipt_path = self.qa.root / "receipts" / f"{row.scene_group_id}.json"
        if sha256(receipt_path) != self.qa.receipts[row.scene_group_id]:
            raise ValueError("Jilin QA receipt changed")
        receipt = json.loads(receipt_path.read_text())
        prefix = row.observation_id + "/"
        arrays = {}
        for name in ["data_valid", "valid", "qa_clear"]:
            array = np.asarray(self.qa.masks[prefix + name])
            if _array_digest(array) != receipt["mask_arrays"][prefix + name]:
                raise ValueError("Jilin QA mask changed")
            arrays[name] = array
        mask = arrays["valid"]
        if (
            not np.array_equal(arrays["data_valid"], frame.valid)
            or not np.array_equal(mask, frame.valid & arrays["qa_clear"][None])
            or not np.array_equal(mask.sum(axis=(1, 2)), qa.valid_pixels_by_band)
            or (qa.quality_status == "qa_missing" and mask.any())
        ):
            raise ValueError("Jilin band validity differs from frozen QA")
        clear = apply_clear_mask(frame, mask, tuple(qa.band_ids))
        return clear, {
            "observation_id": row.observation_id,
            "file_sha256": row.file_sha256,
            "quality_mask_sha256": _array_digest(mask),
            "unit": "reflectance",
            "quality_status": qa.quality_status,
            "band_ids": list(frame.band_ids),
        }
