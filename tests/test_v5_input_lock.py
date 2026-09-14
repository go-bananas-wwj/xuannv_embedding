from types import SimpleNamespace

import pandas as pd
import pytest

from xuannv_embedding.data_process.v5_cli import input_lock
from xuannv_embedding.data_process.v5_sources import sha256


def test_existing_input_lock_rejects_changed_or_missing_dataset_registry(tmp_path):
    args = SimpleNamespace(base_root=tmp_path / "base", dataset_root=tmp_path / "data")
    original = args.base_root / "registry/national_62000.parquet"
    original.parent.mkdir(parents=True)
    pd.DataFrame({"patch_id": ["p1", "p2"], "split": ["train", "test"]}).to_parquet(
        original, index=False
    )
    source = {"revision": "fixed", "manifest_sha256": {"file": "digest"}}
    input_lock(args, source)
    copied = args.dataset_root / "registry/national_62000.parquet"
    assert sha256(copied) == sha256(original)
    input_lock(args, source)
    # A destination-only split edit must not be concealed by an unchanged base lock.
    pd.DataFrame({"patch_id": ["p1", "p2"], "split": ["test", "train"]}).to_parquet(
        copied, index=False
    )
    with pytest.raises(ValueError, match="dataset registry"):
        input_lock(args, source)
    copied.unlink()
    with pytest.raises(ValueError, match="dataset registry"):
        input_lock(args, source)
    assert pd.read_parquet(original).split.tolist() == ["train", "test"]
