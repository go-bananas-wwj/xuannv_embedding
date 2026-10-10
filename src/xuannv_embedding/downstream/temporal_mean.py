"""Export a genuine whole-window mean without labelling it as a monthly observation."""

from pathlib import Path

import numpy as np

from xuannv_embedding.downstream import multitask_features as features
from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.export.context import dump, sha


def export(spec_path):
    spec = primary._load(spec_path)
    if (
        set(spec) != {"protocol", "source", "splits", "output"}
        or spec["protocol"] != "temporal-mean-export-v1"
    ):
        raise ValueError("invalid temporal mean export specification")
    model = spec["source"]
    selection = features.FeatureSelection(**model["selection"])
    if selection.kind != "monthly":
        raise ValueError("temporal mean export requires monthly input")
    path = Path(model["manifest_path"])
    manifest = features._registered_json(path, model["manifest_sha256"], "manifest")
    cache = features._registered_json(Path(model["cache_path"]), model["cache_sha256"], "cache")
    size, _ = features._grid(cache, manifest)
    months = cache["data"]["months"]
    if manifest["months"] != months or manifest["cache_sha256"] != model["cache_sha256"]:
        raise ValueError("mean source window or cache identity differs")
    splits = spec["splits"]
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(cache["split"]):
        raise ValueError("invalid temporal mean partitions")
    indices = sorted(i for split in splits for i in cache["split"][split])
    if manifest.get("exported_indices") and not set(indices) <= set(manifest["exported_indices"]):
        raise ValueError("requested temporal mean inputs are not exported")
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    records = [
        {
            "patch_id": r["patch_id"],
            "bounds": r["bounds"],
            "path": "unexported/" + r["patch_id"] + ".npz",
        }
        for r in cache["records"]
    ]
    digests = {}
    try:
        for i in indices:
            record = manifest["records"][i]
            source = Path(record["path"])
            if not source.is_absolute():
                source = path.parent / source
            if sha(source) != model["tile_sha256"][record["patch_id"]]:
                raise ValueError("monthly source tile digest changed")
            with np.load(source, allow_pickle=False) as z:
                x = z["embedding"]
                if x.shape != (len(months), selection.channels, size, size) or not np.array_equal(
                    z["timestamps"], [int(m.replace("-", "")) for m in months]
                ):
                    raise ValueError("mean source tensor shape or timestamps differ")
                valid = z["valid_mask"] if "valid_mask" in z else np.ones((size, size), bool)
                if valid.dtype != np.bool_ or valid.shape not in [
                    (size, size),
                    (len(months), size, size),
                ]:
                    raise ValueError("invalid source feature validity")
                valid = valid.all(0) if valid.ndim == 3 else valid.copy()
                if not np.isfinite(x[:, :, valid]).all():
                    raise ValueError("nonfinite input on common valid mean domain")
                mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
                mean[:, ~valid] = 0
                if not np.isfinite(mean).all():
                    raise ValueError("nonfinite float32 temporal mean")
            target = out / (record["patch_id"] + ".npz")
            np.savez_compressed(target, embedding=mean[None], valid_mask=valid)
            records[i].update(path=str(target), sha256=sha(target))
            digests[record["patch_id"]] = records[i]["sha256"]
            dump(out / "status.json", {"state": "running", "tiles": len(digests)})
        result = {
            "kind": "temporal_mean",
            "months": [f"mean_{months[0]}_{months[-1]}"],
            "observation_months": months,
            "channels": selection.channels,
            "cache_sha256": model["cache_sha256"],
            "split": cache["split"],
            "records": records,
            "exported_indices": indices,
            "source_manifest_sha256": model["manifest_sha256"],
            "source_checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "aggregation": "equal arithmetic mean; float64 accumulation, float32 storage",
            "validity": "intersection of exported feature validity across all months",
            "normalization": "none here; downstream head preprocessing remains unchanged",
        }
        dump(out / "manifest.json", result)
        proof = {
            "state": "complete",
            "spec_sha256": sha(Path(spec_path)),
            "manifest_sha256": sha(out / "manifest.json"),
            "tile_sha256": digests,
            "implementation_sha256": sha(Path(__file__)),
            "labels_read": False,
        }
        dump(out / "verification.json", proof)
        dump(out / "status.json", {"state": "complete", "tiles": len(digests)})
        return proof
    except BaseException as exc:
        dump(out / "status.json", {"state": "failed", "error": repr(exc)})
        raise
