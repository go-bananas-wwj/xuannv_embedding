"""Partition frozen extraction and restore one regional native feature tensor before PCA."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from xuannv_embedding.export import semantic_features
from xuannv_embedding.export.context import dump, sha


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def verified(reference):
    if set(reference) != {"path", "sha256"} or sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Semantic shard input identity changed")
    return Path(reference["path"])


def prepare(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if (
        set(spec) != {"schema", "source_spec", "shards", "output"}
        or spec["schema"] != "regional-semantic-shards-v1"
        or type(spec["shards"]) is not int
        or not 1 <= spec["shards"] <= 64
    ):
        raise ValueError("Invalid semantic shard preparation specification")
    source = verified(spec["source_spec"])
    original, cache = semantic_features.read_spec(source)
    if spec["shards"] > len(cache["records"]):
        raise ValueError("Every semantic shard must contain observations")
    root = Path(spec["output"])
    if root.resolve() == Path(original["output"]).resolve():
        raise ValueError("Semantic shard outputs must be distinct from the merged output")
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(spec_path), "implementation_sha256": sha(__file__)}
    header = root / "identity.json"
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Semantic shard preparation identity changed")
    dump(header, identity)
    manifest = (
        json.loads(verified(original["rasters"]).read_text()) if "rasters" in original else None
    )
    shards = []
    for number, positions in enumerate(
        np.array_split(np.arange(len(cache["records"])), spec["shards"])
    ):
        positions = positions.tolist()
        directory = root / f"shard_{number:02d}"
        directory.mkdir(exist_ok=True)
        index = directory / "cache.json"
        dump(
            index,
            {
                "state": "complete",
                "months": cache["months"],
                "records": [cache["records"][i] for i in positions],
            },
        )
        child = copy.deepcopy(original)
        child.update(cache=ref(index), output=str(directory / "features"))
        if manifest is not None:
            path = directory / "rasters.json"
            dump(
                path,
                {
                    "schema": manifest["schema"],
                    "records": [manifest["records"][i] for i in positions],
                    "dependencies": manifest.get("dependencies", []),
                },
            )
            child["rasters"] = ref(path)
        path = directory / "spec.json"
        dump(path, child)
        shards.append({"positions": positions, "spec": ref(path)})
    plan = root / "plan.json"
    dump(
        plan,
        {
            "schema": "regional-semantic-merge-v1",
            "source_spec": spec["source_spec"],
            "shards": shards,
            "preparation": ref(spec_path),
            "output": original["output"],
        },
    )
    return plan


def merge(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    if (
        set(plan) != {"schema", "source_spec", "shards", "preparation", "output"}
        or plan["schema"] != "regional-semantic-merge-v1"
        or not plan["shards"]
    ):
        raise ValueError("Invalid semantic merge specification")
    verified(plan["preparation"])
    original, cache = semantic_features.read_spec(verified(plan["source_spec"]))
    if plan["output"] != original["output"]:
        raise ValueError("Semantic merged output differs from its regional source")
    positions = [p for shard in plan["shards"] for p in shard["positions"]]
    if any(type(p) is not int for p in positions) or sorted(positions) != list(
        range(len(cache["records"]))
    ):
        raise ValueError("Semantic shard coverage must be exact and nonoverlapping")
    expected = {k: v for k, v in original.items() if k not in {"cache", "rasters", "output"}}
    source_rasters = (
        json.loads(verified(original["rasters"]).read_text()) if "rasters" in original else None
    )
    checked, shape, source_shape, normalization = [], None, None, None
    for shard in plan["shards"]:
        if set(shard) != {"positions", "spec"}:
            raise ValueError("Unknown semantic shard fields")
        spec_path = verified(shard["spec"])
        child = json.loads(spec_path.read_text())
        parameters = {k: v for k, v in child.items() if k not in {"cache", "rasters", "output"}}
        if parameters != expected:
            raise ValueError("Semantic shard encoder parameters differ from the regional recipe")
        index = json.loads(verified(child["cache"]).read_text())
        if index["months"] != cache["months"] or index["records"] != [
            cache["records"][i] for i in shard["positions"]
        ]:
            raise ValueError("Semantic shard geographic order differs")
        if source_rasters is not None:
            manifest = json.loads(verified(child["rasters"]).read_text())
            if manifest["records"] != [
                source_rasters["records"][i] for i in shard["positions"]
            ] or manifest.get("dependencies", []) != source_rasters.get("dependencies", []):
                raise ValueError("Semantic shard observation inputs differ")
        root = Path(child["output"])
        status = json.loads((root / "native-status.json").read_text())
        header = json.loads((root / "identity.json").read_text())
        if (
            header != {"spec_sha256": sha(spec_path)}
            or status.get("stage") != "complete"
            or status.get("completed") != len(shard["positions"])
        ):
            raise ValueError("Semantic native extraction is incomplete or changed")
        native, mask = root / "native.npy", root / "native-mask.npy"
        values, valid = np.load(native, mmap_mode="r"), np.load(mask, mmap_mode="r")
        if (
            values.ndim != 4
            or values.shape[1] != 1024
            or len(values) != len(shard["positions"])
            or values.dtype != np.float16
            or valid.shape != (len(values), 1, *values.shape[-2:])
            or valid.dtype != np.float32
        ):
            raise ValueError("Semantic native shard dimensions or precision differ")
        normalization_values = {
            k: status.get(k)
            for k in [
                "normalization_mean",
                "normalization_std",
                "prefix_tokens_removed",
                "window_shape",
            ]
        }
        if shape is None:
            shape, source_shape, normalization = (
                values.shape[1:],
                status["source_shape"],
                normalization_values,
            )
        elif (
            shape != values.shape[1:]
            or source_shape != status["source_shape"]
            or normalization != normalization_values
        ):
            raise ValueError("Semantic native shard geometry or normalization differs")
        checked.append(
            {
                "positions": shard["positions"],
                "spec": shard["spec"],
                "native": ref(native),
                "mask": ref(mask),
                "status": ref(root / "native-status.json"),
            }
        )
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "plan_sha256": sha(plan_path),
        "implementation_sha256": sha(__file__),
        "inputs": checked,
    }
    header = root / "merge-identity.json"
    if not header.exists() and any(
        (root / name).exists() for name in ["native.npy", "native-mask.npy", "annual.npy"]
    ):
        raise ValueError("Semantic merge must not overwrite unowned feature outputs")
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Semantic merged sources changed")
    if (root / "merge-status.json").exists():
        previous = json.loads((root / "merge-status.json").read_text())
        if (
            previous["state"] == "complete"
            and sha(root / "native.npy") == previous["native_sha256"]
            and sha(root / "native-mask.npy") == previous["mask_sha256"]
        ):
            return root
        raise ValueError("Semantic completed merge changed")
    dump(header, identity)
    features = np.lib.format.open_memmap(
        root / "native.partial.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(cache["records"]), *shape),
    )
    masks = np.lib.format.open_memmap(
        root / "native-mask.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(cache["records"]), 1, *shape[-2:]),
    )
    count = 0
    for shard in checked:
        values, valid = np.load(shard["native"]["path"], mmap_mode="r"), np.load(
            shard["mask"]["path"], mmap_mode="r"
        )
        for local, position in enumerate(shard["positions"]):
            if (
                not np.isfinite(values[local]).all()
                or not np.isfinite(valid[local]).all()
                or (valid[local] < 0).any()
                or (valid[local] > 1).any()
            ):
                raise ValueError("Nonfinite semantic native values or invalid coverage")
            features[position], masks[position] = values[local], valid[local]
            count += 1
        features.flush()
        masks.flush()
        dump(root / "merge-progress.json", {"completed": count, "total": len(cache["records"])})
    features.flush()
    masks.flush()
    (root / "native.partial.npy").replace(root / "native.npy")
    (root / "native-mask.partial.npy").replace(root / "native-mask.npy")
    dump(root / "identity.json", {"spec_sha256": plan["source_spec"]["sha256"]})
    dump(
        root / "native-status.json",
        {
            "stage": "complete",
            "completed": len(features),
            "source_shape": source_shape,
            "native_shape": list(features.shape),
            **normalization,
            "merged_shards": len(checked),
        },
    )
    dump(
        root / "merge-status.json",
        {
            "state": "complete",
            "native_sha256": sha(root / "native.npy"),
            "mask_sha256": sha(root / "native-mask.npy"),
            "projection_fitted": False,
        },
    )
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment semantic-shards")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "merge"], required=True)
    args = parser.parse_args(argv)
    (prepare if args.phase == "prepare" else merge)(args.spec)
    return 0
