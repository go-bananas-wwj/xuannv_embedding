"""Paired spatial development probes for frozen annual embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LANDCOVER = [
    "tree",
    "shrub",
    "grass",
    "crop",
    "built",
    "bare",
    "snow",
    "water",
    "wetland",
    "mangrove",
    "moss",
]
ROAD_HALF_WIDTH = {
    "motorway": 6,
    "trunk": 6,
    "primary": 5,
    "secondary": 4,
    "tertiary": 3.5,
    "motorway_link": 3.5,
    "trunk_link": 3.5,
    "primary_link": 3.5,
    "secondary_link": 3,
    "tertiary_link": 3,
    "residential": 2.5,
    "unclassified": 2.5,
    "service": 2,
    "living_street": 2.5,
    "track": 1.5,
    "road": 2.5,
    "pedestrian": 2,
    "footway": 1,
    "path": 1,
    "cycleway": 1,
    "steps": 1,
}


def spatial_group(patch_id: str) -> str:
    """Keep patches within a 20.48 km parent-grid block in the same fold."""
    zone, x, y = patch_id.removeprefix("parent_").split(":")
    return f"{zone}:{int(x) // 16}:{int(y) // 16}"


def folds_for(groups: list[str]) -> np.ndarray:
    ordered = sorted(set(groups), key=lambda group: hashlib.sha256(group.encode()).digest())
    if len(ordered) < 5:
        raise ValueError("Spatial development probes require at least five independent groups")
    folds = {group: index % 5 for index, group in enumerate(ordered)}
    return np.asarray([folds[group] for group in groups])


def paired_group_bootstrap(a, b, groups, statistic, *, draws=2000, seed=42):
    """Resample whole cities/spatial groups, preserving pairing and cluster sizes."""
    a, b, groups = np.asarray(a), np.asarray(b), np.asarray(groups)
    unique = np.unique(groups)
    blocks = [np.flatnonzero(groups == group) for group in unique]
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        index = np.concatenate(
            [blocks[i] for i in generator.integers(len(blocks), size=len(blocks))]
        )
        differences.append(statistic(a[index].sum(0)) - statistic(b[index].sum(0)))
    return {
        "difference": float(statistic(a.sum(0)) - statistic(b.sum(0))),
        "CI95": np.percentile(differences, [2.5, 97.5]).tolist(),
        "groups": len(unique),
        "patches": len(a),
        "draws": draws,
    }


def semantic_metrics(confusion, task):
    cm = np.asarray(confusion, dtype=np.float64)
    tp, gt, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    denominator = gt + predicted - tp
    if task == "water":
        return {
            "F1": float(2 * tp[1] / max(gt[1] + predicted[1], 1)),
            "IoU": float(tp[1] / max(denominator[1], 1)),
            "OA": float(tp.sum() / max(cm.sum(), 1)),
        }
    present = gt > 0
    return {
        "mIoU": float((tp[present] / denominator[present]).mean()),
        "macroF1": float((2 * tp[present] / (gt + predicted)[present]).mean()),
        "OA": float(tp.sum() / max(cm.sum(), 1)),
        "per_class_IoU": {
            LANDCOVER[i]: float(tp[i] / denominator[i]) for i in np.flatnonzero(present)
        },
    }


def semantic_probe(plan, root: Path, tag: str, device):
    """Fixed linear probes; fit normalization and class weights on training folds only."""
    from xuannv_embedding.training.annual_experiments import digest, write_json

    labels_root = Path(plan["labels"])
    records = {
        row["patch_id"]: row
        for row in json.loads((labels_root / "labels.json").read_text())["records"]
    }
    ids = sorted(records)
    groups = [spatial_group(patch_id) for patch_id in ids]
    folds = folds_for(groups)
    features, labels, patches, annual_splits = (
        [],
        {task: [] for task in ["water", "landcover"]},
        [],
        [],
    )
    for index, patch_id in enumerate(ids):
        feature_path = root / tag / "semantic" / f"{patch_id}.npz"
        canonical_path = root / "original" / "semantic" / f"{patch_id}.npz"
        with np.load(feature_path) as candidate, np.load(canonical_path) as original:
            valid = original["valid"] > 0.5
            if not np.all(candidate["valid"][valid] > 0.5):
                raise ValueError("Candidate loses support on canonical scored pixels")
            features.append(
                candidate["emb"].reshape(64, -1).T[valid.reshape(-1)].astype(np.float32)
            )
            annual_splits.append(str(candidate["split"]))
        label_path = labels_root / records[patch_id]["file"]
        if digest(label_path) != records[patch_id]["sha256"]:
            raise ValueError("Label asset digest differs")
        with np.load(label_path) as target:
            for task in labels:
                labels[task].append(target[task].reshape(-1)[valid.reshape(-1)])
        patches.append(np.full(valid.sum(), index, dtype=np.int64))
    features = torch.from_numpy(np.concatenate(features))
    labels = {
        task: torch.from_numpy(np.concatenate(values)).long() for task, values in labels.items()
    }
    patches = torch.from_numpy(np.concatenate(patches))
    fold_pixels = torch.from_numpy(folds)[patches]
    result = {
        "development_only": True,
        "fold_unit": "20.48 km spatial block",
        "groups": len(set(groups)),
        "ids": ids,
        "folds": folds.tolist(),
        "annual_splits": annual_splits,
        "tasks": {},
    }
    destination = root / tag / "semantic_probe"
    destination.mkdir(exist_ok=True)
    for task, classes in [("water", 2), ("landcover", 11)]:
        y = labels[task]
        keep = y >= 0
        confusion = np.zeros((len(ids), classes, classes), dtype=np.int64)
        for fold in range(5):
            torch.manual_seed(100 + fold)
            train = (keep & (fold_pixels != fold)).nonzero().flatten()
            test = (keep & (fold_pixels == fold)).nonzero().flatten()
            generator = torch.Generator().manual_seed(fold)
            train = train[torch.randperm(len(train), generator=generator)[:200_000]]
            mean, std = features[train].mean(0), features[train].std(0).clamp(min=1e-6)
            x_train = ((features[train] - mean) / std).to(device)
            y_train = y[train].to(device)
            counts = torch.bincount(y_train, minlength=classes).float()
            weight = torch.where(
                counts > 0, counts.sum() / (counts.clamp(min=1) * (counts > 0).sum()), 0
            )
            head = nn.Linear(64, classes).to(device)
            optimizer = torch.optim.LBFGS(
                head.parameters(), max_iter=300, history_size=20, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                loss = (
                    F.cross_entropy(head(x_train), y_train, weight=weight)
                    + 1e-4 * head.weight.square().sum()
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                for chunk in torch.split(test, 200_000):
                    prediction = head(((features[chunk] - mean) / std).to(device)).argmax(1).cpu()
                    flat = patches[chunk] * classes * classes + y[chunk] * classes + prediction
                    confusion += (
                        torch.bincount(flat, minlength=len(ids) * classes * classes)
                        .reshape(len(ids), classes, classes)
                        .numpy()
                    )
        np.save(destination / f"{task}.npy", confusion)
        result["tasks"][task] = {"all": semantic_metrics(confusion.sum(0), task)}
        for split_name, select in [
            ("pretraining_seen", np.asarray(annual_splits) == "train"),
            ("pretraining_unseen", np.asarray(annual_splits) != "train"),
        ]:
            if select.any():
                result["tasks"][task][split_name] = {
                    **semantic_metrics(confusion[select].sum(0), task),
                    "patches": int(select.sum()),
                }
        print(json.dumps({"semantic": tag, "task": task, **result["tasks"][task]}), flush=True)
    write_json(destination / "metrics.json", result)


def rasterize_osm(path: Path):
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely import wkb
    from shapely.ops import transform

    content = json.loads(path.read_text())
    metadata = content["_meta"]
    x0, _, _, y1 = metadata["bounds"]
    project = Transformer.from_crs(4326, metadata["epsg"], always_xy=True).transform
    output = {}
    for task in ["building", "road", "water"]:
        geometries = []
        for encoded, category in content[task]:
            if task == "road" and category not in ROAD_HALF_WIDTH:
                continue
            geometry = transform(project, wkb.loads(encoded, hex=True))
            if task == "road":
                geometry = geometry.buffer(ROAD_HALF_WIDTH[category], cap_style=2)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if not geometry.is_empty:
                geometries.append(geometry)
        if task == "building":
            output["n_building"] = len(geometries)
        output[task] = (
            rasterize(
                [(geometry, 1) for geometry in geometries],
                out_shape=(256, 256),
                transform=Affine(5, 0, x0, 0, -5, y1),
                fill=0,
                dtype="uint8",
            )
            if geometries
            else np.zeros((256, 256), np.uint8)
        )
    return output


def boundary(mask):
    values = mask.float()[:, None]
    return ((F.max_pool2d(values, 3, 1, 1) + F.max_pool2d(-values, 3, 1, 1)) > 0)[:, 0]


def city_counts(prediction, target):
    truth = target.bool()
    pe, ge = boundary(prediction), boundary(truth)
    pd = F.max_pool2d(pe.float()[:, None], 3, 1, 1)[:, 0] > 0
    gd = F.max_pool2d(ge.float()[:, None], 3, 1, 1)[:, 0] > 0
    masks = [prediction & truth, prediction & ~truth, ~prediction & truth, pe & gd, pe, ge & pd, ge]
    return torch.stack([mask.flatten(1).sum(1) for mask in masks], 1).cpu().numpy()


def city_metrics(counts):
    tp, fp, fn, bp, bpn, br, brn = np.asarray(counts, dtype=np.float64)
    precision, recall = bp / max(bpn, 1), br / max(brn, 1)
    return {
        "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "IoU": float(tp / max(tp + fp + fn, 1)),
        "boundaryF1": float(2 * precision * recall / max(precision + recall, 1e-12)),
    }


def train_city_head(features, labels, train, kind, fold, steps):
    torch.manual_seed(fold)
    device = features.device
    head = (
        nn.Conv2d(64, 1, 1)
        if kind == "linear"
        else nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 3, padding=1)
        )
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    positive = labels[train].float().mean().clamp(1e-4, 1 - 1e-4)
    mean = torch.zeros(64, device=device)
    square = torch.zeros_like(mean)
    for chunk in torch.split(train, 8):
        values = features[chunk].float()
        mean += values.sum((0, 2, 3))
        square += values.square().sum((0, 2, 3))
    count = len(train) * features.shape[-2] * features.shape[-1]
    mean /= count
    std = (square / count - mean.square()).clamp(min=1e-12).sqrt()
    mean, std = mean[None, :, None, None], std[None, :, None, None]
    generator = torch.Generator().manual_seed(fold)
    for _ in range(steps):
        index = train[torch.randint(len(train), (8,), generator=generator).to(device)]
        logits = head((features[index].float() - mean) / std)[:, 0]
        loss = F.binary_cross_entropy_with_logits(
            logits, labels[index].float(), pos_weight=(1 - positive) / positive
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head.eval(), mean, std


def city_probe(plan, root: Path, tag: str, device):
    from xuannv_embedding.training.annual_experiments import digest, write_json

    rows = json.loads(Path(plan["cities"]).read_text())
    city = {row[2]: row[0] for row in rows}
    ids = sorted(city)
    folds = folds_for([city[patch_id] for patch_id in ids])
    features, labels, counts, highres, splits = (
        [],
        {task: [] for task in ["building", "road", "water"]},
        [],
        [],
        [],
    )
    cache = root / "osm_labels"
    cache.mkdir(exist_ok=True)
    label_digests = {}
    for patch_id in ids:
        source = Path(plan["osm"]) / f"{patch_id}.json"
        label_digests[patch_id] = digest(source)
        path = cache / f"{patch_id}.npz"
        if not path.exists():
            np.savez_compressed(
                path, **rasterize_osm(source), source_sha256=label_digests[patch_id]
            )
        with np.load(path) as target:
            if str(target["source_sha256"]) != label_digests[patch_id]:
                raise ValueError("Cached OSM label provenance changed")
            counts.append(int(target["n_building"]))
            for task in labels:
                labels[task].append(target[task].copy())
        with np.load(root / tag / "city5m" / f"{patch_id}.npz") as feature:
            if not np.all(feature["valid"] > 0.5):
                raise ValueError("City protocol requires complete support on every scored patch")
            features.append(feature["emb"].copy())
            highres.append(int(feature["highres_obs"]))
            splits.append(str(feature["split"]))
    features = torch.from_numpy(np.stack(features)).to(device)
    labels = {
        task: torch.from_numpy(np.stack(values)).to(device) for task, values in labels.items()
    }
    scope = {
        "building": np.asarray(counts) >= 20,
        **{task: labels[task].flatten(1).sum(1).cpu().numpy() >= 200 for task in ["road", "water"]},
    }
    destination = root / tag / "city_probe"
    destination.mkdir(exist_ok=True)
    result = {
        "development_only": True,
        "fold_unit": "city",
        "ids": ids,
        "cities": [city[patch_id] for patch_id in ids],
        "folds": folds.tolist(),
        "highres": highres,
        "annual_splits": splits,
        "probe_steps": plan["probe_steps"],
        "label_sha256": label_digests,
        "tasks": {},
    }
    for task, target in labels.items():
        for kind in ["linear", "conv3x3"]:
            stats = np.zeros((len(ids), 7), dtype=np.int64)
            for fold in range(5):
                train = torch.from_numpy(np.flatnonzero(scope[task] & (folds != fold))).to(device)
                test = torch.from_numpy(np.flatnonzero(scope[task] & (folds == fold))).to(device)
                if not len(test):
                    continue
                if not len(train):
                    raise ValueError("City fold has no eligible training patches")
                head, mean, std = train_city_head(
                    features, target, train, kind, fold, plan["probe_steps"]
                )
                with torch.no_grad():
                    for chunk in torch.split(test, 8):
                        prediction = head((features[chunk].float() - mean) / std)[:, 0] > 0
                        stats[chunk.cpu().numpy()] = city_counts(prediction, target[chunk])
                del head
            np.save(destination / f"{task}_{kind}.npy", stats)
            groups = {
                "all": scope[task],
                "with_highres": scope[task] & (np.asarray(highres) > 0),
                "no_highres": scope[task] & (np.asarray(highres) == 0),
                "pretraining_unseen": scope[task] & (np.asarray(splits) != "train"),
            }
            key = f"{task}:{kind}"
            result["tasks"][key] = {
                group: {**city_metrics(stats[select].sum(0)), "patches": int(select.sum())}
                for group, select in groups.items()
                if select.any()
            }
            print(json.dumps({"city": tag, "task": key, **result["tasks"][key]}), flush=True)
    write_json(destination / "metrics.json", result)


def compare(plan, root: Path, arms: list[str], reference="original"):
    from xuannv_embedding.training.annual_experiments import write_json

    result = {"reference": reference, "development_only": True, "comparisons": {}}
    for tag in arms:
        if tag == reference:
            continue
        result["comparisons"][tag] = {}
        for suite in ["semantic_probe", "city_probe"]:
            base = root / reference / suite
            candidate = root / tag / suite
            if not (base / "metrics.json").exists() or not (candidate / "metrics.json").exists():
                continue
            metadata = json.loads((base / "metrics.json").read_text())
            other = json.loads((candidate / "metrics.json").read_text())
            if metadata["ids"] != other["ids"] or metadata["folds"] != other["folds"]:
                raise ValueError("Compared development sets or folds differ")
            if suite == "semantic_probe":
                groups = [spatial_group(patch_id) for patch_id in metadata["ids"]]
                tasks = [("water", "F1"), ("landcover", "mIoU")]
                for task, metric in tasks:
                    a, b = np.load(candidate / f"{task}.npy"), np.load(base / f"{task}.npy")
                    result["comparisons"][tag][f"semantic:{task}:{metric}"] = (
                        paired_group_bootstrap(
                            a, b, groups, lambda cm: semantic_metrics(cm, task)[metric]
                        )
                    )
            else:
                for task in metadata["tasks"]:
                    a = np.load(candidate / f"{task.replace(':', '_')}.npy")
                    b = np.load(base / f"{task.replace(':', '_')}.npy")
                    supported = (b[:, :3].sum(1) > 0) | (a[:, :3].sum(1) > 0)
                    for name, select in [
                        ("all", supported),
                        ("with_highres", supported & (np.asarray(metadata["highres"]) > 0)),
                        ("no_highres", supported & (np.asarray(metadata["highres"]) == 0)),
                    ]:
                        if not select.any():
                            continue
                        for metric in ["F1", "boundaryF1"]:
                            result["comparisons"][tag][f"city:{task}:{name}:{metric}"] = (
                                paired_group_bootstrap(
                                    a[select],
                                    b[select],
                                    np.asarray(metadata["cities"])[select],
                                    lambda values: city_metrics(values)[metric],
                                )
                            )
    write_json(root / f"comparison_{reference}.json", result)
    return result
