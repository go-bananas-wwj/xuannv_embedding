"""Paired low-label development probes; the held-out test split is never scored."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from xuannv_embedding.downstream.comparison_features import read_feature
from xuannv_embedding.downstream.heads import build_head
from xuannv_embedding.downstream.metrics import evaluate_binary
from xuannv_embedding.downstream.protocol import choose_validation_threshold
from xuannv_embedding.training.cli import _git_sha, _setup_device
from xuannv_embedding.training.experiment import _json, _sha

TASKS = {
    "building": ("osm_building",),
    "road": ("osm_major_road", "osm_minor_road"),
    "water": ("osm_water",),
    "green": ("osm_green",),
    "playground": ("osm_playground",),
}


def validate_development_split(split):
    sets = [set(split[k]) for k in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("development and test splits overlap")


def select_support(labels, patch_ids, *, budget, seed):
    candidates = [i for i, y in enumerate(labels) if bool((y > 0).any()) and bool((y == 0).any())]
    ranked = sorted(
        candidates, key=lambda i: hashlib.sha256(f"{seed}:{patch_ids[i]}".encode()).digest()
    )
    if len(ranked) < budget:
        raise ValueError(f"only {len(ranked)} eligible patches for budget {budget}")
    return ranked[:budget]


def knn_logits(train_x, train_y, query_x, device):
    x = train_x.permute(0, 2, 3, 1).reshape(-1, train_x.shape[1])
    y = train_y.reshape(-1)
    generator = torch.Generator().manual_seed(20260916)
    selected = []
    for cls in (0, 1):
        positions = torch.where(y == cls)[0]
        selected.append(positions[torch.randperm(len(positions), generator=generator)[:1024]])
    selected = torch.cat(selected)
    support = F.normalize(x[selected].to(device), dim=1)
    labels = y[selected].to(device)
    query = query_x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    predictions = []
    with torch.no_grad():
        for chunk in query.split(1024):
            similarity = F.normalize(chunk.to(device), dim=1) @ support.T
            indices = similarity.topk(min(5, len(support)), dim=1).indices
            p = labels[indices].mean(dim=1).clamp(1e-5, 1 - 1e-5)
            predictions.append(torch.logit(p).cpu())
    return torch.cat(predictions).reshape(query_x.shape[0], *query_x.shape[-2:]), len(selected)


def run(args):
    from xuannv_embedding.training.probe_followup import cpu_slot

    with cpu_slot(getattr(args, "slot_directory", None)):
        _run(args)


def _run(args):
    torch.set_num_threads(2)
    selected_tasks = getattr(args, "tasks", None) or list(TASKS)
    if len(set(selected_tasks)) != len(selected_tasks) or set(selected_tasks) - set(TASKS):
        raise ValueError("probe tasks must be unique registered tasks")
    tasks = {task: TASKS[task] for task in selected_tasks}
    heads = getattr(args, "heads", None) or ["mlp", "conv3x3", "knn"]
    if len(set(heads)) != len(heads) or set(heads) - {"mlp", "conv3x3", "knn"}:
        raise ValueError("probe heads must be unique registered heads")
    total = len(tasks) * 2 * len(heads)
    cache_path = args.cache / "cache.json"
    cache = json.loads(cache_path.read_text())
    export = json.loads((args.embeddings / "manifest.json").read_text())
    if export["cache_sha256"] != _sha(cache_path):
        raise ValueError("probe cache differs from exported embeddings")
    split = cache["split"]
    validate_development_split(split)
    if args.output.exists():
        raise FileExistsError("probe output exists")
    args.output.mkdir(parents=True)
    device, distributed, _ = _setup_device(args.device)
    if distributed:
        raise ValueError("probe requires an independent process")
    indices = split["train"] + split["validation"]
    images, targets, patch_ids = [], {t: [] for t in tasks}, []
    for n, index in enumerate(indices):
        record = cache["records"][index]
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("label cache changed")
        sample = torch.load(record["path"], weights_only=True, mmap=True)
        # Predeclared last observation month, or the registered static product.
        images.append(read_feature(export["records"][index]["path"]))
        patch_ids.append(record["patch_id"])
        for task, names in tasks.items():
            y = torch.stack([sample["supervised_labels"][k] for k in names]).amax(dim=0)
            valid = all(bool(sample["supervised_label_masks"][k].all()) for k in names)
            targets[task].append(y if valid else torch.full_like(y, -1))
        if (n + 1) % 20 == 0:
            _json(args.output / "status.json", {"state": "loading", "patches": n + 1})
    features = torch.stack(images)
    del images
    count = len(split["train"])
    train_x, val_x = features[:count], features[count:]
    metadata = {
        "git_sha": _git_sha(),
        "export_manifest_sha256": _sha(args.embeddings / "manifest.json"),
        "cache_sha256": _sha(cache_path),
        "month": export["months"][-1],
        "scope": "development validation only; OSM-assisted; not independent final test",
        "budget_unit": "fully labeled patches including foreground and background; not polygons",
        "budgets": [5, 10],
        "heads": heads,
        "support_seed": 20260916,
        "head_seed": 41,
        "tasks": list(tasks),
        "optimizer_steps": 100,
        "batch_size": 2,
        "lr": 0.001,
        "device": str(device),
        "threshold_source": "validation; scores are development selection estimates",
        "validation_patches": len(split["validation"]),
        "test_scored": False,
        "knn": {"k": 5, "max_support_pixels_per_class": 1024, "distance": "cosine"},
    }
    _json(args.output / "run.json", metadata)
    rows = []
    for task in tasks:
        labels = torch.stack(targets[task]).float()
        train_y, val_y = labels[:count], labels[count:]
        for budget in metadata["budgets"]:
            selected = select_support(train_y, patch_ids[:count], budget=budget, seed=20260916)
            x, y = train_x[selected], train_y[selected]
            label_sha = hashlib.sha256(y.numpy().tobytes()).hexdigest()
            for head in metadata["heads"]:
                torch.manual_seed(41)
                started = time.monotonic()
                support_pixels = int((y >= 0).sum())
                if head == "knn":
                    logits, fitted_pixels = knn_logits(x, y, val_x, device)
                else:
                    model = build_head(head, embed_dim=x.shape[1], num_classes=1).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                    xd, yd = x.to(device), y.to(device)
                    positive_weight = ((y == 0).sum() / (y > 0).sum()).clamp(1, 50).to(device)
                    model.train()
                    for step in range(100):
                        picks = torch.randperm(len(x))[:2].to(device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.binary_cross_entropy_with_logits(
                            model(xd[picks])[:, 0], yd[picks], pos_weight=positive_weight
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite probe loss")
                        loss.backward()
                        optimizer.step()
                        if step % 20 == 0:
                            _json(
                                args.output / "status.json",
                                {
                                    "state": "training",
                                    "task": task,
                                    "head": head,
                                    "budget": budget,
                                    "step": step,
                                },
                            )
                    model.eval()
                    with torch.no_grad():
                        logits = torch.cat(
                            [model(v.to(device))[:, 0].cpu() for v in val_x.split(2)]
                        )
                    torch.save(model.cpu().state_dict(), args.output / f"{task}_{budget}_{head}.pt")
                    fitted_pixels = support_pixels
                seconds = time.monotonic() - started
                threshold = choose_validation_threshold(logits, val_y)
                metrics = evaluate_binary(logits, val_y, threshold=threshold)
                metrics["iou"] = metrics["tp"] / max(
                    1, metrics["tp"] + metrics["fp"] + metrics["fn"]
                )
                row = {
                    "task": task,
                    "budget": budget,
                    "head": head,
                    "metrics": metrics,
                    "support_patch_ids": [patch_ids[i] for i in selected],
                    "support_label_sha256": label_sha,
                    "support_pixels": support_pixels,
                    "fitted_pixels": fitted_pixels,
                    "support_positive_pixels": int((y > 0).sum()),
                    "validation_pixels": int((val_y >= 0).sum()),
                    "fit_and_predict_seconds": seconds,
                }
                rows.append(row)
                _json(args.output / "results.json", {"metadata": metadata, "rows": rows})
                _json(
                    args.output / "status.json",
                    {"state": "running", "completed": len(rows), "total": total},
                )
                print(
                    json.dumps(
                        {
                            "task": task,
                            "head": head,
                            "budget": budget,
                            "validation_f1": metrics["f1"],
                            "seconds": seconds,
                        }
                    ),
                    flush=True,
                )
    _json(
        args.output / "status.json", {"state": "complete", "completed": len(rows), "total": total}
    )
