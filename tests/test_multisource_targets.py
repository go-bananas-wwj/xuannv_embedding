import json

import numpy as np
import pytest
import torch
from test_annual_targets import target_bundle  # noqa: F401
from torch import nn

from xuannv_embedding.data.target_contract import OSM_CHANNELS, STRUCTURE_CHANNELS, TARGET_TASKS
from xuannv_embedding.data.targets import AnnualTargetReader, collate_targets
from xuannv_embedding.data_process.target_audit import digest, write_json
from xuannv_embedding.training.target_supervision import TargetSupervision


@pytest.fixture
def local_bundle(target_bundle):  # noqa: F811
    import zarr

    from xuannv_embedding.data_process.local_targets import prepare_local_targets

    root, parent, _, index = target_bundle
    manifest = json.loads((root / "manifest.json").read_text())
    for year in (2020, 2021):
        for task, value in (("clcd", 8), ("nightlights", -0.25 if year == 2020 else 2.0)):
            path = root / f"static/{task}_{year}.zarr.zip"
            values = np.full((2, 128, 128), value, dtype=np.uint8 if task == "clcd" else np.float32)
            with zarr.ZipStore(str(path), mode="w") as store:
                group = zarr.group(store=store)
                group.create_dataset(f"targets/{task}_{year}", data=values)
                group.create_dataset(
                    f"valid_masks/{task}_{year}", data=np.ones_like(values, dtype=bool)
                )
            manifest["files"].append(
                dict(
                    path=str(path.relative_to(root)),
                    bytes=path.stat().st_size,
                    sha256=digest(path),
                    family="static",
                )
            )
    (root / "osm").mkdir()
    for name in ("2020", "2021", "reliable_negative"):
        path = root / f"osm/{name}.zarr.zip"
        with zarr.ZipStore(str(path), mode="w") as store:
            group = zarr.group(store=store)
            for year in (2020, 2021) if name == "reliable_negative" else (int(name),):
                if name == "reliable_negative":
                    for channel in OSM_CHANNELS:
                        states = np.zeros((2, 128, 128), np.uint8)
                        states[:, 64:96, 64:96] = 3
                        group.create_dataset(f"{year}/states/{channel}", data=states)
                        group.create_dataset(
                            f"{year}/confidence/{channel}",
                            data=np.where(states == 3, 224, 0).astype(np.uint8),
                        )
                else:
                    for structure, channels, size in [
                        (False, OSM_CHANNELS, 128),
                        (True, STRUCTURE_CHANNELS, 512),
                    ]:
                        prefix = f"{year}/" + ("structure_2p5m/" if structure else "")
                        for channel in channels:
                            x = np.zeros((2, size, size), np.uint8)
                            x[:, 16:32, 16:32] = 255
                            for field in ("targets", "states", "confidence", "source_bits"):
                                data = (
                                    x
                                    if field in ("targets", "confidence")
                                    else (x > 0).astype(np.uint8)
                                )
                                group.create_dataset(prefix + field + "/" + channel, data=data)
        manifest["files"].append(
            dict(
                path=str(path.relative_to(root)),
                bytes=path.stat().st_size,
                sha256=digest(path),
                family="osm",
            )
        )
    write_json(root / "manifest.json", manifest)
    parent_receipt = json.loads((parent / "preparation.json").read_text())
    parent_receipt["target_manifest_sha256"] = digest(root / "manifest.json")
    write_json(parent / "preparation.json", parent_receipt)
    write_json(parent / "pretraining_split_binding.json", {})
    for scope in ("allowed", "expanded"):
        (parent / f"train_statistics.{scope}.json").write_bytes(
            (parent / "train_statistics.screen.json").read_bytes()
        )
    output = parent.with_name("local_ready")
    prepare_local_targets(parent, root, output, workers=2)
    return root, output, digest(output / "preparation.json"), index


def test_local_targets_are_pinned_year_matched_and_all_heads_backpropagate(local_bundle):
    root, prepared, sha, index = local_bundle
    with AnnualTargetReader(root, prepared, preparation_sha256=sha) as reader:
        first = reader.read(index[0]["patch_id"], 2020, tasks=TARGET_TASKS)
        second = reader.read(index[0]["patch_id"], 2021, tasks=TARGET_TASKS)
        assert set(first) == set(TARGET_TASKS)
        assert first["clcd"]["values"].shape == (9, 32, 32)
        assert first["clcd"]["values"][7].min() == 1
        assert first["nightlights"]["values"].min() == -0.25
        assert second["nightlights"]["values"].min() == 2
        assert first["osm_structure"]["values"].shape == (3, 256, 256)
        for task in ("osm_semantic", "osm_structure"):
            assert (first[task]["valid"] & (first[task]["values"] > 0)).any()
            assert (first[task]["valid"] & (first[task]["values"] == 0)).any()
        targets = collate_targets([first, second])
        module = TargetSupervision(
            4, {task: 0.1 for task in TARGET_TASKS}, dem_mean=100, dem_std=20
        )
        embedding = torch.randn(2, 4, 256, 256, requires_grad=True)
        losses = module.loss(module(embedding), targets)
        losses["total"].backward()
        assert torch.isfinite(embedding.grad).all() and embedding.grad.abs().sum() > 0
        for head in module.heads.values():
            assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
        assert losses["target_osm_semantic_positive_mass"] > 0
        assert losses["target_osm_structure_negative_mass"] > 0
        with pytest.raises(ValueError, match="sealed"):
            reader.read(index[1]["patch_id"], 2020, role="development", tasks=TARGET_TASKS)


def test_local_row_corruption_is_rejected(local_bundle):
    import zarr

    root, prepared, sha, index = local_bundle
    group = zarr.open_group(str(prepared / "local_targets/targets.zarr"), mode="r+")
    group["2020/nightlights/values"][0, 0, 0] += 1
    with AnnualTargetReader(root, prepared, preparation_sha256=sha) as reader:
        with pytest.raises(ValueError, match="row integrity"):
            reader.read(index[0]["patch_id"], 2020, tasks=["nightlights"])


def test_osm_balances_channel_and_positive_negative_mass():
    module = TargetSupervision(1, {"osm_semantic": 1}, dem_mean=0, dem_std=1)
    nn.init.zeros_(module.heads["osm_semantic"].weight)
    nn.init.zeros_(module.heads["osm_semantic"].bias)
    values = torch.zeros(1, 4, 8, 8)
    valid = torch.ones_like(values, dtype=torch.bool)
    values[0, 0, 0, 0] = 1
    valid[:, 1] = False
    values[:, 2] = 1
    result = module.loss(
        module(torch.ones(1, 1, 16, 16)),
        {"osm_semantic": {"values": values, "valid": valid, "weight": valid.float()}},
    )
    result["total"].backward()
    torch.testing.assert_close(
        module.heads["osm_semantic"].bias.grad,
        torch.tensor([0, 0, -1 / 6, 1 / 6]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result["target_osm_semantic_valid_pixels"] == 192
    assert result["target_osm_semantic_positive_mass"] == 65


def test_new_schema_rejects_local_tasks_in_legacy_schema(tmp_path):
    from dataclasses import asdict

    import yaml
    from test_annual_model import annual_config
    from test_target_supervision import target_spec

    from xuannv_embedding.config import Config, ConfigError

    raw = annual_config().to_dict()
    for dataset in raw["data"]["datasets"]:
        dataset["source_map"] = {name: name for name in raw["model"]["input_sources"]}
    target = asdict(target_spec())
    target.update(target_root="/unused", prepared_root="/unused", weights={"clcd": 0.1})
    raw["training"]["independent_targets"] = target
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="unknown"):
        Config.from_yaml(path)
    target["schema"] = "annual-local-multisource-v2"
    path.write_text(yaml.safe_dump(raw))
    assert Config.from_yaml(path).training.independent_targets.weights == {"clcd": 0.1}


def test_all_multisource_tasks_missing_keep_finite_zero_and_complete_gradient_graph():
    from xuannv_embedding.data.target_contract import target_shape

    module = TargetSupervision(2, {task: 0.1 for task in TARGET_TASKS}, dem_mean=100, dem_std=20)
    embedding = torch.randn(1, 2, 256, 256, requires_grad=True)
    targets = {}
    for task, (_, _, kind) in TARGET_TASKS.items():
        shape = (1, *target_shape(task))
        valid_shape = shape if kind != "soft_categorical" else (1, *shape[2:])
        targets[task] = {
            "values": (
                torch.full(shape, -100, dtype=torch.long)
                if kind == "categorical"
                else torch.full(shape, float("nan"))
            ),
            "valid": torch.zeros(valid_shape, dtype=torch.bool),
            "weight": torch.full(valid_shape, float("nan")),
        }
    loss = module.loss(module(embedding), targets)["total"]
    assert loss.item() == 0
    loss.backward()
    assert embedding.grad is not None and not embedding.grad.any()
    for parameter in module.parameters():
        assert parameter.grad is not None and not parameter.grad.any()


def _binary_ddp_worker(rank, rendezvous, directory):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        module = TargetSupervision(1, {"osm_semantic": 1}, dem_mean=0, dem_std=1)
        nn.init.zeros_(module.heads["osm_semantic"].weight)
        nn.init.constant_(module.heads["osm_semantic"].bias, 0.7)
        wrapped = nn.parallel.DistributedDataParallel(module)
        values = torch.zeros(1, 4, 4, 4)
        valid = torch.zeros_like(values, dtype=torch.bool)
        if rank == 0:
            values[0, 0, 0, 0] = 1
            valid[0, 0, 0, 0] = True
        else:
            valid[0, 0, 0, :] = True
        result = module.loss(
            wrapped(torch.ones(1, 1, 8, 8)), {"osm_semantic": {"values": values, "valid": valid}}
        )
        result["total"].backward()
        Path(directory, f"binary_rank{rank}.json").write_text(
            json.dumps(
                {
                    "gradient": module.heads["osm_semantic"].bias.grad[0].item(),
                    "loss": result["target_osm_semantic"].item(),
                    "valid": result["target_osm_semantic_valid_pixels"].item(),
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_binary_ddp_normalizes_global_polarities_with_unequal_rank_support(tmp_path):
    torch.multiprocessing.spawn(
        _binary_ddp_worker, args=(str(tmp_path / "binary_rendezvous"), str(tmp_path)), nprocs=2
    )
    for rank in (0, 1):
        result = json.loads((tmp_path / f"binary_rank{rank}.json").read_text())
        assert result["gradient"] == pytest.approx(
            torch.sigmoid(torch.tensor(0.7)).item() - 0.5, abs=1e-6
        )
        assert result["valid"] == 5
        assert result["loss"] == pytest.approx(
            (
                torch.nn.functional.softplus(torch.tensor(0.7))
                + torch.nn.functional.softplus(torch.tensor(-0.7))
            ).item()
            / 2,
            abs=1e-6,
        )
