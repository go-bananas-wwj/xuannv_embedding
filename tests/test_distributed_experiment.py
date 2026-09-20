import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.training.cli import synthetic_batch
from xuannv_embedding.training.experiment import _sha, public_base_config


def _worker(rank, world, rendezvous, args):
    import xuannv_embedding.training.experiment as experiment

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=world)
    experiment._setup_device = lambda _: (torch.device("cpu"), True, rank)
    try:
        experiment.run(args)
    finally:
        dist.destroy_process_group()


def test_ddp_handles_empty_validation_rank_and_resumes_exactly(tmp_path):
    torch.set_num_threads(1)
    raw = public_base_config(
        yaml.safe_load(Path("configs/production/haidian_p10c_v1.yaml").read_text()),
        seed=41,
        lr=1e-4,
    )
    raw["model"].update(embed_dim=8, stem_dim=8, num_months=2)
    raw["model"]["stp"].update(
        space_dim=16,
        time_dim=16,
        precision_dim=16,
        num_blocks=1,
        num_heads=2,
        time_attention_mode="none",
    )
    raw["data"].update(months=["2025-12", "2026-01"], patch_size=16, batch_size=1, num_workers=0)
    raw["training"].update(amp=False, epochs=2, warmup_epochs=0)
    raw["training"]["input_masking"]["max_months_per_sample"] = 1
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    cfg = Config.from_yaml(config)
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for i in range(4):
        batch = synthetic_batch(cfg, batch_size=1, spatial_size=16)
        sample = {
            k: ({s: v[0] for s, v in val.items()} if isinstance(val, dict) else val[0])
            for k, val in batch.items()
            if k != "patch_ids"
        }
        sample.update(patch_id=f"p{i}", region="haidian")
        path = cache / f"{i}.pt"
        torch.save(sample, path)
        rows.append({"path": str(path), "sha256": _sha(path), "patch_id": f"p{i}"})
    (cache / "cache.json").write_text(
        json.dumps(
            {
                "model_inputs": {k: asdict(v) for k, v in cfg.model.input_sources.items()},
                "model_targets": {k: asdict(v) for k, v in cfg.model.target_heads.items()},
                "data": asdict(cfg.data),
                "records": rows,
                "split": {"train": [0, 1, 2], "validation": [3]},
            },
            default=str,
        )
    )
    args = argparse.Namespace(
        config=config,
        cache=cache,
        output=tmp_path / "full",
        device="cpu",
        epochs=2,
        pilot=False,
        resume=None,
    )
    mp.spawn(_worker, args=(2, str(tmp_path / "r1"), args), nprocs=2, join=True)
    args.output = tmp_path / "resumed"
    args.epochs = 1
    mp.spawn(_worker, args=(2, str(tmp_path / "r2"), args), nprocs=2, join=True)
    args.epochs = 2
    args.resume = args.output / "latest.pt"
    mp.spawn(_worker, args=(2, str(tmp_path / "r3"), args), nprocs=2, join=True)
    full = torch.load(tmp_path / "full" / "latest.pt", weights_only=True)
    resumed = torch.load(args.resume, weights_only=True)
    assert len(resumed["metrics"]["rank_random_states"]) == 2
    assert resumed["metrics"]["validation_samples"] == 1
    for name in ("model", "criterion"):
        for key, value in full[name].items():
            torch.testing.assert_close(value, resumed[name][key], rtol=0, atol=0)
    assert len((args.output / "metrics.jsonl").read_text().splitlines()) == 2
