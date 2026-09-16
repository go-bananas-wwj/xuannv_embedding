import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.training.cli import synthetic_batch
from xuannv_embedding.training.experiment import _sha, public_base_config, run


def test_adaptation_checkpoint_preserves_frozen_base_through_real_training(tmp_path):
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
    raw["training"].update(amp=False, epochs=4, warmup_epochs=1)
    raw["training"]["input_masking"]["max_months_per_sample"] = 1

    def setup(name, config_raw):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config_raw))
        cfg = Config.from_yaml(path)
        cache = tmp_path / (name + "_cache")
        cache.mkdir()
        records = []
        for i in range(4):
            batch = synthetic_batch(cfg, batch_size=1, spatial_size=16)
            sample = {
                k: ({s: v[0] for s, v in val.items()} if isinstance(val, dict) else val[0])
                for k, val in batch.items()
                if k != "patch_ids"
            }
            sample.update(patch_id=f"p{i}", region="haidian")
            f = cache / f"{i}.pt"
            torch.save(sample, f)
            records.append({"path": str(f), "sha256": _sha(f)})
        (cache / "cache.json").write_text(
            json.dumps(
                {
                    "model_inputs": {k: asdict(v) for k, v in cfg.model.input_sources.items()},
                    "model_targets": {k: asdict(v) for k, v in cfg.model.target_heads.items()},
                    "data": asdict(cfg.data),
                    "records": records,
                    "split": {"train": [0, 1, 2], "validation": [3]},
                },
                default=str,
            )
        )
        return argparse.Namespace(
            config=path,
            cache=cache,
            output=tmp_path / name,
            device="cpu",
            epochs=1,
            pilot=False,
            resume=None,
        )

    base_args = setup("base", raw)
    run(base_args)
    adapted_raw = copy.deepcopy(raw)
    adapted_raw["model"]["input_sources"]["extra"] = {"channels": 3, "role": "highres"}
    adapted_raw["model"]["target_heads"]["extra_recon"] = {
        "source": "extra",
        "channels": 3,
        "loss_type": "continuous",
        "weight": 0.9,
    }
    adapted_raw["data"]["datasets"][0]["source_map"]["extra"] = "extra"
    args = setup("adapt", adapted_raw)
    args.initialize = base_args.output / "best.pt"
    args.base_config = base_args.config
    args.freeze_base = True
    args.highres_encoding = "native"
    run(args)
    args.resume = args.output / "latest.pt"
    args.epochs = 2
    run(args)
    base = torch.load(args.initialize, weights_only=True)
    adapted = torch.load(args.resume, weights_only=True)
    for k, v in base["model"].items():
        torch.testing.assert_close(v, adapted["model"]["base." + k], rtol=0, atol=0)
    for k, v in base["criterion"].items():
        torch.testing.assert_close(v, adapted["criterion"][k], rtol=0, atol=0)
    assert adapted["model"]["branches.extra.correction.weight"].abs().sum() > 0
    assert json.loads((args.output / "run.json").read_text())["initialization"] == "registered_base"
