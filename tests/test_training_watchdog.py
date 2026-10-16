import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xuannv_embedding.config import InputSourceConfig
from xuannv_embedding.training.watchdog import (
    decision,
    matching_processes,
    select_checkpoint,
    stop_owned,
)


def test_watchdog_uses_saved_steps_and_falls_back_from_a_corrupt_latest_file(tmp_path):
    source = InputSourceConfig(channels=1, role="temporal")
    config = SimpleNamespace(
        model=SimpleNamespace(input_sources={"x": source}),
        data=SimpleNamespace(datasets=[SimpleNamespace(region="test")]),
    )
    spec = {"config": {"sha256": "a" * 64}, "training": {"steps": 10}}

    def checkpoint(path, step):
        torch.save(
            {
                "format_version": "1",
                "config_sha256": "a" * 64,
                "source_schema": {"x": asdict(source)},
                "regions": ["test"],
                "epoch": step - 1,
                "metrics": {"steps": step},
                "model": {"weight": torch.ones(1)},
                "criterion": {},
                "optimizer": {"param_groups": [{"params": [0]}], "state": {}},
            },
            path,
        )

    checkpoint(tmp_path / "step0400.pt", 4)
    checkpoint(tmp_path / "latest.pt", 2)
    selected, step, _ = select_checkpoint(tmp_path, spec, config)
    assert selected.name == "step0400.pt" and step == 4
    (tmp_path / "latest.pt").write_bytes(b"truncated checkpoint")
    selected, step, rejected = select_checkpoint(tmp_path, spec, config)
    assert selected.name == "step0400.pt" and step == 4
    assert any(r["file"] == "latest.pt" for r in rejected)
    torch.save(torch.ones(1), tmp_path / "latest.pt")
    selected, step, rejected = select_checkpoint(tmp_path, spec, config)
    assert selected.name == "step0400.pt" and step == 4


def test_watchdog_keeps_healthy_runs_and_never_restarts_completed_training():
    settings = dict(
        alive=True,
        complete=False,
        stale=False,
        attempts=0,
        max_restarts=5,
        disk_ok=True,
        checkpoint_ok=True,
    )
    assert decision(**settings) == "running"
    assert decision(**(settings | {"complete": True, "alive": False})) == "complete"
    assert decision(**(settings | {"alive": False})) == "restart"
    assert decision(**(settings | {"stale": True})) == "restart"
    assert decision(**(settings | {"alive": False, "attempts": 5})) == "blocked_retry_limit"
    assert decision(**(settings | {"alive": False, "disk_ok": False})) == "blocked_disk"
    assert (
        decision(**(settings | {"alive": False, "checkpoint_ok": False})) == "blocked_no_checkpoint"
    )


@pytest.mark.skipif(not Path("/proc").exists(), reason="Linux process ownership gate")
def test_watchdog_terminates_only_the_exact_registered_training_group(tmp_path):
    spec = tmp_path / "run.json"
    other_spec = tmp_path / "other.json"
    command = [
        sys.executable,
        "-c",
        "import time;time.sleep(60)",
        "regional-base",
        "--phase",
        "train",
        "--spec",
    ]
    owned = subprocess.Popen([*command, str(spec)], start_new_session=True)
    other = subprocess.Popen([*command, str(other_spec)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while owned.pid not in matching_processes(spec) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert owned.pid in matching_processes(spec)
        assert other.pid not in matching_processes(spec)
        stop_owned(spec, timeout=2)
        owned.wait(timeout=5)
        assert other.poll() is None
    finally:
        for process in [owned, other]:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
