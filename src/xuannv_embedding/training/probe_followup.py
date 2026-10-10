"""Bounded CPU follow-up probes after successful embedding exports."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def cpu_slot(directory: Path | None):
    if directory is None:
        yield None
        return
    directory.mkdir(parents=True, exist_ok=True)
    acquired = None
    while acquired is None:
        for slot in range(3):
            handle = (directory / f"slot{slot}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            acquired = handle
            break
        if acquired is None:
            time.sleep(5)
    try:
        yield slot
    finally:
        fcntl.flock(acquired, fcntl.LOCK_UN)
        acquired.close()


def launch_probe(export: Path, cache: Path, output: Path, slots: Path) -> None:
    from xuannv_embedding.training.cli import _git_sha
    from xuannv_embedding.training.experiment import _json

    registry = export / "probe_process.json"
    log_path = Path(str(output) + ".log")
    if output.exists() or registry.exists() or log_path.exists():
        raise FileExistsError("probe follow-up already registered")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-c",
        "from xuannv_embedding.cli import main; raise SystemExit(main())",
        "experiment",
        "probe",
        "--cache",
        str(cache),
        "--embeddings",
        str(export),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--slot-directory",
        str(slots),
    ]
    env = os.environ.copy()
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=export,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _json(
        registry,
        {
            "pid": process.pid,
            "command": command,
            "output": str(output),
            "log": str(log_path),
            "git_sha": _git_sha(),
            "state": "launched; see probe output status for completion",
        },
    )
