import fcntl
from types import SimpleNamespace

from xuannv_embedding.training.probe_followup import cpu_slot, launch_probe


def test_cpu_probe_slot_skips_occupied_slot_and_releases_it(tmp_path):
    with (tmp_path / "slot0.lock").open("a") as occupied:
        fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with cpu_slot(tmp_path) as slot:
            assert slot == 1
    with cpu_slot(tmp_path) as slot:
        assert slot == 0


def test_probe_launch_records_cpu_command_and_refuses_duplicate(tmp_path, monkeypatch):
    import pytest

    from xuannv_embedding.training import probe_followup as module

    export = tmp_path / "export"
    export.mkdir()
    seen = []
    monkeypatch.setattr("xuannv_embedding.training.cli._git_sha", lambda: "test-sha")
    monkeypatch.setattr(
        module,
        "subprocess",
        SimpleNamespace(
            Popen=lambda command, **kwargs: (seen.append(command) or SimpleNamespace(pid=42)),
            DEVNULL=module.subprocess.DEVNULL,
            STDOUT=module.subprocess.STDOUT,
        ),
    )
    launch_probe(export, tmp_path / "cache", tmp_path / "probe", tmp_path / "slots")
    assert (export / "probe_process.json").is_file()
    assert seen[0][seen[0].index("--device") + 1] == "cpu"
    assert "--slot-directory" in seen[0]
    with pytest.raises(FileExistsError):
        launch_probe(export, tmp_path / "cache", tmp_path / "probe", tmp_path / "slots")
