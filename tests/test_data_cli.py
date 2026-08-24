from __future__ import annotations

import json
from pathlib import Path

import pytest

from xuannv_embedding.cli import main
from xuannv_embedding.utils.manifest import load_manifest


@pytest.mark.parametrize(
    "command",
    ["grid", "registry", "partition", "materialize", "preprocess", "manifest", "validate"],
)
def test_data_commands_expose_their_real_help(command: str) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["data", command, "--help"])
    assert raised.value.code == 0


def test_manifest_and_validate_commands_round_trip(tmp_path: Path, capsys) -> None:
    processed = tmp_path / "processed"
    legacy_path = processed / "legacy" / "manifest.json"
    legacy_path.parent.mkdir(parents=True)
    source_path = processed / "patches" / "s2" / "sample.tif"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"test")
    legacy_path.write_text(
        json.dumps([{"patch_id": "p1", "s2": "../patches/s2/sample.tif"}]),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"

    assert (
        main(
            [
                "data",
                "manifest",
                "--legacy",
                str(legacy_path),
                "--region",
                "test-region",
                "--output",
                str(output),
                "--months",
                "202601",
            ]
        )
        == 0
    )
    document = load_manifest(output)
    assert document.records[0].sources == {"s2": "patches/s2/sample.tif"}
    capsys.readouterr()

    assert main(["data", "validate", "--manifest", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["record_count"] == 1
