"""Measure fresh regional fitting and a complete frozen native-input replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import verified


def _prepare(path, schema, key):
    spec = json.loads(Path(path).read_text())
    if set(spec) != {"schema", key, "output"} or spec["schema"] != schema:
        raise ValueError("Invalid regional cost profile")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists():
        raise ValueError("A completed cost profile must not be timed as fresh preparation")
    return spec, root, verified(spec[key])


def public_profile(path):
    from xuannv_embedding.training import temporal_fusion

    spec, root, fit = _prepare(path, "journal-public-cost-v1", "fit")
    settings = json.loads(fit.read_text())
    if any(Path(settings["output"]).glob("*")):
        raise ValueError("Fresh public profile requires an unused training output")
    started = time.monotonic()
    temporal_fusion.train(fit)
    fitted = time.monotonic()
    status_path = Path(settings["output"]) / "status.json"
    status = json.loads(status_path.read_text())
    if status["state"] != "complete" or status["step"] != settings["steps"]:
        raise ValueError("Public cost profile requires the registered terminal step")
    temporal_fusion.export(fit)
    finished = time.monotonic()
    result = dict(
        state="complete",
        fit_seconds=fitted - started,
        export_seconds=finished - fitted,
        total_seconds=finished - started,
        steps=settings["steps"],
        fit=spec["fit"],
        terminal_status={"path": str(status_path), "sha256": sha(status_path)},
        code_sha256=sha(__file__),
        scope="Fresh regional public fit/export with existing prepared public observations",
    )
    dump(root / "result.json", result)
    return result


def native_profile(path, device):
    from xuannv_embedding.export.journal_observations import run_raw_case

    spec, root, case = _prepare(path, "journal-native-cost-v1", "case")
    settings = json.loads(case.read_text())
    if (Path(settings["output"]) / "encoded/detail").exists():
        raise ValueError("Fresh native cost profile requires unencoded input rasters")
    started = time.monotonic()
    result = run_raw_case(case, device)
    elapsed = time.monotonic() - started
    if result["state"] != "complete":
        raise ValueError("Native profile lacks a terminal frozen readout")
    query_seconds = result["wall_seconds"]
    receipt = dict(
        state="complete",
        total_seconds=elapsed,
        encoding_seconds=elapsed - query_seconds,
        frozen_query_seconds=query_seconds,
        case=spec["case"],
        code_sha256=sha(__file__),
        scope=(
            "Fresh frozen local visual extraction, PCA, regional fusion "
            "and guidance; cached public anchor"
        ),
        pretrained_models_refitted=False,
        regional_models_refitted=False,
    )
    dump(root / "result.json", receipt)
    return receipt
