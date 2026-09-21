import json

import pytest

from xuannv_embedding.downstream.multitask_report import (
    render_pair,
    verified_results,
)
from xuannv_embedding.training.experiment import _sha


def result_fixture(root, *, ap=0.5):
    root.mkdir()
    rows = []
    arrays = []
    for family, source, count in [
        ("C", "osm", 40),
        ("C", "esri", 60),
        ("R", "esri", 60),
        ("Q", "osm", 60),
    ]:
        for i in range(count):
            key = f"{family}_{source}_{i}"
            metrics = dict(
                ap=ap,
                f1=ap,
                iou=ap,
                ba=ap,
                rmse=0.5,
                mae=0.4,
                r2=None,
                bias=-0.1,
                support_tiles=[0],
                support_positions_sha256="same",
                validation_pixels=20,
                support_blocks=1,
                validation_blocks=2,
                queries=[0],
            )
            metrics.update({"precision_top0.01": ap, "recall_top0.01": ap})
            rows.append(
                dict(
                    key=key,
                    family=family,
                    source=source,
                    task=str(i),
                    seed=1,
                    budget=5,
                    error=0.5 if family == "R" else 1 - ap,
                    metrics=metrics,
                )
            )
            p = root / (key + ".npz")
            p.write_bytes(b"verified-predictions")
            arrays.append(dict(key=key, sha256=_sha(p)))
    identity = dict(
        protocol="multitask-v3",
        cache_sha256="same",
        active_indices=[0, 1],
        tasks={},
        test_scored=False,
        code_commit="a" * 40,
        spec_sha256="b" * 64,
    )
    (root / "identity.json").write_text(json.dumps(identity))
    (root / "results.json").write_text(json.dumps(rows))
    (root / "status.json").write_text(
        json.dumps(
            dict(state="complete", conditions_complete=220, test_scored=False, elapsed_seconds=5)
        )
    )
    verification = dict(
        conditions_verified=220,
        max_abs_difference=1e-15,
        results_sha256=_sha(root / "results.json"),
        test_scored=False,
        prediction_arrays=arrays,
    )
    return verification


def test_verified_results_rejects_stale_array_or_incomplete_evaluation(tmp_path):
    root = tmp_path / "run"
    proof = result_fixture(root)
    assert len(verified_results(root, proof)) == 220
    (root / "C_osm_0.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prediction"):
        verified_results(root, proof)
    (root / "status.json").write_text('{"state":"running"}')
    with pytest.raises(ValueError, match="complete"):
        verified_results(root, proof)


def test_verified_results_rejects_mismatched_report_and_test_scoring(tmp_path):
    root = tmp_path / "run"
    proof = result_fixture(root)
    stale = dict(proof, results_sha256="wrong")
    with pytest.raises(ValueError, match="results"):
        verified_results(root, stale)
    identity = json.loads((root / "identity.json").read_text())
    identity["test_scored"] = True
    (root / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="test"):
        verified_results(root, proof)


def test_report_preserves_negative_results_null_r2_and_never_overwrites(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    proofs = [result_fixture(base), result_fixture(candidate, ap=0.45)]
    output = tmp_path / "report"
    render_pair(base, candidate, *proofs, name="T_1", output=output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["paired"]["normalized_score"]["score"] < 0
    assert summary["summaries"]["T_1"]["R_esri"]["r2"] is None
    assert "-5.0000" in (output / "result.tex").read_text()
    assert r"T\_1" in (output / "result.tex").read_text()
    assert str(tmp_path) not in (output / "summary.json").read_text()
    assert len((output / "conditions.csv").read_text().splitlines()) == 441
    with pytest.raises(FileExistsError):
        render_pair(base, candidate, *proofs, name="T_1", output=output)


def test_report_refuses_unpaired_support_before_writing(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    proofs = [result_fixture(base), result_fixture(candidate)]
    path = candidate / "results.json"
    rows = json.loads(path.read_text())
    rows[0]["metrics"]["support_tiles"] = [99]
    path.write_text(json.dumps(rows))
    proofs[1]["results_sha256"] = _sha(path)
    with pytest.raises(ValueError, match="support mismatch"):
        render_pair(base, candidate, *proofs, name="T1", output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_report_compares_candidate_to_t0_with_original_b0_denominator(tmp_path):
    base, original, candidate = (tmp_path / name for name in ("base", "original", "candidate"))
    bproof = result_fixture(base)
    tproof = result_fixture(original, ap=0.6)
    cproof = result_fixture(candidate, ap=0.65)
    output = tmp_path / "report"
    render_pair(
        base,
        candidate,
        bproof,
        cproof,
        name="T1",
        output=output,
        reference=original,
        reference_proof=tproof,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["score_difference_vs_T0"] == pytest.approx((0.05 / 0.5) * 2 / 3)
    assert len((output / "conditions.csv").read_text().splitlines()) == 661
    assert "候选减T0" in (output / "result.tex").read_text()


def test_registered_report_requires_terminal_controller_and_matching_export(tmp_path):
    from argparse import Namespace

    from xuannv_embedding.downstream.multitask_report import run
    from xuannv_embedding.training.multitask_followup import paired_report

    def write(path, value):
        path.write_text(json.dumps(value))

    followup = tmp_path / "followup"
    (followup / "validation").mkdir(parents=True)
    (followup / "export").mkdir()
    baseline, candidate = tmp_path / "base", followup / "validation/T0"
    bproof, cproof = result_fixture(baseline), result_fixture(candidate)
    training = tmp_path / "training"
    training.mkdir()
    checkpoint = training / "epoch_0200.pt"
    checkpoint.write_bytes(b"checked checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text(
        "training: {lr: 0.0001, weight_decay: 0.05, semantic_probe_weight: 0.14}\n"
        "model: {target_heads: {}}\n"
    )
    write(
        training / "run.json",
        dict(
            git_sha="a" * 40,
            config_sha256=_sha(config),
            seed=41,
            world_size=6,
            global_batch_size=48,
            adaptation=dict(base_checkpoint_sha256="b" * 64),
        ),
    )
    write(
        training / "status.json",
        dict(state="complete", epoch=200, elapsed_seconds=50, peak_memory_bytes=100),
    )
    audit = dict(epoch=200, actual_optimizer_steps=[800], checkpoint_sha256=_sha(checkpoint))
    write(followup / "checkpoint_verification.json", audit)
    manifest = followup / "export/manifest.json"
    write(manifest, dict(checkpoint_sha256=_sha(checkpoint), checkpoint_epoch=200))
    identity = json.loads((candidate / "identity.json").read_text())
    identity["feature_identity"] = dict(manifest_sha256=_sha(manifest))
    write(candidate / "identity.json", identity)
    write(followup / "paired_summary.json", paired_report(baseline, candidate))
    write(followup / "verification.json", {"T0": cproof})
    proof_path = tmp_path / "baseline-proof.json"
    write(proof_path, {"B0": bproof})
    plan = tmp_path / "plan.json"
    write(
        plan,
        dict(
            output=str(followup),
            training_directory=str(training),
            config=str(config),
            epochs=200,
            steps=800,
            code_sha="a" * 40,
            config_sha256=_sha(config),
            run_id="T0",
            baseline=str(baseline),
            baseline_results_sha256=bproof["results_sha256"],
        ),
    )
    write(followup / "controller.json", dict(plan_sha256=_sha(plan)))
    args = Namespace(plan=plan, baseline_verification=proof_path, output=tmp_path / "report")
    write(followup / "status.json", dict(state="watching_training"))
    with pytest.raises(ValueError, match="ready_to_publish"):
        run(args)
    assert not args.output.exists()
    write(followup / "status.json", dict(state="ready_to_publish"))
    run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["training"]["optimizer_updates"] == 800
    assert str(tmp_path) not in json.dumps(summary)
    args.output = tmp_path / "changed-export-report"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="export identity"):
        run(args)
