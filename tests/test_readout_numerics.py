import json

import pytest
import torch

from xuannv_embedding.downstream.numerics import city_readout_numerics


def _flags():
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def test_legacy_readout_policy_neither_initializes_cuda_nor_changes_rng_or_flags(monkeypatch):
    def fail():
        raise AssertionError("Legacy policy must not inspect or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_initialized", fail)
    state = torch.get_rng_state().clone()
    flags = _flags()
    with city_readout_numerics(torch.device("cuda:0")) as policy:
        assert policy is None and _flags() == flags
    assert torch.equal(state, torch.get_rng_state()) and _flags() == flags


@pytest.mark.parametrize("exception", [False, True])
def test_strict_readout_restores_prior_warn_only_and_backend_policy_even_on_error(exception):
    original = _flags()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        previous = _flags()
        try:
            with city_readout_numerics(torch.device("cpu"), deterministic=True) as policy:
                assert policy["deterministic_algorithms"] and not policy["warn_only"]
                assert policy["cudnn_deterministic"] and not policy["cudnn_benchmark"]
                assert policy["cuda_matmul_allow_tf32"] == previous[4]
                assert policy["cudnn_allow_tf32"] == previous[5]
                assert policy["cublas_workspace_config"] is None
                assert _flags()[:4] == (True, False, True, False)
                if exception:
                    raise RuntimeError("Readout failed")
        except RuntimeError as exc:
            assert exception and str(exc) == "Readout failed"
        assert _flags() == previous
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
        torch.backends.cudnn.deterministic = original[2]
        torch.backends.cudnn.benchmark = original[3]


@pytest.mark.parametrize(
    "initialized,workspace,valid",
    [
        (False, None, True),
        (True, None, False),
        (True, ":16:8", True),
        (True, ":4096:8", True),
        (False, "", False),
        (False, ":123:4", False),
    ],
)
def test_cuda_workspace_is_configured_before_initialization_or_refused(
    monkeypatch, initialized, workspace, valid
):
    import os

    if workspace is None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "removed-before-test")
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mocked A100")
    flags = _flags()
    if not valid:
        with pytest.raises(ValueError, match="cuBLAS workspace|before CUDA initialization"):
            with city_readout_numerics(torch.device("cuda:0"), deterministic=True):
                raise AssertionError("Invalid CUDA policy must refuse before the readout")
    else:
        with city_readout_numerics(torch.device("cuda:0"), deterministic=True) as policy:
            assert policy["device_name"] == "mocked A100"
            assert policy["cublas_workspace_config"] == (workspace or ":4096:8")
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == (workspace or ":4096:8")
    assert _flags() == flags


def test_strict_readout_rejects_nonboolean_policy_and_unqualified_accelerators():
    with pytest.raises(ValueError, match="boolean"):
        with city_readout_numerics(torch.device("cpu"), deterministic="true"):
            pass
    with pytest.raises(ValueError, match="CPU and CUDA"):
        with city_readout_numerics(torch.device("meta"), deterministic=True):
            pass


@pytest.mark.parametrize("source", [None, "z" * 64, "a" * 63])
def test_deterministic_probe_refuses_missing_or_invalid_feature_export_provenance(tmp_path, source):
    from xuannv_embedding.downstream.annual_ablation import city_probe

    root = tmp_path / "original"
    root.mkdir()
    (root / "export.json").write_text(
        json.dumps({"checkpoint_sha256": "b" * 64, "evaluation_source_code_sha256": source})
    )
    flags = _flags()
    with pytest.raises(ValueError, match="feature-export source provenance"):
        city_probe({}, tmp_path, "original", torch.device("cpu"), deterministic=True)
    assert _flags() == flags and not (root / "city_probe").exists()


@pytest.mark.parametrize("kind", ["linear", "conv3x3"])
def test_actual_tiny_cpu_head_training_preserves_legacy_result_and_repeats_exactly(kind):
    from xuannv_embedding.downstream.annual_ablation import train_city_head

    generator = torch.Generator().manual_seed(31)
    features = torch.randn((6, 64, 8, 8), generator=generator).half()
    labels = torch.randint(0, 2, (6, 8, 8), generator=generator)
    train = torch.arange(4)
    previous = _flags()
    states = []
    for strict in [False, True, True]:
        with city_readout_numerics(torch.device("cpu"), deterministic=strict):
            head, mean, std = train_city_head(features, labels, train, kind, fold=17, steps=3)
            with torch.no_grad():
                output = head((features.float() - mean) / std)
            states.append((head.state_dict(), mean, std, output))
        assert _flags() == previous
    for other in states[1:]:
        assert all(torch.equal(value, other[0][key]) for key, value in states[0][0].items())
        assert all(torch.equal(a, b) for a, b in zip(states[0][1:], other[1:], strict=True))


@pytest.mark.parametrize("deterministic", [False, True])
def test_probe_cli_forwards_numeric_policy_and_rejects_semantic_only(
    tmp_path, monkeypatch, deterministic
):
    from xuannv_embedding.downstream import annual_ablation
    from xuannv_embedding.training.annual_experiments import main

    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    seen = []

    def fake_probe(*args, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(annual_ablation, "city_probe", fake_probe)
    args = ["probe", "--plan", str(plan), "--arm", "original", "--suite", "city", "--device", "cpu"]
    main(args + (["--deterministic"] if deterministic else []))
    assert seen == ([{"deterministic": True}] if deterministic else [{}])
    if deterministic:
        args[args.index("city")] = "semantic"
        with pytest.raises(ValueError, match="--suite city or all"):
            main(args + ["--deterministic"])
