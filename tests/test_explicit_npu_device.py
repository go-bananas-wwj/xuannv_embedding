import builtins
from types import SimpleNamespace

from xuannv_embedding.training import cli


def test_explicit_npu_device_registers_backend_before_constructing_device(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    imported = []
    original_import = builtins.__import__

    def import_backend(name, *args, **kwargs):
        if name == "torch_npu":
            imported.append(name)
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    def device(name):
        assert imported == ["torch_npu"]
        return SimpleNamespace(type="npu")

    monkeypatch.setattr(builtins, "__import__", import_backend)
    monkeypatch.setattr(cli.torch, "device", device)
    monkeypatch.setattr(cli.torch, "npu", SimpleNamespace(set_device=lambda d: None), raising=False)
    cli._setup_device("npu:0")
