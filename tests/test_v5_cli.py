import subprocess
import sys


def test_data_only_cli_does_not_initialize_torch_or_npu_runtime():
    code = """
import sys
from xuannv_embedding.cli import main
try:
    main(["data", "prepare-v5", "--help"])
except SystemExit as exc:
    assert exc.code == 0
assert "torch" not in sys.modules
assert "torch_npu" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
