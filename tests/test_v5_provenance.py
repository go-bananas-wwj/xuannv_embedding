from xuannv_embedding.data_process.v5_provenance import inspect_source_reference
from xuannv_embedding.data_process.v5_sources import sha256


def test_original_hash_and_source_metadata_are_distinct_evidence(tmp_path):
    path = tmp_path / "source.zip"
    path.write_bytes(b"original")
    reference = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }
    metadata = inspect_source_reference(reference)
    assert metadata["status"] == "metadata_matches_current_hash_locked"
    verified = inspect_source_reference({**reference, "sha256": sha256(path)})
    assert verified["status"] == "matches_original_sha256"
    path.write_bytes(b"changed!")
    failed = inspect_source_reference({**reference, "sha256": verified["actual_sha256"]})
    assert failed["status"] == "failed"
    assert "sha256_mismatch" in failed["issues"]
