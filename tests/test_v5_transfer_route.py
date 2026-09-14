import hashlib
import io
from urllib.parse import urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter

from xuannv_embedding.data_process import v5_cli, v5_sources, v5_transfer


@pytest.mark.parametrize("route", ["environment", "direct"])
def test_download_route_survives_cdn_redirect_and_preserves_tls_configuration(
    tmp_path, monkeypatch, route
):
    payload = b"abcdef"
    spec = v5_sources.ArchiveSpec("p.tar.gz", 6, hashlib.sha256(payload).hexdigest(), 1)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:8080")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/example/verified-ca.pem")
    calls = []

    class Adapter(BaseAdapter):
        def send(self, request, **kwargs):
            calls.append((urlsplit(request.url).hostname, kwargs))
            response = requests.Response()
            response.request = request
            response.url = request.url
            response.raw = io.BytesIO(b"")
            response._content_consumed = True
            if len(calls) == 1:
                response.status_code = 302
                response.headers["Location"] = "https://cdn-lfs-cn-1.modelscope.cn/data"
                response._content = b""
            else:
                response.status_code = 206
                response.headers["Content-Range"] = "bytes 0-5/6"
                response._content = payload
            return response

        def close(self):
            pass

    def client():
        result = requests.Session()
        result.mount("https://", Adapter())
        return result

    monkeypatch.setattr(v5_sources, "session", client)
    result = v5_transfer.download_chunked(spec, tmp_path, "fixed", workers=1, route=route)
    assert len(calls) == 2
    assert calls[1][0] == "cdn-lfs-cn-1.modelscope.cn"
    for _, options in calls:
        assert options["verify"] == "/example/verified-ca.pem"
        proxies = options["proxies"]
        assert proxies["https"] == ("" if route == "direct" else "http://proxy.invalid:8080")
        if route == "direct":
            assert all(proxies[k] == "" for k in ["http", "https", "all"])
    assert result["download_route"] == route
    assert (tmp_path / spec.archive).read_bytes() == payload


def test_route_change_resumes_existing_pinned_chunk_without_restarting_bytes(tmp_path, monkeypatch):
    from test_v5_sources import Response, Session

    payload = b"abcdef"
    spec = v5_sources.ArchiveSpec("p.tar.gz", 6, hashlib.sha256(payload).hexdigest(), 1)
    parts = tmp_path / "p.tar.gz.parts"
    parts.mkdir()
    (parts / "000000000000.partial").write_bytes(b"abc")
    v5_sources.write_json(
        parts / "source.json",
        {"revision": "fixed", **spec.__dict__, "chunk_bytes": 16 * 1024 * 1024},
    )
    client = Session([Response(206, b"def", {"Content-Range": "bytes 3-5/6"})])
    client.close = lambda: None
    monkeypatch.setattr(v5_sources, "session", lambda: client)
    result = v5_transfer.download_chunked(spec, tmp_path, "fixed", workers=1, route="direct")
    assert client.ranges == ["bytes=3-5"]
    assert result["download_route"] == "direct"
    assert (tmp_path / spec.archive).read_bytes() == payload


def test_unknown_download_route_is_rejected_before_creating_outputs(tmp_path):
    spec = v5_sources.ArchiveSpec("p.tar.gz", 6, hashlib.sha256(b"abcdef").hexdigest(), 1)
    with pytest.raises(ValueError, match="route"):
        v5_transfer.download_chunked(spec, tmp_path / "new", "fixed", route="guess")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "route_args,expected", [([], "environment"), (["--download-route", "direct"], "direct")]
)
def test_download_cli_passes_explicit_route_without_changing_package_concurrency(
    tmp_path, monkeypatch, route_args, expected
):
    spec = v5_sources.ArchiveSpec("p.tar.gz", 6, hashlib.sha256(b"abcdef").hexdigest(), 1)
    source = {"archives": [spec.__dict__], "revision": "fixed", "manifest_sha256": {}}
    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: source)
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []

    def download(item, directory, revision, **kwargs):
        calls.append((item, revision, kwargs))
        return {**item.__dict__, "status": "complete", "download_route": kwargs["route"]}

    monkeypatch.setattr(v5_cli, "download_chunked", download)
    args = ["--stage", "download", *route_args]
    for field in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + field, str(tmp_path / field)]
    assert v5_cli.main(args) == 0
    assert calls == [(spec, "fixed", {"route": expected})]
