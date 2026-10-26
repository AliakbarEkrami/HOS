"""Offline transport failures must never publish unchecked raw files."""
import hashlib
import io

import pytest

from hos_hrv_reproduction import mitdb_protocol as protocol


class Response(io.BytesIO):
    def __init__(self, payload, status, **headers):
        super().__init__(payload)
        self.status, self.headers = status, headers


@pytest.mark.parametrize("supports_range", [True, False])
def test_download_resumes_truncated_response_or_safely_restarts(tmp_path, monkeypatch, supports_range):
    payload = b"known full signal bytes"
    requests = []
    def open_response(request, timeout):
        requests.append(dict(request.header_items()))
        if len(requests) == 1:
            return Response(payload[:5], 200, **{"Content-Length": str(len(payload))})
        if supports_range:
            return Response(payload[5:], 206,
                            **{"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"})
        return Response(payload, 200, **{"Content-Length": str(len(payload))})
    monkeypatch.setattr(protocol, "urlopen", open_response)
    monkeypatch.setattr(protocol.time, "sleep", lambda _: None)
    target = tmp_path / "100.dat"
    protocol._download_verified_file("https://example.invalid/100.dat", target,
                                     hashlib.sha256(payload).hexdigest())
    assert target.read_bytes() == payload
    assert requests[1]["Range"] == "bytes=5-"
    assert not target.with_suffix(".dat.part").exists()


def test_corrupted_payload_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "urlopen", lambda *a, **k: Response(b"bad", 200))
    monkeypatch.setattr(protocol.time, "sleep", lambda _: None)
    target = tmp_path / "100.dat"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        protocol._download_verified_file("https://example.invalid/100.dat", target, "0" * 64)
    assert not target.exists()


def test_complete_verified_partial_needs_no_network(tmp_path, monkeypatch):
    payload = b"completed before an interrupted rename"
    target = tmp_path / "100.dat"
    target.with_suffix(".dat.part").write_bytes(payload)
    def unavailable(*a, **k):
        raise AssertionError("A verified complete partial should not be downloaded again")
    monkeypatch.setattr(protocol, "urlopen", unavailable)
    protocol._download_verified_file("https://example.invalid/100.dat", target,
                                     hashlib.sha256(payload).hexdigest())
    assert target.read_bytes() == payload


def test_socket_timeout_preserves_short_delivery_for_next_range(tmp_path, monkeypatch):
    payload = b"a short packet then a connection timeout"
    class TimeoutResponse(Response):
        def read1(self, size):
            if self.tell():
                raise TimeoutError("socket stopped delivering")
            return super().read1(3)
    calls = []
    def open_response(request, timeout):
        calls.append(dict(request.header_items()))
        if len(calls) == 1:
            return TimeoutResponse(payload, 200, **{"Content-Length": str(len(payload))})
        return Response(payload[3:], 206,
                        **{"Content-Range": f"bytes 3-{len(payload)-1}/{len(payload)}"})
    monkeypatch.setattr(protocol, "urlopen", open_response)
    monkeypatch.setattr(protocol.time, "sleep", lambda _: None)
    target = tmp_path / "100.dat"
    protocol._download_verified_file("https://example.invalid/100.dat", target,
                                     hashlib.sha256(payload).hexdigest())
    assert calls[1]["Range"] == "bytes=3-"
    assert target.read_bytes() == payload
