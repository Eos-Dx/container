"""upload_zip against a fake EosCloud + fake S3 on a local HTTP server."""

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from container import AlreadyUploaded, UploadError, upload_zip

TOKEN = "tok-123"
UID = "0c7c3ad3-52b6-5d50-b160-3426f8d6e1ad"


class Fake:
    def __init__(self, presign_status=200, complete_status=202, put_status=200):
        self.calls = []
        self.presign_status, self.complete_status, self.put_status = presign_status, complete_status, put_status
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, status, obj):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.calls.append(("POST", self.path, dict(self.headers), data))
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._reply(401, {"detail": "bad token"})
                if self.path == "/api/v1/uploads/":
                    if fake.presign_status != 200:
                        return self._reply(fake.presign_status, {"status": "UPLOADED"})
                    return self._reply(200, {"url": f"http://127.0.0.1:{fake.port}/bucket/key?sig=1"})
                if self.path == "/api/v1/uploads/complete/":
                    return self._reply(fake.complete_status, {"status": "UPLOADED", "session_uid": data["session_uid"]})
                self._reply(404, {})

            def do_PUT(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                fake.calls.append(("PUT", self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                self.send_response(fake.put_status)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_port
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def zip_file(tmp_path):
    p = tmp_path / f"{UID}.zip"
    p.write_bytes(b"PK\x05\x06" + b"\0" * 18)
    return p


def _run(zip_file, **kw):
    fake = Fake(**kw)
    try:
        return fake, upload_zip(zip_file, UID, url=fake.url + "/", token=TOKEN)
    finally:
        fake.close()


def test_happy_path(zip_file):
    fake, state = _run(zip_file)
    assert state == {"status": "UPLOADED", "session_uid": UID}
    digest = hashlib.sha256(zip_file.read_bytes()).digest()
    (m1, p1, h1, d1), (m2, p2, h2, b2), (m3, p3, h3, d3) = fake.calls
    assert (m1, p1) == ("POST", "/api/v1/uploads/") and d1 == {"session_uid": UID, "sha256": digest.hex()}
    assert (m2, p2) == ("PUT", "/bucket/key?sig=1") and b2 == zip_file.read_bytes()
    assert h2["x-amz-checksum-sha256"] == base64.b64encode(digest).decode()
    assert "authorization" not in h2
    assert (m3, p3) == ("POST", "/api/v1/uploads/complete/") and d3 == d1


def test_409_at_presign_is_already_uploaded(zip_file):
    with pytest.raises(AlreadyUploaded) as e:
        _run(zip_file, presign_status=409)
    assert e.value.step == "presign" and e.value.body == {"status": "UPLOADED"}


def test_other_errors_carry_step_and_status(zip_file):
    with pytest.raises(UploadError) as e:
        _run(zip_file, put_status=403)
    assert e.value.step == "put" and e.value.status == 403
    with pytest.raises(UploadError) as e:
        _run(zip_file, complete_status=400)
    assert e.value.step == "complete" and e.value.status == 400 and not isinstance(e.value, AlreadyUploaded)
