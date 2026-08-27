"""EosCloud upload client: sha256 -> presign -> PUT -> complete. Stdlib only.

Shared by every producer, so it reads no settings: URL and token are explicit.
No AWS SDK — the PUT goes to a presigned URL with the checksum the server
signed, so S3 itself rejects bytes that do not match.
"""

import base64
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Union


class UploadError(Exception):
    def __init__(self, step: str, status: int, body):
        super().__init__(f"{step}: HTTP {status}: {body}")
        self.step, self.status, self.body = step, status, body


class AlreadyUploaded(UploadError):
    """The cloud already holds this session (409 at presign or complete);
    ``body`` is its upload state. Resume should skip, not retry."""


def upload_zip(zip_path: Union[str, Path], session_uid: str, *, url: str, token: str,
               timeout: float = 120.0) -> dict:
    """Upload one session zip. Returns the cloud's upload state from ``complete``.

    Raises ``AlreadyUploaded`` when the cloud reports the session as already
    received and ``UploadError`` for any other non-success response.
    """
    body = Path(zip_path).read_bytes()
    digest = hashlib.sha256(body).digest()
    payload = {"session_uid": str(session_uid), "sha256": digest.hex()}
    base = url.rstrip("/")

    presigned = _api("presign", f"{base}/api/v1/uploads/", payload, token, timeout)
    put = urllib.request.Request(
        presigned["url"], data=body, method="PUT",
        headers={"x-amz-checksum-sha256": base64.b64encode(digest).decode()})
    try:
        with urllib.request.urlopen(put, timeout=timeout):
            pass
    except urllib.error.HTTPError as e:
        raise UploadError("put", e.code, e.read().decode(errors="replace")) from None
    return _api("complete", f"{base}/api/v1/uploads/complete/", payload, token, timeout)


def _api(step, endpoint, payload, token, timeout):
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw
        cls = AlreadyUploaded if e.code == 409 else UploadError
        raise cls(step, e.code, body) from None
