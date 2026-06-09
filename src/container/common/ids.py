"""Identifier and filename helpers shared across container versions.

Pure functions, no version coupling. v0_2 re-exports the legacy names from here
(a Python symbol re-export — keeping `container.v0_2.schema.<name>` working —
not anything to do with rebuilding containers) so its surface is unchanged.

Two distinct identities, answering two different questions:

  - ``generate_container_id`` — random 16-hex id for the FILE/artifact. Answers
    "which exact file is this?" New on every build. Its job is physical-artifact
    identity: cloud/storage key, dedup, and telling a rebuilt file apart from the
    original — collision-proof even if ``instance_id`` is misconfigured. Not a
    cross-reference key (random ⇒ not derivable).

  - ``make_global_uid`` — deterministic id for an EoScan DB row. Answers "which
    session/set/measurement does this represent?" Derivable from
    ``(instance_id, entity, pk)`` with no file open, so containers reference each
    other by it before the target exists. This is the edge reference key.

Two timestamp helpers, for two jobs:
  - ``now_timestamp`` — full datetime for in-file attrs (created_at, logs).
  - ``today_token`` — compact, filesystem-safe date for filenames.
"""

import hashlib
import re
import time
import uuid


def now_timestamp() -> str:
    """Full datetime for in-file attributes, e.g. '2026-06-08 14:32:05'."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def today_token() -> str:
    """Compact filename-safe date, e.g. '20260608'."""
    return time.strftime("%Y%m%d")


def generate_container_id() -> str:
    """Random 16-hex-char identity for the container FILE (new each build)."""
    return uuid.uuid4().hex[:16]


def validate_container_id(container_id: str) -> bool:
    return bool(re.match(r"^[0-9a-f]{16}$", container_id))


def sanitize_filename_token(token: str) -> str:
    """Make an arbitrary string safe for use inside a filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", token)


def make_global_uid(instance_id: str, entity: str, pk: int) -> str:
    """Deterministic, globally-unique id for an EoScan row.

    DB primary keys are sequential ints that collide across deployments. This
    derives a stable id from the deployment's ``instance_id`` (namespaced by
    ``entity`` so a session pk and a set pk don't collide):

        sha256("{instance_id}:{entity}:{pk}").hexdigest()[:16]

    Deterministic: the same inputs always yield the same id, with no DB lookup
    — which is what lets one container reference another by id before that other
    container exists. The raw ``pk`` and ``instance_id`` are stored alongside.
    """
    key = f"{instance_id}:{entity}:{pk}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def format_session_container_filename(
    container_id: str,
    sample_id: str = None,
    date_token: str = None,
) -> str:
    """Format session filename: session_<id>_<sample>_<date>.nxs.h5.

    ``container_id`` here is any 16-hex token. v0_3 passes the ``session_uid`` so
    the filename is stable per session (a rebuild overwrites the same name);
    v0_2 passes its random container_id.
    """
    if not validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")

    date_part = date_token or today_token()
    if sample_id:
        safe_sample_id = sanitize_filename_token(sample_id)
        return f"session_{container_id}_{safe_sample_id}_{date_part}.nxs.h5"
    return f"session_{container_id}_{date_part}.nxs.h5"
