"""Combined model-run input — N session containers embedded side by side.

A model run consumes one file. That file is this: each source session
container is copied intact under ``/sessions/<session_uid>/`` — the source's
root attrs are preserved on the group, so every embedded session stays
self-describing — and the run-level user params are stamped as top-level
attributes. Sessions cross-reference each other by ``session_uid`` exactly as
their dependency edges already do, so calibration/system dependencies packed
alongside samples resolve within the file.

Params are the consumer's semantics (e.g. role designations for a pairwise
model) — this module only requires them to be scalars, because HDF5 attrs
hold nothing else. Embedding order is the canonical ``session_uid`` sort:
deterministic regardless of argument order; meaningful ordering, if a model
ever wants one, belongs in params.

Internal soft links (set -> detector-set catalog entry) are expanded into
real copies while embedding: their absolute ``/session/...`` targets would
dangle once re-rooted under ``/sessions/<uid>/``.
"""

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import h5py

from container.common import hdf5 as H
from container.common.hdf5 import _decode
from container.common.ids import generate_container_id, now_timestamp, today_token

from . import schema as S

# Root group holding the embedded session containers, one child per session_uid.
NAME_SESSIONS = "sessions"

# Root identity a param key may not shadow.
_COMBINED_RESERVED = frozenset({
    S.ATTR_FORMAT, S.ATTR_SCHEMA_VERSION, S.ATTR_CONTAINER_TYPE,
    S.ATTR_CONTAINER_ID, S.ATTR_CREATED_AT,
    S.ATTR_PRODUCER_SOFTWARE, S.ATTR_PRODUCER_VERSION,
})


def build_combined_container(
    session_files: Sequence[Union[str, Path]],
    params: Dict[str, Any],
    folder: Union[str, Path],
    producer_software: str = "unknown",
    producer_version: str = "unknown",
) -> Tuple[str, str]:
    """Write the combined container in one pass and close it.

    ``session_files`` are finished v0.3 session containers (validate them
    before combining — this only checks self-description). Returns
    ``(container_id, file_path)``.
    """
    if not session_files:
        raise ValueError("no session containers to combine")

    collisions = _COMBINED_RESERVED & params.keys()
    if collisions:
        raise ValueError(f"params: keys collide with reserved names {sorted(collisions)}")
    for k, v in params.items():
        if not isinstance(v, (str, int, float, bool)):
            raise ValueError(
                f"param {k!r}: top-level attrs hold scalars only, got {type(v).__name__}"
            )

    by_uid = {}
    for path in session_files:
        uid = _check_session_container(path)
        if uid in by_uid:
            raise ValueError(f"duplicate session_uid {uid!r}")
        by_uid[uid] = Path(path)

    container_id = generate_container_id()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"combined_{len(by_uid)}s_{today_token()}_{container_id[:8]}.nxs.h5"

    root_attrs = {
        S.ATTR_FORMAT: S.FORMAT,
        S.ATTR_SCHEMA_VERSION: S.SCHEMA_VERSION,
        S.ATTR_CONTAINER_TYPE: S.CONTAINER_TYPE_COMBINED,
        S.ATTR_CONTAINER_ID: container_id,
        S.ATTR_CREATED_AT: now_timestamp(),
        S.ATTR_PRODUCER_SOFTWARE: producer_software,
        S.ATTR_PRODUCER_VERSION: producer_version,
        **params,
    }

    f = H.create_root(file_path, root_attrs)
    try:
        sessions = f.create_group(NAME_SESSIONS)
        for uid in sorted(by_uid):
            grp = sessions.create_group(uid)
            with h5py.File(by_uid[uid], "r") as src:
                for k in src.attrs:
                    grp.attrs[k] = src.attrs[k]
                for name in src:
                    src.copy(src[name], grp, name=name, expand_soft=True)
    finally:
        f.close()

    return container_id, str(file_path)


def _check_session_container(path) -> str:
    """Root self-description check; returns the file's session_uid."""
    with h5py.File(path, "r") as f:
        ctype = _decode(f.attrs.get(S.ATTR_CONTAINER_TYPE))
        if ctype != S.CONTAINER_TYPE_SESSION:
            raise ValueError(f"{path}: container_type is {ctype!r}, expected 'session'")
        uid = _decode(f.attrs.get(S.ATTR_SESSION_UID))
        if not uid:
            raise ValueError(f"{path}: missing session_uid")
        return uid
