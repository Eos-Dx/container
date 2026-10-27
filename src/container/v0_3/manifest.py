"""The EosCloud upload contract — ``manifest.json`` + zip layout (manifest v2).

One zip per session::

    <session_uid>/
      manifest.json
      sets/<set_uid>/measurements/<measurement_uid>/image.<ext>   original bytes
      sets/<set_uid>/measurements/<measurement_uid>/header.<ext>  only when the
                                                                  header is a separate file
      masks/<detector_id>.npy                                     when supplied

The manifest mirrors ``SessionPayload`` field for field minus the processing
products (``raw``, ``processed``, ``integration``, ``processing_config``,
``processing_steps``, ``preview``) — the cloud derives those itself — plus
``manifest_version``, ``producer``, ``raw_sha256`` and per-member ``sha256``.
Additive fields do not bump ``MANIFEST_VERSION``; removing one or changing its
meaning does. The cloud rejects versions it does not know.

``session.metadata`` and every set's ``metadata`` are dicts of namespaces, one
per writer: the producer's session form values sit under ``form``, a set's
writers under theirs (``lrf``, ``operator``, ...). v1 carried session form
values flat, and the legacy backfill's set values flat beside its
``backfill_provenance`` (upgraded by ``legacy_omniscan``, kept apart because
it is temporary); that change of meaning is what v2 is.
``build_manifest`` refuses a flat dict, so a v2 manifest cannot carry the v1
shape.

``write_zip`` is the only writer of this contract, shared by every producer
(EoScan, container_prep). It reads no settings and takes no ORM object: frame
bytes come from ``frame_source(measurement_uid)``, because producers resolve
them from different layouts.

The reading side lives here too, so the contract and its history are in one
place: ``check`` (required keys, known version), ``upgrade`` (an older
version's shape to the current one — archives are never rewritten, so a
reader upgrades what it reads) and ``members`` (every file a manifest names).
What a particular upload must also satisfy — its zip, its instance, its bytes
— is the reader's business.
"""

import dataclasses
import hashlib
import json
import posixpath
import zipfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from . import legacy_omniscan
from .writer import SessionPayload, SetPayload

MANIFEST_VERSION = 2
# every version a reader can upgrade from
KNOWN_VERSIONS = frozenset({1, 2})

# bytes, or a path to read them from
Source = Union[bytes, bytearray, memoryview, str, Path]
# measurement_uid -> Source
FrameSource = Callable[[str], Source]
# detector_id -> Source (.npy bytes)
MaskSource = Callable[[int], Source]


def write_zip(
    payload: SessionPayload,
    frame_source: FrameSource,
    out_path: Union[str, Path],
    *,
    header_source: Optional[FrameSource] = None,
    mask_source: Optional[MaskSource] = None,
) -> Path:
    """Write the upload zip for ``payload`` to ``out_path`` and return it.

    ``frame_source(measurement_uid)`` yields the original image bytes.
    ``header_source(measurement_uid)`` is consulted only for measurements whose
    ``metadata_file_path`` differs from ``file_path`` (a separate ``.dsc``);
    omitting it while such a measurement exists is an error.
    ``mask_source(detector_id)`` is consulted for every catalog detector that
    declares ``mask_file_path``; when omitted, masks are not shipped and
    ``mask_ref`` is null.
    """
    uid = payload.session_uid
    out_path = Path(out_path)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        mask_refs = {}
        if mask_source is not None:
            for ds in payload.detector_sets:
                for det in ds.detectors:
                    if det.mask_file_path:
                        rel = f"masks/{det.detector_id}.npy"
                        zf.writestr(f"{uid}/{rel}", _read(mask_source(det.detector_id)))
                        mask_refs[det.detector_id] = rel

        manifest = build_manifest(payload, mask_refs)
        for sp, ms in zip(payload.sets, manifest["sets"]):
            for mp, mm in zip(sp.measurements, ms["measurements"]):
                base = f"{uid}/" + _measurement_dir(sp.set_uid, mp.measurement_uid)
                image = _read(frame_source(mp.measurement_uid))
                mm["sha256"] = hashlib.sha256(image).hexdigest()
                zf.writestr(base + mm["file"], image)
                if mm["header"]:
                    if header_source is None:
                        raise ValueError(
                            f"measurement {mp.measurement_uid} has a separate header "
                            f"({mp.metadata_file_path}) but no header_source was given")
                    zf.writestr(base + mm["header"], _read(header_source(mp.measurement_uid)))
        zf.writestr(f"{uid}/manifest.json", json.dumps(manifest, indent=1, default=_json_default))
    return out_path


def build_manifest(payload: SessionPayload, mask_refs: Optional[dict] = None) -> dict:
    """The manifest dict for ``payload`` — everything except the per-member
    ``sha256`` values, which ``write_zip`` fills as it reads the bytes."""
    mask_refs = mask_refs or {}
    _check_namespaces(payload.session_metadata, "session_metadata")
    for sp in payload.sets:
        _check_namespaces(sp.metadata, f"set {sp.set_uid} metadata")
    sample = None
    if payload.sample_clinical_name is not None:
        metadata = {**(payload.sample_metadata or {}), **payload.sample_metadata_attrs}
        sample = {
            "clinical_name": payload.sample_clinical_name,
            "patient_clinical_name": payload.patient_clinical_name,
            "sample_type_name": payload.sample_type_name,
            "metadata": metadata or None,
        }
    return {
        "manifest_version": MANIFEST_VERSION,
        "producer": {
            "software": payload.producer_software,
            "version": payload.producer_version,
            "instance_id": payload.instance_id,
        },
        "session": {
            "uid": payload.session_uid,
            "pk": payload.session_pk,
            "category": payload.session_category,
            "status": payload.status,
            "operator_username": payload.operator_username,
            "started_at": payload.started_at,
            "completed_at": payload.completed_at,
            "machine": {
                "serial": payload.machine_serial,
                "type": payload.machine_type,
                "location": payload.machine_location,
                "wavelength_angstrom": payload.wavelength_angstrom,
                "beam_energy_keV": payload.beam_energy_keV,
                "source_type": payload.source_type,
            },
            "sample": sample,
            "protocol_snapshot": payload.protocol_snapshot,
            "metadata": payload.session_metadata or None,
            "metadata_schema": payload.session_metadata_schema or None,
            "dependencies": [
                {"kind": d.role, "uid": d.session_uid, "session_pk": d.session_pk}
                for d in payload.dependencies
            ],
        },
        "detector_sets": [
            {
                "detector_set_id": ds.detector_set_id,
                "hardware_id": ds.hardware_id,
                "layout": ds.layout,
                "primary_detector_id": ds.primary_detector_id,
                "detectors": [
                    {
                        **{k: v for k, v in dataclasses.asdict(det).items() if k != "mask_file_path"},
                        "mask_ref": mask_refs.get(det.detector_id),
                    }
                    for det in ds.detectors
                ],
            }
            for ds in payload.detector_sets
        ],
        "sets": [_set_entry(sp, mask_refs) for sp in payload.sets],
    }


def _set_entry(sp: SetPayload, mask_refs: dict) -> dict:
    return {
        "set_uid": sp.set_uid,
        "set_pk": sp.set_pk,
        "workflow_id": sp.workflow_id,
        "batch_id": sp.batch_id,
        "detector_set_id": sp.detector_set_id,
        "status": sp.status,
        "is_approved": sp.is_approved,
        "measurement_type_name": sp.measurement_type_name,
        "measurement_type_category": sp.measurement_type_category,
        "workflow_key": sp.workflow_key,
        "distance_mm": sp.distance_mm,
        "voltage_kv": sp.voltage_kv,
        "current_ua": sp.current_ua,
        "exposure_time_s": sp.exposure_time_s,
        "sample_thickness_mm": sp.sample_thickness_mm,
        "stage_position": sp.stage_position,
        "sample_name": sp.sample_name,
        "created_at": sp.created_at,
        "metadata": sp.metadata,
        "poni_text": sp.poni_text,
        "raw_sha256": _array_sha256(sp.raw),
        "measurements": [
            {
                "measurement_uid": m.measurement_uid,
                "measurement_pk": m.measurement_pk,
                "detector_id": m.detector_id,
                "file": "image" + _ext(m.file_path),
                "header": ("header" + _ext(m.metadata_file_path)
                           if m.metadata_file_path and m.metadata_file_path != m.file_path
                           else None),
                "mask_ref": mask_refs.get(m.detector_id),
                "sha256": None,
            }
            for m in sp.measurements
        ],
        "qc_results": [dataclasses.asdict(q) for q in sp.qc_results],
    }


class ManifestError(ValueError):
    """A manifest that breaks the contract of the version it declares."""


_SESSION_KEYS = ("uid", "category", "status", "started_at", "machine")
_SET_KEYS = (
    "set_uid", "detector_set_id", "status", "measurement_type_name", "measurement_type_category",
    "distance_mm", "voltage_kv", "current_ua", "exposure_time_s", "created_at", "measurements",
)
_MEASUREMENT_KEYS = ("measurement_uid", "detector_id", "file")


def check(manifest: dict) -> None:
    """Raise ``ManifestError`` unless ``manifest`` declares a known version and
    has every key the contract requires. Required keys only — every other
    field is optional, and a reader ``.get()``s it, so absence is accepted."""
    _require(manifest, ("manifest_version", "producer", "session", "detector_sets", "sets"), "manifest")
    if manifest["manifest_version"] not in KNOWN_VERSIONS:
        raise ManifestError(f"unknown manifest_version {manifest['manifest_version']!r}")
    _require(manifest["session"], _SESSION_KEYS, "session")
    if "wavelength_angstrom" not in manifest["session"]["machine"]:
        raise ManifestError("manifest session.machine lacks wavelength_angstrom")
    if manifest["session"]["category"] == "SAMPLE":
        sample = manifest["session"].get("sample")
        if not sample or not sample.get("clinical_name"):
            raise ManifestError("SAMPLE session without sample.clinical_name")
    for sp in manifest["sets"]:
        _require(sp, _SET_KEYS, f"set {sp.get('set_uid')}")
        for m in sp["measurements"]:
            _require(m, _MEASUREMENT_KEYS, f"measurement {m.get('measurement_uid')}")


def upgrade(manifest: dict) -> dict:
    """Bring a ``check``ed manifest to the current version's shape, in place,
    and return it. ``manifest_version`` is left as it arrived: it records what
    was sent, not what the dict now holds. The v1 steps are their own
    functions, so rows already stored from a v1 manifest can be brought to the
    same shape by the same code."""
    session = manifest["session"]
    if manifest["manifest_version"] == 1:
        session["metadata"] = upgrade_v1_session_metadata(session.get("metadata"))
        producer = manifest["producer"].get("software")
        for sp in manifest["sets"]:
            sp["metadata"] = upgrade_v1_set_metadata(sp.get("metadata"), sp, producer)
    _check_namespaces(session.get("metadata"), "manifest session.metadata")
    for sp in manifest["sets"]:
        _check_namespaces(sp.get("metadata"), f"manifest set {sp['set_uid']} metadata")
    return manifest


def upgrade_v1_session_metadata(metadata: Optional[dict]) -> Optional[dict]:
    """v1 session metadata in v2's shape: the session form's values, flat in
    v1, go under ``form``."""
    return {"form": metadata} if metadata else metadata


def upgrade_v1_set_metadata(metadata: Optional[dict], set_entry: dict,
                            producer: Optional[str]) -> dict:
    """v1 set metadata in v2's shape. ``set_entry`` is the set's manifest
    entry (or the same fields from a stored row). EoScan's v1 sets were
    already namespaces and pass unchanged; the legacy backfill's are
    ``legacy_omniscan``'s."""
    if producer == legacy_omniscan.PRODUCER:
        return legacy_omniscan.upgrade_v1_set_metadata(metadata, set_entry)
    return dict(metadata or {})


def members(manifest: dict) -> List[Tuple[str, Optional[str]]]:
    """Every file ``manifest`` names, as ``(path under the session root,
    sha256 or None)`` — measurement images and headers, then masks.
    ``manifest.json`` itself is not listed."""
    out = []
    for sp in manifest["sets"]:
        for m in sp["measurements"]:
            base = _measurement_dir(sp["set_uid"], m["measurement_uid"])
            out.append((base + m["file"], m.get("sha256")))
            if m.get("header"):
                out.append((base + m["header"], None))
    masks = {m["mask_ref"] for sp in manifest["sets"] for m in sp["measurements"] if m.get("mask_ref")}
    out.extend((rel, None) for rel in sorted(masks))
    return out


def _require(d, keys, where):
    if not isinstance(d, dict):
        raise ManifestError(f"manifest {where} is not an object")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ManifestError(f"manifest {where} lacks {missing}")


def _check_namespaces(metadata: Optional[dict], where: str) -> None:
    for key, value in (metadata or {}).items():
        if not isinstance(value, dict):
            raise ManifestError(
                f"{where}[{key!r}] is not a namespace (got {type(value).__name__}); "
                f"every value belongs under its writer's namespace")


def _measurement_dir(set_uid: str, measurement_uid: str) -> str:
    return f"sets/{set_uid}/measurements/{measurement_uid}/"


def _ext(path: str) -> str:
    return posixpath.splitext(str(path).replace("\\", "/"))[1]


def _array_sha256(arr) -> Optional[str]:
    if arr is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _read(src: Source) -> bytes:
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    return Path(src).read_bytes()


def _json_default(v: Any):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(f"manifest holds a non-JSON value: {type(v).__name__}")
