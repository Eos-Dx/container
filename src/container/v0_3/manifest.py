"""The EosCloud upload contract — ``manifest.json`` + zip layout (manifest v1).

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

``write_zip`` is the only writer of this contract, shared by every producer
(EoScan, container_prep). It reads no settings and takes no ORM object: frame
bytes come from ``frame_source(measurement_uid)``, because producers resolve
them from different layouts.
"""

import dataclasses
import hashlib
import json
import posixpath
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from .writer import SessionPayload, SetPayload

MANIFEST_VERSION = 1

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
                base = f"{uid}/sets/{sp.set_uid}/measurements/{mp.measurement_uid}/"
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
        "metadata": {**sp.metadata, **sp.metadata_attrs},
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
