"""Structural validation for v0.3 XRD session containers."""

import dataclasses
from pathlib import Path
from typing import List, Tuple, Union

import h5py

from container.common.hdf5 import _decode, read_json_dataset

from . import schema as S

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"


@dataclasses.dataclass(frozen=True)
class ValidationError:
    severity: str
    path: str
    message: str


def _attr(obj, key, default=None):
    if key not in obj.attrs:
        return default
    return _decode(obj.attrs[key])


def validate_session_container(
    file_path: Union[str, Path]
) -> Tuple[bool, List[ValidationError]]:
    """Return ``(is_valid, errors)``; ``is_valid`` is False iff any ERROR found."""
    errors: List[ValidationError] = []

    def err(path, msg):
        errors.append(ValidationError(SEVERITY_ERROR, path, msg))

    def warn(path, msg):
        errors.append(ValidationError(SEVERITY_WARNING, path, msg))

    with h5py.File(file_path, "r") as f:
        # ----- root -----
        if _attr(f, S.ATTR_NX_CLASS) != S.NX_ROOT:
            err("/", f"root NX_class must be {S.NX_ROOT}")
        if _attr(f, S.ATTR_FORMAT) != S.FORMAT:
            err("/", f"format must be '{S.FORMAT}'")
        if _attr(f, S.ATTR_SCHEMA_VERSION) != S.SCHEMA_VERSION:
            err("/", f"schema_version must be '{S.SCHEMA_VERSION}'")
        if _attr(f, S.ATTR_CONTAINER_TYPE) != S.CONTAINER_TYPE_SESSION:
            err("/", "container_type must be 'session'")
        # instance_id is optional provenance; session_uid + container_id are required.
        for key in (S.ATTR_SESSION_UID, S.ATTR_CONTAINER_ID):
            if not _attr(f, key):
                err("/", f"missing root identity attr '{key}'")

        # ----- /session -----
        if S.GROUP_SESSION not in f:
            err("/", "missing /session group")
            return (not _has_error(errors), errors)
        session = f[S.GROUP_SESSION]
        if _attr(session, S.ATTR_NX_CLASS) != S.NX_ENTRY:
            err(S.GROUP_SESSION, f"/session NX_class must be {S.NX_ENTRY}")
        if _attr(session, S.ATTR_SESSION_PK) is None:
            err(S.GROUP_SESSION, "missing session_pk")
        if not _attr(session, S.ATTR_SESSION_UID):
            err(S.GROUP_SESSION, "missing session_uid")
        category = _attr(session, S.ATTR_CATEGORY)
        if category not in S.CATEGORIES:
            err(S.GROUP_SESSION, f"category '{category}' not in {sorted(S.CATEGORIES)}")
        sample_name = None
        if "sample" in session and S.FIELD_SAMPLE_NAME in session["sample"]:
            sample_name = _decode(session["sample"][S.FIELD_SAMPLE_NAME][()])
        if category == S.CATEGORY_SAMPLE and not sample_name:
            err(S.GROUP_SAMPLE, "SAMPLE session requires a non-empty sample/name")

        _validate_dependencies(session, category, err, warn)
        catalog = _detector_catalog(session)
        _validate_sets(session, catalog, err, warn)

    return (not _has_error(errors), errors)


def _detector_catalog(session):
    """``{detector_set_id: {detector_ids}}`` declared in the session catalog."""
    catalog = {}
    instrument = session.get("instrument")
    if instrument is None:
        return catalog
    container = instrument.get(S.NAME_DETECTOR_SETS)
    if container is None:
        return catalog
    for ds in container.values():
        ds_id = _attr(ds, S.ATTR_DETECTOR_SET_ID)
        if ds_id is None:
            continue
        ids = set()
        detectors = ds.get(S.NAME_DETECTORS)
        if detectors is not None:
            for det in detectors.values():
                det_id = _attr(det, S.ATTR_DETECTOR_ID)
                if det_id is not None:
                    ids.add(int(det_id))
        catalog[int(ds_id)] = ids
    return catalog


def _validate_dependencies(session, category, err, warn):
    roles_present = set()
    edges = read_json_dataset(session, S.DS_DEPENDENCIES, default=[]) \
        if S.DS_DEPENDENCIES in session else []
    for idx, edge in enumerate(edges):
        path = f"{S.GROUP_DEPENDENCIES}[{idx}]"
        role = edge.get(S.ATTR_ROLE)
        if role not in S.DEPENDENCY_ROLES:
            err(path, f"dependency role '{role}' invalid")
        else:
            roles_present.add(role)
        # session_pk is optional readability; the uid is the real reference key.
        if not edge.get(S.ATTR_SESSION_UID):
            err(path, "dependency missing session_uid")
    for expected in S.EXPECTED_ROLES.get(category, ()):  # WARNING only
        if expected not in roles_present:
            warn(S.GROUP_DEPENDENCIES,
                 f"{category} session expected a '{expected}' dependency edge")


def _validate_sets(session, catalog, err, warn):
    if "sets" not in session:
        warn(S.GROUP_SETS, "session has no sets")
        return
    for set_name, set_grp in session["sets"].items():
        path = f"{S.GROUP_SETS}/{set_name}"
        if _attr(set_grp, S.ATTR_SET_PK) is None:
            err(path, "set missing set_pk")
        if not _attr(set_grp, S.ATTR_SET_UID):
            err(path, "set missing set_uid")

        # The capture references one catalogued detector set; its measurements'
        # detectors must belong to that set's nested catalog.
        ds_id = _attr(set_grp, S.ATTR_DETECTOR_SET_ID)
        allowed_det_ids = set()
        if ds_id is None:
            err(path, "set missing detector_set_id reference")
        elif int(ds_id) not in catalog:
            err(path, f"detector_set_id {ds_id} not in session detector-set catalog")
        else:
            allowed_det_ids = catalog[int(ds_id)]

        measurements = set_grp.get(S.GROUP_MEASUREMENTS)
        det_names = [n for n in measurements] if measurements else []
        if not det_names:
            err(path, "set has no measurements")
        for det_name in det_names:
            det = measurements[det_name]
            mpath = f"{path}/measurements/{det_name}"
            if _attr(det, S.ATTR_NX_CLASS) != S.NX_DETECTOR:
                warn(mpath, f"measurement NX_class should be {S.NX_DETECTOR}")
            if not _attr(det, S.ATTR_MEASUREMENT_UID):
                err(mpath, "measurement missing measurement_uid")
            det_id = _attr(det, S.ATTR_DETECTOR_ID)
            if det_id is None:
                err(mpath, "measurement missing detector_id reference")
            elif int(det_id) not in allowed_det_ids:
                err(mpath, f"detector_id {det_id} not in detector set {ds_id}'s catalog")

        for prod in (S.DS_RAW_2D, S.DS_PROCESSED):
            if prod in set_grp:
                _validate_2d(set_grp[prod], f"{path}/{prod}", err)

        if S.GROUP_INTEGRATION in set_grp:
            _validate_integration(set_grp[S.GROUP_INTEGRATION],
                                  f"{path}/{S.GROUP_INTEGRATION}", err, warn)

        if S.GROUP_QC in set_grp:
            for check_name, check in set_grp[S.GROUP_QC].items():
                verdict = _attr(check, S.ATTR_VERDICT)
                if verdict not in S.VERDICTS:
                    err(f"{path}/qc/{check_name}", f"verdict '{verdict}' invalid")



def _validate_2d(grp, path, err):
    if _attr(grp, S.ATTR_NX_CLASS) != S.NX_DATA:
        err(path, f"NX_class must be {S.NX_DATA}")
    if _attr(grp, S.ATTR_SIGNAL) != S.DS_DATA:
        err(path, "signal must be 'data'")
    data = grp.get(S.DS_DATA)
    if data is None or data.ndim != 2:
        err(path, "expected a 2D 'data' dataset")


def _validate_integration(grp, path, err, warn):
    if _attr(grp, S.ATTR_NX_CLASS) != S.NX_DATA:
        warn(path, f"integration NX_class should be {S.NX_DATA}")
    if _attr(grp, S.ATTR_SIGNAL) != S.DS_I:
        warn(path, "integration signal should be 'i' (auto-plot contract)")
    if _attr(grp, S.ATTR_AXES) != S.DS_Q:
        warn(path, "integration axes should be 'q' (auto-plot contract)")
    q, i = grp.get(S.DS_Q), grp.get(S.DS_I)
    if q is None or i is None:
        err(path, "integration must have 'q' and 'i' datasets")
    elif q.shape != i.shape:
        err(path, f"q/i length mismatch: {q.shape} vs {i.shape}")


def _has_error(errors: List[ValidationError]) -> bool:
    return any(e.severity == SEVERITY_ERROR for e in errors)
