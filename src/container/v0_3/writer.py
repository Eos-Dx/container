"""One-shot builder for the v0.3 XRD session container.

The whole file is written in a single pass at session end from a fully-populated
``SessionPayload`` and then closed — write-once, no append, no lock step. The
payload dataclasses are the contract a producer (e.g. EoScan) fills.

This container is the *usable* artifact. The rule: **the h5 holds the decoded
products plus what's directly tied to a measurement and usable by analysts/tools;
things not directly usable (only constituent inputs for inference) stay out.** So
each measurement holds its decoded 2D frame (what that detector produced), decoded
mask, and the ``.dsc`` detector-meta header (a blob — directly file-related); the
set holds the decoded ``raw`` composite + ``processed`` matrix + 1D
``integration``; plus QC, metadata, processing and the detector catalog. Per
detector is kept full (frame + mask + meta) even though ``set/raw`` already
stitches all detectors — redundancy on purpose.

The unsummed sub-frames are NOT embedded (constituent, not directly usable). Each
measurement's original vendor source file is embedded as ``raw_file`` so the h5 is
self-contained; ALL original vendor files ALSO ship in a sibling ``.zip`` keyed by
the same ``session_uid`` (the complete cold archive; EoScan-side follow-up), and the
h5 keeps the per-detector file-path pointers (``file_path`` etc.) into that zip.
"""

import dataclasses
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import h5py
import numpy as np

from container.common import hdf5 as H
from container.common.ids import (
    generate_container_id,
    now_timestamp,
    sanitize_filename_token,
    today_token,
)

from . import schema as S


# ====================== Payload contract ======================
@dataclasses.dataclass
class DependencyRef:
    """A container this session depends on, referenced by the target's uid.

    The producer (EoScan) supplies the target session's persisted ``session_uid``
    directly — it holds it in the DB — so no derivation is needed. The optional
    ``session_pk`` rides along for human readability only.
    """
    role: str                 # "calibration" | "system"
    session_uid: str
    session_pk: Optional[int] = None


@dataclasses.dataclass
class QCResultPayload:
    check_name: str
    verdict: str
    message: str
    metrics: dict
    parameters_snapshot: dict
    created_at: str
    # binding priority — producers supply results sorted by it (the group-name
    # index encodes that order; the raw value rides along as an attr)
    priority: Optional[int] = None


@dataclasses.dataclass
class IntegrationPayload:
    q: Any                    # 1D array-like (nm^-1)
    i: Any                    # 1D array-like
    npt: int
    q_unit: str = "nm^-1"
    source: str = "pyFAI"
    sigma: Optional[Any] = None


@dataclasses.dataclass
class ProcessingStepPayload:
    step_name: str
    started_at: str
    finished_at: str
    params: dict
    input_ref: Optional[str] = None
    output_ref: Optional[str] = None


@dataclasses.dataclass
class DetectorSpec:
    """A physical detector — stored once in the session catalog, referenced by
    measurements via the numeric ``detector_id`` (intra-container key). The
    string ``hardware_id`` is retained as the external/human identity."""
    detector_id: int
    hardware_id: str
    manufacturer: str
    model: str
    pixel_size_um: float
    width_px: int
    height_px: int
    sensor_thickness_um: float
    material: str
    mask_file_path: Optional[str] = None


@dataclasses.dataclass
class DetectorSetSpec:
    """A detector set — stored once in the session catalog and referenced by a
    capture (set) via the numeric ``detector_set_id``. Carries the geometry that
    is fixed for this set within the session (``layout`` + ``primary_detector_id``)
    and its own detectors (each detector belongs to exactly one set). Today a
    session has one of these; the catalog generalizes to many (SAXS+WAXS)."""
    detector_set_id: int
    hardware_id: str
    layout: dict
    detectors: List[DetectorSpec]
    primary_detector_id: Optional[int] = None


@dataclasses.dataclass
class MeasurementPayload:
    """One detector's frame within a set — references a catalog detector by
    numeric ``detector_id``. Carries the per-detector identity, the source-file
    pointers (which resolve into the sibling raw ``.zip``), and the *decoded*
    products: the 2D ``data`` frame this detector produced and its decoded
    ``mask``. The original vendor source bytes are embedded as ``raw_file`` so
    the container is self-contained; the zip stays a redundant external copy."""
    measurement_pk: int
    measurement_uid: str        # globally-unique id (supplied by the producer)
    detector_id: int            # references a DetectorSpec in the session catalog
    file_path: str              # pointer into the raw zip
    metadata_file_path: str     # pointer into the raw zip
    mask_file_path: Optional[str] = None
    data: Optional[Any] = None  # decoded 2D frame (what this detector produced)
    mask: Optional[Any] = None  # decoded 2D mask array
    detector_meta: Optional[Union[bytes, Path, str]] = None  # .dsc header blob
    raw_file: Optional[Union[bytes, Path, str]] = None  # original vendor source bytes


@dataclasses.dataclass
class SetPayload:
    set_pk: int
    set_uid: str                # globally-unique id (supplied by the producer)
    workflow_id: str
    batch_id: str               # groups simultaneous captures across detector sets
    detector_set_id: int        # references a DetectorSetSpec in the session catalog
    status: str
    is_approved: Optional[bool]
    measurement_type_name: str
    measurement_type_category: str
    workflow_key: str           # acquisition orchestration workflow (provenance)
    distance_mm: float
    voltage_kv: float
    current_ua: float
    exposure_time_s: float
    sample_thickness_mm: Optional[float]
    stage_position: Optional[float]
    sample_name: Optional[str]
    created_at: str
    metadata: dict
    processing_config: dict
    measurements: List[MeasurementPayload]
    raw: Optional[Any] = None
    processed: Optional[Any] = None
    qc_results: List[QCResultPayload] = dataclasses.field(default_factory=list)
    integration: Optional[IntegrationPayload] = None
    processing_steps: List[ProcessingStepPayload] = dataclasses.field(default_factory=list)
    poni_text: Optional[str] = None
    preview: Optional[Union[bytes, Path, str]] = None


@dataclasses.dataclass
class SessionPayload:
    session_uid: str            # globally-unique id (supplied by the producer)
    session_pk: int
    session_category: str
    status: str
    operator_username: str
    machine_serial: str
    machine_type: str
    machine_location: str
    wavelength_angstrom: float
    beam_energy_keV: float
    source_type: str
    started_at: str
    completed_at: Optional[str]
    # session-level catalog of detector sets. One per session today; each carries
    # its own geometry (layout + primary_detector_id) and detectors. Geometry must
    # not change for a given detector set within the session.
    detector_sets: List[DetectorSetSpec] = dataclasses.field(default_factory=list)
    sample_clinical_name: Optional[str] = None
    patient_clinical_name: Optional[str] = None
    sample_type_name: Optional[str] = None
    protocol_snapshot: Optional[dict] = None
    dependencies: List[DependencyRef] = dataclasses.field(default_factory=list)
    # provenance only — which deployment produced this file; NOT used for ids
    instance_id: Optional[str] = None
    producer_software: str = "unknown"
    producer_version: str = "unknown"
    sets: List[SetPayload] = dataclasses.field(default_factory=list)


# ====================== Builder ======================
def build_session_container(
    payload: SessionPayload,
    folder: Union[str, Path],
) -> Tuple[str, str, str]:
    """Write the whole container in one pass and close it.

    Returns ``(session_uid, container_id, file_path)``.
    """
    session_uid = payload.session_uid
    container_id = generate_container_id()

    # Filename leads with the human identity (category, pk, sample) and keeps a
    # short session_uid token for global uniqueness — deterministic per session
    # (same-day rebuild overwrites the same name). session_uid is an arbitrary
    # producer token, so sanitise it.
    uid_token = sanitize_filename_token(session_uid)[:8]
    sample_token = sanitize_filename_token(payload.sample_clinical_name or payload.machine_serial)
    filename = (f"{payload.session_category.lower()}_{payload.session_pk}_"
                f"{sample_token}_{today_token()}_{uid_token}.nxs.h5")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / filename

    root_attrs = {
        S.ATTR_FORMAT: S.FORMAT,
        S.ATTR_SCHEMA_VERSION: S.SCHEMA_VERSION,
        S.ATTR_CONTAINER_TYPE: S.CONTAINER_TYPE_SESSION,
        S.ATTR_INSTANCE_ID: payload.instance_id,   # provenance; skipped if None
        S.ATTR_SESSION_UID: session_uid,
        S.ATTR_CONTAINER_ID: container_id,
        S.ATTR_CREATED_AT: now_timestamp(),
        S.ATTR_PRODUCER_SOFTWARE: payload.producer_software,
        S.ATTR_PRODUCER_VERSION: payload.producer_version,
    }

    f = H.create_root(file_path, root_attrs)
    try:
        _write_session(f, payload, session_uid)
    finally:
        f.close()

    return session_uid, container_id, str(file_path)


def _write_session(f, payload: SessionPayload, session_uid: str) -> None:
    # /session carries only session identity/status; sample & instrument data
    # live (once, not duplicated) in their NeXus subgroups.
    session = H.make_group(f, "session", S.NX_ENTRY, {
        S.ATTR_SESSION_PK: payload.session_pk,
        S.ATTR_SESSION_UID: session_uid,
        S.ATTR_CATEGORY: payload.session_category,
        S.ATTR_STATUS: payload.status,
        S.ATTR_OPERATOR_USERNAME: payload.operator_username,
        S.ATTR_STARTED_AT: payload.started_at,
        S.ATTR_COMPLETED_AT: payload.completed_at,
    })

    # NXsample — names as string-field datasets. Omitted entirely when the
    # session has no sample (e.g. CALIBRATION/SYSTEM) rather than left empty.
    if any((payload.sample_clinical_name, payload.patient_clinical_name,
            payload.sample_type_name)):
        sample = H.make_group(session, "sample", S.NX_SAMPLE)
        if payload.sample_clinical_name is not None:
            H.write_scalar_dataset(sample, S.FIELD_SAMPLE_NAME, payload.sample_clinical_name)
        if payload.patient_clinical_name is not None:
            H.write_scalar_dataset(sample, S.FIELD_PATIENT_NAME, payload.patient_clinical_name)
        if payload.sample_type_name is not None:
            H.write_scalar_dataset(sample, S.FIELD_SAMPLE_TYPE, payload.sample_type_name)

    # NXinstrument — machine identity as attrs; physics as datasets-with-units.
    instrument = H.make_group(session, "instrument", S.NX_INSTRUMENT, {
        S.ATTR_MACHINE_SERIAL: payload.machine_serial,
        S.ATTR_MACHINE_TYPE: payload.machine_type,
        S.ATTR_MACHINE_LOCATION: payload.machine_location,
        S.ATTR_SOURCE_TYPE: payload.source_type,
    })
    H.write_scalar_dataset(instrument, S.FIELD_WAVELENGTH,
                           payload.wavelength_angstrom, units=S.UNIT_ANGSTROM)
    H.write_scalar_dataset(instrument, S.FIELD_BEAM_ENERGY,
                           payload.beam_energy_keV, units=S.UNIT_KEV)

    # detector-set catalog — each set stored ONCE here (geometry + its detectors),
    # referenced by captures via numeric detector_set_id and by measurements via
    # numeric detector_id, so nothing is duplicated per capture.
    if payload.detector_sets:
        _write_detector_sets(instrument, payload.detector_sets)

    if payload.protocol_snapshot is not None:
        H.write_json_dataset(session, S.DS_PROTOCOL_SNAPSHOT, payload.protocol_snapshot)

    # dependency edges as a single JSON dataset (no attr-only groups). The edge
    # references the target by its persisted session_uid (producer-supplied).
    if payload.dependencies:
        edges = [{
            S.ATTR_ROLE: dep.role,
            S.ATTR_SESSION_PK: dep.session_pk,
            S.ATTR_SESSION_UID: dep.session_uid,
        } for dep in payload.dependencies]
        H.write_json_dataset(session, S.DS_DEPENDENCIES, edges)

    # Group-name labels for detector references — hardware_id is the external/
    # human identity; numeric @*_id attrs stay canonical.
    ds_labels = {ds.detector_set_id: ds.hardware_id for ds in payload.detector_sets}
    det_labels = {det.detector_id: det.hardware_id
                  for ds in payload.detector_sets for det in ds.detectors}

    sets = H.make_group(session, "sets")
    for idx, set_payload in enumerate(payload.sets, start=1):
        _write_set(sets, S.format_set_id(idx, set_payload.measurement_type_name),
                   set_payload, ds_labels, det_labels)


def _write_detector_sets(instrument, detector_sets: List[DetectorSetSpec]) -> None:
    container = H.make_group(instrument, S.NAME_DETECTOR_SETS, S.NX_COLLECTION)
    for ds in detector_sets:
        _write_one_detector_set(container, ds)


def _write_one_detector_set(parent, ds: DetectorSetSpec) -> None:
    grp = H.make_group(parent, S.format_detector_set_id(ds.detector_set_id, ds.hardware_id),
                       S.NX_COLLECTION, {
        S.ATTR_DETECTOR_SET_ID: ds.detector_set_id,
        S.ATTR_DETECTOR_SET_HARDWARE_ID: ds.hardware_id,
        S.ATTR_PRIMARY_DETECTOR_ID: ds.primary_detector_id,
    })
    if ds.layout:
        H.write_json_dataset(grp, S.DS_LAYOUT, ds.layout)
    catalog = H.make_group(grp, S.NAME_DETECTORS)
    for det in ds.detectors:
        _write_detector(catalog, det)


def _write_detector(parent, det: DetectorSpec) -> None:
    # NXdetector — identity/descriptive as attrs; geometry as field datasets
    # with @units. Keyed by numeric detector_id; hardware_id kept as external id.
    grp = H.make_group(parent, S.format_detector_id(det.detector_id, det.hardware_id),
                       S.NX_DETECTOR, {
        S.ATTR_DETECTOR_ID: det.detector_id,
        S.ATTR_DETECTOR_HARDWARE_ID: det.hardware_id,
        S.ATTR_MANUFACTURER: det.manufacturer,
        S.ATTR_MODEL: det.model,
        S.ATTR_MATERIAL: det.material,
        S.ATTR_MASK_FILE_PATH: det.mask_file_path,
    })
    H.write_scalar_dataset(grp, S.FIELD_X_PIXEL_SIZE, det.pixel_size_um, units=S.UNIT_UM)
    H.write_scalar_dataset(grp, S.FIELD_Y_PIXEL_SIZE, det.pixel_size_um, units=S.UNIT_UM)
    H.write_scalar_dataset(grp, S.FIELD_X_PIXEL_COUNT, det.width_px, units=S.UNIT_PIXEL)
    H.write_scalar_dataset(grp, S.FIELD_Y_PIXEL_COUNT, det.height_px, units=S.UNIT_PIXEL)
    H.write_scalar_dataset(grp, S.FIELD_SENSOR_THICKNESS, det.sensor_thickness_um, units=S.UNIT_UM)


def _write_set(parent, name: str, sp: SetPayload, ds_labels: dict, det_labels: dict) -> None:
    # set attrs are pure identity/labels; physical settings live in /acquisition.
    grp = H.make_group(parent, name, S.NX_COLLECTION, {
        S.ATTR_SET_PK: sp.set_pk,
        S.ATTR_SET_UID: sp.set_uid,
        S.ATTR_WORKFLOW_ID: sp.workflow_id,
        S.ATTR_BATCH_ID: sp.batch_id,
        S.ATTR_DETECTOR_SET_ID: sp.detector_set_id,
        S.ATTR_STATUS: sp.status,
        S.ATTR_IS_APPROVED: sp.is_approved,
        S.ATTR_MEASUREMENT_TYPE_NAME: sp.measurement_type_name,
        S.ATTR_MEASUREMENT_TYPE_CATEGORY: sp.measurement_type_category,
        S.ATTR_WORKFLOW_KEY: sp.workflow_key,
        S.ATTR_SAMPLE_NAME: sp.sample_name,
        S.ATTR_CREATED_AT_SET: sp.created_at,
        S.ATTR_DEFAULT: S.GROUP_INTEGRATION,
    })
    # convenience soft link to this capture's detector-set catalog entry —
    # @detector_set_id stays canonical, but grp["detector_set"] resolves to the
    # NX_COLLECTION spec (layout + detectors) in any HDF5 reader.
    grp[S.NAME_DETECTOR_SET] = h5py.SoftLink(
        S.detector_set_path(sp.detector_set_id, ds_labels[sp.detector_set_id]))

    # acquisition conditions for this set — physical fields-with-units, grouped
    # so the set itself stays a tidy identity record.
    acq = H.make_group(grp, S.GROUP_ACQUISITION)
    H.write_scalar_dataset(acq, S.FIELD_DISTANCE, sp.distance_mm, units=S.UNIT_MM)
    H.write_scalar_dataset(acq, S.FIELD_VOLTAGE, sp.voltage_kv, units=S.UNIT_KV)
    H.write_scalar_dataset(acq, S.FIELD_CURRENT, sp.current_ua, units=S.UNIT_UA)
    H.write_scalar_dataset(acq, S.FIELD_EXPOSURE_TIME, sp.exposure_time_s, units=S.UNIT_S)
    if sp.sample_thickness_mm is not None:
        H.write_scalar_dataset(acq, S.FIELD_SAMPLE_THICKNESS,
                               sp.sample_thickness_mm, units=S.UNIT_MM)
    if sp.stage_position is not None:
        H.write_scalar_dataset(acq, S.FIELD_STAGE_POSITION,
                               sp.stage_position, units=S.UNIT_MM)

    if sp.metadata:
        H.write_json_dataset(grp, S.DS_METADATA, sp.metadata)

    if sp.raw is not None:
        _write_2d_product(grp, S.DS_RAW_2D, sp.raw)
    if sp.processed is not None:
        _write_2d_product(grp, S.DS_PROCESSED, sp.processed)

    measurements = H.make_group(grp, S.GROUP_MEASUREMENTS)
    for m in sp.measurements:
        _write_measurement(measurements, m, det_labels[m.detector_id])

    if sp.qc_results:
        qc = H.make_group(grp, S.GROUP_QC)
        for idx, result in enumerate(sp.qc_results, start=1):
            _write_qc(qc, idx, result)

    if sp.integration is not None:
        _write_integration(grp, sp.integration)

    # /processing holds the reduction recipe config + (optional) step log.
    if sp.processing_config or sp.processing_steps:
        _write_processing(grp, sp.processing_config, sp.processing_steps)

    if sp.poni_text is not None or sp.preview is not None:
        artifacts = H.make_group(grp, S.GROUP_ARTIFACTS)
        if sp.poni_text is not None:
            H.write_text_dataset(artifacts, S.DS_PONI, sp.poni_text)
        if sp.preview is not None:
            H.write_bytes_dataset(artifacts, S.DS_PREVIEW, sp.preview)


def _write_2d_product(parent, name: str, arr) -> None:
    grp = H.make_group(parent, name, S.NX_DATA, {S.ATTR_SIGNAL: S.DS_DATA})
    H.write_array_dataset(grp, S.DS_DATA, np.asarray(arr))


def _write_measurement(parent, m: MeasurementPayload, det_label: str) -> None:
    # A measurement is one detector's decoded frame in this set. Detector
    # hardware specs live once in the session catalog (under this measurement's
    # detector set); here we reference them by numeric detector_id (the set's
    # detector_set link reaches the full spec), keep file-path pointers into
    # the raw zip, and store the decoded 2D frame (+ mask) — no vendor bytes.
    attrs = {
        S.ATTR_MEASUREMENT_PK: m.measurement_pk,
        S.ATTR_MEASUREMENT_UID: m.measurement_uid,
        S.ATTR_DETECTOR_ID: m.detector_id,
        S.ATTR_FILE_PATH: m.file_path,
        S.ATTR_METADATA_FILE_PATH: m.metadata_file_path,
        S.ATTR_MASK_FILE_PATH: m.mask_file_path,
    }
    if m.data is not None:                       # NXdata auto-plot on the frame
        attrs[S.ATTR_SIGNAL] = S.DS_DATA
    grp = H.make_group(parent, S.format_detector_id(m.detector_id, det_label),
                       S.NX_DETECTOR, attrs)
    if m.data is not None:
        H.write_array_dataset(grp, S.DS_DATA, np.asarray(m.data))
    if m.mask is not None:
        H.write_array_dataset(grp, S.DS_MASK, np.asarray(m.mask))
    if m.detector_meta is not None:
        H.write_bytes_dataset(grp, S.DS_DETECTOR_META, m.detector_meta)
    if m.raw_file is not None:
        H.write_bytes_dataset(grp, S.DS_RAW_FILE, m.raw_file)


def _write_qc(parent, idx: int, result: QCResultPayload) -> None:
    # Group name = priority-ordered index + check name; identity stays in
    # @check_name (readers key on the attr, never parse the group name).
    grp = H.make_group(parent, S.format_qc_id(idx, result.check_name), attrs={
        S.ATTR_CHECK_NAME: result.check_name,
        S.ATTR_VERDICT: result.verdict,
        S.ATTR_MESSAGE: result.message,
        S.ATTR_PRIORITY: result.priority,
        S.ATTR_QC_CREATED_AT: result.created_at,
    })
    H.write_json_dataset(grp, S.DS_METRICS, result.metrics)
    H.write_json_dataset(grp, S.DS_PARAMETERS_SNAPSHOT, result.parameters_snapshot)


def _write_integration(parent, integ: IntegrationPayload) -> None:
    # q_unit is not duplicated as an attr — it lives on the q dataset's @units.
    grp = H.make_group(parent, S.GROUP_INTEGRATION, S.NX_DATA, {
        S.ATTR_SIGNAL: S.DS_I,
        S.ATTR_AXES: S.DS_Q,
        S.ATTR_NPT: integ.npt,
        S.ATTR_SOURCE: integ.source,
    })
    H.write_array_dataset(grp, S.DS_Q, np.asarray(integ.q, dtype=np.float64),
                          attrs={S.ATTR_UNITS: integ.q_unit})
    H.write_array_dataset(grp, S.DS_I, np.asarray(integ.i, dtype=np.float64))
    if integ.sigma is not None:
        H.write_array_dataset(grp, "sigma", np.asarray(integ.sigma, dtype=np.float64))


def _write_processing(parent, config: dict, steps: List[ProcessingStepPayload]) -> None:
    grp = H.make_group(parent, S.GROUP_PROCESSING)
    if config:
        H.write_json_dataset(grp, S.DS_PROCESSING_CONFIG, config)
    for idx, step in enumerate(steps, start=1):
        step_grp = H.make_group(grp, S.format_step_id(idx, step.step_name), attrs={
            S.ATTR_STEP_NAME: step.step_name,
            S.ATTR_STEP_STARTED_AT: step.started_at,
            S.ATTR_STEP_FINISHED_AT: step.finished_at,
            S.ATTR_INPUT_REF: step.input_ref,
            S.ATTR_OUTPUT_REF: step.output_ref,
        })
        H.write_json_dataset(step_grp, S.DS_PARAMS, step.params)
