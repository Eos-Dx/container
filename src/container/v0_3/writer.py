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

The unsummed sub-frames are NOT embedded (constituent, not directly usable), and
ALL original vendor files ship in a sibling ``.zip`` keyed by the same
``session_uid`` (the complete cold archive; EoScan-side follow-up). The h5 keeps
the per-detector file-path pointers (``file_path`` etc.) into that zip.
"""

import dataclasses
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import h5py
import numpy as np

from container.common import hdf5 as H
from container.common.ids import (
    format_session_container_filename,
    generate_container_id,
    make_global_uid,
    now_timestamp,
)

from . import schema as S


# ====================== Payload contract ======================
@dataclasses.dataclass
class DependencyRef:
    """A container this session depends on, referenced by the target's pk.

    The builder derives the target ``session_uid`` from ``session_pk`` — no
    lookup, no need for the target container to exist yet.
    """
    role: str                 # "calibration" | "system"
    session_pk: int


@dataclasses.dataclass
class QCResultPayload:
    check_name: str
    verdict: str
    message: str
    metrics: dict
    parameters_snapshot: dict
    created_at: str


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
class MeasurementPayload:
    """One detector's frame within a set — references a catalog detector by
    numeric ``detector_id``. Carries the per-detector identity, the source-file
    pointers (which resolve into the sibling raw ``.zip``), and the *decoded*
    products: the 2D ``data`` frame this detector produced and its decoded
    ``mask``. No vendor bytes — originals live in the zip."""
    measurement_pk: int
    detector_id: int            # references a DetectorSpec in the session catalog
    file_path: str              # pointer into the raw zip
    metadata_file_path: str     # pointer into the raw zip
    mask_file_path: Optional[str] = None
    data: Optional[Any] = None  # decoded 2D frame (what this detector produced)
    mask: Optional[Any] = None  # decoded 2D mask array
    detector_meta: Optional[Union[bytes, Path, str]] = None  # .dsc header blob


@dataclasses.dataclass
class SetPayload:
    set_pk: int
    workflow_id: str
    batch_id: str
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
    instance_id: str
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
    # session-level detector set (one per session — geometry must not change mid-session)
    detector_set_hardware_id: str = ""
    detector_set_layout: dict = dataclasses.field(default_factory=dict)
    detectors: List[DetectorSpec] = dataclasses.field(default_factory=list)
    sample_clinical_name: Optional[str] = None
    patient_clinical_name: Optional[str] = None
    sample_type_name: Optional[str] = None
    protocol_snapshot: Optional[dict] = None
    dependencies: List[DependencyRef] = dataclasses.field(default_factory=list)
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
    instance_id = payload.instance_id
    session_uid = make_global_uid(instance_id, S.ENTITY_SESSION, payload.session_pk)
    container_id = generate_container_id()

    sample_token = payload.sample_clinical_name or payload.machine_serial
    filename = format_session_container_filename(session_uid, sample_id=sample_token)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / filename

    root_attrs = {
        S.ATTR_FORMAT: S.FORMAT,
        S.ATTR_SCHEMA_VERSION: S.SCHEMA_VERSION,
        S.ATTR_CONTAINER_TYPE: S.CONTAINER_TYPE_SESSION,
        S.ATTR_INSTANCE_ID: instance_id,
        S.ATTR_SESSION_UID: session_uid,
        S.ATTR_CONTAINER_ID: container_id,
        S.ATTR_CREATED_AT: now_timestamp(),
        S.ATTR_PRODUCER_SOFTWARE: payload.producer_software,
        S.ATTR_PRODUCER_VERSION: payload.producer_version,
    }

    f = H.create_root(file_path, root_attrs)
    try:
        _write_session(f, payload, instance_id, session_uid)
    finally:
        f.close()

    return session_uid, container_id, str(file_path)


def _write_session(f, payload: SessionPayload, instance_id: str, session_uid: str) -> None:
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

    # detector_set — one per session (geometry is fixed for the session). The
    # detector catalog is stored ONCE here and referenced by measurements via
    # numeric detector_id, so shared detectors aren't duplicated per set.
    if payload.detectors or payload.detector_set_hardware_id or payload.detector_set_layout:
        _write_detector_set(instrument, payload)

    if payload.protocol_snapshot is not None:
        H.write_json_dataset(session, S.DS_PROTOCOL_SNAPSHOT, payload.protocol_snapshot)

    # dependency edges as a single JSON dataset (no attr-only groups)
    if payload.dependencies:
        edges = [{
            S.ATTR_ROLE: dep.role,
            S.ATTR_SESSION_PK: dep.session_pk,
            S.ATTR_SESSION_UID: make_global_uid(
                instance_id, S.ENTITY_SESSION, dep.session_pk),
        } for dep in payload.dependencies]
        H.write_json_dataset(session, S.DS_DEPENDENCIES, edges)

    sets = H.make_group(session, "sets")
    for idx, set_payload in enumerate(payload.sets, start=1):
        _write_set(sets, S.format_set_id(idx), set_payload, instance_id)


def _write_detector_set(instrument, payload: SessionPayload) -> None:
    ds = H.make_group(instrument, S.NAME_DETECTOR_SET, S.NX_COLLECTION, {
        S.ATTR_DETECTOR_SET_HARDWARE_ID: payload.detector_set_hardware_id,
    })
    if payload.detector_set_layout:
        H.write_json_dataset(ds, S.DS_LAYOUT, payload.detector_set_layout)
    if payload.detectors:
        catalog = H.make_group(ds, S.NAME_DETECTORS)
        for det in payload.detectors:
            _write_detector(catalog, det)


def _write_detector(parent, det: DetectorSpec) -> None:
    # NXdetector — identity/descriptive as attrs; geometry as field datasets
    # with @units. Keyed by numeric detector_id; hardware_id kept as external id.
    grp = H.make_group(parent, S.format_detector_id(det.detector_id), S.NX_DETECTOR, {
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


def _write_set(parent, name: str, sp: SetPayload, instance_id: str) -> None:
    set_uid = make_global_uid(instance_id, S.ENTITY_SET, sp.set_pk)
    # set attrs are pure identity/labels; physical settings live in /acquisition.
    grp = H.make_group(parent, name, S.NX_COLLECTION, {
        S.ATTR_SET_PK: sp.set_pk,
        S.ATTR_SET_UID: set_uid,
        S.ATTR_WORKFLOW_ID: sp.workflow_id,
        S.ATTR_BATCH_ID: sp.batch_id,
        S.ATTR_STATUS: sp.status,
        S.ATTR_IS_APPROVED: sp.is_approved,
        S.ATTR_MEASUREMENT_TYPE_NAME: sp.measurement_type_name,
        S.ATTR_MEASUREMENT_TYPE_CATEGORY: sp.measurement_type_category,
        S.ATTR_WORKFLOW_KEY: sp.workflow_key,
        S.ATTR_SAMPLE_NAME: sp.sample_name,
        S.ATTR_CREATED_AT_SET: sp.created_at,
        S.ATTR_DEFAULT: S.GROUP_INTEGRATION,
    })

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
        _write_measurement(measurements, m, instance_id)

    if sp.qc_results:
        qc = H.make_group(grp, S.GROUP_QC)
        for result in sp.qc_results:
            _write_qc(qc, result)

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


def _write_measurement(parent, m: MeasurementPayload, instance_id: str) -> None:
    # A measurement is one detector's decoded frame in this set. Detector
    # hardware specs live once in the session catalog; here we reference them by
    # numeric detector_id, keep file-path pointers into the raw zip, and store
    # the decoded 2D frame (+ mask) — no opaque vendor bytes.
    attrs = {
        S.ATTR_MEASUREMENT_PK: m.measurement_pk,
        S.ATTR_MEASUREMENT_UID: make_global_uid(
            instance_id, S.ENTITY_MEASUREMENT, m.measurement_pk),
        S.ATTR_DETECTOR_ID: m.detector_id,
        S.ATTR_FILE_PATH: m.file_path,
        S.ATTR_METADATA_FILE_PATH: m.metadata_file_path,
        S.ATTR_MASK_FILE_PATH: m.mask_file_path,
    }
    if m.data is not None:                       # NXdata auto-plot on the frame
        attrs[S.ATTR_SIGNAL] = S.DS_DATA
    grp = H.make_group(parent, S.format_detector_id(m.detector_id), S.NX_DETECTOR, attrs)
    # convenience soft link to the catalog entry — @detector_id stays canonical,
    # but `grp["detector"]` resolves to the NXdetector spec in any HDF5 reader.
    grp[S.NAME_DETECTOR] = h5py.SoftLink(
        f"{S.GROUP_DETECTORS}/{S.format_detector_id(m.detector_id)}")
    if m.data is not None:
        H.write_array_dataset(grp, S.DS_DATA, np.asarray(m.data))
    if m.mask is not None:
        H.write_array_dataset(grp, S.DS_MASK, np.asarray(m.mask))
    if m.detector_meta is not None:
        H.write_bytes_dataset(grp, S.DS_DETECTOR_META, m.detector_meta)


def _write_qc(parent, result: QCResultPayload) -> None:
    grp = H.make_group(parent, result.check_name, attrs={
        S.ATTR_VERDICT: result.verdict,
        S.ATTR_MESSAGE: result.message,
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
        step_grp = H.make_group(grp, S.format_step_id(idx), attrs={
            S.ATTR_STEP_NAME: step.step_name,
            S.ATTR_STEP_STARTED_AT: step.started_at,
            S.ATTR_STEP_FINISHED_AT: step.finished_at,
            S.ATTR_INPUT_REF: step.input_ref,
            S.ATTR_OUTPUT_REF: step.output_ref,
        })
        H.write_json_dataset(step_grp, S.DS_PARAMS, step.params)
