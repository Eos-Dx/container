"""XRD acquisition session container — v0.3 schema constants and ID helpers.

A vendor-neutral layout for an XRD acquisition modelled as
**session -> sets -> measurements**, with per-set quality-control results, 1D
azimuthal integration, metadata and (optional) processing provenance.

NeXus base classes are worn only where they earn it (NXroot/NXentry/NXsample/
NXinstrument/NXdetector, and NXdata @signal/@axes on /integration so the 1D
curve auto-plots in generic NeXus viewers). No /entry wrapper, no application
definition. Identifiers carry the producing system's row ids (``*_pk``) plus a
deterministic global ``*_uid`` so containers stay self-describing and
cross-referenceable regardless of who produced them.
"""

from container.common.constants import (  # noqa: F401  (re-exported for writers)
    ATTR_AXES,
    ATTR_CONTAINER_TYPE,
    ATTR_DEFAULT,
    ATTR_FORMAT,
    ATTR_NX_CLASS,
    ATTR_SCHEMA_VERSION,
    ATTR_SIGNAL,
    ATTR_UNITS,
    CONTAINER_TYPE_SESSION,
    NX_COLLECTION,
    NX_DATA,
    NX_DETECTOR,
    NX_ENTRY,
    NX_INSTRUMENT,
    NX_ROOT,
    NX_SAMPLE,
)
from container.common.ids import sanitize_filename_token

# ================== Self-description =======================
SCHEMA_VERSION = "0.3"
FORMAT = "xrd-session"

# ================== make_global_uid entity namespaces =====
ENTITY_SESSION = "session"
ENTITY_SET = "set"
ENTITY_MEASUREMENT = "measurement"

# ================== Session categories ====================
CATEGORY_SAMPLE = "SAMPLE"
CATEGORY_CALIBRATION = "CALIBRATION"
CATEGORY_SYSTEM = "SYSTEM"
CATEGORIES = frozenset({CATEGORY_SAMPLE, CATEGORY_CALIBRATION, CATEGORY_SYSTEM})

# ================== QC verdicts ===========================
VERDICTS = frozenset({"PASS", "WARNING", "FAIL", "ERROR"})

# ================== Dependency roles ======================
ROLE_CALIBRATION = "calibration"
ROLE_SYSTEM = "system"
DEPENDENCY_ROLES = frozenset({ROLE_CALIBRATION, ROLE_SYSTEM})

# Which dependency roles each session category is expected to declare.
EXPECTED_ROLES = {
    CATEGORY_SAMPLE: (ROLE_CALIBRATION, ROLE_SYSTEM),
    CATEGORY_CALIBRATION: (ROLE_SYSTEM,),
    CATEGORY_SYSTEM: (),
}

# ================== Root attributes =======================
ATTR_INSTANCE_ID = "instance_id"
ATTR_SESSION_UID = "session_uid"          # logical identity + edge key + filename
ATTR_CONTAINER_ID = "container_id"        # physical artifact id (cloud key/audit)
ATTR_CREATED_AT = "created_at"
ATTR_PRODUCER_SOFTWARE = "producer_software"
ATTR_PRODUCER_VERSION = "producer_version"

# ================== Group paths ===========================
GROUP_SESSION = "/session"
GROUP_SAMPLE = "/session/sample"
GROUP_INSTRUMENT = "/session/instrument"
# A session-level CATALOG of detector sets (ds_<pk>), each holding its own
# layout + nested detector catalog. A capture (set) references one by numeric
# @detector_set_id + a soft link, mirroring how measurements reference detectors.
# Today every session uses a single detector set; the catalog generalizes so a
# future multi-detector-set capture (SAXS+WAXS at one batch) needs no reshape.
GROUP_DETECTOR_SETS = "/session/instrument/detector_sets"
GROUP_DEPENDENCIES = "/session/dependencies"
GROUP_SETS = "/session/sets"

# detector-set / detector child names (relative)
NAME_DETECTOR_SETS = "detector_sets"
NAME_DETECTOR_SET = "detector_set"  # per-set soft link → catalog detector-set entry
NAME_DETECTORS = "detectors"
DS_LAYOUT = "layout"

# Per-set child group names (relative)
GROUP_ACQUISITION = "acquisition"
GROUP_MEASUREMENTS = "measurements"
GROUP_QC = "qc"
GROUP_INTEGRATION = "integration"
GROUP_PROCESSING = "processing"
GROUP_ARTIFACTS = "artifacts"

# ================== Session attributes ====================
ATTR_SESSION_PK = "session_pk"
ATTR_CATEGORY = "category"
ATTR_STATUS = "status"
ATTR_OPERATOR_USERNAME = "operator_username"
ATTR_STARTED_AT = "started_at"
ATTR_COMPLETED_AT = "completed_at"
ATTR_MACHINE_SERIAL = "machine_serial"
ATTR_MACHINE_TYPE = "machine_type"
ATTR_MACHINE_LOCATION = "machine_location"
ATTR_WAVELENGTH_ANGSTROM = "wavelength_angstrom"
ATTR_BEAM_ENERGY_KEV = "beam_energy_keV"
ATTR_SOURCE_TYPE = "source_type"
ATTR_SAMPLE_CLINICAL_NAME = "sample_clinical_name"
ATTR_PATIENT_CLINICAL_NAME = "patient_clinical_name"
ATTR_SAMPLE_TYPE_NAME = "sample_type_name"

DS_PROTOCOL_SNAPSHOT = "protocol_snapshot"

# ================== Dependency-edge attributes ============
ATTR_ROLE = "role"
# (reuses ATTR_SESSION_PK and ATTR_SESSION_UID for the target)

# ================== Set attributes ========================
ATTR_SET_PK = "set_pk"
ATTR_SET_UID = "set_uid"
ATTR_WORKFLOW_ID = "workflow_id"
ATTR_BATCH_ID = "batch_id"
ATTR_IS_APPROVED = "is_approved"
ATTR_MEASUREMENT_TYPE_NAME = "measurement_type_name"
ATTR_MEASUREMENT_TYPE_CATEGORY = "measurement_type_category"
ATTR_WORKFLOW_KEY = "workflow_key"   # acquisition orchestration workflow (provenance)
ATTR_DETECTOR_SET_ID = "detector_set_id"            # numeric intra-container detector-set key
ATTR_DETECTOR_SET_HARDWARE_ID = "detector_set_hardware_id"  # external/human id on catalog
ATTR_PRIMARY_DETECTOR_ID = "primary_detector_id"    # which detector anchors the geometry
ATTR_DISTANCE_MM = "distance_mm"
ATTR_VOLTAGE_KV = "voltage_kv"
ATTR_CURRENT_UA = "current_ua"
ATTR_EXPOSURE_TIME_S = "exposure_time_s"
ATTR_SAMPLE_THICKNESS_MM = "sample_thickness_mm"
ATTR_STAGE_POSITION = "stage_position"
ATTR_SAMPLE_NAME = "sample_name"
ATTR_CREATED_AT_SET = "created_at"

DS_METADATA = "metadata"
DS_PROCESSING_CONFIG = "config"   # reduction recipe — lives inside /processing

# Set-level 2D products — the set is the primary unit: raw (stitched across
# detectors per geometry) and processed (final matrix).
DS_RAW_2D = "raw"
DS_PROCESSED = "processed"
DS_DATA = "data"

# ================== Detector catalog + measurement refs ===
ATTR_DETECTOR_ID = "detector_id"          # numeric intra-container detector key
ATTR_MEASUREMENT_PK = "measurement_pk"
ATTR_MEASUREMENT_UID = "measurement_uid"
ATTR_DETECTOR_HARDWARE_ID = "detector_hardware_id"  # external/human id on catalog
ATTR_MANUFACTURER = "manufacturer"
ATTR_MODEL = "model"
ATTR_PIXEL_SIZE_UM = "pixel_size_um"
ATTR_WIDTH_PX = "width_px"
ATTR_HEIGHT_PX = "height_px"
ATTR_SENSOR_THICKNESS_UM = "sensor_thickness_um"
ATTR_MATERIAL = "material"
ATTR_FILE_PATH = "file_path"
ATTR_METADATA_FILE_PATH = "metadata_file_path"
ATTR_MASK_FILE_PATH = "mask_file_path"

# per-measurement: decoded products (DS_DATA reused; mask as decoded array) +
# the .dsc detector-meta header, kept as a blob since it's tied to the file.
DS_MASK = "mask"
DS_DETECTOR_META = "detector_meta"   # .dsc header bytes (directly file-related)
DS_RAW_FILE = "raw_file"             # original vendor source bytes (.gfrm/.png/.h5)

# ================== QC attributes =========================
ATTR_CHECK_NAME = "check_name"
ATTR_VERDICT = "verdict"
ATTR_MESSAGE = "message"
ATTR_PRIORITY = "priority"   # binding priority (execution scheduling)
ATTR_QC_CREATED_AT = "created_at"
DS_METRICS = "metrics"
DS_PARAMETERS_SNAPSHOT = "parameters_snapshot"

# ================== Integration (NXdata) ==================
DS_Q = "q"
DS_I = "i"
ATTR_NPT = "npt"
ATTR_Q_UNIT = "q_unit"
ATTR_SOURCE = "source"

# ================== Processing log ========================
ATTR_STEP_NAME = "step_name"
ATTR_STEP_STARTED_AT = "started_at"
ATTR_STEP_FINISHED_AT = "finished_at"
ATTR_INPUT_REF = "input_ref"
ATTR_OUTPUT_REF = "output_ref"
DS_PARAMS = "params"

# ================== Artifacts =============================
DS_PONI = "poni"
DS_PREVIEW = "preview"

# ================== NeXus field datasets (with @units) ====
# Physics quantities are stored as datasets-with-units (not attrs) so generic
# NeXus tools (pyFAI/DAWN/silx) can read geometry/energy directly.
UNIT_ANGSTROM = "angstrom"
UNIT_KEV = "keV"
UNIT_MM = "mm"
UNIT_UM = "um"
UNIT_KV = "kV"
UNIT_UA = "uA"
UNIT_S = "s"
UNIT_PIXEL = "pixel"

# Frame-data units — the @units attr on set-level raw/data and each
# measurement data dataset. "raw" in the tree means "as read, un-postprocessed";
# the *unit* of that read changed over producer history (legacy corpus: ADU
# counts; EoScan since 2026-07: photons), so the producer must declare it.
UNIT_PHOTON = "photon"
UNIT_ADU = "adu"

# instrument (NXinstrument) fields
FIELD_WAVELENGTH = "wavelength"
FIELD_BEAM_ENERGY = "beam_energy"
# sample (NXsample) fields
FIELD_SAMPLE_NAME = "name"
FIELD_PATIENT_NAME = "patient_name"
FIELD_SAMPLE_TYPE = "sample_type"
# optional free-form descriptive sample/patient metadata (JSON), e.g. clinical
# attributes. EoScan's live export leaves it unset; backfill producers populate it.
DS_SAMPLE_METADATA = "metadata"
# set geometry + per-set acquisition (source settings snapshot for this set)
FIELD_DISTANCE = "distance"
FIELD_VOLTAGE = "voltage"
FIELD_CURRENT = "current"
FIELD_EXPOSURE_TIME = "exposure_time"
FIELD_SAMPLE_THICKNESS = "sample_thickness"
FIELD_STAGE_POSITION = "stage_position"
# detector (NXdetector) fields
FIELD_X_PIXEL_SIZE = "x_pixel_size"
FIELD_Y_PIXEL_SIZE = "y_pixel_size"
FIELD_X_PIXEL_COUNT = "x_pixel_count"
FIELD_Y_PIXEL_COUNT = "y_pixel_count"
FIELD_SENSOR_THICKNESS = "sensor_thickness"

# dependencies: single JSON dataset under /session (array of edge dicts)
DS_DEPENDENCIES = "dependencies"


# ================== ID formatters =========================
# Group names carry a zero-padded index plus a human label. The index is
# mandatory where order is semantic (sets = measured order, steps = execution
# order, qc = priority order) because HDF5 iterates children alphabetically;
# the label makes the tree readable without opening attributes. Lookups parse
# the index prefix only — labels are display, never identity. Labels come from
# producer DB fields (measurement type, hardware_id, check name), so renaming
# those changes the group names in a REBUILT container — fine: files are
# write-once and the canonical keys live in attrs, untouched by label drift.
def _label_token(label: str) -> str:
    return sanitize_filename_token(label).lower()


def format_set_id(index: int, label: str) -> str:
    """1-based set group name in measured order: set_001_dark."""
    return f"set_{index:03d}_{_label_token(label)}"


def format_step_id(index: int, name: str) -> str:
    """1-based step group name in execution order: step_01_denoise_normalize."""
    return f"step_{index:02d}_{_label_token(name)}"


def format_qc_id(index: int, check_name: str) -> str:
    """1-based qc group name in priority order: qc_01_beam_intensity."""
    return f"qc_{index:02d}_{_label_token(check_name)}"


def format_dep_id(index: int) -> str:
    return f"dep_{index:03d}"


def format_detector_id(detector_id: int, label: str) -> str:
    """Detector group name: det_<id>_<hardware_id>. Used for both catalog
    entries and measurement groups (one measurement per detector per set)."""
    return f"det_{detector_id}_{_label_token(label)}"


def format_detector_set_id(detector_set_id: int, label: str) -> str:
    """Detector-set catalog group name: ds_<id>_<hardware_id>."""
    return f"ds_{detector_set_id}_{_label_token(label)}"


def detector_set_path(detector_set_id: int, label: str) -> str:
    """Absolute path to a detector set's catalog entry."""
    return f"{GROUP_DETECTOR_SETS}/{format_detector_set_id(detector_set_id, label)}"
