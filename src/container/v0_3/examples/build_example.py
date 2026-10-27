"""Build one representative v0.3 session container and print its structure.

Run:  python src/container/v0_3/examples/build_example.py [sample|calibration]
Produces an .nxs.h5 under this folder's out/ and dumps the full tree.
"""

import uuid
from pathlib import Path

import h5py
import numpy as np

from container.v0_3 import (
    DependencyRef,
    DetectorSetSpec,
    DetectorSpec,
    IntegrationPayload,
    MeasurementPayload,
    ProcessingStepPayload,
    QCResultPayload,
    SessionPayload,
    SetPayload,
    build_session_container,
)

OUT = Path(__file__).resolve().parent / "out"

# Detector-set catalog — each set stored ONCE per session (geometry + its
# detectors), referenced by a capture via numeric detector_set_id and by
# measurements via numeric detector_id.
DETECTOR_SETS = [
    DetectorSetSpec(
        detector_set_id=1, hardware_id="DS-WIDEPIX-01", primary_detector_id=1,
        layout={"detectors": [{"detector_id": 1, "x_mm": 0.0, "y_mm": 0.0},
                              {"detector_id": 2, "x_mm": 14.1, "y_mm": 0.0}],
                "primary_detector_id": 1},
        detectors=[
            DetectorSpec(detector_id=1, hardware_id="Advacam-WidePIX-A", manufacturer="Advacam",
                         model="WidePIX", pixel_size_um=55.0, width_px=256, height_px=256,
                         sensor_thickness_um=500.0, material="Si", mask_file_path="/masks/A.npy"),
            DetectorSpec(detector_id=2, hardware_id="Advacam-WidePIX-B", manufacturer="Advacam",
                         model="WidePIX", pixel_size_um=55.0, width_px=256, height_px=256,
                         sensor_thickness_um=500.0, material="Si", mask_file_path="/masks/B.npy"),
        ],
    ),
]


def _measurement(pk, detector_id):
    # decoded products + the embedded original vendor source bytes (raw_file)
    return MeasurementPayload(
        measurement_pk=pk, measurement_uid=uuid.uuid4().hex, detector_id=detector_id,
        file_path=f"/data/sample_42/det_{detector_id}.txt",
        metadata_file_path=f"/data/sample_42/det_{detector_id}.dsc",
        mask_file_path=f"/masks/det_{detector_id}.npy",
        data=np.random.randint(0, 500, size=(16, 16)).astype(np.float32),
        mask=np.ones((16, 16), dtype=np.uint8),
        detector_meta=b"<.dsc PIXet detector header>",
        raw_file=f"<.gfrm vendor frame bytes det {detector_id}>".encode(),
    )


def _set(pk, mt_name, mt_cat, measurements, with_integration=True):
    return SetPayload(
        set_pk=pk, set_uid=uuid.uuid4().hex,
        workflow_id=f"wf-{pk}", batch_id=f"batch-{pk}", detector_set_id=1, status="COMPLETED",
        is_approved=True, measurement_type_name=mt_name, measurement_type_category=mt_cat,
        workflow_key="advacam_xrd_fixed",
        distance_mm=170.0, voltage_kv=40.0,
        current_ua=30.0, exposure_time_s=60.0, sample_thickness_mm=1.2, stage_position=0.0,
        sample_name=f"{mt_cat.lower()}_{mt_name}_{pk}", created_at="2026-06-08 10:00:00",
        metadata={"lrf": {"average_mm": 1.2, "n_valid": 5},
                  "advacam": {"frame_count": 2, "raw_frames": ["frames/f000.txt", "frames/f001.txt"]}},
        processing_config={"postprocessing_key": "sample",
                           "postprocessing_params": {"integrate_npt": 2000, "apply_hotpix": False},
                           "background_set_uid": None},
        measurements=measurements,
        raw=np.random.randint(0, 500, size=(16, 16)).astype(np.float32),
        processed=np.random.rand(16, 16).astype(np.float32),
        qc_results=[
            QCResultPayload("symmetry", "PASS", "symmetry 98.5%",
                            {"symmetry_pct": 98.5, "sum_a": 1000.0, "sum_b": 1015.0},
                            {"x1": 100, "x2": 150, "y1": 100, "y2": 150}, "2026-06-08 10:01:00"),
            QCResultPayload("transmission", "PASS", "transmission 42%",
                            {"transmission_pct": 42.0, "reference_set_id": 3},
                            {"ratio_pct": 42.0}, "2026-06-08 10:01:05"),
        ],
        integration=(IntegrationPayload(q=np.linspace(0.1, 30.0, 2000),
                                        i=np.abs(np.random.rand(2000)) * 1000,
                                        npt=2000)
                     if with_integration else None),
        processing_steps=[
            ProcessingStepPayload("denoise_normalize", "10:00:01", "10:00:02",
                                  {"snap_values": [64, 128]}, "raw", "processed"),
            ProcessingStepPayload("integrate", "10:00:02", "10:00:03",
                                  {"npt": 2000}, "processed", "integration"),
        ],
        poni_text="Distance: 0.1706\nPoni1: 0.0123\nPoni2: 0.0145\nWavelength: 1.5406e-10\n",
        preview=b"\x89PNG\r\n<preview png bytes>",
    )


def build() -> str:
    payload = SessionPayload(
        session_uid=uuid.uuid4().hex,
        instance_id="7f3c2a10-eos-labA", session_pk=42, session_category="SAMPLE",
        status="COMPLETED", operator_username="alice", machine_serial="SN-001",
        machine_type="EosDx-Bench", machine_location="LabA", wavelength_angstrom=1.5406,
        beam_energy_keV=8.0478, source_type="Cu", started_at="2026-06-08 09:00:00",
        completed_at="2026-06-08 10:30:00", sample_clinical_name="PAT001-S01",
        patient_clinical_name="PAT001", sample_type_name="tissue",
        protocol_snapshot={"protocol_id": 1, "name": "Standard tissue scan",
                           "category": "SAMPLE", "blocks": [{"sequence_order": 0,
                           "repeat_count": 1, "steps": [{"measurement_type_name": "sample_main"}]}]},
        detector_sets=DETECTOR_SETS,
        # in real use the target uid is the dep session's persisted uid (looked
        # up in the DB); here it's illustrative.
        dependencies=[DependencyRef("calibration", uuid.uuid4().hex, session_pk=7),
                      DependencyRef("system", uuid.uuid4().hex, session_pk=3),
                      DependencyRef("system", uuid.uuid4().hex, session_pk=4)],
        producer_software="eoscan", producer_version="1.0.0",
        sets=[
            _set(99, "sample_main", "SAMPLE",
                 [_measurement(701, 1), _measurement(702, 2)]),
            _set(100, "sample_far", "SAMPLE",
                 [_measurement(703, 1)]),
        ],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    session_uid, container_id, path = build_session_container(payload, OUT)
    print(f"session_uid = {session_uid}")
    print(f"container_id = {container_id}")
    print(f"file = {path}\n")
    return path


def dump(path: str) -> None:
    def fmt_attrs(obj):
        if not obj.attrs:
            return ""
        parts = []
        for k, v in obj.attrs.items():
            v = v.decode() if isinstance(v, bytes) else v
            parts.append(f"{k}={v}")
        return "  {" + ", ".join(parts) + "}"

    with h5py.File(path, "r") as f:
        print("/" + fmt_attrs(f))

        def visit(name, obj):
            depth = name.count("/") + 1
            indent = "  " * depth
            short = name.split("/")[-1]
            if isinstance(obj, h5py.Group):
                print(f"{indent}{short}/{fmt_attrs(obj)}")
            else:
                print(f"{indent}{short}  <dataset {obj.shape} {obj.dtype}>{fmt_attrs(obj)}")

        f.visititems(visit)


def build_calibration() -> str:
    """A CALIBRATION session: no sample, system-only deps, AGBH + DARK sets."""
    agbh = _set(201, "agbh", "CALIBRATION", [_measurement(801, 1)])
    dark = _set(202, "dark", "CALIBRATION", [_measurement(802, 1)],
                with_integration=False)
    dark.poni_text = None  # dark frame yields no calibration geometry

    payload = SessionPayload(
        session_uid=uuid.uuid4().hex,
        instance_id="7f3c2a10-eos-labA", session_pk=7, session_category="CALIBRATION",
        status="COMPLETED", operator_username="alice", machine_serial="SN-001",
        machine_type="EosDx-Bench", machine_location="LabA", wavelength_angstrom=1.5406,
        beam_energy_keV=8.0478, source_type="Cu", started_at="2026-06-08 08:00:00",
        completed_at="2026-06-08 08:30:00",
        # no sample/patient/sample_type on a calibration session
        detector_sets=DETECTOR_SETS,
        dependencies=[DependencyRef("system", uuid.uuid4().hex, session_pk=3),
                      DependencyRef("system", uuid.uuid4().hex, session_pk=4)],
        producer_software="eoscan", producer_version="1.0.0", sets=[agbh, dark],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    session_uid, container_id, path = build_session_container(payload, OUT)
    print(f"session_uid = {session_uid}")
    print(f"container_id = {container_id}")
    print(f"file = {path}\n")
    return path


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "sample"
    dump(build_calibration() if which == "calibration" else build())
