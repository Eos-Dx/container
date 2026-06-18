"""Payload builders for v0.3 container tests (imported by test modules)."""

import numpy as np

from container.v0_3 import (
    DependencyRef,
    DetectorSetSpec,
    DetectorSpec,
    IntegrationPayload,
    MeasurementPayload,
    QCResultPayload,
    SessionPayload,
    SetPayload,
)


# Producer-supplied uids — in real use EoScan passes its persisted row uids;
# here we derive readable, deterministic ones from the pk so dependency edges
# (which reference a target session's uid) are easy to assert.
def sess_uid(pk):
    return f"sess-uid-{pk}"


def make_detector_spec(detector_id=1, hardware_id="Det-A", **over):
    kw = dict(
        detector_id=detector_id, hardware_id=hardware_id, manufacturer="Advacam",
        model="WidePIX", pixel_size_um=55.0, width_px=256, height_px=256,
        sensor_thickness_um=500.0, material="Si", mask_file_path=None,
    )
    kw.update(over)
    return DetectorSpec(**kw)


def make_detector_set_spec(detector_set_id=1, hardware_id="DS-1", detectors=None, **over):
    if detectors is None:
        detectors = [make_detector_spec(1, "Det-A"), make_detector_spec(2, "Det-B")]
    kw = dict(
        detector_set_id=detector_set_id, hardware_id=hardware_id,
        layout={"detectors": [{"detector_id": 1, "x_mm": 0, "y_mm": 0}],
                "primary_detector_id": 1},
        detectors=detectors, primary_detector_id=1,
    )
    kw.update(over)
    return DetectorSetSpec(**kw)


def make_measurement(pk=7, detector_id=1, **over):
    kw = dict(
        measurement_pk=pk, measurement_uid=f"meas-uid-{pk}", detector_id=detector_id,
        file_path=f"/data/det_{detector_id}.txt",
        metadata_file_path=f"/data/det_{detector_id}.dsc",
        mask_file_path=None,
        data=np.full((4, 4), detector_id, dtype=np.float32),
        mask=np.ones((4, 4), dtype=np.uint8),
        detector_meta=b"<.dsc header>",
        raw_file=f"<raw vendor bytes det {detector_id}>".encode(),
    )
    kw.update(over)
    return MeasurementPayload(**kw)


def make_set(pk=99, measurements=None, qc=True, integration=True,
             processing_steps=None, raw=None, processed=None, **over):
    if measurements is None:
        measurements = [make_measurement()]
    qc_results = []
    if qc:
        qc_results = [QCResultPayload(
            check_name="symmetry", verdict="PASS", message="ok",
            metrics={"symmetry_pct": 98.5, "reference_set_id": 5},
            parameters_snapshot={"x1": 1, "x2": 2}, created_at="2026-06-08 10:01:00",
            priority=10)]
    integ = None
    if integration:
        integ = IntegrationPayload(q=np.linspace(0, 30, 2000),
                                   i=np.ones(2000), npt=2000)
    kw = dict(
        set_pk=pk, set_uid=f"set-uid-{pk}", workflow_id="wf-1", batch_id="b-1",
        detector_set_id=1, status="COMPLETED",
        is_approved=True, measurement_type_name="sample_main",
        measurement_type_category="SAMPLE", workflow_key="bruker_xrd_fixed",
        distance_mm=170.0, voltage_kv=40.0, current_ua=30.0, exposure_time_s=60.0,
        sample_thickness_mm=1.2, stage_position=0.0,
        sample_name=f"sample_sample_main_{pk}", created_at="2026-06-08 10:00:00",
        metadata={"lrf": {"average_mm": 1.2}, "advacam": {"frame_count": 3}},
        processing_config={"postprocessing_key": "sample",
                           "postprocessing_params": {"integrate_npt": 2000}},
        measurements=measurements, raw=raw, processed=processed,
        qc_results=qc_results, integration=integ,
        processing_steps=processing_steps or [], poni_text="Distance: 0.17\n",
        preview=b"\x89PNG-preview",
    )
    kw.update(over)
    return SetPayload(**kw)


def make_session(category="SAMPLE", sets=None, dependencies=None, detector_sets=None,
                 instance_id="inst-1", session_pk=42, **over):
    if sets is None:
        sets = [make_set()]
    if detector_sets is None:
        detector_sets = [make_detector_set_spec()]
    if dependencies is None:
        if category == "SAMPLE":
            dependencies = [DependencyRef("calibration", sess_uid(7), session_pk=7),
                            DependencyRef("system", sess_uid(3), session_pk=3)]
        elif category == "CALIBRATION":
            dependencies = [DependencyRef("system", sess_uid(3), session_pk=3)]
        else:
            dependencies = []
    sample_name = "PAT001-S01" if category == "SAMPLE" else None
    kw = dict(
        session_uid=sess_uid(session_pk), instance_id=instance_id,
        session_pk=session_pk, session_category=category,
        status="COMPLETED", operator_username="alice", machine_serial="SN-001",
        machine_type="EosDx", machine_location="LabA", wavelength_angstrom=1.5406,
        beam_energy_keV=8.047, source_type="Cu", started_at="2026-06-08 09:00:00",
        completed_at="2026-06-08 10:30:00",
        detector_sets=detector_sets,
        sample_clinical_name=sample_name,
        patient_clinical_name="PAT001" if category == "SAMPLE" else None,
        sample_type_name="tissue" if category == "SAMPLE" else None,
        protocol_snapshot={"name": "p1"} if category == "SAMPLE" else None,
        dependencies=dependencies, producer_software="eoscan",
        producer_version="1.0", sets=sets,
    )
    kw.update(over)
    return SessionPayload(**kw)
