"""write_zip emits the EosCloud upload contract (manifest v1) from a SessionPayload."""

import hashlib
import json
import zipfile

import numpy as np
import pytest

from container.v0_3 import MANIFEST_VERSION, build_manifest, write_zip

from _factory_v0_3 import make_detector_set_spec, make_detector_spec, make_measurement, make_session, make_set

FRAMES = {"meas-uid-7": b"gfrm bytes 7", "meas-uid-8": b"gfrm bytes 8"}
HEADERS = {"meas-uid-7": b"dsc 7", "meas-uid-8": b"dsc 8"}

# what the cloud's ingest requires (apps/jobs/handlers/ingest.py) — kept in step by hand
REQUIRED_SESSION = ("uid", "category", "status", "started_at", "machine")
REQUIRED_SET = (
    "set_uid", "detector_set_id", "status", "measurement_type_name", "measurement_type_category",
    "distance_mm", "voltage_kv", "current_ua", "exposure_time_s", "created_at", "measurements",
)
REQUIRED_MEASUREMENT = ("measurement_uid", "detector_id", "file")
NOT_SHIPPED = ("raw", "processed", "integration", "processing_config", "processing_steps", "preview")


def build(tmp_path, payload=None, **kw):
    payload = payload or make_session()
    out = write_zip(payload, FRAMES.__getitem__, tmp_path / "s.zip",
                    header_source=HEADERS.__getitem__, **kw)
    zf = zipfile.ZipFile(out)
    return payload, zf, json.loads(zf.read(f"{payload.session_uid}/manifest.json"))


def test_layout_and_bytes(tmp_path):
    p, zf, m = build(tmp_path)
    uid, s, mu = p.session_uid, p.sets[0], p.sets[0].measurements[0]
    base = f"{uid}/sets/{s.set_uid}/measurements/{mu.measurement_uid}/"
    assert set(zf.namelist()) == {f"{uid}/manifest.json", base + "image.txt", base + "header.dsc"}
    assert zf.read(base + "image.txt") == FRAMES[mu.measurement_uid]
    assert zf.read(base + "header.dsc") == HEADERS[mu.measurement_uid]
    assert zf.testzip() is None
    mm = m["sets"][0]["measurements"][0]
    assert mm["file"] == "image.txt" and mm["header"] == "header.dsc"
    assert mm["sha256"] == hashlib.sha256(FRAMES[mu.measurement_uid]).hexdigest()


def test_manifest_has_what_ingest_requires_and_nothing_derived(tmp_path):
    p, _, m = build(tmp_path)
    assert set(m) == {"manifest_version", "producer", "session", "detector_sets", "sets"}
    assert m["manifest_version"] == MANIFEST_VERSION == 1
    assert m["producer"] == {"software": "eoscan", "version": "1.0", "instance_id": "inst-1"}
    assert all(k in m["session"] for k in REQUIRED_SESSION)
    assert m["session"]["machine"]["wavelength_angstrom"] == 1.5406
    assert m["session"]["sample"] == {"clinical_name": "PAT001-S01", "patient_clinical_name": "PAT001",
                                      "sample_type_name": "tissue", "metadata": None}
    assert m["session"]["dependencies"][0] == {"kind": "calibration", "uid": "sess-uid-7", "session_pk": 7}
    det = m["detector_sets"][0]["detectors"][0]
    assert det["hardware_id"] == "Det-A" and det["mask_ref"] is None and "mask_file_path" not in det
    s = m["sets"][0]
    assert all(k in s for k in REQUIRED_SET)
    assert not any(k in s for k in NOT_SHIPPED)
    assert s["poni_text"] == "Distance: 0.17\n" and s["raw_sha256"] is None
    assert s["metadata"] == {"lrf": {"average_mm": 1.2}, "advacam": {"frame_count": 3}}
    assert s["qc_results"][0]["check_name"] == "symmetry" and s["qc_results"][0]["priority"] == 10
    assert all(k in s["measurements"][0] for k in REQUIRED_MEASUREMENT)


def test_calibration_has_no_sample_block(tmp_path):
    _, _, m = build(tmp_path, make_session(category="CALIBRATION"))
    assert m["session"]["sample"] is None
    assert m["session"]["dependencies"] == [{"kind": "system", "uid": "sess-uid-3", "session_pk": 3}]


def test_embedded_header_means_no_header_member(tmp_path):
    meas = make_measurement(file_path="/d/f.gfrm", metadata_file_path="/d/f.gfrm")
    p = make_session(sets=[make_set(measurements=[meas])])
    out = write_zip(p, FRAMES.__getitem__, tmp_path / "s.zip")  # no header_source needed
    zf = zipfile.ZipFile(out)
    names = zf.namelist()
    assert len(names) == 2 and names[0].endswith("image.gfrm")
    m = json.loads(zf.read(f"{p.session_uid}/manifest.json"))
    assert m["sets"][0]["measurements"][0]["header"] is None


def test_separate_header_without_source_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="header_source"):
        write_zip(make_session(), FRAMES.__getitem__, tmp_path / "s.zip")


def test_masks_hoisted_per_detector(tmp_path):
    dets = [make_detector_spec(1, "Det-A", mask_file_path="/m/1.npy"), make_detector_spec(2, "Det-B")]
    p = make_session(detector_sets=[make_detector_set_spec(detectors=dets)],
                     sets=[make_set(measurements=[make_measurement(7, 1), make_measurement(8, 2)])])
    _, zf, m = build(tmp_path, p, mask_source={1: b"npy 1"}.__getitem__)
    assert zf.read(f"{p.session_uid}/masks/1.npy") == b"npy 1"
    assert [d["mask_ref"] for d in m["detector_sets"][0]["detectors"]] == ["masks/1.npy", None]
    assert [x["mask_ref"] for x in m["sets"][0]["measurements"]] == ["masks/1.npy", None]


def test_masks_not_shipped_without_source(tmp_path):
    dets = [make_detector_spec(1, "Det-A", mask_file_path="/m/1.npy")]
    p = make_session(detector_sets=[make_detector_set_spec(detectors=dets)])
    _, zf, m = build(tmp_path, p)
    assert not any("masks/" in n for n in zf.namelist())
    assert m["detector_sets"][0]["detectors"][0]["mask_ref"] is None


def test_metadata_attrs_merge_and_numpy_scalars_serialize(tmp_path):
    raw = np.arange(6, dtype=np.uint16).reshape(2, 3)
    s = make_set(raw=raw, metadata_attrs={"position": "P1"}, distance_mm=np.float64(170.5))
    p = make_session(sets=[s], sample_metadata={"age": np.int64(40)}, sample_metadata_attrs={"side": "Left"})
    _, _, m = build(tmp_path, p)
    assert m["sets"][0]["metadata"]["position"] == "P1"
    assert m["sets"][0]["distance_mm"] == 170.5
    assert m["sets"][0]["raw_sha256"] == hashlib.sha256(raw.tobytes()).hexdigest()
    assert m["session"]["sample"]["metadata"] == {"age": 40, "side": "Left"}


def test_frame_source_may_return_a_path(tmp_path):
    f = tmp_path / "frame.gfrm"
    f.write_bytes(b"from disk")
    meas = make_measurement(file_path="x.gfrm", metadata_file_path="x.gfrm")
    p = make_session(sets=[make_set(measurements=[meas])])
    out = write_zip(p, lambda uid: f, tmp_path / "s.zip")
    m = json.loads(zipfile.ZipFile(out).read(f"{p.session_uid}/manifest.json"))
    assert m["sets"][0]["measurements"][0]["sha256"] == hashlib.sha256(b"from disk").hexdigest()


def test_build_manifest_is_pure_and_leaves_sha_unfilled():
    m = build_manifest(make_session())
    assert m["sets"][0]["measurements"][0]["sha256"] is None
    json.dumps(m)
