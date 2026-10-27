"""legacy_omniscan: the legacy backfill's v1 set metadata through manifest.upgrade.
Delete with the module."""

import json
import zipfile

from container.v0_3 import legacy_omniscan, manifest

from _factory_v0_3 import make_session

FRAMES = {"meas-uid-7": b"gfrm bytes 7", "meas-uid-8": b"gfrm bytes 8"}
HEADERS = {"meas-uid-7": b"dsc 7", "meas-uid-8": b"dsc 8"}


def written(tmp_path, payload):
    out = manifest.write_zip(payload, FRAMES.__getitem__, tmp_path / "s.zip",
                             header_source=HEADERS.__getitem__)
    zf = zipfile.ZipFile(out)
    return payload, zf, json.loads(zf.read(f"{payload.session_uid}/manifest.json"))


PROVENANCE = {"software": "OmniScan", "data_batch": "1"}


def legacy_v1(tmp_path, metadata, type_name="Sample Main", thickness=2.5):
    """A v1 manifest from the legacy backfill with one set carrying ``metadata``."""
    _, _, m = written(tmp_path, make_session())
    m["manifest_version"] = 1
    m["producer"]["software"] = legacy_omniscan.PRODUCER
    sp = m["sets"][0]
    sp.update(metadata=metadata, measurement_type_name=type_name, sample_thickness_mm=thickness)
    return m


def test_legacy_sample_main_moves_under_omniscan(tmp_path):
    m = legacy_v1(tmp_path, {"position": "P1", "is_rescan": False, "thickness_raw_mm": 2.5,
                             "backfill_provenance": PROVENANCE})
    manifest.upgrade(m)
    # thickness_raw_mm restated the thickness column, so it is gone
    assert m["sets"][0]["metadata"] == {"omniscan": {
        "position": "P1", "is_rescan": False, "provenance": PROVENANCE}}


def test_legacy_transmission_keeps_its_values_and_drops_phase(tmp_path):
    m = legacy_v1(tmp_path, {"position": "P2", "phase": "Phase1", "is_rescan": True,
                             "transmission_pct": 10.0, "correction_factor": 1.6,
                             "backfill_provenance": PROVENANCE}, type_name="Sample Transmission")
    manifest.upgrade(m)
    assert m["sets"][0]["metadata"] == {"omniscan": {
        "position": "P2", "is_rescan": True, "transmission_pct": 10.0,
        "correction_factor": 1.6, "provenance": PROVENANCE}}


def test_legacy_calibration_frame_kind_is_the_type_name(tmp_path):
    m = legacy_v1(tmp_path, {"frame_kind": "AgBh", "backfill_provenance": PROVENANCE},
                  type_name="Silver Behenate", thickness=None)
    manifest.upgrade(m)
    assert m["sets"][0]["metadata"] == {"omniscan": {"provenance": PROVENANCE}}


def test_legacy_value_that_disagrees_is_kept(tmp_path):
    m = legacy_v1(tmp_path, {"thickness_raw_mm": 3.0, "frame_kind": "Dark", "phase": "Phase2"})
    manifest.upgrade(m)
    assert m["sets"][0]["metadata"] == {"omniscan": {
        "thickness_raw_mm": 3.0, "frame_kind": "Dark", "phase": "Phase2"}}
