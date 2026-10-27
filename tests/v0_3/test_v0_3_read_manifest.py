"""The reading side of the upload contract: check, upgrade, members."""

import copy
import json
import zipfile

import pytest

from container.v0_3 import ManifestError, manifest

from _factory_v0_3 import make_detector_set_spec, make_detector_spec, make_measurement, make_session, make_set

FRAMES = {"meas-uid-7": b"gfrm bytes 7", "meas-uid-8": b"gfrm bytes 8"}
HEADERS = {"meas-uid-7": b"dsc 7", "meas-uid-8": b"dsc 8"}


def written(tmp_path, payload=None, **kw):
    payload = payload or make_session(session_metadata={"form": {"suspicious": True}})
    out = manifest.write_zip(payload, FRAMES.__getitem__, tmp_path / "s.zip",
                             header_source=HEADERS.__getitem__, **kw)
    zf = zipfile.ZipFile(out)
    return payload, zf, json.loads(zf.read(f"{payload.session_uid}/manifest.json"))


def as_v1(m):
    """The same session as a v1 producer sent it: form values flat."""
    v1 = copy.deepcopy(m)
    v1["manifest_version"] = 1
    v1["session"]["metadata"] = v1["session"]["metadata"]["form"]
    return v1


def test_current_manifest_checks_and_upgrades_unchanged(tmp_path):
    _, _, m = written(tmp_path)
    before = copy.deepcopy(m)
    manifest.check(m)
    assert manifest.upgrade(m) == before


def test_v1_upgrades_to_current_shape_and_keeps_its_version(tmp_path):
    _, _, m = written(tmp_path)
    v1 = as_v1(m)
    manifest.check(v1)
    manifest.upgrade(v1)
    assert v1["session"]["metadata"] == {"form": {"suspicious": True}}
    # the version records what arrived, not the shape the dict now holds
    assert v1["manifest_version"] == 1


def test_v1_without_metadata_upgrades_to_none(tmp_path):
    _, _, m = written(tmp_path, make_session())
    v1 = copy.deepcopy(m)
    v1["manifest_version"] = 1
    assert manifest.upgrade(v1)["session"]["metadata"] is None


def test_unknown_version_refused(tmp_path):
    _, _, m = written(tmp_path)
    m["manifest_version"] = 9
    with pytest.raises(ManifestError, match="unknown manifest_version 9"):
        manifest.check(m)


def test_flat_metadata_under_v2_refused(tmp_path):
    _, _, m = written(tmp_path)
    m["session"]["metadata"] = {"suspicious": True}
    with pytest.raises(ManifestError, match=r"session.metadata\['suspicious'\] is not a namespace"):
        manifest.upgrade(m)


@pytest.mark.parametrize("edit, message", [
    (lambda m: m.pop("producer"), r"manifest lacks \['producer'\]"),
    (lambda m: m["session"].pop("status"), r"manifest session lacks \['status'\]"),
    (lambda m: m["session"]["machine"].pop("wavelength_angstrom"), "lacks wavelength_angstrom"),
    (lambda m: m["session"].update(sample=None), "without sample.clinical_name"),
    (lambda m: m["sets"][0].pop("voltage_kv"), r"lacks \['voltage_kv'\]"),
    (lambda m: m["sets"][0]["measurements"][0].pop("file"), r"lacks \['file'\]"),
])
def test_missing_required_key_refused(tmp_path, edit, message):
    _, _, m = written(tmp_path)
    edit(m)
    with pytest.raises(ManifestError, match=message):
        manifest.check(m)


def test_members_are_exactly_the_files_written(tmp_path):
    dets = [make_detector_spec(1, "Det-A", mask_file_path="/m/1.npy"), make_detector_spec(2, "Det-B")]
    p = make_session(detector_sets=[make_detector_set_spec(detectors=dets)],
                     sets=[make_set(measurements=[make_measurement(7, 1), make_measurement(8, 2)])])
    _, zf, m = written(tmp_path, p, mask_source={1: b"npy 1"}.__getitem__)
    listed = manifest.members(m)
    prefix = f"{p.session_uid}/"
    assert {prefix + rel for rel, _ in listed} == set(zf.namelist()) - {prefix + "manifest.json"}
    # images carry the sha256 the writer recorded; headers and masks carry none
    for rel, sha in listed:
        assert (sha is not None) == rel.split("/")[-1].startswith("image")
