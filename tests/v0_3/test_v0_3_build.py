"""v0.3 build round-trip, identity, NeXus classes, set matrices, metadata, write-once."""

import h5py
import numpy as np

from container import open_container
from container.v0_3 import build_session_container
from container.v0_3.utils import read_json_dataset

from _factory_v0_3 import make_measurement, make_session, make_set, sess_uid


def build(tmp_path, **session_kw):
    return build_session_container(make_session(**session_kw), tmp_path)


def test_build_roundtrip_and_autodetect(tmp_path):
    suid, cid, path = build(tmp_path)
    c = open_container(path)
    assert type(c).__name__ == "SessionContainer"
    meta = c.session_meta()
    assert meta["schema_version"] == "0.3"
    assert meta["format"] == "xrd-session"
    assert meta["container_type"] == "session"
    assert [s["name"] for s in c.sets()] == ["set_001_sample_main"]
    # per-detector decoded frame round-trips (det_1 frame filled with 1.0)
    frame = c.frame(1, 1)
    assert frame.shape == (4, 4) and frame[0, 0] == 1.0


def test_set_and_detector_counts(tmp_path):
    sets = [make_set(pk=1, measurements=[make_measurement(pk=10, detector_id=1),
                                         make_measurement(pk=11, detector_id=2)]),
            make_set(pk=2)]
    _, _, path = build_session_container(make_session(sets=sets), tmp_path)
    c = open_container(path)
    assert len(c.sets()) == 2
    assert len(c.measurements(1)) == 2


def test_measurement_references_detector_by_id_only(tmp_path):
    """@detector_id is the canonical reference; no per-measurement soft link —
    the full spec is reachable via the set's `detector_set` link."""
    _, _, path = build(tmp_path)
    with h5py.File(path, "r") as f:
        meas = f["/session/sets/set_001_sample_main/measurements/det_1_det-a"]
        assert meas.attrs["detector_id"] == 1
        assert "detector" not in meas


def test_root_identity_attrs(tmp_path):
    suid, cid, path = build(tmp_path, instance_id="inst-X", session_pk=42)
    assert suid == sess_uid(42)        # producer-supplied uid, written verbatim
    assert cid != suid                 # distinct ids (physical vs logical)
    with h5py.File(path, "r") as f:
        assert f.attrs["session_uid"] == suid
        assert f.attrs["container_id"] == cid
        assert f.attrs["instance_id"] == "inst-X"   # provenance, optional


def test_instance_id_optional(tmp_path):
    """instance_id is provenance-only — omitted entirely when not supplied."""
    _, _, path = build(tmp_path, instance_id=None)
    with h5py.File(path, "r") as f:
        assert "instance_id" not in f.attrs
        assert f.attrs["session_uid"] == sess_uid(42)


def test_container_id_random_session_uid_stable(tmp_path):
    """Rebuilding the same session keeps session_uid, mints a new container_id."""
    s1, c1, _ = build(tmp_path / "a", session_pk=42)
    s2, c2, _ = build(tmp_path / "b", session_pk=42)
    assert s1 == s2          # stable logical id (same producer uid)
    assert c1 != c2          # random physical id


def test_nexus_base_classes(tmp_path):
    _, _, path = build(tmp_path, sets=[make_set(processed=np.zeros((4, 4)))])
    with h5py.File(path, "r") as f:
        assert f.attrs["NX_class"] == "NXroot"
        assert f["/session"].attrs["NX_class"] == "NXentry"
        assert f["/session/sample"].attrs["NX_class"] == "NXsample"
        assert f["/session/instrument"].attrs["NX_class"] == "NXinstrument"
        det = list(f["/session/sets/set_001_sample_main/measurements"].values())[0]
        assert det.attrs["NX_class"] == "NXdetector"
        assert f["/session/sets/set_001_sample_main/integration"].attrs["NX_class"] == "NXdata"
        # custom groups carry no misleading base class
        assert "NX_class" not in f["/session/sets/set_001_sample_main/qc"].attrs


def test_set_matrices_optional(tmp_path):
    # present
    _, _, path = build(tmp_path / "p",
                       sets=[make_set(processed=np.ones((4, 5)), raw=np.ones((4, 5)))])
    with h5py.File(path, "r") as f:
        proc = f["/session/sets/set_001_sample_main/processed"]
        assert proc.attrs["NX_class"] == "NXdata"
        assert proc.attrs["signal"] == "data"
        assert proc["data"].shape == (4, 5)
        assert f["/session/sets/set_001_sample_main/raw/data"].shape == (4, 5)
    # absent
    _, _, path2 = build(tmp_path / "a", sets=[make_set()])
    with h5py.File(path2, "r") as f:
        assert "processed" not in f["/session/sets/set_001_sample_main"]
        assert "raw" not in f["/session/sets/set_001_sample_main"]


def test_metadata_payload(tmp_path):
    _, _, path = build(tmp_path)
    with h5py.File(path, "r") as f:
        set_grp = f["/session/sets/set_001_sample_main"]
        assert read_json_dataset(set_grp, "metadata")["lrf"]["average_mm"] == 1.2
        # geometry/layout lives once in the detector-set catalog, not per set
        assert "geometry" not in set_grp
        ds = f["/session/instrument/detector_sets/ds_1_ds-1"]
        assert read_json_dataset(ds, "layout")["primary_detector_id"] == 1
        assert ds.attrs["primary_detector_id"] == 1
        # the capture references its detector set by id + a resolving soft link
        assert set_grp.attrs["detector_set_id"] == 1
        assert set_grp.get("detector_set", getlink=True).path == \
            "/session/instrument/detector_sets/ds_1_ds-1"
        pc = read_json_dataset(set_grp["processing"], "config")
        assert pc["postprocessing_key"] == "sample"
        assert pc["postprocessing_params"]["integrate_npt"] == 2000
        assert read_json_dataset(f["/session"], "protocol_snapshot")["name"] == "p1"
        # machine identity stays an attr; sample names are string-field datasets
        assert f["/session/instrument"].attrs["source_type"] == "Cu"
        assert f["/session/sample/name"][()].decode() == "PAT001-S01"
        assert f["/session/sample/sample_type"][()].decode() == "tissue"


def test_physics_scalars_are_fields_with_units(tmp_path):
    """Physics quantities are datasets carrying @units (NeXus tool hygiene)."""
    _, _, path = build(tmp_path)
    with h5py.File(path, "r") as f:
        wl = f["/session/instrument/wavelength"]
        assert wl[()] == 1.5406 and wl.attrs["units"] == "angstrom"
        assert f["/session/instrument/beam_energy"].attrs["units"] == "keV"
        # per-set acquisition conditions grouped, as fields-with-units
        acq = f["/session/sets/set_001_sample_main/acquisition"]
        assert acq["distance"][()] == 170.0 and acq["distance"].attrs["units"] == "mm"
        assert acq["voltage"][()] == 40.0 and acq["voltage"].attrs["units"] == "kV"
        assert acq["current"].attrs["units"] == "uA"
        assert acq["exposure_time"].attrs["units"] == "s"
        # set attrs are pure identity now — no physical quantities mixed in
        assert "voltage_kv" not in f["/session/sets/set_001_sample_main"].attrs
        # measurements are slim references — detector specs live in the catalog
        meas = f["/session/sets/set_001_sample_main/measurements/det_1_det-a"]
        assert meas.attrs["detector_id"] == 1
        assert "x_pixel_size" not in meas               # spec lives in catalog, not here
        det = f["/session/instrument/detector_sets/ds_1_ds-1/detectors/det_1_det-a"]
        assert det["x_pixel_size"].attrs["units"] == "um"
        assert det["x_pixel_count"][()] == 256 and det["x_pixel_count"].attrs["units"] == "pixel"
        assert det["sensor_thickness"][()] == 500.0
        assert "width_px" not in det.attrs           # promoted to a field
        assert det.attrs["detector_hardware_id"] == "Det-A"   # external id retained
        # duplicated session attrs are gone (single home in subgroups)
        assert "wavelength_angstrom" not in f["/session"].attrs
        assert "machine_serial" not in f["/session"].attrs


def test_write_once_no_lifecycle_attrs(tmp_path):
    _, _, path = build(tmp_path)
    with h5py.File(path, "r") as f:
        for forbidden in ("locked", "lock_status", "transfer_status"):
            assert forbidden not in f.attrs
    # opening again read-only works
    assert open_container(path) is not None
