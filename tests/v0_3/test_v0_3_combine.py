"""Combined container: embedding, self-description, params, link expansion, guards."""

import h5py
import numpy as np
import pytest

from container.common.hdf5 import _decode
from container.v0_3 import build_combined_container, build_session_container
from container.v0_3.combine import NAME_SESSIONS

from _factory_v0_3 import make_session, sess_uid


PARAMS = {"target": sess_uid(2), "contralateral": sess_uid(1), "threshold": 0.3, "paired": True}


def build_sources(tmp_path):
    _, _, sample = build_session_container(make_session(session_pk=2), tmp_path / "src")
    _, _, calib = build_session_container(
        make_session(category="CALIBRATION", session_pk=1), tmp_path / "src")
    return sample, calib


def test_roundtrip_root_and_params(tmp_path):
    sample, calib = build_sources(tmp_path)
    cid, path = build_combined_container([sample, calib], PARAMS, tmp_path)
    with h5py.File(path, "r") as f:
        assert _decode(f.attrs["container_type"]) == "combined"
        assert _decode(f.attrs["format"]) == "xrd-session"
        assert _decode(f.attrs["schema_version"]) == "0.3"
        assert _decode(f.attrs["container_id"]) == cid
        for k, v in PARAMS.items():
            assert _decode(f.attrs[k]) == v
        assert set(f[NAME_SESSIONS]) == {sess_uid(1), sess_uid(2)}


def test_embedded_sessions_stay_self_describing(tmp_path):
    sample, calib = build_sources(tmp_path)
    _, path = build_combined_container([sample, calib], {}, tmp_path)
    with h5py.File(path, "r") as f:
        for pk, category in ((2, "SAMPLE"), (1, "CALIBRATION")):
            emb = f[f"{NAME_SESSIONS}/{sess_uid(pk)}"]
            # source root attrs preserved on the group
            assert _decode(emb.attrs["container_type"]) == "session"
            assert _decode(emb.attrs["session_uid"]) == sess_uid(pk)
            assert _decode(emb.attrs["producer_software"]) == "eoscan"
            assert _decode(emb["session"].attrs["category"]) == category
        # tree copied intact — one set with its measurement frame
        st = f[f"{NAME_SESSIONS}/{sess_uid(2)}/session/sets/set_001_sample_main"]
        det = next(iter(st["measurements"]))
        np.testing.assert_array_equal(
            st[f"measurements/{det}/data"][()], np.full((4, 4), 1, dtype=np.float32))


def test_soft_links_expanded(tmp_path):
    sample, _ = build_sources(tmp_path)
    _, path = build_combined_container([sample], {}, tmp_path)
    with h5py.File(path, "r") as f:
        st = f[f"{NAME_SESSIONS}/{sess_uid(2)}/session/sets/set_001_sample_main"]
        # absolute /session/... soft link would dangle re-rooted; must be a real copy
        assert isinstance(st.get("detector_set", getlink=True), h5py.HardLink)
        assert _decode(st["detector_set"].attrs["detector_set_hardware_id"]) == "DS-1"


def test_guards(tmp_path):
    sample, calib = build_sources(tmp_path)
    with pytest.raises(ValueError, match="no session containers"):
        build_combined_container([], {}, tmp_path)
    with pytest.raises(ValueError, match="duplicate session_uid"):
        build_combined_container([sample, sample], {}, tmp_path)
    with pytest.raises(ValueError, match="scalars only"):
        build_combined_container([sample], {"roles": {"target": 1}}, tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        build_combined_container([sample], {"container_id": "x"}, tmp_path)
    _, combined = build_combined_container([sample, calib], {}, tmp_path)
    with pytest.raises(ValueError, match="expected 'session'"):
        build_combined_container([combined], {}, tmp_path)
