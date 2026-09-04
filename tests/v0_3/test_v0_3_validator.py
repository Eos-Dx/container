"""Validator: valid passes; structural breakages are flagged."""

import h5py
import numpy as np

from container.v0_3 import IntegrationPayload, QCResultPayload, build_session_container
from container.v0_3.validator import SEVERITY_ERROR, validate_session_container

from _factory_v0_3 import make_session, make_set


def _build(tmp_path, **session_kw):
    _, _, path = build_session_container(make_session(**session_kw), tmp_path)
    return path


def _errors(path):
    ok, errs = validate_session_container(path)
    return ok, [e for e in errs if e.severity == SEVERITY_ERROR]


def test_valid_container_passes(tmp_path):
    ok, errs = validate_session_container(_build(tmp_path))
    assert ok
    assert [e for e in errs if e.severity == SEVERITY_ERROR] == []


def test_missing_session_uid_errors(tmp_path):
    path = _build(tmp_path)
    with h5py.File(path, "r+") as f:
        del f.attrs["session_uid"]
    ok, errs = _errors(path)
    assert not ok
    assert any("session_uid" in e.message for e in errs)


def test_wrong_schema_version_errors(tmp_path):
    path = _build(tmp_path)
    with h5py.File(path, "r+") as f:
        f.attrs["schema_version"] = "0.2"
    ok, errs = _errors(path)
    assert not ok
    assert any("schema_version" in e.message for e in errs)


def test_integration_length_mismatch_errors(tmp_path):
    sp = make_set()
    sp.integration = IntegrationPayload(q=np.linspace(0, 30, 2000),
                                        i=np.ones(1999), npt=2000)
    _, _, path = build_session_container(make_session(sets=[sp]), tmp_path)
    ok, errs = _errors(path)
    assert not ok
    assert any("mismatch" in e.message for e in errs)


def test_bad_verdict_errors(tmp_path):
    sp = make_set(qc=False)
    sp.qc_results = [QCResultPayload("symmetry", "BOGUS", "x", {}, {}, "t")]
    _, _, path = build_session_container(make_session(sets=[sp]), tmp_path)
    ok, errs = _errors(path)
    assert not ok
    assert any("verdict" in e.message for e in errs)


def test_missing_frame_units_warns_not_errors(tmp_path):
    """Pre-units archives lack @units on frame data: warning, never error."""
    path = _build(tmp_path)
    with h5py.File(path, "r+") as f:
        del f["/session/sets/set_001_sample_main/measurements/det_1_det-a/data"].attrs["units"]
    ok, errs = validate_session_container(path)
    assert ok
    assert any("lacks @units" in e.message for e in errs)
