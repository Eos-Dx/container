"""Cross-container dependency edges."""

from container import open_container
from container.v0_3 import build_session_container

from _factory_v0_3 import make_session, sess_uid


def _deps(tmp_path, category):
    _, _, path = build_session_container(make_session(category=category), tmp_path)
    return open_container(path).dependencies()


def test_sample_depends_on_calibration_and_system(tmp_path):
    deps = _deps(tmp_path, "SAMPLE")
    by_role = {d["role"]: d for d in deps}
    assert set(by_role) == {"calibration", "system"}
    # edge references the target session by its producer-supplied uid
    assert by_role["calibration"]["session_uid"] == sess_uid(7)
    assert by_role["calibration"]["session_pk"] == 7


def test_calibration_depends_on_system_only(tmp_path):
    deps = _deps(tmp_path, "CALIBRATION")
    assert {d["role"] for d in deps} == {"system"}


def test_system_has_no_dependencies(tmp_path):
    assert _deps(tmp_path, "SYSTEM") == []
