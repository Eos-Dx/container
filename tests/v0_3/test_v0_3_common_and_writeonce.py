"""Shared-helper extraction safety net, global-uid properties, and lock-absence."""

import pytest

from container.common.ids import make_global_uid
from container.v0_3 import build_session_container

from _factory_v0_3 import make_session


def test_v0_2_public_names_survive_common_extraction():
    # v0_2's historical import paths must still resolve after moving the helpers
    # into container.common.
    from container.v0_2.schema import (  # noqa: F401
        format_session_container_filename,
        generate_container_id,
        now_timestamp,
        today_token,
        validate_container_id,
    )
    assert validate_container_id(generate_container_id())


def test_global_uid_is_deterministic_and_collision_safe():
    a = make_global_uid("inst-A", "session", 42)
    assert a == make_global_uid("inst-A", "session", 42)          # deterministic
    assert a != make_global_uid("inst-B", "session", 42)          # instance disambiguates
    assert a != make_global_uid("inst-A", "set", 42)              # entity disambiguates
    assert len(a) == 16


def test_v0_3_has_no_lock(tmp_path):
    _, _, path = build_session_container(make_session(), tmp_path)
    from container import is_container_locked
    with pytest.raises(Exception):
        # v0.3 is write-once; container_manager / lock is intentionally absent.
        is_container_locked(path)
