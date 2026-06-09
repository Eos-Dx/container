from container import open_container


def test_public_imports_are_exposed():
    assert callable(open_container)


def test_v0_3_module_surface():
    from container.v0_3 import (  # noqa: F401
        SessionContainer,
        TechnicalContainer,
        build_session_container,
        utils,
    )
    assert callable(SessionContainer.open)
    assert callable(utils.get_container_info)
    assert callable(build_session_container)
