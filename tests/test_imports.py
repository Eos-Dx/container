from hardware.container import open_container


def test_public_imports_are_exposed():
    assert callable(open_container)
