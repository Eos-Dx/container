from container.v0_2 import schema


def test_technical_container_filename_contract():
    assert (
        schema.format_technical_container_filename(
            "123044177fc94dca",
            distance_cm=17.0,
            date_token="20260506",
        )
        == "technical_123044177fc94dca_17cm_20260506.nxs.h5"
    )


def test_technical_container_filename_keeps_decimal_distance_safe():
    assert (
        schema.format_technical_container_filename(
            "123044177fc94dca",
            distance_cm=17.25,
            date_token="20260506",
        )
        == "technical_123044177fc94dca_17p25cm_20260506.nxs.h5"
    )


def test_session_container_filename_contract():
    assert (
        schema.format_session_container_filename(
            "fcf2fbc8f0c3419b",
            sample_id="378776__377667_P116_WH_S03",
            date_token="20260429",
        )
        == "session_fcf2fbc8f0c3419b_378776__377667_P116_WH_S03_20260429.nxs.h5"
    )


def test_h5_group_id_contract():
    assert schema.format_technical_event_id(3) == "tech_evt_000003"
    assert schema.format_point_id(1) == "pt_001"
    assert schema.format_measurement_id(1) == "meas_000000001"
    assert schema.format_analytical_measurement_id(2) == "ana_000000002"
