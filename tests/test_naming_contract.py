import h5py
import pytest

from container.v0_2 import schema, technical_container


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


@pytest.mark.parametrize(
    ("distance_cm", "expected_token"),
    [
        (2.0, "2cm"),
        (17.0, "17cm"),
    ],
)
def test_created_technical_container_filename_matches_root_distance_attr(
    tmp_path,
    distance_cm,
    expected_token,
):
    _container_id, file_path = technical_container.create_technical_container(
        folder=tmp_path,
        distance_cm=distance_cm,
    )

    assert f"_{expected_token}_" in file_path
    with h5py.File(file_path, "r") as h5f:
        assert float(h5f.attrs[schema.ATTR_DISTANCE_CM]) == distance_cm


def test_generate_from_aux_table_uses_primary_distance_as_root(tmp_path):
    raw_path = tmp_path / "raw.npy"
    import numpy as np

    np.save(raw_path, np.ones((2, 2), dtype=np.float32))
    _container_id, file_path = technical_container.generate_from_aux_table(
        folder=tmp_path,
        aux_measurements={"DARK": {"PRIMARY": str(raw_path)}},
        poni_data={},
        detector_config=[
            {"id": "det_secondary", "alias": "SECONDARY"},
            {"id": "det_primary", "alias": "PRIMARY"},
        ],
        active_detector_ids=["det_primary", "det_secondary"],
        distances_cm={"SECONDARY": 17.0, "PRIMARY": 2.0},
    )

    assert "_2cm_" in file_path
    with h5py.File(file_path, "r") as h5f:
        assert float(h5f.attrs[schema.ATTR_DISTANCE_CM]) == 2.0
