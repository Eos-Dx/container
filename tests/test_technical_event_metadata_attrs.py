from pathlib import Path

import h5py
import numpy as np

from container.v0_2 import schema, technical_container


def _make_detector_config():
    return [
        {
            "id": "det_primary",
            "alias": "PRIMARY",
            "type": "Advacam",
            "size": {"width": 8, "height": 8},
            "pixel_size_um": [55.0, 55.0],
        }
    ]


def test_technical_event_stores_acquisition_metadata(tmp_path):
    data_path = Path(tmp_path) / "dark_primary.npy"
    np.save(data_path, np.ones((8, 8), dtype=np.float32))

    aux_measurements = {
        "DARK": {
            "PRIMARY": {
                "file_path": str(data_path),
                "integration_time_ms": 2500.0,
                "n_frames": 7,
                "thickness": 1.3,
            }
        }
    }

    _container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(tmp_path),
        aux_measurements=aux_measurements,
        poni_data={},
        detector_config=_make_detector_config(),
        active_detector_ids=["det_primary"],
        distances_cm={"PRIMARY": 17.0},
    )

    with h5py.File(file_path, "r") as file_handle:
        event_id = schema.format_technical_event_id(1)
        det_group = file_handle[f"{schema.GROUP_TECHNICAL}/{event_id}/det_primary"]
        assert float(det_group.attrs[schema.ATTR_INTEGRATION_TIME_MS]) == 2500.0
        assert int(det_group.attrs[schema.ATTR_N_FRAMES]) == 7
        assert float(det_group.attrs[schema.ATTR_THICKNESS]) == 1.3


def test_technical_container_filename_uses_compact_integer_distance(tmp_path):
    _container_id, file_path = technical_container.create_technical_container(
        folder=tmp_path,
        distance_cm=17.0,
    )

    filename = Path(file_path).name
    assert "_17cm_" in filename
    assert "17p00cm" not in filename


def test_technical_event_stores_txt_dsc_sidecar_as_raw_dsc(tmp_path):
    raw_dir = Path(tmp_path) / "raw"
    raw_dir.mkdir()
    txt_path = raw_dir / "AGBH_PRIMARY.txt"
    dsc_path = raw_dir / "AGBH_PRIMARY.txt.dsc"
    txt_path.write_text("1 2\n3 4\n", encoding="utf-8")
    dsc_path.write_text("[F0]\nType=i16\n", encoding="utf-8")

    _container_id, file_path = technical_container.create_technical_container(
        folder=tmp_path,
        distance_cm=17.0,
    )
    technical_container.write_detector_config(
        file_path=file_path,
        detectors_config=_make_detector_config(),
        active_detector_ids=["det_primary"],
    )
    technical_container.add_technical_event(
        file_path=file_path,
        event_index=1,
        technical_type="AGBH",
        measurements={
            "PRIMARY": {
                "data": np.ones((8, 8), dtype=np.float32),
                "detector_id": "det_primary",
                "source_file": str(txt_path),
            }
        },
        timestamp="2026-05-05 12:00:00",
        distances_cm={"PRIMARY": 17.0},
    )

    with h5py.File(file_path, "r") as file_handle:
        event_id = schema.format_technical_event_id(1)
        blob_group = file_handle[f"{schema.GROUP_TECHNICAL}/{event_id}/det_primary/blob"]
        assert sorted(blob_group.keys()) == ["raw_dsc", "raw_txt"]
        assert blob_group["raw_dsc"].attrs["file_format"] == "dsc"
        assert blob_group["raw_dsc"].attrs["source_filename"] == dsc_path.name


def test_technical_event_stores_only_raw_txt_and_txt_dsc_sidecars(tmp_path):
    raw_dir = Path(tmp_path) / "raw"
    raw_dir.mkdir()
    npy_path = raw_dir / "AGBH_PRIMARY.npy"
    txt_path = raw_dir / "AGBH_PRIMARY.txt"
    dsc_path = raw_dir / "AGBH_PRIMARY.txt.dsc"
    np.save(npy_path, np.ones((8, 8), dtype=np.float32))
    txt_path.write_text("1 2\n3 4\n", encoding="utf-8")
    dsc_path.write_text("[F0]\nType=i16\n", encoding="utf-8")

    _container_id, file_path = technical_container.create_technical_container(
        folder=tmp_path,
        distance_cm=17.0,
    )
    technical_container.write_detector_config(
        file_path=file_path,
        detectors_config=_make_detector_config(),
        active_detector_ids=["det_primary"],
    )
    technical_container.add_technical_event(
        file_path=file_path,
        event_index=1,
        technical_type="AGBH",
        measurements={
            "PRIMARY": {
                "data": np.ones((8, 8), dtype=np.float32),
                "detector_id": "det_primary",
                "source_file": str(npy_path),
            }
        },
        timestamp="2026-05-05 12:00:00",
        distances_cm={"PRIMARY": 17.0},
    )

    with h5py.File(file_path, "r") as file_handle:
        event_id = schema.format_technical_event_id(1)
        blob_group = file_handle[f"{schema.GROUP_TECHNICAL}/{event_id}/det_primary/blob"]
        assert sorted(blob_group.keys()) == ["raw_dsc", "raw_txt"]
        assert blob_group["raw_txt"].attrs["source_filename"] == txt_path.name
        assert blob_group["raw_dsc"].attrs["source_filename"] == dsc_path.name
        processed = file_handle[f"{schema.GROUP_TECHNICAL}/{event_id}/det_primary/{schema.DATASET_PROCESSED_SIGNAL}"]
        np.testing.assert_array_equal(processed[()], np.ones((8, 8), dtype=np.float32))


def test_technical_event_finds_extensionless_raw_txt_for_npy_source(tmp_path):
    raw_dir = Path(tmp_path) / "raw"
    raw_dir.mkdir()
    npy_path = raw_dir / "AGBH_PRIMARY.npy"
    txt_path = raw_dir / "AGBH_PRIMARY"
    dsc_path = raw_dir / "AGBH_PRIMARY.txt.dsc"
    np.save(npy_path, np.ones((8, 8), dtype=np.float32))
    txt_path.write_text("1 2\n3 4\n", encoding="utf-8")
    dsc_path.write_text("[F0]\nType=i16\n", encoding="utf-8")

    _container_id, file_path = technical_container.create_technical_container(
        folder=tmp_path,
        distance_cm=17.0,
    )
    technical_container.write_detector_config(
        file_path=file_path,
        detectors_config=_make_detector_config(),
        active_detector_ids=["det_primary"],
    )
    technical_container.add_technical_event(
        file_path=file_path,
        event_index=1,
        technical_type="AGBH",
        measurements={
            "PRIMARY": {
                "data": np.ones((8, 8), dtype=np.float32),
                "detector_id": "det_primary",
                "source_file": str(npy_path),
            }
        },
        timestamp="2026-05-05 12:00:00",
        distances_cm={"PRIMARY": 17.0},
    )

    with h5py.File(file_path, "r") as file_handle:
        event_id = schema.format_technical_event_id(1)
        blob_group = file_handle[f"{schema.GROUP_TECHNICAL}/{event_id}/det_primary/blob"]
        assert sorted(blob_group.keys()) == ["raw_dsc", "raw_txt"]
        assert blob_group["raw_txt"].attrs["source_filename"] == txt_path.name
        assert blob_group["raw_dsc"].attrs["source_filename"] == dsc_path.name
