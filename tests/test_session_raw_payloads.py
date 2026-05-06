from pathlib import Path

import h5py
import numpy as np

from container.v0_2 import writer


def test_session_measurement_stores_npy_with_txt_and_txt_dsc_sidecars(tmp_path):
    npy_path = Path(tmp_path) / "capture_PRIMARY.npy"
    txt_path = Path(tmp_path) / "capture_PRIMARY.txt"
    dsc_path = Path(tmp_path) / "capture_PRIMARY.txt.dsc"
    np.save(npy_path, np.ones((4, 4), dtype=np.float32))
    txt_path.write_text("1 2\n3 4\n", encoding="utf-8")
    dsc_path.write_text("[F0]\nType=i16\n", encoding="utf-8")

    _container_id, file_path = writer.create_session_container(
        folder=tmp_path,
        sample_id="sample",
        operator_id="operator",
        site_id="site",
        machine_name="machine",
        beam_energy_keV=12.0,
        acquisition_date="2026-05-06",
    )
    measurement_path = writer.add_measurement(
        file_path=file_path,
        point_index=1,
        measurement_data={"det_primary": np.ones((4, 4), dtype=np.float32)},
        detector_metadata={"det_primary": {"integration_time_ms": 1000.0}},
        poni_alias_map={"PRIMARY": "det_primary"},
        raw_files={"det_primary": {"raw_npy": str(npy_path)}},
    )

    with h5py.File(file_path, "r") as file_handle:
        blob_group = file_handle[f"{measurement_path}/det_primary/blob"]
        assert sorted(blob_group.keys()) == ["raw_dsc", "raw_npy", "raw_txt"]
        assert blob_group["raw_txt"].attrs["source_filename"] == txt_path.name
        assert blob_group["raw_dsc"].attrs["source_filename"] == dsc_path.name
        assert blob_group["raw_npy"].attrs["source_filename"] == npy_path.name
        assert blob_group["raw_dsc"].attrs["file_format"] == "dsc"
