"""Per-set QC, 1D integration (NeXus auto-plot contract), and processing log."""

import h5py
import numpy as np

from container import open_container
from container.v0_3 import ProcessingStepPayload, QCResultPayload, build_session_container

from _factory_v0_3 import make_session, make_set


def _build(tmp_path, set_kw=None):
    sp = make_set(**(set_kw or {}))
    _, _, path = build_session_container(make_session(sets=[sp]), tmp_path)
    return path


# ---------------- QC ----------------
def test_qc_roundtrip_and_multiple(tmp_path):
    checks = [
        QCResultPayload("symmetry", "PASS", "ok", {"symmetry_pct": 98.5}, {"x1": 1}, "t",
                        priority=10),
        QCResultPayload("transmission", "FAIL", "low", {"transmission_pct": 5.0}, {"y1": 2}, "t",
                        priority=20),
    ]
    sp = make_set(qc=False)
    sp.qc_results = checks
    _, _, path = build_session_container(make_session(sets=[sp]), tmp_path)
    c = open_container(path)
    qc = c.qc(1)
    # keyed by @check_name, priority order preserved
    assert list(qc) == ["symmetry", "transmission"]
    assert qc["transmission"]["verdict"] == "FAIL"
    assert qc["symmetry"]["priority"] == 10
    with h5py.File(path, "r") as f:
        from container.v0_3.utils import read_json_dataset
        g = f["/session/sets/set_001_sample_main/qc/qc_01_symmetry"]
        assert read_json_dataset(g, "metrics")["symmetry_pct"] == 98.5
        assert read_json_dataset(g, "parameters_snapshot")["x1"] == 1


def test_zero_qc_has_no_group(tmp_path):
    path = _build(tmp_path, {"qc": False})
    with h5py.File(path, "r") as f:
        assert "qc" not in f["/session/sets/set_001_sample_main"]


# ---------------- integration ----------------
def test_integration_nexus_plot_contract(tmp_path):
    path = _build(tmp_path)
    with h5py.File(path, "r") as f:
        integ = f["/session/sets/set_001_sample_main/integration"]
        assert integ.attrs["NX_class"] == "NXdata"
        assert integ.attrs["signal"] == "i"
        assert integ.attrs["axes"] == "q"
        assert integ["q"].attrs["units"] == "nm^-1"
        assert integ["q"].dtype == np.float64
        assert integ["q"].shape == integ["i"].shape == (2000,)


def test_absent_integration_has_no_group(tmp_path):
    path = _build(tmp_path, {"integration": False})
    with h5py.File(path, "r") as f:
        assert "integration" not in f["/session/sets/set_001_sample_main"]


# ---------------- processing ----------------
def test_processing_holds_config_no_steps(tmp_path):
    """With a recipe but no step log, /processing exists with config but no steps."""
    path = _build(tmp_path)
    with h5py.File(path, "r") as f:
        proc = f["/session/sets/set_001_sample_main/processing"]
        assert "config" in proc
        assert not [n for n in proc if n.startswith("step_")]
    c = open_container(path)
    assert c.processing(1) == []                         # no steps logged
    assert c.processing_config(1)["postprocessing_key"] == "sample"


def test_processing_steps_ordered_with_params(tmp_path):
    steps = [
        ProcessingStepPayload("normalize", "t0", "t1", {"snap": [64, 128]}, None, None),
        ProcessingStepPayload("integrate", "t1", "t2", {"npt": 2000}, "processed", "integration"),
    ]
    path = _build(tmp_path, {"processing_steps": steps})
    c = open_container(path)
    log = c.processing(1)
    assert [s["name"] for s in log] == ["step_01_normalize", "step_02_integrate"]
    assert log[1]["step_name"] == "integrate"
    assert log[1]["output_ref"] == "integration"
    with h5py.File(path, "r") as f:
        from container.v0_3.utils import read_json_dataset
        params = read_json_dataset(
            f["/session/sets/set_001_sample_main/processing/step_01_normalize"], "params")
        assert params["snap"] == [64, 128]
