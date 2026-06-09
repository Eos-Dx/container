"""Read-side access for v0.3 XRD session containers."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from container.common.hdf5 import _decode, read_json_dataset

from . import schema as S
from . import validator as V


def _read_fields(group):
    """Decoded {attr/scalar-dataset name -> value} for a metadata group."""
    out = {k: _decode(v) for k, v in group.attrs.items()}
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset) and obj.shape == ():
            out[name] = _decode(obj[()])
    return out


class SessionContainer:
    """Lazy read accessor over a v0.3 session container."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    @classmethod
    def open(cls, file_path: Union[str, Path], validate: bool = True) -> "SessionContainer":
        inst = cls(file_path)
        if validate:
            ok, errors = V.validate_session_container(file_path)
            if not ok:
                detail = "; ".join(f"{e.path}: {e.message}"
                                   for e in errors if e.severity == V.SEVERITY_ERROR)
                raise ValueError(f"Invalid v0.3 session container: {detail}")
        return inst

    def validate(self) -> Tuple[bool, List[V.ValidationError]]:
        return V.validate_session_container(self.file_path)

    # ---- metadata ----
    def session_meta(self) -> Dict[str, Any]:
        """Flat view: root identity + /session + sample + instrument fields."""
        with h5py.File(self.file_path, "r") as f:
            meta = {k: _decode(v) for k, v in f.attrs.items()}
            if S.GROUP_SESSION in f:
                session = f[S.GROUP_SESSION]
                meta.update({k: _decode(v) for k, v in session.attrs.items()})
                if "sample" in session:
                    meta.update(_read_fields(session["sample"]))
                if "instrument" in session:
                    meta.update(_read_fields(session["instrument"]))
            return meta

    def dependencies(self) -> List[Dict[str, Any]]:
        with h5py.File(self.file_path, "r") as f:
            session = f.get(S.GROUP_SESSION)
            if session is None or S.DS_DEPENDENCIES not in session:
                return []
            return read_json_dataset(session, S.DS_DEPENDENCIES, default=[])

    # ---- detector-set catalog (session-level; one entry per detector set) ----
    def detector_sets(self) -> List[Dict[str, Any]]:
        """Each catalogued detector set: its attrs + ``layout`` + ``detectors``."""
        out: List[Dict[str, Any]] = []
        with h5py.File(self.file_path, "r") as f:
            container = f.get(S.GROUP_DETECTOR_SETS)
            if container is None:
                return out
            for name in sorted(container):
                ds = container[name]
                entry: Dict[str, Any] = {"name": name,
                                         **{k: _decode(v) for k, v in ds.attrs.items()}}
                if S.DS_LAYOUT in ds:
                    entry[S.DS_LAYOUT] = read_json_dataset(ds, S.DS_LAYOUT, default={})
                catalog = ds.get(S.NAME_DETECTORS)
                entry[S.NAME_DETECTORS] = (
                    [{"name": dn, **_read_fields(catalog[dn])} for dn in sorted(catalog)]
                    if catalog is not None else []
                )
                out.append(entry)
        return out

    def detectors(self) -> List[Dict[str, Any]]:
        """Flat list of every detector across all detector sets (convenience)."""
        out: List[Dict[str, Any]] = []
        for ds in self.detector_sets():
            out.extend(ds[S.NAME_DETECTORS])
        return out

    # ---- sets ----
    def _sets_group(self, f):
        return f.get(S.GROUP_SETS)

    def sets(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with h5py.File(self.file_path, "r") as f:
            grp = self._sets_group(f)
            if grp is None:
                return out
            for name in sorted(grp):
                entry = {"name": name,
                         **{k: _decode(v) for k, v in grp[name].attrs.items()}}
                if S.GROUP_ACQUISITION in grp[name]:
                    entry[S.GROUP_ACQUISITION] = _read_fields(grp[name][S.GROUP_ACQUISITION])
                out.append(entry)
        return out

    def _set_by_index(self, f, set_idx: int):
        return f.get(f"{S.GROUP_SETS}/{S.format_set_id(set_idx)}")

    def measurements(self, set_idx: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_MEASUREMENTS not in grp:
                return out
            meas = grp[S.GROUP_MEASUREMENTS]
            for name in sorted(meas):
                out.append({"name": name, **_read_fields(meas[name])})
        return out

    def frame(self, set_idx: int, detector_id: int) -> Optional[np.ndarray]:
        """Decoded 2D frame a single detector produced in this set."""
        key = f"{S.GROUP_MEASUREMENTS}/{S.format_detector_id(detector_id)}/{S.DS_DATA}"
        return self._set_dataset(set_idx, key)

    def raw(self, set_idx: int) -> Optional[np.ndarray]:
        """Set-level decoded raw composite (stitched across detectors)."""
        return self._set_dataset(set_idx, f"{S.DS_RAW_2D}/{S.DS_DATA}")

    def processed(self, set_idx: int) -> Optional[np.ndarray]:
        """Set-level final processed 2D matrix."""
        return self._set_dataset(set_idx, f"{S.DS_PROCESSED}/{S.DS_DATA}")

    def _set_dataset(self, set_idx: int, rel_key: str) -> Optional[np.ndarray]:
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or rel_key not in grp:
                return None
            return grp[rel_key][()]

    def qc(self, set_idx: int) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_QC not in grp:
                return out
            for name, check in grp[S.GROUP_QC].items():
                out[name] = {k: _decode(v) for k, v in check.attrs.items()}
        return out

    def integration(self, set_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_INTEGRATION not in grp:
                return None
            integ = grp[S.GROUP_INTEGRATION]
            return integ[S.DS_Q][()], integ[S.DS_I][()]

    def processing(self, set_idx: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_PROCESSING not in grp:
                return out
            proc = grp[S.GROUP_PROCESSING]
            for name in sorted(n for n in proc if n.startswith("step_")):
                out.append({"name": name,
                            **{k: _decode(v) for k, v in proc[name].attrs.items()}})
        return out

    def processing_config(self, set_idx: int) -> Optional[Dict[str, Any]]:
        """Reduction recipe (workflow/postprocessing keys + params)."""
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_PROCESSING not in grp:
                return None
            proc = grp[S.GROUP_PROCESSING]
            if S.DS_PROCESSING_CONFIG not in proc:
                return None
            return read_json_dataset(proc, S.DS_PROCESSING_CONFIG, default=None)


class TechnicalContainer:
    """v0.3 defines no standalone technical container (contract stub)."""

    @classmethod
    def open(cls, file_path: Union[str, Path], validate: bool = True):
        raise NotImplementedError("v0.3 does not define a standalone technical container.")
