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
    out.update(_scalar_datasets(group))
    return out


def _scalar_datasets(group):
    """Decoded {name -> value} for the group's scalar (shape ()) datasets only."""
    return {name: _decode(obj[()]) for name, obj in group.items()
            if isinstance(obj, h5py.Dataset) and obj.shape == ()}


def _child_by_prefix(group, prefix):
    """First child whose name starts with ``prefix`` — group names are
    `<kind>_<index/id>_<label>` and lookups match on the indexed prefix only."""
    for name in group:
        if name.startswith(prefix):
            return group[name]
    return None


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
                    # structural identity = scalar datasets; descriptive metadata
                    # lives in group attrs (+ a leftover JSON blob), surfaced via
                    # sample_metadata() and kept out of this flat identity view.
                    sample_fields = _scalar_datasets(session["sample"])
                    sample_fields.pop(S.DS_SAMPLE_METADATA, None)
                    meta.update(sample_fields)
                if "instrument" in session:
                    meta.update(_read_fields(session["instrument"]))
            return meta

    def sample_metadata(self) -> Optional[Dict[str, Any]]:
        """Free-form descriptive (e.g. clinical) sample metadata, or None.

        Reassembled from the flat scalar attrs on the sample group plus any
        nested/None leftovers kept in the JSON blob."""
        with h5py.File(self.file_path, "r") as f:
            session = f.get(S.GROUP_SESSION)
            if session is None or "sample" not in session:
                return None
            sample = session["sample"]
            out = {k: _decode(v) for k, v in sample.attrs.items()
                   if k != S.ATTR_NX_CLASS}
            blob = read_json_dataset(sample, S.DS_SAMPLE_METADATA, default=None)
            if blob:
                out.update(blob)
            return out or None

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
                blob = read_json_dataset(grp[name], S.DS_METADATA, default=None)
                if blob:
                    entry[S.DS_METADATA] = blob
                out.append(entry)
        return out

    def _set_by_index(self, f, set_idx: int):
        grp = f.get(S.GROUP_SETS)
        return _child_by_prefix(grp, f"set_{set_idx:03d}_") if grp is not None else None

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
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_MEASUREMENTS not in grp:
                return None
            m = _child_by_prefix(grp[S.GROUP_MEASUREMENTS], f"det_{detector_id}_")
            return m[S.DS_DATA][()] if m is not None and S.DS_DATA in m else None

    def raw_file(self, set_idx: int, detector_id: int) -> Optional[bytes]:
        """Original vendor source bytes (.gfrm/.png/.h5) a detector wrote."""
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_MEASUREMENTS not in grp:
                return None
            m = _child_by_prefix(grp[S.GROUP_MEASUREMENTS], f"det_{detector_id}_")
            if m is None or S.DS_RAW_FILE not in m:
                return None
            return m[S.DS_RAW_FILE][()].tobytes()

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
        """Results keyed by @check_name, in priority order (group names sort
        by the priority-ordered index prefix)."""
        out: Dict[str, Dict[str, Any]] = {}
        with h5py.File(self.file_path, "r") as f:
            grp = self._set_by_index(f, set_idx)
            if grp is None or S.GROUP_QC not in grp:
                return out
            qc = grp[S.GROUP_QC]
            for name in sorted(qc):
                attrs = {k: _decode(v) for k, v in qc[name].attrs.items()}
                out[attrs[S.ATTR_CHECK_NAME]] = attrs
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
