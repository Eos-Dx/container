"""Version-agnostic HDF5 write/read primitives.

Thin helpers over h5py used by the v0_3 builder/reader. No NeXus-layout opinions
live here — only generic group/attr/dataset I/O.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np

from . import constants as C

_STR = h5py.string_dtype("utf-8")


def _decode(value: Any) -> Any:
    """Normalise an h5py attr/scalar to a plain Python value."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_decode(v) for v in value.tolist()]
    return value


def set_attrs(obj, attrs: Optional[Dict[str, Any]]) -> None:
    """Set attributes, skipping any whose value is None."""
    if not attrs:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        obj.attrs[key] = value


def create_root(file_path: Union[str, Path], root_attrs: Dict[str, Any]) -> h5py.File:
    """Create a fresh container file (mode 'w'), tag it NXroot, write root attrs.

    Returns the open ``h5py.File`` for the caller to keep writing, then close.
    ``schema_version``/``format`` are passed in via ``root_attrs`` — no version
    is hardcoded here.
    """
    file_path = Path(file_path)
    f = h5py.File(file_path, "w")
    f.attrs[C.ATTR_NX_CLASS] = C.NX_ROOT
    set_attrs(f, root_attrs)
    return f


def make_group(parent, name: str, nx_class: Optional[str] = None,
               attrs: Optional[Dict[str, Any]] = None):
    """Create (or get) a child group, optionally tagging NX_class + attrs."""
    group = parent.require_group(name)
    if nx_class is not None:
        group.attrs[C.ATTR_NX_CLASS] = nx_class
    set_attrs(group, attrs)
    return group


def write_json_dataset(group, name: str, obj: Any) -> None:
    """Serialise ``obj`` to a UTF-8 JSON string dataset."""
    group.create_dataset(name, data=json.dumps(obj), dtype=_STR)


def write_text_dataset(group, name: str, text: str) -> None:
    group.create_dataset(name, data=text, dtype=_STR)


def write_scalar_dataset(group, name: str, value: Any,
                         units: Optional[str] = None):
    """Store a scalar as a NeXus-style field dataset, with an optional @units.

    Numeric physics quantities become datasets-with-units so generic NeXus tools
    (pyFAI/DAWN/silx) can read them; strings become UTF-8 string datasets.
    """
    if isinstance(value, str):
        ds = group.create_dataset(name, data=value, dtype=_STR)
    else:
        ds = group.create_dataset(name, data=value)
    if units:
        ds.attrs[C.ATTR_UNITS] = units
    return ds


def write_bytes_dataset(group, name: str, data: Union[bytes, Path, str],
                        compression: int = C.COMPRESSION_BLOB) -> None:
    """Store raw bytes (or a file's contents) as a gzip'd uint8 dataset."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    group.create_dataset(name, data=arr, compression="gzip",
                         compression_opts=compression)


def write_array_dataset(group, name: str, arr,
                        compression: int = C.COMPRESSION_ARRAY,
                        attrs: Optional[Dict[str, Any]] = None):
    """Store a numeric ndarray as a gzip'd dataset; return the dataset."""
    arr = np.asarray(arr)
    ds = group.create_dataset(name, data=arr, compression="gzip",
                              compression_opts=compression)
    set_attrs(ds, attrs)
    return ds


def read_attr(obj, key: str, default: Any = None) -> Any:
    if key not in obj.attrs:
        return default
    return _decode(obj.attrs[key])


def read_json_dataset(group, name: str, default: Any = None) -> Any:
    if name not in group:
        return default
    return json.loads(_decode(group[name][()]))


def get_container_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Return decoded root attributes (used by the loader for type dispatch)."""
    with h5py.File(file_path, "r") as f:
        return {key: _decode(val) for key, val in f.attrs.items()}
