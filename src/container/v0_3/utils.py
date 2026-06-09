"""v0.3 utility surface — generic HDF5 helpers plus container introspection."""

from pathlib import Path
from typing import Any, Dict, Union

import h5py

from container.common.hdf5 import _decode  # noqa: F401
from container.common.hdf5 import (  # noqa: F401  (re-exported surface)
    make_group,
    read_attr,
    read_json_dataset,
    set_attrs,
    write_array_dataset,
    write_bytes_dataset,
    write_json_dataset,
    write_text_dataset,
)

from . import schema as S


def get_container_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Root identity/type info, enriched with the session pk.

    The loader reads ``container_type`` from here to dispatch open(). We also
    surface the session identifiers for convenience.
    """
    with h5py.File(file_path, "r") as f:
        info = {key: _decode(val) for key, val in f.attrs.items()}
        if S.GROUP_SESSION in f:
            session = f[S.GROUP_SESSION]
            for key in (S.ATTR_SESSION_PK, S.ATTR_CATEGORY,
                        S.ATTR_SAMPLE_CLINICAL_NAME, S.ATTR_PATIENT_CLINICAL_NAME):
                if key in session.attrs:
                    info[key] = _decode(session.attrs[key])
    return info
