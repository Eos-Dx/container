"""XRD acquisition session container — v0.3.

A vendor-neutral, write-once HDF5 container modelled as
session -> sets -> measurements with per-set QC, 1D integration and metadata.

Module surface required by the loader: ``utils`` (with ``get_container_info``),
``SessionContainer`` and ``TechnicalContainer``. v0.3 has no lock/transfer and no
standalone technical container, so those are intentionally absent (TechnicalContainer
is a stub that raises on open).
"""

__version__ = "0.3.0"

from . import schema
from . import utils
from . import validator
from . import writer
from .reader import SessionContainer, TechnicalContainer
from .writer import (
    DependencyRef,
    DetectorSpec,
    IntegrationPayload,
    MeasurementPayload,
    ProcessingStepPayload,
    QCResultPayload,
    SessionPayload,
    SetPayload,
    build_session_container,
)

__all__ = [
    "schema",
    "utils",
    "validator",
    "writer",
    "SessionContainer",
    "TechnicalContainer",
    "build_session_container",
    "SessionPayload",
    "SetPayload",
    "MeasurementPayload",
    "DetectorSpec",
    "QCResultPayload",
    "IntegrationPayload",
    "ProcessingStepPayload",
    "DependencyRef",
    "__version__",
]
