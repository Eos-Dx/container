"""Shared HDF5 / NeXus attribute and class-name strings.

Trivial constants both versions can agree on. Kept here so a version package
never has to reach into another version for them.
"""

# NeXus base-class machinery
ATTR_NX_CLASS = "NX_class"
NX_ROOT = "NXroot"
NX_ENTRY = "NXentry"
NX_SAMPLE = "NXsample"
NX_INSTRUMENT = "NXinstrument"
NX_DETECTOR = "NXdetector"
NX_DATA = "NXdata"
NX_COLLECTION = "NXcollection"

# NXdata plotting convention
ATTR_SIGNAL = "signal"
ATTR_AXES = "axes"
ATTR_UNITS = "units"
ATTR_DEFAULT = "default"

# Self-description
ATTR_FORMAT = "format"
ATTR_SCHEMA_VERSION = "schema_version"
ATTR_CONTAINER_TYPE = "container_type"

CONTAINER_TYPE_SESSION = "session"
CONTAINER_TYPE_COMBINED = "combined"   # model-run input: N sessions embedded side by side

# Compression levels
COMPRESSION_BLOB = 9
COMPRESSION_ARRAY = 4
