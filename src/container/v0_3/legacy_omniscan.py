"""The legacy backfill's v1 set metadata, brought to manifest v2's shape.

container_prep assembled the OmniScan-era corpus (producer
``omniscan-backfill``) with each set's values flat beside its
``backfill_provenance`` — the one v1 producer whose set metadata was not
namespaces. ``manifest.upgrade`` hands those sets here so a legacy v1 zip
still reads as v2.

Temporary: delete this module, and its call in
``manifest.upgrade_v1_set_metadata``, once no legacy v1 zip needs reading.
Rows already stored were converted by the reader's own migration.
"""

from typing import Optional

PRODUCER = "omniscan-backfill"
NAMESPACE = "omniscan"

# its calibration frame kinds, each the same fact as the set's type name
_FRAME_KIND_TYPE = {
    "AgBh": "Silver Behenate", "Background": "Background", "Dark": "Dark", "LaB6": "LaB6",
}


def upgrade_v1_set_metadata(metadata: Optional[dict], set_entry: dict) -> dict:
    """The flat values (``position``, ``is_rescan``, transmission, ...) and
    ``backfill_provenance`` go under ``omniscan``, provenance as
    ``provenance``. Three values only restate another field and are dropped
    when they do: ``thickness_raw_mm`` (= ``sample_thickness_mm``),
    ``frame_kind`` (= the type name) and ``phase`` (always ``Phase1``). One
    that disagrees is kept, under ``omniscan``."""
    metadata = dict(metadata or {})
    flat = {k: v for k, v in metadata.items() if not isinstance(v, dict)}
    out = {k: v for k, v in metadata.items()
           if isinstance(v, dict) and k != "backfill_provenance"}
    if "thickness_raw_mm" in flat and flat["thickness_raw_mm"] == set_entry.get("sample_thickness_mm"):
        del flat["thickness_raw_mm"]
    if "frame_kind" in flat and \
            _FRAME_KIND_TYPE.get(flat["frame_kind"]) == set_entry.get("measurement_type_name"):
        del flat["frame_kind"]
    if flat.get("phase") == "Phase1":
        del flat["phase"]
    omniscan = flat
    if "backfill_provenance" in metadata:
        omniscan["provenance"] = metadata["backfill_provenance"]
    if omniscan:
        out[NAMESPACE] = omniscan
    return out
