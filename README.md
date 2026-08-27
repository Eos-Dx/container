# eosdx-container

Primary standalone HDF5 container library for Eos-Dx.

This repo is the canonical Python package for reading, writing, validating, and
locking DiFRA-compatible data containers outside the old monorepo layout.

## Repository Role

- `container` is a reusable library package.
- It does not depend on `difra`.
- `difra` consumes it at runtime for technical/session container workflows.

## Package Surface

Top-level helpers exposed by `container`:

- `open_container(...)`
- `open_container_bundle(...)`
- `create_container_bundle(...)`
- `lock_container(...)`
- `unlock_container(...)`
- `is_container_locked(...)`

Versioned implementations are provided under:

- `container.v0_1`
- `container.v0_2`
- `container.v0_3`

## EosCloud upload contract

The library is the single writer of the session upload zip and the single
upload client, shared by every producer (EoScan, `container_prep`):

- `container.v0_3.write_zip(payload, frame_source, out_path, *, header_source=None, mask_source=None)`
  — `manifest.json` (manifest v1, `container.v0_3.MANIFEST_VERSION`) plus the
  original frame bytes, headers and masks. `frame_source(measurement_uid)`
  returns bytes or a path; producers resolve frames from their own layouts.
- `container.upload_zip(zip_path, session_uid, *, url, token)` — sha256 →
  presign → PUT → complete against an EosCloud instance. Stdlib only, no AWS
  SDK. Raises `AlreadyUploaded` (409) or `UploadError`.

Neither reads settings nor takes an ORM object; URL, token and frame
resolution are explicit arguments.

## Layout

- `src/container/` contains the installable Python package.
- `src/container/v0_1/` contains legacy format support.
- `src/container/v0_2/` contains the NeXus-based DIFRA container model.
- `src/container/v0_3/` contains the current XRD session container and the EosCloud upload contract.
- `tests/` contains standalone import and behavior checks.

## Development

Install in editable mode and run the package tests:

```bash
pip install -e .
pytest
```

Minimal usage:

```python
from container import open_container

container = open_container("path/to/file.nxs.h5")
```
