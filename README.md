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

## Layout

- `src/container/` contains the installable Python package.
- `src/container/v0_1/` contains legacy format support.
- `src/container/v0_2/` contains the current NeXus-based container model.
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
