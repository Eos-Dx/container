# eosdx-container

Standalone `container` package extracted from `xrd-analysis`.

## Layout

- `src/` contains the installable Python package.
- `tests/` contains lightweight smoke tests for the standalone repo.

## Package

This repo provides versioned HDF5 container support used by DiFRA:

- `container.v0_1`
- `container.v0_2`

## Development

Install in editable mode:

```bash
pip install -e .
pytest
```
