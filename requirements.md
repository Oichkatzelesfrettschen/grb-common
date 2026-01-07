# Installation Requirements (grb-common)

`grb-common` is a Python library submodule (`src/grb_common`) providing shared
infrastructure for GRB astrophysics analysis.

## Prerequisites

- Python `>=3.9`

## Install (dev)

```bash
cd grb-common
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

Optional extras (choose what you need):
- `pip install -e '.[plotting]'`
- `pip install -e '.[fitting]'`
- `pip install -e '.[io]'`

## Tests and gates

- Tests: `pytest`
- Strict gates (warnings-as-errors on the contract surface): run from repo root
  with `scripts/audit/run_tiers.sh`.

