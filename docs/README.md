# grb-common Documentation

Shared infrastructure for Gamma-Ray Burst (GRB) astrophysics.

## Modules

- `grb_common.constants`: Physical constants in CGS units.
- `grb_common.cosmology`: Cosmological distance calculations.
- `grb_common.extinction`: Dust extinction laws and corrections.
- `grb_common.fitting`: Bayesian fitting utilities.
- `grb_common.plotting`: Standardized plotting styles.

## Quick Start

```python
from grb_common.cosmology import luminosity_distance
from grb_common.constants import C

dL = luminosity_distance(z=1.0)
print(f"Luminosity distance at z=1: {dL:.2e} cm")
```
