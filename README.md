# grb-common

Shared infrastructure for gamma-ray burst (GRB) astrophysics analysis.

## Overview

This package consolidates common utilities used across multiple GRB afterglow modeling codes:

- **ASGARD_GRBAfterglow**: High-accuracy afterglow simulation (Fortran/Python)
- **boxfit**: Pre-computed hydrodynamics fitting (C++)
- **JetFit**: Structured jet model fitting (Python)
- **PyGRB**: Prompt emission pulse fitting (Python)

## Features

- **Physical Constants**: CGS units derived from astropy.constants
- **Cosmology**: Distance calculations with standard cosmologies (Planck18, etc.)
- **Extinction**: Dust corrections (Fitzpatrick99, Cardelli89, SMC, LMC)
- **Data I/O**: Multi-format loaders with standardized HDF5 schema
- **Fitting**: Unified sampler interface (emcee, dynesty, pymultinest)
- **Plotting**: Publication-quality figures for light curves, SEDs, corners

## Installation

```bash
# Core package (constants, cosmology)
pip install grb-common

# With all optional dependencies
pip install grb-common[all]

# Specific extras
pip install grb-common[fitting]     # emcee
pip install grb-common[nested]      # dynesty
pip install grb-common[plotting]    # matplotlib, corner
pip install grb-common[io]          # h5py, pandas
```

## Quick Start

```python
from grb_common import constants
from grb_common.cosmology import luminosity_distance

# Physical constants (CGS)
print(f"Speed of light: {constants.C_LIGHT:.3e} cm/s")
print(f"Thomson cross-section: {constants.SIGMA_T:.3e} cm^2")

# Cosmological distances
z = 0.0098  # GRB 170817A redshift
dL = luminosity_distance(z, cosmology='Planck18')
print(f"Luminosity distance: {dL:.3e} cm")
```

## Data Loading

```python
from grb_common.io import load_lightcurve, GRBObservation

# Auto-detect format (ASGARD txt, boxfit txt, FITS, CSV, HDF5)
lc = load_lightcurve("grb170817_xray.txt")

# Access standardized fields
print(f"Times: {lc.time[:5]} s")
print(f"Fluxes: {lc.flux[:5]} erg/cm^2/s")
```

## Fitting

```python
from grb_common.fitting import GRBFitter, UniformPrior, LogUniformPrior

# Define model (your afterglow code)
def my_model(params, times, frequencies):
    E_iso, n, epsilon_e, epsilon_B, p = params
    # ... compute flux ...
    return flux

# Set up fitter
fitter = GRBFitter(
    data="grb170817.h5",
    model=my_model,
    parameters={
        'E_iso': LogUniformPrior(1e50, 1e54),
        'n': LogUniformPrior(1e-5, 1),
        'epsilon_e': UniformPrior(0.01, 0.5),
        'epsilon_B': LogUniformPrior(1e-5, 0.1),
        'p': UniformPrior(2.0, 3.0),
    },
    sampler='dynesty',
)

# Run
result = fitter.run(n_samples=10000)
result.corner_plot()
result.save("chains.h5")
```

## Modules

| Module | Description |
|--------|-------------|
| `constants` | Physical constants (CGS, astropy-based) |
| `cosmology` | Distance calculations, time dilation, k-corrections |
| `extinction` | Dust extinction laws and dereddening |
| `io` | Data loaders, writers, standardized schemas |
| `fitting` | Sampler interface, likelihoods, priors |
| `plotting` | Light curves, SEDs, corner plots |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPL-3.0-or-later

## Citation

If you use grb-common in your research, please cite:

```bibtex
@software{grb_common,
  author = {Afrauthihinngreygaard, Deirikr Jaiusadastra},
  title = {grb-common: Shared infrastructure for GRB astrophysics},
  year = {2025},
  url = {https://github.com/Oichkatzelesfrettschen/grb-common}
}
```
