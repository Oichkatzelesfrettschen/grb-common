# Migration Guide: Using grb-common

This guide shows how to migrate existing GRB analysis code to use the consolidated
`grb-common` library.

## Installation

```bash
# Basic installation
pip install grb-common

# With fitting backends (emcee, dynesty)
pip install grb-common[fitting]

# With plotting support
pip install grb-common[plotting]

# Everything
pip install grb-common[all]
```

## Migration Examples

### 1. Physical Constants

**Before (manual definitions):**
```python
import numpy as np

c = 2.998e10          # cm/s
m_e = 9.109e-28       # g
sigma_T = 6.652e-25   # cm^2
```

**After (grb-common):**
```python
from grb_common.constants import C_LIGHT, M_ELECTRON, SIGMA_T
```

### 2. Cosmology Calculations

**Before (direct astropy):**
```python
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_L = cosmo.luminosity_distance(z).to(u.cm).value
```

**After (grb-common):**
```python
from grb_common.cosmology import luminosity_distance

# Returns cm directly, ready for CGS calculations
d_L = luminosity_distance(z)  # Uses Planck18 by default

# Or specify cosmology
d_L = luminosity_distance(z, cosmology="WMAP9")
```

### 3. Extinction Correction

**Before (ASGARD extinc.py):**
```python
from extinction import fitzpatrick99 as f99
import numpy as np

def opt_extinction(mag_data, mag_err, frequency, Rv, Ebv, zeropointflux):
    wave = np.array([2.997e10/frequency*1e8])
    Av = Rv * Ebv
    mag_data_deredden = mag_data - f99(wave, Av, Rv)
    flux_data_deredden = 10**(0.4*(zeropointflux-mag_data_deredden))
    flux_data_err = 0.4*np.log(10.0)*flux_data_deredden*mag_err
    return flux_data_deredden, flux_data_err
```

**After (grb-common):**
```python
from grb_common.extinction import deredden, fitzpatrick99

# Direct flux dereddening
wavelength = 2.997e10 / frequency * 1e8  # Angstroms
flux_corrected = deredden(flux, wavelength, Av=Rv*Ebv, Rv=Rv)

# Or using magnitude
A_lambda = fitzpatrick99(wavelength, Av=Rv*Ebv, Rv=Rv)
mag_corrected = mag - A_lambda
```

### 4. MCMC Priors

**Before (custom prior functions):**
```python
def lnprior(theta):
    E_iso, p, n, epsilon_e, epsilon_b = theta
    if (50 < E_iso < 55 and 2.0 < p < 2.9 and ...):
        return 0.0
    return -np.inf
```

**After (grb-common):**
```python
from grb_common.fitting import (
    LogUniformPrior, UniformPrior, CompositePrior
)

priors = CompositePrior({
    'E_iso': LogUniformPrior(1e50, 1e55),  # Linear space, not log
    'p': UniformPrior(2.0, 2.9),
    'n': LogUniformPrior(1e-5, 1.0),
    'epsilon_e': LogUniformPrior(1e-4, 0.5),
    'epsilon_b': LogUniformPrior(1e-8, 0.1),
})

# For emcee: use log_prob
def log_posterior(theta):
    return priors.log_prob(theta) + log_likelihood(theta)

# For nested sampling: use prior_transform
# Maps unit cube [0,1]^n to parameter space
theta = priors.prior_transform(u)
```

### 5. Sampler Interface

**Before (direct emcee):**
```python
import emcee

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
sampler.run_mcmc(initial, nsteps, progress=True)
samples = sampler.flatchain[burnin:]
```

**After (grb-common unified interface):**
```python
from grb_common.fitting.backends import get_sampler

# Works with emcee, dynesty, or pymultinest
sampler = get_sampler(
    'emcee',
    log_prob=log_posterior,
    n_params=5,
    param_names=['E_iso', 'p', 'n', 'epsilon_e', 'epsilon_b'],
)
result = sampler.run(n_walkers=32, n_steps=5000)

# Unified result interface
print(result.mean('E_iso'))
print(result.percentile('p', [16, 50, 84]))
result.corner_plot()
result.save('chains.h5')
```

### 6. Corner Plots

**Before (direct corner):**
```python
import corner

fig = corner.corner(
    samples,
    labels=parameters,
    truths=[E[0], p[0], ...],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
)
fig.savefig("corner.pdf")
```

**After (grb-common):**
```python
from grb_common.plotting import corner_plot, set_style

set_style('apj')  # Publication-ready formatting

fig = corner_plot(
    result,
    truths={'E_iso': 1e52, 'p': 2.3},
    quantiles=[0.16, 0.5, 0.84],
)
fig.savefig("corner.pdf")
```

### 7. Light Curve Plotting

**Before (matplotlib directly):**
```python
import matplotlib.pyplot as plt

plt.errorbar(time, flux, yerr=flux_err, fmt='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time [s]')
plt.ylabel('Flux [erg/cm^2/s/Hz]')
```

**After (grb-common):**
```python
from grb_common.plotting import plot_lightcurve, set_style
from grb_common.io import load_lightcurve

set_style('mnras')
lc = load_lightcurve('data.txt')
ax = plot_lightcurve(lc, flux_unit='mJy')
ax.figure.savefig('lightcurve.pdf')
```

## Data Format Compatibility

grb-common can read data formats from multiple GRB projects:

```python
from grb_common.io import load_grb, detect_format

# Auto-detect format
format_name = detect_format('data.txt')
print(f"Detected format: {format_name}")

# Load observation data
obs = load_grb('grb170817_data.txt')

# Access light curves by band
xray = obs.get_band('X-ray')
optical = obs.get_band('optical_R')

# Export to HDF5
from grb_common.io import save_grb
save_grb(obs, 'grb170817.h5')
```

## Gradual Migration Strategy

1. **Phase 1**: Add grb-common as optional dependency
   ```toml
   [project.optional-dependencies]
   grb-common = ["grb-common>=0.1.0"]
   ```

2. **Phase 2**: Create compatibility wrappers
   ```python
   try:
       from grb_common.constants import C_LIGHT, M_ELECTRON
   except ImportError:
       C_LIGHT = 2.998e10
       M_ELECTRON = 9.109e-28
   ```

3. **Phase 3**: Replace implementations incrementally
4. **Phase 4**: Remove local implementations and require grb-common

## API Reference

See the full API documentation at: https://grb-common.readthedocs.io/
