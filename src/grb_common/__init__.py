"""
grb-common: Shared infrastructure for GRB astrophysics analysis.

This package provides common utilities for gamma-ray burst (GRB) afterglow
modeling, fitting, and analysis across multiple specialized codes:
- ASGARD_GRBAfterglow (Fortran/Python)
- boxfit (C++)
- JetFit (Python)
- PyGRB (Python)

Modules:
    constants: Physical constants in CGS units (astropy-based)
    cosmology: Cosmological distance calculations
    extinction: Dust extinction corrections
    io: Data loading and standardized schemas
    fitting: MCMC/nested sampling infrastructure
    plotting: Publication-quality figures

Usage:
    from grb_common import constants
    from grb_common.cosmology import luminosity_distance
    from grb_common.fitting import GRBFitter

Installation:
    pip install grb-common              # Core only
    pip install grb-common[all]         # All optional dependencies
    pip install grb-common[fitting]     # With emcee
    pip install grb-common[nested]      # With dynesty
"""

__version__ = "0.1.0"
__author__ = "Deirikr Jaiusadastra Afrauthihinngreygaard"

# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name):
    """Lazy import submodules."""
    if name == "constants":
        from grb_common import constants
        return constants
    elif name == "cosmology":
        from grb_common import cosmology
        return cosmology
    elif name == "extinction":
        from grb_common import extinction
        return extinction
    elif name == "io":
        from grb_common import io
        return io
    elif name == "fitting":
        from grb_common import fitting
        return fitting
    elif name == "plotting":
        from grb_common import plotting
        return plotting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "constants",
    "cosmology",
    "extinction",
    "io",
    "fitting",
    "plotting",
]
