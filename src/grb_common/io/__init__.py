"""
Data I/O utilities for GRB astrophysics.

This module provides standardized data loading, writing, and schema
definitions for gamma-ray burst observations across multiple formats.

Supported formats:
    - ASGARD text format (.txt)
    - boxfit output format (.txt)
    - FITS tables (.fits)
    - CSV with auto-detection
    - HDF5 with standardized schema (.h5, .hdf5)

Schemas:
    LightCurve: Time-series flux measurements
    Spectrum: Frequency-dependent flux measurements
    GRBMetadata: Observation metadata
    GRBObservation: Complete observation container

Loaders:
    load_lightcurve: Auto-detect format and load light curve
    load_spectrum: Auto-detect format and load spectrum
    load_grb: Load complete observation from HDF5
    detect_format: Detect file format

Writers:
    save_lightcurve: Save light curve to HDF5
    save_spectrum: Save spectrum to HDF5
    save_grb: Save complete observation to HDF5

Usage:
    from grb_common.io import load_lightcurve, save_grb, LightCurve

    # Load data
    lc = load_lightcurve("grb170817_xray.txt")

    # Access fields
    print(f"Times: {lc.time[:5]} s")
    print(f"Fluxes: {lc.flux[:5]} erg/cm^2/s")

    # Save to HDF5
    save_lightcurve(lc, "output.h5")
"""

from .schemas import (
    LightCurvePoint,
    LightCurve,
    Spectrum,
    GRBMetadata,
    GRBObservation,
)

from .loaders import (
    detect_format,
    load_lightcurve,
    load_spectrum,
    load_grb,
)

from .writers import (
    save_lightcurve,
    save_spectrum,
    save_grb,
)

__all__ = [
    # Schemas
    "LightCurvePoint",
    "LightCurve",
    "Spectrum",
    "GRBMetadata",
    "GRBObservation",
    # Loaders
    "detect_format",
    "load_lightcurve",
    "load_spectrum",
    "load_grb",
    # Writers
    "save_lightcurve",
    "save_spectrum",
    "save_grb",
]
