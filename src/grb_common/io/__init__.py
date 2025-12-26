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

Key classes:
    GRBObservation: Container for GRB observational data
    LightCurve: Time-series flux measurements
    Spectrum: Frequency-dependent flux measurements

Key functions:
    load_lightcurve: Auto-detect format and load light curve
    load_spectrum: Auto-detect format and load spectrum
    save_hdf5: Save to standardized HDF5 schema
"""

__all__ = [
    "GRBObservation",
    "LightCurve",
    "Spectrum",
    "load_lightcurve",
    "load_spectrum",
    "save_hdf5",
]

# Implementations will be added in Phase 2.5
