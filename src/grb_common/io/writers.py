"""
Data writers for GRB observations.

Writes data to HDF5 format with standardized schema for interoperability
between GRB analysis codes.

Usage:
    from grb_common.io import save_lightcurve, save_grb
    save_lightcurve(lc, "output.h5")
    save_grb(obs, "grb170817.h5")
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np

from .schemas import LightCurve, Spectrum, GRBMetadata, GRBObservation


def save_lightcurve(
    lc: LightCurve,
    filepath: Union[str, Path],
    group: str = "light_curves",
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save light curve to HDF5 file.

    Parameters
    ----------
    lc : LightCurve
        Light curve to save.
    filepath : str or Path
        Output file path.
    group : str
        HDF5 group name for light curves.
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', None).
    compression_opts : int
        Compression level (1-9 for gzip).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install grb-common[io]"
        )

    filepath = Path(filepath)

    with h5py.File(filepath, "a") as f:
        # Create or get light_curves group
        if group not in f:
            grp = f.create_group(group)
        else:
            grp = f[group]

        # Create dataset for this band
        if lc.band in grp:
            del grp[lc.band]

        band_grp = grp.create_group(lc.band)

        # Store arrays
        kwargs = {}
        if compression:
            kwargs["compression"] = compression
            kwargs["compression_opts"] = compression_opts

        band_grp.create_dataset("time", data=lc.time, **kwargs)
        band_grp.create_dataset("flux", data=lc.flux, **kwargs)
        band_grp.create_dataset("flux_err", data=lc.flux_err, **kwargs)

        if lc.flux_err_lo is not None:
            band_grp.create_dataset("flux_err_lo", data=lc.flux_err_lo, **kwargs)
        if lc.flux_err_hi is not None:
            band_grp.create_dataset("flux_err_hi", data=lc.flux_err_hi, **kwargs)
        if lc.upper_limits is not None:
            band_grp.create_dataset("upper_limits", data=lc.upper_limits, **kwargs)

        # Store metadata as attributes
        band_grp.attrs["band"] = lc.band
        band_grp.attrs["flux_type"] = lc.flux_type
        band_grp.attrs["time_frame"] = lc.time_frame

        if lc.frequency is not None:
            band_grp.attrs["frequency"] = lc.frequency
        if lc.wavelength is not None:
            band_grp.attrs["wavelength"] = lc.wavelength
        if lc.energy is not None:
            band_grp.attrs["energy"] = lc.energy

        for key, val in lc.metadata.items():
            if isinstance(val, (str, int, float, bool)):
                band_grp.attrs[key] = val


def save_spectrum(
    spec: Spectrum,
    filepath: Union[str, Path],
    group: str = "spectra",
    name: Optional[str] = None,
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save spectrum to HDF5 file.

    Parameters
    ----------
    spec : Spectrum
        Spectrum to save.
    filepath : str or Path
        Output file path.
    group : str
        HDF5 group name for spectra.
    name : str, optional
        Name for this spectrum (default: time-based).
    compression : str, optional
        Compression algorithm.
    compression_opts : int
        Compression level.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 support")

    filepath = Path(filepath)

    if name is None:
        name = f"t_{spec.time:.3e}"

    with h5py.File(filepath, "a") as f:
        if group not in f:
            grp = f.create_group(group)
        else:
            grp = f[group]

        if name in grp:
            del grp[name]

        spec_grp = grp.create_group(name)

        kwargs = {}
        if compression:
            kwargs["compression"] = compression
            kwargs["compression_opts"] = compression_opts

        spec_grp.create_dataset("frequency", data=spec.frequency, **kwargs)
        spec_grp.create_dataset("flux", data=spec.flux, **kwargs)
        spec_grp.create_dataset("flux_err", data=spec.flux_err, **kwargs)

        spec_grp.attrs["time"] = spec.time
        spec_grp.attrs["flux_unit"] = spec.flux_unit

        for key, val in spec.metadata.items():
            if isinstance(val, (str, int, float, bool)):
                spec_grp.attrs[key] = val


def save_grb(
    obs: GRBObservation,
    filepath: Union[str, Path],
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save complete GRB observation to HDF5 file.

    Creates standardized HDF5 structure:
        /metadata           - GRB metadata attributes
        /light_curves/      - Light curve datasets by band
        /spectra/           - Spectral datasets by time

    Parameters
    ----------
    obs : GRBObservation
        Complete observation to save.
    filepath : str or Path
        Output file path.
    compression : str, optional
        Compression algorithm.
    compression_opts : int
        Compression level.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 support")

    filepath = Path(filepath)

    with h5py.File(filepath, "w") as f:
        # Store metadata
        meta_grp = f.create_group("metadata")
        meta = obs.metadata

        meta_grp.attrs["name"] = meta.name
        if meta.trigger_time is not None:
            meta_grp.attrs["trigger_time"] = meta.trigger_time
        if meta.redshift is not None:
            meta_grp.attrs["redshift"] = meta.redshift
        if meta.redshift_err is not None:
            meta_grp.attrs["redshift_err"] = meta.redshift_err
        if meta.ra is not None:
            meta_grp.attrs["ra"] = meta.ra
        if meta.dec is not None:
            meta_grp.attrs["dec"] = meta.dec
        if meta.host_galaxy is not None:
            meta_grp.attrs["host_galaxy"] = meta.host_galaxy
        if meta.ebv_mw is not None:
            meta_grp.attrs["ebv_mw"] = meta.ebv_mw
        if meta.ebv_host is not None:
            meta_grp.attrs["ebv_host"] = meta.ebv_host
        if meta.source is not None:
            meta_grp.attrs["source"] = meta.source
        if meta.reference is not None:
            meta_grp.attrs["reference"] = meta.reference

    # Save light curves
    for lc in obs.light_curves:
        save_lightcurve(
            lc, filepath,
            compression=compression,
            compression_opts=compression_opts,
        )

    # Save spectra
    if obs.spectra:
        for spec in obs.spectra:
            save_spectrum(
                spec, filepath,
                compression=compression,
                compression_opts=compression_opts,
            )


__all__ = [
    "save_lightcurve",
    "save_spectrum",
    "save_grb",
]
