"""
Data loaders for various GRB file formats.

Supports automatic format detection and parsing for:
- ASGARD text format
- boxfit output format
- FITS tables (Swift-XRT, BATSE, Fermi-GBM)
- CSV files (generic)
- HDF5 (grb-common schema)

Usage:
    from grb_common.io import load_lightcurve
    lc = load_lightcurve("grb170817_xray.txt")
"""

from pathlib import Path
from typing import Optional, Literal, Union, List
import re
import numpy as np

from .schemas import LightCurve, Spectrum, GRBMetadata, GRBObservation


FileFormat = Literal["asgard", "boxfit", "fits", "csv", "hdf5", "auto"]


def detect_format(filepath: Union[str, Path]) -> str:
    """
    Detect file format from extension and content.

    Parameters
    ----------
    filepath : str or Path
        Path to data file.

    Returns
    -------
    str
        Detected format: 'asgard', 'boxfit', 'fits', 'csv', or 'hdf5'.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    # Extension-based detection
    if suffix in (".h5", ".hdf5"):
        return "hdf5"
    if suffix in (".fits", ".fit"):
        return "fits"
    if suffix in (".csv",):
        return "csv"

    # For text files, peek at content
    if suffix in (".txt", ".dat", ""):
        try:
            with open(filepath, "r") as f:
                first_lines = [f.readline() for _ in range(10)]
            content = "".join(first_lines)

            # ASGARD format markers
            if "ASGARD" in content or "afterglow" in content.lower():
                return "asgard"

            # boxfit format markers
            if "boxfit" in content.lower() or "time[s]" in content.lower():
                return "boxfit"

            # Default to CSV-like for text files
            return "csv"
        except Exception:
            pass

    return "csv"  # Default fallback


def load_lightcurve(
    filepath: Union[str, Path],
    format: FileFormat = "auto",
    band: Optional[str] = None,
    **kwargs,
) -> LightCurve:
    """
    Load light curve from file.

    Parameters
    ----------
    filepath : str or Path
        Path to data file.
    format : str
        File format: 'asgard', 'boxfit', 'fits', 'csv', 'hdf5', or 'auto'.
    band : str, optional
        Band name to assign (overrides auto-detection).
    **kwargs
        Additional arguments passed to format-specific parser.

    Returns
    -------
    LightCurve
        Loaded light curve data.
    """
    filepath = Path(filepath)

    if format == "auto":
        format = detect_format(filepath)

    if format == "asgard":
        return _parse_asgard_txt(filepath, band=band, **kwargs)
    elif format == "boxfit":
        return _parse_boxfit_txt(filepath, band=band, **kwargs)
    elif format == "fits":
        return _parse_fits_lightcurve(filepath, band=band, **kwargs)
    elif format == "csv":
        return _parse_csv_lightcurve(filepath, band=band, **kwargs)
    elif format == "hdf5":
        return _parse_hdf5_lightcurve(filepath, band=band, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_spectrum(
    filepath: Union[str, Path],
    format: FileFormat = "auto",
    **kwargs,
) -> Spectrum:
    """
    Load spectrum from file.

    Parameters
    ----------
    filepath : str or Path
        Path to data file.
    format : str
        File format.
    **kwargs
        Additional arguments.

    Returns
    -------
    Spectrum
        Loaded spectral data.
    """
    filepath = Path(filepath)

    if format == "auto":
        format = detect_format(filepath)

    if format == "hdf5":
        return _parse_hdf5_spectrum(filepath, **kwargs)
    elif format == "csv":
        return _parse_csv_spectrum(filepath, **kwargs)
    else:
        raise ValueError(f"Spectrum loading not supported for format: {format}")


def load_grb(filepath: Union[str, Path]) -> GRBObservation:
    """
    Load complete GRB observation from HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file.

    Returns
    -------
    GRBObservation
        Complete observation dataset.
    """
    filepath = Path(filepath)
    return _parse_hdf5_grb(filepath)


# -----------------------------------------------------------------------------
# Format-specific parsers
# -----------------------------------------------------------------------------


def _parse_asgard_txt(
    filepath: Path,
    band: Optional[str] = None,
    time_col: int = 0,
    flux_col: int = 1,
    err_col: int = 2,
    skip_header: int = 0,
    comment: str = "#",
) -> LightCurve:
    """
    Parse ASGARD text file format.

    Expected format:
        # Time[s]  Flux[erg/cm^2/s/Hz]  Error
        1.0e3     1.0e-26              1.0e-27
        ...
    """
    data = np.loadtxt(
        filepath,
        skiprows=skip_header,
        comments=comment,
        unpack=True,
    )

    # Handle different column counts
    if data.ndim == 1:
        # Single column - assume time only
        raise ValueError("ASGARD file needs at least 2 columns (time, flux)")

    ncols = data.shape[0]

    time = data[time_col]
    flux = data[flux_col]

    if ncols > 2 and err_col < ncols:
        flux_err = data[err_col]
    else:
        # Estimate error as 10% of flux
        flux_err = np.abs(flux) * 0.1

    # Try to extract band from filename
    if band is None:
        fname = filepath.stem.lower()
        if "xray" in fname or "x-ray" in fname:
            band = "X-ray"
        elif "radio" in fname:
            band = "radio"
        elif "optical" in fname:
            band = "optical"
        else:
            band = "unknown"

    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        band=band,
        metadata={"source": "ASGARD", "filepath": str(filepath)},
    )


def _parse_boxfit_txt(
    filepath: Path,
    band: Optional[str] = None,
    time_col: int = 0,
    flux_col: int = 1,
    err_col: Optional[int] = None,
    skip_header: int = 1,
    comment: str = "#",
) -> LightCurve:
    """
    Parse boxfit output text file format.

    Expected format:
        # time[s] flux[mJy] ...
        1.0e3    1.0
        ...
    """
    data = np.loadtxt(
        filepath,
        skiprows=skip_header,
        comments=comment,
        unpack=True,
    )

    time = data[time_col]
    flux_mjy = data[flux_col]

    # Convert mJy to erg/cm^2/s/Hz (1 mJy = 1e-26 erg/cm^2/s/Hz)
    flux = flux_mjy * 1e-26

    if err_col is not None and err_col < data.shape[0]:
        flux_err = data[err_col] * 1e-26
    else:
        # boxfit output typically doesn't have errors
        flux_err = np.zeros_like(flux)

    if band is None:
        band = "model"

    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        band=band,
        metadata={"source": "boxfit", "filepath": str(filepath)},
    )


def _parse_fits_lightcurve(
    filepath: Path,
    band: Optional[str] = None,
    hdu: int = 1,
    time_col: str = "TIME",
    flux_col: str = "RATE",
    err_col: str = "ERROR",
) -> LightCurve:
    """
    Parse FITS light curve file.

    Supports Swift-XRT, BATSE, and Fermi-GBM formats.
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError(
            "astropy is required for FITS support. "
            "Install with: pip install astropy"
        )

    with fits.open(filepath) as hdul:
        table = hdul[hdu].data
        header = hdul[hdu].header

        # Try to find columns
        col_names = [c.name.upper() for c in table.columns]

        if time_col.upper() in col_names:
            time = table[time_col]
        elif "TIME" in col_names:
            time = table["TIME"]
        elif "MJD" in col_names:
            time = (table["MJD"] - table["MJD"][0]) * 86400  # Convert to seconds
        else:
            raise ValueError(f"Cannot find time column in {col_names}")

        if flux_col.upper() in col_names:
            flux = table[flux_col]
        elif "RATE" in col_names:
            flux = table["RATE"]
        elif "FLUX" in col_names:
            flux = table["FLUX"]
        else:
            raise ValueError(f"Cannot find flux column in {col_names}")

        if err_col.upper() in col_names:
            flux_err = table[err_col]
        elif "ERROR" in col_names:
            flux_err = table["ERROR"]
        elif "STAT_ERR" in col_names:
            flux_err = table["STAT_ERR"]
        else:
            flux_err = np.abs(flux) * 0.1

        # Auto-detect band from header
        if band is None:
            if "TELESCOP" in header:
                telescop = header["TELESCOP"]
                if "Swift" in telescop:
                    band = "Swift-XRT"
                elif "Fermi" in telescop:
                    band = "Fermi-GBM"
                else:
                    band = telescop
            else:
                band = "FITS"

    return LightCurve(
        time=np.asarray(time),
        flux=np.asarray(flux),
        flux_err=np.asarray(flux_err),
        band=band,
        metadata={"source": "FITS", "filepath": str(filepath)},
    )


def _parse_csv_lightcurve(
    filepath: Path,
    band: Optional[str] = None,
    delimiter: str = ",",
    time_col: Union[int, str] = 0,
    flux_col: Union[int, str] = 1,
    err_col: Union[int, str, None] = 2,
) -> LightCurve:
    """
    Parse CSV light curve file.

    Auto-detects column names if header present.
    """
    # Try to detect if header exists
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

    has_header = not first_line[0].isdigit() and not first_line[0] == "-"

    if has_header:
        try:
            import pandas as pd
            df = pd.read_csv(filepath, delimiter=delimiter)

            # Find time column
            time_candidates = ["time", "t", "mjd", "jd", "time_s", "time[s]"]
            time_col_name = None
            for cand in time_candidates:
                matches = [c for c in df.columns if cand in c.lower()]
                if matches:
                    time_col_name = matches[0]
                    break
            if time_col_name is None:
                time_col_name = df.columns[0]

            # Find flux column
            flux_candidates = ["flux", "rate", "f_nu", "flux_density"]
            flux_col_name = None
            for cand in flux_candidates:
                matches = [c for c in df.columns if cand in c.lower()]
                if matches:
                    flux_col_name = matches[0]
                    break
            if flux_col_name is None:
                flux_col_name = df.columns[1]

            # Find error column
            err_col_name = None
            err_candidates = ["error", "err", "sigma", "uncertainty"]
            for cand in err_candidates:
                matches = [c for c in df.columns if cand in c.lower()]
                if matches:
                    err_col_name = matches[0]
                    break

            time = df[time_col_name].values
            flux = df[flux_col_name].values
            if err_col_name:
                flux_err = df[err_col_name].values
            else:
                flux_err = np.abs(flux) * 0.1

        except ImportError:
            # Fallback without pandas
            data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1, unpack=True)
            time = data[0]
            flux = data[1]
            flux_err = data[2] if data.shape[0] > 2 else np.abs(flux) * 0.1
    else:
        data = np.loadtxt(filepath, delimiter=delimiter, unpack=True)
        time = data[time_col] if isinstance(time_col, int) else data[0]
        flux = data[flux_col] if isinstance(flux_col, int) else data[1]
        if err_col is not None:
            flux_err = data[err_col] if isinstance(err_col, int) else data[2]
        else:
            flux_err = np.abs(flux) * 0.1

    if band is None:
        band = filepath.stem

    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        band=band,
        metadata={"source": "CSV", "filepath": str(filepath)},
    )


def _parse_hdf5_lightcurve(
    filepath: Path,
    band: Optional[str] = None,
    group: str = "light_curves",
) -> LightCurve:
    """Parse HDF5 light curve in grb-common schema."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install grb-common[io]"
        )

    with h5py.File(filepath, "r") as f:
        if group not in f:
            raise ValueError(f"Group '{group}' not found in HDF5 file")

        grp = f[group]

        # If band specified, look for that dataset
        if band:
            if band not in grp:
                raise ValueError(f"Band '{band}' not found in {list(grp.keys())}")
            dset = grp[band]
        else:
            # Take first band
            band = list(grp.keys())[0]
            dset = grp[band]

        time = dset["time"][:]
        flux = dset["flux"][:]
        flux_err = dset["flux_err"][:]

        metadata = dict(dset.attrs)

    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        band=band,
        metadata=metadata,
    )


def _parse_hdf5_spectrum(filepath: Path, **kwargs) -> Spectrum:
    """Parse HDF5 spectrum."""
    raise NotImplementedError("HDF5 spectrum parsing not yet implemented")


def _parse_csv_spectrum(filepath: Path, **kwargs) -> Spectrum:
    """Parse CSV spectrum."""
    raise NotImplementedError("CSV spectrum parsing not yet implemented")


def _parse_hdf5_grb(filepath: Path) -> GRBObservation:
    """Parse complete GRB observation from HDF5."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 support")

    with h5py.File(filepath, "r") as f:
        # Load metadata
        meta_grp = f.get("metadata", {})
        metadata = GRBMetadata(
            name=meta_grp.attrs.get("name", "Unknown"),
            redshift=meta_grp.attrs.get("redshift"),
            ra=meta_grp.attrs.get("ra"),
            dec=meta_grp.attrs.get("dec"),
        )

        # Load light curves
        light_curves = []
        if "light_curves" in f:
            for band in f["light_curves"]:
                lc = _parse_hdf5_lightcurve(filepath, band=band)
                light_curves.append(lc)

    return GRBObservation(
        metadata=metadata,
        light_curves=light_curves,
    )


__all__ = [
    "detect_format",
    "load_lightcurve",
    "load_spectrum",
    "load_grb",
]
