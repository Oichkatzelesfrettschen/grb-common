"""
Data schemas for GRB observations.

This module defines standardized data structures for GRB light curves,
spectra, and metadata. All data classes are immutable dataclasses with
validation and serialization support.

Units Convention:
    - Time: seconds (observer frame unless specified)
    - Flux: erg/cm^2/s or erg/cm^2/s/Hz (spectral)
    - Frequency: Hz
    - Wavelength: Angstroms
    - Energy: keV
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import numpy as np


@dataclass(frozen=True)
class LightCurvePoint:
    """
    Single point in a light curve.

    Attributes
    ----------
    time : float
        Observation time in seconds (observer frame).
    flux : float
        Flux in erg/cm^2/s or erg/cm^2/s/Hz.
    flux_err : float
        1-sigma flux uncertainty.
    flux_err_lo : float, optional
        Lower flux uncertainty (if asymmetric).
    flux_err_hi : float, optional
        Upper flux uncertainty (if asymmetric).
    is_upper_limit : bool
        True if this is an upper limit, not a detection.
    """
    time: float
    flux: float
    flux_err: float
    flux_err_lo: Optional[float] = None
    flux_err_hi: Optional[float] = None
    is_upper_limit: bool = False


@dataclass
class LightCurve:
    """
    Multi-band light curve data.

    Attributes
    ----------
    time : ndarray
        Observation times in seconds.
    flux : ndarray
        Fluxes in erg/cm^2/s or erg/cm^2/s/Hz.
    flux_err : ndarray
        1-sigma flux uncertainties.
    flux_err_lo : ndarray, optional
        Lower uncertainties (asymmetric errors).
    flux_err_hi : ndarray, optional
        Upper uncertainties (asymmetric errors).
    upper_limits : ndarray, optional
        Boolean mask for upper limits.
    band : str
        Observation band name (e.g., 'X-ray', 'optical_r', 'radio_5GHz').
    frequency : float, optional
        Central frequency in Hz.
    wavelength : float, optional
        Central wavelength in Angstroms.
    energy : float, optional
        Central energy in keV.
    flux_type : str
        Type of flux: 'flux_density' (Jy or erg/cm^2/s/Hz) or 'flux' (erg/cm^2/s).
    time_frame : str
        Time frame: 'observer' or 'source'.
    metadata : dict
        Additional metadata.
    """
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    band: str
    flux_err_lo: Optional[np.ndarray] = None
    flux_err_hi: Optional[np.ndarray] = None
    upper_limits: Optional[np.ndarray] = None
    frequency: Optional[float] = None
    wavelength: Optional[float] = None
    energy: Optional[float] = None
    flux_type: Literal["flux_density", "flux"] = "flux_density"
    time_frame: Literal["observer", "source"] = "observer"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency."""
        n = len(self.time)
        if len(self.flux) != n:
            raise ValueError(f"flux length {len(self.flux)} != time length {n}")
        if len(self.flux_err) != n:
            raise ValueError(f"flux_err length {len(self.flux_err)} != time length {n}")

        # Convert to numpy arrays if needed
        if not isinstance(self.time, np.ndarray):
            object.__setattr__(self, 'time', np.asarray(self.time))
        if not isinstance(self.flux, np.ndarray):
            object.__setattr__(self, 'flux', np.asarray(self.flux))
        if not isinstance(self.flux_err, np.ndarray):
            object.__setattr__(self, 'flux_err', np.asarray(self.flux_err))

    def __len__(self) -> int:
        return len(self.time)

    @property
    def detections(self) -> "LightCurve":
        """Return light curve with only detections (no upper limits)."""
        if self.upper_limits is None:
            return self
        mask = ~self.upper_limits
        return LightCurve(
            time=self.time[mask],
            flux=self.flux[mask],
            flux_err=self.flux_err[mask],
            band=self.band,
            flux_err_lo=self.flux_err_lo[mask] if self.flux_err_lo is not None else None,
            flux_err_hi=self.flux_err_hi[mask] if self.flux_err_hi is not None else None,
            upper_limits=None,
            frequency=self.frequency,
            wavelength=self.wavelength,
            energy=self.energy,
            flux_type=self.flux_type,
            time_frame=self.time_frame,
            metadata=self.metadata,
        )

    def to_source_frame(self, z: float) -> "LightCurve":
        """
        Convert to source frame time.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        LightCurve
            New light curve with time_frame='source'.
        """
        if self.time_frame == "source":
            return self
        return LightCurve(
            time=self.time / (1 + z),
            flux=self.flux,
            flux_err=self.flux_err,
            band=self.band,
            flux_err_lo=self.flux_err_lo,
            flux_err_hi=self.flux_err_hi,
            upper_limits=self.upper_limits,
            frequency=self.frequency,
            wavelength=self.wavelength,
            energy=self.energy,
            flux_type=self.flux_type,
            time_frame="source",
            metadata={**self.metadata, "redshift": z},
        )


@dataclass
class Spectrum:
    """
    Spectral energy distribution at a single epoch.

    Attributes
    ----------
    frequency : ndarray
        Frequencies in Hz.
    flux : ndarray
        Flux densities in erg/cm^2/s/Hz or Jy.
    flux_err : ndarray
        1-sigma flux uncertainties.
    time : float
        Observation time in seconds.
    flux_unit : str
        Flux unit: 'cgs' (erg/cm^2/s/Hz) or 'Jy'.
    metadata : dict
        Additional metadata.
    """
    frequency: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    time: float
    flux_unit: Literal["cgs", "Jy"] = "cgs"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency."""
        n = len(self.frequency)
        if len(self.flux) != n:
            raise ValueError(f"flux length {len(self.flux)} != frequency length {n}")
        if len(self.flux_err) != n:
            raise ValueError(f"flux_err length {len(self.flux_err)} != frequency length {n}")

    def __len__(self) -> int:
        return len(self.frequency)


@dataclass
class GRBMetadata:
    """
    Metadata for a GRB observation.

    Attributes
    ----------
    name : str
        GRB name (e.g., 'GRB 170817A', 'GRB 130427A').
    trigger_time : float, optional
        Trigger time (MJD or Unix timestamp).
    redshift : float, optional
        Spectroscopic redshift.
    redshift_err : float, optional
        Redshift uncertainty.
    ra : float, optional
        Right ascension in degrees.
    dec : float, optional
        Declination in degrees.
    host_galaxy : str, optional
        Host galaxy name.
    ebv_mw : float, optional
        Milky Way E(B-V) extinction.
    ebv_host : float, optional
        Host galaxy E(B-V) extinction.
    source : str, optional
        Data source (e.g., 'Swift-XRT', 'VLA', 'HST').
    reference : str, optional
        Literature reference (ADS bibcode).
    """
    name: str
    trigger_time: Optional[float] = None
    redshift: Optional[float] = None
    redshift_err: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    host_galaxy: Optional[str] = None
    ebv_mw: Optional[float] = None
    ebv_host: Optional[float] = None
    source: Optional[str] = None
    reference: Optional[str] = None


@dataclass
class GRBObservation:
    """
    Complete GRB observation dataset.

    Attributes
    ----------
    metadata : GRBMetadata
        GRB metadata.
    light_curves : list of LightCurve
        Multi-band light curves.
    spectra : list of Spectrum, optional
        Spectral energy distributions.
    """
    metadata: GRBMetadata
    light_curves: List[LightCurve]
    spectra: Optional[List[Spectrum]] = None

    def get_band(self, band: str) -> Optional[LightCurve]:
        """Get light curve for a specific band."""
        for lc in self.light_curves:
            if lc.band == band:
                return lc
        return None

    @property
    def bands(self) -> List[str]:
        """List of available bands."""
        return [lc.band for lc in self.light_curves]


__all__ = [
    "LightCurvePoint",
    "LightCurve",
    "Spectrum",
    "GRBMetadata",
    "GRBObservation",
]
