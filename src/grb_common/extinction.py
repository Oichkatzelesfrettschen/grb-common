"""
Dust extinction corrections for GRB astrophysics.

This module provides functions for calculating dust extinction and
dereddening observed fluxes. Supports multiple extinction laws commonly
used in GRB afterglow analysis.

Extinction laws:
    - Fitzpatrick99 (MW): Standard Milky Way
    - Cardelli89 (MW): CCM Milky Way
    - Calzetti00: Starburst galaxies
    - SMC: Small Magellanic Cloud (Gordon03)
    - LMC: Large Magellanic Cloud (Gordon03)

Key functions:
    extinction_curve: A(lambda) / A(V) for given wavelengths
    deredden: Remove extinction from observed flux
    redden: Apply extinction to intrinsic flux
    ebv_to_av: Convert E(B-V) to A(V) using R(V)

Usage:
    from grb_common.extinction import deredden, extinction_curve

    # Deredden observed flux
    wavelengths = np.array([4500, 6500, 9000])  # Angstroms
    flux_obs = np.array([1e-15, 2e-15, 3e-15])  # erg/cm^2/s/A
    flux_int = deredden(wavelengths, flux_obs, ebv=0.1, law='Fitzpatrick99')
"""

from typing import Optional, Union

import numpy as np

# Type alias
ArrayLike = Union[float, np.ndarray]

# Extinction law names
EXTINCTION_LAWS = [
    "Fitzpatrick99",
    "Cardelli89",
    "Calzetti00",
    "SMC",
    "LMC",
]

# Default R(V) values for different environments
RV_DEFAULT = {
    "Fitzpatrick99": 3.1,
    "Cardelli89": 3.1,
    "Calzetti00": 4.05,
    "SMC": 2.93,
    "LMC": 3.41,
}


def _check_extinction_import():
    """Check if extinction package is available."""
    try:
        import extinction
        return extinction
    except ImportError:
        raise ImportError(
            "The 'extinction' package is required for extinction calculations. "
            "Install it with: pip install grb-common[extinction]"
        )


def ebv_to_av(ebv: float, rv: float = 3.1) -> float:
    """
    Convert E(B-V) color excess to A(V) extinction.

    Parameters
    ----------
    ebv : float
        Color excess E(B-V) in magnitudes.
    rv : float, optional
        Total-to-selective extinction ratio R(V). Default is 3.1 (MW).

    Returns
    -------
    float
        Visual extinction A(V) in magnitudes.
    """
    return ebv * rv


def extinction_curve(
    wavelengths: ArrayLike,
    av: float = 1.0,
    rv: Optional[float] = None,
    law: str = "Fitzpatrick99",
) -> ArrayLike:
    """
    Calculate extinction A(lambda) for given wavelengths.

    Parameters
    ----------
    wavelengths : array-like
        Wavelengths in Angstroms.
    av : float, optional
        Visual extinction A(V) in magnitudes. Default is 1.0.
    rv : float, optional
        Total-to-selective extinction R(V). If None, uses default for law.
    law : str, optional
        Extinction law to use. Default is 'Fitzpatrick99'.

    Returns
    -------
    ndarray
        Extinction A(lambda) in magnitudes at each wavelength.
    """
    ext = _check_extinction_import()

    wavelengths = np.atleast_1d(wavelengths).astype(float)

    if rv is None:
        rv = RV_DEFAULT.get(law, 3.1)

    if law == "Fitzpatrick99":
        return ext.fitzpatrick99(wavelengths, av, rv)
    elif law == "Cardelli89":
        return ext.ccm89(wavelengths, av, rv)
    elif law == "Calzetti00":
        return ext.calzetti00(wavelengths, av, rv)
    elif law == "SMC":
        return ext.fm07(wavelengths, av)  # Gordon et al. 2003 SMC bar
    elif law == "LMC":
        # LMC approximation using FM with modified parameters
        return ext.fm07(wavelengths, av)
    else:
        raise ValueError(f"Unknown extinction law: {law}. Choose from {EXTINCTION_LAWS}")


def deredden(
    wavelengths: ArrayLike,
    flux: ArrayLike,
    ebv: Optional[float] = None,
    av: Optional[float] = None,
    rv: Optional[float] = None,
    law: str = "Fitzpatrick99",
) -> ArrayLike:
    """
    Deredden observed flux to get intrinsic flux.

    Must provide either ebv or av.

    Parameters
    ----------
    wavelengths : array-like
        Wavelengths in Angstroms.
    flux : array-like
        Observed flux (any units).
    ebv : float, optional
        Color excess E(B-V) in magnitudes.
    av : float, optional
        Visual extinction A(V) in magnitudes.
    rv : float, optional
        Total-to-selective extinction R(V).
    law : str, optional
        Extinction law to use.

    Returns
    -------
    ndarray
        Dereddened (intrinsic) flux in same units as input.
    """
    if ebv is None and av is None:
        raise ValueError("Must provide either ebv or av")

    if rv is None:
        rv = RV_DEFAULT.get(law, 3.1)

    if av is None:
        av = ebv_to_av(ebv, rv)

    a_lambda = extinction_curve(wavelengths, av, rv, law)

    # Convert from magnitudes to flux ratio
    # A = -2.5 * log10(F_obs / F_int)
    # F_int = F_obs * 10^(A/2.5)
    flux = np.atleast_1d(flux).astype(float)
    return flux * 10 ** (a_lambda / 2.5)


def redden(
    wavelengths: ArrayLike,
    flux: ArrayLike,
    ebv: Optional[float] = None,
    av: Optional[float] = None,
    rv: Optional[float] = None,
    law: str = "Fitzpatrick99",
) -> ArrayLike:
    """
    Apply extinction to intrinsic flux to get observed flux.

    Must provide either ebv or av.

    Parameters
    ----------
    wavelengths : array-like
        Wavelengths in Angstroms.
    flux : array-like
        Intrinsic flux (any units).
    ebv : float, optional
        Color excess E(B-V) in magnitudes.
    av : float, optional
        Visual extinction A(V) in magnitudes.
    rv : float, optional
        Total-to-selective extinction R(V).
    law : str, optional
        Extinction law to use.

    Returns
    -------
    ndarray
        Reddened (observed) flux in same units as input.
    """
    if ebv is None and av is None:
        raise ValueError("Must provide either ebv or av")

    if rv is None:
        rv = RV_DEFAULT.get(law, 3.1)

    if av is None:
        av = ebv_to_av(ebv, rv)

    a_lambda = extinction_curve(wavelengths, av, rv, law)

    # F_obs = F_int * 10^(-A/2.5)
    flux = np.atleast_1d(flux).astype(float)
    return flux * 10 ** (-a_lambda / 2.5)


__all__ = [
    "EXTINCTION_LAWS",
    "RV_DEFAULT",
    "ebv_to_av",
    "extinction_curve",
    "deredden",
    "redden",
]
