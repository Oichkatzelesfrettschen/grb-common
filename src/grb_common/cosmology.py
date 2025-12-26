"""
Cosmological calculations for GRB astrophysics.

This module provides distance and time calculations using standard
cosmological parameters from astropy. Supports multiple cosmologies
and custom parameter sets.

Key functions:
    luminosity_distance: d_L(z) in cm
    angular_diameter_distance: d_A(z) in cm
    comoving_distance: d_C(z) in cm
    lookback_time: t_lb(z) in seconds
    age_at_redshift: t(z) in seconds
    hubble_parameter: H(z) in km/s/Mpc

Supported cosmologies:
    - Planck18 (default)
    - Planck15
    - WMAP9
    - Custom (provide H0, Om0, Ode0)

Usage:
    from grb_common.cosmology import luminosity_distance

    z = 0.0098  # GRB 170817A
    dL = luminosity_distance(z)  # Returns distance in cm
"""

from typing import Optional, Union

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18, Planck15, WMAP9, FlatLambdaCDM

# Type alias for array-like inputs
ArrayLike = Union[float, np.ndarray]

# Available cosmologies
COSMOLOGIES = {
    "Planck18": Planck18,
    "Planck15": Planck15,
    "WMAP9": WMAP9,
}

# Default cosmology
DEFAULT_COSMOLOGY = "Planck18"


def get_cosmology(
    cosmology: Optional[str] = None,
    H0: Optional[float] = None,
    Om0: Optional[float] = None,
    Ode0: Optional[float] = None,
):
    """
    Get cosmology object.

    Parameters
    ----------
    cosmology : str, optional
        Name of predefined cosmology ('Planck18', 'Planck15', 'WMAP9').
        If None and custom params not provided, uses DEFAULT_COSMOLOGY.
    H0 : float, optional
        Hubble constant in km/s/Mpc for custom cosmology.
    Om0 : float, optional
        Matter density parameter for custom cosmology.
    Ode0 : float, optional
        Dark energy density parameter for custom cosmology.

    Returns
    -------
    astropy.cosmology.Cosmology
        Cosmology object.
    """
    if H0 is not None and Om0 is not None:
        # Custom cosmology
        if Ode0 is None:
            Ode0 = 1 - Om0  # Flat universe
        return FlatLambdaCDM(H0=H0, Om0=Om0)

    name = cosmology or DEFAULT_COSMOLOGY
    if name not in COSMOLOGIES:
        raise ValueError(f"Unknown cosmology: {name}. Choose from {list(COSMOLOGIES.keys())}")
    return COSMOLOGIES[name]


def luminosity_distance(
    z: ArrayLike,
    cosmology: Optional[str] = None,
    **kwargs,
) -> ArrayLike:
    """
    Calculate luminosity distance in cm.

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    cosmology : str, optional
        Cosmology name ('Planck18', 'Planck15', 'WMAP9').
    **kwargs
        Custom cosmology parameters (H0, Om0, Ode0).

    Returns
    -------
    float or ndarray
        Luminosity distance in cm.
    """
    cosmo = get_cosmology(cosmology, **kwargs)
    return cosmo.luminosity_distance(z).to(u.cm).value


def angular_diameter_distance(
    z: ArrayLike,
    cosmology: Optional[str] = None,
    **kwargs,
) -> ArrayLike:
    """
    Calculate angular diameter distance in cm.

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    cosmology : str, optional
        Cosmology name.
    **kwargs
        Custom cosmology parameters.

    Returns
    -------
    float or ndarray
        Angular diameter distance in cm.
    """
    cosmo = get_cosmology(cosmology, **kwargs)
    return cosmo.angular_diameter_distance(z).to(u.cm).value


def comoving_distance(
    z: ArrayLike,
    cosmology: Optional[str] = None,
    **kwargs,
) -> ArrayLike:
    """
    Calculate comoving distance in cm.

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    cosmology : str, optional
        Cosmology name.
    **kwargs
        Custom cosmology parameters.

    Returns
    -------
    float or ndarray
        Comoving distance in cm.
    """
    cosmo = get_cosmology(cosmology, **kwargs)
    return cosmo.comoving_distance(z).to(u.cm).value


def lookback_time(
    z: ArrayLike,
    cosmology: Optional[str] = None,
    **kwargs,
) -> ArrayLike:
    """
    Calculate lookback time in seconds.

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    cosmology : str, optional
        Cosmology name.
    **kwargs
        Custom cosmology parameters.

    Returns
    -------
    float or ndarray
        Lookback time in seconds.
    """
    cosmo = get_cosmology(cosmology, **kwargs)
    return cosmo.lookback_time(z).to(u.s).value


def age_at_redshift(
    z: ArrayLike,
    cosmology: Optional[str] = None,
    **kwargs,
) -> ArrayLike:
    """
    Calculate age of universe at redshift z in seconds.

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    cosmology : str, optional
        Cosmology name.
    **kwargs
        Custom cosmology parameters.

    Returns
    -------
    float or ndarray
        Age of universe in seconds.
    """
    cosmo = get_cosmology(cosmology, **kwargs)
    return cosmo.age(z).to(u.s).value


def hubble_parameter(
    z: ArrayLike,
    cosmology: Optional[str] = None,
    **kwargs,
) -> ArrayLike:
    """
    Calculate Hubble parameter H(z) in km/s/Mpc.

    Parameters
    ----------
    z : float or array-like
        Redshift(s).
    cosmology : str, optional
        Cosmology name.
    **kwargs
        Custom cosmology parameters.

    Returns
    -------
    float or ndarray
        Hubble parameter in km/s/Mpc.
    """
    cosmo = get_cosmology(cosmology, **kwargs)
    return cosmo.H(z).value


__all__ = [
    "get_cosmology",
    "luminosity_distance",
    "angular_diameter_distance",
    "comoving_distance",
    "lookback_time",
    "age_at_redshift",
    "hubble_parameter",
    "COSMOLOGIES",
    "DEFAULT_COSMOLOGY",
]
