"""Tests for grb_common.cosmology module."""

import numpy as np
import pytest


def test_luminosity_distance_low_z():
    """Test luminosity distance at low redshift."""
    from grb_common.constants import C_LIGHT, MPC
    from grb_common.cosmology import luminosity_distance

    # At very low z, d_L ~ c*z/H0
    z = 0.01
    dL = luminosity_distance(z)

    # Should be ~40 Mpc for z=0.01 with H0~70
    dL_mpc = dL / MPC
    assert 30 < dL_mpc < 50


def test_luminosity_distance_grb170817():
    """Test luminosity distance for GRB 170817A."""
    from grb_common.constants import MPC
    from grb_common.cosmology import luminosity_distance

    z = 0.0098  # GRB 170817A redshift
    dL = luminosity_distance(z, cosmology="Planck18")

    # GRB 170817A is at ~40 Mpc
    dL_mpc = dL / MPC
    assert 35 < dL_mpc < 45


def test_angular_diameter_distance():
    """Test angular diameter distance relation."""
    from grb_common.cosmology import angular_diameter_distance, luminosity_distance

    z = 0.5
    dL = luminosity_distance(z)
    dA = angular_diameter_distance(z)

    # d_L = d_A * (1+z)^2
    assert np.isclose(dL, dA * (1 + z) ** 2, rtol=1e-10)


def test_cosmology_selection():
    """Test different cosmology selections."""
    from grb_common.cosmology import luminosity_distance

    z = 1.0
    dL_planck18 = luminosity_distance(z, cosmology="Planck18")
    dL_planck15 = luminosity_distance(z, cosmology="Planck15")
    dL_wmap9 = luminosity_distance(z, cosmology="WMAP9")

    # All should be similar but not identical
    assert dL_planck18 != dL_planck15
    assert dL_planck18 != dL_wmap9
    # But within 5% of each other
    assert np.isclose(dL_planck18, dL_planck15, rtol=0.05)


def test_custom_cosmology():
    """Test custom cosmology parameters."""
    from grb_common.cosmology import luminosity_distance

    z = 0.5
    dL = luminosity_distance(z, H0=70, Om0=0.3)
    assert dL > 0


def test_array_input():
    """Test array input for redshifts."""
    from grb_common.cosmology import luminosity_distance

    z = np.array([0.1, 0.5, 1.0, 2.0])
    dL = luminosity_distance(z)

    assert len(dL) == len(z)
    # Distance should increase with redshift
    assert np.all(np.diff(dL) > 0)


def test_invalid_cosmology():
    """Test error for invalid cosmology name."""
    from grb_common.cosmology import get_cosmology

    with pytest.raises(ValueError, match="Unknown cosmology"):
        get_cosmology("InvalidCosmo")


def test_hubble_parameter():
    """Test Hubble parameter calculation."""
    from grb_common.cosmology import hubble_parameter

    # H(z=0) should be H0 ~ 67-70 km/s/Mpc
    H0 = hubble_parameter(0)
    assert 65 < H0 < 72

    # H(z) should increase with z in LCDM
    H1 = hubble_parameter(1.0)
    assert H1 > H0
