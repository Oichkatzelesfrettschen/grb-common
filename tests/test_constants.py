"""Tests for grb_common.constants module."""

import pytest
import numpy as np


def test_speed_of_light():
    """Test speed of light value."""
    from grb_common.constants import C_LIGHT
    # Should be approximately 3e10 cm/s
    assert 2.99e10 < C_LIGHT < 3.01e10


def test_electron_mass():
    """Test electron mass value."""
    from grb_common.constants import M_ELECTRON
    # Should be approximately 9.1e-28 g
    assert 9.0e-28 < M_ELECTRON < 9.2e-28


def test_thomson_cross_section():
    """Test Thomson cross-section value."""
    from grb_common.constants import SIGMA_T
    # Should be approximately 6.65e-25 cm^2
    assert 6.6e-25 < SIGMA_T < 6.7e-25


def test_solar_mass():
    """Test solar mass value."""
    from grb_common.constants import M_SUN
    # Should be approximately 2e33 g
    assert 1.98e33 < M_SUN < 2.0e33


def test_parsec():
    """Test parsec value."""
    from grb_common.constants import PC, MPC
    # 1 pc ~ 3.086e18 cm
    assert 3.08e18 < PC < 3.09e18
    # 1 Mpc = 1e6 pc
    assert np.isclose(MPC, PC * 1e6, rtol=1e-10)


def test_derived_quantities():
    """Test derived quantities are consistent."""
    from grb_common.constants import M_ELECTRON, C_LIGHT, M_E_C2
    # m_e * c^2 should equal M_E_C2
    calculated = M_ELECTRON * C_LIGHT**2
    assert np.isclose(calculated, M_E_C2, rtol=1e-10)


def test_all_exports():
    """Test that __all__ contains expected constants."""
    from grb_common import constants
    expected = ["C_LIGHT", "M_ELECTRON", "SIGMA_T", "M_SUN", "PC"]
    for name in expected:
        assert name in constants.__all__
        assert hasattr(constants, name)
