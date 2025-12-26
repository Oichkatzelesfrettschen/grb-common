"""Basic package tests for grb-common."""

import pytest


def test_import_package():
    """Test that the package can be imported."""
    import grb_common
    assert hasattr(grb_common, "__version__")


def test_version():
    """Test version string format."""
    from grb_common import __version__
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_lazy_import_constants():
    """Test lazy import of constants module."""
    from grb_common import constants
    # Module should be accessible but may not have implementation yet
    assert constants is not None


def test_lazy_import_cosmology():
    """Test lazy import of cosmology module."""
    from grb_common import cosmology
    assert cosmology is not None


def test_lazy_import_extinction():
    """Test lazy import of extinction module."""
    from grb_common import extinction
    assert extinction is not None


def test_lazy_import_io():
    """Test lazy import of io module."""
    from grb_common import io
    assert io is not None


def test_lazy_import_fitting():
    """Test lazy import of fitting module."""
    from grb_common import fitting
    assert fitting is not None


def test_lazy_import_plotting():
    """Test lazy import of plotting module."""
    from grb_common import plotting
    assert plotting is not None


def test_all_exports():
    """Test that __all__ contains expected exports."""
    import grb_common
    expected = [
        "__version__",
        "constants",
        "cosmology",
        "extinction",
        "io",
        "fitting",
        "plotting",
    ]
    for name in expected:
        assert name in grb_common.__all__
