"""Tests for grb_common.io module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestLightCurve:
    """Tests for LightCurve schema."""

    def test_create_lightcurve(self):
        """Test basic LightCurve creation."""
        from grb_common.io import LightCurve

        time = np.array([1.0, 2.0, 3.0])
        flux = np.array([1e-26, 2e-26, 1.5e-26])
        flux_err = np.array([1e-27, 2e-27, 1.5e-27])

        lc = LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            band="X-ray",
        )

        assert len(lc) == 3
        assert lc.band == "X-ray"
        np.testing.assert_array_equal(lc.time, time)

    def test_lightcurve_validation(self):
        """Test LightCurve validates array lengths."""
        from grb_common.io import LightCurve

        with pytest.raises(ValueError, match="flux length"):
            LightCurve(
                time=np.array([1.0, 2.0]),
                flux=np.array([1e-26]),  # Wrong length
                flux_err=np.array([1e-27, 2e-27]),
                band="test",
            )

    def test_lightcurve_detections(self):
        """Test filtering to detections only."""
        from grb_common.io import LightCurve

        lc = LightCurve(
            time=np.array([1.0, 2.0, 3.0]),
            flux=np.array([1e-26, 2e-26, 1.5e-26]),
            flux_err=np.array([1e-27, 2e-27, 1.5e-27]),
            band="X-ray",
            upper_limits=np.array([False, True, False]),
        )

        detections = lc.detections
        assert len(detections) == 2
        np.testing.assert_array_equal(detections.time, [1.0, 3.0])

    def test_lightcurve_to_source_frame(self):
        """Test conversion to source frame."""
        from grb_common.io import LightCurve

        z = 1.0
        lc = LightCurve(
            time=np.array([100.0, 200.0]),
            flux=np.array([1e-26, 2e-26]),
            flux_err=np.array([1e-27, 2e-27]),
            band="X-ray",
        )

        lc_source = lc.to_source_frame(z)

        assert lc_source.time_frame == "source"
        np.testing.assert_array_equal(lc_source.time, lc.time / (1 + z))


class TestGRBObservation:
    """Tests for GRBObservation schema."""

    def test_create_observation(self):
        """Test creating complete observation."""
        from grb_common.io import GRBMetadata, GRBObservation, LightCurve

        meta = GRBMetadata(name="GRB 170817A", redshift=0.0098)

        lc_xray = LightCurve(
            time=np.array([1.0, 2.0]),
            flux=np.array([1e-26, 2e-26]),
            flux_err=np.array([1e-27, 2e-27]),
            band="X-ray",
        )

        lc_radio = LightCurve(
            time=np.array([10.0, 20.0]),
            flux=np.array([1e-25, 2e-25]),
            flux_err=np.array([1e-26, 2e-26]),
            band="radio_5GHz",
        )

        obs = GRBObservation(
            metadata=meta,
            light_curves=[lc_xray, lc_radio],
        )

        assert obs.metadata.name == "GRB 170817A"
        assert len(obs.light_curves) == 2
        assert obs.bands == ["X-ray", "radio_5GHz"]

    def test_get_band(self):
        """Test retrieving specific band."""
        from grb_common.io import GRBMetadata, GRBObservation, LightCurve

        meta = GRBMetadata(name="Test")
        lc = LightCurve(
            time=np.array([1.0]),
            flux=np.array([1e-26]),
            flux_err=np.array([1e-27]),
            band="X-ray",
        )

        obs = GRBObservation(metadata=meta, light_curves=[lc])

        assert obs.get_band("X-ray") is not None
        assert obs.get_band("optical") is None


class TestFormatDetection:
    """Tests for format detection."""

    def test_detect_hdf5(self):
        """Test HDF5 format detection."""
        from grb_common.io import detect_format

        assert detect_format("data.h5") == "hdf5"
        assert detect_format("data.hdf5") == "hdf5"

    def test_detect_fits(self):
        """Test FITS format detection."""
        from grb_common.io import detect_format

        assert detect_format("data.fits") == "fits"
        assert detect_format("data.fit") == "fits"

    def test_detect_csv(self):
        """Test CSV format detection."""
        from grb_common.io import detect_format

        assert detect_format("data.csv") == "csv"


class TestHDF5RoundTrip:
    """Tests for HDF5 write/read round-trip."""

    def test_lightcurve_roundtrip(self):
        """Test saving and loading light curve."""
        pytest.importorskip("h5py")
        from grb_common.io import LightCurve, load_lightcurve, save_lightcurve

        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        flux = np.array([1e-26, 2e-26, 1.5e-26, 1.2e-26, 0.8e-26])
        flux_err = np.array([1e-27, 2e-27, 1.5e-27, 1.2e-27, 0.8e-27])

        lc_orig = LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            band="X-ray",
            frequency=1e18,
        )

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_lightcurve(lc_orig, filepath)
            lc_loaded = load_lightcurve(filepath, band="X-ray")

            np.testing.assert_array_almost_equal(lc_loaded.time, lc_orig.time)
            np.testing.assert_array_almost_equal(lc_loaded.flux, lc_orig.flux)
            np.testing.assert_array_almost_equal(lc_loaded.flux_err, lc_orig.flux_err)
            assert lc_loaded.band == lc_orig.band
        finally:
            filepath.unlink()

    def test_grb_roundtrip(self):
        """Test saving and loading complete GRB observation."""
        pytest.importorskip("h5py")
        from grb_common.io import GRBMetadata, GRBObservation, LightCurve, load_grb, save_grb

        meta = GRBMetadata(
            name="GRB 170817A",
            redshift=0.0098,
            ra=197.45,
            dec=-23.38,
        )

        lc = LightCurve(
            time=np.array([1.0, 2.0, 3.0]),
            flux=np.array([1e-26, 2e-26, 1.5e-26]),
            flux_err=np.array([1e-27, 2e-27, 1.5e-27]),
            band="X-ray",
        )

        obs_orig = GRBObservation(metadata=meta, light_curves=[lc])

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_grb(obs_orig, filepath)
            obs_loaded = load_grb(filepath)

            assert obs_loaded.metadata.name == "GRB 170817A"
            assert obs_loaded.metadata.redshift == 0.0098
            assert len(obs_loaded.light_curves) == 1
            assert obs_loaded.light_curves[0].band == "X-ray"
        finally:
            filepath.unlink()


class TestCSVParser:
    """Tests for CSV parsing."""

    def test_parse_csv_with_header(self):
        """Test parsing CSV with header."""
        from grb_common.io import load_lightcurve

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,flux,error\n")
            f.write("1.0,1e-26,1e-27\n")
            f.write("2.0,2e-26,2e-27\n")
            f.write("3.0,1.5e-26,1.5e-27\n")
            filepath = Path(f.name)

        try:
            lc = load_lightcurve(filepath, format="csv")
            assert len(lc) == 3
            np.testing.assert_array_almost_equal(lc.time, [1.0, 2.0, 3.0])
        finally:
            filepath.unlink()

    def test_parse_csv_without_header(self):
        """Test parsing CSV without header."""
        from grb_common.io import load_lightcurve

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("1.0,1e-26,1e-27\n")
            f.write("2.0,2e-26,2e-27\n")
            filepath = Path(f.name)

        try:
            lc = load_lightcurve(filepath, format="csv")
            assert len(lc) == 2
        finally:
            filepath.unlink()
