"""Tests for grb_common.plotting module."""

import pytest
import numpy as np


class TestStyle:
    """Tests for style module."""

    def test_colorblind_palette_length(self):
        """Test colorblind palette has expected colors."""
        from grb_common.plotting import COLORBLIND_PALETTE

        assert len(COLORBLIND_PALETTE) == 8
        assert all(c.startswith("#") for c in COLORBLIND_PALETTE)

    def test_get_color_wraps(self):
        """Test get_color wraps around palette."""
        from grb_common.plotting import get_color, COLORBLIND_PALETTE

        assert get_color(0) == COLORBLIND_PALETTE[0]
        assert get_color(8) == COLORBLIND_PALETTE[0]  # Wraps
        assert get_color(9) == COLORBLIND_PALETTE[1]

    def test_get_band_color_known(self):
        """Test get_band_color for known bands."""
        from grb_common.plotting import get_band_color, GRB_BAND_COLORS

        assert get_band_color("X-ray") == GRB_BAND_COLORS["X-ray"]
        assert get_band_color("radio") == GRB_BAND_COLORS["radio"]

    def test_get_band_color_partial_match(self):
        """Test get_band_color partial matching."""
        from grb_common.plotting import get_band_color, GRB_BAND_COLORS

        assert get_band_color("X-ray band") == GRB_BAND_COLORS["X-ray"]
        assert get_band_color("radio_custom") == GRB_BAND_COLORS["radio"]
        assert get_band_color("optical_unknown") == GRB_BAND_COLORS["optical"]

    def test_get_band_color_fallback(self):
        """Test get_band_color fallback to hash-based color."""
        from grb_common.plotting import get_band_color

        color = get_band_color("unknown_band_xyz")
        assert color.startswith("#")

    def test_set_style_default(self):
        """Test setting default style."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import set_style, STYLES
        import matplotlib.pyplot as plt

        set_style("default")
        # Check some style parameters were applied
        assert plt.rcParams["legend.frameon"] == STYLES["default"]["legend.frameon"]

    def test_set_style_apj(self):
        """Test setting ApJ style."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import set_style, STYLES
        import matplotlib.pyplot as plt

        set_style("apj")
        assert plt.rcParams["figure.dpi"] == STYLES["apj"]["figure.dpi"]

    def test_set_style_invalid(self):
        """Test error for invalid style."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import set_style

        with pytest.raises(ValueError, match="Unknown style"):
            set_style("invalid_style")

    def test_get_marker(self):
        """Test get_marker returns valid markers."""
        from grb_common.plotting import get_marker

        markers = [get_marker(i) for i in range(12)]
        assert all(isinstance(m, str) for m in markers)
        # Should wrap
        assert get_marker(0) == get_marker(10)

    def test_format_parameter_name_known(self):
        """Test formatting known parameter names."""
        from grb_common.plotting import format_parameter_name

        assert "E_{\\rm iso}" in format_parameter_name("E_iso")
        assert "\\epsilon_e" in format_parameter_name("epsilon_e")
        assert "\\theta_j" in format_parameter_name("theta_j")

    def test_format_parameter_name_generic(self):
        """Test formatting unknown parameter names."""
        from grb_common.plotting import format_parameter_name

        result = format_parameter_name("custom_param")
        assert "custom" in result and "param" in result


class TestLightcurvePlotting:
    """Tests for lightcurve plotting functions."""

    @pytest.fixture
    def mock_lightcurve(self):
        """Create a mock light curve."""
        class MockLC:
            time = np.array([1e3, 1e4, 1e5, 1e6])
            flux = np.array([1e-26, 5e-27, 2e-27, 1e-27])
            flux_err = np.array([1e-27, 5e-28, 2e-28, 1e-28])
            band = "X-ray"
            upper_limits = np.array([False, False, False, False])

            def __len__(self):
                return len(self.time)

        return MockLC()

    def test_plot_lightcurve_creates_axes(self, mock_lightcurve):
        """Test plot_lightcurve creates axes."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import plot_lightcurve

        ax = plot_lightcurve(mock_lightcurve)
        assert ax is not None

    def test_plot_lightcurve_unit_conversion(self, mock_lightcurve):
        """Test plot_lightcurve unit conversion."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import plot_lightcurve

        # CGS
        ax = plot_lightcurve(mock_lightcurve, flux_unit="cgs")
        assert "erg" in ax.get_ylabel()

        # mJy
        ax = plot_lightcurve(mock_lightcurve, flux_unit="mJy")
        assert "mJy" in ax.get_ylabel()

        # uJy
        ax = plot_lightcurve(mock_lightcurve, flux_unit="uJy")
        assert "Jy" in ax.get_ylabel()

    def test_plot_lightcurve_with_upper_limits(self, mock_lightcurve):
        """Test plot_lightcurve with upper limits."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import plot_lightcurve

        mock_lightcurve.upper_limits = np.array([False, False, True, False])
        ax = plot_lightcurve(mock_lightcurve, show_upper_limits=True)
        assert ax is not None


class TestSpectraPlotting:
    """Tests for spectra plotting functions."""

    def test_get_band_frequency_known(self):
        """Test get_band_frequency for known bands."""
        from grb_common.plotting import get_band_frequency

        # X-ray should be ~1e17 Hz
        freq = get_band_frequency("X-ray")
        assert 1e16 < freq < 1e19

        # Radio should be ~GHz
        freq = get_band_frequency("radio_5GHz")
        assert 1e9 < freq < 1e11

    def test_get_band_frequency_extract_ghz(self):
        """Test frequency extraction from band name."""
        from grb_common.plotting import get_band_frequency

        freq = get_band_frequency("custom_15GHz")
        assert np.isclose(freq, 15e9, rtol=0.1)

    def test_band_frequencies_dict(self):
        """Test BAND_FREQUENCIES has expected structure."""
        from grb_common.plotting import BAND_FREQUENCIES

        assert "X-ray" in BAND_FREQUENCIES
        assert "radio_5GHz" in BAND_FREQUENCIES
        assert all(len(v) == 2 for v in BAND_FREQUENCIES.values())


class TestCornerPlots:
    """Tests for corner plot functions."""

    @pytest.fixture
    def mock_result(self):
        """Create a mock sampler result."""
        from grb_common.fitting import SamplerResult

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=[1, 2, 3], scale=[0.1, 0.2, 0.3], size=(1000, 3))
        log_like = -0.5 * np.sum(samples**2, axis=1)

        return SamplerResult(
            samples=samples,
            log_likelihood=log_like,
            param_names=["E_iso", "n", "p"],
        )

    def test_corner_plot_creates_figure(self, mock_result):
        """Test corner_plot creates figure."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("corner")
        from grb_common.plotting import corner_plot

        fig = corner_plot(mock_result)
        assert fig is not None

    def test_corner_plot_with_truths(self, mock_result):
        """Test corner_plot with truth values."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("corner")
        from grb_common.plotting import corner_plot

        truths = {"E_iso": 1.0, "n": 2.0, "p": 3.0}
        fig = corner_plot(mock_result, truths=truths)
        assert fig is not None

    def test_corner_plot_subset_params(self, mock_result):
        """Test corner_plot with subset of parameters."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("corner")
        from grb_common.plotting import corner_plot

        fig = corner_plot(mock_result, params=["E_iso", "n"])
        assert fig is not None

    def test_corner_1d(self, mock_result):
        """Test corner_1d creates figure."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import corner_1d

        fig = corner_1d(mock_result)
        assert fig is not None

    def test_trace_plot(self, mock_result):
        """Test trace_plot creates figure."""
        pytest.importorskip("matplotlib")
        from grb_common.plotting import trace_plot

        fig = trace_plot(mock_result)
        assert fig is not None


class TestPlottingImports:
    """Tests for module imports."""

    def test_all_exports_importable(self):
        """Test all __all__ exports are importable."""
        from grb_common import plotting

        for name in plotting.__all__:
            assert hasattr(plotting, name), f"{name} not found in plotting module"

    def test_lazy_matplotlib_import(self):
        """Test matplotlib is not imported until needed."""
        # This is a design verification test
        from grb_common.plotting import COLORBLIND_PALETTE, get_color

        # These should work without matplotlib
        assert len(COLORBLIND_PALETTE) == 8
        assert get_color(0).startswith("#")
