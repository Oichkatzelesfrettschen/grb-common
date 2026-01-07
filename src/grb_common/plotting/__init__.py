"""
Publication-quality plotting utilities for GRB analysis.

This module provides standardized plotting functions for GRB afterglow
data visualization, following astronomy journal style guidelines.

Style:
    set_style: Apply publication style settings
    get_color: Get colorblind-safe color
    get_band_color: Get color for observational band

Light curves:
    plot_lightcurve: Single-band light curve
    plot_multiband: Multi-band light curve overlay
    plot_with_model: Light curve with model overlay
    plot_lightcurve_comparison: Compare multiple datasets

Spectra:
    plot_sed: Spectral energy distribution
    plot_sed_evolution: Multi-epoch SED evolution
    plot_broadband_sed: Broadband SED with breaks

Corner plots:
    corner_plot: Parameter posterior visualization
    corner_comparison: Compare multiple results
    corner_1d: 1D marginal distributions
    trace_plot: MCMC convergence diagnostic

Style presets:
    'default': Matplotlib defaults with GRB tweaks
    'mnras': Monthly Notices of the RAS
    'apj': Astrophysical Journal
    'aa': Astronomy & Astrophysics
    'nature': Nature/Science compact style
    'presentation': Large fonts for slides

Usage:
    from grb_common.plotting import set_style, plot_lightcurve
    from grb_common.io import load_lightcurve

    set_style('mnras')
    lc = load_lightcurve("grb_xray.txt")
    fig = plot_lightcurve(lc)
    fig.savefig("lightcurve.pdf")
"""

from .corner import (
    corner_1d,
    corner_comparison,
    corner_plot,
    trace_plot,
)
from .lightcurves import (
    plot_lightcurve,
    plot_lightcurve_comparison,
    plot_multiband,
    plot_with_model,
)
from .spectra import (
    BAND_FREQUENCIES,
    get_band_frequency,
    plot_broadband_sed,
    plot_sed,
    plot_sed_evolution,
)
from .style import (
    COLORBLIND_PALETTE,
    GRB_BAND_COLORS,
    STYLES,
    format_parameter_name,
    get_band_color,
    get_color,
    get_marker,
    set_style,
)

__all__ = [
    # Style
    "COLORBLIND_PALETTE",
    "GRB_BAND_COLORS",
    "STYLES",
    "set_style",
    "get_color",
    "get_band_color",
    "get_marker",
    "format_parameter_name",
    # Light curves
    "plot_lightcurve",
    "plot_multiband",
    "plot_with_model",
    "plot_lightcurve_comparison",
    # Spectra
    "BAND_FREQUENCIES",
    "get_band_frequency",
    "plot_sed",
    "plot_sed_evolution",
    "plot_broadband_sed",
    # Corner plots
    "corner_plot",
    "corner_comparison",
    "corner_1d",
    "trace_plot",
]
