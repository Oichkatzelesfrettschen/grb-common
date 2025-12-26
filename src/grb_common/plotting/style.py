"""
Publication style configuration for GRB plots.

This module provides style presets for major astronomy journals and
colorblind-safe color palettes following Wong (2011).

Supported styles:
    - 'default': Matplotlib defaults with GRB-appropriate tweaks
    - 'apj': Astrophysical Journal
    - 'mnras': Monthly Notices of the RAS
    - 'aa': Astronomy & Astrophysics
    - 'nature': Nature/Science compact style

Usage:
    from grb_common.plotting import set_style, get_color

    set_style('mnras')
    plt.plot(t, flux, color=get_color(0))
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Wong (2011) colorblind-safe palette
# https://www.nature.com/articles/nmeth.1618
COLORBLIND_PALETTE = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
]

# Extended palette for multi-band GRB data
GRB_BAND_COLORS = {
    # X-ray
    "X-ray": "#0072B2",
    "Swift-XRT": "#0072B2",
    "Chandra": "#0072B2",
    "XMM": "#0072B2",
    # Optical bands
    "optical": "#009E73",
    "optical_U": "#CC79A7",
    "optical_B": "#56B4E9",
    "optical_V": "#009E73",
    "optical_R": "#E69F00",
    "optical_I": "#D55E00",
    "optical_J": "#D55E00",
    "optical_H": "#A52A2A",
    "optical_K": "#8B0000",
    # Radio
    "radio": "#E69F00",
    "radio_1GHz": "#F0E442",
    "radio_5GHz": "#E69F00",
    "radio_10GHz": "#D55E00",
    "radio_15GHz": "#CC79A7",
    # Gamma-ray
    "gamma": "#000000",
    "Fermi-GBM": "#000000",
    "Fermi-LAT": "#333333",
}

# Journal-specific style configurations
STYLES: Dict[str, Dict[str, Any]] = {
    "default": {
        "figure.figsize": (8, 6),
        "figure.dpi": 100,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.linewidth": 1.0,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "errorbar.capsize": 2,
    },
    "apj": {
        # ApJ single column: 3.5 inches, double: 7.1 inches
        "figure.figsize": (3.5, 2.8),
        "figure.dpi": 300,
        "font.size": 8,
        "font.family": "serif",
        "font.serif": ["Times", "DejaVu Serif"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "errorbar.capsize": 1.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    },
    "mnras": {
        # MNRAS single column: 84mm (3.3 inches), double: 174mm (6.85 inches)
        "figure.figsize": (3.3, 2.6),
        "figure.dpi": 300,
        "font.size": 8,
        "font.family": "serif",
        "font.serif": ["Times", "DejaVu Serif"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "errorbar.capsize": 1.5,
    },
    "aa": {
        # A&A single column: 88mm (3.46 inches), double: 180mm (7.09 inches)
        "figure.figsize": (3.46, 2.8),
        "figure.dpi": 300,
        "font.size": 8,
        "font.family": "serif",
        "axes.linewidth": 0.5,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
    },
    "nature": {
        # Nature: very compact, 89mm single column
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 300,
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 6,
        "legend.frameon": False,
        "lines.linewidth": 0.75,
        "lines.markersize": 3,
    },
    "presentation": {
        # For slides/talks
        "figure.figsize": (10, 7),
        "figure.dpi": 150,
        "font.size": 16,
        "font.family": "sans-serif",
        "axes.linewidth": 1.5,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "errorbar.capsize": 4,
    },
}


def set_style(style: str = "default") -> None:
    """
    Set matplotlib style for publication.

    Parameters
    ----------
    style : str
        Style name: 'default', 'apj', 'mnras', 'aa', 'nature', 'presentation'.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install grb-common[plotting]"
        )

    if style not in STYLES:
        raise ValueError(f"Unknown style: {style}. Choose from {list(STYLES.keys())}")

    plt.rcParams.update(STYLES[style])


def get_color(index: int) -> str:
    """
    Get color from colorblind-safe palette.

    Parameters
    ----------
    index : int
        Color index (wraps around).

    Returns
    -------
    str
        Hex color code.
    """
    return COLORBLIND_PALETTE[index % len(COLORBLIND_PALETTE)]


def get_band_color(band: str) -> str:
    """
    Get color for a specific observational band.

    Parameters
    ----------
    band : str
        Band name (e.g., 'X-ray', 'optical_R', 'radio_5GHz').

    Returns
    -------
    str
        Hex color code.
    """
    if band in GRB_BAND_COLORS:
        return GRB_BAND_COLORS[band]

    # Try partial matches
    band_lower = band.lower()
    if "x-ray" in band_lower or "xray" in band_lower:
        return GRB_BAND_COLORS["X-ray"]
    if "radio" in band_lower:
        return GRB_BAND_COLORS["radio"]
    if "optical" in band_lower:
        return GRB_BAND_COLORS["optical"]

    # Fallback to palette
    return get_color(hash(band) % len(COLORBLIND_PALETTE))


def get_marker(index: int) -> str:
    """
    Get marker style.

    Parameters
    ----------
    index : int
        Marker index.

    Returns
    -------
    str
        Matplotlib marker code.
    """
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]
    return markers[index % len(markers)]


def format_parameter_name(name: str) -> str:
    """
    Format parameter name for plot labels.

    Converts snake_case to LaTeX math notation.

    Parameters
    ----------
    name : str
        Parameter name (e.g., 'E_iso', 'epsilon_e', 'theta_obs').

    Returns
    -------
    str
        Formatted LaTeX string.
    """
    # Common GRB parameter substitutions
    substitutions = {
        "E_iso": r"$E_{\rm iso}$ [erg]",
        "E_k": r"$E_k$ [erg]",
        "n": r"$n$ [cm$^{-3}$]",
        "n_0": r"$n_0$ [cm$^{-3}$]",
        "epsilon_e": r"$\epsilon_e$",
        "epsilon_B": r"$\epsilon_B$",
        "epsilon_b": r"$\epsilon_B$",
        "p": r"$p$",
        "theta_obs": r"$\theta_{\rm obs}$ [rad]",
        "theta_j": r"$\theta_j$ [rad]",
        "theta_c": r"$\theta_c$ [rad]",
        "Gamma_0": r"$\Gamma_0$",
        "z": r"$z$",
        "d_L": r"$d_L$ [cm]",
        "A_V": r"$A_V$ [mag]",
        "t_0": r"$t_0$ [s]",
    }

    if name in substitutions:
        return substitutions[name]

    # Generic formatting
    if "_" in name:
        parts = name.split("_")
        if len(parts) == 2:
            return f"${parts[0]}_{{{parts[1]}}}$"

    return name


__all__ = [
    "COLORBLIND_PALETTE",
    "GRB_BAND_COLORS",
    "STYLES",
    "set_style",
    "get_color",
    "get_band_color",
    "get_marker",
    "format_parameter_name",
]
