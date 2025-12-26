"""
Spectral energy distribution (SED) plotting utilities.

This module provides functions for creating publication-quality
SED plots with multi-epoch visualization.

Usage:
    from grb_common.plotting import plot_sed, plot_sed_evolution
    from grb_common.io import load_grb

    obs = load_grb("grb170817.h5")
    fig = plot_sed(obs, epoch=1e5)
    fig.savefig("sed.pdf")
"""

from typing import Optional, List, Union, Any, Tuple
import numpy as np

from .style import get_band_color, get_color, get_marker, COLORBLIND_PALETTE


# Frequency/wavelength ranges for standard bands
BAND_FREQUENCIES = {
    # Gamma-ray (Hz)
    "gamma": (1e19, 1e24),
    "Fermi-GBM": (2.4e18, 7.3e20),
    "Fermi-LAT": (2.4e22, 7.3e25),
    # X-ray
    "X-ray": (2.4e16, 2.4e18),
    "Swift-XRT": (7.3e16, 2.4e18),
    "Chandra": (7.3e16, 2.4e18),
    # UV/Optical (Hz)
    "optical_U": (7.5e14, 9.5e14),
    "optical_B": (6.0e14, 7.5e14),
    "optical_V": (5.2e14, 6.0e14),
    "optical_R": (4.3e14, 5.2e14),
    "optical_I": (3.3e14, 4.3e14),
    "optical_J": (2.2e14, 2.8e14),
    "optical_H": (1.6e14, 2.0e14),
    "optical_K": (1.2e14, 1.5e14),
    # Radio
    "radio_1GHz": (0.5e9, 2e9),
    "radio_5GHz": (3e9, 8e9),
    "radio_10GHz": (8e9, 15e9),
    "radio_15GHz": (12e9, 20e9),
}


def get_band_frequency(band: str) -> float:
    """
    Get representative frequency for a band.

    Parameters
    ----------
    band : str
        Band name.

    Returns
    -------
    float
        Representative frequency in Hz.
    """
    if band in BAND_FREQUENCIES:
        low, high = BAND_FREQUENCIES[band]
        return np.sqrt(low * high)  # Geometric mean

    # Try to extract frequency from band name
    band_lower = band.lower()
    if "ghz" in band_lower:
        # Extract number before GHz
        import re
        match = re.search(r"(\d+(?:\.\d+)?)\s*ghz", band_lower)
        if match:
            return float(match.group(1)) * 1e9

    # Fallback heuristics
    if "x-ray" in band_lower or "xray" in band_lower:
        return 1e17
    if "optical" in band_lower:
        return 5e14
    if "radio" in band_lower:
        return 5e9

    # Default to optical
    return 5e14


def plot_sed(
    spectra: Union["Spectrum", List["Spectrum"]],
    ax: Optional[Any] = None,
    xscale: str = "log",
    yscale: str = "log",
    flux_unit: str = "cgs",
    show_upper_limits: bool = True,
    show_model: bool = False,
    model_frequencies: Optional[np.ndarray] = None,
    model_flux: Optional[np.ndarray] = None,
    **kwargs,
) -> Any:
    """
    Plot spectral energy distribution.

    Parameters
    ----------
    spectra : Spectrum or list of Spectrum
        Spectral data to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    xscale : str
        X-axis scale ('log' or 'linear').
    yscale : str
        Y-axis scale ('log' or 'linear').
    flux_unit : str
        Flux units: 'cgs' (erg/cm^2/s/Hz), 'mJy', 'uJy'.
    show_upper_limits : bool
        Show upper limits as downward arrows.
    show_model : bool
        Show model curve if model data provided.
    model_frequencies : ndarray, optional
        Model frequency array for overlay.
    model_flux : ndarray, optional
        Model flux array for overlay.
    **kwargs
        Additional arguments passed to plot functions.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install grb-common[plotting]"
        )

    if ax is None:
        fig, ax = plt.subplots()

    # Convert single spectrum to list
    if not isinstance(spectra, list):
        spectra = [spectra]

    for i, spectrum in enumerate(spectra):
        freq = spectrum.frequency
        flux = spectrum.flux.copy()
        flux_err = spectrum.flux_err.copy()

        # Unit conversion
        if flux_unit == "mJy":
            flux = flux / 1e-26
            flux_err = flux_err / 1e-26
        elif flux_unit == "uJy":
            flux = flux / 1e-29
            flux_err = flux_err / 1e-29

        # Color
        color = kwargs.pop("color", get_color(i))

        # Separate detections and upper limits
        if hasattr(spectrum, 'upper_limits') and spectrum.upper_limits is not None and show_upper_limits:
            det_mask = ~spectrum.upper_limits
            ul_mask = spectrum.upper_limits
        else:
            det_mask = np.ones(len(spectrum.frequency), dtype=bool)
            ul_mask = np.zeros(len(spectrum.frequency), dtype=bool)

        # Label
        label = kwargs.pop("label", f"t={spectrum.time:.1e} s" if hasattr(spectrum, 'time') else None)

        # Plot detections
        if np.any(det_mask):
            ax.errorbar(
                freq[det_mask],
                flux[det_mask],
                yerr=flux_err[det_mask],
                fmt=get_marker(i),
                color=color,
                label=label,
                capsize=kwargs.pop("capsize", 2),
                **kwargs,
            )

        # Plot upper limits
        if np.any(ul_mask) and show_upper_limits:
            ax.scatter(
                freq[ul_mask],
                flux[ul_mask],
                marker="v",
                s=kwargs.get("ms", 30),
                color=color,
                alpha=0.7,
            )

    # Plot model if provided
    if show_model and model_frequencies is not None and model_flux is not None:
        model_flux_plot = model_flux.copy()
        if flux_unit == "mJy":
            model_flux_plot = model_flux_plot / 1e-26
        elif flux_unit == "uJy":
            model_flux_plot = model_flux_plot / 1e-29

        ax.plot(
            model_frequencies,
            model_flux_plot,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Model",
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(r"Frequency [Hz]")

    if flux_unit == "mJy":
        ax.set_ylabel(r"Flux Density [mJy]")
    elif flux_unit == "uJy":
        ax.set_ylabel(r"Flux Density [$\mu$Jy]")
    else:
        ax.set_ylabel(r"Flux Density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")

    return ax


def plot_sed_evolution(
    obs: "GRBObservation",
    epochs: List[float],
    ax: Optional[Any] = None,
    time_tolerance: float = 0.2,
    offset_epochs: bool = True,
    offset_factor: float = 10.0,
    colormap: str = "viridis",
    **kwargs,
) -> Any:
    """
    Plot SED evolution across multiple epochs.

    Parameters
    ----------
    obs : GRBObservation
        Observation with light curves.
    epochs : list of float
        Times (in seconds) at which to extract SEDs.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    time_tolerance : float
        Fractional tolerance for matching times to epochs.
    offset_epochs : bool
        Vertically offset epochs for clarity.
    offset_factor : float
        Multiplication factor between epochs.
    colormap : str
        Matplotlib colormap for epoch colors.
    **kwargs
        Additional arguments passed to errorbar().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        raise ImportError("matplotlib required")

    if ax is None:
        fig, ax = plt.subplots()

    # Get colormap
    cmap = cm.get_cmap(colormap)
    colors = [cmap(i / (len(epochs) - 1)) for i in range(len(epochs))] if len(epochs) > 1 else [cmap(0.5)]

    for i, epoch in enumerate(epochs):
        frequencies = []
        fluxes = []
        flux_errs = []

        # Extract data near this epoch from each band
        for band in obs.bands:
            lc = obs.get_band(band)
            if lc is None:
                continue

            # Find closest time within tolerance
            dt = np.abs(lc.time - epoch) / epoch
            idx = np.argmin(dt)

            if dt[idx] < time_tolerance:
                # Skip upper limits
                if lc.upper_limits is not None and lc.upper_limits[idx]:
                    continue

                freq = get_band_frequency(band)
                flux = lc.flux[idx]
                flux_err = lc.flux_err[idx]

                # Apply offset
                if offset_epochs:
                    offset = offset_factor ** i
                    flux *= offset
                    flux_err *= offset

                frequencies.append(freq)
                fluxes.append(flux)
                flux_errs.append(flux_err)

        if frequencies:
            # Sort by frequency
            sort_idx = np.argsort(frequencies)
            frequencies = np.array(frequencies)[sort_idx]
            fluxes = np.array(fluxes)[sort_idx]
            flux_errs = np.array(flux_errs)[sort_idx]

            if offset_epochs:
                offset = offset_factor ** i
                label = f"t={epoch:.1e} s ($\\times${offset:.0e})" if offset != 1 else f"t={epoch:.1e} s"
            else:
                label = f"t={epoch:.1e} s"

            ax.errorbar(
                frequencies,
                fluxes,
                yerr=flux_errs,
                fmt="o-",
                color=colors[i],
                label=label,
                capsize=kwargs.pop("capsize", 2),
                **kwargs,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel(r"Flux Density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")
    ax.legend(loc="best")

    if hasattr(obs, 'metadata') and obs.metadata.name:
        ax.set_title(f"{obs.metadata.name} - SED Evolution")

    return ax


def plot_broadband_sed(
    frequency: np.ndarray,
    flux: np.ndarray,
    ax: Optional[Any] = None,
    highlight_breaks: bool = True,
    break_frequencies: Optional[List[float]] = None,
    break_labels: Optional[List[str]] = None,
    **kwargs,
) -> Any:
    """
    Plot broadband SED with optional spectral break annotations.

    Parameters
    ----------
    frequency : ndarray
        Frequency array in Hz.
    flux : ndarray
        Flux density array.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    highlight_breaks : bool
        Annotate spectral breaks.
    break_frequencies : list of float, optional
        Frequencies of spectral breaks to annotate.
    break_labels : list of str, optional
        Labels for break frequencies.
    **kwargs
        Additional arguments passed to plot().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required")

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        frequency,
        flux,
        color=kwargs.pop("color", "black"),
        linewidth=kwargs.pop("linewidth", 1.5),
        **kwargs,
    )

    # Annotate breaks
    if highlight_breaks and break_frequencies:
        if break_labels is None:
            break_labels = [f"$\\nu_{{{i+1}}}$" for i in range(len(break_frequencies))]

        for freq, label in zip(break_frequencies, break_labels):
            # Find flux at break
            idx = np.argmin(np.abs(frequency - freq))
            flux_at_break = flux[idx]

            ax.axvline(freq, color="gray", linestyle="--", alpha=0.5)
            ax.annotate(
                label,
                xy=(freq, flux_at_break),
                xytext=(freq * 1.5, flux_at_break * 2),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel(r"Flux Density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")

    return ax


__all__ = [
    "BAND_FREQUENCIES",
    "get_band_frequency",
    "plot_sed",
    "plot_sed_evolution",
    "plot_broadband_sed",
]
