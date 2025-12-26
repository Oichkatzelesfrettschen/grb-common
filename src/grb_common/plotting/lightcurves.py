"""
Light curve plotting utilities for GRB afterglow data.

This module provides functions for creating publication-quality
light curve plots with proper error bars, upper limits, and
multi-band visualization.

Usage:
    from grb_common.plotting import plot_lightcurve, plot_multiband
    from grb_common.io import load_lightcurve

    lc = load_lightcurve("grb170817_xray.txt")
    fig = plot_lightcurve(lc)
    fig.savefig("lightcurve.pdf")
"""

from typing import Optional, List, Union, Tuple, Any
import numpy as np

from .style import get_band_color, get_marker


def plot_lightcurve(
    lc: "LightCurve",
    ax: Optional[Any] = None,
    xscale: str = "log",
    yscale: str = "log",
    flux_unit: str = "cgs",
    show_upper_limits: bool = True,
    color: Optional[str] = None,
    marker: str = "o",
    label: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Plot a single light curve.

    Parameters
    ----------
    lc : LightCurve
        Light curve data.
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
    color : str, optional
        Plot color. Auto-detected from band if None.
    marker : str
        Marker style.
    label : str, optional
        Legend label. Uses band name if None.
    **kwargs
        Additional arguments passed to errorbar().

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

    # Unit conversion
    flux = lc.flux.copy()
    flux_err = lc.flux_err.copy()

    if flux_unit == "mJy":
        flux = flux / 1e-26
        flux_err = flux_err / 1e-26
        ylabel = r"Flux Density [mJy]"
    elif flux_unit == "uJy":
        flux = flux / 1e-29
        flux_err = flux_err / 1e-29
        ylabel = r"Flux Density [$\mu$Jy]"
    else:
        ylabel = r"Flux Density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]"

    # Auto-detect color
    if color is None:
        color = get_band_color(lc.band)

    # Label
    if label is None:
        label = lc.band

    # Separate detections and upper limits
    if lc.upper_limits is not None and show_upper_limits:
        det_mask = ~lc.upper_limits
        ul_mask = lc.upper_limits
    else:
        det_mask = np.ones(len(lc), dtype=bool)
        ul_mask = np.zeros(len(lc), dtype=bool)

    # Plot detections
    if np.any(det_mask):
        ax.errorbar(
            lc.time[det_mask],
            flux[det_mask],
            yerr=flux_err[det_mask],
            fmt=marker,
            color=color,
            label=label,
            capsize=kwargs.pop("capsize", 2),
            **kwargs,
        )

    # Plot upper limits
    if np.any(ul_mask) and show_upper_limits:
        ax.scatter(
            lc.time[ul_mask],
            flux[ul_mask],
            marker="v",
            s=kwargs.get("ms", 30),
            color=color,
            alpha=0.7,
        )
        # Add downward arrows
        for t, f in zip(lc.time[ul_mask], flux[ul_mask]):
            ax.annotate(
                "",
                xy=(t, f * 0.3),
                xytext=(t, f),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
            )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(ylabel)

    return ax


def plot_multiband(
    obs: "GRBObservation",
    ax: Optional[Any] = None,
    bands: Optional[List[str]] = None,
    xscale: str = "log",
    yscale: str = "log",
    flux_unit: str = "cgs",
    offset_bands: bool = False,
    offset_factor: float = 10.0,
    show_legend: bool = True,
    **kwargs,
) -> Any:
    """
    Plot multi-band light curves.

    Parameters
    ----------
    obs : GRBObservation
        Observation with multiple light curves.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    bands : list of str, optional
        Bands to include. All if None.
    xscale : str
        X-axis scale.
    yscale : str
        Y-axis scale.
    flux_unit : str
        Flux units.
    offset_bands : bool
        Vertically offset bands for clarity.
    offset_factor : float
        Multiplication factor between bands if offset.
    show_legend : bool
        Show legend.
    **kwargs
        Additional arguments passed to plot_lightcurve().

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

    if bands is None:
        bands = obs.bands

    for i, band in enumerate(bands):
        lc = obs.get_band(band)
        if lc is None:
            continue

        # Create offset copy if requested
        if offset_bands:
            from ..io import LightCurve
            offset = offset_factor ** i
            lc_plot = LightCurve(
                time=lc.time,
                flux=lc.flux * offset,
                flux_err=lc.flux_err * offset,
                band=lc.band,
                upper_limits=lc.upper_limits,
            )
            label = f"{band} ($\\times${offset:.0e})" if offset != 1 else band
        else:
            lc_plot = lc
            label = band

        plot_lightcurve(
            lc_plot,
            ax=ax,
            xscale=xscale,
            yscale=yscale,
            flux_unit=flux_unit,
            label=label,
            marker=get_marker(i),
            **kwargs,
        )

    if show_legend:
        ax.legend(loc="best")

    # Title
    if hasattr(obs, 'metadata') and obs.metadata.name:
        ax.set_title(obs.metadata.name)

    return ax


def plot_with_model(
    lc: "LightCurve",
    model_time: np.ndarray,
    model_flux: np.ndarray,
    ax: Optional[Any] = None,
    show_residuals: bool = False,
    flux_unit: str = "cgs",
    data_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
) -> Union[Any, Tuple[Any, Any]]:
    """
    Plot light curve with model overlay.

    Parameters
    ----------
    lc : LightCurve
        Observed light curve.
    model_time : ndarray
        Model time array.
    model_flux : ndarray
        Model flux array.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    show_residuals : bool
        Show residual panel below main plot.
    flux_unit : str
        Flux units.
    data_kwargs : dict, optional
        Keyword arguments for data points.
    model_kwargs : dict, optional
        Keyword arguments for model line.

    Returns
    -------
    matplotlib.axes.Axes or tuple
        Main axes, or (main_ax, residual_ax) if show_residuals.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required")

    data_kwargs = data_kwargs or {}
    model_kwargs = model_kwargs or {}

    if show_residuals:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1,
            figsize=(6, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        if ax is None:
            fig, ax_main = plt.subplots()
        else:
            ax_main = ax

    # Plot data
    color = get_band_color(lc.band)
    plot_lightcurve(
        lc,
        ax=ax_main,
        flux_unit=flux_unit,
        color=data_kwargs.pop("color", color),
        **data_kwargs,
    )

    # Unit conversion for model
    if flux_unit == "mJy":
        model_flux_plot = model_flux / 1e-26
    elif flux_unit == "uJy":
        model_flux_plot = model_flux / 1e-29
    else:
        model_flux_plot = model_flux

    # Plot model
    ax_main.plot(
        model_time,
        model_flux_plot,
        color=model_kwargs.pop("color", "black"),
        linestyle=model_kwargs.pop("linestyle", "-"),
        linewidth=model_kwargs.pop("linewidth", 1.5),
        label=model_kwargs.pop("label", "Model"),
        **model_kwargs,
    )

    ax_main.legend()

    if show_residuals:
        # Interpolate model to data times
        from scipy.interpolate import interp1d
        model_interp = interp1d(
            model_time, model_flux,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        model_at_data = model_interp(lc.time)

        # Compute residuals
        residuals = (lc.flux - model_at_data) / lc.flux_err

        # Filter valid points
        valid = np.isfinite(residuals)
        if lc.upper_limits is not None:
            valid &= ~lc.upper_limits

        ax_res.errorbar(
            lc.time[valid],
            residuals[valid],
            yerr=np.ones(np.sum(valid)),
            fmt="o",
            color=color,
            capsize=2,
        )
        ax_res.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax_res.set_ylabel(r"Residual [$\sigma$]")
        ax_res.set_xlabel(r"Time [s]")
        ax_res.set_xscale("log")

        return ax_main, ax_res

    return ax_main


def plot_lightcurve_comparison(
    light_curves: List["LightCurve"],
    labels: Optional[List[str]] = None,
    ax: Optional[Any] = None,
    **kwargs,
) -> Any:
    """
    Plot multiple light curves for comparison.

    Useful for comparing different model fits or data reductions.

    Parameters
    ----------
    light_curves : list of LightCurve
        Light curves to compare.
    labels : list of str, optional
        Labels for each light curve.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Additional arguments passed to plot_lightcurve().

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

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(light_curves))]

    for i, (lc, label) in enumerate(zip(light_curves, labels)):
        plot_lightcurve(
            lc,
            ax=ax,
            color=get_band_color(lc.band) if i == 0 else None,
            marker=get_marker(i),
            label=label,
            **kwargs,
        )

    ax.legend()
    return ax


__all__ = [
    "plot_lightcurve",
    "plot_multiband",
    "plot_with_model",
    "plot_lightcurve_comparison",
]
