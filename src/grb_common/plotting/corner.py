"""
Corner plot utilities for MCMC/nested sampling results.

This module provides a wrapper around corner.py with GRB-specific
formatting and integration with SamplerResult.

Usage:
    from grb_common.plotting import corner_plot
    from grb_common.fitting import SamplerResult

    result = SamplerResult.load("chains.h5")
    fig = corner_plot(result)
    fig.savefig("corner.pdf")
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np

from .style import format_parameter_name, COLORBLIND_PALETTE


def corner_plot(
    result: "SamplerResult",
    params: Optional[List[str]] = None,
    truths: Optional[Dict[str, float]] = None,
    labels: Optional[List[str]] = None,
    quantiles: List[float] = [0.16, 0.5, 0.84],
    show_titles: bool = True,
    title_fmt: str = ".2e",
    color: Optional[str] = None,
    truth_color: str = "#D55E00",
    **kwargs,
) -> Any:
    """
    Create corner plot from sampler result.

    Parameters
    ----------
    result : SamplerResult
        Sampler output containing samples and parameter names.
    params : list of str, optional
        Parameters to include. All if None.
    truths : dict, optional
        True values to mark on plot.
    labels : list of str, optional
        Labels for parameters. Uses formatted names if None.
    quantiles : list of float
        Quantiles to show in 1D histograms.
    show_titles : bool
        Show parameter estimates as titles.
    title_fmt : str
        Format string for titles.
    color : str, optional
        Plot color.
    truth_color : str
        Color for truth markers.
    **kwargs
        Additional arguments passed to corner.corner().

    Returns
    -------
    matplotlib.figure.Figure
        The corner plot figure.
    """
    try:
        import corner
    except ImportError:
        raise ImportError(
            "corner package required for corner plots. "
            "Install with: pip install corner"
        )

    # Select parameters
    if params is None:
        params = result.param_names
        param_indices = list(range(result.n_params))
    else:
        param_indices = [result.param_names.index(p) for p in params]

    samples = result.samples[:, param_indices]

    # Format labels
    if labels is None:
        labels = [format_parameter_name(p) for p in params]

    # Extract truths
    if truths is not None:
        truth_values = [truths.get(p, None) for p in params]
    else:
        truth_values = None

    # Get weights if available
    weights = getattr(result, 'weights', None)

    # Set color
    if color is None:
        color = COLORBLIND_PALETTE[5]  # Blue

    # Default kwargs
    corner_kwargs = {
        "labels": labels,
        "quantiles": quantiles,
        "show_titles": show_titles,
        "title_fmt": title_fmt,
        "color": color,
        "truth_color": truth_color,
        "plot_datapoints": kwargs.pop("plot_datapoints", True),
        "plot_density": kwargs.pop("plot_density", True),
        "fill_contours": kwargs.pop("fill_contours", True),
        "levels": kwargs.pop("levels", [0.68, 0.95]),
        "smooth": kwargs.pop("smooth", 1.0),
    }
    corner_kwargs.update(kwargs)

    if truth_values is not None:
        corner_kwargs["truths"] = truth_values

    if weights is not None:
        corner_kwargs["weights"] = weights

    fig = corner.corner(samples, **corner_kwargs)

    return fig


def corner_comparison(
    results: List["SamplerResult"],
    names: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    **kwargs,
) -> Any:
    """
    Create corner plot comparing multiple sampler results.

    Parameters
    ----------
    results : list of SamplerResult
        Sampler outputs to compare.
    names : list of str, optional
        Names for each result set.
    params : list of str, optional
        Parameters to include. All shared parameters if None.
    colors : list of str, optional
        Colors for each result set.
    **kwargs
        Additional arguments passed to corner.corner().

    Returns
    -------
    matplotlib.figure.Figure
        The corner plot figure.
    """
    try:
        import corner
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("corner and matplotlib required")

    # Find shared parameters
    if params is None:
        shared_params = set(results[0].param_names)
        for r in results[1:]:
            shared_params &= set(r.param_names)
        params = list(shared_params)

    # Names
    if names is None:
        names = [f"Result {i+1}" for i in range(len(results))]

    # Colors
    if colors is None:
        colors = COLORBLIND_PALETTE[:len(results)]

    # Labels
    labels = [format_parameter_name(p) for p in params]

    fig = None
    for i, result in enumerate(results):
        param_indices = [result.param_names.index(p) for p in params]
        samples = result.samples[:, param_indices]

        weights = getattr(result, 'weights', None)

        corner_kwargs = {
            "labels": labels,
            "color": colors[i],
            "plot_datapoints": False,
            "plot_density": True,
            "fill_contours": False,
            "levels": [0.68, 0.95],
            "smooth": 1.0,
        }
        corner_kwargs.update(kwargs)

        if weights is not None:
            corner_kwargs["weights"] = weights

        if fig is None:
            fig = corner.corner(samples, **corner_kwargs)
        else:
            corner.corner(samples, fig=fig, **corner_kwargs)

    # Add legend
    axes = fig.get_axes()
    lines = [plt.Line2D([0], [0], color=c, linewidth=2) for c in colors[:len(results)]]
    axes[0].legend(lines, names, loc="upper right")

    return fig


def corner_1d(
    result: "SamplerResult",
    params: Optional[List[str]] = None,
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    show_quantiles: bool = True,
    quantiles: List[float] = [0.16, 0.5, 0.84],
    **kwargs,
) -> Any:
    """
    Create 1D marginal distributions for parameters.

    Parameters
    ----------
    result : SamplerResult
        Sampler output.
    params : list of str, optional
        Parameters to include.
    ncols : int
        Number of columns in subplot grid.
    figsize : tuple, optional
        Figure size.
    show_quantiles : bool
        Show quantile lines.
    quantiles : list of float
        Quantiles to show.
    **kwargs
        Additional arguments passed to hist().

    Returns
    -------
    matplotlib.figure.Figure
        The figure with 1D histograms.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required")

    if params is None:
        params = result.param_names

    nparams = len(params)
    nrows = int(np.ceil(nparams / ncols))

    if figsize is None:
        figsize = (3 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    weights = getattr(result, 'weights', None)

    for i, param in enumerate(params):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        idx = result.param_names.index(param)
        samples = result.samples[:, idx]

        # Histogram
        hist_kwargs = {
            "bins": kwargs.pop("bins", 50),
            "density": True,
            "alpha": kwargs.pop("alpha", 0.7),
            "color": COLORBLIND_PALETTE[5],
        }

        if weights is not None:
            hist_kwargs["weights"] = weights

        ax.hist(samples, **hist_kwargs, **kwargs)

        # Quantiles
        if show_quantiles:
            if weights is not None:
                from ..fitting.result import weighted_percentile
                q_values = weighted_percentile(samples, weights, np.array(quantiles) * 100)
            else:
                q_values = np.percentile(samples, np.array(quantiles) * 100)

            for q_val in q_values:
                ax.axvline(q_val, color="black", linestyle="--", alpha=0.7)

            # Title with median and uncertainty
            med = q_values[1]
            lower = med - q_values[0]
            upper = q_values[2] - med

            ax.set_title(f"{med:.2e}$^{{+{upper:.2e}}}_{{-{lower:.2e}}}$", fontsize=9)

        ax.set_xlabel(format_parameter_name(param))
        ax.set_ylabel("Probability Density")

    # Hide unused subplots
    for i in range(nparams, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def trace_plot(
    result: "SamplerResult",
    params: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Any:
    """
    Create trace plot showing parameter evolution.

    Useful for diagnosing MCMC convergence.

    Parameters
    ----------
    result : SamplerResult
        Sampler output.
    params : list of str, optional
        Parameters to include.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Additional arguments passed to plot().

    Returns
    -------
    matplotlib.figure.Figure
        The trace plot figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required")

    if params is None:
        params = result.param_names

    nparams = len(params)

    if figsize is None:
        figsize = (10, 2.5 * nparams)

    fig, axes = plt.subplots(nparams, 1, figsize=figsize, sharex=True)
    if nparams == 1:
        axes = [axes]

    for i, param in enumerate(params):
        idx = result.param_names.index(param)
        samples = result.samples[:, idx]

        axes[i].plot(
            samples,
            color=COLORBLIND_PALETTE[5],
            alpha=0.7,
            linewidth=0.5,
            **kwargs,
        )
        axes[i].set_ylabel(format_parameter_name(param))

        # Add mean line
        mean_val = np.mean(samples)
        axes[i].axhline(mean_val, color="black", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Sample")

    plt.tight_layout()
    return fig


__all__ = [
    "corner_plot",
    "corner_comparison",
    "corner_1d",
    "trace_plot",
]
